# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routed experts on 2-bit tensor-sym planes (cubit moe_w2) for the
DeepSeek-V4 / GLM-5.x MoE family.

Opt-in via VLLM_MOE_W2=1. Replaces the stock routed-expert GEMM path:

  weights : checkpoint mxfp4 e2m1 codes -> {-4,-1,1,4} 2-bit planes built on
            GPU at load (the small QUANT_PROBE ablation reported acceptance
            2.73 vs 2.68 and 12/12 short coherent outputs; this is not a
            long-agent parity result). Base-only MXFP4 blocks refit their
            UE8M0 exponent when exact SSE decreases. FP8 block-quant
            checkpoints (DS4-FP8, GLM-5.2-FP8) are re-quantized at load via
            build_layer_planes_fp8.
  compute : cubit `moe_w2_mm` SASS GEMM (M<=4 per pair, PRMT-LUT decode,
            QMMA.SF block-32 sfb, f32 act-scale fold per k32) for BOTH
            w13 and w2.
  glue    : moe_align_block_size(block=4) pairs, a32 activation quant
            (fp8 e4m3 + exact f32 PER-32-GROUP scales; the a128 lineage
            quantized per-128 — the 4x-coarser groups plus the group-128
            mid-pipeline requant were the last measured source of the
            +8-11% completion-token inflation vs native; the e8m0-scale
            MXFP8-parity variant lost accuracy: GSM8K 95.5% vs 97.0%),
            silu*up in torch, weighted scatter-add unpermute. All
            steps are tensor ops or driver launches on the current stream:
            CUDA-graph capturable, registered as one custom op.

VRAM: planes+scales ~1.73 GiB/layer (vs ~3.2 GiB raw fp4) -> 43 layers fit
a single 96 GB SM120 board together with the fp8 dense stack and KV.
The MTP drafter keeps the stock DeepGEMM-MXFP4 path: layer names containing
"mtp" are excluded, matching the QUANT_PROBE protocol (drafter unmodified).
"""

import ctypes
import functools
import os
import re
import time

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.moe_w2_planes import (
    mxfp4_to_codes,
    mxfp4_refit_codes_scales,
    pack_fragment_major,
    pack_scales,
    scale_refit_enabled,
)
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

_KERN = b"moe_w2_mm"
_DIR = os.getenv("VLLM_MOE_W2_CUBIT_DIR", "/cubit-share")
_BLOCK = 4                      # tokens per pair == kernel M limit
_NTHR = 256                     # NWARP=8 (K>=1024)


def _nwarp_for_k(k: int) -> int:
    """Split-K warp count baked into each cubin by gen_moe_w2.py (KSLICE=K/NWARP
    must be a multiple of 128). K>=1024 -> 8 warps; K=512 (the w2 GEMM under TP4)
    shards to 4. The launch block MUST match the cubin or the extra warps index
    past K (KSLICE*wid) and read garbage. Mirrors the generator's `_nwarp`."""
    nb = k // 128
    cap = 8 if k >= 1024 else 4
    for n in range(min(cap, nb), 0, -1):
        if nb % n == 0:
            return n
    return 1

_cu = None
_fns: dict = {}
_state = "uninit"
# PREFILL LEVER (default ON since the mc4afrag cubins ship): fragment-major
# activations so each lane's m16k32 QMMA A-fragment loads in ONE LDG.128 (vs 8
# strided 4-byte loads). Profile showed prefill moe_w2_mm is L1/load-issue bound
# (NOT weight-DRAM bound), so this cuts the dominant load class ~4x at identical
# occupancy -> measured 1.30x (K=4096) / 1.27x (K=2048) on the prefill GEMM.
# Numerics are bit-identical to mc4. Needs moe_w2_mm_mc4afrag_k{K}_a32.cubin present
# (loader degrades to mc4 when missing). Opt out: VLLM_MOE_W2_AFRAG=0.
_AFRAG = os.getenv("VLLM_MOE_W2_AFRAG", "1") == "1"
_afrag_ok = False
_sm89 = False                   # set by the (8,9) loader branch
# sm_89 NATIVE decode cubin registry: {(K, N): cuFunction}. Populated
# only on Ada by _load_sm89_native; empty everywhere else, so the
# _launch fast-path check is a no-op on sm_120.
_native_fns: dict = {}
_scale_refit_logged = False
# One-shot latch for the w8 POOL parity gate (w8_pool_self_test). Set the
# first time the gate runs successfully (or is skipped because the tier has
# no pool yet); prevents re-running it on every subsequent layer build.
_w8_pool_gate_done = False
# One-shot latch for the w8 POOL gate SKIP warning (FP8 enabled but the gate
# could not run): the per-layer _w8_pool_gate wrapper emits the WARNING once.
_w8_pool_gate_skip_logged = False
# Stashed gate dims from _w8_pool_gate when the pool was deferred (n_slots==0
# at build time). w8_pool_gate_deferred() re-runs the gate post-finalize_auto.
_w8_gate_dims: tuple[int, int] | None = None
# PREFILL QUALITY LEVER (default ON): prefill-sized calls consume the FP4
# delta tier exactly like decode — pairs whose expert is FP4-resident divert
# to moe_w4(q)_mm, the rest stay on the 2-bit planes. Before this, prefill
# ALWAYS computed on bare 2-bit planes, which was measured to be the whole
# source of the +8-11% completion-token inflation vs native (GSM8K-200 DS4
# TP2 tau1.0: 125 tok -> 116-117 = native parity when prefill reads the FP4
# tier; the KV built from 2-bit prefill flattens decode logits cumulatively
# with context length — internal/PREFILL_KV_INFLATION_FINDINGS.md). The
# w4/w4q kernels take M<=4, so each 16-token prefill pair dispatches as 4
# sub-entries; the 2-bit majority keeps the MC4/AFRAG fast path. Prefill
# quality tracks pool coverage (partial pool = partial recovery — graceful).
# Cost: FP4-resident pairs read the slot plane once per sub-entry (4x plane
# traffic on the diverted minority). Opt out: VLLM_MOE_W2_PREFILL_FP4=0.
# BASE-cache prefill is untouched (its need-pool is gate-scoped and tiny).
#
# "ensure" mode (VLLM_MOE_W2_PREFILL_FP4=ensure) upgrades the opportunistic
# lever to the DECODE-CLASS GUARANTEE: before each layer's desc build the
# chunk's routed set is made resident via tier.ensure_resident (the base
# cache's prefill idiom — synchronous fetch + per-layer pin scope), so
# EVERY pair diverts to FP4 regardless of steady-state coverage. The pool
# only needs to hold one layer's chunk working set (<= E slots); parity no
# longer costs a step-union-sized pool. Prices: H2D on cold working sets
# (a broad prompt can stream the whole quintal store once, ~1 s/40 GiB on
# PCIe5), the decode hot set gets displaced by long prompts (tau-driven
# fires re-promote it after the prompt — transient), and the guarantee
# only holds on EAGER prefill: chunks replayed from captured piecewise
# graphs skip host code and stay opportunistic (same boundary the base
# cache accepts).
_PF4_ENV = os.getenv("VLLM_MOE_W2_PREFILL_FP4", "1")
_PREFILL_FP4 = _PF4_ENV not in ("0", "")
_PREFILL_FP4_ENSURE = _PF4_ENV == "ensure"
_pf4_ensure_logged = False
# One-shot w8 coverage trace (gated by VLLM_MOE_W2_FP8_TRACE): logs pool
# residency for the first few w8 prefills so we can confirm the dynamic pool
# is covering the hot-expert subset during the prefill. Zero-overhead off.
_w8_cov_logged = 0
# Decode/prefill routing threshold: calls with T > this take the prefill
# path (MC4/AFRAG kernels + prefill-FP4). The default 96 is the LARGEST
# CUDAGRAPH CAPTURE SIZE of the standing multi-seq configs — a captured
# decode graph must never cross into the prefill path. Low-concurrency
# quality configs can lower it (e.g. 16 at max-num-seqs 1, where the
# biggest decode step is seqs x (1+spec) = 3 tokens) so that SHORT PROMPTS
# (a 90-token chunk has T <= 96) reach the prefill path and the ensure
# guarantee instead of the opportunistic decode path. Keep it ABOVE the
# config's largest captured decode size or captured decode graphs would
# bake prefill-path behaviour.
_PREFILL_T = int(os.getenv("VLLM_MOE_W2_PREFILL_T", "96"))


def _to_fragment_major(a: torch.Tensor, pairs: int, K: int) -> torch.Tensor:
    """[pairs*16, K] fp8 row-major -> fragment-major per 16-token tile (matches the
    AFRAG kernel layout / tools.moe_w2_prefill_bench.pack_a_fragment_major):
    dims [pair, g2, g, j, quad, t, b] -> [pair, j, g, t, quad, g2, b].

    `a` MUST have EXACTLY pairs*16 rows (complete tiles). Callers pass the
    tile-aligned region ws['a1'][:pairs*16] -- NOT ws['a1'][:slots] (slots is the
    over-allocated, non-16-multiple sorted_ids size)."""
    assert a.shape[0] == pairs * 16, (a.shape, pairs)
    v = a.view(torch.uint8).view(pairs, 2, 8, K // 64, 4, 4, 4)
    v = v.permute(0, 3, 2, 5, 4, 1, 6).reshape(pairs * 16, K)
    return v.contiguous().view(a.dtype)

# layer_key -> dict(planes13, sc13, planes2, sc2, top_k, inter)
_LAYERS: dict[int, dict] = {}
_WS: dict = {}                  # shared workspaces, sized lazily

# ---- adaptive expert top-p (VLLM_MOE_W2_TOPP, colibri's --topp) ----------
# Keep each token's routed experts only up to cumulative router weight p:
# the tail of the top-k carries little output mass but full fetch/compute
# cost. Measured on colibri (GLM-5.2, same sigmoid+norm_topk router family):
# p=0.7 cut expert loads 30-40% and bought 1.6x end-to-end on a cold cache.
# Here it shrinks the per-step expert union — on the BASE cache that is
# fewer misses and fewer replay triggers; GPU-resident it is less HBM
# traffic. 0 (default) = off, exact stock routing.
#   VLLM_MOE_W2_TOPP        cumulative-weight cutoff p in (0,1)
#   VLLM_MOE_W2_TOPP_MIN    experts always kept per token (default 2)
#   VLLM_MOE_W2_TOPP_RENORM 1 (default): renormalize kept weights so the
#                           token's total routed weight is preserved
#                           (colibri semantics for norm_topk models);
#                           0: keep original weights (mass shrinks).
_TOPP = float(os.getenv("VLLM_MOE_W2_TOPP", "0"))
_TOPP_MIN = max(1, int(os.getenv("VLLM_MOE_W2_TOPP_MIN", "2")))
_TOPP_RENORM = os.getenv("VLLM_MOE_W2_TOPP_RENORM", "1") == "1"

# ---- activation-scale group size per GEMM (the a32 kernel format) --------
# The _a32 kernels read f32 A scales at PER-32 stride; quantizing with a
# coarser group and repeating each scale over the 32-groups it covers is
# mathematically identical to quantizing at that coarser group — so one
# cubin set serves any {32, 64, 128} combination. G1 = the x -> w13 GEMM,
# G2 = the silu·up requant -> w2 GEMM; UE8M0 rounds scales up to powers of
# two (per_token_group_quant_fp8's platform default on this stack, i.e.
# what the retired a128 lineage actually served).
#
# DEFAULTS = 128/128/UE8M0: the measured-best combination. The activation-
# precision handoff's plan A (per-32 groups) was built and E2E-FALSIFIED
# here — GSM8K-200, 2x6000 quintal tau1.0, all with the identical stack:
#   G1=128 G2=128 ue8m0 (a128-equivalent):  97.0%  122 tok  (flips 1<->1)
#   G1=32  G2=128 f32:                      96.5%  124 tok
#   G1=32  G2=32  f32:                      95.5%  127 tok
#   G1=32  G2=32  e8m0 (native MXFP8 fmt):  95.5%  122 tok
#   native / a128 anchors:                  97.0%  116 / 125 tok
# Finer A groups do NOT shorten completions (the +8-11% inflation vs native
# does not come from activation-scale granularity) and consistently cost
# accuracy against the 2-bit/quintal weight planes. The envs stay for
# format experiments; the per-32-capable kernels are the delivery.
_G1 = int(os.getenv("VLLM_MOE_W2_A32_G1", "128"))
_G2 = int(os.getenv("VLLM_MOE_W2_A32_G2", "128"))
_A32_UE8M0 = os.getenv("VLLM_MOE_W2_A32_UE8M0", "1") == "1"
assert _G1 in (32, 64, 128) and _G2 in (32, 64, 128), (_G1, _G2)


def _quant_a32(x, out_q, out_s, group: int):
    """Quantize rows of `x` into the per-32-stride a32 scale plane using
    `group`-sized amax groups. For group > 32 the scale broadcast into the
    strided plane is a single view-copy (no repeat_interleave temporary)."""
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
    )
    _, s = per_token_group_quant_fp8(x, group, out_q=out_q,
                                     use_ue8m0=_A32_UE8M0)
    r = group // 32
    if r == 1:
        out_s.copy_(s)
    else:
        out_s.view(s.shape[0], -1, r).copy_(s.unsqueeze(-1))


def _apply_topp(topk_weights: torch.Tensor, topk_ids: torch.Tensor):
    """Drop each token's routed-weight tail past cumulative fraction _TOPP.

    Dropped entries get weight 0 and their expert id REDIRECTED to the
    token's heaviest expert — the redirected pair never fetches a new
    expert (top-1 is always kept) and its zero weight makes the unpermute
    contribution exactly zero, so the drop needs no kernel changes and is
    invisible to moe_align/desc. Pure static-shape tensor ops:
    CUDA-graph-capture-safe. Returns (weights, ids) untouched when off."""
    k = topk_ids.shape[1]
    if not (0.0 < _TOPP < 1.0) or k <= _TOPP_MIN:
        return topk_weights, topk_ids
    w = topk_weights.float()
    order = torch.argsort(w, dim=1, descending=True)
    w_sorted = w.gather(1, order)
    cum = torch.cumsum(w_sorted, dim=1)
    tot = cum[:, -1:]
    # keep ranks whose PRECEDING cumulative mass is still below p*tot
    # (the first expert crossing the threshold is kept, colibri semantics)
    keep_sorted = (cum - w_sorted) < (_TOPP * tot)
    keep_sorted[:, :_TOPP_MIN] = True
    keep = torch.zeros_like(keep_sorted).scatter(1, order, keep_sorted)
    if _TOPP_RENORM:
        kept_sum = (w * keep).sum(dim=1, keepdim=True).clamp_min(1e-20)
        w = w * (tot / kept_sum)
    top1 = topk_ids.gather(1, order[:, :1])
    new_ids = torch.where(keep, topk_ids, top1.expand_as(topk_ids))
    new_w = torch.where(keep, w, torch.zeros_like(w)).to(topk_weights.dtype)
    return new_w, new_ids


def _sanitize_topk_routes(topk_weights: torch.Tensor,
                          topk_ids: torch.Tensor,
                          n_experts: int):
    """Prepare router output for CUDA MoE alignment.

    DSpark graph padding can leave a complete top-k row at ``-1``.  vLLM's
    alignment kernel skips IDs >= ``n_experts`` but does not reject negative
    IDs before indexing shared memory.  Map every invalid route to the
    supported positive skip sentinel and zero its contribution.  All tensor
    shapes stay static, so the operation is CUDA-graph-capture-safe.
    """
    valid = (topk_ids >= 0) & (topk_ids < n_experts)
    weights = torch.where(valid, topk_weights,
                          torch.zeros_like(topk_weights))
    ids = torch.where(valid, topk_ids,
                      torch.full_like(topk_ids, n_experts))
    return weights, ids


def enabled() -> bool:
    return os.getenv("VLLM_MOE_W2", "0") == "1"


_cutoff_cache: int | None = None


def _layer_cutoff() -> int:
    """Main-stack layer count: layers >= this are the MTP drafter. Taken from
    the model config when available (43 for DS4-Flash, 78 for GLM-5.2, 61 for
    Kimi-K2.7); VLLM_MOE_W2_NUM_LAYERS overrides.

    get_text_config() unwraps composite VLM configs (KimiK25Config keeps
    num_hidden_layers on .text_config; a bare hf_config lookup would raise
    and silently fall back to 43, sending layers 43+ down the stock path);
    for text-only configs it returns self.

    Manual memoization on SUCCESS ONLY (was @functools.cache): a transient
    config-not-current exception must not freeze the guessed 43 forever —
    on a 78-layer GLM that silently sent layers 43+ down the stock path
    (mixed precision, no log — review finding 2.4). The fallback is now
    loud and uncached."""
    global _cutoff_cache
    if _cutoff_cache is not None:
        return _cutoff_cache
    v = os.getenv("VLLM_MOE_W2_NUM_LAYERS")
    if v is not None:
        _cutoff_cache = int(v)
        return _cutoff_cache
    try:
        from vllm.config import get_current_vllm_config
        cfg = get_current_vllm_config().model_config.hf_config
        cfg = cfg.get_text_config()
        n = cfg.num_hidden_layers
        if n:
            _cutoff_cache = int(n)
            return _cutoff_cache
    except Exception as e:  # noqa: BLE001
        logger.error(
            "moe_w2 _layer_cutoff: no current vllm config (%s) — TEMPORARY "
            "uncached fallback 43 (DS4 layout); set VLLM_MOE_W2_NUM_LAYERS "
            "explicitly if this repeats past load.", e)
    return 43


def is_w2_layer(layer_name: str) -> bool:
    """Main-model routed experts only. The MTP drafter (layer index >=
    num_hidden_layers, e.g. model.layers.43.* for the 43-layer main stack)
    keeps its original path: QUANT_PROBE's acceptance numbers were
    measured with the drafter unmodified."""
    if not enabled():
        return False
    name = layer_name or ""
    if "mtp" in name:
        return False
    import re
    m = re.search(r"\.layers\.(\d+)\.", name)
    if m is None:
        return False
    return int(m.group(1)) < _layer_cutoff()


def _driver():
    global _cu
    if _cu is None:
        cu = ctypes.CDLL("libcuda.so.1")
        cu.cuLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 6 + [
            ctypes.c_uint, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p]
        cu.cuModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                    ctypes.c_char_p]
        cu.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                           ctypes.c_void_p, ctypes.c_char_p]
        _cu = cu
    return _cu


def _ck(r, what):
    if r:
        raise RuntimeError(f"moe_w2_cubit: CUDA error {r} in {what}")


def _warn_env_typos() -> None:
    """VLLM_MOE_W2* env names nothing reads are SILENT no-ops: the moe_w2
    stack reads os.getenv directly and envs.py does not register these
    knobs (so vllm's own "Unknown vLLM environment variable" warning fires
    even for the CORRECT names — benign, ignore it). The failure mode of a
    typo is "my knob did nothing", which costs a deploy cycle to notice
    (measured: ..PLANE_CACHE typo for ..PLANES_CACHE). Harvest the
    canonical names from the moe_w2_* sources and warn with a suggestion.
    """
    import difflib
    import glob
    here = os.path.dirname(os.path.abspath(__file__))
    known = {"VLLM_MOE_W2_DENSE_FP8"}    # read from fp8.py, not utils/
    for f in glob.glob(os.path.join(here, "moe_w2_*.py")):
        try:
            with open(f) as fh:
                known |= set(re.findall(r"VLLM_MOE_W2\w*", fh.read()))
        except OSError:
            pass
    for k in sorted(os.environ):
        if k.startswith("VLLM_MOE_W2") and k not in known:
            hint = difflib.get_close_matches(k, sorted(known), n=1)
            logger.warning(
                "moe_w2: env %s is not read by any moe_w2 module and does "
                "NOTHING%s", k,
                f" — did you mean {hint[0]}?" if hint else "")


# sm_89 native decode cubin (kernels/cuda/moe_w2_sm89_decode.cu):
# hand-written CUDA using Ada's native e4m3 tensor core
# (mma.sync.m16n8k32 f32.e4m3.e4m3) with in-register PRMT decode of the
# 2-bit codes to e4m3 — no BF16 weight materialization. Measured
# 3.1-4.7x over the Triton emulation on the two decode GEMM shapes of
# DS4-Flash @ TP2 (K=1024/N=4096 down, K=4096/N=2048 gate-up). It is
# DECODE-TIER ONLY: every "w2"-tier desc pair has m_rows <= 4 (mblock ==
# _BLOCK) and the kernel RETURNS WITHOUT WRITING for m_rows > 4, so it
# must never see an mc4/afrag prefill desc — _launch keys the fast path
# on tier == "w2" plus an exact (K, n_rows) registry hit; every other
# shape/tier stays on the Triton emulation. Launch contract (differs
# from the sm_120 cubins): grid (N/64, pairs), block 64, ONE pointer
# arg (the desc table). Same driver-API launch on the same torch stream
# as the sm_120 path, so CUDA graph capture works unchanged.
_SM89_NATIVE_SHAPES = ((1024, 4096), (4096, 2048))


def _launch_sm89_native(fn, desc, n_rows: int, pairs: int, stream) -> None:
    arg = ctypes.c_uint64(desc.data_ptr())
    argv = (ctypes.c_void_p * 1)(
        ctypes.cast(ctypes.byref(arg), ctypes.c_void_p))
    _ck(_driver().cuLaunchKernel(fn, n_rows // 64, pairs, 1, 64, 1, 1, 0,
                                 stream, argv, None), "sm89 native launch")


def _load_sm89_native() -> None:
    """Opportunistically register the native decode cubin on Ada.

    Any failure — knob off, cubin absent, driver refuses the module, a
    symbol is missing, the boot parity gate fails — logs a clear line
    and leaves _native_fns empty so the Triton emulation serves
    everything. Never fatal: the cubin is an optimization, not a
    requirement."""
    if os.getenv("VLLM_MOE_W2_SM89_NATIVE", "1").lower() in (
            "0", "false", "no", "off"):
        logger.info("moe_w2_cubit: sm_89 native decode cubin disabled "
                    "via VLLM_MOE_W2_SM89_NATIVE")
        return
    path = os.getenv("VLLM_MOE_W2_SM89_NATIVE_CUBIN",
                     os.path.join(_DIR, "moe_w2_sm89_decode.cubin"))
    if not os.path.exists(path):
        logger.info("moe_w2_cubit: no sm_89 native decode cubin at %s; "
                    "decode stays on the Triton emulation", path)
        return
    try:
        from vllm.model_executor.layers.quantization.utils import (
            moe_w2_sm89)
        cu = _driver()
        mod = ctypes.c_void_p()
        _ck(cu.cuModuleLoad(ctypes.byref(mod), path.encode()),
            f"cuModuleLoad {path}")
        selftest = os.getenv("VLLM_MOE_W2_SM89_SELFTEST", "1").lower() \
            not in ("0", "false", "no", "off")
        loaded = {}
        for k, n in _SM89_NATIVE_SHAPES:
            fn = ctypes.c_void_p()
            _ck(cu.cuModuleGetFunction(
                ctypes.byref(fn), mod,
                f"moe_w2_sm89_decode_k{k}_n{n}".encode()),
                f"cuModuleGetFunction sm89 native k{k}n{n}")
            if selftest:
                def _lnch(desc, n_rows, pairs, _fn=fn):
                    _launch_sm89_native(
                        _fn, desc, n_rows, pairs,
                        ctypes.c_void_p(
                            torch.cuda.current_stream().cuda_stream))
                worst = moe_w2_sm89.native_self_test(_lnch, k, n)
                logger.info(
                    "moe_w2_cubit: sm_89 native decode K=%d N=%d parity "
                    "worst_rel=%.3e (gate 2.5e-2)", k, n, worst)
            loaded[(k, n)] = fn
        _native_fns.update(loaded)
        logger.info(
            "moe_w2_cubit: sm_89 NATIVE decode cubin ACTIVE for (K, N)=%s "
            "from %s%s", sorted(_native_fns), path,
            "" if selftest else " (parity gate SKIPPED via env)")
    except Exception as e:  # noqa: BLE001
        _native_fns.clear()
        logger.warning(
            "moe_w2_cubit: sm_89 native decode cubin unavailable (%s); "
            "decode stays on the Triton emulation", e)


def _ensure_ready() -> bool:
    global _state
    if _state == "ready":
        return True
    if _state == "unavailable":
        return False
    try:
        _warn_env_typos()
        torch.cuda.init()
        torch.zeros(1, device="cuda")
        # Hand-written SASS = sm_120 ONLY, and the cubins carry no PTX to
        # JIT elsewhere. Ada (sm_89) gets a Triton EMULATION of the same
        # GEMM instead (below); everything else refuses EARLY with a clear
        # message instead of the late shape-assert at weight load (review
        # finding 2.6): SM121 (GB10/DGX Spark), SM100 and SM90 cannot run
        # these kernels.
        cap = torch.cuda.get_device_capability()
        if cap == (8, 9):
            # Ada (RTX 4090 / RTX 2000-6000 Ada): FP8 e4m3 tensor cores
            # but no QMMA/QMMA.SF, no FP4 — the cubins cannot run there.
            # moe_w2_sm89 emulates the GEMM in Triton: SAME host-packed
            # planes, SAME 6-field desc ABI, 2-bit codes decoded to BF16
            # in registers, both per-32 scales folded as explicit
            # multiplies (Ada's MMA has no scale-factor operand). Experts
            # still stream from HBM at 2 bits/elem — the Moet win
            # survives; only the tensor-core instruction changes. ONE
            # kernel serves decode (w2) and prefill (w2mc4) via the
            # m_rows mask. AFRAG is a QMMA A-fragment LDG.128 trick and
            # does not transfer: w2mc4afrag stays unregistered and
            # prefill falls back to w2mc4 (_afrag_ok stays False). The
            # w4/w4q delta tiers are NOT ported — serve with
            # VLLM_MOE_W2_DELTA=0 (_require_kernels fails loudly
            # otherwise). K set: same contract as the cubin loader below
            # (VLLM_MOE_W2_KS extends it without a code edit).
            from vllm.model_executor.layers.quantization.utils import (
                moe_w2_sm89)
            ks = tuple(int(x) for x in os.getenv(
                "VLLM_MOE_W2_KS", "7168,6144,4096,2048,1024,512").split(","))
            _fns.update(moe_w2_sm89.make_launchers(ks))
            global _sm89
            _sm89 = True
            # STARTUP SELF-TEST (observability: deploy cycles are more
            # expensive than one JIT compile at init). A tiny op-level
            # GEMM runs against a torch reference on the actual silicon:
            # a Triton codegen/driver regression fails HERE, attributed,
            # instead of as silent output corruption. Opt out:
            # VLLM_MOE_W2_SM89_SELFTEST=0.
            if os.getenv("VLLM_MOE_W2_SM89_SELFTEST", "1").lower() \
                    not in ("0", "false", "no", "off"):
                import triton as _triton
                worst = moe_w2_sm89.self_test()
                logger.info(
                    "moe_w2_cubit: sm_89 Triton emulation ready on %s "
                    "(triton %s): self-test worst_rel=%.3e (gate 2.5e-2); "
                    "w2/w2mc4 registered for K=%s",
                    torch.cuda.get_device_name(), _triton.__version__,
                    worst, sorted(ks))
            else:
                logger.info(
                    "moe_w2_cubit: sm_89 Triton emulation ready "
                    "(self-test SKIPPED via env); w2/w2mc4 for K=%s",
                    sorted(ks))
            # native decode cubin on top of the (already validated)
            # emulation: opportunistic, parity-gated, never fatal.
            _load_sm89_native()
            # FP8-e4m3 delta tier parity gate: when the w8 launchers are
            # registered (VLLM_MOE_W2_FP8_DELTA=1), a tiny op-level GEMM
            # validates the FP8 fragment-major byte layout + the native fp8
            # tl.dot at engine init. A wrong w8 kernel fails HERE, attributed,
            # instead of as silent prefill corruption. The addressing half is
            # also CPU-validated (kernels/gen/moe_w8_sm89_cpu_check.py).
            if ("w8", ks[0]) in _fns:
                try:
                    from vllm.model_executor.layers.quantization.utils import (
                        moe_w2_sm89 as _w8m)
                    worst8 = _w8m.w8_self_test()
                    logger.info(
                        "moe_w2_cubit: sm_89 FP8 delta tier ready: w8 "
                        "self-test worst_rel=%.3e (gate 2.5e-2); w8 "
                        "registered for K=%s", worst8, sorted(ks))
                except Exception as e:  # noqa: BLE001 - attribution, not silent
                    raise RuntimeError(
                        f"moe_w2_cubit: sm_89 FP8 delta tier PARITY GATE "
                        f"failed at engine init: {e}") from e
            _state = "ready"
            return True
        if cap != (12, 0):
            raise RuntimeError(
                f"moe_w2 kernels ship as sm_120 SASS cubins plus an "
                f"sm_89 Triton emulation; this device reports "
                f"sm_{cap[0]}{cap[1]}. Unset VLLM_MOE_W2 on this "
                "hardware.")
        cu = _driver()
        # ONLY `_a32` cubins load (activation format rev: f32 A scales at
        # PER-32-GROUP granularity, folded per k32). The a128 lineage reads
        # the desc `as` field with a (K/128)*4 row stride — feeding it the
        # a32 (K/32)*4 plane would be silent garbage, so the filename suffix
        # IS the format contract: a cubin dir without the complete _a32 set
        # fails loudly in _require_kernels (no old/new mixing possible).
        # The a128-era mc2 prefill experiment is retired (superseded by
        # mc4/AFRAG; never launched by the serving path).
        # K set: the shipped families cover DS4/GLM/Kimi at TP1-8;
        # VLLM_MOE_W2_KS extends it WITHOUT a code edit when a new model
        # brings a new contraction (comma-separated; cubins must exist).
        ks = tuple(int(x) for x in os.getenv(
            "VLLM_MOE_W2_KS", "7168,6144,4096,2048,1024,512").split(","))
        for tier, kern in (("w2", b"moe_w2_mm"), ("w4", b"moe_w4_mm"),
                           ("w4q", b"moe_w4q_mm"), ("w2mc4", b"moe_w2_mm")):
            # GEMM contraction K: gate-up needs K=hidden (4096 DS4-Flash,
            # 6144 GLM-5.x, 7168 Kimi-K2.x); down needs K=I/TP (2048 @ TP1,
            # 1024 @ TP2, 512 @ TP4). Cubins are loaded opportunistically --
            # the plane builders assert the shapes the model actually needs
            # are present (_require_kernels fails loudly at weight load).
            for k in ks:
                if tier == "w2mc4":
                    fname = f"moe_w2_mm_mc4_k{k}_a32.cubin"
                else:
                    fname = f"moe_{tier}_mm_k{k}_a32.cubin"
                path = os.path.join(_DIR, fname)
                if not os.path.exists(path):
                    continue
                mod = ctypes.c_void_p()
                _ck(cu.cuModuleLoad(ctypes.byref(mod), path.encode()),
                    f"cuModuleLoad {path}")
                fn = ctypes.c_void_p()
                _ck(cu.cuModuleGetFunction(ctypes.byref(fn), mod, kern),
                    "cuModuleGetFunction")
                _fns[(tier, k)] = fn
        global _afrag_ok
        if _AFRAG:
            try:
                for k in ks:
                    path = os.path.join(
                        _DIR, f"moe_w2_mm_mc4afrag_k{k}_a32.cubin")
                    if not os.path.exists(path):
                        continue
                    mod = ctypes.c_void_p()
                    _ck(cu.cuModuleLoad(ctypes.byref(mod), path.encode()),
                        f"cuModuleLoad {path}")
                    fn = ctypes.c_void_p()
                    _ck(cu.cuModuleGetFunction(ctypes.byref(fn), mod, b"moe_w2_mm"),
                        "cuModuleGetFunction afrag")
                    _fns[("w2mc4afrag", k)] = fn
                _afrag_ok = True
                logger.info("moe_w2_cubit: AFRAG prefill cubins loaded")
            except Exception as e:  # noqa: BLE001
                logger.warning("moe_w2_cubit: AFRAG unavailable (%s); using mc4", e)
                _afrag_ok = False
        _state = "ready"
        logger.info("moe_w2_cubit: cubins loaded: %s", sorted(_fns))
        return True
    except Exception as e:  # noqa: BLE001
        logger.error("moe_w2_cubit unavailable: %s", e)
        import traceback; logger.error("moe_w2_cubit FULL TRACEBACK:\n%s", traceback.format_exc())
        _state = "unavailable"
        return False


# --------------------------------------------------------------------------
# Load-time plane building
# --------------------------------------------------------------------------


def w8_pool_self_test(tier, k13: int, n13: int) -> float:
    """End-to-end FP8 gate THROUGH a pool slot: catches slot/stride/staging
    sizing bugs that the standalone w8_self_test / _op_case_fp8 cannot (e.g.
    a tier built with FP4-sized slots — the singleton-poisoning IMA root
    cause). Run at boot after the FP8 tier has its first staged expert plane.

    Mirrors _op_case_fp8 + _launch but rewrites the W pointer to land inside
    the tier's actual pool (TAIL slot — overruns fault here first), then
    syncs and compares against the torch fp32 reference. Raises on any
    non-finite output or relative error >= 2.5e-2 (the standard w8 gate)."""
    from vllm.model_executor.layers.quantization.utils import (
        moe_w2_sm89 as _w8m)
    from vllm.model_executor.layers.quantization.utils import moe_w2_delta
    case = (_w8m._op_case_w8fp4 if moe_w2_delta.fp8_store_fp4()
            else _w8m._op_case_fp8)
    desc, bufs, ref = case(k13, n13, m_rows=16)
    plane = bufs[2]                                   # the w plane
    exact = moe_w2_delta.exact_cache_enabled()
    expected = plane.numel() + (bufs[3].numel() if exact else 0)
    if expected != tier.w13_bytes:
        raise RuntimeError(
            f"w8 pool gate: staged section {expected} B != tier.w13_bytes "
            f"{tier.w13_bytes} B — tier sized for a different format")
    slot = tier.n_slots - 1                           # TAIL slot: overruns
                                                     # fault here first
    tier.pool[slot, :tier.w13_bytes].copy_(
        torch.cat((plane.view(-1), bufs[3].view(-1))) if exact
        else plane.view(-1)[:tier.w13_bytes])
    desc[0, 2] = tier.pool[slot].data_ptr()           # W ptr now inside the
                                                     # pool
    if exact:
        desc[0, 3] = tier.pool[slot].data_ptr() + plane.numel()
    _launch("w8", k13, desc[0], n13, 1, torch.cuda.current_stream())
    torch.cuda.synchronize()
    got = bufs[4].float().cpu()
    worst = (got - ref).abs().max().item() / ref.abs().max().clamp_min(
        1e-12).item()
    if not torch.isfinite(got).all() or worst >= 2.5e-2:
        raise RuntimeError(
            f"w8 POOL parity FAILED: worst_rel={worst:.3e} "
            f"(gate 2.5e-2) — slot/stride/staging sizing bug")
    return worst


def _w8_pool_gate(fp8_tier, k13: int, n13: int) -> None:
    """One-shot wrapper around w8_pool_self_test: runs the gate the first
    time an FP8 tier with a non-empty pool has a staged expert plane, then
    latches. Failures are raised loudly (boot attribution)."""
    global _w8_pool_gate_done, _w8_pool_gate_skip_logged, _w8_gate_dims
    _w8_gate_dims = (k13, n13)
    if _w8_pool_gate_done:
        return
    if fp8_tier is None or fp8_tier.n_slots <= 0:
        # Tier not built yet, or auto-deferred / zero-slot: try again on a
        # later layer (the latch is NOT set). Do NOT stay silent -- emit a
        # one-shot WARNING so the inactive condition is attributable in the
        # boot log (the build-path call sites gate on fp8_tier is not None,
        # so in practice this branch is the AUTO-DEFERRED cold pool; a
        # genuinely never-staged tier is caught HARD by _fp8_boot_guard at
        # first forward).
        from vllm.model_executor.layers.quantization.utils import moe_w2_delta
        if (not _w8_pool_gate_skip_logged) and moe_w2_delta.fp8_enabled():
            _w8_pool_gate_skip_logged = True
            logger.warning(
                "moe_w2_cubit: VLLM_MOE_W2_FP8_DELTA=1 but the w8 POOL "
                "parity gate could NOT run at build (fp8_tier=%s, "
                "n_slots=%s) -- end-to-end pool slot/stride sizing is "
                "UNVERIFIED (auto-deferred cold pool, or a build path "
                "skipped the gate). _fp8_boot_guard reports at first "
                "forward.",
                "None" if fp8_tier is None else "built",
                "n/a" if fp8_tier is None else fp8_tier.n_slots)
        return
    worst = w8_pool_self_test(fp8_tier, k13, n13)
    _w8_pool_gate_done = True
    logger.info(
        "moe_w2_cubit: sm_89 FP8 POOL parity OK: worst_rel=%.3e "
        "(gate 2.5e-2); w8 slot/stride sizing verified end-to-end through "
        "pool slot %d", worst, fp8_tier.n_slots - 1)


def w8_pool_gate_deferred() -> None:
    """Run the one-shot w8 POOL parity gate for a DEFERRED pool: at build time
    the FP8 tier had 0 slots (allocation deferred to finalize_auto — see
    DeltaTier defer_pool), so _w8_pool_gate could only stash the gate dims.
    The worker calls this right after finalize_auto (pool real, still before
    cudagraph capture). Parity failure raises — a mis-sized slot would serve
    garbage bytes, so it must abort the boot, attributed."""
    from vllm.model_executor.layers.quantization.utils import moe_w2_delta
    if _w8_pool_gate_done or not moe_w2_delta.fp8_enabled():
        return
    tier = moe_w2_delta._TIER
    if tier is None or tier.n_slots <= 0 or _w8_gate_dims is None:
        return
    _w8_pool_gate(tier, *_w8_gate_dims)


# One-shot latch for the post-load FP8 boot guard. Runs at the FIRST forward
# (after model load / finalize_auto, before real serving traffic -- vLLM
# warmup exercises it, so a raise aborts the boot, attributed).
_fp8_boot_guard_done = False


def _fp8_boot_guard() -> None:
    """LOUD boot guard: ``VLLM_MOE_W2_FP8_DELTA=1`` but the FP8 delta tier
    was never created/staged -> the build path did not wire FP8, prefill
    silently runs on bare 2-bit, and a test cycle burns against an
    FP8-OFF baseline. Fail loudly instead of silently skipping.

    Decisive signal: ``moe_w2_delta._TIER`` (the FP8 singleton) must EXIST
    and have at least one staged layer (``add_layer_host_planes`` ran). This
    is exactly what the DS4-Flash mxfp4 path missed before build_layer_planes
    wired the FP8 tier -- _TIER stayed None, the standalone w8 self-test
    passed (it validates the kernel in isolation), and the resident tier/pool
    was never built. Tolerates a legit auto-deferred cold pool (n_slots == 0
    pending finalize_auto): the tier EXISTS and is staged, so the guard
    passes; the POOL parity gate is reported separately (info/warn).
    """
    global _fp8_boot_guard_done, _w8_pool_gate_done
    if _fp8_boot_guard_done:
        return
    _fp8_boot_guard_done = True
    from vllm.model_executor.layers.quantization.utils import moe_w2_delta
    if not moe_w2_delta.fp8_enabled():
        return
    tier = moe_w2_delta._TIER
    staged = 0 if tier is None else len(tier._store)
    if tier is None or staged == 0:
        raise RuntimeError(
            "moe_w2: VLLM_MOE_W2_FP8_DELTA=1 but the FP8 delta tier was "
            "never created/staged -- FP8 prefill is INACTIVE (the build "
            "path did not wire FP8). Set VLLM_MOE_W2_FP8_DELTA=0 or fix "
            "the build path (build_layer_planes / build_layer_planes_fp8 / "
            "build_layer_planes_nvfp4 must call _fp8_tier_for_build + "
            "add_layer_host_planes for the mxfp4/fp8/nvfp4 checkpoint "
            "formats respectively).")
    if _w8_pool_gate_done:
        logger.info(
            "moe_w2: FP8 BOOT GUARD PASSED -- delta tier active (%d layers "
            "staged, %d slots, POOL parity verified); FP8 prefill is LIVE",
            staged, tier.n_slots)
    elif tier.n_slots == 0 and getattr(tier, "_auto_pending", False):
        logger.info(
            "moe_w2: FP8 BOOT GUARD: delta tier staged (%d layers); pool "
            "allocation DEFERRED to post-KV finalize_auto — the POOL parity "
            "gate runs there (w8_pool_gate_deferred)", staged)
    else:
        logger.warning(
            "moe_w2: FP8 BOOT GUARD: delta tier active (%d layers staged, "
            "%d slots) but the w8 POOL parity gate did NOT run "
            "(auto-deferred pool, or a build path skipped the gate) -- "
            "end-to-end slot/stride sizing is UNVERIFIED; a sizing "
            "regression would not be caught until prefill", staged,
            tier.n_slots)


def _require_kernels(K13: int, K2: int, need_w4: bool) -> None:
    """Fail loudly at weight load when the cubins this model's shapes need are
    missing from _DIR (they are loaded opportunistically in _ensure_ready)."""
    from vllm.model_executor.layers.quantization.utils import moe_w2_delta
    need = [("w2", K13), ("w2", K2), ("w2mc4", K13), ("w2mc4", K2)]
    if need_w4:
        w4tier = "w4q" if moe_w2_delta.split_enabled() else "w4"
        need += [(w4tier, K13), (w4tier, K2)]
    if moe_w2_delta.fp8_enabled():
        # FP8-e4m3 delta tier (mutually exclusive with FP4; the w8 Triton
        # launchers are registered by make_launchers under VLLM_MOE_W2_FP8_DELTA).
        need += [("w8", K13), ("w8", K2)]
    missing = [f"{t}_k{k}" for t, k in need if (t, k) not in _fns]
    if _sm89:
        # Ada: there is no cubin dir to fix — either an FP4 delta tier is
        # enabled without its SM89 Triton port, or K13/K2 fell outside
        # VLLM_MOE_W2_KS. Name the actual remedy.
        assert not missing, (
            f"moe_w2 sm_89: no kernels for {missing} (K13={K13}, K2={K2})."
            f" w4 full-FP4 is not ported to Ada; w4q split-FP4 is "
            f"experimental behind VLLM_MOE_W2_SM89_W4Q=1. Set "
            f"VLLM_MOE_W2_DELTA=0 (or VLLM_MOE_W2_DELTA_GB=0), or enable "
            f"the w4q port for split-FP4, and make sure K13/K2 are in "
            f"VLLM_MOE_W2_KS. See docs/ada-sm89-port.md")
    assert not missing, (
        f"moe_w2_cubit: missing cubins for K13={K13}/K2={K2}: {missing} "
        f"(dir {_DIR}; set VLLM_MOE_W2_CUBIT_DIR)")


def _fp4_tier_for_build(E: int, dev, n13k13: int, n2k2: int):
    """FP4 delta tier sized for this model's PER-RANK shapes (n13k13 =
    N13*K13, n2k2 = N2*K2 elements). Over the base cache the FP4 slots must
    carry their OWN block-32 scale sections ([fp4_13|sc13|fp4_2|sc2]) — the
    base planes, and with them the GPU-resident scale planes the standalone
    delta shares, are host-resident there. Split mode (DELTA_SPLIT): slots
    hold RADIX-5 quintal planes (2.5 bits/elem, bit-exact e2m1 — see
    moe_w2_planes.pack_quintal_fragment_major) and NO scale sections even
    over the base cache — the quintal kernel reads class/sign/scales from
    the base slot the refinement is residency-coupled to."""
    from vllm.model_executor.layers.quantization.utils import moe_w2_delta
    split = moe_w2_delta.split_enabled()
    sc13, sc2 = ((n13k13 // 32, n2k2 // 32)
                 if moe_w2_delta.base_enabled() and not split else (0, 0))
    if split:
        return moe_w2_delta.get_tier(n_experts=E, dev=dev,
                                     w13_bytes=n13k13 * 5 // 16,
                                     w2_bytes=n2k2 * 5 // 16)
    return moe_w2_delta.get_tier(n_experts=E, dev=dev,
                                 w13_bytes=n13k13 // 2 + sc13,
                                 w2_bytes=n2k2 // 2 + sc2)


def _fp8_tier_for_build(E: int, dev, n13k13: int, n2k2: int):
    """FP8-e4m3 delta tier sized for this model's PER-RANK shapes (n13k13 =
    N13*K13, n2k2 = N2*K2 elements). Slot layout depends on storage mode:
    fp8-store: 1 byte/elem fragment-major e4m3 (w13_bytes=n13k13);
    fp4-store: 0.5 byte/elem fragment-major e2m1 nibble plane (w13_bytes=
    n13k13//2) — the kernel decodes nibble->e4m3 in registers. NO scale
    sections in either mode — the UE8M0 block-32 scale plane is the SHARED
    resident base scale. Returns None when the FP8 delta tier is disabled."""
    from vllm.model_executor.layers.quantization.utils import moe_w2_delta
    if not moe_w2_delta.fp8_enabled():
        return None
    # fp8=True: this is the ONLY call site permitted to construct/receive
    # the FP8 singleton tier (see the poisoning guard in get_tier). The FP4
    # call site _fp4_tier_for_build runs BEFORE this one and would otherwise
    # poison _TIER with FP4 half-sizes.
    half = 2 if moe_w2_delta.fp8_store_fp4() else 1
    # Exact-primary has no resident base scales to share.  Keep the original
    # block-32 UE8M0 scales in each slot beside the checkpoint FP4 nibbles:
    # [fp4_13|sc13|fp4_2|sc2].
    sc13 = n13k13 // 32 if moe_w2_delta.exact_cache_enabled() else 0
    sc2 = n2k2 // 32 if moe_w2_delta.exact_cache_enabled() else 0
    return moe_w2_delta.get_tier(n_experts=E, dev=dev,
                                 w13_bytes=n13k13 // half + sc13,
                                 w2_bytes=n2k2 // half + sc2, fp8=True)


def _stage_fp8_host(tier, layer_key: int, fp13, sc13, fp2, sc2) -> None:
    """Stage the w8 tier using its mode-specific slot geometry."""
    from vllm.model_executor.layers.quantization.utils import moe_w2_delta
    if moe_w2_delta.exact_cache_enabled():
        tier.add_layer_host_sections(layer_key, (fp13, sc13), (fp2, sc2))
    else:
        tier.add_layer_host_planes(layer_key, fp13, fp2)


def _stage_fp4_host(tier, layer_key: int, fp13, sc13, fp2, sc2) -> None:
    """Stage a layer's FP4 planes into the tier's pinned host store; over the
    base cache the scale planes ride along inside the slot sections (copied
    section-by-section — no GPU-side cat temporaries). Split slots carry
    refinement only — their scales live in the coupled base slot."""
    from vllm.model_executor.layers.quantization.utils import moe_w2_delta
    if moe_w2_delta.base_enabled() and not moe_w2_delta.split_enabled():
        tier.add_layer_host_sections(layer_key, (fp13, sc13), (fp2, sc2))
    else:
        tier.add_layer_host_planes(layer_key, fp13, fp2)


def _pack_fp4_plane(nib):
    """One expert's FP4-tier plane row from its e2m1 nibbles: the full
    fragment-major nibble plane (moe_w4_mm), or — split mode — the radix-5
    QUINTAL plane (moe_w4q_mm reads it alongside the resident base;
    bit-exact e2m1 at 2.5 bits/elem)."""
    from vllm.model_executor.layers.quantization.utils import moe_w2_delta
    from vllm.model_executor.layers.quantization.utils.moe_w2_planes import (
        pack_fp4_fragment_major, pack_quintal_fragment_major)
    if moe_w2_delta.split_enabled():
        return pack_quintal_fragment_major(nib)
    return pack_fp4_fragment_major(nib)


def _fp4_plane_nbytes(n: int, k: int) -> int:
    """Per-expert FP4-tier plane bytes for one [n, k] matrix: nibbles
    (n*k/2) or the split quintal plane (n*k*5/16)."""
    from vllm.model_executor.layers.quantization.utils import moe_w2_delta
    return n * k * 5 // 16 if moe_w2_delta.split_enabled() else n * k // 2


def _mxfp4_fp8_unit_bytes(nib, ck_scale_bytes, serving_scale_bytes):
    """UNIT-SPACE e4m3 bytes [N, K] u8 for one mxfp4 expert, for the FP8
    delta tier's slot content.

    The moe_w8 kernel folds the SHARED resident base scale -- the same
    UE8M0 block-32 plane (``sc13``/``sc2``) the 2-bit tier reads -- so the
    slot's e4m3 bytes must be the true weight DIVIDED by that serving
    scale. For mxfp4 the true weight is ``e2m1_val(nib) * 2^(ck-127)``;
    dividing by the serving scale (the checkpoint scale, or the
    refit-decremented scale when SCALE_REFIT is active) yields the
    unit-space ``u = e2m1_val * 2^(ck - serving)``.

    This is the EXACT ``want_fp8=True`` math of ``_f64_to_codes_scales``
    (``u = wb / 2^exp; fp8 = u.clamp(-448, 448).to(e4m3)``), using the
    tier's serving scale instead of recomputing ``exp`` so the shared
    UE8M0 plane still reconstructs the weight (the mxfp4 2-bit path keeps
    the checkpoint scale, unlike the fp8-checkpoint path which recomputes
    it -- so we must NOT call ``_f64_to_codes_scales`` here). f32 math is
    exact for the e2m1 grid and the +/-1 exponent deltas. Because ``u`` is
    derived from the same requant as the 2-bit codes (incl. any refit),
    the FP8 tier has NO scale-refit conflict (per
    ``pack_fp8_fragment_major``'s contract).
    """
    from vllm.model_executor.layers.quantization.utils.moe_w2_planes import (
        _E2M1_VALS)
    N, K = nib.shape
    w = _E2M1_VALS.to(nib.device)[nib.long()].view(N, K // 32, 32)
    ck = ck_scale_bytes.to(torch.float32).view(N, K // 32, 1)
    sv = serving_scale_bytes.to(torch.float32).view(N, K // 32, 1)
    u = w.to(torch.float32) * torch.exp2(ck - sv)
    return (u.reshape(N, K).clamp(-448.0, 448.0)
            .to(torch.float8_e4m3fn).view(torch.uint8))


# Loader-level skip (the planes-cache/pack "v1.5 follow-up"): layers the
# pack already serves never need their checkpoint experts in host RAM at
# all. Decided at CREATE time (sidecar probe), executed by stubbing the big
# params + no-op'ing their weight loaders — the vLLM loader then streams
# past those tensors without allocating or copying. Measured motivation
# (GLM-5.2 TP2 boot-from-pack): ~190 s of giant host allocations before the
# shard read, 222 s of staged copies during it, and a ~0.5 TB transient
# that intermittently OOM'd the box when boots overlapped.
_n_created = 0
_skip_logged = False
_pack_io_skip_logged = False
_pack_io_skip_installed = False
_pack_io_skip_layers: set[int] = set()
_PACK_EXPERT_WEIGHT_RE = re.compile(
    r"(?:^|\.)layers\.(\d+)\.ffn\.experts\.\d+\."
    r"w[123]\.(?:weight|scale)$"
)


def _pack_io_should_skip(weight_name: str) -> bool:
    """True only for raw routed-expert tensors whose transformer layer is
    already validated in the persistent pack/planes cache."""
    match = _PACK_EXPERT_WEIGHT_RE.search(weight_name)
    return (match is not None
            and int(match.group(1)) in _pack_io_skip_layers)


def _install_pack_io_skip(layer_idx: int) -> None:
    """Teach the ordinary safetensors iterator to reject a cached layer's
    expert payloads before ``safe_open.get_tensor`` reads them.

    ``plan_pack_skip`` calls this only after the identity- and geometry-keyed
    pack/planes probe succeeds.  The wrapper preserves vLLM's EP filtering and
    changes no non-expert, shared-expert, MTP, or uncached-layer name.
    """
    global _pack_io_skip_installed, _pack_io_skip_logged
    _pack_io_skip_layers.add(int(layer_idx))
    if _pack_io_skip_installed:
        return
    from vllm.model_executor.model_loader import weight_utils as _weight_utils

    original = _weight_utils.should_skip_weight

    @functools.wraps(original)
    def cached_or_original(weight_name, local_expert_ids):
        if _pack_io_should_skip(weight_name):
            return True
        return original(weight_name, local_expert_ids)

    _weight_utils.should_skip_weight = cached_or_original
    _pack_io_skip_installed = True
    if not _pack_io_skip_logged:
        _pack_io_skip_logged = True
        logger.info(
            "moe_w2 PACK I/O-SKIP installed: identity-validated routed "
            "expert tensors are filtered before safetensors get_tensor")


def _noop_loader(*args, **kwargs):
    """Weight-loader stand-in for pack-skipped params: the expert loading
    loop calls with return_success=True and must see truthy, or it treats
    the shard as unmapped and keeps probing replicas."""
    return True if kwargs.get("return_success") else None


def plan_pack_skip(layer) -> bool:
    """CREATE-time twin of the boot-from-cache paths: assign this layer's
    key (the same build-order counter process_weights_after_loading uses),
    probe whichever store this config will serve from — the pack sidecars
    (BASE cache / host-resident) or the planes cache (GPU-resident) — and
    when the layer is already served, stub the four big params and disarm
    their loaders. Returns True when the layer boots with zero checkpoint
    staging."""
    from vllm.model_executor.layers.quantization.utils import moe_w2_delta
    from vllm.model_executor.layers.quantization.utils import (
        moe_w2_planes_cache as _pc)
    from vllm.model_executor.layers.quantization.utils.moe_w2_store import (
        pack_has_layer)
    global _n_created
    key = _n_created
    _n_created += 1
    if key == 0:
        # A process normally constructs one target model, but clearing here
        # keeps an in-process rebuild/reload from inheriting stale layer IDs.
        _pack_io_skip_layers.clear()
    layer._moe_w2_create_key = key
    if not enabled():
        return False
    try:
        E, N13, K13h = layer.w13_weight.shape
        _, N2, K2h = layer.w2_weight.shape
    except Exception:  # noqa: BLE001 - unexpected layout: stage as before
        return False
    K13, K2 = K13h * 2, K2h * 2
    c13len, s13len = N13 * K13 // 4, N13 * K13 // 32
    c2len, s2len = N2 * K2 // 4, N2 * K2 // 32
    if moe_w2_delta.exact_cache_enabled():
        n_keys = _layer_cutoff() + 1
        exact_slot = ((N13 * K13 // 2 + s13len)
                      + (N2 * K2 // 2 + s2len))
        if not pack_has_layer("w8x", key, n_keys, E, exact_slot):
            return False
    elif moe_w2_delta.base_enabled():
        # host-resident: the pack store serves the base (and the FP4
        # need-pool when configured) — probe the sidecars.
        n_keys = _layer_cutoff() + 1
        if not pack_has_layer("base", key, n_keys, E,
                              c13len + s13len + c2len + s2len):
            return False
        if moe_w2_delta.enabled():
            # over-base FP4 need-pool: mirror _fp4_tier_for_build's sizing
            # and get_tier's pack tag (split quintal slots live in a
            # SEPARATE pack — different geometry, "fp4q")
            if moe_w2_delta.split_enabled():
                ftag = "fp4q"
                fslot = N13 * K13 * 5 // 16 + N2 * K2 * 5 // 16
            else:
                ftag = "fp4"
                fslot = (N13 * K13 // 2 + s13len) + (N2 * K2 // 2 + s2len)
            if not pack_has_layer(ftag, key, n_keys, E, fslot):
                return False
    else:
        # GPU-resident: the planes cache is the only source that can
        # replace the checkpoint requant — probe it (keyed by transformer
        # layer index, sized exactly like process-time try_load). The probe
        # MUST include the FP8 delta parts (fp8u13/fp8u2) when the FP8 tier
        # is armed: an old cache written without them is a MISS (rebuilds),
        # but a cache that has them lets an FP8-enabled boot skip the
        # checkpoint staging too -- the build path serves FP8 from the cache
        # via _consume_planes_cache. (mxfp4/DS4 never reaches plan_pack_skip
        # -- it streams via arm_stream_build -- so this only lifts the
        # nvfp4/fp8-block GPU-resident loader skip under FP8.)
        lidx = _pc.layer_idx_from_name(getattr(layer, "layer_name", ""))
        if lidx is None or not _pc.cache_has_layer(
                lidx, _pc.expected_sizes(
                    E, N13, K13, N2, K2,
                    want_fp4=moe_w2_delta.enabled(),
                    want_fp8=moe_w2_delta.fp8_enabled())):
            return False
    for pname in ("w13_weight", "w13_weight_scale",
                  "w2_weight", "w2_weight_scale"):
        p = getattr(layer, pname)
        p.data = torch.empty(0, dtype=p.data.dtype, device="cpu")
        p.weight_loader = _noop_loader
    layer._moe_w2_shapes = (E, N13, K13, N2, K2)
    layer._moe_w2_pack_skip = True
    lidx = _pc.layer_idx_from_name(getattr(layer, "layer_name", ""))
    if lidx is not None:
        _install_pack_io_skip(lidx)
    else:
        logger.warning(
            "moe_w2: pack layer key %d has no transformer layer index; "
            "parameter staging is skipped but checkpoint I/O cannot be "
            "filtered safely", key)
    global _skip_logged
    if not _skip_logged:
        _skip_logged = True
        logger.info(
            "moe_w2 LOADER-SKIP armed: pack-resident expert layers are "
            "neither host-staged nor copied from the checkpoint "
            "(first: key %d)", key)
    logger.debug("moe_w2: layer key %d loader-skipped", key)
    return True


# ---- streaming FIRST boot (VLLM_MOE_W2_STREAM_BUILD, default on) ---------
# With no pack and no planes cache to skip from, the loader used to stage
# the FULL expert checkpoint in host RAM before a single layer was
# requantized — ~400+ GB transient on GLM-5.2, reported as a 4-5 h
# swap-through first boot on a 354 GB host. Streaming build requants each
# layer the moment its LAST expected expert tensor lands (exact per-param
# load counting, no ordering assumptions) and stubs its staging right
# after: peak staging = O(one layer) ≈ 6 GB instead of the checkpoint.
# The GPU is idle during load anyway, so the per-layer requant overlaps
# shard I/O instead of serializing after it. VLLM_MOE_W2_STREAM_BUILD=0
# restores the stage-everything-then-build behaviour.
_STREAM = os.getenv("VLLM_MOE_W2_STREAM_BUILD", "1") == "1"
_stream_logged = False


class _StreamLoader:
    """Per-param weight_loader wrapper: counts SUCCESSFUL (expert, shard)
    loads and triggers the layer build when every big param is complete.
    A load arriving after the build would be silent data loss (the params
    are stubs by then) — fail loudly instead."""

    def __init__(self, layer, pname, inner):
        self._layer = layer
        self._pname = pname
        self._inner = inner

    def __call__(self, param, loaded_weight, *args, **kwargs):
        # LAZY staging: the create hook left this param as a 0-byte stub
        # (allocating every layer's host buffer up front peaked at ~300+ GB
        # before the first shard was even read — measured). Materialize the
        # layer's buffer the moment its first tensor arrives; with a
        # layer-major checkpoint only a couple of layers are ever
        # in-flight, an unordered one merely degrades to the old profile.
        if param.data.numel() == 0:
            shape = self._layer._moe_w2_stream_shapes[self._pname]
            param.data = torch.empty(shape, dtype=param.data.dtype,
                                     device="cpu")
        ret = self._inner(param, loaded_weight, *args, **kwargs)
        ok = (ret is True) if kwargs.get("return_success") else True
        if not ok:
            return ret
        pend = self._layer._moe_w2_pending
        if pend.get(self._pname, 0) <= 0:
            raise RuntimeError(
                f"moe_w2 stream-build: {self._pname} load arrived after "
                f"the layer was already built — more (expert, shard) "
                f"tensors than expected; set VLLM_MOE_W2_STREAM_BUILD=0 "
                f"and report the checkpoint")
        pend[self._pname] -= 1
        if all(v == 0 for v in pend.values()):
            # nvfp4 layers got their key from the create-time planner
            # (plan_pack_skip always runs first there); mxfp4 layers have
            # no planner — build order IS the key, as their classic
            # process_weights path always did.
            key = getattr(self._layer, "_moe_w2_create_key", None)
            if key is None:
                key = len(_LAYERS)
            self._layer._moe_w2_stream_key = key
            self._layer._moe_w2_stream_builder(self._layer, key)
            # Drop the staging storage IN PLACE on the ORIGINAL Parameter
            # objects. _finish_layer replaced the layer's attributes with
            # stub Parameters, but load_weights' params_dict (built once,
            # up front) still references the originals for the rest of the
            # load — without this, nothing frees until load_weights returns
            # and the "streaming" peak is the whole checkpoint again
            # (measured: 517 GB at 29/47 shards on GLM TP2).
            for p in self._layer._moe_w2_stream_orig:
                p.data = torch.empty(0, dtype=p.data.dtype, device="cpu")
            self._layer._moe_w2_stream_orig = ()
            self._layer._moe_w2_stream_built = True
            logger.debug("moe_w2: layer key %d stream-built during load",
                         key)
        return ret


_full_staging_warned = False


def warn_full_staging(layer, why: str) -> None:
    """One-time LOUD warning when a w2 layer falls back to classic
    full-checkpoint host staging (stream-build did not arm). The silent
    version of this state OOM-killed three Ada deploy cycles in a row:
    every expert tensor of every layer accumulates in host RAM before
    the first requant, and on a host smaller than the checkpoint the
    kernel's OOM-killer ends the engine with no Python traceback."""
    global _full_staging_warned
    if _full_staging_warned:
        return
    _full_staging_warned = True
    per_layer = 0
    for pname in ("w13_weight", "w13_weight_scale", "w2_weight",
                  "w2_weight_scale"):
        p = getattr(layer, pname, None)
        if p is not None:
            per_layer += p.data.numel() * p.data.element_size()
    total_gib = per_layer * _layer_cutoff() / 2**30
    avail = ""
    try:
        with open("/proc/meminfo") as fh:
            kb = {l.split(":")[0]: int(l.split()[1]) for l in fh}
        avail = (f"; MemAvailable now "
                 f"{kb.get('MemAvailable', 0) / 2**20:.1f} GiB")
    except Exception:  # noqa: BLE001 — diagnostics must not fail the boot
        pass
    logger.warning(
        "moe_w2: stream-build NOT armed (%s) — the FULL expert checkpoint "
        "(~%.1f GiB over %d layers) will stage in host RAM before the "
        "first requant%s. On a smaller host this ends as a tracebackless "
        "OOM-kill mid-load. Expected on exotic layouts only; report the "
        "checkpoint format if you see this on DS4/GLM/Kimi.",
        why, total_gib, _layer_cutoff(), avail)


def arm_stream_build(layer) -> bool:
    """Arm the streaming per-layer build on a layer plan_pack_skip missed
    (first boot, or a store that does not yet hold it). Expected loads per
    param: w13-side shards land twice per expert (w1, w3), w2-side once —
    exact counts over the layout's FULL param set (nvfp4: six incl. scale_2:
    building before they land would bake uninitialized per-tensor scales
    into the planes AND the caches), so completeness needs no
    checkpoint-ordering assumption.

    Staging is LAZY: the four big params become 0-byte stubs here and a
    layer's buffers materialize on its FIRST loaded tensor (the up-front
    create_weights allocation of every layer peaked at ~300 GB before the
    first shard was read — measured), then drop IN PLACE at build (the
    attribute swap alone frees nothing: load_weights' params_dict holds
    the original objects until the load returns — measured 517 GB).
    Returns True when armed (the caller then skips its own staging).
    No-op unless VLLM_MOE_W2_STREAM_BUILD=1 (default)."""
    global _stream_logged
    if not (_STREAM and enabled()):
        return False
    try:
        E = layer.w13_weight.shape[0]
    except Exception:  # noqa: BLE001 - unexpected layout: staged path
        return False
    if hasattr(layer, "w13_weight_scale_2"):
        # NVFP4 (modelopt) layout — per-tensor scale_2 params ride along
        expected = {"w13_weight": 2 * E, "w13_weight_scale": 2 * E,
                    "w13_weight_scale_2": 2 * E,
                    "w2_weight": E, "w2_weight_scale": E,
                    "w2_weight_scale_2": E}
        builder = build_layer_planes_nvfp4
    else:
        # mxfp4 layout (DS4-Flash official checkpoints): same shard
        # cardinality (w13 lands twice per expert - w1, w3 - w2 once),
        # no scale_2. Until this branch existed, mxfp4 checkpoints
        # SILENTLY fell back to staging the full expert checkpoint in
        # host RAM (~140+ GB on DS4-Flash) - an OOM-kill with no
        # traceback on any host smaller than the checkpoint, and the
        # root cause of three burned Ada deploy cycles.
        expected = {"w13_weight": 2 * E, "w13_weight_scale": 2 * E,
                    "w2_weight": E, "w2_weight_scale": E}
        builder = build_layer_planes
    big = ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
    wrappers = {}
    for pname in expected:
        p = getattr(layer, pname, None)
        inner = getattr(p, "weight_loader", None)
        if p is None or inner is None:
            return False                # leave the layer fully staged
        wrappers[pname] = (p, _StreamLoader(layer, pname, inner))
    layer._moe_w2_pending = expected
    layer._moe_w2_stream_builder = builder
    # originals of the BIG params: their storage is dropped in place at
    # build time, and their SHAPES feed the lazy materialization
    layer._moe_w2_stream_orig = tuple(getattr(layer, p) for p in big)
    layer._moe_w2_stream_shapes = {
        p: tuple(getattr(layer, p).shape) for p in big}
    for pname in big:                   # lazy: nothing staged until loaded
        p = getattr(layer, pname)
        p.data = torch.empty(0, dtype=p.data.dtype, device="cpu")
    for pname, (p, wrap) in wrappers.items():
        p.weight_loader = wrap
    if not _stream_logged:
        _stream_logged = True
        logger.info(
            "moe_w2 STREAM-BUILD armed: layer staging materializes on its "
            "first loaded tensor and requants on its last (peak staging = "
            "layers in flight, not the checkpoint); "
            "VLLM_MOE_W2_STREAM_BUILD=0 restores the old path")
    return True


def _try_skip_requant(layer, layer_key: int, E: int, N13: int, K13: int,
                      N2: int, K2: int, param_names) -> bool:
    """Boot-from-pack: when every host store this config serves from already
    holds this layer's rows (valid pack written by a previous boot), the
    dequant->requant of the checkpoint experts produces bytes NOBODY reads —
    the base planes live in the pack, and so do the FP4 need-pool sections.
    Skip it: register the layer's slot-layout metadata and the param stubs
    exactly as _finish_layer's base path would, and let the tiers serve
    from the pack. On GLM the requant is the dominant boot cost (NVFP4 ->
    f64 -> 2-bit, hundreds of GiB of transients); with the pack it reduces
    to open+read.

    Only applies over the base cache (base_enabled): the GPU-resident plane
    path needs the planes materialized regardless. A PinnedHostStore never
    contains layers at boot -> configs without VLLM_MOE_W2_STORE_DIR are
    untouched. Layers absent from the pack (e.g. the MTP drafter, or a
    partially written pack) requant as before. When the FP4 tier is enabled
    but ITS pack misses the layer, we also requant (the fp4 sections can
    only be rebuilt from the checkpoint bytes)."""
    from vllm.model_executor.layers.quantization.utils import moe_w2_delta
    if moe_w2_delta.exact_cache_enabled():
        dev = torch.device("cuda")
        tier = _fp8_tier_for_build(E, dev, N13 * K13, N2 * K2)
        if tier is None or layer_key not in tier._store:
            return False
        f13len, s13len = N13 * K13 // 2, N13 * K13 // 32
        f2len, s2len = N2 * K2 // 2, N2 * K2 // 32
        from vllm.model_executor.layers.quantization.utils import (
            moe_w2_planes_cache as _pc)
        _LAYERS[layer_key] = dict(
            N13=N13, K13=K13, N2=N2, K2=K2, E=E, exact=True,
            tl_idx=_pc.layer_idx_from_name(getattr(layer, "layer_name", "")),
            off_s13=f13len, off_c2=f13len + s13len,
            off_s2=f13len + s13len + f2len,
        )
        _w8_pool_gate(tier, K13, N13)
        stub = torch.empty(0, dtype=torch.uint8, device=dev)
        for name in param_names:
            layer.register_parameter(
                name, torch.nn.Parameter(stub, requires_grad=False))
        logger.info(
            "moe_w2: layer %d requant SKIPPED — exact FP4-storage cache "
            "serving from pack (boot-from-pack)", layer_key)
        return True
    if not moe_w2_delta.base_enabled():
        return False
    dev = torch.device("cuda")
    c13len, s13len = N13 * K13 // 4, N13 * K13 // 32
    c2len, s2len = N2 * K2 // 4, N2 * K2 // 32
    btier = moe_w2_delta.get_base_tier(
        _layer_cutoff() + 1, E, dev,
        w13_bytes=c13len + s13len, w2_bytes=c2len + s2len)
    if layer_key not in btier._store:
        return False
    tier = _fp4_tier_for_build(E, dev, N13 * K13, N2 * K2)
    if tier is not None and layer_key not in tier._store:
        return False
    from vllm.model_executor.layers.quantization.utils import (
        moe_w2_planes_cache as _pc)
    _LAYERS[layer_key] = dict(
        N13=N13, K13=K13, N2=N2, K2=K2, E=E, base=True,
        tl_idx=_pc.layer_idx_from_name(getattr(layer, "layer_name", "")),
        off_s13=c13len, off_c2=c13len + s13len,
        off_s2=c13len + s13len + c2len,
        off4_s13=2 * c13len, off4_c2=2 * c13len + s13len,
        off4_s2=2 * c13len + s13len + 2 * c2len,
    )
    stub = torch.empty(0, dtype=torch.uint8, device=dev)
    for name in param_names:
        layer.register_parameter(
            name, torch.nn.Parameter(stub, requires_grad=False))
    logger.info(
        "moe_w2: layer %d requant SKIPPED — %s serving from pack "
        "(boot-from-pack)", layer_key,
        "base+fp4" if tier is not None else "base")
    return True


_budget_logged = False


def _log_plane_budget(E: int, N13: int, K13: int, N2: int, K2: int) -> None:
    """One-time projection of the FULL model's plane footprint at the
    FIRST plane build — observability bought before the failure, not
    after: resident mode on a card that cannot possibly hold the planes
    used to walk ~40 layers of torch.empty into a bare CUDA OOM whose
    traceback names nothing moe_w2. Now the impossible config fails here
    with the remedy, the merely-tight one warns, and every boot logs one
    budget line (also the "moe_w2 is ACTIVE" beacon — the stock backend
    banner, e.g. "Using 'MARLIN' Mxfp4 MoE backend", still prints and
    routinely misleads readers of the log)."""
    global _budget_logged
    if _budget_logged:
        return
    _budget_logged = True
    from vllm.model_executor.layers.quantization.utils import moe_w2_delta
    exact = moe_w2_delta.exact_cache_enabled()
    per_layer = E * (N13 * K13 + N2 * K2) * (
        (1 / 2 if exact else 1 / 4) + 1 / 32)
    n_layers = _layer_cutoff()
    try:
        from vllm.config import get_current_vllm_config
        pp = get_current_vllm_config().parallel_config.pipeline_parallel_size
        n_layers = -(-n_layers // max(pp, 1))
    except Exception:  # noqa: BLE001 — PP unknown: project full depth
        pass
    total_gib = per_layer * n_layers / 2**30
    if moe_w2_delta.base_enabled() or exact:
        logger.info(
            "moe_w2 planes: %.2f GiB/layer x %d layers = %.1f GiB -> "
            "PINNED HOST RAM (%s; the GPU holds the expert slot pool)",
            per_layer / 2**30, n_layers, total_gib,
            "exact FP4-storage cache" if exact else "2-bit base cache")
        # host-side preflight, same rationale as the VRAM one: a pinned
        # arena that outgrows host RAM dies mid-load as a bare
        # cudaHostAlloc failure (or the OOM-killer takes the engine with
        # NO traceback at all). Warn with the remedies while the boot log
        # is still short. MemAvailable is the kernel's own estimate;
        # /proc is Linux-only — skip silently elsewhere.
        try:
            with open("/proc/meminfo") as fh:
                kb = {l.split(":")[0]: int(l.split()[1]) for l in fh}
            avail_gib = kb.get("MemAvailable", 0) / 2**20
            if total_gib > avail_gib * 0.9:
                logger.warning(
                    "moe_w2 base cache: projected pinned host planes "
                    "(%.1f GiB) exceed 90%% of MemAvailable (%.1f GiB) — "
                    "expect a host-alloc failure or the OOM-killer "
                    "(engine dies with no traceback). Cap the pinned "
                    "arena with VLLM_MOE_W2_BASE_RAM_GB and give the "
                    "spill an NVMe home via VLLM_MOE_W2_STORE_DIR",
                    total_gib, avail_gib)
        except Exception:  # noqa: BLE001 — a preflight must never be
            pass           # the failure it exists to diagnose
        return
    free_b, total_b = torch.cuda.mem_get_info()
    logger.info(
        "moe_w2 planes: %.2f GiB/layer x %d layers = %.1f GiB "
        "GPU-RESIDENT (device now: %.1f GiB free / %.1f GiB total)",
        per_layer / 2**30, n_layers, total_gib,
        free_b / 2**30, total_b / 2**30)
    if per_layer * n_layers > total_b:
        raise RuntimeError(
            f"moe_w2: GPU-resident planes need ~{total_gib:.1f} GiB but "
            f"this device has {total_b / 2**30:.1f} GiB TOTAL — this "
            "cannot fit at any utilization. Use the base cache "
            "(VLLM_MOE_W2_BASE_CACHE_GB=<pool GiB> keeps planes in pinned "
            "host RAM and streams a GPU slot pool), shard with TP, or "
            "serve a smaller model. See docs/ada-sm89-port.md")
    if per_layer * n_layers > free_b * 0.9:
        logger.warning(
            "moe_w2: projected resident planes (%.1f GiB) exceed 90%% of "
            "currently free VRAM (%.1f GiB) — weight load will likely "
            "OOM; consider VLLM_MOE_W2_BASE_CACHE_GB", total_gib,
            free_b / 2**30)


def _build_exact_mxfp4_layer(layer, layer_key: int, dev, w13, s13, w2, s2,
                             E: int, N13: int, K13: int, N2: int, K2: int,
                             param_names) -> None:
    """Cold-build one checkpoint-MXFP4 layer into the mandatory exact tier.

    Exact mode serves the original checkpoint E2M1 nibbles and UE8M0 scales
    through ``moe_w8``; it never reads the legacy 2-bit planes.  Building
    those planes on GPU first therefore only adds quantization work, VRAM
    transients, and a planes-cache write whose bytes are discarded.  Repack
    the checkpoint bytes directly on CPU into the persistent ``w8x`` slot
    layout instead.  Subsequent boots bypass this function through the pack
    I/O-skip path in ``_try_skip_requant``.
    """
    from vllm.model_executor.layers.quantization.utils import moe_w2_delta
    from vllm.model_executor.layers.quantization.utils.moe_w2_planes import (
        mxfp4_to_nibbles, pack_fp4_fragment_major_batch)

    if not moe_w2_delta.exact_cache_enabled():
        raise RuntimeError("exact MXFP4 builder called outside exact mode")
    tier = _fp8_tier_for_build(E, dev, N13 * K13, N2 * K2)
    if tier is None:
        raise RuntimeError("exact MXFP4 builder has no mandatory w8 tier")

    chunk = max(1, int(os.getenv("VLLM_MOE_W2_EXACT_PACK_CHUNK", "32")))
    started = time.perf_counter()

    def pack_matrix(weight, scale, N, K):
        # Outputs stay pageable CPU here. add_layer_host_sections performs the
        # single copy into the pinned arena (or persistent pack writer), so a
        # cold build does not consume GPU memory beyond the already-budgeted
        # exact slot pool.
        fp = torch.empty(E, N * K // 2, dtype=torch.uint8, device="cpu")
        sc = torch.empty(E, N * K // 32, dtype=torch.uint8, device="cpu")
        for e0 in range(0, E, chunk):
            e1 = min(e0 + chunk, E)
            fp[e0:e1] = pack_fp4_fragment_major_batch(
                mxfp4_to_nibbles(weight[e0:e1]))
            # Batched twin of pack_scales: [B,N,K/32] ->
            # [B,N/16,K/32,16] -> [B,N*K/32].
            b = e1 - e0
            sc[e0:e1] = scale[e0:e1].view(
                b, N // 16, 16, K // 32).transpose(
                    2, 3).contiguous().view(b, -1)
        return fp, sc

    fp13, sc13 = pack_matrix(w13, s13, N13, K13)
    fp2, sc2 = pack_matrix(w2, s2, N2, K2)
    _stage_fp8_host(tier, layer_key, fp13, sc13, fp2, sc2)
    del fp13, sc13, fp2, sc2
    _w8_pool_gate(tier, K13, N13)
    empty = torch.empty(0, dtype=torch.uint8, device=dev)
    _finish_layer(layer, layer_key, dev, empty, empty, empty, empty,
                  N13, K13, N2, K2, E, param_names)
    logger.info(
        "moe_w2 exact cold-pack: layer %d checkpoint FP4+scales packed "
        "directly on CPU in %.2fs (chunk=%d; q2 requant/cache skipped)",
        layer_key, time.perf_counter() - started, chunk)


def build_layer_planes(layer, layer_key: int) -> None:
    """Quantize one FusedMoE layer's experts to 2-bit planes (GPU, chunked).

    Reads the CPU-resident checkpoint params (w13_weight [E,2I,K/2] u8 etc.),
    builds fragment-major code planes + scale planes on the GPU, then
    replaces the originals with empty stubs.
    """
    assert _ensure_ready(), (
        "moe_w2 kernels unavailable — the root cause was logged at "
        "engine init as 'moe_w2_cubit unavailable: ...'")
    dev = torch.device("cuda")
    w13 = layer.w13_weight.data          # [E, 2I, H/2] u8 (cpu)
    s13 = layer.w13_weight_scale.data    # [E, 2I, H/32] u8
    w2 = layer.w2_weight.data            # [E, H, I/2] u8
    s2 = layer.w2_weight_scale.data      # [E, H, I/32] u8
    E, N13, _ = w13.shape
    _, N2, _ = w2.shape
    K13, K2 = N2, N13 // 2               # H, I (4096/2048 on DS4-Flash TP1)
    from vllm.model_executor.layers.quantization.utils import moe_w2_delta
    has_fp4_delta = moe_w2_delta.enabled()
    scale_refit = scale_refit_enabled(has_fp4_delta)
    if scale_refit and has_fp4_delta:
        raise RuntimeError(
            "VLLM_MOE_W2_SCALE_REFIT=1 cannot be used with the FP4 delta "
            "tier because the two tiers currently share scale planes")
    if scale_refit and moe_w2_delta.fp8_enabled() and moe_w2_delta.fp8_store_fp4():
        raise RuntimeError(
            "VLLM_MOE_W2_FP8_STORE=fp4 requires VLLM_MOE_W2_SCALE_REFIT=0: "
            "fp4-storage slots hold the checkpoint e2m1 nibbles verbatim, and "
            "a refit-decremented serving scale would need u = 2*e2m1 — values "
            "the e2m1 grid cannot represent. Set SCALE_REFIT=0 (quality-"
            "prefill mode) or use VLLM_MOE_W2_FP8_STORE=fp8.")
    global _scale_refit_logged
    if scale_refit and not _scale_refit_logged:
        _scale_refit_logged = True
        logger.info(
            "moe_w2 MXFP4 scale refit enabled: each block can decrease its "
            "UE8M0 exponent by one when exact block SSE decreases")
    _require_kernels(K13, K2, need_w4=has_fp4_delta)
    _log_plane_budget(E, N13, K13, N2, K2)
    if _try_skip_requant(layer, layer_key, E, N13, K13, N2, K2,
                         ("w13_weight", "w13_weight_scale", "w2_weight",
                          "w2_weight_scale")):
        return
    if moe_w2_delta.exact_cache_enabled():
        _build_exact_mxfp4_layer(
            layer, layer_key, dev, w13, s13, w2, s2,
            E, N13, K13, N2, K2,
            ("w13_weight", "w13_weight_scale", "w2_weight",
             "w2_weight_scale"))
        return
    # Planes cache (VLLM_MOE_W2_PLANES_CACHE): the mxfp4 requant + FP8
    # derivation below is deterministic given (checkpoint, TP layout, zero
    # mode, scale refit), so cached planes (2-bit base + FP4 + FP8 delta)
    # stream back instead of rebuilding -- the FP8 derivation was the slow
    # step this path re-ran every boot. Mirrors build_layer_planes_nvfp4.
    from vllm.model_executor.layers.quantization.utils import (
        moe_w2_planes_cache as _pc)
    planes_cache_enabled = _pc.enabled()
    lidx = _pc.layer_idx_from_name(getattr(layer, "layer_name", ""))
    planes_cache_store = _pc.store
    if _consume_planes_cache(layer, layer_key, dev, E, N13, K13, N2, K2):
        return

    planes13 = torch.empty(E, N13 * K13 // 4, dtype=torch.uint8, device=dev)
    sc13 = torch.empty(E, N13 * K13 // 32, dtype=torch.uint8, device=dev)
    planes2 = torch.empty(E, N2 * K2 // 4, dtype=torch.uint8, device=dev)
    sc2 = torch.empty(E, N2 * K2 // 32, dtype=torch.uint8, device=dev)

    from vllm.model_executor.layers.quantization.utils.moe_w2_planes import (
        mxfp4_to_nibbles, pack_fp4_fragment_major, pack_fp8_fragment_major)
    # Pass the PER-RANK FP4 plane sizes (N*K//2 bytes/expert) so the delta tier's
    # slots, host store, and pool indexing match the (TP-sharded) planes. On TP1
    # these equal the module constants -> the single-GPU path is unchanged.
    tier = _fp4_tier_for_build(E, dev, N13 * K13, N2 * K2)
    # FP8-e4m3 delta PREFILL tier (Ada native FP8 MMA; mutually exclusive with
    # FP4 -- when FP8 is active the FP4 ``tier`` above is None). Mirrors
    # build_layer_planes_fp8: create the tier, build the unit-space planes,
    # stage them, and run the one-shot POOL parity gate. Without this block
    # DS4-Flash (mxfp4) booted with FP8 ENABLED but INACTIVE -- the tier
    # stayed None, prefill ran on bare 2-bit, and the w8 launcher's standalone
    # parity gate passed but the resident tier/pool was never built.
    fp8_tier = _fp8_tier_for_build(E, dev, N13 * K13, N2 * K2)
    store_fp4 = moe_w2_delta.fp8_store_fp4()
    fp13 = fp2 = None
    if tier is not None:
        # full nibble planes (w4) or quintal planes (w4q, split)
        fp13 = torch.empty(E, _fp4_plane_nbytes(N13, K13),
                           dtype=torch.uint8, device=dev)
        fp2 = torch.empty(E, _fp4_plane_nbytes(N2, K2),
                          dtype=torch.uint8, device=dev)
    fp8_13 = fp8_2 = None
    if fp8_tier is not None:
        # CPU-resident: staged into the HOST arena and promoted into the GPU
        # pool on demand — build transients must NEVER consume VRAM (measured:
        # ~3 GiB/layer FP8 GPU transients OOM'd mxfp4_refit_codes_scales at
        # layer ~41). fp4-store: 0.5 byte/elem nibble plane (N13*K13//2);
        # fp8-store: 1 byte/elem e4m3 byte plane (N13*K13). Derivation is pure
        # element-wise/permutation math -> CPU-correct in both modes.
        n13_elem = N13 * K13 // (2 if store_fp4 else 1)
        n2_elem = N2 * K2 // (2 if store_fp4 else 1)
        fp8_13 = torch.empty(E, n13_elem, dtype=torch.uint8, device="cpu")
        fp8_2 = torch.empty(E, n2_elem, dtype=torch.uint8, device="cpu")

    from vllm.model_executor.layers.quantization.utils.moe_w2_planes import (
        mxfp4_fp8_unit_bytes_batch, pack_fp8_fragment_major_batch,
        pack_fp4_fragment_major_batch)
    # Mode-dependent cache part keys: fp4-store and fp8-store caches coexist
    # in one dir; distinct part names (fp4u13/fp4u2 vs fp8u13/fp8u2) mean a
    # cache written in one mode MISSES in the other and rebuilds, which is the
    # correct behaviour (presence + exact size is the validity contract).
    w8k13, w8k2 = ("fp4u13", "fp4u2") if store_fp4 else ("fp8u13", "fp8u2")
    chunk = 32
    for e0 in range(0, E, chunk):
        e1 = min(e0 + chunk, E)
        wg = w13[e0:e1].to(dev, non_blocking=True)
        sg = s13[e0:e1].to(dev, non_blocking=True)
        # Refit / codes-once-per-chunk: mxfp4_refit_codes_scales and
        # mxfp4_to_codes are ROW/BLOCK-LOCAL (independent per [N, K] row), so
        # flattening the chunk's experts into the N dim is EXACT -> cb[i] ==
        # the per-expert codes, sb_[i] == the per-expert serving scales. The
        # 2-bit base pack below (planes13/sc13) is therefore byte-identical to
        # the old per-expert loop; only the FP8/FP4 derivation moved from
        # per-expert to one batched call per chunk.
        if scale_refit:
            cb, sb_ = mxfp4_refit_codes_scales(
                wg.reshape(-1, wg.shape[-1]),
                sg.reshape(-1, sg.shape[-1]))
            cb = cb.view(e1 - e0, N13, K13)
            sb_ = sb_.view(e1 - e0, N13, K13 // 32)
        else:
            cb, sb_ = mxfp4_to_codes(wg), sg
        for i in range(e1 - e0):
            nib = mxfp4_to_nibbles(wg[i])
            planes13[e0 + i] = pack_fragment_major(cb[i])
            sc13[e0 + i] = pack_scales(sb_[i])
            if fp13 is not None:
                fp13[e0 + i] = _pack_fp4_plane(nib)
        if fp8_13 is not None:
            if store_fp4:
                # fp4-store: refit=0 (enforced by 3g) -> serving scale ==
                # checkpoint scale -> unit-space u = e2m1 exactly. The
                # checkpoint nibbles ARE the unit-space nibble plane; pack
                # them directly (no sb_ needed).
                fp8_13[e0:e1] = pack_fp4_fragment_major_batch(
                    mxfp4_to_nibbles(w13[e0:e1]))
            else:
                # fp8-store: ONE batched derivation per chunk (kills ~E
                # per-expert nib D2H syncs). Reads the CPU checkpoint bytes
                # directly (w13/s13 = ck) and the chunk's serving scales
                # (sb_ = s13 unless refit decremented it); the ONE D2H is
                # sb_.to("cpu"). CPU math -> byte-identical to the old
                # per-expert path (validated by
                # kernels/gen/moe_w8_sm89_batch_check.py).
                fp8_13[e0:e1] = pack_fp8_fragment_major_batch(
                    mxfp4_fp8_unit_bytes_batch(
                        w13[e0:e1], s13[e0:e1], sb_.to("cpu")))
        wg = w2[e0:e1].to(dev, non_blocking=True)
        sg = s2[e0:e1].to(dev, non_blocking=True)
        if scale_refit:
            cb, sb_ = mxfp4_refit_codes_scales(
                wg.reshape(-1, wg.shape[-1]),
                sg.reshape(-1, sg.shape[-1]))
            cb = cb.view(e1 - e0, N2, K2)
            sb_ = sb_.view(e1 - e0, N2, K2 // 32)
        else:
            cb, sb_ = mxfp4_to_codes(wg), sg
        for i in range(e1 - e0):
            nib = mxfp4_to_nibbles(wg[i])
            planes2[e0 + i] = pack_fragment_major(cb[i])
            sc2[e0 + i] = pack_scales(sb_[i])
            if fp2 is not None:
                fp2[e0 + i] = _pack_fp4_plane(nib)
        if fp8_2 is not None:
            if store_fp4:
                # w2 mirror of the fp4-store w13 block above.
                fp8_2[e0:e1] = pack_fp4_fragment_major_batch(
                    mxfp4_to_nibbles(w2[e0:e1]))
            else:
                # w2 mirror of the fp8-store w13 batched block above.
                fp8_2[e0:e1] = pack_fp8_fragment_major_batch(
                    mxfp4_fp8_unit_bytes_batch(
                        w2[e0:e1], s2[e0:e1], sb_.to("cpu")))

    # Persist the built planes (2-bit base + FP4 + FP8/FP4-store delta) for
    # later boots: the cache turns the delta derivation (the slow step) into
    # an NVMe read on every restart after the first. store() skips None parts
    # and never fails the load. Mirrors build_layer_planes_nvfp4.
    if planes_cache_enabled and lidx is not None:
        planes_cache_store(lidx, dict(
            planes13=planes13, sc13=sc13, planes2=planes2, sc2=sc2,
            fp13=fp13, fp2=fp2, **{w8k13: fp8_13, w8k2: fp8_2}))

    if tier is not None:
        _stage_fp4_host(tier, layer_key, fp13, sc13, fp2, sc2)
        del fp13, fp2
        # (the background manager is started by get_tier when the tier is
        # created; the old "start on layer NUM_LAYERS-1" trigger never fired
        # under PP, where layer_keys are local per rank and never reach 42)
    if fp8_tier is not None:
        # FP8 delta: stage the e4m3 byte planes (no scale sections -- the
        # shared resident base scales serve both tiers), then the one-shot
        # end-to-end POOL parity gate (catches a tier sized for the wrong
        # format through a real pool slot, loudly at the first staged layer).
        _stage_fp8_host(fp8_tier, layer_key, fp8_13, sc13, fp8_2, sc2)
        del fp8_13, fp8_2
        _w8_pool_gate(fp8_tier, K13, N13)

    _finish_layer(layer, layer_key, dev, planes13, sc13, planes2, sc2,
                  N13, K13, N2, K2, E,
                  ("w13_weight", "w13_weight_scale", "w2_weight",
                   "w2_weight_scale"))


def build_layer_planes_fp8(layer, layer_key: int,
                           scale_suffix: str = "weight_scale_inv") -> None:
    """FP8 block-quant checkpoint variant of build_layer_planes (Fp8MoEMethod:
    DS4-Flash-FP8, GLM-5.2-FP8 — models without an FP4 release).

    Reads the CPU-staged fp8 params (w13_weight [E,2I,H] e4m3 +
    w13_weight_scale_inv [E,ceil(2I/128),ceil(H/128)] f32 etc.), re-quantizes
    each expert on GPU to the sweep-validated 2-bit pipeline (block-32 UE8M0 +
    e2m1 snap + tensor-sym {-4,-1,1,4}; internal/glm52-sweep/sweep.py), packs
    fragment-major planes, then replaces the originals with empty stubs. The
    e2m1 nibbles of the same requant feed the optional FP4 delta tier.
    """
    from vllm.model_executor.layers.quantization.utils import moe_w2_delta
    from vllm.model_executor.layers.quantization.utils.moe_w2_planes import (
        fp8_block_to_codes_scales, pack_fp4_fragment_major,
        pack_fp8_fragment_major)

    assert _ensure_ready(), (
        "moe_w2 kernels unavailable — the root cause was logged at "
        "engine init as 'moe_w2_cubit unavailable: ...'")
    dev = torch.device("cuda")
    w13 = layer.w13_weight.data                       # [E, 2I, H] e4m3 (cpu)
    s13 = getattr(layer, f"w13_{scale_suffix}").data  # [E, 2I/128, H/128] f32
    w2 = layer.w2_weight.data                         # [E, H, I] e4m3
    s2 = getattr(layer, f"w2_{scale_suffix}").data    # [E, H/128, I/128] f32
    assert w13.dtype == torch.float8_e4m3fn, w13.dtype
    E, N13, K13 = w13.shape
    _, N2, K2 = w2.shape
    _require_kernels(K13, K2, need_w4=moe_w2_delta.enabled())
    _log_plane_budget(E, N13, K13, N2, K2)
    if _try_skip_requant(layer, layer_key, E, N13, K13, N2, K2,
                         ("w13_weight", f"w13_{scale_suffix}", "w2_weight",
                          f"w2_{scale_suffix}")):
        return

    planes13 = torch.empty(E, N13 * K13 // 4, dtype=torch.uint8, device=dev)
    sc13 = torch.empty(E, N13 * K13 // 32, dtype=torch.uint8, device=dev)
    planes2 = torch.empty(E, N2 * K2 // 4, dtype=torch.uint8, device=dev)
    sc2 = torch.empty(E, N2 * K2 // 32, dtype=torch.uint8, device=dev)

    tier = _fp4_tier_for_build(E, dev, N13 * K13, N2 * K2)
    # FP8-e4m3 delta tier (Ada native FP8 MMA; mutually exclusive with FP4).
    # Slot layout: fp8-store = 1 byte/elem e4m3 byte plane; fp4-store = 0.5
    # byte/elem e2m1 nibble plane (kernel decodes nibble->e4m3 in registers).
    # No scale sections in either mode — the UE8M0 scale plane is the shared
    # resident base scale.
    fp8_tier = _fp8_tier_for_build(E, dev, N13 * K13, N2 * K2)
    store_fp4 = moe_w2_delta.fp8_store_fp4()
    fp13 = fp2 = None
    fp8_13 = fp8_2 = None
    if fp8_tier is not None:
        n13_elem = N13 * K13 // (2 if store_fp4 else 1)
        n2_elem = N2 * K2 // (2 if store_fp4 else 1)
        fp8_13 = torch.empty(E, n13_elem, dtype=torch.uint8, device=dev)
        fp8_2 = torch.empty(E, n2_elem, dtype=torch.uint8, device=dev)
    if tier is not None:
        # full nibble planes (w4) or quintal planes (w4q, split)
        fp13 = torch.empty(E, _fp4_plane_nbytes(N13, K13),
                           dtype=torch.uint8, device=dev)
        fp2 = torch.empty(E, _fp4_plane_nbytes(N2, K2),
                          dtype=torch.uint8, device=dev)

    # Mode-dependent cache part keys (see build_layer_planes comment).
    w8k13, w8k2 = ("fp4u13", "fp4u2") if store_fp4 else ("fp8u13", "fp8u2")
    # fp8 experts are 4x the bytes of the mxfp4 path and the requant makes f32
    # temporaries -> smaller H2D chunks, per-expert quantize.
    chunk = 8
    for e0 in range(0, E, chunk):
        e1 = min(e0 + chunk, E)
        wg = w13[e0:e1].to(dev, non_blocking=True)
        sg = s13[e0:e1].to(dev, non_blocking=True)
        for i in range(e1 - e0):
            codes, sbytes, nib, fp8u = fp8_block_to_codes_scales(
                wg[i], sg[i],
                want_nibbles=(fp13 is not None) or (fp8_13 is not None and store_fp4),
                want_fp8=fp8_13 is not None and not store_fp4)
            planes13[e0 + i] = pack_fragment_major(codes)
            sc13[e0 + i] = pack_scales(sbytes)
            if fp13 is not None:
                fp13[e0 + i] = _pack_fp4_plane(nib)
            if fp8_13 is not None:
                if store_fp4:
                    fp8_13[e0 + i] = pack_fp4_fragment_major(nib)
                else:
                    fp8_13[e0 + i] = pack_fp8_fragment_major(fp8u)
        wg = w2[e0:e1].to(dev, non_blocking=True)
        sg = s2[e0:e1].to(dev, non_blocking=True)
        for i in range(e1 - e0):
            codes, sbytes, nib, fp8u = fp8_block_to_codes_scales(
                wg[i], sg[i],
                want_nibbles=(fp2 is not None) or (fp8_2 is not None and store_fp4),
                want_fp8=fp8_2 is not None and not store_fp4)
            planes2[e0 + i] = pack_fragment_major(codes)
            sc2[e0 + i] = pack_scales(sbytes)
            if fp2 is not None:
                fp2[e0 + i] = _pack_fp4_plane(nib)
            if fp8_2 is not None:
                if store_fp4:
                    fp8_2[e0 + i] = pack_fp4_fragment_major(nib)
                else:
                    fp8_2[e0 + i] = pack_fp8_fragment_major(fp8u)

    if tier is not None:
        _stage_fp4_host(tier, layer_key, fp13, sc13, fp2, sc2)
        del fp13, fp2
    if fp8_tier is not None:
        # Stage the delta planes (no scale sections — the shared resident base
        # scales serve both tiers), then the one-shot end-to-end POOL parity
        # gate (catches a tier sized for the wrong format through a real pool
        # slot, loudly at the first staged layer).
        _stage_fp8_host(fp8_tier, layer_key, fp8_13, sc13, fp8_2, sc2)
        del fp8_13, fp8_2
        _w8_pool_gate(fp8_tier, K13, N13)

    _finish_layer(layer, layer_key, dev, planes13, sc13, planes2, sc2,
                  N13, K13, N2, K2, E,
                  ("w13_weight", f"w13_{scale_suffix}", "w2_weight",
                   f"w2_{scale_suffix}"))


def _consume_planes_cache(layer, layer_key: int, dev,
                          E: int, N13: int, K13: int, N2: int,
                          K2: int) -> bool:
    """Serve one layer's planes from the planes cache (GPU-resident
    configs). CPU tensors from the cache feed the same _stage_fp4_host/
    _finish_layer sinks as a fresh requant (their copy_ calls are
    device-agnostic). Shared by the staged path (cache hit replaces the
    requant) and the loader-skip path (stubs; the cache is the ONLY
    source). Returns True on a hit.

    Serves BOTH delta tiers when present: the FP4 tier (cached fp13/fp2)
    and the FP8 tier (cached fp8u13/fp8u2). A cache written without the
    FP8 parts simply MISSes when the FP8 tier is requested (and rebuilds);
    a cache written WITH them skips the per-expert FP8 derivation too."""
    from vllm.model_executor.layers.quantization.utils import (
        moe_w2_planes_cache as planes_cache)
    lidx = planes_cache.layer_idx_from_name(getattr(layer, "layer_name", ""))
    if not planes_cache.enabled() or lidx is None:
        return False
    tier = _fp4_tier_for_build(E, dev, N13 * K13, N2 * K2)
    fp8_tier = _fp8_tier_for_build(E, dev, N13 * K13, N2 * K2)
    from vllm.model_executor.layers.quantization.utils import moe_w2_delta
    store_fp4 = moe_w2_delta.fp8_store_fp4()
    w8k13, w8k2 = ("fp4u13", "fp4u2") if store_fp4 else ("fp8u13", "fp8u2")
    cached = planes_cache.try_load(
        lidx, planes_cache.expected_sizes(
            E, N13, K13, N2, K2,
            want_fp4=(tier is not None),
            want_fp8=(fp8_tier is not None)))
    if cached is None:
        return False
    planes13 = cached["planes13"].view(E, -1).to(dev)
    sc13 = cached["sc13"].view(E, -1).to(dev)
    planes2 = cached["planes2"].view(E, -1).to(dev)
    sc2 = cached["sc2"].view(E, -1).to(dev)
    if tier is not None:
        _stage_fp4_host(tier, layer_key, cached["fp13"].view(E, -1),
                        sc13, cached["fp2"].view(E, -1), sc2)
    if fp8_tier is not None:
        # Delta planes are CPU u8 (the cache stores CPU tensors), matching
        # the build path's CPU-resident fp8_13/fp8_2. Mode-dependent keys:
        # fp4-store cache carries fp4u13/fp4u2; fp8-store carries fp8u13/fp8u2.
        _stage_fp8_host(
            fp8_tier, layer_key, cached[w8k13].view(E, -1), sc13,
            cached[w8k2].view(E, -1), sc2)
        _w8_pool_gate(fp8_tier, K13, N13)
    _finish_layer(layer, layer_key, dev, planes13, sc13, planes2,
                  sc2, N13, K13, N2, K2, E,
                  ("w13_weight", "w13_weight_scale", "w2_weight",
                   "w2_weight_scale"))
    logger.info("moe_w2: layer %d planes from cache", lidx)
    return True


def build_layer_planes_nvfp4(layer, layer_key: int) -> None:
    """NVFP4 (modelopt) checkpoint variant of build_layer_planes
    (ModelOptNvFp4FusedMoE: nvidia/GLM-5.2-NVFP4 — e2m1 codes + e4m3
    block-16 scales + per-tensor scale_2).

    Reads the CPU-staged params (w13_weight [E,2I,H/2] u8 packed +
    w13_weight_scale [E,2I,H/16] e4m3 + w13_weight_scale_2 [E,2] f32 etc.),
    dequantizes each expert to f64 on GPU (exact) and re-quantizes to the
    sweep-validated sign-symmetric 2-bit pipeline; the e2m1 nibbles of the
    same requant feed the optional FP4 delta tier. The UE8M0 block-32 output
    scales absorb scale_2, so serving needs no extra per-tensor factor.
    """
    from vllm.model_executor.layers.quantization.utils import moe_w2_delta
    from vllm.model_executor.layers.quantization.utils.moe_w2_planes import (
        nvfp4_to_codes_scales, pack_fp4_fragment_major, pack_fp8_fragment_major)

    assert _ensure_ready(), (
        "moe_w2 kernels unavailable — the root cause was logged at "
        "engine init as 'moe_w2_cubit unavailable: ...'")
    dev = torch.device("cuda")
    if getattr(layer, "_moe_w2_pack_skip", False):
        # loader-level skip: the params are 0-byte stubs (plan_pack_skip),
        # shapes travel via the create-time stash. The probed store MUST
        # still serve the layer — there is no checkpoint copy to fall
        # back to.
        E, N13, K13, N2, K2 = layer._moe_w2_shapes
        _require_kernels(K13, K2, need_w4=moe_w2_delta.enabled())
        _log_plane_budget(E, N13, K13, N2, K2)
        if moe_w2_delta.base_enabled():
            assert _try_skip_requant(
                layer, layer_key, E, N13, K13, N2, K2,
                ("w13_weight", "w13_weight_scale", "w2_weight",
                 "w2_weight_scale")), (
                f"moe_w2: layer {layer_key} was loader-skipped on a pack "
                f"sidecar hit but the pack no longer serves it "
                f"(dir/sidecar changed mid-load?) — restart without the "
                f"stale VLLM_MOE_W2_STORE_DIR state")
            return
        # GPU-resident: materialize the planes from the planes cache
        # (probed at create time; a miss here means the cache dir changed
        # under a live load).
        assert _consume_planes_cache(layer, layer_key, dev,
                                     E, N13, K13, N2, K2), (
            f"moe_w2: layer {layer_key} was loader-skipped on a planes-"
            f"cache hit but the cache no longer serves it — restart "
            f"without the stale VLLM_MOE_W2_PLANES_CACHE state")
        return
    w13 = layer.w13_weight.data                 # [E, 2I, H/2] u8 (cpu)
    s13 = layer.w13_weight_scale.data           # [E, 2I, H/16] e4m3
    s13_2 = layer.w13_weight_scale_2.data       # [E, 2] f32 (w1, w3)
    w2 = layer.w2_weight.data                   # [E, H, I/2] u8
    s2 = layer.w2_weight_scale.data             # [E, H, I/16] e4m3
    s2_2 = layer.w2_weight_scale_2.data         # [E] f32
    assert w13.dtype == torch.uint8 and s13.dtype == torch.float8_e4m3fn, (
        w13.dtype, s13.dtype)
    E, N13, K13h = w13.shape
    K13 = K13h * 2
    _, N2, K2h = w2.shape
    K2 = K2h * 2
    group = K13 // s13.shape[2]                 # 16 for NVFP4
    _require_kernels(K13, K2, need_w4=moe_w2_delta.enabled())
    _log_plane_budget(E, N13, K13, N2, K2)
    if _try_skip_requant(layer, layer_key, E, N13, K13, N2, K2,
                         ("w13_weight", "w13_weight_scale", "w2_weight",
                          "w2_weight_scale")):
        return

    # Planes cache (VLLM_MOE_W2_PLANES_CACHE): the requant below is
    # deterministic given (checkpoint, TP layout, zero mode), so cached
    # planes can be streamed back instead of rebuilt (~9 min saved on
    # Kimi-K2.7 restarts). Complements the pack store's boot-from-pack
    # above: the cache serves GPU-RESIDENT plane configs (planes must be
    # materialized), the pack store serves host-resident tiers (planes
    # never materialize).
    if _consume_planes_cache(layer, layer_key, dev, E, N13, K13, N2, K2):
        return
    from vllm.model_executor.layers.quantization.utils import (
        moe_w2_planes_cache as planes_cache)
    lidx = planes_cache.layer_idx_from_name(getattr(layer, "layer_name", ""))
    tier = _fp4_tier_for_build(E, dev, N13 * K13, N2 * K2)

    planes13 = torch.empty(E, N13 * K13 // 4, dtype=torch.uint8, device=dev)
    sc13 = torch.empty(E, N13 * K13 // 32, dtype=torch.uint8, device=dev)
    planes2 = torch.empty(E, N2 * K2 // 4, dtype=torch.uint8, device=dev)
    sc2 = torch.empty(E, N2 * K2 // 32, dtype=torch.uint8, device=dev)

    fp8_tier = _fp8_tier_for_build(E, dev, N13 * K13, N2 * K2)
    store_fp4 = moe_w2_delta.fp8_store_fp4()
    fp13 = fp2 = None
    fp8_13 = fp8_2 = None
    if fp8_tier is not None:
        n13_elem = N13 * K13 // (2 if store_fp4 else 1)
        n2_elem = N2 * K2 // (2 if store_fp4 else 1)
        fp8_13 = torch.empty(E, n13_elem, dtype=torch.uint8, device=dev)
        fp8_2 = torch.empty(E, n2_elem, dtype=torch.uint8, device=dev)
    if tier is not None:
        # full nibble planes (w4) or quintal planes (w4q, split)
        fp13 = torch.empty(E, _fp4_plane_nbytes(N13, K13),
                           dtype=torch.uint8, device=dev)
        fp2 = torch.empty(E, _fp4_plane_nbytes(N2, K2),
                          dtype=torch.uint8, device=dev)

    # Mode-dependent cache part keys (see build_layer_planes comment).
    w8k13, w8k2 = ("fp4u13", "fp4u2") if store_fp4 else ("fp8u13", "fp8u2")
    # f64 temporaries are 16x the packed nibbles -> small H2D chunks,
    # per-expert quantize (mirrors the fp8 loader).
    chunk = 8
    for e0 in range(0, E, chunk):
        e1 = min(e0 + chunk, E)
        wg = w13[e0:e1].to(dev, non_blocking=True)
        sg = s13[e0:e1].to(dev, non_blocking=True)
        s2g = s13_2[e0:e1].to(dev, non_blocking=True)
        half = N13 // 2                          # rows [0:I]=w1, [I:2I]=w3
        for i in range(e1 - e0):
            s2_row = torch.cat((s2g[i, 0].expand(half), s2g[i, 1].expand(half)))
            codes, sbytes, nib, fp8u = nvfp4_to_codes_scales(
                wg[i], sg[i], s2_row, group=group,
                want_nibbles=(fp13 is not None) or (fp8_13 is not None and store_fp4),
                want_fp8=fp8_13 is not None and not store_fp4)
            planes13[e0 + i] = pack_fragment_major(codes)
            sc13[e0 + i] = pack_scales(sbytes)
            if fp13 is not None:
                fp13[e0 + i] = _pack_fp4_plane(nib)
            if fp8_13 is not None:
                if store_fp4:
                    fp8_13[e0 + i] = pack_fp4_fragment_major(nib)
                else:
                    fp8_13[e0 + i] = pack_fp8_fragment_major(fp8u)
        wg = w2[e0:e1].to(dev, non_blocking=True)
        sg = s2[e0:e1].to(dev, non_blocking=True)
        s2g = s2_2[e0:e1].to(dev, non_blocking=True)
        for i in range(e1 - e0):
            codes, sbytes, nib, fp8u = nvfp4_to_codes_scales(
                wg[i], sg[i], s2g[i], group=group,
                want_nibbles=(fp2 is not None) or (fp8_2 is not None and store_fp4),
                want_fp8=fp8_2 is not None and not store_fp4)
            planes2[e0 + i] = pack_fragment_major(codes)
            sc2[e0 + i] = pack_scales(sbytes)
            if fp2 is not None:
                fp2[e0 + i] = _pack_fp4_plane(nib)
            if fp8_2 is not None:
                if store_fp4:
                    fp8_2[e0 + i] = pack_fp4_fragment_major(nib)
                else:
                    fp8_2[e0 + i] = pack_fp8_fragment_major(fp8u)

    if planes_cache.enabled() and lidx is not None:
        # Persist the delta planes alongside the 2-bit/FP4 parts so an
        # FP8-enabled boot can loader-skip (_consume_planes_cache can serve
        # both tiers). Mode-dependent keys: fp4-store and fp8-store caches
        # coexist (distinct part names mean a cache written in one mode
        # MISSes in the other and rebuilds — correct behaviour).
        planes_cache.store(lidx, dict(planes13=planes13, sc13=sc13,
                                      planes2=planes2, sc2=sc2,
                                      fp13=fp13, fp2=fp2,
                                      **{w8k13: fp8_13, w8k2: fp8_2}))

    if tier is not None:
        _stage_fp4_host(tier, layer_key, fp13, sc13, fp2, sc2)
        del fp13, fp2
    if fp8_tier is not None:
        _stage_fp8_host(fp8_tier, layer_key, fp8_13, sc13, fp8_2, sc2)
        del fp8_13, fp8_2
        # One-shot end-to-end POOL parity gate (boot hook).
        _w8_pool_gate(fp8_tier, K13, N13)

    _finish_layer(layer, layer_key, dev, planes13, sc13, planes2, sc2,
                  N13, K13, N2, K2, E,
                  ("w13_weight", "w13_weight_scale", "w2_weight",
                   "w2_weight_scale"))


def _finish_layer(layer, layer_key, dev, planes13, sc13, planes2, sc2,
                  N13, K13, N2, K2, E, param_names) -> None:
    from vllm.model_executor.layers.quantization.utils import moe_w2_delta
    # transformer layer index of this layer_key (dense-offset models: GLM's
    # first sparse layer 3 -> key 0). LOOKA uses it to pair each key with
    # its transformer layer's router (mlp.gate) weights.
    from vllm.model_executor.layers.quantization.utils import (
        moe_w2_planes_cache as _pc)
    _tl = _pc.layer_idx_from_name(getattr(layer, "layer_name", ""))
    if moe_w2_delta.exact_cache_enabled():
        tier = moe_w2_delta._TIER
        if tier is None or layer_key not in tier._store:
            raise RuntimeError(
                f"moe_w2 exact cache: layer {layer_key} was not staged")
        f13len, s13len = N13 * K13 // 2, N13 * K13 // 32
        f2len, s2len = N2 * K2 // 2, N2 * K2 // 32
        _LAYERS[layer_key] = dict(
            N13=N13, K13=K13, N2=N2, K2=K2, E=E, exact=True,
            tl_idx=_tl, off_s13=f13len, off_c2=f13len + s13len,
            off_s2=f13len + s13len + f2len)
        del planes13, sc13, planes2, sc2
        stub = torch.empty(0, dtype=torch.uint8, device=dev)
        for name in param_names:
            layer.register_parameter(
                name, torch.nn.Parameter(stub, requires_grad=False))
        logger.info(
            "moe_w2: layer %d exact FP4-storage HOST-staged (%.2f GiB)",
            layer_key, E * tier.slot_bytes / 2**30)
        return
    if moe_w2_delta.base_enabled():
        # BASE cache (inverted delta): the 2-bit planes go to PINNED HOST RAM
        # instead of staying GPU-resident; the GPU holds only the base tier's
        # slot pool. Slot layout per expert: [codes13 | sc13 | codes2 | sc2]
        # (the tier's "w13 section" = codes13+sc13, "w2 section" = codes2+sc2,
        # so add_layer_host_planes packs it verbatim).
        c13len, s13len = planes13.shape[1], sc13.shape[1]
        c2len, s2len = planes2.shape[1], sc2.shape[1]
        btier = moe_w2_delta.get_base_tier(
            _layer_cutoff() + 1, E, dev,
            w13_bytes=c13len + s13len, w2_bytes=c2len + s2len)
        btier.add_layer_host_planes(
            layer_key,
            torch.cat((planes13, sc13), dim=1),
            torch.cat((planes2, sc2), dim=1))
        _LAYERS[layer_key] = dict(
            N13=N13, K13=K13, N2=N2, K2=K2, E=E, base=True, tl_idx=_tl,
            off_s13=c13len, off_c2=c13len + s13len,
            off_s2=c13len + s13len + c2len,
            # FP4 need-pool slot sections ([fp4_13|sc13|fp4_2|sc2]; fp4 codes
            # are 2x the 2-bit codes, scale sections identical) — read by the
            # base+delta desc kernel when the FP4 tier coexists.
            off4_s13=2 * c13len, off4_c2=2 * c13len + s13len,
            off4_s2=2 * c13len + s13len + 2 * c2len,
        )
        del planes13, sc13, planes2, sc2
        stub = torch.empty(0, dtype=torch.uint8, device=dev)
        for name in param_names:
            layer.register_parameter(
                name, torch.nn.Parameter(stub, requires_grad=False))
        logger.info("moe_w2: layer %d planes HOST-staged (base cache, "
                    "%.2f GiB pinned)", layer_key,
                    E * btier.slot_bytes / 2**30)
        return

    _check_resident_fit(layer_key, dev,
                        planes13.nbytes + sc13.nbytes
                        + planes2.nbytes + sc2.nbytes)
    _LAYERS[layer_key] = dict(
        planes13=planes13, sc13=sc13, planes2=planes2, sc2=sc2,
        N13=N13, K13=K13, N2=N2, K2=K2, E=E, tl_idx=_tl,
    )
    # Release checkpoint copies; keep CUDA stubs so device probes stay happy.
    stub = torch.empty(0, dtype=torch.uint8, device=dev)
    for name in param_names:
        layer.register_parameter(
            name, torch.nn.Parameter(stub, requires_grad=False))
    logger.info("moe_w2: layer %d planes built (%.2f GiB)", layer_key,
                (planes13.nbytes + sc13.nbytes + planes2.nbytes + sc2.nbytes)
                / 2**30)


_resident_fit_checked = False


def _check_resident_fit(layer_key: int, dev, bytes_per_layer: int) -> None:
    """RESIDENCY HARDSTOP (user directive 2026-07-13): if resident planes
    cannot fit in VRAM without degrading serving (no room for KV/graphs),
    refuse EARLY with actionable guidance instead of a raw CUDA OOM deep
    into the requant. Mirrors the base-cache working-set hardstop one rung
    down the residency ladder.

    Runs once, at the FIRST resident layer (its exact per-layer bytes x
    the model's remaining MoE layer count = total plane demand; uniform
    layers assumed - true for DS4/GLM/Kimi). Reserves are estimates and
    deliberately conservative in the SAFE direction only (a false PASS
    just falls through to the torch OOM backstop below; a false FAIL is
    avoided by keeping reserves minimal: 2 GiB graphs/workspaces + ~1.7
    GiB MTP drafter scratch when speculative decoding is on + a KV floor
    of one max_model_len sequence at the measured fp8 MLA rate)."""
    global _resident_fit_checked
    if _resident_fit_checked:
        return
    _resident_fit_checked = True
    try:
        from vllm.config import get_current_vllm_config
        vcfg = get_current_vllm_config()
        hf = vcfg.model_config.hf_text_config
        n_layers = int(getattr(hf, "num_hidden_layers", 0) or 0)
        n_dense = int(getattr(hf, "first_k_dense_replace", 0) or 0)
        n_moe = max(n_layers - n_dense, 1)
        max_len = int(vcfg.model_config.max_model_len)
        spec_on = vcfg.speculative_config is not None
        free_b, _total_b = torch.cuda.mem_get_info(dev)
        planes_total = bytes_per_layer * (n_moe - len(_LAYERS))
        kv_floor = max_len * 16 * 1024            # ~15.6 KB/tok measured
        reserve = 2 * 2**30 + (int(1.7 * 2**30) if spec_on else 0) + kv_floor
        budget = free_b - reserve
        if planes_total > budget and os.getenv(
                "VLLM_MOE_W2_FORCE_RESIDENT", "0") == "1":
            # Explicit consent valve (the FORCE_POOL pattern): the reserve
            # model is deliberately conservative and refuses knife-edge 1x
            # configs that DO serve (the validated pro6000x1 recipe sits
            # ~3-4 GiB inside this guard's reserves). Forced boots fall
            # through to the torch OOM backstop if the guard was right.
            logger.warning(
                "moe_w2 RESIDENT planes exceed the estimated budget by "
                "%.1f GiB but VLLM_MOE_W2_FORCE_RESIDENT=1 - continuing "
                "on user consent (a real shortfall will OOM at load or "
                "capture).", (planes_total - budget) / 2**30)
            return
        if planes_total > budget:
            deficit = (planes_total - budget) / 2**30
            # suggested pool: what actually fits, rounded down to .5 GiB
            fit_gb = max((budget + bytes_per_layer * len(_LAYERS))
                         / 2**30, 0.0)
            sug = max(int(fit_gb * 2) / 2, 1.0)
            raise ValueError(
                f"moe_w2 RESIDENT planes do not fit: {n_moe} MoE layers x "
                f"{bytes_per_layer / 2**30:.2f} GiB = "
                f"{n_moe * bytes_per_layer / 2**30:.1f} GiB of 2-bit "
                f"planes vs ~{budget / 2**30:.1f} GiB of VRAM budget "
                f"(free {free_b / 2**30:.1f} minus KV floor for "
                f"{max_len} tokens, graphs/workspaces"
                f"{' and MTP scratch' if spec_on else ''}) - short by "
                f"~{deficit:.1f} GiB. Serving would OOM or degrade. Use "
                f"the BASE-CACHE rung of the residency ladder instead: "
                f"VLLM_MOE_W2_BASE_CACHE_GB={sug:.1f} (host-resident "
                f"planes, GPU expert cache; add VLLM_MOE_W2_STORE_DIR + "
                f"VLLM_MOE_W2_BASE_RAM_GB for the NVMe tier if host RAM "
                f"is short). The boot guard will then verify the pool "
                f"against its working-set floor.")
    except ValueError:
        raise
    except Exception as e:  # noqa: BLE001 - the check must not break boot
        logger.warning("moe_w2 resident-fit check skipped: %s", e)


# --------------------------------------------------------------------------
# Forward
# --------------------------------------------------------------------------

def _workspaces(slots: int, tokens: int, dev, inter: int = 2048,
                hidden: int = 4096, n_experts: int = 256) -> dict:
    # `inter` = per-rank expert intermediate size I (2048 on 1 GPU; 1024 @ TP2,
    # 512 @ TP4 as the experts shard). The hidden H (4096 DS4, 6144 GLM-5.x) is
    # NOT sharded, so the A-side (a1), x-quant (xq) and w2 output (c2) buffers
    # stay H-wide; only the gate/up output (c13 = 2I), the intermediate
    # activation (act/a2 = I) and its per-32 scales (as2 = I/32) follow the
    # shard.
    #
    # Activation scales are per-32-GROUP f32 (a32 cubin rev): [rows, K/32]
    # f32 — 4x the scale elements of the retired per-128 format (row stride
    # (K/32)*4 bytes; the desc-build strides moved with it).
    if (_WS.get("slots", 0) < slots or _WS.get("tokens", 0) < tokens
            or _WS.get("inter") != inter or _WS.get("hidden") != hidden
            or _WS.get("n_experts", 0) < n_experts):
        # Captured graphs bake these buffer ADDRESSES; a realloc after any
        # capture would leave every captured replay reading freed memory.
        # The system invariant is "profile_run maxes the sizes before
        # capture" — enforce it instead of assuming it (review 2.5): any
        # growth after the first capture (or during one) is a hard error.
        if _WS.get("frozen"):
            raise RuntimeError(
                f"moe_w2 workspace growth after graph capture: have "
                f"slots={_WS.get('slots')}, tokens={_WS.get('tokens')}, "
                f"need slots={slots}, tokens={tokens}. Captured graphs "
                "hold the old buffers - the profile run must cover the "
                "largest schedulable batch (check max_num_batched_tokens/"
                "max_num_seqs vs the profile shapes).")
        if torch.cuda.is_available() and torch.cuda.is_initialized() \
                and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "moe_w2 workspace realloc DURING cudagraph capture "
                f"(slots {_WS.get('slots', 0)}->{slots}, tokens "
                f"{_WS.get('tokens', 0)}->{tokens}) - the profile run "
                "must pre-size the workspaces.")
        slots = max(slots, _WS.get("slots", 0))
        tokens = max(tokens, _WS.get("tokens", 0))
        n_experts = max(n_experts, _WS.get("n_experts", 0))
        _WS.update(
            slots=slots,
            tokens=tokens,
            inter=inter,
            hidden=hidden,
            n_experts=n_experts,
            # token-side quant buffers; the LAST row is the permanent zero
            # pad row (gather source for filler slots) — quant only ever
            # writes rows [:T].
            xq=torch.zeros(tokens + 1, hidden, dtype=torch.float8_e4m3fn,
                           device=dev),
            xs=torch.zeros(tokens + 1, hidden // 32, dtype=torch.float32,
                           device=dev),
            a1=torch.zeros(slots + 4, hidden, dtype=torch.float8_e4m3fn,
                           device=dev),
            as1=torch.zeros(slots + 4, hidden // 32, dtype=torch.float32,
                            device=dev),
            # zeros, not empty: pad-pair rows are never written by the kernel
            # (early EXIT) yet flow through silu/scatter math with weight 0;
            # uninitialized inf/nan would poison 0*x.
            c13=torch.zeros(slots + 4, 2 * inter, dtype=torch.bfloat16,
                            device=dev),
            act=torch.zeros(slots + 4, inter, dtype=torch.bfloat16, device=dev),
            a2=torch.zeros(slots + 4, inter, dtype=torch.float8_e4m3fn,
                           device=dev),
            as2=torch.zeros(slots + 4, inter // 32,
                            dtype=torch.float32, device=dev),
            c2=torch.zeros(slots + 4, hidden, dtype=torch.bfloat16,
                           device=dev),
            # Inverse of moe_align's slot -> flattened-route permutation.
            # One int32 per padded slot is enough because sorted_ids contains
            # at least T*top_k entries.  The fused deterministic unpermute
            # writes only valid route coordinates and checks route_ids before
            # reading the inverse, so sentinel routes need no initialization.
            route_slot=torch.empty(slots, dtype=torch.int32, device=dev),
            desc=torch.empty(4, slots // _BLOCK, 6, dtype=torch.int64,
                             device=dev),
            # split-FP4 (moe_w4q_mm) desc tables: 8 u64 per pair, 64 B ABI
            desc4s=torch.empty(2, slots // _BLOCK, 8, dtype=torch.int64,
                               device=dev),
            # -1 slot row for the tier-less desc path; sized to the MODEL's
            # expert count (256 = DS4 default; 384 Kimi-K2.x reads past a
            # fixed 256-row table).
            no_slots=torch.full((max(n_experts, 256),), -1,
                                dtype=torch.int32, device=dev),
        )
        if _afrag_ok:
            # AFRAG destination buffers: the triton repack streams row-major
            # a1/a2 into these (single pass, no copy-back); the desc tables
            # point the GEMM at them instead of a1/a2.
            _WS.update(
                a1f=torch.zeros(slots + 4, hidden, dtype=torch.float8_e4m3fn,
                                device=dev),
                a2f=torch.zeros(slots + 4, inter, dtype=torch.float8_e4m3fn,
                                device=dev),
            )
    return _WS


import triton
import triton.language as tl


@triton.jit
def _route_slot_invert_kernel(
    sorted_ids_ptr,
    route_slot_ptr,
    n_slots,
    n_routes,
    BLOCK: tl.constexpr,
):
    """Invert moe_align's collision-free valid slot -> route permutation."""
    slot = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_bounds = slot < n_slots
    route = tl.load(sorted_ids_ptr + slot, mask=in_bounds, other=n_routes)
    valid = in_bounds & (route >= 0) & (route < n_routes)
    # Every valid flattened route occurs at most once in sorted_ids.  Stores
    # therefore have no collisions and need no atomics (deterministic).
    tl.store(route_slot_ptr + route, slot, mask=valid)


@triton.jit
def _deterministic_unpermute_kernel(
    c2_ptr,
    route_slot_ptr,
    route_ids_ptr,
    weights_ptr,
    miss_rows_ptr,
    out_ptr,
    hidden,
    n_slots,
    n_experts,
    TOP_K: tl.constexpr,
    HAS_MISS_ROWS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Fixed-order weighted route reduction without FP32 route tensors."""
    token = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    col_mask = cols < hidden
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for j in tl.static_range(0, TOP_K):
        route = token * TOP_K + j
        expert = tl.load(route_ids_ptr + route)
        valid_route = (expert >= 0) & (expert < n_experts)
        slot = tl.load(route_slot_ptr + route,
                       mask=valid_route, other=0).to(tl.int32)
        valid_slot = valid_route & (slot >= 0) & (slot < n_slots)
        value = tl.load(c2_ptr + slot * hidden + cols,
                        mask=valid_slot & col_mask,
                        other=0.0).to(tl.float32)
        if HAS_MISS_ROWS:
            resident = tl.load(miss_rows_ptr + slot,
                               mask=valid_slot, other=0).to(tl.int1)
            # Select before multiplication so stale NaN/Inf workspace rows
            # belonging to cache misses cannot poison a zero contribution.
            value = tl.where(resident, value, 0.0)
        weight = tl.load(weights_ptr + route,
                         mask=valid_route, other=0.0).to(tl.float32)
        acc += value * weight
    tl.store(out_ptr + token * hidden + cols, acc, mask=col_mask)


def _deterministic_unpermute(
    c2: torch.Tensor,
    sorted_ids: torch.Tensor,
    route_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    miss_rows: torch.Tensor | None,
    route_slot: torch.Tensor,
    out: torch.Tensor,
    n_experts: int,
) -> torch.Tensor:
    """Deterministically undo moe_align using bounded persistent scratch.

    The prior torch implementation allocated the FP32 slot conversion/product
    plus an FP32 [T*top_k, H] gather.  At the 1,044-token failing shape,
    top_k=6, H=4,096, and E=256 padding, that peaks near 414 MiB per rank.
    This path uses a 4-byte inverse per route and accumulates each
    [token, hidden-tile] result in registers.
    """
    tokens, top_k = topk_weights.shape
    n_slots = sorted_ids.numel()
    hidden = c2.shape[1]
    _route_slot_invert_kernel[(triton.cdiv(n_slots, 256),)](
        sorted_ids, route_slot, n_slots, tokens * top_k, BLOCK=256)
    miss_ptr = miss_rows if miss_rows is not None else c2
    _deterministic_unpermute_kernel[
        (tokens, triton.cdiv(hidden, 256))
    ](
        c2, route_slot, route_ids, topk_weights, miss_ptr, out,
        hidden, n_slots, n_experts,
        TOP_K=top_k, HAS_MISS_ROWS=miss_rows is not None, BLOCK_H=256,
        num_warps=4,
    )
    return out


@triton.jit
def _afrag_repack_kernel(src_ptr, dst_ptr, K: tl.constexpr):
    """Row-major fp8 [pairs*16, K] -> AFRAG fragment-major, single pass.

    One program = one (pair, j=k64) 16-row x 64-byte block = 256 u32 words;
    the permutation [pair, g2, g, j, quad, t, b] -> [pair, j, g, t, quad, g2, b]
    lands each program's words in one contiguous 1 KiB dst run. Bit-identical
    to _to_fragment_major (validated), ~3x faster than the torch permute+copy
    and needs no intermediate tensor."""
    p = tl.program_id(0)
    j = tl.program_id(1)
    w = tl.arange(0, 256)
    g2 = w & 1
    quad = (w >> 1) & 3
    t = (w >> 3) & 3
    g = (w >> 5) & 7
    src_off = (p * 16 + g2 * 8 + g) * (K // 4) + j * 16 + quad * 4 + t
    dst_off = p * 16 * (K // 4) + j * 256 + w
    tl.store(dst_ptr + dst_off, tl.load(src_ptr + src_off))


def _afrag_repack(src: torch.Tensor, dst: torch.Tensor, pairs: int, K: int):
    """Repack rows [:pairs*16] of `src` (fp8 row-major) into `dst` (AFRAG)."""
    src32 = src.view(torch.uint8).view(-1).view(torch.int32)
    dst32 = dst.view(torch.uint8).view(-1).view(torch.int32)
    _afrag_repack_kernel[(pairs, K // 64)](src32, dst32, K=K)


def a32_dequant_ref(x: torch.Tensor, gemm: int = 1) -> torch.Tensor:
    """Torch-only reference roundtrip of the activation quant the forward
    serves for the given GEMM (group _G1/_G2, optional UE8M0 rounding —
    mirrors per_token_group_quant_fp8): rows [M, K] -> f32 dequant. Used
    by the tools/ test references so they cannot drift from the serving
    format regardless of the VLLM_MOE_W2_A32_* envs in effect."""
    group = _G1 if gemm == 1 else _G2
    m, k = x.shape
    xb = x.float().view(m, k // group, group)
    scale = (xb.abs().amax(-1).clamp_min(1e-10) / 448.0)
    if _A32_UE8M0:
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    q = (xb / scale[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.float().view(m, k) * scale.repeat_interleave(group, 1)


@triton.jit
def _desc_build_kernel(
    eids_ptr, npost_ptr, slot_ptr, d_ptr,
    a1b, as1b, c13b, a2b, as2b, c2b,
    p13b, s13b, p2b, s2b, poolb,
    p13s, s13s, p2s, s2s,
    slot_bytes, w13_bytes,
    a1_rb, as1_rb, c13_rb, a2_rb, as2_rb, c2_rb,
    n_experts, pairs, cap6, mblock,
    BLOCK: tl.constexpr,
):
    """All four moe desc tables in one launch (24 columns per pair).

    d_ptr = [4, cap, 6] i64: 0 = w2-tier w13, 1 = w2-tier w2,
    2 = w4-tier w13, 3 = w4-tier w2. A pair is routed to exactly one tier
    via the m_rows field (the other tier's kernel sees m=0 -> early EXIT).
    slot_ptr = this layer's row of the delta slot table (-1 = base tier);
    poolb = delta pool base (w13 plane at slot start, w2 at +w13_bytes).
    """
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = p < pairs
    e = tl.load(eids_ptr + p, mask=mask, other=0).to(tl.int64)
    e = tl.minimum(tl.maximum(e, 0), n_experts - 1)
    slot = tl.load(slot_ptr + e, mask=mask, other=-1).to(tl.int64)
    npost = tl.load(npost_ptr).to(tl.int64)
    live = p < npost // mblock
    is4 = slot >= 0
    m2 = tl.where(live & ~is4, mblock, 0).to(tl.int64)
    m4 = tl.where(live & is4, mblock, 0).to(tl.int64)
    base = p.to(tl.int64) * mblock
    slot_c = tl.maximum(slot, 0)
    a1 = a1b + base * a1_rb
    as1 = as1b + base * as1_rb
    c13 = c13b + base * c13_rb
    a2 = a2b + base * a2_rb
    as2 = as2b + base * as2_rb
    c2 = c2b + base * c2_rb
    bs13 = s13b + e * s13s
    bs2 = s2b + e * s2s
    for gi in tl.static_range(4):
        d = d_ptr + gi * cap6 + p * 6
        if gi == 0:
            b, s, a, as_, c, m = p13b + e * p13s, bs13, a1, as1, c13, m2
        elif gi == 1:
            b, s, a, as_, c, m = p2b + e * p2s, bs2, a2, as2, c2, m2
        elif gi == 2:
            b, s, a, as_, c, m = (poolb + slot_c * slot_bytes, bs13,
                                  a1, as1, c13, m4)
        else:
            b, s, a, as_, c, m = (poolb + slot_c * slot_bytes + w13_bytes,
                                  bs2, a2, as2, c2, m4)
        tl.store(d + 0, a, mask=mask)
        tl.store(d + 1, as_, mask=mask)
        tl.store(d + 2, b, mask=mask)
        tl.store(d + 3, s, mask=mask)
        tl.store(d + 4, c, mask=mask)
        tl.store(d + 5, m, mask=mask)


@triton.jit
def _desc_build_kernel_w4s(
    eids_ptr, npost_ptr, slot_ptr, d_ptr,
    a1b, as1b, c13b, a2b, as2b, c2b,
    p13b, s13b, p2b, s2b, poolb,
    p13s, s13s, p2s, s2s,
    slot_bytes, w13r_bytes,
    a1_rb, as1_rb, c13_rb, a2_rb, as2_rb, c2_rb,
    n_experts, pairs, cap8, mblock,
    BLOCK: tl.constexpr,
):
    """Split-FP4 desc tables (moe_w4q_mm, 8 x u64 per pair, 64 B ABI):
    {a, as, base, ref, bs, c, m_rows, pad}. `base`/`bs` point at the
    RESIDENT 2-bit plane / scale rows (exactly the w2 tier's pointers);
    `ref` at the delta slot's quintal sections ([q13 | q2], w13r_bytes =
    q13 section size). Written alongside the main kernel's w2 tables;
    pairs not FP4-resident get m=0 (w4q early-EXITs)."""
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = p < pairs
    e = tl.load(eids_ptr + p, mask=mask, other=0).to(tl.int64)
    e = tl.minimum(tl.maximum(e, 0), n_experts - 1)
    slot = tl.load(slot_ptr + e, mask=mask, other=-1).to(tl.int64)
    npost = tl.load(npost_ptr).to(tl.int64)
    live = p < npost // mblock
    is4 = slot >= 0
    m4 = tl.where(live & is4, mblock, 0).to(tl.int64)
    base = p.to(tl.int64) * mblock
    ref = poolb + tl.maximum(slot, 0) * slot_bytes
    a1 = a1b + base * a1_rb
    as1 = as1b + base * as1_rb
    c13 = c13b + base * c13_rb
    a2 = a2b + base * a2_rb
    as2 = as2b + base * as2_rb
    c2 = c2b + base * c2_rb
    for gi in tl.static_range(2):
        d = d_ptr + gi * cap8 + p * 8
        if gi == 0:
            bb, rr, ss, a, as_, c = (p13b + e * p13s, ref, s13b + e * s13s,
                                     a1, as1, c13)
        else:
            bb, rr, ss, a, as_, c = (p2b + e * p2s, ref + w13r_bytes,
                                     s2b + e * s2s, a2, as2, c2)
        tl.store(d + 0, a, mask=mask)
        tl.store(d + 1, as_, mask=mask)
        tl.store(d + 2, bb, mask=mask)
        tl.store(d + 3, rr, mask=mask)
        tl.store(d + 4, ss, mask=mask)
        tl.store(d + 5, c, mask=mask)
        tl.store(d + 6, m4, mask=mask)
        tl.store(d + 7, tl.zeros_like(m4), mask=mask)


@triton.jit
def _desc_build_kernel_prefill4(
    eids_ptr, npost_ptr, slot_ptr, d_ptr,
    a1b, as1b, c13b, a2b, as2b, c2b,
    a1b_rm, a2b_rm,
    p13b, s13b, p2b, s2b, poolb,
    p13s, s13s, p2s, s2s,
    slot_bytes, w13_bytes,
    a1_rb, as1_rb, c13_rb, a2_rb, as2_rb, c2_rb,
    n_experts, pairs, cap6, mblock,
    BLOCK: tl.constexpr,
):
    """Prefill variant of _desc_build_kernel (VLLM_MOE_W2_PREFILL_FP4):
    FP4-resident pairs divert to the w4 tier like decode, but the w4/w4q
    kernels take M<=4, so each diverted mblock(16)-token pair is emitted as
    FOUR w4 sub-entries (rows +0/+4/+8/+12, m=4 each) at desc index
    p*4+sub — capacity fits because prefill pairs = slots/16 and the desc
    tables are sized slots/4. The w2 tables keep pair granularity (m=0 for
    diverted pairs -> MC4/AFRAG early-EXIT). w2 A-pointers follow the
    (possibly fragment-major) a1b/a2b bases; the w4 sub-entries always read
    the ROW-MAJOR a1b_rm/a2b_rm (repack leaves them intact)."""
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = p < pairs
    e = tl.load(eids_ptr + p, mask=mask, other=0).to(tl.int64)
    e = tl.minimum(tl.maximum(e, 0), n_experts - 1)
    slot = tl.load(slot_ptr + e, mask=mask, other=-1).to(tl.int64)
    npost = tl.load(npost_ptr).to(tl.int64)
    live = p < npost // mblock
    is4 = slot >= 0
    m2 = tl.where(live & ~is4, mblock, 0).to(tl.int64)
    base = p.to(tl.int64) * mblock
    slot_c = tl.maximum(slot, 0)
    bs13 = s13b + e * s13s
    bs2 = s2b + e * s2s
    # w2 tables (pair granularity, diverted pairs zeroed)
    for gi in tl.static_range(2):
        d = d_ptr + gi * cap6 + p * 6
        if gi == 0:
            b, s, a, as_, c = (p13b + e * p13s, bs13, a1b + base * a1_rb,
                               as1b + base * as1_rb, c13b + base * c13_rb)
        else:
            b, s, a, as_, c = (p2b + e * p2s, bs2, a2b + base * a2_rb,
                               as2b + base * as2_rb, c2b + base * c2_rb)
        tl.store(d + 0, a, mask=mask)
        tl.store(d + 1, as_, mask=mask)
        tl.store(d + 2, b, mask=mask)
        tl.store(d + 3, s, mask=mask)
        tl.store(d + 4, c, mask=mask)
        tl.store(d + 5, m2, mask=mask)
    # w4 tables (sub-entry granularity: 4 x m<=4 rows per diverted pair)
    for sub in tl.static_range(4):
        rbase = base + sub * 4
        m4 = tl.where(live & is4, 4, 0).to(tl.int64)
        for gi in tl.static_range(2):
            d = d_ptr + (2 + gi) * cap6 + (p * 4 + sub) * 6
            if gi == 0:
                b, s, a, as_, c = (poolb + slot_c * slot_bytes, bs13,
                                   a1b_rm + rbase * a1_rb,
                                   as1b + rbase * as1_rb,
                                   c13b + rbase * c13_rb)
            else:
                b, s, a, as_, c = (poolb + slot_c * slot_bytes + w13_bytes,
                                   bs2, a2b_rm + rbase * a2_rb,
                                   as2b + rbase * as2_rb,
                                   c2b + rbase * c2_rb)
            tl.store(d + 0, a, mask=mask)
            tl.store(d + 1, as_, mask=mask)
            tl.store(d + 2, b, mask=mask)
            tl.store(d + 3, s, mask=mask)
            tl.store(d + 4, c, mask=mask)
            tl.store(d + 5, m4, mask=mask)


@triton.jit
def _desc_build_kernel_w4s_prefill4(
    eids_ptr, npost_ptr, slot_ptr, d_ptr,
    a1b_rm, as1b, c13b, a2b_rm, as2b, c2b,
    p13b, s13b, p2b, s2b, poolb,
    p13s, s13s, p2s, s2s,
    slot_bytes, w13r_bytes,
    a1_rb, as1_rb, c13_rb, a2_rb, as2_rb, c2_rb,
    n_experts, pairs, cap8, mblock,
    BLOCK: tl.constexpr,
):
    """Prefill variant of _desc_build_kernel_w4s: split-FP4 sub-entries
    (moe_w4q_mm, M<=4) at desc index p*4+sub for FP4-resident pairs. base/
    bs stay per-expert resident-plane pointers; a/as/c take the sub-block
    row offset (ROW-MAJOR activation bases — w4q never reads AFRAG)."""
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = p < pairs
    e = tl.load(eids_ptr + p, mask=mask, other=0).to(tl.int64)
    e = tl.minimum(tl.maximum(e, 0), n_experts - 1)
    slot = tl.load(slot_ptr + e, mask=mask, other=-1).to(tl.int64)
    npost = tl.load(npost_ptr).to(tl.int64)
    live = p < npost // mblock
    is4 = slot >= 0
    base = p.to(tl.int64) * mblock
    ref = poolb + tl.maximum(slot, 0) * slot_bytes
    for sub in tl.static_range(4):
        rbase = base + sub * 4
        m4 = tl.where(live & is4, 4, 0).to(tl.int64)
        for gi in tl.static_range(2):
            d = d_ptr + gi * cap8 + (p * 4 + sub) * 8
            if gi == 0:
                bb, rr, ss, a, as_, c = (
                    p13b + e * p13s, ref, s13b + e * s13s,
                    a1b_rm + rbase * a1_rb, as1b + rbase * as1_rb,
                    c13b + rbase * c13_rb)
            else:
                bb, rr, ss, a, as_, c = (
                    p2b + e * p2s, ref + w13r_bytes, s2b + e * s2s,
                    a2b_rm + rbase * a2_rb, as2b + rbase * as2_rb,
                    c2b + rbase * c2_rb)
            tl.store(d + 0, a, mask=mask)
            tl.store(d + 1, as_, mask=mask)
            tl.store(d + 2, bb, mask=mask)
            tl.store(d + 3, rr, mask=mask)
            tl.store(d + 4, ss, mask=mask)
            tl.store(d + 5, c, mask=mask)
            tl.store(d + 6, m4, mask=mask)
            tl.store(d + 7, tl.zeros_like(m4), mask=mask)


@triton.jit
def _desc_build_kernel_basecache(
    eids_ptr, npost_ptr, slot_ptr, miss_ptr, d_ptr,
    a1b, as1b, c13b, a2b, as2b, c2b,
    poolb, slot_bytes, off_s13, off_c2, off_s2,
    a1_rb, as1_rb, c13_rb, a2_rb, as2_rb, c2_rb,
    n_experts, pairs, cap6, mblock,
    BLOCK: tl.constexpr,
):
    """Base-cache variant of _desc_build_kernel: the 2-bit BASE planes live in
    a GPU pool (slot sections per expert: [codes13 | sc13 | codes2 | sc2]),
    not in resident per-layer planes. A live pair whose expert is NOT resident
    (slot < 0) gets m=0 (the GEMM early-EXITs; its c13/c2 rows stay zero, so
    the pair contributes nothing) and bumps `miss_ptr` — the runner fetches
    the missing experts and replays the step. Only the w2-tier tables d[0]
    (w13 GEMM) and d[1] (w2 GEMM) are written; the w4 tier is not used with
    the base cache."""
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = p < pairs
    e = tl.load(eids_ptr + p, mask=mask, other=0).to(tl.int64)
    e = tl.minimum(tl.maximum(e, 0), n_experts - 1)
    slot = tl.load(slot_ptr + e, mask=mask, other=-1).to(tl.int64)
    npost = tl.load(npost_ptr).to(tl.int64)
    live = p < npost // mblock
    hit = slot >= 0
    m = tl.where(live & hit, mblock, 0).to(tl.int64)
    n_miss = tl.sum(tl.where(mask & live & ~hit, 1, 0))
    tl.atomic_add(miss_ptr, n_miss)
    base = p.to(tl.int64) * mblock
    slot_c = tl.maximum(slot, 0)
    sbase = poolb + slot_c * slot_bytes
    a1 = a1b + base * a1_rb
    as1 = as1b + base * as1_rb
    c13 = c13b + base * c13_rb
    a2 = a2b + base * a2_rb
    as2 = as2b + base * as2_rb
    c2 = c2b + base * c2_rb
    for gi in tl.static_range(2):
        d = d_ptr + gi * cap6 + p * 6
        if gi == 0:
            b, s, a, as_, c = sbase, sbase + off_s13, a1, as1, c13
        else:
            b, s, a, as_, c = sbase + off_c2, sbase + off_s2, a2, as2, c2
        tl.store(d + 0, a, mask=mask)
        tl.store(d + 1, as_, mask=mask)
        tl.store(d + 2, b, mask=mask)
        tl.store(d + 3, s, mask=mask)
        tl.store(d + 4, c, mask=mask)
        tl.store(d + 5, m, mask=mask)


@triton.jit
def _desc_build_kernel_exactcache(
    eids_ptr, npost_ptr, slot_ptr, miss_ptr, d_ptr,
    a1b, as1b, c13b, a2b, as2b, c2b,
    poolb, slot_bytes, off_s13, off_c2, off_s2,
    a1_rb, as1_rb, c13_rb, a2_rb, as2_rb, c2_rb,
    n_experts, pairs, cap6, mblock,
    BLOCK: tl.constexpr,
):
    """Build w8 descriptors for the mandatory exact FP4-storage cache.

    Slot layout is [fp4_13|sc13|fp4_2|sc2].  Misses write m=0, increment
    the shared counter, and are zero-weighted by the epilogue until the
    runner fetches and replays the step.
    """
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = p < pairs
    e = tl.load(eids_ptr + p, mask=mask, other=0).to(tl.int64)
    e = tl.minimum(tl.maximum(e, 0), n_experts - 1)
    slot = tl.load(slot_ptr + e, mask=mask, other=-1).to(tl.int64)
    npost = tl.load(npost_ptr).to(tl.int64)
    live = p < npost // mblock
    hit = slot >= 0
    m = tl.where(live & hit, mblock, 0).to(tl.int64)
    tl.atomic_add(miss_ptr, tl.sum(tl.where(mask & live & ~hit, 1, 0)))
    row = p.to(tl.int64) * mblock
    sbase = poolb + tl.maximum(slot, 0) * slot_bytes
    for gi in tl.static_range(2):
        # d[2]/d[3] are the native-FP8/w8 tables.  d[0]/d[1] deliberately
        # remain unused: there is no approximate resident base in this mode.
        d = d_ptr + (2 + gi) * cap6 + p * 6
        if gi == 0:
            b, s, a, as_, c = (
                sbase, sbase + off_s13,
                a1b + row * a1_rb, as1b + row * as1_rb,
                c13b + row * c13_rb)
        else:
            b, s, a, as_, c = (
                sbase + off_c2, sbase + off_s2,
                a2b + row * a2_rb, as2b + row * as2_rb,
                c2b + row * c2_rb)
        tl.store(d + 0, a, mask=mask)
        tl.store(d + 1, as_, mask=mask)
        tl.store(d + 2, b, mask=mask)
        tl.store(d + 3, s, mask=mask)
        tl.store(d + 4, c, mask=mask)
        tl.store(d + 5, m, mask=mask)


@triton.jit
def _desc_build_kernel_base_delta(
    eids_ptr, npost_ptr, bslot_ptr, fslot_ptr, miss_ptr, d_ptr,
    a1b, as1b, c13b, a2b, as2b, c2b,
    bpoolb, bslot_bytes, off_s13, off_c2, off_s2,
    fpoolb, fslot_bytes, off4_s13, off4_c2, off4_s2,
    a1_rb, as1_rb, c13_rb, a2_rb, as2_rb, c2_rb,
    n_experts, pairs, cap6, mblock,
    BLOCK: tl.constexpr,
):
    """Base cache + FP4 need-pool coexistence variant: TWO slot tables with
    priority FP4 > 2-bit base slot > miss. FP4-resident pairs go to the w4
    tier (d[2]/d[3]) reading [fp4_13|sc13|fp4_2|sc2] sections from the FP4
    pool (the slots carry their own scales — no GPU-resident scale planes
    exist with a host-resident base); the rest go to the w2 tier (d[0]/d[1])
    from the base pool. A live pair resident in NEITHER pool gets m=0 in both
    tiers (contributes zero) and bumps `miss_ptr` — same replay contract as
    the plain base-cache kernel. All four desc tables are written."""
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = p < pairs
    e = tl.load(eids_ptr + p, mask=mask, other=0).to(tl.int64)
    e = tl.minimum(tl.maximum(e, 0), n_experts - 1)
    bslot = tl.load(bslot_ptr + e, mask=mask, other=-1).to(tl.int64)
    fslot = tl.load(fslot_ptr + e, mask=mask, other=-1).to(tl.int64)
    npost = tl.load(npost_ptr).to(tl.int64)
    live = p < npost // mblock
    is4 = fslot >= 0
    bhit = bslot >= 0
    m2 = tl.where(live & bhit & ~is4, mblock, 0).to(tl.int64)
    m4 = tl.where(live & is4, mblock, 0).to(tl.int64)
    n_miss = tl.sum(tl.where(mask & live & ~bhit & ~is4, 1, 0))
    tl.atomic_add(miss_ptr, n_miss)
    base = p.to(tl.int64) * mblock
    bs = bpoolb + tl.maximum(bslot, 0) * bslot_bytes
    fs = fpoolb + tl.maximum(fslot, 0) * fslot_bytes
    a1 = a1b + base * a1_rb
    as1 = as1b + base * as1_rb
    c13 = c13b + base * c13_rb
    a2 = a2b + base * a2_rb
    as2 = as2b + base * as2_rb
    c2 = c2b + base * c2_rb
    for gi in tl.static_range(4):
        d = d_ptr + gi * cap6 + p * 6
        if gi == 0:
            b, s, a, as_, c, m = bs, bs + off_s13, a1, as1, c13, m2
        elif gi == 1:
            b, s, a, as_, c, m = bs + off_c2, bs + off_s2, a2, as2, c2, m2
        elif gi == 2:
            b, s, a, as_, c, m = fs, fs + off4_s13, a1, as1, c13, m4
        else:
            b, s, a, as_, c, m = fs + off4_c2, fs + off4_s2, a2, as2, c2, m4
        tl.store(d + 0, a, mask=mask)
        tl.store(d + 1, as_, mask=mask)
        tl.store(d + 2, b, mask=mask)
        tl.store(d + 3, s, mask=mask)
        tl.store(d + 4, c, mask=mask)
        tl.store(d + 5, m, mask=mask)


@triton.jit
def _desc_build_kernel_base_delta_split(
    eids_ptr, npost_ptr, bslot_ptr, fslot_ptr, miss_ptr, d_ptr, d4s_ptr,
    a1b, as1b, c13b, a2b, as2b, c2b,
    bpoolb, bslot_bytes, off_s13, off_c2, off_s2,
    fpoolb, fslot_bytes, w13r_bytes,
    a1_rb, as1_rb, c13_rb, a2_rb, as2_rb, c2_rb,
    n_experts, pairs, cap6, cap8, mblock,
    BLOCK: tl.constexpr,
):
    """Base cache + SPLIT FP4 need-pool: quintal slots are read AGAINST
    the base pool slot (codes + scales), so a pair routes to the w4q tier
    only when its expert is resident in BOTH slot tables. FP4-mapped but
    base-missing counts as a MISS (contributes zero, bumps miss_ptr — the
    runner's base fetch + replay restores it; the base tier's eviction
    hard-excludes FP4-mapped experts so this is a transient, not a steady
    state). w2 tables (d_ptr[0..1], 6-field) serve base-resident pairs not
    in FP4; w4q tables (d4s_ptr[0..1], 8-field/64 B) carry
    {a, as, base=bslot codes section, ref=fslot section,
    bs=bslot scale section, c, m, pad}."""
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = p < pairs
    e = tl.load(eids_ptr + p, mask=mask, other=0).to(tl.int64)
    e = tl.minimum(tl.maximum(e, 0), n_experts - 1)
    bslot = tl.load(bslot_ptr + e, mask=mask, other=-1).to(tl.int64)
    fslot = tl.load(fslot_ptr + e, mask=mask, other=-1).to(tl.int64)
    npost = tl.load(npost_ptr).to(tl.int64)
    live = p < npost // mblock
    bhit = bslot >= 0
    is4 = (fslot >= 0) & bhit          # split serve needs BOTH resident
    m2 = tl.where(live & bhit & ~is4, mblock, 0).to(tl.int64)
    m4 = tl.where(live & is4, mblock, 0).to(tl.int64)
    n_miss = tl.sum(tl.where(mask & live & ~bhit, 1, 0))
    tl.atomic_add(miss_ptr, n_miss)
    base = p.to(tl.int64) * mblock
    bs = bpoolb + tl.maximum(bslot, 0) * bslot_bytes
    fs = fpoolb + tl.maximum(fslot, 0) * fslot_bytes
    a1 = a1b + base * a1_rb
    as1 = as1b + base * as1_rb
    c13 = c13b + base * c13_rb
    a2 = a2b + base * a2_rb
    as2 = as2b + base * as2_rb
    c2 = c2b + base * c2_rb
    for gi in tl.static_range(2):      # w2 tables (base pool sections)
        d = d_ptr + gi * cap6 + p * 6
        if gi == 0:
            b, s, a, as_, c = bs, bs + off_s13, a1, as1, c13
        else:
            b, s, a, as_, c = bs + off_c2, bs + off_s2, a2, as2, c2
        tl.store(d + 0, a, mask=mask)
        tl.store(d + 1, as_, mask=mask)
        tl.store(d + 2, b, mask=mask)
        tl.store(d + 3, s, mask=mask)
        tl.store(d + 4, c, mask=mask)
        tl.store(d + 5, m2, mask=mask)
    for gi in tl.static_range(2):      # w4s tables (base + refinement)
        d = d4s_ptr + gi * cap8 + p * 8
        if gi == 0:
            bb, rr, ss, a, as_, c = bs, fs, bs + off_s13, a1, as1, c13
        else:
            bb, rr, ss, a, as_, c = (bs + off_c2, fs + w13r_bytes,
                                     bs + off_s2, a2, as2, c2)
        tl.store(d + 0, a, mask=mask)
        tl.store(d + 1, as_, mask=mask)
        tl.store(d + 2, bb, mask=mask)
        tl.store(d + 3, rr, mask=mask)
        tl.store(d + 4, ss, mask=mask)
        tl.store(d + 5, c, mask=mask)
        tl.store(d + 6, m4, mask=mask)
        tl.store(d + 7, tl.zeros_like(m4), mask=mask)


@triton.jit
def _desc_build_kernel_base_delta_prefill4(
    eids_ptr, npost_ptr, bslot_ptr, fslot_ptr, miss_ptr, d_ptr,
    a1b, as1b, c13b, a2b, as2b, c2b,
    a1b_rm, a2b_rm,
    bpoolb, bslot_bytes, off_s13, off_c2, off_s2,
    fpoolb, fslot_bytes, off4_s13, off4_c2, off4_s2,
    a1_rb, as1_rb, c13_rb, a2_rb, as2_rb, c2_rb,
    n_experts, pairs, cap6, mblock,
    BLOCK: tl.constexpr,
):
    """PREFILL variant of _desc_build_kernel_base_delta (prefill-FP4 over
    the base cache): FP4-resident pairs divert to the w4 tier exactly like
    decode, but as FOUR M<=4 sub-entries per 16-token pair (d[2]/d[3] at
    index p*4+sub — the same capacity argument as the resident prefill4
    builder). w2 pair entries read the BASE pool; w4 sub-entries read the
    FP4 pool sections and the ROW-MAJOR activation bases (AFRAG repack
    leaves them valid). A live pair resident in NEITHER pool contributes
    zero and bumps miss_ptr (with ensure_resident on both tiers this is
    the pool-too-small warning path, not a steady state)."""
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = p < pairs
    e = tl.load(eids_ptr + p, mask=mask, other=0).to(tl.int64)
    e = tl.minimum(tl.maximum(e, 0), n_experts - 1)
    bslot = tl.load(bslot_ptr + e, mask=mask, other=-1).to(tl.int64)
    fslot = tl.load(fslot_ptr + e, mask=mask, other=-1).to(tl.int64)
    npost = tl.load(npost_ptr).to(tl.int64)
    live = p < npost // mblock
    is4 = fslot >= 0
    bhit = bslot >= 0
    m2 = tl.where(live & bhit & ~is4, mblock, 0).to(tl.int64)
    n_miss = tl.sum(tl.where(mask & live & ~bhit & ~is4, 1, 0))
    tl.atomic_add(miss_ptr, n_miss)
    base = p.to(tl.int64) * mblock
    bs = bpoolb + tl.maximum(bslot, 0) * bslot_bytes
    fs = fpoolb + tl.maximum(fslot, 0) * fslot_bytes
    # w2 tables (pair granularity, base pool)
    for gi in tl.static_range(2):
        d = d_ptr + gi * cap6 + p * 6
        if gi == 0:
            b, s, a, as_, c = (bs, bs + off_s13, a1b + base * a1_rb,
                               as1b + base * as1_rb, c13b + base * c13_rb)
        else:
            b, s, a, as_, c = (bs + off_c2, bs + off_s2, a2b + base * a2_rb,
                               as2b + base * as2_rb, c2b + base * c2_rb)
        tl.store(d + 0, a, mask=mask)
        tl.store(d + 1, as_, mask=mask)
        tl.store(d + 2, b, mask=mask)
        tl.store(d + 3, s, mask=mask)
        tl.store(d + 4, c, mask=mask)
        tl.store(d + 5, m2, mask=mask)
    # w4 tables (sub-entries, FP4 pool sections)
    for sub in tl.static_range(4):
        rbase = base + sub * 4
        m4 = tl.where(live & is4, 4, 0).to(tl.int64)
        for gi in tl.static_range(2):
            d = d_ptr + (2 + gi) * cap6 + (p * 4 + sub) * 6
            if gi == 0:
                b, s, a, as_, c = (fs, fs + off4_s13,
                                   a1b_rm + rbase * a1_rb,
                                   as1b + rbase * as1_rb,
                                   c13b + rbase * c13_rb)
            else:
                b, s, a, as_, c = (fs + off4_c2, fs + off4_s2,
                                   a2b_rm + rbase * a2_rb,
                                   as2b + rbase * as2_rb,
                                   c2b + rbase * c2_rb)
            tl.store(d + 0, a, mask=mask)
            tl.store(d + 1, as_, mask=mask)
            tl.store(d + 2, b, mask=mask)
            tl.store(d + 3, s, mask=mask)
            tl.store(d + 4, c, mask=mask)
            tl.store(d + 5, m4, mask=mask)


@triton.jit
def _desc_build_kernel_base_delta_split_prefill4(
    eids_ptr, npost_ptr, bslot_ptr, fslot_ptr, miss_ptr, d_ptr, d4s_ptr,
    a1b, as1b, c13b, a2b, as2b, c2b,
    a1b_rm, a2b_rm,
    bpoolb, bslot_bytes, off_s13, off_c2, off_s2,
    fpoolb, fslot_bytes, w13r_bytes,
    a1_rb, as1_rb, c13_rb, a2_rb, as2_rb, c2_rb,
    n_experts, pairs, cap6, cap8, mblock,
    BLOCK: tl.constexpr,
):
    """PREFILL variant of _desc_build_kernel_base_delta_split: quintal
    refinement read AGAINST the base slot, so a pair diverts to w4q only
    when resident in BOTH tables (same coupling as decode); diverted pairs
    are emitted as FOUR M<=4 w4q sub-entries (d4s at p*4+sub) reading the
    ROW-MAJOR activation bases. FP4-mapped/base-missing pairs count as
    misses exactly like decode (the base tier's eviction hard-excludes
    FP4-mapped experts, and prefill ensure_resident fetches the base first,
    so this is transient)."""
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = p < pairs
    e = tl.load(eids_ptr + p, mask=mask, other=0).to(tl.int64)
    e = tl.minimum(tl.maximum(e, 0), n_experts - 1)
    bslot = tl.load(bslot_ptr + e, mask=mask, other=-1).to(tl.int64)
    fslot = tl.load(fslot_ptr + e, mask=mask, other=-1).to(tl.int64)
    npost = tl.load(npost_ptr).to(tl.int64)
    live = p < npost // mblock
    bhit = bslot >= 0
    is4 = (fslot >= 0) & bhit          # split serve needs BOTH resident
    m2 = tl.where(live & bhit & ~is4, mblock, 0).to(tl.int64)
    n_miss = tl.sum(tl.where(mask & live & ~bhit, 1, 0))
    tl.atomic_add(miss_ptr, n_miss)
    base = p.to(tl.int64) * mblock
    bs = bpoolb + tl.maximum(bslot, 0) * bslot_bytes
    fs = fpoolb + tl.maximum(fslot, 0) * fslot_bytes
    for gi in tl.static_range(2):      # w2 tables (base pool sections)
        d = d_ptr + gi * cap6 + p * 6
        if gi == 0:
            b, s, a, as_, c = (bs, bs + off_s13, a1b + base * a1_rb,
                               as1b + base * as1_rb, c13b + base * c13_rb)
        else:
            b, s, a, as_, c = (bs + off_c2, bs + off_s2, a2b + base * a2_rb,
                               as2b + base * as2_rb, c2b + base * c2_rb)
        tl.store(d + 0, a, mask=mask)
        tl.store(d + 1, as_, mask=mask)
        tl.store(d + 2, b, mask=mask)
        tl.store(d + 3, s, mask=mask)
        tl.store(d + 4, c, mask=mask)
        tl.store(d + 5, m2, mask=mask)
    for sub in tl.static_range(4):     # w4s sub-entries (base + refinement)
        rbase = base + sub * 4
        m4 = tl.where(live & is4, 4, 0).to(tl.int64)
        for gi in tl.static_range(2):
            d = d4s_ptr + gi * cap8 + (p * 4 + sub) * 8
            if gi == 0:
                bb, rr, ss, a, as_, c = (
                    bs, fs, bs + off_s13,
                    a1b_rm + rbase * a1_rb, as1b + rbase * as1_rb,
                    c13b + rbase * c13_rb)
            else:
                bb, rr, ss, a, as_, c = (
                    bs + off_c2, fs + w13r_bytes, bs + off_s2,
                    a2b_rm + rbase * a2_rb, as2b + rbase * as2_rb,
                    c2b + rbase * c2_rb)
            tl.store(d + 0, a, mask=mask)
            tl.store(d + 1, as_, mask=mask)
            tl.store(d + 2, bb, mask=mask)
            tl.store(d + 3, rr, mask=mask)
            tl.store(d + 4, ss, mask=mask)
            tl.store(d + 5, c, mask=mask)
            tl.store(d + 6, m4, mask=mask)
            tl.store(d + 7, tl.zeros_like(m4), mask=mask)


def _launch(tier: str, K: int, desc: torch.Tensor, n_rows: int, pairs: int,
            stream):
    if _native_fns and tier == "w2":
        # sm_89 native decode cubin, exact-(K, N) fast path. Decode-tier
        # pairs always have m_rows <= 4 (mblock == _BLOCK); m_rows == 0
        # dead pairs early-EXIT in the kernel like the emulation. Same
        # stream, same driver-API launch class as the sm_120 cubins —
        # CUDA graph capture works unchanged. Everything else (prefill
        # mc4, w4/w4q tiers, other shapes) falls through below.
        nfn = _native_fns.get((K, n_rows))
        if nfn is not None:
            _launch_sm89_native(nfn, desc, n_rows, pairs, stream)
            return
    fn = _fns[(tier, K)]
    if not isinstance(fn, ctypes.c_void_p):
        # sm_89 path: fn is a moe_w2_sm89 Triton launcher. Same grid
        # geometry contract (one program per 16-row N-tile per pair),
        # launched on the current torch stream — the very stream `stream`
        # wraps — so ordering vs the desc build and the epilogue is
        # unchanged and capture works as before.
        fn(desc, n_rows, pairs)
        return
    args = [ctypes.c_uint64(desc.data_ptr()),
            ctypes.c_uint32(K),
            ctypes.c_uint32(K // 64),
            ctypes.c_uint32(n_rows * 2),
            ctypes.c_uint32(K // 128)]
    argv = (ctypes.c_void_p * len(args))(
        *[ctypes.cast(ctypes.byref(x), ctypes.c_void_p) for x in args])
    _ck(_driver().cuLaunchKernel(fn, n_rows // 16, pairs, 1,
                                 _nwarp_for_k(K) * 32, 1, 1, 0,
                                 stream, argv, None), "launch")


def _moe_w2_forward(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    layer_key: int,
) -> torch.Tensor:
    from vllm.model_executor.layers.quantization.utils import prefill_timers
    with prefill_timers.span("moe_w2"):
        return _moe_w2_forward_timed(x, topk_weights, topk_ids, layer_key)


def _moe_w2_forward_timed(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    layer_key: int,
) -> torch.Tensor:
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )
    global _pf4_ensure_logged, _w8_cov_logged
    # One-shot post-load FP8 boot guard (Fix B): runs on the FIRST forward
    # (after model load + finalize_auto, before serving). vLLM warmup hits
    # it, so a raise aborts the boot, attributed -- exactly the loud, early
    # failure that catches "FP8 enabled but inactive" instead of burning a
    # test cycle. No-op after the first call and when FP8 is off.
    _fp8_boot_guard()

    st = _LAYERS[layer_key]
    exact_mode = st.get("exact", False)
    T, H = x.shape
    # adaptive expert top-p (env-gated; identity when off). Must run before
    # moe_align/mark_seen/route_log so dropped experts are neither fetched
    # nor counted as routed.
    topk_weights, topk_ids = _apply_topp(topk_weights, topk_ids)
    topk_weights, route_ids = _sanitize_topk_routes(
        topk_weights, topk_ids, st["E"])
    top_k = topk_ids.shape[1]
    dev = x.device
    stream = ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream)

    # decode-sized calls use the proven 4-token kernel + delta tier;
    # prefill-sized calls use the MC4 kernel (16 tokens per pair-entry = full
    # QMMA-M, plane reads amortized 4x, ~1.5x over MC2) plus the prefill-FP4
    # tier dispatch. Threshold default 96 = the largest cudagraph capture
    # size of the standing configs: anything above is necessarily a prefill
    # chunk; short tail chunks keep the decode path. VLLM_MOE_W2_PREFILL_T
    # lowers it on low-concurrency configs so short prompts reach the
    # prefill path (and the ensure guarantee) — see _PREFILL_T.
    prefill = T > _PREFILL_T
    mblock = 16 if prefill else _BLOCK
    sorted_ids, expert_blocks, num_post = moe_align_block_size(
        route_ids, mblock, st["E"])
    slots = sorted_ids.numel()
    pairs = slots // mblock
    # st["K2"] = per-rank expert intermediate I (w2 contraction), st["K13"] =
    # hidden H (w13 contraction) -> size the workspaces for the model's shapes
    # (and correctly under tensor parallelism).
    ws = _workspaces(slots, T, dev, inter=st["K2"], hidden=st["K13"],
                     n_experts=st["E"])

    # ---- activation quant (a32: exact f32 scales, group _G1, default 32)
    # into the padded buffer; the buffer's last row is the permanent zero
    # pad row for filler slots.
    xq = ws["xq"]
    pad_row = xq.shape[0] - 1
    _quant_a32(x, xq[:T], ws["xs"][:T], _G1)
    valid = sorted_ids < T * top_k
    rows = torch.where(valid, sorted_ids // top_k,
                       torch.full_like(sorted_ids, pad_row))
    torch.index_select(xq.view(torch.uint8), 0, rows,
                       out=ws["a1"][:slots].view(torch.uint8))
    torch.index_select(ws["xs"], 0, rows, out=ws["as1"][:slots])

    # ---- desc tables in ONE triton launch
    from vllm.model_executor.layers.quantization.utils import moe_w2_delta
    base_mode = st.get("base", False)
    # FP8-delta is a PREFILL-only quality lever on sm89: only "w8" is
    # registered for FP8, the FP4 decode tiers ("w4"/"w4q") are not. When
    # FP8 is active, decode must stay on the base 2-bit path (the native
    # decode cubin / "w2" tier) -- exactly like FP8-off. This predicate
    # gates every decode-delta branch below so FP8 decode skips all FP4
    # tier machinery (slot/pool setup, FP4 desc build, w4/w4q launches).
    fp8_prefill_only = moe_w2_delta.fp8_enabled() and not exact_mode
    # AFRAG (prefill): the GEMM reads fragment-major activations from the
    # dedicated a1f/a2f buffers (filled by the single-pass triton repack
    # below); point the w2 desc 'a' fields there. The w4 tier never reads
    # AFRAG: decode is row-major anyway and the prefill-FP4 sub-entries
    # take the row-major bases explicitly (the repack copies OUT of a1/a2,
    # leaving them valid).
    # w8 consumes row-major activations. AFRAG is a w2-prefill optimization.
    use_afrag = prefill and _afrag_ok and not exact_mode
    a1_base = ws["a1f"] if use_afrag else ws["a1"]
    a2_base = ws["a2f"] if use_afrag else ws["a2"]
    d = ws["desc"]
    cap = d.shape[1]
    miss_rows = None
    use_pf4 = False        # prefill-FP4 (resident AND base modes; _PREFILL_FP4)
    # Assigned below only by the resident-plane branch.  The host/base-cache
    # branch deliberately has no FP8-delta dispatch, but the shared GEMM
    # epilogue still tests this flag.  Keep its inactive value explicit so
    # base-mode profile/warm-up forwards cannot read an unbound local.
    use_w8 = False
    if torch.cuda.is_current_stream_capturing():
        # first capture freezes the workspace sizes (see _workspaces)
        _WS["frozen"] = True
    if exact_mode:
        # Mandatory FP4-storage cache: there is no resident 2-bit fallback.
        # Eager forwards fetch the layer working set before the GEMMs; captured
        # decode graphs record misses in-graph for the runner's strict replay.
        tier = moe_w2_delta._TIER
        if tier is None or tier.n_slots <= 0:
            raise RuntimeError("moe_w2 exact cache is not initialized")
        use_w8 = True
        if torch.cuda.is_current_stream_capturing():
            tier.notify_capture()
        else:
            tier.ensure_resident(layer_key, route_ids.view(-1))
        moe_w2_delta.mark_seen(
            tier.seen[layer_key], route_ids.view(-1).long())
        if layer_key == 0:
            tier.miss_count.zero_()
        slot_row = tier.slot_table[layer_key]
        _desc_build_kernel_exactcache[(triton.cdiv(pairs, 256),)](
            expert_blocks, num_post, slot_row, tier.miss_count, d,
            a1_base.data_ptr(), ws["as1"].data_ptr(),
            ws["c13"].data_ptr(), a2_base.data_ptr(),
            ws["as2"].data_ptr(), ws["c2"].data_ptr(),
            tier.pool.data_ptr(), tier.slot_bytes,
            st["off_s13"], st["off_c2"], st["off_s2"],
            st["K13"], (st["K13"] // 32) * 4, 4 * st["K2"],
            st["K2"], (st["K2"] // 32) * 4, 2 * st["K13"],
            st["E"], pairs, cap * 6, mblock, BLOCK=256)
        e_pair = expert_blocks.to(torch.long).clamp_(0, st["E"] - 1)
        resident = slot_row[e_pair] >= 0
        miss_rows = resident.repeat_interleave(mblock)[:slots]
        # Downstream w4 launch predicates key off tier. Exact dispatch is w8
        # only; retain use_w8 and deliberately hide the tier from those paths.
        tier = None
    elif base_mode:
        # BASE cache: 2-bit planes come from the base tier's GPU pool; a live
        # pair with a non-resident expert contributes zero and bumps the miss
        # counter (runner fetches + replays). Prefill fetches its whole layer
        # working set up-front (outside capture) — decode must stay
        # capturable, so misses are handled post-hoc. The FP4 need-pool
        # (delta tier over the base cache, gate-filled) coexists on the
        # decode path: FP4-resident pairs divert to the w4 tier.
        btier = moe_w2_delta._BASE_TIER
        tier = moe_w2_delta._TIER        # FP4 need-pool (None unless opted in)
        use_pf4 = (_PREFILL_FP4 and prefill and tier is not None
                   and tier.n_slots > 0)
        if torch.cuda.is_current_stream_capturing():
            btier.notify_capture()
            if tier is not None:
                tier.notify_capture()
        elif prefill:
            btier.ensure_resident(layer_key, route_ids.view(-1))
            if use_pf4 and _PREFILL_FP4_ENSURE:
                # decode-class guarantee for the FP4 side too: fetch the
                # chunk's routed set into the need-pool AFTER the base rows
                # (split w4q reads the refinement AGAINST the base slot, and
                # the base eviction hard-excludes FP4-mapped experts, so the
                # base ensure must land first). Layer pins rotate per tier.
                if not _pf4_ensure_logged:
                    _pf4_ensure_logged = True
                    logger.info("moe_w2 prefill-FP4 ensure mode (base "
                                "cache): eager chunk working sets fetched "
                                "to both tiers (first call: layer %d, T=%d)",
                                layer_key, T)
                tier.ensure_resident(layer_key, route_ids.view(-1))
        moe_w2_delta.mark_seen(btier.seen[layer_key], route_ids.view(-1).long())
        if tier is not None:
            # the gate's force_promote reads the FP4 tier's own seen scatter
            moe_w2_delta.mark_seen(tier.seen[layer_key],
                                   route_ids.view(-1).long())
        if not prefill:
            # LOOKA/PILOT (router-lookahead): score predictors + write the
            # next layer's prediction. Must run BEFORE the route_log
            # overwrite below (predictor [0] reads last step's ids from it).
            # In-graph safe (persistent buffers, static shapes); no-op
            # unless armed.
            from vllm.model_executor.layers.quantization.utils import (
                moe_w2_looka)
            if moe_w2_looka.enabled():
                moe_w2_looka.record(layer_key, x, topk_ids, btier.route_log)
        if not prefill and btier.route_log is not None:
            # per-(token,layer) routing log for the draft-prefetch predictor:
            # a static [n_layers, T_cap, k_cap] buffer the runner reads back
            # post-step (~KBs). In-graph safe: fixed shapes per captured
            # size, static destination. Rows beyond this step's real token
            # count hold stale ids — the host slices by the true T.
            _t = min(topk_ids.shape[0], btier.route_log.shape[1])
            _k = min(topk_ids.shape[1], btier.route_log.shape[2])
            btier.route_log[layer_key, :_t, :_k].copy_(
                topk_ids[:_t, :_k], non_blocking=True)
        if layer_key == 0:
            # per-step counter reset, in-graph (layer 0 runs first each step)
            btier.miss_count.zero_()
        slot_row = btier.slot_table[layer_key]
        use_fp4 = tier is not None and not prefill and not fp8_prefill_only
        use_w4s_base = use_fp4 and moe_w2_delta.split_enabled()
        if use_pf4 and moe_w2_delta.split_enabled():
            # prefill-FP4 over the base cache, SPLIT: w2 pair entries from
            # the base pool + w4q SUB-entries (M<=4) coupling base codes
            # with the quintal refinement. See the prefill4 builders.
            fslot_row = tier.slot_table[layer_key]
            d4s = ws["desc4s"]
            _desc_build_kernel_base_delta_split_prefill4[
                    (triton.cdiv(pairs, 256),)](
                expert_blocks, num_post, slot_row, fslot_row,
                btier.miss_count, d, d4s,
                a1_base.data_ptr(), ws["as1"].data_ptr(), ws["c13"].data_ptr(),
                a2_base.data_ptr(), ws["as2"].data_ptr(), ws["c2"].data_ptr(),
                ws["a1"].data_ptr(), ws["a2"].data_ptr(),
                btier.pool.data_ptr(), btier.slot_bytes,
                st["off_s13"], st["off_c2"], st["off_s2"],
                tier.pool.data_ptr(), tier.slot_bytes, tier.w13_bytes,
                st["K13"], (st["K13"] // 32) * 4, 4 * st["K2"], st["K2"],
                (st["K2"] // 32) * 4, 2 * st["K13"],
                st["E"], pairs, cap * 6, d4s.shape[1] * 8, mblock, BLOCK=256)
        elif use_pf4:
            # prefill-FP4 over the base cache, NON-split: w4 sub-entries
            # read full-FP4 sections from the need-pool slots.
            fslot_row = tier.slot_table[layer_key]
            _desc_build_kernel_base_delta_prefill4[
                    (triton.cdiv(pairs, 256),)](
                expert_blocks, num_post, slot_row, fslot_row,
                btier.miss_count, d,
                a1_base.data_ptr(), ws["as1"].data_ptr(), ws["c13"].data_ptr(),
                a2_base.data_ptr(), ws["as2"].data_ptr(), ws["c2"].data_ptr(),
                ws["a1"].data_ptr(), ws["a2"].data_ptr(),
                btier.pool.data_ptr(), btier.slot_bytes,
                st["off_s13"], st["off_c2"], st["off_s2"],
                tier.pool.data_ptr(), tier.slot_bytes,
                st["off4_s13"], st["off4_c2"], st["off4_s2"],
                st["K13"], (st["K13"] // 32) * 4, 4 * st["K2"], st["K2"],
                (st["K2"] // 32) * 4, 2 * st["K13"],
                st["E"], pairs, cap * 6, mblock, BLOCK=256)
        elif use_w4s_base:
            fslot_row = tier.slot_table[layer_key]
            d4s = ws["desc4s"]
            _desc_build_kernel_base_delta_split[(triton.cdiv(pairs, 256),)](
                expert_blocks, num_post, slot_row, fslot_row,
                btier.miss_count, d, d4s,
                a1_base.data_ptr(), ws["as1"].data_ptr(), ws["c13"].data_ptr(),
                a2_base.data_ptr(), ws["as2"].data_ptr(), ws["c2"].data_ptr(),
                btier.pool.data_ptr(), btier.slot_bytes,
                st["off_s13"], st["off_c2"], st["off_s2"],
                tier.pool.data_ptr(), tier.slot_bytes, tier.w13_bytes,
                st["K13"], (st["K13"] // 32) * 4, 4 * st["K2"], st["K2"],
                (st["K2"] // 32) * 4, 2 * st["K13"],
                st["E"], pairs, cap * 6, d4s.shape[1] * 8, mblock, BLOCK=256)
        elif use_fp4:
            fslot_row = tier.slot_table[layer_key]
            _desc_build_kernel_base_delta[(triton.cdiv(pairs, 256),)](
                expert_blocks, num_post, slot_row, fslot_row,
                btier.miss_count, d,
                a1_base.data_ptr(), ws["as1"].data_ptr(), ws["c13"].data_ptr(),
                a2_base.data_ptr(), ws["as2"].data_ptr(), ws["c2"].data_ptr(),
                btier.pool.data_ptr(), btier.slot_bytes,
                st["off_s13"], st["off_c2"], st["off_s2"],
                tier.pool.data_ptr(), tier.slot_bytes,
                st["off4_s13"], st["off4_c2"], st["off4_s2"],
                st["K13"], (st["K13"] // 32) * 4, 4 * st["K2"], st["K2"],
                (st["K2"] // 32) * 4, 2 * st["K13"],
                st["E"], pairs, cap * 6, mblock, BLOCK=256)
        else:
            _desc_build_kernel_basecache[(triton.cdiv(pairs, 256),)](
                expert_blocks, num_post, slot_row,
                btier.miss_count, d,
                a1_base.data_ptr(), ws["as1"].data_ptr(), ws["c13"].data_ptr(),
                a2_base.data_ptr(), ws["as2"].data_ptr(), ws["c2"].data_ptr(),
                btier.pool.data_ptr(), btier.slot_bytes,
                st["off_s13"], st["off_c2"], st["off_s2"],
                st["K13"], (st["K13"] // 32) * 4, 4 * st["K2"], st["K2"],
                (st["K2"] // 32) * 4, 2 * st["K13"],
                st["E"], pairs, cap * 6, mblock, BLOCK=256)
        # Miss pairs get scatter weight 0: the GEMMs early-EXIT on m=0 and
        # never write their c13/c2 rows, but those workspace rows hold STALE
        # values from a previous forward — zeroing the WEIGHT (not the rows)
        # makes the miss contribution an exact 0 for free. Graph-safe (pure
        # tensor ops on captured buffers). FP4-resident pairs are NOT misses
        # — except under split, where serving needs the BASE slot too (an
        # FP4-mapped/base-missing pair contributed zero and must replay).
        e_pair = expert_blocks.to(torch.long).clamp_(0, st["E"] - 1)
        resident = (slot_row[e_pair] >= 0)
        if (use_fp4 or use_pf4) and not moe_w2_delta.split_enabled():
            # non-split: a full-FP4 slot serves the pair without its base
            # row (decode d[2]/d[3] or the prefill4 sub-entries).
            resident |= (fslot_row[e_pair] >= 0)
        miss_rows = resident.repeat_interleave(mblock)[:slots]
        if not use_fp4 and not use_pf4:
            tier = None      # downstream w4 launches key off `tier`
    else:
        tier = moe_w2_delta._TIER       # peek only; created by the plane builder
        # FP8-e4m3 delta PREFILL tier (Ada native FP8 MMA; mutually exclusive
        # with FP4 — when FP8 is active, use_w8 wins and use_pf4 stays False).
        # w8 takes M<=16 in one shot (BLOCK_M=16 masked by m_rows), so prefill
        # pairs dispatch at PAIR granularity (no sub-entry split like w4).
        use_w8 = (prefill and moe_w2_delta.fp8_enabled()
                  and tier is not None and tier.n_slots > 0)
        # prefill consumes the FP4 tier too (quality lever, see _PREFILL_FP4)
        use_pf4 = (_PREFILL_FP4 and prefill and tier is not None
                   and tier.n_slots > 0 and not use_w8)
        if tier is not None and not prefill and not fp8_prefill_only:
            if torch.cuda.is_current_stream_capturing():
                tier.notify_capture()
            slot_row = tier.slot_table[layer_key]
            pool_ptr = tier.pool.data_ptr()
            moe_w2_delta.mark_seen(tier.seen[layer_key],
                                   route_ids.view(-1).long())
        else:
            if tier is not None:
                # seen marks land BEFORE the desc build: the manager's
                # victim selection excludes this window's experts, so a
                # background pass cannot rewrite a slot between the desc
                # build below and the GEMMs reading it (same protection
                # class as decode's step-scoped windows).
                moe_w2_delta.mark_seen(tier.seen[layer_key],
                                       route_ids.view(-1).long())
            if use_pf4 or use_w8:
                if torch.cuda.is_current_stream_capturing():
                    tier.notify_capture()
                elif _PREFILL_FP4_ENSURE:
                    # decode-class guarantee: fetch the chunk's routed set
                    # into the pool before the desc build reads slot_table
                    # (base-cache prefill idiom; layer pins rotate).
                    if not _pf4_ensure_logged:
                        _pf4_ensure_logged = True
                        logger.info("moe_w2 prefill-FP4/FP8 ensure mode: "
                                    "eager chunk working sets fetched to "
                                    "the delta tier (first call: layer %d, "
                                    "T=%d)", layer_key, T)
                    tier.ensure_resident(layer_key, route_ids.view(-1))
                slot_row = tier.slot_table[layer_key]
                pool_ptr = tier.pool.data_ptr()
                if use_w8 and os.getenv(
                        "VLLM_MOE_W2_FP8_TRACE", "0") not in (
                        "0", "false", "no", "off"):
                    # One-shot w8 coverage probe: confirm the (now-dynamic) w8
                    # pool is actually covering the hot-expert subset during
                    # the prefill. Fires for the first N w8 prefills only.
                    if _w8_cov_logged < 8:
                        _w8_cov_logged += 1
                        _row = tier.slot_table[layer_key]
                        _n_res = int((_row >= 0).sum())
                        _n_free = len(tier._free) if hasattr(
                            tier, "_free") else 0
                        logger.info(
                            "moe_w2 w8 prefill: layer %d use_w8=True "
                            "pool_slots=%d resident_this_layer=%d "
                            "free=%d T=%d",
                            layer_key, tier.n_slots, _n_res, _n_free, T)
            else:
                slot_row = ws["no_slots"]
                pool_ptr = ws["a1"].data_ptr()  # never dereferenced (m4=0)
        if use_pf4:
            # w2 tables at pair granularity (diverted pairs m=0) + w4 tables
            # at SUB-ENTRY granularity (p*4+sub, m<=4 — the w4/w4q kernel M
            # limit). w4 entries always read ROW-MAJOR a1/a2 (AFRAG repack
            # leaves the row-major source buffers intact).
            _desc_build_kernel_prefill4[(triton.cdiv(pairs, 256),)](
                expert_blocks, num_post, slot_row, d,
                a1_base.data_ptr(), ws["as1"].data_ptr(),
                ws["c13"].data_ptr(), a2_base.data_ptr(),
                ws["as2"].data_ptr(), ws["c2"].data_ptr(),
                ws["a1"].data_ptr(), ws["a2"].data_ptr(),
                st["planes13"].data_ptr(), st["sc13"].data_ptr(),
                st["planes2"].data_ptr(), st["sc2"].data_ptr(), pool_ptr,
                st["planes13"].shape[1], st["sc13"].shape[1],
                st["planes2"].shape[1], st["sc2"].shape[1],
                tier.slot_bytes, tier.w13_bytes,
                st["K13"], (st["K13"] // 32) * 4, 4 * st["K2"], st["K2"],
                (st["K2"] // 32) * 4, 2 * st["K13"],
                st["E"], pairs, cap * 6, mblock, BLOCK=256)
        else:
            _desc_build_kernel[(triton.cdiv(pairs, 256),)](
                expert_blocks, num_post, slot_row, d,
                a1_base.data_ptr(), ws["as1"].data_ptr(),
                ws["c13"].data_ptr(),
                a2_base.data_ptr(), ws["as2"].data_ptr(), ws["c2"].data_ptr(),
                st["planes13"].data_ptr(), st["sc13"].data_ptr(),
                st["planes2"].data_ptr(), st["sc2"].data_ptr(), pool_ptr,
                st["planes13"].shape[1], st["sc13"].shape[1],
                st["planes2"].shape[1], st["sc2"].shape[1],
                (tier.slot_bytes if tier is not None
                 else moe_w2_delta.SLOT_BYTES),
                (tier.w13_bytes if tier is not None
                 else moe_w2_delta.W13_BYTES),
                # row strides (bytes). H-side: a1 fp8 [H], as1 f32 [H/32]
                # (a32 per-32 groups; the a128 lineage was f32 [H/128]), c2
                # bf16 [H]. per-rank intermediate side: c13 bf16 [2I], a2
                # fp8 [I], as2 f32 [I/32]. K13 = H, K2 = I; GLM-5.x gets
                # H=6144, TP shards shrink I.
                st["K13"], (st["K13"] // 32) * 4, 4 * st["K2"], st["K2"],
                (st["K2"] // 32) * 4, 2 * st["K13"],
                st["E"], pairs, cap * 6, mblock, BLOCK=256)
        if (tier is not None and (not prefill or use_pf4)
                and moe_w2_delta.split_enabled()):
            # split-FP4: the extra 8-field tables for moe_w4q_mm (base/bs =
            # the resident plane rows, ref = the slot's quintal sections).
            # Prefill emits sub-entries (M<=4) via the prefill4 variant.
            d4s = ws["desc4s"]
            builder = (_desc_build_kernel_w4s_prefill4 if use_pf4
                       else _desc_build_kernel_w4s)
            a1_w4s = ws["a1"] if use_pf4 else a1_base
            a2_w4s = ws["a2"] if use_pf4 else a2_base
            builder[(triton.cdiv(pairs, 256),)](
                expert_blocks, num_post, slot_row, d4s,
                a1_w4s.data_ptr(), ws["as1"].data_ptr(),
                ws["c13"].data_ptr(), a2_w4s.data_ptr(),
                ws["as2"].data_ptr(), ws["c2"].data_ptr(),
                st["planes13"].data_ptr(), st["sc13"].data_ptr(),
                st["planes2"].data_ptr(), st["sc2"].data_ptr(), pool_ptr,
                st["planes13"].shape[1], st["sc13"].shape[1],
                st["planes2"].shape[1], st["sc2"].shape[1],
                tier.slot_bytes, tier.w13_bytes,
                st["K13"], (st["K13"] // 32) * 4, 4 * st["K2"], st["K2"],
                (st["K2"] // 32) * 4, 2 * st["K13"],
                st["E"], pairs, d4s.shape[1] * 8, mblock, BLOCK=256)

    # ---- w13 GEMMs (both tiers) -> fused silu*up -> quant -> w2 GEMMs
    # AFRAG prefill: single-pass triton repack row-major a1/a2 -> fragment-major
    # a1f/a2f (desc built against a1f/a2f above) so the GEMM loads each m16k32
    # A-fragment in one LDG.128. Numerics bit-identical to mc4.
    w2tier = ("w2mc4afrag" if use_afrag else "w2mc4") if prefill else "w2"
    # AFRAG repacks COMPLETE 16-row tiles. `slots` is moe_align's OVER-ALLOCATED
    # row count (sorted_ids.numel() = topk*T + E*15), NOT a multiple of 16; the
    # desc/kernel only ever touch the first `pairs*16` rows (num_post <= pairs*16),
    # so repack exactly that tile-aligned region. Rows [pairs*16:slots] are unused
    # filler (never read). Capacity is fine: pairs*16 <= slots <= a1.shape[0]-4.
    if use_afrag:
        _afrag_repack(ws["a1"], ws["a1f"], pairs, st["K13"])
    if not exact_mode:
        _launch(w2tier, st["K13"], d[0], st["N13"], pairs, stream)
    # split-FP4 dispatch: both residency modes fill ws["desc4s"] (classic:
    # _desc_build_kernel_w4s against resident planes; base cache:
    # _desc_build_kernel_base_delta_split against the coupled base slots).
    # Prefill-FP4 launches the w4 tier over SUB-ENTRIES (4 x M<=4 rows per
    # 16-token pair -> grid pairs*4), see _desc_build_kernel_prefill4.
    use_w4s = (tier is not None and not prefill
               and moe_w2_delta.split_enabled())
    if tier is not None and not prefill and not fp8_prefill_only:
        if use_w4s:
            _launch("w4q", st["K13"], ws["desc4s"][0], st["N13"], pairs,
                    stream)
        else:
            _launch("w4", st["K13"], d[2], st["N13"], pairs, stream)
    elif use_w8:
        # FP8 delta prefill: 6-field desc at PAIR granularity (w8 takes M<=16,
        # no sub-entry split). d[2] was built by _desc_build_kernel above.
        _launch("w8", st["K13"], d[2], st["N13"], pairs, stream)
    elif use_pf4:
        if moe_w2_delta.split_enabled():
            _launch("w4q", st["K13"], ws["desc4s"][0], st["N13"], pairs * 4,
                    stream)
        else:
            _launch("w4", st["K13"], d[2], st["N13"], pairs * 4, stream)
    act = ws["act"][:slots]
    torch.ops._C.silu_and_mul(act, ws["c13"][:slots])
    # mid-pipeline requant (group _G2, default 32 — the a128-era group-128
    # requant here was one of the two activation-precision gaps vs native)
    _quant_a32(act, ws["a2"][:slots], ws["as2"][:slots], _G2)
    if use_afrag:
        _afrag_repack(ws["a2"], ws["a2f"], pairs, st["K2"])
    if not exact_mode:
        _launch(w2tier, st["K2"], d[1], st["N2"], pairs, stream)
    if tier is not None and not prefill and not fp8_prefill_only:
        if use_w4s:
            _launch("w4q", st["K2"], ws["desc4s"][1], st["N2"], pairs, stream)
        else:
            _launch("w4", st["K2"], d[3], st["N2"], pairs, stream)
    elif use_w8:
        _launch("w8", st["K2"], d[3], st["N2"], pairs, stream)
    elif use_pf4:
        if moe_w2_delta.split_enabled():
            _launch("w4q", st["K2"], ws["desc4s"][1], st["N2"], pairs * 4,
                    stream)
        else:
            _launch("w4", st["K2"], d[3], st["N2"], pairs * 4, stream)

    # ---- weighted unpermute (pad/sentinel/miss rows masked), DETERMINISTIC.
    # Atomic index_add was nondeterministic; the first deterministic torch
    # replacement materialized ~414 MiB of FP32 route intermediates at the
    # 1,044-token prefill shape and OOMed a 100K request.  Invert moe_align's
    # collision-free valid permutation into persistent int32 scratch, then
    # have one Triton program accumulate each output tile in fixed route-j
    # order.  This preserves deterministic, graph-safe behavior while the
    # only per-call allocation is the returned BF16 [T,H] tensor.
    out = torch.empty_like(x)
    return _deterministic_unpermute(
        ws["c2"][:slots], sorted_ids, route_ids, topk_weights, miss_rows,
        ws["route_slot"], out, st["E"])


def _moe_w2_forward_fake(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    layer_key: int,
) -> torch.Tensor:
    return torch.empty_like(x)


direct_register_custom_op(
    "moe_w2_forward",
    _moe_w2_forward,
    fake_impl=_moe_w2_forward_fake,
)


def moe_w2_forward(x, topk_weights, topk_ids, layer_key):
    return torch.ops.vllm.moe_w2_forward(x, topk_weights, topk_ids, layer_key)


@functools.cache
def ready() -> bool:
    return enabled() and _ensure_ready()
