#!/usr/bin/env python3
"""Offline GPU validation for the IQ2_XXS / Q2_K MoE forward wiring (Step 5a).

End-to-end exercise of ``Mxfp4MoEMethod._iq2_forward`` -- the per-expert
Python-loop routed-expert forward that wires the proven Step 4b/4c Triton
GEMMs (``iq2_xxs_mm`` for gate/up, ``q2_k_mm`` for down) into the MoE
dispatch+combine. This test does NOT boot vLLM and does NOT import the full
vLLM tree; it AST-extracts the ``_iq2_forward`` source from the overlay
mxfp4.py, exec's it against a small fake ``_iq2_active`` layer carrying
random IQ2_XXS / Q2_K block bytes, and compares against an independent
per-expert reference (dequant -> torch matmul -> SwiGLU -> dequant ->
matmul -> weighted combine).

Why AST extraction (not ``import vllm...``): the dev worktree has no vLLM
installed, and mxfp4.py's top-level imports (vllm.logger, fused_moe,
modular_kernel, ...) are heavy. The wiring under test is self-contained --
``_iq2_forward`` only touches ``layer`` attributes, ``torch``, the two
constants ``_IQ2_XXS_BLOCK_BYTES`` / ``_Q2_K_BLOCK_BYTES`` / ``_QK_K``, and
the two GEMM modules it imports lazily. We exec exactly that body so the
REAL code path runs (not a replay), with the lazy GEMM imports resolving to
the overlay files via stub packages.

What this validates:
  1. The wiring is call-shape correct: ``_iq2_forward`` runs end-to-end on a
     GPU; output is finite + the right shape/dtype.
  2. No gross wiring bug (transpose flip, expert-id swap, combine-weight
     inversion) -- caught by output-direction + magnitude checks against the
     fp32 reference. Single-expert kernel precision (~1-3e-3 rel) is taken as
     given from Step 4b/4c; this test compounds three GEMMs + SwiGLU in bf16,
     so the relative band is wider and we rely primarily on cosine direction.

What this does NOT validate:
  - The real DSv4 checkpoint (only random bytes here; the boot test, Step 5b,
    exercises real weights at serving scale).
  - Peak throughput (this is a correctness path, not a benchmark).

Requires a CUDA GPU. Free the GPUs first (``docker stop moet`` -- production
moet serves on both).

Run:    python3 tools/test_iq2_moe_forward.py
Exit:   0 on success, non-zero on any mismatch.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
import textwrap
import types
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
UTILS = HERE.parent / "overlay" / "vllm" / "vllm" / "model_executor" / \
    "layers" / "quantization" / "utils"
sys.path.insert(0, str(UTILS))

from iq2_xxs_ref import (  # noqa: E402
    QK_K, IQ2_XXS_BLOCK_BYTES, Q2_K_BLOCK_BYTES,
    dequant_iq2_xxs, dequant_q2_k,
)

# Register stub packages so the lazy import inside _iq2_forward
# (`from vllm.model_executor.layers.quantization.utils import
# iq2_xxs_mm_triton, q2_k_mm_triton`) resolves to the overlay files.
OVERLAY_ROOT = HERE.parent / "overlay"
_vllm = types.ModuleType("vllm")
_vllm.__path__ = [str(OVERLAY_ROOT / "vllm")]
sys.modules["vllm"] = _vllm


def _stub_pkg(dotted, rel):
    mod = types.ModuleType(dotted)
    mod.__path__ = [str(OVERLAY_ROOT / "vllm" / rel)]
    sys.modules[dotted] = mod


for d, r in [
    ("vllm.model_executor", "model_executor"),
    ("vllm.model_executor.layers", "model_executor/layers"),
    ("vllm.model_executor.layers.quantization",
     "model_executor/layers/quantization"),
    ("vllm.model_executor.layers.quantization.utils",
     "model_executor/layers/quantization/utils"),
]:
    _stub_pkg(d, r)

# The lazy import inside _iq2_forward resolves
# `from vllm.model_executor.layers.quantization.utils import
# iq2_xxs_mm_triton, q2_k_mm_triton`. Stub __path__ alone is unreliable when
# a real vllm is installed in the image; load the overlay modules directly
# by path and register them under the expected qualified names so the lazy
# import finds them (and their proven _get_luts / iq2_xxs_mm / q2_k_mm).
_UTILS_PKG = "vllm.model_executor.layers.quantization.utils"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    setattr(sys.modules[_UTILS_PKG], path.stem, mod)
    return mod


_load_module(
    _UTILS_PKG + ".iq2_xxs_ref",
    UTILS / "iq2_xxs_ref.py")
_load_module(
    _UTILS_PKG + ".iq2_xxs_mm_triton",
    UTILS / "iq2_xxs_mm_triton.py")
_load_module(
    _UTILS_PKG + ".q2_k_mm_triton",
    UTILS / "q2_k_mm_triton.py")


def _extract_method(src_path: Path, method_name: str) -> str:
    """AST-extract a method's source from a .py file (no exec of the file).

    Returns the method as a standalone `def` (self-receiving) source string.
    """
    src = src_path.read_text()
    tree = ast.parse(src)
    cls_node = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.ClassDef)
                and node.name == "Mxfp4MoEMethod"):
            cls_node = node
            break
    if cls_node is None:
        raise RuntimeError("Mxfp4MoEMethod class not found in mxfp4.py")
    for node in cls_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            seg = ast.get_source_segment(src, node)
            if seg is None:
                raise RuntimeError(f"could not extract {method_name} source")
            return textwrap.dedent(seg)
    raise RuntimeError(f"method {method_name} not found on Mxfp4MoEMethod")


MXFP4 = HERE.parent / "overlay" / "vllm" / "vllm" / "model_executor" / \
    "layers" / "quantization" / "mxfp4.py"
_method_src = _extract_method(MXFP4, "_iq2_forward")

# Exec the method in a namespace that provides its free globals.
_WIRED_NS = {
    "torch": torch,
    "_IQ2_XXS_BLOCK_BYTES": IQ2_XXS_BLOCK_BYTES,
    "_Q2_K_BLOCK_BYTES": Q2_K_BLOCK_BYTES,
    "_QK_K": QK_K,
    "__name__": "iq2_forward_test",
}
exec(compile(_method_src, str(MXFP4), "exec"), _WIRED_NS)
iq2_forward = _WIRED_NS["_iq2_forward"]
print("[setup] AST-extracted Mxfp4MoEMethod._iq2_forward and exec'd it "
      "against a stubbed namespace (no vLLM import chain).")

# ---------------------------------------------------------------------------
# Test shapes -- QK_K-aligned for both IQ2_XXS (gate/up K = hidden) and Q2_K
# (down K = intermediate).
# ---------------------------------------------------------------------------
HIDDEN = 512        # gate/up K  (2 IQ2_XXS blocks per row)
INTERMEDIATE = 256  # down K     (1 Q2_K block per row)
NUM_EXPERTS = 4
TOPK = 2
NUM_TOKENS = 8
SEED = 7


def _expert_blocks_np(make_fn, n, k, base_seed, d_scale=1.0):
    """Stack NUM_EXPERTS valid [N, row_bytes] uint8 blocks via the proven
    Step 4b/4c random-weight makers (they sample finite fp16 block scales --
    raw randint can produce inf/nan fp16 d-fields). Optionally scale the
    fp16 block scales (d / dmin) by ``d_scale`` to keep output magnitudes
    in a range where bf16 precision is meaningful (default 1.0 = as generated).
    """
    slabs = [make_fn(n, k, seed=base_seed + e) for e in range(NUM_EXPERTS)]
    arr = np.stack(slabs, axis=0)  # [E, N, row_bytes]
    if d_scale != 1.0:
        arr = _scale_block_scales(arr, make_fn, d_scale)
    return arr


def _scale_block_scales(arr, make_fn, d_scale):
    """Scale the fp16 d-field(s) in-place by d_scale. IQ2_XXS: d at bytes
    [0:2] of each 66-B block. Q2_K: d at [80:82], dmin at [82:84] of each
    84-B block. Detected from block_bytes via the maker's module."""
    import importlib
    # Resolve the constants from the maker's module.
    mod = importlib.import_module(make_fn.__module__)
    bb = getattr(mod, "IQ2_XXS_BLOCK_BYTES", None) or \
        getattr(mod, "Q2_K_BLOCK_BYTES")
    is_iq2 = hasattr(mod, "IQ2_XXS_BLOCK_BYTES")
    flat = arr.reshape(-1, bb).copy()  # [n_blocks_total, bb]
    if is_iq2:
        d = flat[:, 0:2].copy().view(np.float16).astype(np.float32)
        d *= np.float16(d_scale)
        flat[:, 0:2] = np.ascontiguousarray(d.astype(np.float16)).view(np.uint8)
    else:
        for off in (80, 82):  # d and dmin
            v = flat[:, off:off + 2].copy().view(np.float16).astype(np.float32)
            v *= np.float16(d_scale)
            flat[:, off:off + 2] = np.ascontiguousarray(
                v.astype(np.float16)).view(np.uint8)
    return flat.reshape(arr.shape)


def _build_inputs(rng):
    from vllm.model_executor.layers.quantization.utils.iq2_xxs_mm_triton \
        import _make_random_iq2xxs_weights
    from vllm.model_executor.layers.quantization.utils.q2_k_mm_triton \
        import _make_random_q2_k_weights

    # d_scale=0.1 keeps the 3-GEMM compound output in a sane range so bf16
    # rounding does not dominate the relative-error / cosine signal (the
    # random d-field in [0.001, 1.0] otherwise produces ~1e6 outputs).
    gate_np = _expert_blocks_np(
        _make_random_iq2xxs_weights, INTERMEDIATE, HIDDEN,
        base_seed=11, d_scale=0.1)
    up_np = _expert_blocks_np(
        _make_random_iq2xxs_weights, INTERMEDIATE, HIDDEN,
        base_seed=23, d_scale=0.1)
    down_np = _expert_blocks_np(
        _make_random_q2_k_weights, HIDDEN, INTERMEDIATE,
        base_seed=37, d_scale=0.1)
    gate_w = torch.from_numpy(np.ascontiguousarray(gate_np)).cuda()
    up_w = torch.from_numpy(np.ascontiguousarray(up_np)).cuda()
    down_w = torch.from_numpy(np.ascontiguousarray(down_np)).cuda()

    x = torch.randn(NUM_TOKENS, HIDDEN, dtype=torch.bfloat16,
                    generator=rng).cuda()
    topk_ids = torch.empty(NUM_TOKENS, TOPK, dtype=torch.long)
    for t in range(NUM_TOKENS):
        perm = torch.randperm(NUM_EXPERTS, generator=rng)[:TOPK]
        topk_ids[t] = perm
    logits = torch.randn(NUM_TOKENS, NUM_EXPERTS, generator=rng)
    weights = torch.softmax(logits, dim=-1).to(torch.float32)
    topk_weights = weights.gather(1, topk_ids)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return gate_w, up_w, down_w, x, topk_weights.cuda(), topk_ids.cuda()


class _FakeLayer:
    """Stand-in exposing only what _iq2_forward touches."""

    def __init__(self, gate_w, up_w, down_w, expert_map=None,
                 global_num_experts=NUM_EXPERTS):
        # _iq2_forward only reads .shape and indexes [expert_id]; the real
        # loader registers these as Parameters(requires_grad=False), but for
        # the test the raw uint8 tensors suffice (torch 2.11 rejects grad on
        # integer dtypes anyway).
        self.gate_weight_iq2_xxs = gate_w
        self.up_weight_iq2_xxs = up_w
        self.down_weight_q2_k = down_w
        self.expert_map = expert_map  # None or int32[num_global]
        self.global_num_experts = global_num_experts
        self.apply_router_weight_on_input = False


class _FakeSelf:
    """The method reads no `self` attributes (the _iq2_active guard is in
    apply(), not _iq2_forward), so a bare object suffices as the receiver."""
    pass


def _reference_forward(layer, x, topk_weights, topk_ids):
    """Per-expert numpy fp32 reference.

    For each expert e: dequant gate/up/down -> matmul -> SwiGLU -> matmul.
    Combine = sum over topk of weight * expert_out (router weights applied
    AFTER the down GEMM, matching cpu_fused_moe_torch). Honors ``expert_map``:
    a dispatch whose global expert maps to -1 (off-rank under EP) contributes
    zero (the runner's all-reduce fills it from the peer rank).
    """
    gate_np = layer.gate_weight_iq2_xxs.cpu().numpy()
    up_np = layer.up_weight_iq2_xxs.cpu().numpy()
    down_np = layer.down_weight_q2_k.cpu().numpy()
    x_np = x.float().cpu().numpy()
    tk_w = topk_weights.float().cpu().numpy()
    tk_ids = topk_ids.cpu().numpy()
    e_local = gate_np.shape[0]
    emap = (layer.expert_map.cpu().numpy()
            if layer.expert_map is not None else None)

    W_gate = [dequant_iq2_xxs(gate_np[e].tobytes(),
                              (INTERMEDIATE, HIDDEN)) for e in range(e_local)]
    W_up = [dequant_iq2_xxs(up_np[e].tobytes(),
                            (INTERMEDIATE, HIDDEN)) for e in range(e_local)]
    W_down = [dequant_q2_k(down_np[e].tobytes(),
                           (HIDDEN, INTERMEDIATE)) for e in range(e_local)]

    out = np.zeros((NUM_TOKENS, HIDDEN), dtype=np.float32)
    for t in range(NUM_TOKENS):
        for k in range(TOPK):
            g_id = int(tk_ids[t, k])
            if emap is not None:
                local = int(emap[g_id])
                if local < 0:
                    continue  # off-rank: contributes zero here
            else:
                local = g_id
            w = float(tk_w[t, k])
            g = x_np[t] @ W_gate[local].T
            u = x_np[t] @ W_up[local].T
            mid = _silu_np(g) * u  # SwiGLU = SiLU(gate) * up
            o = mid @ W_down[local].T
            out[t] += w * o
    return out


def _silu_np(g):
    """Numerically stable SiLU = g * sigmoid(g) (avoids np.exp overflow on
    large-negative g)."""
    out = np.empty_like(g)
    pos = g >= 0
    out[pos] = g[pos] / (1.0 + np.exp(-g[pos]))
    eg = np.exp(g[~pos])
    out[~pos] = g[~pos] * eg / (1.0 + eg)
    return out


def _cosine(a, b):
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a.ravel(), b.ravel()) / (na * nb))


def _run_case(label, layer, x, tk_w, tk_ids, self_obj) -> bool:
    """Run _iq2_forward on one layer/input config, compare to the reference.

    Returns True on PASS, False on FAIL (and prints diagnostics)."""
    print(f"\n[run] {label}: {NUM_TOKENS} tokens, "
          f"global={layer.global_num_experts} experts, "
          f"E_local={layer.gate_weight_iq2_xxs.shape[0]}, topk={TOPK}, "
          f"hidden={HIDDEN}, intermediate={INTERMEDIATE}, "
          f"expert_map={'set' if layer.expert_map is not None else 'None'}")

    got = iq2_forward(self_obj, layer, x, tk_w, tk_ids)
    if got.shape != (NUM_TOKENS, HIDDEN):
        print(f"FAIL: output shape {got.shape} != ({NUM_TOKENS}, {HIDDEN})")
        return False
    if got.dtype != torch.bfloat16:
        print(f"FAIL: output dtype {got.dtype} != bf16")
        return False
    if not torch.isfinite(got).all():
        print("FAIL: non-finite output from _iq2_forward")
        return False

    ref = _reference_forward(layer, x, tk_w, tk_ids)
    got_np = got.float().cpu().numpy()

    abs_err = np.abs(got_np - ref)
    max_abs = float(abs_err.max())
    ref_mag = np.maximum(np.abs(ref), 1e-3)
    rel = abs_err / ref_mag
    mean_rel = float(rel.mean())
    cos = _cosine(got_np, ref)
    ref_finite = bool(np.all(np.isfinite(ref)))

    print(f"[cmp] cosine={cos:.5f}  finite(got,ref)=(True,{ref_finite})  "
          f"max_abs={max_abs:.4f}  mean_rel={mean_rel:.4f}")

    COS_OK = 0.95
    if cos < COS_OK:
        print(f"FAIL: cosine {cos:.5f} < {COS_OK} -- direction mismatch; "
              "likely a wiring bug (transpose/combine/swap).")
        return False
    if not ref_finite:
        print("FAIL: reference output non-finite (adjust d_scale).")
        return False
    MEAN_REL_OK = 0.10
    if mean_rel > MEAN_REL_OK:
        print(f"FAIL: mean_rel {mean_rel:.4f} > {MEAN_REL_OK} -- compounding "
              "error larger than expected; investigate.")
        return False

    print(f"[PASS] {label}: wiring OK "
          f"(cosine={cos:.5f} >= {COS_OK}, mean_rel={mean_rel:.4f} "
          f"< {MEAN_REL_OK}).")
    return True


def main() -> int:
    if not torch.cuda.is_available():
        print("FAIL: CUDA required (Triton GEMMs are CUDA-only). "
              "Free GPUs first: docker stop moet")
        return 2

    torch.manual_seed(SEED)
    rng = torch.Generator(device="cpu").manual_seed(SEED + 1)
    gate_w, up_w, down_w, x, tk_w, tk_ids = _build_inputs(rng)
    self_obj = _FakeSelf()

    results = []

    # Case A: all experts local (TP=1 or experts-replicated). The default
    # production-IQ2 path for a single-GPU serve.
    layer_a = _FakeLayer(gate_w, up_w, down_w, expert_map=None)
    results.append(_run_case(
        "Case A (no expert_map, all-local)", layer_a, x, tk_w, tk_ids,
        self_obj))

    # Case B: expert-parallel rank-0 of a 2-way split. global=4, local=2
    # (experts 0,1 on this rank; 2,3 on the peer). expert_map[g] = local id
    # or -1. Dispatches to globals 2,3 contribute zero here; the runner's
    # all-reduce fills them from the peer. THIS IS THE TP=2 BOOT-PATH.
    gate_w_loc = gate_w[:2].clone()
    up_w_loc = up_w[:2].clone()
    down_w_loc = down_w[:2].clone()
    expert_map = torch.tensor([0, 1, -1, -1], dtype=torch.int32).cuda()
    layer_b = _FakeLayer(
        gate_w_loc, up_w_loc, down_w_loc, expert_map=expert_map,
        global_num_experts=NUM_EXPERTS)
    results.append(_run_case(
        "Case B (EP rank-0, expert_map=[0,1,-1,-1])",
        layer_b, x, tk_w, tk_ids, self_obj))

    print()
    if all(results):
        print(f"[PASS] All {len(results)} IQ2 MoE forward cases passed. "
              "Step 5a offline validation green: _iq2_forward produces "
              "correct, finite routed-expert output for both the all-local "
              "and expert-parallel dispatch paths. Ready for the boot test "
              "(Step 5b).")
        return 0
    print(f"[FAIL] {sum(1 for r in results if not r)}/{len(results)} case(s) "
          "failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
