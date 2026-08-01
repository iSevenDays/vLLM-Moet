#!/usr/bin/env python3
"""Does the decode indexer's TF32 score flip borderline top-512 membership?

The sm_89-specific suspect from docs/dsv4-sm89-longcontext/runlogs/dig_decode_path.md:
the capture-safe Triton port of DeepGEMM's ``fp8_paged_mqa_logits`` scores every
cached context token against the decode query with a TF32 dot
(``tl.dot(q, tl.trans(k), allow_tf32=True)`` at
``overlay/vllm/vllm/v1/attention/ops/triton_paged_mqa_logits_dsv4.py:185``),
while the fp32 torch reference (and the prefill path) score in pure fp32. TF32
shrinks the mantissa to 10 bits; the boot self-test reports ``worst_row_rel =
4.691e-4`` -- exactly the TF32 rounding band. The hypothesis: on a token whose
fp32 score lands just above the ``index_topk = 512`` cut (rank ~508-516), the
TF32 perturbation can push it BELOW the cut, so its KV is never loaded during
the digit-emitting decode steps and the digit silently drops.

This harness constructs that borderline scenario DIRECTLY and measures the flip
rate. It builds a paged FP8 indexer cache with > 512 scored candidates (shapes
mirrored from the kernel's ``_self_test_case``), places a TARGET token's fp32
score at each rank in [505, 520] by scaling its key (exact, because ``score`` is
linear in the key scale for non-negative per-head weights), then runs BOTH
``paged_mqa_logits_dsv4_triton`` (TF32) and ``paged_mqa_logits_torch_ref`` (fp32
oracle) on the IDENTICAL packed cache + Q + weights and asks: does the TARGET's
top-512 membership AGREE? A disagreement is a TF32 flip. A second pass samples
natural random scenarios (signed weights, no placement) and reports the observed
flip rate among candidates whose fp32 rank naturally falls in the borderline
band. The TF32 score residual is also reported so the ~4.7e-4 noise can be
confirmed on this silicon.

Verdict: PASS = no borderline membership flips (TF32 rounding never moves a
rank-505..520 token across the 512-cut on this GPU); FLIP-DETECTED = TF32 can
change which tokens the indexer selects (the suspected decode-selection
mechanism). Either way the residual + per-rank table are printed.

Run inside the serving image (the overlay kernel is installed there):

  docker run --rm --gpus '"device=0"' --entrypoint python3 \
    -v $PWD/tools/test_paged_mqa_tf32_flip.py:/opt/t.py:ro \
    vllm-moet-sm89:v0251 /opt/t.py

Free-GPU note: a loaded server leaves only ~1 GiB on each card. This harness
needs < 50 MiB (768-token cache, one draft row), so it coexists with a live
server on the same GPU; if you prefer isolation, stop the server first or use a
free device index.

Falsifier cross-check: the production one-boot falsifier is
``VLLM_DSV4_PAGED_MQA_LOGITS_FORCE_TORCH=1`` (decode logits fall back to the
fp32 reference). This harness tests the SAME mechanism at the op level without
a server boot.
"""
from __future__ import annotations

import os
import sys

import torch

# Prefer the installed (in-container) vllm copy -- that is the code the server
# actually runs. Fall back to the in-tree overlay for bare-metal runs.
try:
    from vllm.v1.attention.ops.triton_paged_mqa_logits_dsv4 import (
        pack_fp8_indexer_cache_reference as pack_cache,
        paged_mqa_logits_dsv4_triton as run_triton,
        paged_mqa_logits_torch_ref as run_ref,
    )
except ImportError:  # in-tree: run from a checkout with the overlay beside tools/
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _OVL = os.path.abspath(os.path.join(_HERE, "..", "overlay"))
    if os.path.isdir(_OVL):
        sys.path.insert(0, _OVL)
    from vllm.v1.attention.ops.triton_paged_mqa_logits_dsv4 import (  # noqa: E402
        pack_fp8_indexer_cache_reference as pack_cache,
        paged_mqa_logits_dsv4_triton as run_triton,
        paged_mqa_logits_torch_ref as run_ref,
    )

# --- Geometry (mirrors _self_test_case at :371-401 of the kernel module) -----
BLOCK_SIZE = 64            # the self-test block; precision is per-block so this
                           # is representative (dig notes production uses 256)
NUM_BLOCKS = 12            # -> 768 candidate tokens (> 512, so top-512 excludes)
NUM_TOKENS = BLOCK_SIZE * NUM_BLOCKS
H, D = 32, 128
BSZ, NEXT_N = 1, 1
MAX_MODEL_LEN = NUM_TOKENS
TOPK = 512                 # the DSv4 lightning-indexer index_topk
TARGET = 0                 # the candidate whose fp32 rank we dial

# Ranks to probe (1-indexed: rank 1 = highest score). 512 = last IN, 513 = 1st OUT.
RANK_SWEEP = range(505, 521)
NATURAL_BORDER_LO, NATURAL_BORDER_HI = 505, 520


def _device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "needs CUDA: the Triton TF32 port is the code under test")
    return torch.device("cuda", torch.cuda.current_device())


def _base_inputs(seed: int, signed_weights: bool):
    """Random FP8 query, non-negative (or signed) per-head weights, random fp32
    keys for every candidate token. Generator-seeded for reproducibility."""
    g = torch.Generator().manual_seed(seed)
    kv_rows = torch.randn(NUM_TOKENS, D, generator=g) * 2.0
    q = (torch.randn(BSZ, NEXT_N, H, D, generator=g)
         ).clamp(-448, 448).to(torch.float8_e4m3fn)
    w = torch.randn(BSZ * NEXT_N, H, generator=g)
    if not signed_weights:
        w = w.abs()           # non-negative -> scores >= 0 -> exact rank dialing
    return kv_rows, q, w


def _ctx_bt() -> tuple[torch.Tensor, torch.Tensor]:
    """Identity block table (block i -> physical block i); every token scored."""
    ctx = torch.tensor([[NUM_TOKENS]], dtype=torch.int32)
    max_blocks = -(-MAX_MODEL_LEN // BLOCK_SIZE)
    bt = (torch.arange(max_blocks, dtype=torch.int32)
          .unsqueeze(0).expand(BSZ, max_blocks).contiguous())
    return ctx, bt


def _scores(kv_rows: torch.Tensor, q: torch.Tensor, w: torch.Tensor,
            ctx: torch.Tensor, bt: torch.Tensor, fn, dev: torch.device
            ) -> torch.Tensor:
    """Pack kv_rows into the FP8 indexer cache and score every candidate with
    ``fn``. Returns logits[0] over [max_model_len] (-inf outside context)."""
    kv = pack_cache(kv_rows.to(dev), BLOCK_SIZE)
    logits = fn(q_values=q.to(dev), kv_cache=kv, weights=w.to(dev),
                context_lens=ctx.to(dev), block_tables=bt.to(dev),
                max_model_len=MAX_MODEL_LEN)
    return logits[0]


def _topk_set(scores: torch.Tensor, k: int) -> set[int]:
    fin = scores[scores.isfinite()]
    kk = min(k, int(fin.numel()))
    return set(scores.topk(kk).indices.tolist())


def _place_target_at_rank(kv_rows: torch.Tensor, q: torch.Tensor, w: torch.Tensor,
                          ctx: torch.Tensor, bt: torch.Tensor, dev: torch.device,
                          target_rank: int) -> torch.Tensor:
    """Return a copy of kv_rows with the TARGET token's fp32-ref score placed at
    ``target_rank`` (1-indexed among all NUM_TOKENS candidates).

    Score = sum_h relu(q_h . k) * w_h is homogeneous in the key for a FIXED relu
    sign pattern; with non-negative weights every score is >= 0, so scaling a
    high-positive-score base key reaches any target score >= 0 in one step. The
    other candidates' scores are untouched (only kv_rows[TARGET] changes), so
    the rank is set by the (precomputed) ordering of the others.
    """
    s_all = _scores(kv_rows, q, w, ctx, bt, run_ref, dev)
    others = s_all.clone()
    others[TARGET] = float("-inf")
    oth_desc = torch.sort(others[others.isfinite()], descending=True).values
    # 0-indexed oth_desc: oth_desc[0] = largest. For target at 1-indexed rank R,
    # exactly R-1 others are above it -> score in (oth_desc[R-1], oth_desc[R-2]).
    lo = float(oth_desc[target_rank - 1].item())
    hi = float(oth_desc[target_rank - 2].item()) if target_rank >= 2 else float("inf")
    want_v = 0.5 * (lo + hi)

    # Base key guaranteed to have a positive score: the sum of the dequantized
    # query heads (q_h . sum_j q_j > 0 for random Q; relu active on every head).
    qf = q[0, 0].float().to(dev)            # [H, D]
    base_key = qf.sum(0)                    # [D]
    kv2 = kv_rows.clone()
    kv2[TARGET] = base_key.cpu()
    s0 = _scores(kv2, q, w, ctx, bt, run_ref, dev)[TARGET].item()
    if s0 <= 0.0:                           # extremely unlikely for random Q
        base_key = -base_key
        kv2[TARGET] = base_key.cpu()
        s0 = _scores(kv2, q, w, ctx, bt, run_ref, dev)[TARGET].item()
    if s0 == 0.0:
        raise RuntimeError("base target key has zero score; rerun with another seed")

    alpha = want_v / s0
    kv2[TARGET] = (alpha * base_key).cpu()
    return kv2


def _membership(kv_rows: torch.Tensor, q: torch.Tensor, w: torch.Tensor,
                ctx: torch.Tensor, bt: torch.Tensor, dev: torch.device, fn):
    scores = _scores(kv_rows, q, w, ctx, bt, fn, dev)
    return TARGET in _topk_set(scores, TOPK), float(scores[TARGET].item())


def controlled_sweep(dev: torch.device, seeds: list[int]) -> dict:
    """Dial the TARGET to each fp32 rank in RANK_SWEEP and compare TF32 vs fp32
    top-512 membership. Returns aggregated stats."""
    print("\n=== Controlled rank placement (non-negative weights, exact dial) ===")
    ctx, bt = _ctx_bt()
    header = f"{'seed':>4} {'rank':>4} {'fp32_in':>7} {'tf32_in':>7} {'flip':>5} {'tf32_residual':>14}"
    print(header)
    print("-" * len(header))
    flips = 0
    trials = 0
    max_resid = 0.0
    for seed in seeds:
        kv_rows, q, w = _base_inputs(seed, signed_weights=False)
        for rank in RANK_SWEEP:
            kv_placed = _place_target_at_rank(kv_rows, q, w, ctx, bt, dev, rank)
            fp32_in, s_ref = _membership(kv_placed, q, w, ctx, bt, dev, run_ref)
            tf32_in, s_tri = _membership(kv_placed, q, w, ctx, bt, dev, run_triton)
            resid = abs(s_tri - s_ref) / max(abs(s_ref), 1.0)
            max_resid = max(max_resid, resid)
            flip = (fp32_in != tf32_in)
            flips += int(flip)
            trials += 1
            print(f"{seed:>4} {rank:>4} {str(fp32_in):>7} {str(tf32_in):>7} "
                  f"{('YES' if flip else 'no'):>5} {resid:>14.3e}")
    rate = flips / trials if trials else 0.0
    print(f"\ncontrolled: flips={flips}/{trials} ({rate:.2%}); "
          f"max |tf32-fp32|/|fp32| residual={max_resid:.3e}")
    return {"flips": flips, "trials": trials, "rate": rate,
            "max_residual": max_resid}


def natural_sweep(dev: torch.device, seeds: list[int]) -> dict:
    """Random scenarios with SIGNED weights (as in the kernel self-test); no
    score placement. For every candidate whose fp32 rank naturally lands in the
    borderline band, compare TF32 vs fp32 top-512 membership."""
    print("\n=== Natural sampling (signed randn weights, no placement) ===")
    ctx, bt = _ctx_bt()
    flips = 0
    border = 0
    resid_samples = []
    per_rank_flips: dict[int, int] = {}
    per_rank_tot: dict[int, int] = {}
    for seed in seeds:
        kv_rows, q, w = _base_inputs(seed, signed_weights=True)
        s_ref = _scores(kv_rows, q, w, ctx, bt, run_ref, dev)
        s_tri = _scores(kv_rows, q, w, ctx, bt, run_triton, dev)
        ref_order = s_ref.argsort(descending=True)
        ref_rank = torch.empty_like(ref_order)
        ref_rank[ref_order] = torch.arange(1, s_ref.numel() + 1,
                                           device=ref_order.device)
        ref_top = _topk_set(s_ref, TOPK)
        tri_top = _topk_set(s_tri, TOPK)
        fin = s_ref.isfinite()
        for pos in range(NUM_TOKENS):
            if not bool(fin[pos]):
                continue
            r = int(ref_rank[pos].item())
            if not (NATURAL_BORDER_LO <= r <= NATURAL_BORDER_HI):
                continue
            border += 1
            per_rank_tot[r] = per_rank_tot.get(r, 0) + 1
            fp32_in = pos in ref_top
            tf32_in = pos in tri_top
            if fp32_in != tf32_in:
                flips += 1
                per_rank_flips[r] = per_rank_flips.get(r, 0) + 1
            sv = float(s_ref[pos].item())
            resid_samples.append(
                abs(float(s_tri[pos].item()) - sv) / max(abs(sv), 1.0))
    print(f"natural: borderline candidates={border} over {len(seeds)} seeds; "
          f"flips={flips} ({flips / border:.2%})" if border else
          f"natural: borderline candidates=0 over {len(seeds)} seeds")
    if per_rank_tot:
        print("  per-rank (fp32_rank: flips/candidates):")
        for r in sorted(per_rank_tot):
            f = per_rank_flips.get(r, 0)
            print(f"    rank {r}: {f}/{per_rank_tot[r]}")
    if resid_samples:
        rs = torch.tensor(resid_samples)
        print(f"  borderline |tf32-fp32|/|fp32| residual: "
              f"max={rs.max().item():.3e} mean={rs.mean().item():.3e}")
    return {"flips": flips, "border": border, "per_rank_flips": per_rank_flips,
            "per_rank_tot": per_rank_tot}


def main() -> int:
    dev = _device()
    cap = torch.cuda.get_device_capability(dev)
    print(f"device: sm_{cap[0]}{cap[1]} {torch.cuda.get_device_name(dev)} "
          f"(torch {torch.__version__})")
    print(f"geometry: {NUM_TOKENS} candidates, topk={TOPK}, "
          f"block_size={BLOCK_SIZE}, H={H}, D={D}")

    seeds_ctrl = [1, 2, 3, 7, 42]
    seeds_nat = list(range(100, 116))  # 16 seeds for borderline coverage

    ctrl = controlled_sweep(dev, seeds_ctrl)
    nat = natural_sweep(dev, seeds_nat)

    print("\n=== VERDICT ===")
    any_flip = (ctrl["flips"] > 0) or (nat["flips"] > 0)
    if any_flip:
        print("FLIP-DETECTED: TF32 decode-indexer scoring can change which "
              "tokens land in the top-512 candidate set.")
        print(f"  controlled: {ctrl['flips']}/{ctrl['trials']} ranks flipped "
              f"(max tf32 residual {ctrl['max_residual']:.3e})")
        print(f"  natural:    {nat['flips']}/{nat['border']} borderline "
              "candidates flipped")
        print("  -> Consistent with the decode-SELECTION mechanism in "
              "dig_decode_path.md. Confirm end-to-end with "
              "VLLM_DSV4_PAGED_MQA_LOGITS_FORCE_TORCH=1; fix = allow_tf32=False "
              "at triton_paged_mqa_logits_dsv4.py:185.")
        return 2
    print("PASS: no borderline top-512 membership flips on this silicon.")
    print(f"  controlled: {ctrl['flips']}/{ctrl['trials']} ranks flipped; "
          f"max |tf32-fp32|/|fp32| residual = {ctrl['max_residual']:.3e}")
    print(f"  natural:    {nat['flips']}/{nat['border']} borderline "
          "candidates flipped")
    print("  -> TF32 perturbs scores by ~4.7e-4 but the rank-505..520 score "
          "gaps exceed the noise band on these random scenarios.")
    print("  NOTE: random-data gaps may be wider than real digit-critical "
          "activations; a flip-free verdict here does NOT exonerate TF32 "
          "everywhere. If the end-to-end needle still fails with "
          "FORCE_TORCH=1, retest with production-derived Q/K/weights.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
