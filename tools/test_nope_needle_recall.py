#!/usr/bin/env python3
"""How much exact-value recall does the NoPE encoding cost, vs attended-key count?

Cheap CPU-only harness for the long-context needle regression. No model, no
GPU, no server: it isolates the ONE thing the packed KV encoding controls --
how faithfully a single attended row's value survives a round trip through the
cache -- and how that degrades as the number of competing keys grows.

Why this is the right question. End-to-end, the model always recovers the
needle's WORD but loses its DIGITS, and with added redundancy the digits come
back as NEAR-MISSES (1606 for 1605). That is a precision signature: the needle
IS attended, but the value read back is noisy. In MLA, K and V are the SAME
512-dim cache row, so attention output ~= sum_i softmax_i * row_i. Two things
can spoil exact recall:
  1. quantization of the needle's OWN row (a floor on readback precision), and
  2. logit noise letting softmax mass leak to distractors, whose quantization
     noise then contaminates the output -- this is what grows with N.

Compares stock FP8-E4M3 (~3 mantissa bits) against VLLM_DSV4_KV_INT8=1 signed
INT8 (~7 bits) at IDENTICAL byte cost, over the same UE8M0 tile scale.

  docker run --rm --entrypoint python3 \
    -v $PWD/tools/test_nope_needle_recall.py:/opt/t.py:ro \
    vllm-moet-sm89:v0251 /opt/t.py
"""
import sys

import torch

from vllm.v1.attention.ops.triton_sparse_mla_dsv4 import (
    dequant_dsv4_packed_cache,
    pack_dsv4_reference_cache,
)

D, D_NOPE = 512, 448
PBS = 64
SOFTMAX_SCALE = D ** -0.5


def roundtrip(rows: torch.Tensor, int8: bool) -> torch.Tensor:
    """rows [P*PBS, 512] f32 -> packed bytes -> decoded f32 (same shape)."""
    packed = pack_dsv4_reference_cache(rows, pbs=PBS, int8=int8)
    return dequant_dsv4_packed_cache(packed, pbs=PBS, int8=int8).float()


def build(n_keys: int, gen: torch.Generator):
    """n_keys distractors + 1 needle; query aligned to the needle.

    The needle is made the top match by a MODEST margin (as in a real
    retrieval, not a trivial one), so ranking is sensitive to logit noise.
    """
    n = ((n_keys + PBS - 1) // PBS) * PBS          # pad to whole pages
    rows = torch.randn(n, D, generator=gen) * 0.5
    needle_idx = n // 2                            # mid-context
    needle = torch.randn(D, generator=gen) * 0.5
    rows[needle_idx] = needle
    # query points along the needle so it wins by ~a few sigma, not by 100x
    q = needle.clone()
    q += torch.randn(D, generator=gen) * 0.5 * 0.9
    return rows, q, needle_idx, n


def attend(rows: torch.Tensor, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logits = (rows @ q) * SOFTMAX_SCALE
    w = torch.softmax(logits, dim=0)
    return w @ rows, w


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def main() -> int:
    print(f"{'N_keys':>8} {'encoding':>6} {'row_rel':>9} {'needle_rank':>12} "
          f"{'needle_w':>9} {'w_exact':>9} {'out_rel':>9} {'val_rel':>9}")
    print("-" * 84)

    worst = {"e4m3": 0.0, "int8": 0.0}
    for n_keys in (512, 2048, 8192, 32768):
        gen = torch.Generator().manual_seed(20260801 + n_keys)
        rows, q, ni, n = build(n_keys, gen)
        out_exact, w_exact = attend(rows, q)
        rank_exact = int((( rows @ q) > (rows[ni] @ q)).sum())

        for label, int8 in (("e4m3", False), ("int8", True)):
            deq = roundtrip(rows, int8)
            out_q, w_q = attend(deq, q)
            # rank of the needle under the quantized keys (0 == still argmax)
            rank = int(((deq @ q) > (deq[ni] @ q)).sum())
            # val_rel: how precisely the needle's OWN value is read back --
            # the ceiling on exact-token (digit) recall.
            val_rel = rel(deq[ni], rows[ni])
            out_rel = rel(out_q, out_exact)
            row_rel = rel(deq, rows)
            worst[label] = max(worst[label], out_rel)
            print(f"{n_keys:>8} {label:>6} {row_rel:>9.3e} "
                  f"{rank:>5} (ex {rank_exact:>3}) {w_q[ni]:>9.4f} "
                  f"{w_exact[ni]:>9.4f} {out_rel:>9.3e} {val_rel:>9.3e}")
        print()

    print(f"worst output rel-err: E4M3={worst['e4m3']:.3e}  "
          f"INT8={worst['int8']:.3e}")
    if worst["e4m3"] > 0:
        print(f"INT8 reduces attention-output error {worst['e4m3']/max(worst['int8'],1e-12):.1f}x "
              f"at identical bytes/token")
    return 0


if __name__ == "__main__":
    sys.exit(main())
