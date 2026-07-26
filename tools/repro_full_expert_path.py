#!/usr/bin/env python3
"""Reproduce by simulating the FULL per-expert path (gate/up/down) on real
GGUF weights, matching production _iq2_forward exactly.  This checks whether
mid = SiLU(gate)*up saturates and the down GEMM's final acc.to(bf16) overflows."""
from __future__ import annotations

import os
import sys
import mmap
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REF = os.path.normpath(os.path.join(
    _HERE, "..", "overlay", "vllm", "vllm", "model_executor", "layers",
    "quantization", "utils"))
sys.path.insert(0, _REF)

from iq2_xxs_ref import (  # noqa: E402
    QK_K, IQ2_XXS_BLOCK_BYTES, Q2_K_BLOCK_BYTES,
    parse_gguf_header, GGUF_TYPE_IQ2_XXS, GGUF_TYPE_Q2_K,
)
from iq2_xxs_mm_triton import iq2_xxs_mm  # noqa: E402
from q2_k_mm_triton import q2_k_mm  # noqa: E402

GGUF = "/root/antirez/ds4/ds4flash.gguf"
LAYER = 0
HIDDEN = 4096
INTERMEDIATE = 2048
N_EXPERTS = 256
GATE_UP_ROW_BYTES = (HIDDEN // QK_K) * IQ2_XXS_BLOCK_BYTES      # 1056
DOWN_ROW_BYTES = (INTERMEDIATE // QK_K) * Q2_K_BLOCK_BYTES      # 672
GATE_UP_BYTES = N_EXPERTS * INTERMEDIATE * GATE_UP_ROW_BYTES
DOWN_BYTES = N_EXPERTS * HIDDEN * DOWN_ROW_BYTES


def load_expert_tensor(name: str, expert: int, nbytes_total: int,
                       per_expert: int, gguf_type: int) -> bytes:
    with open(GGUF, "rb") as fh:
        with mmap.mmap(fh.fileno(), 0, prot=mmap.PROT_READ) as mm:
            buf = bytes(mm[:64 * 1024 * 1024])
            tensors, data_off, _a = parse_gguf_header(buf)
    cands = [t for t in tensors if t.name == name and t.gguf_type == gguf_type]
    t = cands[0]
    abs_off = data_off + t.offset + expert * per_expert
    with open(GGUF, "rb") as bf:
        bf.seek(abs_off)
        return bf.read(per_expert)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sim] device={device}")

    from iq2_xxs_mm_triton import _get_luts
    grid, ksigns = _get_luts(device)

    # Production: x is the hidden activation [num_tokens, hidden=4096] bf16.
    # Use a realistic-scale activation (unit-ish, like a real residual stream).
    M = 4
    rng = torch.Generator().manual_seed(2026)
    x = (torch.randn(M, HIDDEN, generator=rng) * 1.0).to(torch.bfloat16).to(device)
    print(f"[sim] M={M} hidden={HIDDEN} intermediate={INTERMEDIATE}")
    print(f"[sim] x scale: mean|x|={float(x.abs().mean()):.4g} "
          f"max|x|={float(x.abs().max()):.4g}")

    gate_shape = (INTERMEDIATE, HIDDEN)   # (N, K)
    up_shape = (INTERMEDIATE, HIDDEN)
    down_shape = (HIDDEN, INTERMEDIATE)

    for expt in [0, 1, 34, 35]:
        gate_b = load_expert_tensor(f"blk.{LAYER}.ffn_gate_exps.weight", expt,
                                    GATE_UP_BYTES,
                                    INTERMEDIATE * GATE_UP_ROW_BYTES,
                                    GGUF_TYPE_IQ2_XXS)
        up_b = load_expert_tensor(f"blk.{LAYER}.ffn_up_exps.weight", expt,
                                  GATE_UP_BYTES,
                                  INTERMEDIATE * GATE_UP_ROW_BYTES,
                                  GGUF_TYPE_IQ2_XXS)
        down_b = load_expert_tensor(f"blk.{LAYER}.ffn_down_exps.weight", expt,
                                    DOWN_BYTES,
                                    HIDDEN * DOWN_ROW_BYTES,
                                    GGUF_TYPE_Q2_K)

        gate_w = torch.from_numpy(np.frombuffer(gate_b, dtype=np.uint8).copy()).to(device)
        up_w = torch.from_numpy(np.frombuffer(up_b, dtype=np.uint8).copy()).to(device)
        down_w = torch.from_numpy(np.frombuffer(down_b, dtype=np.uint8).copy()).to(device)

        gate_out = iq2_xxs_mm(x, gate_w, gate_shape, grid, ksigns)
        up_out = iq2_xxs_mm(x, up_w, up_shape, grid, ksigns)
        mid = torch.nn.functional.silu(gate_out) * up_out
        out = q2_k_mm(mid, down_w, down_shape)

        def _stats(t, lbl):
            f = bool(torch.isfinite(t).all())
            return (f"{lbl}: finite={f} "
                    f"max|x|={float(t.abs().max()):.4g} "
                    f"mean|x|={float(t.abs().mean()):.4g} "
                    f"dtype={t.dtype}")

        print(f"\n=== expert {expt} ===")
        print("  " + _stats(gate_out, "gate_out"))
        print("  " + _stats(up_out, "  up_out"))
        print("  " + _stats(mid, "     mid"))
        print("  " + _stats(out, "    out"))

        # Also compare: down GEMM done in PURE fp32 (no bf16 cast on output).
        # This isolates whether the overflow is in the final bf16 cast.
        from iq2_xxs_ref import dequant_q2_k
        w_fp32 = dequant_q2_k(down_b, down_shape)
        w_fp32_t = torch.from_numpy(np.ascontiguousarray(w_fp32)).to(device)
        out_fp32 = (mid.float() @ w_fp32_t.t())  # pure fp32
        print("  " + _stats(out_fp32, " out_fp32 (pure fp32 mm, no bf16 cast)"))

        # And: down GEMM with mid clamped to a smaller range.
        mid_clamped = mid.clamp(-1.5, 1.5)
        out_clamped = q2_k_mm(mid_clamped, down_w, down_shape)
        print("  " + _stats(out_clamped, " out_clamped (|mid|<=1.5)"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
