#!/usr/bin/env python3
"""Test q2_k_mm with SATURATING mid activations to find the overflow point.

If the down GEMM's fp32 accumulator exceeds 65504 (bf16 max), the final
acc.to(bf16) cast produces Inf -- which matches the production 'non-finite
expert_out' symptom even though gate/up/mid are all bf16-finite."""
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
    QK_K, Q2_K_BLOCK_BYTES, parse_gguf_header, GGUF_TYPE_Q2_K,
)
from q2_k_mm_triton import q2_k_mm  # noqa: E402

GGUF = "/root/antirez/ds4/ds4flash.gguf"
N = 4096
K = 2048
ROW_BYTES = (K // QK_K) * Q2_K_BLOCK_BYTES
EXPERT_BYTES = N * ROW_BYTES
M = 4


def load_expert(expert: int) -> bytes:
    with open(GGUF, "rb") as fh:
        with mmap.mmap(fh.fileno(), 0, prot=mmap.PROT_READ) as mm:
            buf = bytes(mm[:64 * 1024 * 1024])
            tensors, data_off, _a = parse_gguf_header(buf)
    t = [x for x in tensors if x.name == "blk.0.ffn_down_exps.weight"
         and x.gguf_type == GGUF_TYPE_Q2_K][0]
    with open(GGUF, "rb") as bf:
        bf.seek(data_off + t.offset + expert * EXPERT_BYTES)
        return bf.read(EXPERT_BYTES)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for expt in [0, 1, 34, 35]:
        raw = load_expert(expt)
        w_t = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy()).to(device)
        print(f"\n=== expert {expt} ===")
        # Sweep the activation scale. mid in production = SiLU(gate)*up, bf16.
        # Find the scale at which the down output overflows bf16.
        for scale in [1.0, 10.0, 100.0, 1000.0, 10000.0, 65504.0]:
            # All-ones-sign activation at this scale (worst-case alignment).
            mid = torch.full((M, K), scale, dtype=torch.bfloat16, device=device)
            out = q2_k_mm(mid, w_t, (N, K))
            finite = bool(torch.isfinite(out).all())
            nan_n = int(torch.isnan(out).sum())
            inf_n = int(torch.isinf(out).sum())
            mx = float(out.abs().max()) if finite else float("nan")
            print(f"  mid scale={scale:>9.1f}: finite={finite} nan={nan_n} "
                  f"inf={inf_n} max|out|={mx:.4g}")

        # Also: random activation at large scale (more realistic).
        for scale in [100.0, 1000.0, 10000.0]:
            g = torch.Generator().manual_seed(99)
            mid = (torch.randn(M, K, generator=g) * scale).to(torch.bfloat16).to(device)
            out = q2_k_mm(mid, w_t, (N, K))
            finite = bool(torch.isfinite(out).all())
            nan_n = int(torch.isnan(out).sum())
            inf_n = int(torch.isinf(out).sum())
            mx = float(out.abs().max()) if finite else float("nan")
            print(f"  mid randn*{scale:>6.0f}: finite={finite} nan={nan_n} "
                  f"inf={inf_n} max|out|={mx:.4g}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
