#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline repro for the Q2_K down NaN bug (experts 1/34/35).

Reads the per-expert Q2_K down bytes directly from the antirez GGUF, then:
  1. Inspects d/dmin fp16 ranges (any Inf/NaN/huge?).
  2. Runs the fp32 CPU reference dequant (dequant_q2_k) -> is it finite?
  3. Runs a PURE fp32 reference matmul (no bf16 cast) -> finite?
  4. Runs the Step-4c reference matmul (dequant fp32 -> cast bf16 -> fp32 mm) -> NaN?
  5. Runs the Triton q2_k_mm kernel -> NaN?

This isolates: bad blocks (d=Inf) vs bf16-overflow-in-cast (d large but finite).
"""
from __future__ import annotations

import os
import sys
import struct
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REF = os.path.normpath(os.path.join(
    _HERE, "..", "overlay", "vllm", "vllm", "model_executor", "layers",
    "quantization", "utils"))
sys.path.insert(0, _REF)

from iq2_xxs_ref import (  # noqa: E402
    QK_K, Q2_K_BLOCK_BYTES, dequant_q2_k, dequant_q2_k_block,
    parse_gguf_header, find_tensors_by_prefix, GGUF_TYPE_Q2_K,
)
from q2_k_mm_triton import q2_k_mm  # noqa: E402

GGUF = "/root/antirez/ds4/ds4flash.gguf"
LAYER = 0
# Offending local experts (rank 0): 1, 34, 35. Plus expert 0 as control.
EXPERTS = [0, 1, 34, 35]
# Production shape (Step 5a): down_w = [E_local, N=hidden=4096, rb=672].
# Per expert: N=4096 rows, K=2048 (=intermediate), 8 blocks/row, 672 B/row.
N = 4096
K = 2048
BLOCKS_PER_ROW = K // QK_K  # 8
ROW_BYTES = BLOCKS_PER_ROW * Q2_K_BLOCK_BYTES  # 672
EXPERT_BYTES = N * ROW_BYTES  # 2,752,512
M = 4


def load_expert_bytes(gguf_path: str, layer: int, expert: int) -> bytes:
    import mmap
    with open(gguf_path, "rb") as fh:
        with mmap.mmap(fh.fileno(), 0, prot=mmap.PROT_READ) as mm:
            buf = bytes(mm[:64 * 1024 * 1024])
            tensors, data_off, _align = parse_gguf_header(buf)
    name = f"blk.{layer}.ffn_down_exps.weight"
    cands = [t for t in tensors if t.name == name and t.gguf_type == GGUF_TYPE_Q2_K]
    if not cands:
        raise SystemExit(f"tensor {name} (Q2_K) not found")
    t = cands[0]
    abs_off = data_off + t.offset + expert * EXPERT_BYTES
    with open(gguf_path, "rb") as bf:
        bf.seek(abs_off)
        raw = bf.read(EXPERT_BYTES)
    if len(raw) != EXPERT_BYTES:
        raise SystemExit(f"short read: {len(raw)} != {EXPERT_BYTES}")
    return raw


def inspect_d_dmin(raw: bytes, label: str) -> dict:
    """Scan every Q2_K block's d (fp16) and dmin (fp16)."""
    n_blocks = len(raw) // Q2_K_BLOCK_BYTES
    d_bits = np.frombuffer(raw, dtype=np.uint8)[80::84].astype(np.uint16)
    # Reconstruct the full u16: byte 80 (lo) + byte 81 (hi)<<8.
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(n_blocks, Q2_K_BLOCK_BYTES)
    d_u16 = arr[:, 80].astype(np.uint16) | (arr[:, 81].astype(np.uint16) << 8)
    dm_u16 = arr[:, 82].astype(np.uint16) | (arr[:, 83].astype(np.uint16) << 8)
    d_f16 = np.frombuffer(d_u16.tobytes(), dtype=np.float16).astype(np.float32)
    dm_f16 = np.frombuffer(dm_u16.tobytes(), dtype=np.float16).astype(np.float32)

    def _cls(v):
        return dict(
            n=len(v),
            finite=int(np.sum(np.isfinite(v))),
            inf=int(np.sum(np.isinf(v))),
            nan=int(np.sum(np.isnan(v))),
            sub=int(np.sum((np.abs(v) > 0) & (np.abs(v) < np.finfo(np.float16).tiny))),
            max=float(np.max(v[v == v])) if np.any(np.isfinite(v)) else float("nan"),
            min=float(np.min(v[v == v])) if np.any(np.isfinite(v)) else float("nan"),
        )

    info = {"d": _cls(d_f16), "dmin": _cls(dm_f16), "n_blocks": n_blocks}
    print(f"\n[{label}] {n_blocks} blocks")
    print(f"  d   : finite={info['d']['finite']}/{info['d']['n']} "
          f"inf={info['d']['inf']} nan={info['d']['nan']} "
          f"sub={info['d']['sub']} max={info['d']['max']:.4g} "
          f"min={info['d']['min']:.4g}")
    print(f"  dmin: finite={info['dmin']['finite']}/{info['dmin']['n']} "
          f"inf={info['dmin']['inf']} nan={info['dmin']['nan']} "
          f"sub={info['dmin']['sub']} max={info['dmin']['max']:.4g} "
          f"min={info['dmin']['min']:.4g}")

    # Distribution of d (how many blocks have d > 1455 = the bf16 overflow
    # threshold for d*15*3 > 65504).
    thr = 65504.0 / (15 * 3)
    n_over = int(np.sum((d_f16 > thr) & np.isfinite(d_f16)))
    print(f"  d > {thr:.1f} (bf16-overflow threshold d*15*3>65504): "
          f"{n_over} blocks")
    # Top 5 d values
    top = np.sort(d_f16[np.isfinite(d_f16)])[-5:][::-1]
    print(f"  top-5 d: {top}")
    return info


def run_kernel(raw: bytes, a_bf16: torch.Tensor, device: torch.device, label: str):
    w_bytes = np.frombuffer(raw, dtype=np.uint8).copy()
    w_t = torch.from_numpy(w_bytes).to(device)
    c = q2_k_mm(a_bf16.to(device), w_t, (N, K))
    got = c.float().cpu().numpy()
    finite = bool(np.all(np.isfinite(got)))
    print(f"  [{label}] kernel q2_k_mm  shape={got.shape} "
          f"finite={finite} "
          f"nan={int(np.sum(np.isnan(got)))} inf={int(np.sum(np.isinf(got)))} "
          f"max={float(np.nanmax(np.abs(got))) if finite else float('nan'):.4g}")
    return got, finite


def run_reference(raw: bytes, a_bf16: torch.Tensor, label: str):
    """Three references:
       (a) fp32 dequant only -> finite?
       (b) pure fp32 matmul (no bf16 cast) -> finite?
       (c) Step-4c ref (dequant -> bf16 cast -> fp32 mm) -> finite? (matches kernel's overflow)
    """
    w_fp32 = dequant_q2_k(raw, (N, K))  # numpy fp32 [N, K]
    a_np = a_bf16.cpu().float().numpy()

    # (a)
    finite_deq = bool(np.all(np.isfinite(w_fp32)))
    amax_deq = float(np.max(np.abs(w_fp32))) if finite_deq else float("nan")
    print(f"  [{label}] (a) dequant fp32      finite={finite_deq} "
          f"max|x|={amax_deq:.4g}")

    # (b) pure fp32 matmul
    c_fp32 = a_np @ w_fp32.T
    finite_b = bool(np.all(np.isfinite(c_fp32)))
    print(f"  [{label}] (b) ref matmul fp32    finite={finite_b} "
          f"nan={int(np.sum(np.isnan(c_fp32)))} "
          f"inf={int(np.sum(np.isinf(c_fp32)))} "
          f"max={float(np.nanmax(np.abs(c_fp32))) if finite_b else float('nan'):.4g}")

    # (c) Step-4c ref: cast to bf16 (mirrors the kernel's bf16 cast), then fp32 mm
    w_bf16 = torch.from_numpy(np.ascontiguousarray(w_fp32)).to(torch.bfloat16)
    c_ref = a_bf16.cpu().float() @ w_bf16.float().t()
    c_ref_np = c_ref.numpy()
    finite_c = bool(np.all(np.isfinite(c_ref_np)))
    print(f"  [{label}] (c) ref matmul bf16-cast finite={finite_c} "
          f"nan={int(np.sum(np.isnan(c_ref_np)))} "
          f"inf={int(np.sum(np.isinf(c_ref_np)))} "
          f"max={float(np.nanmax(np.abs(c_ref_np))) if finite_c else float('nan'):.4g}")

    return c_fp32, finite_b, c_ref_np, finite_c


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[repro] device={device}, M={M} N={N} K={K}, experts={EXPERTS}")

    rng = np.random.default_rng(1234)
    a_np = (rng.standard_normal((M, K)) * 0.3).astype(np.float32)
    a_bf16 = torch.from_numpy(a_np).to(torch.bfloat16)

    summary = {}
    for expt in EXPERTS:
        label = f"expert {expt}"
        print(f"\n=== {label} ===")
        raw = load_expert_bytes(GGUF, LAYER, expt)
        info = inspect_d_dmin(raw, label)
        c_fp32, finite_b, c_ref, finite_c = run_reference(raw, a_bf16, label)
        if device.type == "cuda":
            c_kern, finite_k = run_kernel(raw, a_bf16, device, label)
        else:
            c_kern, finite_k = None, None
        summary[label] = dict(
            d_max=info["d"]["max"], d_inf=info["d"]["inf"],
            dequant_finite=finite_b and np.all(np.isfinite(
                dequant_q2_k(raw, (N, K)))),
            ref_fp32_finite=finite_b,
            ref_bf16_finite=finite_c,
            kernel_finite=finite_k,
        )

    print("\n=== SUMMARY ===")
    print(f"{'expert':<10} {'d_max':>12} {'d_inf':>6} {'dequant_finite':>15} "
          f"{'ref_fp32':>10} {'ref_bf16':>10} {'kernel':>10}")
    for lbl, s in summary.items():
        print(f"{lbl:<10} {s['d_max']:>12.4g} {s['d_inf']:>6} "
              f"{str(s['dequant_finite']):>15} "
              f"{str(s['ref_fp32_finite']):>10} "
              f"{str(s['ref_bf16_finite']):>10} "
              f"{str(s['kernel_finite']):>10}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
