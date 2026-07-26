#!/usr/bin/env python3
"""Scan ALL 43 layers x experts 1/34/35 for anomalous d/dmin (Inf/NaN/huge).
Also reports the per-block dequant max to find any block whose dequant
exceeds sane bounds (which would indicate a bad block vs a kernel bug)."""
from __future__ import annotations

import os
import sys
import mmap
import struct
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REF = os.path.normpath(os.path.join(
    _HERE, "..", "overlay", "vllm", "vllm", "model_executor", "layers",
    "quantization", "utils"))
sys.path.insert(0, _REF)

from iq2_xxs_ref import (  # noqa: E402
    QK_K, Q2_K_BLOCK_BYTES, dequant_q2_k_block,
    parse_gguf_header, GGUF_TYPE_Q2_K,
)

GGUF = "/root/antirez/ds4/ds4flash.gguf"
N_LAYERS = 43
HIDDEN = 4096
INTERMEDIATE = 2048
ROW_BYTES = (INTERMEDIATE // QK_K) * Q2_K_BLOCK_BYTES  # 672
EXPERT_BYTES = HIDDEN * ROW_BYTES  # 2,752,512
EXPERTS = [0, 1, 34, 35]


def main():
    with open(GGUF, "rb") as fh:
        with mmap.mmap(fh.fileno(), 0, prot=mmap.PROT_READ) as mm:
            buf = bytes(mm[:64 * 1024 * 1024])
            tensors, data_off, _a = parse_gguf_header(buf)

    print(f"Scanning {N_LAYERS} layers x experts {EXPERTS}...")
    worst_d = 0.0
    worst_layer = -1
    worst_expt = -1
    n_anomalies = 0
    for L in range(N_LAYERS):
        name = f"blk.{L}.ffn_down_exps.weight"
        cands = [t for t in tensors if t.name == name and t.gguf_type == GGUF_TYPE_Q2_K]
        if not cands:
            continue
        t = cands[0]
        for expt in EXPERTS:
            abs_off = data_off + t.offset + expt * EXPERT_BYTES
            with open(GGUF, "rb") as bf:
                bf.seek(abs_off)
                raw = bf.read(EXPERT_BYTES)
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(-1, Q2_K_BLOCK_BYTES)
            d_u16 = arr[:, 80].astype(np.uint16) | (arr[:, 81].astype(np.uint16) << 8)
            dm_u16 = arr[:, 82].astype(np.uint16) | (arr[:, 83].astype(np.uint16) << 8)
            d_f16 = np.frombuffer(d_u16.tobytes(), dtype=np.float16).astype(np.float32)
            dm_f16 = np.frombuffer(dm_u16.tobytes(), dtype=np.float16).astype(np.float32)
            d_bad = int(np.sum(~np.isfinite(d_f16)))
            dm_bad = int(np.sum(~np.isfinite(dm_f16)))
            d_max = float(d_f16.max()) if np.all(np.isfinite(d_f16)) else float("inf")
            if d_bad or dm_bad or d_max > 0.1:
                print(f"  L{L:2d} expert {expt:3d}: d_bad={d_bad} dm_bad={dm_bad} "
                      f"d_max={d_max:.4g} -- ANOMALOUS")
                n_anomalies += 1
            if d_max > worst_d:
                worst_d = d_max
                worst_layer = L
                worst_expt = expt

    print(f"\nDone. anomalies={n_anomalies}")
    print(f"worst d_max={worst_d:.6g} at L{worst_layer} expert {worst_expt}")
    # Spot-check: dequant the worst expert's blocks, find max dequant value.
    name = f"blk.{worst_layer}.ffn_down_exps.weight"
    t = [x for x in tensors if x.name == name and x.gguf_type == GGUF_TYPE_Q2_K][0]
    abs_off = data_off + t.offset + worst_expt * EXPERT_BYTES
    with open(GGUF, "rb") as bf:
        bf.seek(abs_off)
        raw = bf.read(EXPERT_BYTES)
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(-1, Q2_K_BLOCK_BYTES)
    # Sample 256 blocks spread across the expert, dequant, find max.
    n_blocks = arr.shape[0]
    idx = np.linspace(0, n_blocks - 1, 256).astype(int)
    max_deq = 0.0
    for i in idx:
        v = dequant_q2_k_block(bytes(arr[i].tobytes()))
        m = float(np.max(np.abs(v)))
        if not np.isfinite(m):
            print(f"  NON-FINITE dequant at block {i}!")
        elif m > max_deq:
            max_deq = m
    print(f"sampled max dequant |y| = {max_deq:.6g} (256 blocks)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
