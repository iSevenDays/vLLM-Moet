#!/usr/bin/env python3
"""Compare GGUF bytes vs safetensors bytes for Q2_K down experts, and run
the kernel on the safetensors bytes to see if production bytes differ."""
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
    QK_K, Q2_K_BLOCK_BYTES, dequant_q2_k,
    parse_gguf_header, find_tensors_by_prefix, GGUF_TYPE_Q2_K,
)
from q2_k_mm_triton import q2_k_mm  # noqa: E402

GGUF = "/root/antirez/ds4/ds4flash.gguf"
ST = "/root/models/DeepSeek-V4-Flash-IQ2/dsv4_iq2.safetensors"
LAYER = 0
N = 4096
K = 2048
BLOCKS_PER_ROW = K // QK_K
ROW_BYTES = BLOCKS_PER_ROW * Q2_K_BLOCK_BYTES
EXPERT_BYTES = N * ROW_BYTES
M = 4


def gguf_expert_bytes(expert: int) -> bytes:
    with open(GGUF, "rb") as fh:
        with mmap.mmap(fh.fileno(), 0, prot=mmap.PROT_READ) as mm:
            buf = bytes(mm[:64 * 1024 * 1024])
            tensors, data_off, _a = parse_gguf_header(buf)
    name = f"blk.{LAYER}.ffn_down_exps.weight"
    cands = [t for t in tensors if t.name == name and t.gguf_type == GGUF_TYPE_Q2_K]
    t = cands[0]
    abs_off = data_off + t.offset + expert * EXPERT_BYTES
    with open(GGUF, "rb") as bf:
        bf.seek(abs_off)
        return bf.read(EXPERT_BYTES)


def main():
    from safetensors import safe_open

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[cmp] device={device}")

    st_name = f"model.layers.{LAYER}.mlp.experts.down_proj.weight_q2_k"
    with safe_open(ST, framework="numpy", device="cpu") as sf:
        full = sf.get_tensor(st_name)  # [256, 4096, 672] uint8 memmap
        print(f"[cmp] safetensors {st_name} shape={full.shape} dtype={full.dtype}")
        # Sanity: total bytes
        print(f"[cmp] safetensors total bytes = {full.nbytes:,d} "
              f"(expect {256 * EXPERT_BYTES:,d})")

    rng = np.random.default_rng(7)
    a_np = (rng.standard_normal((M, K)) * 0.3).astype(np.float32)
    a_bf16 = torch.from_numpy(a_np).to(torch.bfloat16)

    for expt in [0, 1, 34, 35, 129, 162, 163]:
        gguf_b = gguf_expert_bytes(expt)
        st_b = bytes(full[expt].tobytes())
        match = gguf_b == st_b
        n_diff = sum(1 for a, b in zip(gguf_b, st_b) if a != b) if not match else 0
        print(f"\n[expert {expt}] GGUF==ST bytes: {match}  "
              f"({n_diff} differing bytes)" if not match
              else f"\n[expert {expt}] GGUF==ST bytes: {match}")

        # Inspect d/dmin of the safetensors bytes
        arr = np.frombuffer(st_b, dtype=np.uint8).reshape(
            -1, Q2_K_BLOCK_BYTES)
        d_u16 = arr[:, 80].astype(np.uint16) | (arr[:, 81].astype(np.uint16) << 8)
        dm_u16 = arr[:, 82].astype(np.uint16) | (arr[:, 83].astype(np.uint16) << 8)
        d_f16 = np.frombuffer(d_u16.tobytes(), dtype=np.float16).astype(np.float32)
        dm_f16 = np.frombuffer(dm_u16.tobytes(), dtype=np.float16).astype(np.float32)
        d_inf = int(np.sum(np.isinf(d_f16))) + int(np.sum(np.isnan(d_f16)))
        dm_inf = int(np.sum(np.isinf(dm_f16))) + int(np.sum(np.isnan(dm_f16)))
        print(f"  ST d: max={float(d_f16.max()):.4g} inf/nan={d_inf}  "
              f"dmin: max={float(dm_f16.max()):.4g} inf/nan={dm_inf}")

        # dequant the ST bytes
        w_fp32 = dequant_q2_k(st_b, (N, K))
        finite = bool(np.all(np.isfinite(w_fp32)))
        print(f"  ST dequant fp32 finite={finite} max|x|="
              f"{float(np.max(np.abs(w_fp32))):.4g}")

        # kernel on ST bytes
        w_t = torch.from_numpy(np.frombuffer(st_b, dtype=np.uint8).copy()).to(device)
        c = q2_k_mm(a_bf16.to(device), w_t, (N, K))
        got = c.float().cpu().numpy()
        kfinite = bool(np.all(np.isfinite(got)))
        print(f"  ST kernel finite={kfinite} "
              f"nan={int(np.sum(np.isnan(got)))} "
              f"inf={int(np.sum(np.isinf(got)))}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
