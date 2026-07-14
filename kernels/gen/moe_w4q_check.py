#!/usr/bin/env python3
"""Op-level validation of moe_w4q_mm (RADIX-5 split FP4: resident 2-bit
base + 2.5-bit quintal refinement = BIT-EXACT FP4 at 5/8 of the
non-split slot bytes).

Random e2m1 nibbles -> base codes (NIBBLE_TO_CODE) + quintal records
(digit = mag idx, small 0..4 / big 0..2; word = base-5 pack of 4 digits;
80-bit lane record split P8/P2). Reference dequant is TRUE e2m1 — no
merge, unlike the historical moe_w4s_check whose reference bakes the
mag-0 merge in (the split-FP4 zero-loss trap).

Desc ABI (64B/pair): {a, as, base, ref, bs, c, m_rows, pad}.
Env: CUBIN, W4S_CUBIN (optional perf reference), K, N, E, M, RUNS, BENCH.
"""
import ctypes
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/workspace/cubit/tools")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from culaunch import Cuda  # noqa: E402

CUBIN = os.environ["CUBIN"]
W4S_CUBIN = os.environ.get("W4S_CUBIN")
NWARP = int(os.environ.get("NWARP", "8"))
N = int(os.environ.get("N", "4096"))
K = int(os.environ.get("K", "4096"))
E = int(os.environ.get("E", "5"))
M = int(os.environ.get("M", "4"))
RUNS = int(os.environ.get("RUNS", "4"))
BENCH = int(os.environ.get("BENCH", "0"))
torch.manual_seed(int(os.environ.get("SEED", "3")))

E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6] * 2)
E2M1[8:] *= -1
NIBBLE_TO_CODE = torch.tensor([2] * 5 + [3] * 3 + [1] * 5 + [0] * 3,
                              dtype=torch.uint8)


def pack_fragment_major(codes):
    n, k = codes.shape
    c = codes.view(n // 16, 2, 8, k // 64, 2, 2, 4, 4)
    c = c.permute(0, 3, 2, 6, 1, 4, 5, 7).contiguous().view(-1, 4).to(torch.int32)
    return (c[:, 0] | (c[:, 1] << 2) | (c[:, 2] << 4) | (c[:, 3] << 6)).to(torch.uint8)


def pack_quintal_fragment_major(nibs):
    """[N, K] e2m1 nibbles -> radix-5 plane [N*K*5/16] u8:
    per (nb, kb64) block, 32 lanes x 8 B (record bits 0-64), then
    32 lanes x 2 B (bits 64-80). Word w = elems of plane byte w."""
    n, k = nibs.shape
    mag = (nibs & 7).to(torch.int64)
    code = NIBBLE_TO_CODE[nibs.long()]
    big = (code == 0) | (code == 3)
    digit = torch.where(big, mag - 5, mag)
    d = digit.view(n // 16, 2, 8, k // 64, 2, 2, 4, 4)
    d = d.permute(0, 3, 2, 6, 1, 4, 5, 7).contiguous().view(-1, 8, 4)
    words = (d[..., 0] + 5 * d[..., 1] + 25 * d[..., 2] + 125 * d[..., 3])
    # 8x10-bit words -> 5x u16 (little-endian 80-bit stream)
    shorts = torch.zeros(words.shape[0], 5, dtype=torch.int64)
    for w in range(8):
        b = 10 * w
        s, off = b // 16, b % 16
        shorts[:, s] |= (words[:, w] << off)
        if off > 6:
            shorts[:, s + 1] |= words[:, w] >> (16 - off)
    shorts &= 0xFFFF
    by = torch.stack([shorts & 0xFF, shorts >> 8], dim=-1).view(-1, 10)
    lanes_per_blk = 32
    by = by.view(-1, lanes_per_blk, 10).to(torch.uint8)
    return torch.cat([by[:, :, :8].reshape(-1, 256),
                      by[:, :, 8:].reshape(-1, 64)], dim=1).flatten()


def pack_scales(s):
    n, ks = s.shape
    return s.view(n // 16, 16, ks).transpose(1, 2).contiguous().flatten()


def quant_a32(a):
    """[M, K] f32 -> a32 activations: e4m3 codes + exact f32 per-32-group
    scales + f32 dequant (amax/448 rule, same as production group-128 quant
    just 4x finer; the e8m0-scale variant lost GSM8K accuracy — see
    gen_moe_w2.py)."""
    m, k = a.shape
    ab = a.view(m, k // 32, 32)
    a_s = (ab.abs().amax(-1).clamp_min(1e-10) / 448.0)
    a8 = (ab / a_s[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).view(m, k)
    deq = a8.float() * a_s.repeat_interleave(32, 1)
    return a8, a_s, deq


cu = Cuda()
fn = cu.load_kernel(CUBIN, "moe_w4q_mm")
fn_s = cu.load_kernel(W4S_CUBIN, "moe_w4s_mm") if W4S_CUBIN else None

descs = np.zeros((E, 8), dtype=np.uint64)
descs_s = np.zeros((E, 8), dtype=np.uint64)
refs, d_cs = [], []
for e in range(E):
    nibs = torch.randint(0, 16, (N, K), dtype=torch.uint8)
    nibs.view(-1)[:16] = torch.arange(16, dtype=torch.uint8)  # all nibbles
    sexp = torch.randint(120, 132, (N, K // 32), dtype=torch.uint8)
    a8, a_s, a_deq = quant_a32(torch.randn(M, K) * 0.5)

    code = NIBBLE_TO_CODE[nibs.long()]
    scale = torch.exp2(sexp.float() - 127.0).repeat_interleave(32, 1)
    w_true = E2M1[nibs.long()] * scale        # TRUE FP4 — merge-free
    refs.append(a_deq @ w_true.T)

    d_a = cu.to_device(a8.view(torch.uint8).numpy())
    d_as = cu.to_device(a_s.float().numpy().view(np.uint8))
    d_base = cu.to_device(pack_fragment_major(code).numpy())
    d_ref = cu.to_device(pack_quintal_fragment_major(nibs).numpy())
    d_bs = cu.to_device(pack_scales(sexp).numpy())
    d_c = cu.alloc(M * N * 2)
    d_cs.append(d_c)
    descs[e] = [d_a.value, d_as.value, d_base.value, d_ref.value, d_bs.value,
                d_c.value, np.uint64(M), 0]
    if fn_s is not None:
        # w4s slot for the SAME experts (2-bit refinement, merge included)
        MAG_TO_REF = torch.tensor([0, 0, 1, 2, 3, 0, 1, 2], dtype=torch.uint8)
        d_refs = cu.to_device(pack_fragment_major(
            MAG_TO_REF[(nibs & 7).long()]).numpy())
        descs_s[e] = [d_a.value, d_as.value, d_base.value, d_refs.value,
                      d_bs.value, d_c.value, np.uint64(M), 0]

d_desc = cu.to_device(descs.view(np.uint8))
args = [d_desc, ctypes.c_uint32(K), ctypes.c_uint32(K // 64),
        ctypes.c_uint32(N * 2), ctypes.c_uint32(K // 128)]

outs, worst = [], 0.0
for r in range(RUNS):
    for d_c in d_cs:
        cu.memset32(d_c, 0, M * N // 2)
    cu.launch(fn, (N // 16, E, 1), (NWARP * 32, 1, 1), args)
    cu.synchronize()
    blob = b""
    for e, d_c in enumerate(d_cs):
        raw = cu.from_device(d_c, M * N * 2, dtype=np.uint16).copy()
        blob += raw.tobytes()
        got = torch.from_numpy(raw.reshape(M, N).copy()).view(torch.bfloat16).float()
        rel = (got - refs[e]).abs().max().item() / refs[e].abs().max().item()
        worst = max(worst, rel)
    outs.append(blob)

ok = worst < 2.5e-2 and len(set(outs)) == 1
print(f"moe_w4q NWARP={NWARP} N={N} K={K} E={E} M={M}: worst_rel={worst:.3e} "
      f"distinct={len(set(outs))} (reference is TRUE e2m1 — no merge)")

if BENCH and fn_s is not None:
    d_desc_s = cu.to_device(descs_s.view(np.uint8))
    for name, f, dd in (("w4q", fn, d_desc), ("w4s", fn_s, d_desc_s)):
        bargs = [dd] + args[1:]
        for _ in range(20):
            cu.launch(f, (N // 16, E, 1), (NWARP * 32, 1, 1), bargs)
        cu.synchronize()
        t0 = time.perf_counter()
        REP = 200
        for _ in range(REP):
            cu.launch(f, (N // 16, E, 1), (NWARP * 32, 1, 1), bargs)
        cu.synchronize()
        dt = (time.perf_counter() - t0) / REP
        print(f"  bench {name}: {dt*1e6:.1f} us/launch "
              f"(grid {N//16}x{E}, {NWARP*32} thr)")

print(f"RESULT: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
