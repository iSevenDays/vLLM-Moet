#!/usr/bin/env python3
"""CPU golden for the Ada (sm_89) Triton moe_w8_mm port (FP8-e4m3 delta tier).

Replicates, in torch CPU ops, EXACTLY the FP8 fragment-major byte address
arithmetic and fold order of overlay/.../moe_w2_sm89.py::moe_w8_mm_sm89 and
checks it two ways against a torch reference:

  1. EXACT: the loaded W tile (via the kernel's byte offsets + k_perm) must
     equal the source e4m3 bytes element-for-element at the permuted k
     positions — an addressing proof over every (nb, kb, row, k).
  2. NUMERIC: full pair GEMM vs the f32 reference (worst_rel < 2.5e-2, same
     gate as the 2-bit op check) plus RUNS IDENTICAL outputs.

The FP8 layout is 1 byte/elem in the SAME fragment-major element order as the
2-bit codes (pack_fragment_major's [nb, kb, g, t, tile, k32, half, k4]
permutation), but WITHOUT k4 packing — each element is one e4m3 byte. The
in-block byte offset is g*128 + t*32 + tile*16 + k32*8 + half*4 + k4 (4x the
2-bit g*32 + t*8 + tile*4 + k32*2 + half, because FP8 unpacks the 4
codes/byte to one byte/elem). The kernel loads 32 bytes per (N-row,
32-group) via a 32-element gather whose byte-major order f=0..31 maps through
the SAME k_perm as the 2-bit kernel — so the A-gather is identical.

Scale math: the UE8M0 block-32 weight scale (2^(sb-127) via exponent
bitcast) and the f32 per-32-group activation scale fold as one [M, N] outer
product, exactly as the 2-bit kernel.

This validates everything about the FP8 layout + math EXCEPT the Triton fp8
tl.dot codegen — that half is moe_w2_sm89.py::w8_self_test on real sm_89
silicon (runs at boot, gated at 2.5e-2).

Run: python3 kernels/gen/moe_w8_sm89_cpu_check.py   (all K, ~a minute)
"""
import os
import sys

import torch

torch.manual_seed(int(os.environ.get("SEED", "7")))
KS = tuple(int(x) for x in os.environ.get(
    "KS", "512,1024,2048,4096,6144,7168").split(","))
N = int(os.environ.get("N", "64"))
E = int(os.environ.get("E", "3"))
RUNS = int(os.environ.get("RUNS", "4"))


# ---- host packer (verbatim from moe_w2_planes.pack_fp8_fragment_major) -----
def pack_fp8_fragment_major(w_fp8):
    """[N, K] u8 e4m3 bytes -> fragment-major FP8 plane [N*K] u8.

    Same [nb, kb, g, t, tile, k32, half, k4] permutation as the 2-bit
    pack_fragment_major, but 1 byte/elem (no k4 packing)."""
    N, K = w_fp8.shape
    c = w_fp8.view(N // 16, 2, 8, K // 64, 2, 2, 4, 4)
    c = c.permute(0, 3, 2, 6, 1, 4, 5, 7).contiguous()
    return c.flatten().to(torch.uint8)


def pack_scales(s):
    n, ks = s.shape
    return s.view(n // 16, 16, ks).transpose(1, 2).contiguous().flatten()


def quant_a32(a):
    m, k = a.shape
    ab = a.view(m, k // 32, 32)
    a_s = (ab.abs().amax(-1).clamp_min(1e-10) / 448.0)
    a8 = (ab / a_s[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).view(m, k)
    deq = a8.float() * a_s.repeat_interleave(32, 1)
    return a8, a_s, deq


# ---- the kernel's FP8 index arithmetic, replicated 1:1 -------------------
def kernel_indices_fp8():
    """(row_off[16], byte_col[32], k_perm[32]) — the constant index vectors
    moe_w8_mm_sm89 builds with tl.arange. Compare to the 2-bit
    kernel_indices: row_off is g*128+tile*16 (vs g*32+tile*4), byte_col is a
    32-element gather (vs 8-element), k_perm is IDENTICAL."""
    r = torch.arange(16)
    row_off = (r % 8) * 128 + (r // 8) * 16
    f = torch.arange(32)
    byte_col = (f // 8) * 32 + ((f // 4) % 2) * 4 + (f % 4)
    k_perm = ((f // 4) % 2) * 16 + (f // 8) * 4 + (f % 4)
    return row_off, byte_col, k_perm


def ue8m0(sb):
    """2^(sb-127) via exponent-field bitcast — the kernel's exact fold."""
    w = (sb.to(torch.int32) << 23).view(torch.float32)
    return torch.where(sb == 0, torch.tensor(5.877471754111438e-39), w)


def emulate_pair_fp8(plane, sbytes, a8, a_s, m_rows, n_rows, K, w_src):
    """One (pair) GEMM exactly as the w8 kernel computes it, all 16-row N
    tiles. Returns C [m_rows, n_rows] f32 (pre-bf16-store) + the layout
    mismatch count (addressing proof)."""
    row_off, byte_col, k_perm = kernel_indices_fp8()
    a8f = a8.float()                       # e4m3 -> exact in f32
    C = torch.zeros(16, n_rows)
    exact_bad = 0
    for pid_n in range(n_rows // 16):
        acc = torch.zeros(16, 16)
        for g in range(K // 32):
            kb, k32 = g // 2, g % 2
            k_glob = kb * 64 + k32 * 32 + k_perm            # [32]
            a = a8f[:, k_glob]                              # [16, 32]
            blk = (pid_n * (K // 64) + kb) * 1024           # FP8 block = 1024 B
            boff = blk + row_off[:, None] + (byte_col + k32 * 8)[None, :]
            w_bytes = plane[boff.long()]                    # [16, 32] u8
            w = w_bytes.view(torch.float8_e4m3fn).float()   # e4m3 -> f32
            # addressing proof: loaded RAW BYTES == source bytes at (row, k_glob)
            rows = pid_n * 16 + torch.arange(16)
            exact_bad += int(
                (w_bytes != w_src[rows][:, k_glob]).sum())
            sb = sbytes[(pid_n * (K // 32) + g) * 16
                        + torch.arange(16)]
            w_s = ue8m0(sb)                                 # [16]
            asg = a_s[:, g]                                 # [16]
            acc = acc + (a @ w.T) * (asg[:, None] * w_s[None, :])
        C[:, pid_n * 16:(pid_n + 1) * 16] = acc
    return C[:m_rows], exact_bad


fail = False
for K in KS:
    worst, bad, outs = 0.0, 0, []
    for run in range(RUNS):
        torch.manual_seed(2000 + K + run * 0)   # same data every run
        run_blob = b""
        for e in range(E):
            m_rows = (1, 4, 16)[e % 3]
            # random e4m3 weights [N, K] (unit-space values * 2^(sb-127))
            w_fp8 = (torch.randn(N, K) * 0.5).clamp(-448, 448).to(
                torch.float8_e4m3fn)
            sexp = torch.randint(120, 132, (N, K // 32), dtype=torch.uint8)
            a_full = torch.randn(16, K) * 0.5
            a8, a_s, a_deq = quant_a32(a_full)
            w_deq = w_fp8.float() * torch.exp2(
                sexp.float() - 127.0).repeat_interleave(32, 1)
            ref = a_deq[:m_rows] @ w_deq.T

            got, exact_bad = emulate_pair_fp8(
                pack_fp8_fragment_major(w_fp8.view(torch.uint8)),
                pack_scales(sexp), a8, a_s, m_rows, N, K,
                w_fp8.view(torch.uint8))
            got = got.to(torch.bfloat16).float()   # the kernel's bf16 store
            bad += exact_bad
            rel = (got - ref).abs().max().item() / ref.abs().max().item()
            worst = max(worst, rel)
            run_blob += got.to(torch.bfloat16).view(torch.uint16).numpy().tobytes()
        outs.append(run_blob)
    ok = worst < 2.5e-2 and bad == 0 and len(set(outs)) == 1
    print(f"K={K:5d}: worst_rel={worst:.3e} layout_mismatches={bad} "
          f"distinct={len(set(outs))} -> {'PASS' if ok else 'FAIL'}")
    fail |= not ok

print(f"RESULT: {'FAIL' if fail else 'PASS'}")
sys.exit(1 if fail else 0)
