#!/usr/bin/env python3
"""CPU golden for the Ada (sm_89) Triton moe_w2_mm port — runs anywhere.

Replicates, in torch CPU ops, EXACTLY the address arithmetic and fold
order of kernels/triton/moe_w2_sm89.py (fragment-major byte offsets, the
byte-major k permutation the kernel gathers A through, the UE8M0
exponent-bitcast, the per-32-group outer-product scale fold, the m_rows
mask) and checks it two ways against moe_w2_check.py's reference math:

  1. EXACT: the decoded W tile (via the kernel's byte offsets + k_perm)
     must equal LEVELS[codes] element-for-element at the permuted k
     positions — an addressing proof over every (nb, kb, row, k).
  2. NUMERIC: full pair GEMM vs the f32 reference, same verdict as the
     op check (worst_rel < 2.5e-2) plus RUNS identical outputs.

This validates everything about the port EXCEPT Triton codegen/launch —
that half is kernels/gen/moe_w2_check_sm89.py on real sm_89 silicon.

Run: python3 kernels/gen/moe_w2_sm89_cpu_check.py   (all K, ~a minute)
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

LEVELS = torch.tensor([-4.0, -1.0, 1.0, 4.0])


# ---- host packers (verbatim from moe_w2_check.py / moe_w2_planes) ----------
def pack_fragment_major(codes):
    n, k = codes.shape
    c = codes.view(n // 16, 2, 8, k // 64, 2, 2, 4, 4)
    c = c.permute(0, 3, 2, 6, 1, 4, 5, 7).contiguous().view(-1, 4).to(torch.int32)
    return (c[:, 0] | (c[:, 1] << 2) | (c[:, 2] << 4) | (c[:, 3] << 6)).to(torch.uint8)


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


# ---- the kernel's index arithmetic, replicated 1:1 --------------------------
def kernel_indices(K):
    """(row_off[16], byte_col[8], k_perm[32], shift[32]) — the constant
    index vectors moe_w2_mm_sm89 builds with tl.arange."""
    r = torch.arange(16)
    row_off = (r % 8) * 32 + (r // 8) * 4
    b_idx = torch.arange(8)
    byte_col = (b_idx // 2) * 8 + (b_idx % 2)
    f = torch.arange(32)
    k_perm = ((f // 4) % 2) * 16 + (f // 8) * 4 + (f % 4)
    shift = (f % 4) * 2
    return row_off, byte_col, k_perm, shift


def ue8m0(sb):
    """2^(sb-127) via exponent-field bitcast — the kernel's exact fold."""
    w = (sb.to(torch.int32) << 23).view(torch.float32)
    return torch.where(sb == 0, torch.tensor(5.877471754111438e-39), w)


def emulate_pair(plane, sbytes, a8, a_s, m_rows, n_rows, K, codes_ref):
    """One (pair) GEMM exactly as the Triton kernel computes it, all
    16-row N tiles. Returns C [m_rows, n_rows] f32 (pre-bf16-store)."""
    row_off, byte_col, k_perm, shift = kernel_indices(K)
    a8f = a8.float()                       # e4m3 -> exact in f32/bf16
    C = torch.zeros(16, n_rows)
    exact_bad = 0
    for pid_n in range(n_rows // 16):
        acc = torch.zeros(16, 16)
        for g in range(K // 32):
            kb, k32 = g // 2, g % 2
            k_glob = kb * 64 + k32 * 32 + k_perm            # [32]
            a = a8f[:, k_glob]                              # [16, 32] masked below
            blk = (pid_n * (K // 64) + kb) * 256
            boff = blk + row_off[:, None] + (byte_col + k32 * 2)[None, :]
            code_bytes = plane[boff.long()].to(torch.int32)  # [16, 8]
            expanded = code_bytes[:, :, None].expand(16, 8, 4).reshape(16, 32)
            codes = (expanded >> shift[None, :]) & 3
            # addressing proof: decoded codes == source codes at (row, k)
            rows = pid_n * 16 + torch.arange(16)
            exact_bad += int((codes != codes_ref[rows][:, k_glob]).sum())
            mag = torch.where((codes == 0) | (codes == 3), 4.0, 1.0)
            w = torch.where(codes < 2, -mag, mag)           # [16, 32]
            sb = sbytes[(pid_n * (K // 32) + g) * 16
                        + torch.arange(16)]
            w_s = ue8m0(sb)                                 # [16]
            asg = a_s[:, g]                                 # [16]
            acc = acc + (a.to(torch.bfloat16).float()
                         @ w.to(torch.bfloat16).float().T) \
                * (asg[:, None] * w_s[None, :])
        C[:, pid_n * 16:(pid_n + 1) * 16] = acc
    return C[:m_rows], exact_bad


fail = False
for K in KS:
    worst, bad, outs = 0.0, 0, []
    for run in range(RUNS):
        torch.manual_seed(1000 + K + run * 0)   # same data every run
        run_blob = b""
        for e in range(E):
            m_rows = (1, 4, 16)[e % 3]
            codes = torch.randint(0, 4, (N, K), dtype=torch.uint8)
            sexp = torch.randint(120, 132, (N, K // 32), dtype=torch.uint8)
            a_full = torch.randn(16, K) * 0.5
            a8, a_s, a_deq = quant_a32(a_full)
            w_deq = LEVELS[codes.long()] * torch.exp2(
                sexp.float() - 127.0).repeat_interleave(32, 1)
            ref = a_deq[:m_rows] @ w_deq.T

            got, exact_bad = emulate_pair(
                pack_fragment_major(codes), pack_scales(sexp),
                a8, a_s, m_rows, N, K, codes.to(torch.int32))
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
