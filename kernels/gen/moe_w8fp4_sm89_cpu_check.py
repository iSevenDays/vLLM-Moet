#!/usr/bin/env python3
"""CPU golden for the Ada (sm_89) Triton moe_w8fp4_mm port (FP4-STORAGE mode).

Replicates, in torch CPU ops, EXACTLY the FP4-storage fragment-major nibble
address arithmetic and fold order of
overlay/.../moe_w2_sm89.py::moe_w8fp4_mm_sm89 and checks it two ways against
a torch reference:

  1. EXACT: the extracted nibble (via the kernel's byte offsets) must equal the
     source nibble at the permuted k position — an addressing proof over every
     (nb, kb, row, k4-pair).
  2. NUMERIC: full pair GEMM vs the f32 reference (worst_rel < 2.5e-2, same
     gate as the 2-bit op check) plus RUNS IDENTICAL outputs.

FP4-STORAGE layout: e2m1 nibble plane packed by pack_fp4_fragment_major — the
SAME [nb, kb, g, t, tile, k32, half, k4] element order as the FP8 byte plane,
two nibbles per byte. Even k4 maps to the LOW nibble (bits [3:0]), odd k4 to
the HIGH nibble (bits [7:4]). Every offset is the FP8 kernel's halved:

  FP8  in-block byte offset: g*128 + t*32 + tile*16 + k32*8  + half*4 + k4
  FP4  in-block byte offset: g*64  + t*16 + tile*8  + k32*4  + half*2 + k4//2
  nibble selector:           k4 & 1  (0 = low nibble, 1 = high nibble)

Each (nb, kb) FP8 block is 1024 B; the FP4 block is 512 B.

Each nibble decodes to its EXACT e4m3 bit pattern:
  s = nib >> 3, e2 = (nib>>1)&3, m = nib&1
  pat = (e2>0 ? ((e2+6)<<3)|(m<<2) : (m>0 ? 48 : 0)) | (s<<7)

This is validated by an inline decode proof (all 16 nibbles) at the top of the
main loop.

Scale math: the UE8M0 block-32 weight scale (2^(sb-127) via exponent bitcast)
and the f32 per-32-group activation scale fold as one [M, N] outer product,
exactly as the FP8 kernel.

This validates everything about the FP4-STORAGE layout + math EXCEPT the
Triton fp8 tl.dot codegen — that half is
moe_w2_sm89.py::w8_self_test on real sm_89 silicon (runs at boot).

Run: python3 kernels/gen/moe_w8fp4_sm89_cpu_check.py   (all K, ~a minute)
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

# e2m1 value table: index = nibble (unsigned 4-bit e2m1 — sign in bit 3)
_E2M1_VALS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
              0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


# ---- host packers (verbatim from moe_w2_planes) ----------------------------

def pack_fp4_fragment_major(nib):
    """[N, K] u8 nibbles -> fragment-major FP4 nibble plane [N*K//2] u8.

    Same [nb, kb, g, t, tile, k32, half, k4] permutation as pack_fp8, but
    two nibbles per byte: even k4 in the LOW nibble, odd k4 in the HIGH
    nibble. The halved byte offset derivation: every FP8 offset is //2 (2
    nibbles/byte share one byte), and k4//2 is the byte sub-index, k4&1
    selects the nibble within that byte."""
    N, K = nib.shape
    # view as [nb, 2, 8, kb, 2, 2, 4, 4] = [nb, tile, g, kb, k32, half, t, k4]
    # then permute to [nb, kb, g, t, tile, k32, half, k4] (k4 dim is 4; the
    # flattened order pairs (k4=0,1) and (k4=2,3) into consecutive bytes)
    c = nib.view(N // 16, 2, 8, K // 64, 2, 2, 4, 4)
    c = c.permute(0, 3, 2, 6, 1, 4, 5, 7).contiguous()
    # pack pairs of nibbles: even k4 = low nibble, odd k4 = high nibble
    flat = c.flatten().to(torch.int32)
    lo = flat[0::2] & 0xF
    hi = flat[1::2] & 0xF
    return (lo | (hi << 4)).to(torch.uint8)


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


# ---- e2m1 nibble -> e4m3 bit-pattern decode, all 16 nibbles ----------------

def e2m1_to_e4m3_pat(nib_int: int) -> int:
    """Decode one nibble to its exact e4m3 bit-pattern (unsigned byte).

    Formula from the kernel: s = nib>>3, e2 = (nib>>1)&3, m = nib&1.
    Normals (e2>0): e4m3 exp = e2-1+7 = e2+6, mantissa = m<<2.
    Subnormal (e2==0, m==1): 0.5 = 2^-1 -> e4m3 pattern 0x30 (exp=6, mant=0).
    Zero (e2==0, m==0): pattern 0 (both +0 and -0 are correct; sign set below).
    Sign bit: s -> bit 7."""
    s = (nib_int >> 3) & 1
    e2 = (nib_int >> 1) & 3
    m = nib_int & 1
    if e2 > 0:
        body = ((e2 + 6) << 3) | (m << 2)
    else:
        body = 48 if m > 0 else 0  # 48 = 0x30
    return body | (s << 7)


def e4m3_pat_to_float(pat: int) -> float:
    """Decode e4m3 bit-pattern (unsigned byte) to float (float8_e4m3fn rules)."""
    s = (pat >> 7) & 1
    exp = (pat >> 3) & 0xF
    mant = pat & 0x7
    if exp == 0:
        val = (mant / 8.0) * (2.0 ** -6)   # subnormal: 2^(-bias-2+1) = 2^-6 mantissa
    else:
        val = (1.0 + mant / 8.0) * (2.0 ** (exp - 7))
    return -val if s else val


def verify_decode_table():
    """Assert the e4m3 bit-pattern formula agrees with the _E2M1_VALS table
    for all 16 nibbles. Returns the 16 (e2m1_val, e4m3_decoded_val) pairs."""
    pairs = []
    for nib in range(16):
        pat = e2m1_to_e4m3_pat(nib)
        decoded = e4m3_pat_to_float(pat)
        expected = _E2M1_VALS[nib]
        assert abs(decoded - expected) < 1e-9, (
            f"nibble {nib:#x}: formula decoded {decoded} != table {expected} "
            f"(pat={pat:#04x})")
        pairs.append((expected, decoded))
    return pairs


# ---- the kernel's FP4 index arithmetic, replicated 1:1 --------------------

def kernel_indices_fp4():
    """(row_off[16], byte_col[32], nib_shift[32], k_perm[32]) — the constant
    index vectors moe_w8fp4_mm_sm89 builds with tl.arange.

    Derivation: FP8 offsets halved (2 nibbles/byte). row_off = g*64 + tile*8
    (was g*128 + tile*16). byte_col = (f//8)*16 + ((f//4)%2)*2 + (f%4)//2
    (was (f//8)*32 + ((f//4)%2)*4 + (f%4)). nib_shift = (f%2)*4 selects low
    (shift=0, even k4) or high (shift=4, odd k4) nibble. k_perm is identical
    to the w2/w8 kernels."""
    r = torch.arange(16)
    row_off = (r % 8) * 64 + (r // 8) * 8
    f = torch.arange(32)
    byte_col = (f // 8) * 16 + ((f // 4) % 2) * 2 + (f % 4) // 2
    nib_shift = (f % 2) * 4
    k_perm = ((f // 4) % 2) * 16 + (f // 8) * 4 + (f % 4)
    return row_off, byte_col, nib_shift, k_perm


def ue8m0(sb):
    """2^(sb-127) via exponent-field bitcast — the kernel's exact fold."""
    w = (sb.to(torch.int32) << 23).view(torch.float32)
    return torch.where(sb == 0, torch.tensor(5.877471754111438e-39), w)


def emulate_pair_fp4(plane, sbytes, a8, a_s, m_rows, n_rows, K, nib_src):
    """One (pair) GEMM exactly as the w8fp4 kernel computes it, all 16-row N
    tiles. Returns C [m_rows, n_rows] f32 (pre-bf16-store) + the nibble
    mismatch count (addressing proof)."""
    row_off, byte_col, nib_shift, k_perm = kernel_indices_fp4()
    a8f = a8.float()                                # e4m3 -> exact in f32
    C = torch.zeros(16, n_rows)
    exact_bad = 0
    for pid_n in range(n_rows // 16):
        acc = torch.zeros(16, 16)
        for g in range(K // 32):
            kb, k32 = g // 2, g % 2
            k_glob = kb * 64 + k32 * 32 + k_perm            # [32]
            a = a8f[:, k_glob]                               # [16, 32]
            # FP4 block = 512 B (vs 1024 B for FP8)
            blk = (pid_n * (K // 64) + kb) * 512
            boff = blk + row_off[:, None] + (byte_col + k32 * 4)[None, :]
            wb = plane[boff.long()].to(torch.int32)          # [16, 32] u8
            nib = (wb >> nib_shift[None, :]) & 0xF           # [16, 32]

            # addressing proof: extracted nibble == source nibble at (row, k_glob)
            rows = pid_n * 16 + torch.arange(16)
            exact_bad += int((nib != nib_src[rows][:, k_glob].to(torch.int32)).sum())

            # decode nibble -> e4m3 bit-pattern (exact: e2m1 subset of e4m3)
            s = nib >> 3
            e2 = (nib >> 1) & 3
            m = nib & 1
            body = torch.where(e2 > 0, ((e2 + 6) << 3) | (m << 2),
                               torch.where(m > 0, torch.tensor(48), torch.tensor(0)))
            pat = (body | (s << 7)).to(torch.uint8)
            # decode proof: formula vs _E2M1_VALS table
            w_via_table = torch.tensor(_E2M1_VALS, dtype=torch.float32)[nib.long()]
            w_via_formula = pat.view(torch.float8_e4m3fn).float()
            assert torch.allclose(w_via_table, w_via_formula, atol=1e-6), (
                f"decode mismatch at g={g} pid_n={pid_n}")
            w = w_via_formula

            sb = sbytes[(pid_n * (K // 32) + g) * 16 + torch.arange(16)]
            w_s = ue8m0(sb)                                  # [16]
            asg = a_s[:, g]                                  # [16]
            acc = acc + (a @ w.T) * (asg[:, None] * w_s[None, :])
        C[:, pid_n * 16:(pid_n + 1) * 16] = acc
    return C[:m_rows], exact_bad


# ---- decode table self-check (pure ints, no GPU, runs unconditionally) -----
print("Decode table self-check (all 16 nibbles):")
pairs = verify_decode_table()
for nib, (e2m1_val, e4m3_val) in enumerate(pairs):
    pat = e2m1_to_e4m3_pat(nib)
    print(f"  nib={nib:#04x} e2m1={e2m1_val:+5.2f}  e4m3_pat={pat:#04x} "
          f"decoded={e4m3_val:+5.2f}")
print("Decode table: PASS")


fail = False
for K in KS:
    worst, bad, outs = 0.0, 0, []
    for run in range(RUNS):
        torch.manual_seed(2000 + K + run * 0)   # same data every run
        run_blob = b""
        for e in range(E):
            m_rows = (1, 4, 16)[e % 3]
            # random e2m1 nibbles [N, K]
            nib_src = torch.randint(0, 16, (N, K), dtype=torch.uint8)
            sexp = torch.randint(120, 132, (N, K // 32), dtype=torch.uint8)
            a_full = torch.randn(16, K) * 0.5
            a8, a_s, a_deq = quant_a32(a_full)
            # reference: _E2M1_VALS dequant * 2^(sexp-127)
            w_deq = torch.tensor(_E2M1_VALS, dtype=torch.float32)[
                nib_src.long()] * torch.exp2(
                sexp.float() - 127.0).repeat_interleave(32, 1)
            ref = a_deq[:m_rows] @ w_deq.T

            got, exact_bad = emulate_pair_fp4(
                pack_fp4_fragment_major(nib_src),
                pack_scales(sexp), a8, a_s, m_rows, N, K, nib_src)
            got = got.to(torch.bfloat16).float()    # the kernel's bf16 store
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
