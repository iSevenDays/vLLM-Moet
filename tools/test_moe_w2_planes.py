#!/usr/bin/env python3
"""Golden tests for moe_w2_planes vs the QUANT_PROBE-validated repack tool.

1. nibble->code map == repack_expert_bits' tensor-sym cmap16 (levels
   {-4,-1,1,4}, odd-symmetric tie-break).
2. pack_fragment_major layout: every byte lands where the kernel doc says.
3. round-trip: codes -> plane -> (python unpack) -> codes.
4. QUINTAL split (moe_w4q_mm): decode(base(n), digits(n)) == e2m1(n) for
   ALL 16 nibbles — the split-FP4 bit-exactness criterion (the legacy
   2-bit refinement fails it on nibble 0/8: mag-0 merge).
5. pack_quintal_fragment_major layout: every 10-bit word lands where the
   kernel doc says (P8/P2 sections, per-lane 80-bit LE record).
6. MXFP4 scale refit never increases exact block SSE and cannot underflow.
7. Base-only W2 enables scale refit when the environment does not override it.

Run: python3 tools/test_moe_w2_planes.py  (CPU, ~seconds)
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "tools")
sys.path.insert(0, ".")

from repack_expert_bits import _subset_tables, _candidate_level_sets  # noqa: E402

import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location(
    "moe_w2_planes",
    "overlay/vllm/vllm/model_executor/layers/quantization/utils/moe_w2_planes.py")
_m = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_m)
_CODE_TO_NIBBLE = _m._CODE_TO_NIBBLE
_NIBBLE_TO_CODE = _m._NIBBLE_TO_CODE
mxfp4_to_codes = _m.mxfp4_to_codes
mxfp4_refit_codes_scales = _m.mxfp4_refit_codes_scales
pack_fragment_major = _m.pack_fragment_major
pack_quintal_fragment_major = _m.pack_quintal_fragment_major
nibbles_to_quintal_digits = _m.nibbles_to_quintal_digits
quintal_dequant = _m.quintal_dequant
scale_refit_enabled = _m.scale_refit_enabled
split_fp4_dequant = _m.split_fp4_dequant

# ---- 1. quantization map equals the validated tool -------------------------
err16, cmap16 = _subset_tables(4, symmetric=True)
sets = _candidate_level_sets(4, symmetric=True)
target = sets.index((-4.0, -1.0, 1.0, 4.0))
cmap = cmap16[target]            # e2m1 code -> e2m1 code of nearest level
CODE_VALUES = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6] * 2)
CODE_VALUES[8:] *= -1
lv = {-4.0: 0, -1.0: 1, 1.0: 2, 4.0: 3}
ok = True
for c in range(16):
    expect_code = lv[float(CODE_VALUES[cmap[c]])]
    got_code = int(_NIBBLE_TO_CODE[c])
    if expect_code != got_code:
        print(f"MISMATCH nibble {c:#x}: tool->{expect_code} ours->{got_code}")
        ok = False
assert ok, "nibble->code map diverges from the validated tool"
print("1. quantization map == repack tool (16/16 nibbles)")

# also: reconstruction nibbles match the tool's cmap
for code in range(4):
    pass
recon = {int(_NIBBLE_TO_CODE[c]): int(cmap[c]) for c in range(16)}
for c in range(16):
    assert int(_CODE_TO_NIBBLE[int(_NIBBLE_TO_CODE[c])]) == int(cmap[c]), c
print("2. reconstruction nibbles == tool cmap")

# ---- 3. fragment-major layout positions ------------------------------------
N, K = 32, 128
codes = torch.arange(N * K, dtype=torch.int64).reshape(N, K) % 4
codes = codes.to(torch.uint8)
plane = pack_fragment_major(codes)
assert plane.numel() == N * K // 4


def plane_byte(nb, kb, g, t, j):
    """Index of fragment byte j (0..7) for lane (g,t) of block (nb,kb)."""
    lanes_per_blk = 32 * 8
    blk = (nb * (K // 64) + kb)
    lane = g * 4 + t
    return blk * lanes_per_blk + lane * 8 + j


def expect_byte(nb, kb, g, t, j):
    tile, rest = divmod(j, 4)
    k32, half = divmod(rest, 2)
    row = nb * 16 + tile * 8 + g
    kbase = kb * 64 + k32 * 32 + half * 16 + t * 4
    b = 0
    for k4 in range(4):
        b |= int(codes[row, kbase + k4]) << (2 * k4)
    return b


bad = 0
for nb in range(N // 16):
    for kb in range(K // 64):
        for g in range(8):
            for t in range(4):
                for j in range(8):
                    got = int(plane[plane_byte(nb, kb, g, t, j)])
                    exp = expect_byte(nb, kb, g, t, j)
                    if got != exp:
                        bad += 1
assert bad == 0, f"{bad} plane bytes misplaced"
print("3. fragment-major layout exact (all bytes)")

# ---- 4. mxfp4 nibble order --------------------------------------------------
w = torch.tensor([[0x52, 0xE0]], dtype=torch.uint8)   # nibbles: 2,5 then 0,E
c = mxfp4_to_codes(w)
assert c.tolist() == [[2, 3, 2, 0]], c.tolist()        # 1.0->+1, 3->+4, +0->+1, -4->-4
print("4. mxfp4 nibble order (lo=even k)")

# ---- 5. QUINTAL split bit-exactness (all 16 nibbles) ------------------------
E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6] * 2, dtype=torch.float64)
E2M1[8:] *= -1
nib16 = torch.arange(16, dtype=torch.uint8)
got = quintal_dequant(nib16).double()
assert torch.equal(got, E2M1), (got, E2M1)
# digits stay in base-5 and reconstruct the mag index with the class
d = nibbles_to_quintal_digits(nib16)
assert d.min() >= 0 and d.max() <= 4, d
# the legacy 2-bit refinement fails exactly on the zeros (documents the
# reason quintal replaced it)
legacy = split_fp4_dequant(nib16).double()
mism = (legacy != E2M1).nonzero().flatten().tolist()
assert mism == [0, 8], mism
print("5. quintal decode == e2m1 for 16/16 nibbles (legacy fails 0x0/0x8)")

# ---- 6. quintal plane layout (every word, every lane) -----------------------
N, K = 32, 128
nibs = torch.randint(0, 16, (N, K), dtype=torch.uint8)
nibs.view(-1)[:16] = torch.arange(16, dtype=torch.uint8)
plane = pack_quintal_fragment_major(nibs)
assert plane.numel() == N * K * 5 // 16
digits = nibbles_to_quintal_digits(nibs)


def elem(nb, kb, g, t, j, k4):
    """(row, k) of element k4 of plane byte j for lane (g,t) in (nb,kb)."""
    tile, rest = divmod(j, 4)
    k32, half = divmod(rest, 2)
    row = nb * 16 + tile * 8 + g
    k = kb * 64 + k32 * 32 + half * 16 + t * 4 + k4
    return row, k


BLK = 320                        # 256 B P8 + 64 B P2 per (nb, kb)
bad = 0
for nb in range(N // 16):
    for kb in range(K // 64):
        blk = (nb * (K // 64) + kb) * BLK
        for g in range(8):
            for t in range(4):
                lane = g * 4 + t
                p8 = plane[blk + lane * 8: blk + lane * 8 + 8]
                p2 = plane[blk + 256 + lane * 2: blk + 256 + lane * 2 + 2]
                rec = int.from_bytes(bytes(p8.tolist() + p2.tolist()),
                                     "little")
                for w in range(8):
                    c = (rec >> (10 * w)) & 0x3FF
                    want = 0
                    for k4 in range(4):
                        r, k = elem(nb, kb, g, t, w, k4)
                        want += int(digits[r, k]) * 5 ** k4
                    if c != want:
                        bad += 1
assert bad == 0, f"{bad} quintal words misplaced"
print("6. quintal fragment-major layout exact (all words)")

# ---- 7. block scale refit ---------------------------------------------------
# Magnitude 2 is exact with the lower scale and its large code.
nib = torch.full((16, 64), 4, dtype=torch.uint8)
w = (nib[:, 0::2] | (nib[:, 1::2] << 4)).to(torch.uint8)
scales = torch.full((16, 2), 100, dtype=torch.uint8)
codes, refit_scales = mxfp4_refit_codes_scales(w, scales)
assert torch.all(codes == 3)
assert torch.all(refit_scales == 99)

# Byte zero cannot decrease. The normal conversion must remain unchanged.
zero_scales = torch.zeros_like(scales)
codes, refit_scales = mxfp4_refit_codes_scales(w, zero_scales)
assert torch.equal(codes, mxfp4_to_codes(w))
assert torch.equal(refit_scales, zero_scales)

# Test all nibble values in random block mixtures. Express the refit output
# in units of the original checkpoint block scale.
torch.manual_seed(1)
nib = torch.randint(0, 16, (64, 256), dtype=torch.uint8)
w = (nib[:, 0::2] | (nib[:, 1::2] << 4)).to(torch.uint8)
scales = torch.randint(0, 255, (64, 8), dtype=torch.uint8)
codes, refit_scales = mxfp4_refit_codes_scales(w, scales)
orig = _m._E2M1_VALS[nib.long()].view(64, 8, 32)
levels = torch.tensor([-4.0, -1.0, 1.0, 4.0], dtype=torch.float64)
base = levels[mxfp4_to_codes(w).long()].view(64, 8, 32)
scale_ratio = torch.exp2(
    refit_scales.double() - scales.double()).repeat_interleave(32, dim=-1)
refit = (levels[codes.long()] * scale_ratio).view(64, 8, 32)
assert torch.all(
    ((refit - orig) ** 2).sum(-1) <= ((base - orig) ** 2).sum(-1))
print("7. MXFP4 scale refit lowers or preserves exact block SSE")

# ---- 8. base-only default ---------------------------------------------------
old_refit = os.environ.pop("VLLM_MOE_W2_SCALE_REFIT", None)
assert scale_refit_enabled(has_fp4_delta=False)
assert not scale_refit_enabled(has_fp4_delta=True)
os.environ["VLLM_MOE_W2_SCALE_REFIT"] = "0"
assert not scale_refit_enabled(has_fp4_delta=False)
os.environ["VLLM_MOE_W2_SCALE_REFIT"] = "1"
assert scale_refit_enabled(has_fp4_delta=False)
if old_refit is None:
    os.environ.pop("VLLM_MOE_W2_SCALE_REFIT")
else:
    os.environ["VLLM_MOE_W2_SCALE_REFIT"] = old_refit
print("8. MXFP4 scale refit defaults on for base-only W2")
print("ALL PASS")
