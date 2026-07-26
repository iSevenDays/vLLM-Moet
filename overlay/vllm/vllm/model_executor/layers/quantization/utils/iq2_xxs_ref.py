# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU reference dequant for IQ2_XXS (GGML type 16) and Q2_K (GGML type 10).

This is Step 0 of the IQ2_XXS -> Ada (sm_89) native-decode port: a numpy-only
golden reference that bit-exactly matches antirez's serving implementation so
the later Triton kernel has something to validate against.

Sources mirrored (verbatim tables + math):
  * /root/antirez/ds4/ds4_iq2_tables_cuda.inc     - cuda_iq2xxs_grid[256], cuda_ksigns_iq2xs[128]
  * /root/antirez/ds4/metal/moe.metal             - ds4_metal_kmask_iq2xs[8] = {1..128},
                                                     dequantize_iq2_xxs (moe.metal:2887-2906)
  * /root/antirez/ds4/ds4_cuda.cu                 - dev_dot_iq2_xxs_q8_K_block (16962-16983),
                                                     dev_dot_q2_K_q8_K_block   (17239-17261)
  * /root/antirez/ds4/ds4.c                       - block_iq2_xxs (66B), block_q2_K (84B),
                                                     gguf_types[16] / [10] metadata

Block layouts (QK_K = 256 elements per block):

  block_iq2_xxs  (66 bytes):
      uint16_t d          bytes 0..1    (fp16 super-block scale)
      uint16_t qs[32]     bytes 2..65   (8 sub-blocks x 4 u16 = 8 bytes each)

  block_q2_K     (84 bytes):
      uint8_t  scales[16] bytes 0..15   (low nibble = 4-bit per-sub-block scale,
                                         high nibble = 4-bit per-sub-block min)
      uint8_t  qs[64]     bytes 16..79  (256 x 2-bit codes, interleaved layout)
      uint16_t d          bytes 80..81  (fp16 scale)
      uint16_t dmin       bytes 82..83  (fp16 min scale)

IQ2_XXS dequant math (per 8-element group k=0..3 inside each 32-elem sub-block):
      grid   = iq2xxs_grid[a_k]                 (8 bytes, each in {8, 25, 43})
      signs  = ksigns_iq2xs[sign_k]             (8 sign bits)
      ls     = 2 * (aux1 >> 28) + 1             (odd 4-bit local scale in [1, 31])
      w[i]   = grid[i] * (-1 if signs & (1<<i) else +1)
      y[i]   = 0.125 * d * ls * w[i]
  Algebraic note: antirez's metal form ``d * (0.5 + ls_field) * 0.25`` is
  identical to ``0.125 * (2*ls_field + 1)`` because 0.25*(0.5+L) == 0.125*(2L+1).

Q2_K dequant math (per 16-element sub-block is=0..15):
      scale  = scales[is] & 0x0f
      min    = scales[is] >> 4
      code[i]= (qs[byte_base + i] >> shift) & 3      (raw 2-bit code in {0,1,2,3})
      y[is*16 + i] = d * scale * code[i] - dmin * min
  (antirez's CUDA dot-product path uses the raw {0,1,2,3} code with no lookup
  table -- so this reference does the same. The standard llama.cpp Q2_K path
  uses a `kvalues_iq2xs` offset lookup; that is NOT what antirez serves, so we
  intentionally diverge to match the production CUDA math bit-for-bit.)

Usage:
    python3 iq2_xxs_ref.py            # validate against the real GGUF
    python3 iq2_xxs_ref.py --selftest # block-layout self-test (no GGUF needed)

The validation path mmaps /root/antirez/ds4/ds4flash.gguf, parses the GGUF
tensor directory, locates layer-0 expert-0 ffn_gate_exps (IQ2_XXS) and
ffn_down_exps (Q2_K), and dequantizes the first block of each.
"""

from __future__ import annotations

import argparse
import mmap
import os
import struct
import sys
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
QK_K = 256
IQ2_XXS_BLOCK_BYTES = 66
Q2_K_BLOCK_BYTES = 84

# GGUF type ids (antirez/ds4.c gguf_types[]).
GGUF_TYPE_Q2_K = 10
GGUF_TYPE_IQ2_XXS = 16

DEFAULT_GGUF = "/root/antirez/ds4/ds4flash.gguf"

# ---------------------------------------------------------------------------
# Lookup tables -- copied verbatim from antirez (ds4_iq2_tables_cuda.inc /
# moe.metal). Do not edit by hand; the assertions below guard them.
# ---------------------------------------------------------------------------

# 128 uint8 sign masks -- ds4_iq2_tables_cuda.inc:1-8 / moe.metal:21-30.
_KSIGNS_IQ2XS: List[int] = [
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
]

# 256 uint64 grid entries -- ds4_iq2_tables_cuda.inc:11-76 / moe.metal:32-97.
# Each entry packs 8 little-endian bytes, each byte in {0x08, 0x19, 0x2b}
# = {8, 25, 43}.
_IQ2XXS_GRID: List[int] = [
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08, 0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819, 0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b, 0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08, 0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819, 0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808, 0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808, 0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919, 0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08, 0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819, 0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808, 0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808, 0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819, 0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819, 0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19, 0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b, 0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908, 0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08, 0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819, 0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808, 0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819, 0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b, 0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808, 0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b, 0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08, 0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808, 0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b, 0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819, 0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908, 0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808, 0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b, 0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19, 0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
]

# moe.metal:17-19 -- {1, 2, 4, 8, 16, 32, 64, 128}.
_KMASK_IQ2XS: List[int] = [1, 2, 4, 8, 16, 32, 64, 128]


class _Tables:
    """Lazily-decoded table tensors (module-level singletons)."""

    def __init__(self) -> None:
        self.ksigns = np.asarray(_KSIGNS_IQ2XS, dtype=np.uint8)  # (128,)
        self.kmask = np.asarray(_KMASK_IQ2XS, dtype=np.uint8)    # (8,)
        assert self.ksigns.shape == (128,), "ksigns_iq2xs must have 128 entries"
        assert self.kmask.shape == (8,), "kmask_iq2xs must have 8 entries"

        grid_u64 = np.asarray(_IQ2XXS_GRID, dtype=np.uint64)     # (256,)
        assert grid_u64.shape == (256,), "iq2xxs_grid must have 256 entries"
        # Decode each uint64 into 8 little-endian bytes.
        # grid_bytes[g, i] = (grid_u64[g] >> (8*i)) & 0xff
        shifts = np.arange(8, dtype=np.uint64) * np.uint64(8)
        self.grid_bytes = (
            (grid_u64[:, None] >> shifts) & np.uint64(0xff)
        ).astype(np.uint8)                                       # (256, 8)
        # Every byte must be in {8, 25, 43} (the IQ2_XXS alphabet).
        present = sorted(np.unique(self.grid_bytes).tolist())
        assert present == [8, 25, 43], (
            f"iq2xxs_grid bytes must be {{8, 25, 43}}, got {present}"
        )


TABLES = _Tables()


# ---------------------------------------------------------------------------
# fp16 helper -- numpy can view uint16 bit patterns directly as float16, then
# upcast to float32. This is exact for normals/subnormals/inf and gives a
# quiet NaN for NaN payloads (dequant scales are never NaN in practice).
# ---------------------------------------------------------------------------
def f16_bits_to_f32(u16: np.ndarray) -> np.ndarray:
    """Convert IEEE-754 binary16 bit patterns (uint16) to float32.

    Mirrors CUDA ``__half2float``: every normal/subnormal/inf bit pattern is
    preserved exactly through the float16 -> float32 upcast.
    """
    u16 = np.asarray(u16, dtype=np.uint16)
    # Reinterpret the raw bytes as float16, then widen to float32. frombuffer
    # gives a read-only view of the underlying buffer with no copy of the data.
    f16 = np.frombuffer(u16.tobytes(), dtype=np.float16)
    return f16.astype(np.float32)


def _f16_scalar(u16_bits: int) -> float:
    """Convenience scalar wrapper around :func:`f16_bits_to_f32`."""
    return float(f16_bits_to_f32(np.array([u16_bits], dtype=np.uint16))[0])


# ---------------------------------------------------------------------------
# IQ2_XXS dequant
# ---------------------------------------------------------------------------
def dequant_iq2_xxs_block(block_bytes: bytes) -> np.ndarray:
    """Dequantize one 66-byte IQ2_XXS block -> float32[256].

    Output order: sub-block ib32 (0..7) -> 32 contiguous values; within each
    sub-block, 4 groups of 8 values (a0, a1, a2, a3). Matches the CUDA
    dot-product layout so a vectorized gather followed by a dot with q8
    reproduces antirez's dev_dot_iq2_xxs_q8_K_block bit-for-bit.
    """
    if len(block_bytes) != IQ2_XXS_BLOCK_BYTES:
        raise ValueError(
            f"IQ2_XXS block must be {IQ2_XXS_BLOCK_BYTES} bytes, "
            f"got {len(block_bytes)}"
        )

    # d: fp16 super-block scale.
    d_u16 = struct.unpack_from("<H", block_bytes, 0)[0]
    d = _f16_scalar(d_u16)

    # qs: 32 uint16 little-endian.
    qs = np.frombuffer(block_bytes, dtype="<u2", count=32, offset=2)  # (32,)

    # Reshape into 8 sub-blocks of 4 uint16 each.
    qs32 = qs.reshape(8, 4).astype(np.uint32)
    aux0 = qs32[:, 0] | (qs32[:, 1] << np.uint32(16))   # (8,) 4 grid indices
    aux1 = qs32[:, 2] | (qs32[:, 3] << np.uint32(16))   # (8,) sign idxs + ls

    # Local scale: ls = 2*(aux1>>28) + 1 (odd, in [1, 31]).
    ls = (np.uint32(2) * (aux1 >> np.uint32(28)) + np.uint32(1)).astype(
        np.int32
    )                                                      # (8,)

    # Decode 4 (grid_idx, sign_idx) pairs per sub-block.
    out = np.empty((8, 32), dtype=np.float32)  # ib32, within-sub-block pos
    for k in range(4):
        a_k = ((aux0 >> np.uint32(8 * k)) & np.uint32(0xff)).astype(np.int64)
        sign_k = (
            (aux1 >> np.uint32(7 * k)) & np.uint32(0x7f)
        ).astype(np.int64)                                 # (8,)

        grid_vals = TABLES.grid_bytes[a_k]                 # (8 ib32, 8 bytes)
        signs_mask = TABLES.ksigns[sign_k]                 # (8 ib32, 8 bits)
        # sign[i] = -1 if (signs_mask & kmask[i]) else +1
        sign_neg = (
            (signs_mask & TABLES.kmask[None, :]) != 0
        )                                                   # (8, 8) bool
        signed = grid_vals.astype(np.int32) * np.where(
            sign_neg, np.int32(-1), np.int32(1)
        )                                                   # (8, 8)
        out[:, k * 8:(k + 1) * 8] = signed.astype(np.float32)

    # Apply d and ls (per ib32).
    scale = np.float32(0.125) * np.float32(d) * ls.astype(np.float32)
    out *= scale[:, None]
    return out.reshape(QK_K)


def dequant_iq2_xxs(tensor_bytes: bytes, shape: Tuple[int, ...]) -> np.ndarray:
    """Dequantize a full IQ2_XXS tensor.

    ``shape`` is GGUF dims in the canonical (n_dims leading) order; the
    returned ndarray has the same shape and dtype float32. The element count
    must be a multiple of 256 (the IQ2_XXS block size).
    """
    n_elems = int(np.prod(shape)) if shape else 0
    if n_elems % QK_K != 0:
        raise ValueError(
            f"IQ2_XXS element count {n_elems} not a multiple of {QK_K}"
        )
    n_blocks = n_elems // QK_K
    expected = n_blocks * IQ2_XXS_BLOCK_BYTES
    if len(tensor_bytes) < expected:
        raise ValueError(
            f"IQ2_XXS tensor truncated: have {len(tensor_bytes)} bytes, "
            f"need {expected}"
        )

    out = np.empty((n_blocks, QK_K), dtype=np.float32)
    for b in range(n_blocks):
        chunk = tensor_bytes[
            b * IQ2_XXS_BLOCK_BYTES:(b + 1) * IQ2_XXS_BLOCK_BYTES
        ]
        out[b] = dequant_iq2_xxs_block(chunk)
    return out.reshape(shape)


# ---------------------------------------------------------------------------
# Q2_K dequant
# ---------------------------------------------------------------------------
def dequant_q2_k_block(block_bytes: bytes) -> np.ndarray:
    """Dequantize one 84-byte Q2_K block -> float32[256].

    Matches antirez's dev_dot_q2_K_q8_K_block layout: the 16 sub-blocks map
    sequentially to output positions [is*16 .. is*16+15], and each sub-block
    reads 16 codes from ``qs`` at a sub-block-specific byte base and 2-bit
    shift (see module docstring for the indexing derivation).
    """
    if len(block_bytes) != Q2_K_BLOCK_BYTES:
        raise ValueError(
            f"Q2_K block must be {Q2_K_BLOCK_BYTES} bytes, "
            f"got {len(block_bytes)}"
        )

    scales = np.frombuffer(block_bytes, dtype=np.uint8, count=16, offset=0)
    qs = np.frombuffer(block_bytes, dtype=np.uint8, count=64, offset=16)
    d_u16 = struct.unpack_from("<H", block_bytes, 16 + 64)[0]
    dmin_u16 = struct.unpack_from("<H", block_bytes, 16 + 64 + 2)[0]
    d = _f16_scalar(d_u16)
    dmin = _f16_scalar(dmin_u16)

    out = np.empty(QK_K, dtype=np.float32)
    for isub in range(16):
        k = isub // 8
        within = isub % 8
        j = within // 2
        half = within % 2
        byte_base = k * 32 + half * 16
        shift = j * 2
        scale = int(scales[isub] & 0x0f)
        min_v = int(scales[isub] >> 4)
        codes = (qs[byte_base:byte_base + 16] >> shift) & 0x03  # (16,) in 0..3
        vals = np.float32(d) * np.float32(scale) * codes.astype(np.float32) \
            - np.float32(dmin) * np.float32(min_v)
        out[isub * 16:isub * 16 + 16] = vals
    return out


def dequant_q2_k(tensor_bytes: bytes, shape: Tuple[int, ...]) -> np.ndarray:
    """Dequantize a full Q2_K tensor -> float32 with the requested shape."""
    n_elems = int(np.prod(shape)) if shape else 0
    if n_elems % QK_K != 0:
        raise ValueError(
            f"Q2_K element count {n_elems} not a multiple of {QK_K}"
        )
    n_blocks = n_elems // QK_K
    expected = n_blocks * Q2_K_BLOCK_BYTES
    if len(tensor_bytes) < expected:
        raise ValueError(
            f"Q2_K tensor truncated: have {len(tensor_bytes)} bytes, "
            f"need {expected}"
        )

    out = np.empty((n_blocks, QK_K), dtype=np.float32)
    for b in range(n_blocks):
        chunk = tensor_bytes[b * Q2_K_BLOCK_BYTES:(b + 1) * Q2_K_BLOCK_BYTES]
        out[b] = dequant_q2_k_block(chunk)
    return out.reshape(shape)


# ---------------------------------------------------------------------------
# Minimal GGUF parser (header + tensor directory only; metadata skipped).
# ---------------------------------------------------------------------------
class GgufTensor:
    __slots__ = ("name", "ndim", "dims", "gguf_type", "offset", "nbytes")

    def __init__(self, name: str, ndim: int, dims: List[int],
                 gguf_type: int, offset: int, nbytes: int) -> None:
        self.name = name
        self.ndim = ndim
        self.dims = dims
        self.gguf_type = gguf_type
        self.offset = offset
        self.nbytes = nbytes

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.dims:
            n *= d
        return n


_GGUF_TYPE_INFO: Dict[int, Tuple[str, int, int]] = {
    # gguf_type_id: (name, block_elems, block_bytes)  -- only what we need.
    0:  ("f32",     1,   4),
    1:  ("f16",     1,   2),
    10: ("q2_k",    256, 84),
    16: ("iq2_xxs", 256, 66),
    30: ("bf16",    1,   2),
}


def _tensor_nbytes(gguf_type: int, n_elements: int) -> int:
    info = _GGUF_TYPE_INFO.get(gguf_type)
    if info is None:
        raise ValueError(f"Unsupported GGUF type {gguf_type}")
    _, block_elems, block_bytes = info
    n_blocks = (n_elements + block_elems - 1) // block_elems
    return n_blocks * block_bytes


def _read_gguf_string(buf: bytes, pos: int) -> Tuple[str, int]:
    (n,) = struct.unpack_from("<Q", buf, pos)
    pos += 8
    s = buf[pos:pos + n].decode("utf-8", errors="replace")
    pos += n
    return s, pos


# GGUF metadata value types we need to skip to reach the tensor directory.
_GGUF_VALUE_UINT8 = 0
_GGUF_VALUE_INT8 = 1
_GGUF_VALUE_UINT16 = 2
_GGUF_VALUE_INT16 = 3
_GGUF_VALUE_UINT32 = 4
_GGUF_VALUE_INT32 = 5
_GGUF_VALUE_FLOAT32 = 6
_GGUF_VALUE_BOOL = 7
_GGUF_VALUE_STRING = 8
_GGUF_VALUE_ARRAY = 9
_GGUF_VALUE_UINT64 = 10
_GGUF_VALUE_INT64 = 11
_GGUF_VALUE_FLOAT64 = 12

_SCALAR_SIZE = {
    _GGUF_VALUE_UINT8: 1, _GGUF_VALUE_INT8: 1, _GGUF_VALUE_UINT16: 2,
    _GGUF_VALUE_INT16: 2, _GGUF_VALUE_UINT32: 4, _GGUF_VALUE_INT32: 4,
    _GGUF_VALUE_FLOAT32: 4, _GGUF_VALUE_BOOL: 1, _GGUF_VALUE_UINT64: 8,
    _GGUF_VALUE_INT64: 8, _GGUF_VALUE_FLOAT64: 8,
}


def _skip_gguf_value(buf: bytes, pos: int, vtype: int) -> int:
    if vtype in _SCALAR_SIZE:
        return pos + _SCALAR_SIZE[vtype]
    if vtype == _GGUF_VALUE_STRING:
        (n,) = struct.unpack_from("<Q", buf, pos)
        pos += 8 + n
        return pos
    if vtype == _GGUF_VALUE_ARRAY:
        (item_type,) = struct.unpack_from("<I", buf, pos)
        pos += 4
        (length,) = struct.unpack_from("<Q", buf, pos)
        pos += 8
        for _ in range(length):
            pos = _skip_gguf_value(buf, pos, item_type)
        return pos
    raise ValueError(f"Unknown GGUF metadata value type {vtype}")


def parse_gguf_header(buf: bytes) -> Tuple[List[GgufTensor], int, int]:
    """Parse the GGUF header + tensor directory.

    Returns (tensors, data_offset, alignment). ``data_offset`` is the absolute
    byte offset within ``buf`` where the tensor data section starts.
    """
    magic = struct.unpack_from("<I", buf, 0)[0]
    if magic != 0x46554747:  # "GGUF" little-endian.
        raise ValueError(f"Bad GGUF magic 0x{magic:08x}")
    version = struct.unpack_from("<I", buf, 4)[0]
    if version != 3:
        # v1/v2 had different field widths; antirez's model is v3.
        sys.stderr.write(
            f"[iq2_xxs_ref] warning: GGUF version {version} (only v3 "
            f"validated); parsing may fail.\n"
        )
    n_tensors = struct.unpack_from("<Q", buf, 8)[0]
    n_kv = struct.unpack_from("<Q", buf, 16)[0]

    pos = 24
    alignment = 32  # default per GGUF spec.

    # Skip the metadata KV block, watching for "general.alignment".
    for _ in range(n_kv):
        key, pos = _read_gguf_string(buf, pos)
        (vtype,) = struct.unpack_from("<I", buf, pos)
        pos += 4
        if key == "general.alignment" and vtype in _SCALAR_SIZE:
            sz = _SCALAR_SIZE[vtype]
            alignment = int.from_bytes(buf[pos:pos + sz], "little")
        pos = _skip_gguf_value(buf, pos, vtype)

    # Parse the tensor directory.
    tensors: List[GgufTensor] = []
    for _ in range(n_tensors):
        name, pos = _read_gguf_string(buf, pos)
        (ndim,) = struct.unpack_from("<I", buf, pos)
        pos += 4
        dims = list(struct.unpack_from(f"<{ndim}Q", buf, pos))
        pos += 8 * ndim
        (gguf_type,) = struct.unpack_from("<I", buf, pos)
        pos += 4
        (rel_offset,) = struct.unpack_from("<Q", buf, pos)
        pos += 8
        n_elems = 1
        for d in dims:
            n_elems *= d
        nbytes = _tensor_nbytes(gguf_type, n_elems) \
            if gguf_type in _GGUF_TYPE_INFO else 0
        tensors.append(
            GgufTensor(name, ndim, dims, gguf_type, rel_offset, nbytes)
        )

    data_offset = pos
    # Align the data section start to ``alignment``.
    if alignment > 1:
        data_offset = (data_offset + alignment - 1) & ~(alignment - 1)
    return tensors, data_offset, alignment


def find_tensors_by_prefix(tensors: List[GgufTensor], prefix: str,
                           gguf_type: int) -> List[GgufTensor]:
    return [t for t in tensors
            if t.name.startswith(prefix) and t.gguf_type == gguf_type]


# ---------------------------------------------------------------------------
# Validation entry points.
# ---------------------------------------------------------------------------
def _selftest() -> bool:
    """Block-layout self-test: exercise the dequant with hand-crafted blocks.

    Builds deterministic IQ2_XXS and Q2_K blocks, runs the dequant, and
    asserts finiteness + expected values. Useful when the real GGUF is not
    reachable. Returns True on success.
    """
    print("[selftest] IQ2_XXS grid alphabet:",
          sorted(np.unique(TABLES.grid_bytes).tolist()))
    assert TABLES.grid_bytes.shape == (256, 8)
    assert TABLES.ksigns.shape == (128,)
    assert TABLES.kmask.tolist() == [1, 2, 4, 8, 16, 32, 64, 128]

    # fp16 round-trip on a few scale values.
    for v in (0.0, 1.0, 0.5, 0.0625, 0.001, -0.01, 65504.0):
        bits = int(np.array([v], dtype=np.float16).view(np.uint16)[0])
        back = _f16_scalar(bits)
        assert abs(back - v) <= 1e-3 * (1.0 + abs(v)), \
            f"fp16 round-trip failed for {v}: got {back}"

    # Build an IQ2_XXS block: d=1.0, all-zero qs (ls=1, grid idx 0, signs 0).
    # grid[0] = 0x0808...08 -> all 8s; signs=0 -> all +1; ls=1 -> 0.125*1*1*8=1.0.
    block = bytearray(IQ2_XXS_BLOCK_BYTES)
    block[0] = 0x00
    block[1] = 0x3c  # fp16 1.0
    out = dequant_iq2_xxs_block(bytes(block))
    assert out.shape == (QK_K,)
    assert np.all(np.isfinite(out)), "IQ2_XXS dequant produced non-finite"
    assert np.allclose(out, 1.0), \
        f"Expected all 1.0, got min={out.min()} max={out.max()}"

    # Build a Q2_K block: d=1, dmin=0, all-zero codes -> all zeros.
    q2block = bytearray(Q2_K_BLOCK_BYTES)
    q2block[80] = 0x00
    q2block[81] = 0x3c  # d = 1.0
    # dmin=0 (fp16 zero) -- bytes 82,83 already 0.
    out2 = dequant_q2_k_block(bytes(q2block))
    assert out2.shape == (QK_K,)
    assert np.all(np.isfinite(out2))
    assert np.allclose(out2, 0.0), \
        f"Expected all 0.0, got min={out2.min()} max={out2.max()}"

    print("[selftest] IQ2_XXS + Q2_K block dequant OK "
          "(finite, expected values)")
    return True


def _validate_gguf(gguf_path: str) -> int:
    """Validate against the real antirez GGUF.

    Returns the process exit code (0 on success).
    """
    if not os.path.exists(gguf_path):
        sys.stderr.write(f"[iq2_xxs_ref] GGUF not found: {gguf_path}\n")
        return 1

    print(f"[gguf] opening {gguf_path} "
          f"({os.path.getsize(gguf_path) / 1e9:.1f} GB)")
    with open(gguf_path, "rb") as fh:
        with mmap.mmap(fh.fileno(), 0, prot=mmap.PROT_READ) as mm:
            buf = bytes(mm[:64 * 1024 * 1024])  # header lives well under 64MB
            tensors, data_off, align = parse_gguf_header(buf)
            print(f"[gguf] {len(tensors)} tensors, data section at byte "
                  f"{data_off}, alignment {align}")

            # Spot-check: every grid byte in {8, 25, 43}.
            print("[gguf] IQ2_XXS grid alphabet:",
                  sorted(np.unique(TABLES.grid_bytes).tolist()))

            # ----- locate layer-0 expert-0 weights -----
            gate_t = find_tensors_by_prefix(
                tensors, "blk.0.ffn_gate_exps.", GGUF_TYPE_IQ2_XXS)
            up_t = find_tensors_by_prefix(
                tensors, "blk.0.ffn_up_exps.", GGUF_TYPE_IQ2_XXS)
            down_t = find_tensors_by_prefix(
                tensors, "blk.0.ffn_down_exps.", GGUF_TYPE_Q2_K)

            for label, lst in (("ffn_gate_exps (IQ2_XXS)", gate_t),
                               ("ffn_up_exps   (IQ2_XXS)", up_t),
                               ("ffn_down_exps (Q2_K)   ", down_t)):
                if not lst:
                    print(f"[gguf] no layer-0 {label} tensors found")
                    continue
                lst.sort(key=lambda t: t.name)
                print(f"[gguf] {label}: {len(lst)} expert tensors; "
                      f"first names: {[t.name for t in lst[:3]]}")

            # Pick the first expert (weight name typically ends in ".0").
            def _pick_expert0(cands: List[GgufTensor]) -> GgufTensor:
                for t in cands:
                    if t.name.endswith(".0"):
                        return t
                return cands[0]

            exit_code = 0

            # ----- IQ2_XXS first block of expert-0 ffn_gate_exps -----
            if gate_t:
                t = _pick_expert0(gate_t)
                abs_off = data_off + t.offset
                print(f"\n[iq2_xxs] tensor {t.name!r} "
                      f"type={t.gguf_type} dims={t.dims} "
                      f"nbytes={t.nbytes} (data_off={data_off}, "
                      f"abs_off={abs_off})")
                with open(gguf_path, "rb") as bf:
                    bf.seek(abs_off)
                    first_block = bf.read(IQ2_XXS_BLOCK_BYTES)
                if len(first_block) != IQ2_XXS_BLOCK_BYTES:
                    print("[iq2_xxs] ERROR: short read")
                    exit_code = 1
                else:
                    vals = dequant_iq2_xxs_block(first_block)
                    finite = bool(np.all(np.isfinite(vals)))
                    amax = float(np.max(np.abs(vals)))
                    print(f"[iq2_xxs] first 32 dequant values (block 0):")
                    print("        ", np.array2string(vals[:32], precision=5,
                                                     max_line_width=120))
                    print(f"[iq2_xxs] finite={finite} max|x|={amax:.6f} "
                          f"min={float(vals.min()):.6f} "
                          f"max={float(vals.max()):.6f}")
                    if not finite:
                        print("[iq2_xxs] FAIL: non-finite values")
                        exit_code = 1
                    # Magnitude envelope: typical post-scale O(0.001..0.1).
                    if amax > 10.0:
                        print(f"[iq2_xxs] WARN: max|x|={amax:.4f} looks large; "
                              f"verify d/ls interpretation")
                    else:
                        print(f"[iq2_xxs] magnitude envelope OK "
                              f"(max|x|={amax:.4f})")

            # ----- Q2_K first block of expert-0 ffn_down_exps -----
            if down_t:
                t = _pick_expert0(down_t)
                abs_off = data_off + t.offset
                print(f"\n[q2_k] tensor {t.name!r} "
                      f"type={t.gguf_type} dims={t.dims} "
                      f"nbytes={t.nbytes} (data_off={data_off}, "
                      f"abs_off={abs_off})")
                with open(gguf_path, "rb") as bf:
                    bf.seek(abs_off)
                    first_block = bf.read(Q2_K_BLOCK_BYTES)
                if len(first_block) != Q2_K_BLOCK_BYTES:
                    print("[q2_k] ERROR: short read")
                    exit_code = 1
                else:
                    vals = dequant_q2_k_block(first_block)
                    finite = bool(np.all(np.isfinite(vals)))
                    amax = float(np.max(np.abs(vals)))
                    print(f"[q2_k] first 32 dequant values (block 0):")
                    print("        ", np.array2string(vals[:32], precision=5,
                                                     max_line_width=120))
                    print(f"[q2_k] finite={finite} max|x|={amax:.6f} "
                          f"min={float(vals.min()):.6f} "
                          f"max={float(vals.max()):.6f}")
                    if not finite:
                        print("[q2_k] FAIL: non-finite values")
                        exit_code = 1
                    if amax > 10.0:
                        print(f"[q2_k] WARN: max|x|={amax:.4f} looks large; "
                              f"verify d/dmin nibble interpretation")
                    else:
                        print(f"[q2_k] magnitude envelope OK "
                              f"(max|x|={amax:.4f})")

            return exit_code


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[3])
    parser.add_argument(
        "--gguf", default=DEFAULT_GGUF,
        help=f"Path to the GGUF to validate against (default: {DEFAULT_GGUF})",
    )
    parser.add_argument(
        "--selftest", action="store_true",
        help="Run the block-layout self-test (no GGUF needed)",
    )
    args = parser.parse_args(argv)

    ok = _selftest()
    if not ok:
        return 1
    if args.selftest:
        return 0
    return _validate_gguf(args.gguf)


if __name__ == "__main__":
    raise SystemExit(main())
