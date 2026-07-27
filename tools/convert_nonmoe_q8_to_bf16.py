#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GGUF -> vLLM safetensors converter for the non-MoE Q8_0 tensors.

Hypothesis test (branch feat/moe-w2-sm89-native-decode): the DSv4 attention
projections are loaded at fp8 e4m3 in our vLLM checkpoint (~9.7% rel error,
3-bit mantissa), while antirez serves them at Q8_0 (~0.78% rel error, 12x more
precise). The precision loss in the Q/K/V/O projections compounds across 43
layers and may be the root cause of the 16K needle retrieval failure.

This converter extracts the non-MoE Linear weights from antirez's GGUF (which
are stored at Q8_0), dequantizes them to bf16 (matching llama.cpp's
quantize_row_q8_0_ref math: per-32-block amax -> int8 + fp16 scale; dequant:
qs * scale), and writes them under the vLLM-expected param names. The fp8
shards ALSO have these tensors (as fp8 + per-tensor `.scale`); the boot path
adds the matching layer prefixes to ``ignored_layers`` in config.json so vLLM
loads them with ``UnquantizedLinearMethod`` (plain bf16 Linear, no scale), and
the safetensors index is rewritten to point these names at the new bf16 file.

Targets per layer L:
    blk.L.attn_q_a.weight        (Q8_0) -> layers.L.attn.wq_a.weight
    blk.L.attn_kv.weight         (Q8_0) -> layers.L.attn.wkv.weight
    blk.L.attn_q_b.weight        (Q8_0) -> layers.L.attn.wq_b.weight
    blk.L.attn_output_a.weight   (Q8_0) -> layers.L.attn.wo_a.weight
    blk.L.attn_output_b.weight   (Q8_0) -> layers.L.attn.wo_b.weight
    blk.L.ffn_gate_shexp.weight  (Q8_0) -> layers.L.ffn.shared_experts.w1.weight
    blk.L.ffn_up_shexp.weight    (Q8_0) -> layers.L.ffn.shared_experts.w3.weight
    blk.L.ffn_down_shexp.weight  (Q8_0) -> layers.L.ffn.shared_experts.w2.weight
Plus the LM head:
    output.weight                (Q8_0) -> head.weight  (= lm_head after mapper)

The wq_a/wkv shards are fused into ``attn.fused_wqa_wkv`` by the
MergedColumnParallelLinear weight loader at load time (identical to the fp8
checkpoint layout), so we emit them UN-fused.

GGUF block layout reference: ds4.c:tensor_expect_tensor / quantize_row_q8_0_ref
in llama.cpp. Per 32-element block:
    uint16_t d (fp16 scale)    bytes 0..1
    int8_t qs[32]              bytes 2..33
Dequant math:
    y[i] = d * qs[i]

The element ordering inside a Q8_0 tensor matches the row-major layout of the
GGUF dims (ne[0] is the contiguous dimension). Reshaping the flat dequant
result to (ne[1], ne[0]) yields the PyTorch [out_features, in_features]
convention that vLLM's Linear weight loader expects.

Usage:
    python3 tools/convert_nonmoe_q8_to_bf16.py \\
        --gguf /root/antirez/ds4/ds4flash.gguf \\
        --out  /root/models/DeepSeek-V4-Flash-IQ2/dsv4_nonmoe_bf16.safetensors \\
        --model-dir /root/models/DeepSeek-V4-Flash
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import struct
import sys
import time
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch  # for bf16 handling in the validation path

# ----------------------------------------------------------------------------
# Locate the Step 0 reference parser (reuse, do not duplicate).
# ----------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REF_PATH = os.path.normpath(os.path.join(
    _HERE, "..", "overlay", "vllm", "vllm", "model_executor", "layers",
    "quantization", "utils", "iq2_xxs_ref.py"))
if not os.path.exists(_REF_PATH):
    sys.exit(f"[convert] cannot find iq2_xxs_ref.py at {_REF_PATH}")
sys.path.insert(0, os.path.dirname(_REF_PATH))

# Reuse the Step 0 parser.
from iq2_xxs_ref import parse_gguf_header  # noqa: E402

# ----------------------------------------------------------------------------
# GGUF type ids we care about (antirez/ds4.c gguf_types[]).
# ----------------------------------------------------------------------------
GGUF_TYPE_F32 = 0
GGUF_TYPE_F16 = 1
GGUF_TYPE_Q8_0 = 8
GGUF_TYPE_BF16 = 30

QK8_0 = 32
Q8_0_BLOCK_BYTES = 34  # 2 (fp16 scale) + 32 (int8)

# ----------------------------------------------------------------------------
# DeepSeek-V4-Flash shape constants (for sanity-checks).
# ----------------------------------------------------------------------------
N_LAYERS = 43
HIDDEN = 4096         # DS4_N_EMBD
INTERMEDIATE = 2048   # DS4_N_FF_EXP (per-expert; shared expert uses 1x)
Q_LORA = 1024         # q_lora_rank
HEAD_DIM = 512        # head_dim (kv proj out_features)
O_LORA = 1024         # o_lora_rank
N_HEADS = 64
N_GROUPS = 8  # o_groups
VOCAB = 129280


# ----------------------------------------------------------------------------
# GGUF -> vLLM name mapping for the tensors we extract.
# ----------------------------------------------------------------------------
GGUF_TO_VLLM_LAYER_MAP: Dict[str, str] = {
    # GGUF tail (after blk.L.) -> vLLM tail (after layers.L.)
    "attn_q_a.weight":         "attn.wq_a.weight",
    "attn_kv.weight":          "attn.wkv.weight",
    "attn_q_b.weight":         "attn.wq_b.weight",
    "attn_output_a.weight":    "attn.wo_a.weight",
    "attn_output_b.weight":    "attn.wo_b.weight",
    "ffn_gate_shexp.weight":   "ffn.shared_experts.w1.weight",
    "ffn_up_shexp.weight":     "ffn.shared_experts.w3.weight",
    "ffn_down_shexp.weight":   "ffn.shared_experts.w2.weight",
}

# Subsets used by --targets. The MLA_QKV set is the hypothesis-target set
# (Q/K/V precision); the full set adds the output projection + shared expert
# + lm_head (broke the boot on the DSv4 custom fp8 o_proj kernel).
TARGET_SETS: Dict[str, set] = {
    "mla_qkv": {"attn.wq_a.weight", "attn.wkv.weight", "attn.wq_b.weight"},
    "all": set(GGUF_TO_VLLM_LAYER_MAP.values()) | {"head.weight"},
}

# Top-level (non-blk) mappings.
GGUF_TO_VLLM_TOP_MAP: Dict[str, str] = {
    "output.weight": "head.weight",  # the weights_mapper renames head.weight -> lm_head.weight
}


def gguf_to_vllm_name(gguf_name: str) -> str | None:
    """Map a GGUF tensor name to the vLLM (pre-mapper) checkpoint name.

    Returns None for tensors we don't extract (MoE experts, LUTs, norms,
    embeddings, HC routing, MTP -- we only replace the fp8-quantized non-MoE
    Linear weights that the hypothesis is about).
    """
    if gguf_name in GGUF_TO_VLLM_TOP_MAP:
        return GGUF_TO_VLLM_TOP_MAP[gguf_name]
    if gguf_name.startswith("blk."):
        parts = gguf_name.split(".", 2)  # ["blk", "L", "tail"]
        if len(parts) != 3:
            return None
        try:
            L = int(parts[1])
        except ValueError:
            return None
        tail = parts[2]
        if tail in GGUF_TO_VLLM_LAYER_MAP:
            return f"layers.{L}.{GGUF_TO_VLLM_LAYER_MAP[tail]}"
    return None


# ----------------------------------------------------------------------------
# Q8_0 dequant (vectorized, exact match to llama.cpp quantize_row_q8_0_ref).
# ----------------------------------------------------------------------------
def dequant_q8_0_to_f32(raw: bytes, n_elements: int) -> np.ndarray:
    """Dequantize Q8_0 bytes -> 1-D float32 array of length n_elements.

    Q8_0 = 32 int8 values + 1 fp16 scale per block (34 bytes). The output is
    the flat row-major array (``y[i] = d * qs[i]`` per block), which the caller
    reshapes to the PyTorch [out, in] convention.
    """
    if n_elements % QK8_0 != 0:
        raise ValueError(f"Q8_0 element count {n_elements} not a multiple of {QK8_0}")
    n_blocks = n_elements // QK8_0
    expected = n_blocks * Q8_0_BLOCK_BYTES
    if len(raw) < expected:
        raise ValueError(
            f"Q8_0 tensor truncated: have {len(raw)} bytes, need {expected}")
    arr = np.frombuffer(raw, dtype=np.uint8, count=expected
                        ).reshape(n_blocks, Q8_0_BLOCK_BYTES)
    # fp16 scale (first 2 bytes per block). copy() -> contiguous -> view as fp16.
    d = arr[:, :2].copy().view(np.float16).astype(np.float32).reshape(n_blocks, 1)
    # int8 values (next 32 bytes). view as signed int8.
    qs = arr[:, 2:].copy().view(np.int8).astype(np.float32)
    return (qs * d).reshape(-1)  # flat float32 array of length n_elements


def fp32_to_bf16_bytes(arr_f32: np.ndarray) -> bytes:
    """Convert float32 numpy array -> bf16 bit patterns as little-endian bytes.

    bf16 is the top 16 bits of fp32 with round-to-nearest-even. We do the
    rounding in uint32 space, then pack to uint16. This bit-exactly matches
    torch.tensor.to(torch.bfloat16) for finite values.
    """
    if arr_f32.dtype != np.float32:
        arr_f32 = arr_f32.astype(np.float32)
    u32 = arr_f32.view(np.uint32)
    # Round-to-nearest-even: add 0x7FFF plus the LSB of the resulting bf16
    # (which is bit 16 of the fp32). This is the canonical RNE shortcut.
    lsb = (u32 >> 16) & 1
    rounded = u32 + np.uint32(0x7FFF) + lsb.astype(np.uint32)
    bf16 = (rounded >> np.uint32(16)).astype(np.uint16)
    return bf16.tobytes()  # little-endian by default


def tensor_bf16_bytes(raw: bytes, gguf_type: int, dims: List[int]
                      ) -> Tuple[bytes, Tuple[int, ...]]:
    """Dequantize/cast a GGUF tensor to bf16 bytes + return the PyTorch shape.

    PyTorch shape is the reversed GGUF dims (ne[0]=contiguous becomes the
    innermost/trailing dim of a row-major layout, so a 2D GGUF tensor with
    dims=[in, out] becomes the PyTorch [out, in] Linear weight convention).
    """
    n_elems = int(np.prod(dims)) if dims else 0
    if gguf_type == GGUF_TYPE_Q8_0:
        arr_f32 = dequant_q8_0_to_f32(raw, n_elems)
    elif gguf_type == GGUF_TYPE_F32:
        arr_f32 = np.frombuffer(raw, dtype=np.float32, count=n_elems).copy()
    elif gguf_type == GGUF_TYPE_F16:
        arr_f32 = np.frombuffer(raw, dtype=np.float16, count=n_elems
                                ).astype(np.float32)
    elif gguf_type == GGUF_TYPE_BF16:
        # Already bf16 bits; pass through.
        out_shape = tuple(reversed(dims))
        if len(raw) < n_elems * 2:
            raise ValueError(
                f"BF16 tensor truncated: have {len(raw)}, need {n_elems*2}")
        return raw[:n_elems * 2], out_shape
    else:
        raise ValueError(f"Unsupported GGUF type {gguf_type}")

    out_shape = tuple(reversed(dims))  # PyTorch [out, in] (or just shape for 1D)
    return fp32_to_bf16_bytes(arr_f32), out_shape


# ----------------------------------------------------------------------------
# GGUF access wrapper (mmap).
# ----------------------------------------------------------------------------
class GgufView:
    def __init__(self, gguf_path: str) -> None:
        self.path = gguf_path
        self.fh = open(gguf_path, "rb")
        self.mm = mmap.mmap(self.fh.fileno(), 0, prot=mmap.PROT_READ)
        header_buf = bytes(self.mm[:64 * 1024 * 1024])
        tensors, data_off, align = parse_gguf_header(header_buf)
        self.tensors_by_name = {t.name: t for t in tensors}
        self.data_off = data_off
        self.alignment = align
        print(f"[gguf] opened {gguf_path} "
              f"({os.path.getsize(gguf_path) / 1e9:.2f} GB, "
              f"{len(tensors)} tensors, data_off={data_off}, align={align})")

    def tensor_abs_off(self, name: str) -> int:
        return self.data_off + self.tensors_by_name[name].offset

    def read_tensor_raw(self, name: str) -> bytes:
        t = self.tensors_by_name[name]
        off = self.tensor_abs_off(name)
        # Compute actual byte count from type+dims (the GGUF directory stores
        # nbytes for known types already, but recompute for safety).
        n_elems = int(np.prod(t.dims)) if t.dims else 0
        if t.gguf_type == GGUF_TYPE_Q8_0:
            nbytes = (n_elems // QK8_0) * Q8_0_BLOCK_BYTES
        elif t.gguf_type == GGUF_TYPE_F32:
            nbytes = n_elems * 4
        elif t.gguf_type == GGUF_TYPE_F16:
            nbytes = n_elems * 2
        elif t.gguf_type == GGUF_TYPE_BF16:
            nbytes = n_elems * 2
        else:
            raise ValueError(f"Unsupported GGUF type {t.gguf_type} for {name}")
        if nbytes != t.nbytes and t.nbytes != 0:
            raise ValueError(
                f"{name}: recomputed nbytes={nbytes} != directory nbytes={t.nbytes}")
        return bytes(self.mm[off:off + nbytes])

    def close(self) -> None:
        try:
            self.mm.close()
        finally:
            self.fh.close()


# ----------------------------------------------------------------------------
# Minimal streaming safetensors writer (reused pattern from IQ2 converter).
# ----------------------------------------------------------------------------
_NUMPY_DTYPE_NAME: List[Tuple[np.dtype, str]] = [
    (np.dtype(np.bool_), "BOOL"),
    (np.dtype(np.uint8), "U8"),   (np.dtype(np.int8), "I8"),
    (np.dtype(np.uint16), "U16"), (np.dtype(np.int16), "I16"),
    (np.dtype(np.uint32), "U32"), (np.dtype(np.int32), "I32"),
    (np.dtype(np.uint64), "U64"), (np.dtype(np.int64), "I64"),
    (np.dtype(np.float16), "F16"), (np.dtype(np.float32), "F32"),
    (np.dtype(np.float64), "F64"),
]


def _bf16_dtype_str() -> str:
    """Return the safetensors dtype name for bf16."""
    return "BF16"


WriterFn = Callable[[Any], None]


def _make_bytes_writer(b: bytes) -> WriterFn:
    def _w(out_fh) -> None:
        out_fh.write(b)
    return _w


def write_safetensors_streaming(
    out_path: str,
    entries: List[Dict],
    metadata: Dict[str, str],
    progress_cb: Callable[[Dict, int, int, float], None] | None = None,
) -> int:
    """Stream ``entries`` into ``out_path`` as a safetensors file.

    Each entry is a dict with keys:
        name   : str              -- tensor key in the safetensors
        dtype_str : str           -- safetensors dtype name (e.g. "BF16")
        shape  : tuple of ints    -- tensor shape
        bytes_ : int              -- total byte count
        writer : callable(out_fh) -- writes the bytes; fh is pre-seeked
    """
    offset = 0
    for e in entries:
        e["_start"] = offset
        e["_end"] = offset + e["bytes_"]
        offset += e["bytes_"]
    total_data = offset

    header: Dict[str, Any] = {"__metadata__": metadata}
    for e in entries:
        header[e["name"]] = {
            "dtype": e["dtype_str"],
            "shape": list(e["shape"]),
            "data_offsets": [e["_start"], e["_end"]],
        }
    header_json = json.dumps(header, separators=(",", ":"))
    header_bytes = header_json.encode("utf-8")
    pad = (8 - (len(header_bytes) % 8)) % 8
    header_bytes += b" " * pad
    preamble = struct.pack("<Q", len(header_bytes)) + header_bytes
    preamble_len = len(preamble)
    total_size = preamble_len + total_data

    print(f"[st] preamble {preamble_len} B (header JSON {len(header_json)} B + "
          f"{pad} B pad); data section {total_data} B; total {total_size} B "
          f"({total_size / 1e9:.2f} GB)")
    with open(out_path, "wb") as f:
        f.truncate(total_size)
        f.seek(0)
        f.write(preamble)

    written = 0
    t0 = time.time()
    with open(out_path, "r+b") as f:
        for e in entries:
            f.seek(preamble_len + e["_start"])
            e["writer"](f)
            written += e["bytes_"]
            if progress_cb is not None:
                progress_cb(e, written, total_data, t0)

    with open(out_path, "r+b") as f:
        f.flush()
        os.fsync(f.fileno())
    return total_size


# ----------------------------------------------------------------------------
# Build the entries for the converter.
# ----------------------------------------------------------------------------
# Per-tensor expected PyTorch shape (for sanity-checks against the dequant).
EXPECTED_PT_SHAPE: Dict[str, Tuple[int, ...]] = {
    "attn.wq_a.weight":            (Q_LORA, HIDDEN),                   # (1024, 4096)
    "attn.wkv.weight":             (HEAD_DIM, HIDDEN),                 # (512, 4096)
    "attn.wq_b.weight":            (N_HEADS * HEAD_DIM, Q_LORA),       # (32768, 1024)
    "attn.wo_a.weight":            (N_GROUPS * O_LORA, N_HEADS * HEAD_DIM // N_GROUPS),
    "attn.wo_b.weight":            (HIDDEN, N_GROUPS * O_LORA),        # (4096, 8192)
    "ffn.shared_experts.w1.weight": (INTERMEDIATE, HIDDEN),             # (2048, 4096)
    "ffn.shared_experts.w3.weight": (INTERMEDIATE, HIDDEN),
    "ffn.shared_experts.w2.weight": (HIDDEN, INTERMEDIATE),             # (4096, 2048)
}

# Per-GGUF-tensor-name expected type.
GGUF_NAME_TO_TYPE: Dict[str, int] = {
    "attn_q_a.weight":         GGUF_TYPE_Q8_0,
    "attn_kv.weight":          GGUF_TYPE_Q8_0,
    "attn_q_b.weight":         GGUF_TYPE_Q8_0,
    "attn_output_a.weight":    GGUF_TYPE_Q8_0,
    "attn_output_b.weight":    GGUF_TYPE_Q8_0,
    "ffn_gate_shexp.weight":   GGUF_TYPE_Q8_0,
    "ffn_up_shexp.weight":     GGUF_TYPE_Q8_0,
    "ffn_down_shexp.weight":   GGUF_TYPE_Q8_0,
    "output.weight":           GGUF_TYPE_Q8_0,
}


def build_entries(gguf: GgufView, n_layers: int = N_LAYERS,
                  target_set: str = "all") -> List[Dict]:
    """Build one safetensors entry per extracted non-MoE weight.

    target_set selects which weights to extract:
      "all"     -- all 8 attn Linears + shexp + lm_head.
      "mla_qkv" -- only the MLA Q/K/V projections (wq_a, wkv, wq_b); the
                   minimum viable hypothesis test. The other tensors (wo_a,
                   wo_b, shexp, lm_head) hit DSv4 custom fp8 kernels (o_proj,
                   tilelang) that hardcode fp8 weight_scale_inv access.
    """
    bf16_str = _bf16_dtype_str()
    entries: List[Dict] = []
    if target_set not in TARGET_SETS:
        sys.exit(f"[convert] unknown --targets {target_set!r}; "
                 f"pick one of {list(TARGET_SETS)}")
    wanted = TARGET_SETS[target_set]

    # Per-layer non-MoE Linears.
    for L in range(n_layers):
        for gguf_tail, vllm_tail in GGUF_TO_VLLM_LAYER_MAP.items():
            if vllm_tail not in wanted:
                continue
            gguf_name = f"blk.{L}.{gguf_tail}"
            vllm_name = f"layers.{L}.{vllm_tail}"
            if gguf_name not in gguf.tensors_by_name:
                sys.exit(f"[convert] missing GGUF tensor {gguf_name}")
            t = gguf.tensors_by_name[gguf_name]
            expect_type = GGUF_NAME_TO_TYPE[gguf_tail]
            if t.gguf_type != expect_type:
                sys.exit(f"[convert] {gguf_name}: type {t.gguf_type} != expected {expect_type}")
            expect_pt_shape = EXPECTED_PT_SHAPE[vllm_tail]
            raw = gguf.read_tensor_raw(gguf_name)
            bf16, pt_shape = tensor_bf16_bytes(raw, t.gguf_type, t.dims)
            if tuple(pt_shape) != expect_pt_shape:
                sys.exit(
                    f"[convert] {gguf_name}: dequant shape {pt_shape} != "
                    f"expected {expect_pt_shape} (gguf dims={t.dims})")
            entries.append({
                "name": vllm_name,
                "dtype_str": bf16_str,
                "shape": pt_shape,
                "bytes_": len(bf16),
                "writer": _make_bytes_writer(bf16),
                "_src": gguf_name,
            })

    # LM head (top-level GGUF tensor).
    if "head.weight" in wanted:
        gguf_name = "output.weight"
        vllm_name = "head.weight"
        if gguf_name not in gguf.tensors_by_name:
            sys.exit(f"[convert] missing GGUF tensor {gguf_name}")
        t = gguf.tensors_by_name[gguf_name]
        if t.gguf_type != GGUF_TYPE_Q8_0:
            sys.exit(f"[convert] {gguf_name}: type {t.gguf_type} != Q8_0")
        raw = gguf.read_tensor_raw(gguf_name)
        bf16, pt_shape = tensor_bf16_bytes(raw, t.gguf_type, t.dims)
        expect_pt_shape = (VOCAB, HIDDEN)
        if tuple(pt_shape) != expect_pt_shape:
            sys.exit(
                f"[convert] {gguf_name}: dequant shape {pt_shape} != expected {expect_pt_shape}")
        entries.append({
            "name": vllm_name,
            "dtype_str": bf16_str,
            "shape": pt_shape,
            "bytes_": len(bf16),
            "writer": _make_bytes_writer(bf16),
            "_src": gguf_name,
        })
    return entries


def build_metadata(gguf_path: str) -> Dict[str, str]:
    return {
        "format": "ds4-nonmoe-q8-bf16-v1",
        "source_gguf": os.path.abspath(gguf_path),
        "model": "DeepSeek-V4-Flash",
        "n_layers": str(N_LAYERS),
        "description": (
            "Non-MoE Linear weights dequantized from antirez's Q8_0 GGUF to "
            "bf16. Loaded via Fp8Config.ignored_layers + safetensors index "
            "remap. Hypothesis: fp8 attention causes 16K needle failure."
        ),
    }


# ----------------------------------------------------------------------------
# Update model.safetensors.index.json so the loader picks up the bf16 file.
# ----------------------------------------------------------------------------
def update_index(
    idx_path: str,
    bf16_shard_name: str,
    bf16_keys: List[str],
    bf16_size: int,
    drop_scales: bool = True,
) -> None:
    """Rewrite the index so each bf16 key points at the new shard.

    With ``drop_scales=True`` also remove the matching ``.scale`` siblings
    (they're unused once the layer is in ignored_layers; leaving them in the
    index is harmless but clutters diagnostic listings).
    """
    with open(idx_path) as f:
        idx = json.load(f)
    wm = idx.get("weight_map", {})
    affected_files = set(wm.values())

    redirected = 0
    scales_dropped = 0
    for key in bf16_keys:
        if key in wm:
            redirected += 1
        wm[key] = bf16_shard_name
        if drop_scales:
            scale_key = key[:-len(".weight")] + ".scale"
            if scale_key in wm:
                del wm[scale_key]
                scales_dropped += 1

    # Bump total_size by the new shard's size (approximate; the unused fp8
    # bytes in the original shards remain counted, which is fine for vLLM's
    # loader -- it doesn't assert against total_size).
    meta = idx.get("metadata", {})
    meta["total_size"] = meta.get("total_size", 0) + bf16_size
    idx["metadata"] = meta
    idx["weight_map"] = wm

    # Re-marshal pretty (compact-ish) and atomically swap.
    tmp = idx_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(idx, f, separators=(",", ":"))
    os.replace(tmp, idx_path)
    print(f"[index] {idx_path}: redirected {redirected} keys to "
          f"{bf16_shard_name}; dropped {scales_dropped} .scale siblings; "
          f"total_size += {bf16_size}")


# ----------------------------------------------------------------------------
# ignored_layers list for config.json (printed + optionally applied).
# ----------------------------------------------------------------------------
def build_ignored_layers() -> List[str]:
    """Per-layer prefixes for Fp8Config.ignored_layers (post-mapper names).

    The model's weights_mapper maps ``layers.`` -> ``model.layers.`` and
    ``head.weight`` -> ``lm_head.weight``. Fp8Config.apply_vllm_mapper applies
    the mapper to ignored_layers too, so we could use either form; we emit the
    post-mapper form to be explicit.
    """
    ignored: List[str] = []
    for L in range(N_LAYERS):
        # fused_wqa_wkv is a MergedColumnParallelLinear whose shards are
        # wq_a + wkv; is_layer_skipped matches on the fused prefix when the
        # whole module is listed.
        ignored.append(f"model.layers.{L}.attn.fused_wqa_wkv")
        ignored.append(f"model.layers.{L}.attn.wq_b")
        ignored.append(f"model.layers.{L}.attn.wo_a")
        ignored.append(f"model.layers.{L}.attn.wo_b")
        # shared_experts MLP: MergedColumnParallelLinear (gate_up_proj) +
        # RowParallelLinear (down_proj). The checkpoint stores w1/w3 (which
        # the loader fuses into gate_up_proj) and w2 (mapped to down_proj).
        ignored.append(f"model.layers.{L}.ffn.shared_experts.gate_up_proj")
        ignored.append(f"model.layers.{L}.ffn.shared_experts.down_proj")
    ignored.append("lm_head")
    return ignored


def update_config_json(
    config_path: str, ignored_layers: List[str], backup: bool = True,
) -> None:
    """Insert/merge ignored_layers into config.json's quantization_config."""
    with open(config_path) as f:
        cfg = json.load(f)
    qc = cfg.get("quantization_config", {})
    if "ignored_layers" in qc:
        existing = set(qc["ignored_layers"])
        new = [x for x in ignored_layers if x not in existing]
        if not new:
            print(f"[config] {config_path}: ignored_layers already complete "
                  f"({len(existing)} entries)")
            return
        qc["ignored_layers"] = list(existing) | set(ignored_layers)
    else:
        qc["ignored_layers"] = list(ignored_layers)
    cfg["quantization_config"] = qc

    if backup:
        bak = config_path + ".bak"
        if not os.path.exists(bak):
            os.replace(config_path, bak)
            # Write the new content to the original path.
            with open(config_path, "w") as f:
                json.dump(cfg, f, indent=2)
            print(f"[config] {config_path}: backed up to {bak}; "
                  f"added {len(ignored_layers)} ignored_layers entries")
        else:
            with open(config_path, "w") as f:
                json.dump(cfg, f, indent=2)
            print(f"[config] {config_path}: backup {bak} exists; "
                  f"added {len(ignored_layers)} ignored_layers entries")
    else:
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"[config] {config_path}: added {len(ignored_layers)} ignored_layers entries")


# ----------------------------------------------------------------------------
# Round-trip validation.
# ----------------------------------------------------------------------------
def validate_against_gguf(
    safetensors_path: str, gguf: GgufView, layers_to_check: List[int],
    target_set: str = "all",
) -> bool:
    """For each layer: dequant from GGUF matches the bf16 in the safetensors
    (within bf16 rounding). Uses safetensors.torch which understands bf16.
    """
    try:
        from safetensors import safe_open
    except ImportError as e:
        print(f"[validate] safetensors not available: {e}; skipping")
        return True

    if target_set not in TARGET_SETS:
        sys.exit(f"[validate] unknown target_set {target_set!r}")
    wanted = TARGET_SETS[target_set]

    print(f"[validate] checking layers {layers_to_check} (target_set={target_set})")
    ok = True

    with safe_open(safetensors_path, framework="pt", device="cpu") as sf:
        keys = list(sf.keys())
        print(f"[validate] safetensors has {len(keys)} tensors")

        for L in layers_to_check:
            for gguf_tail, vllm_tail in GGUF_TO_VLLM_LAYER_MAP.items():
                if vllm_tail not in wanted:
                    continue
                gguf_name = f"blk.{L}.{gguf_tail}"
                vllm_name = f"layers.{L}.{vllm_tail}"
                if vllm_name not in keys:
                    print(f"[validate] FAIL: missing {vllm_name}")
                    ok = False
                    continue
                # Re-dequant the GGUF to f32 (ground truth).
                t = gguf.tensors_by_name[gguf_name]
                raw = gguf.read_tensor_raw(gguf_name)
                f32_ref_flat = dequant_q8_0_to_f32(
                    raw, int(np.prod(t.dims))).astype(np.float32)
                expected_shape = EXPECTED_PT_SHAPE[vllm_tail]
                f32_ref = f32_ref_flat.reshape(expected_shape)

                # Read back as bf16 torch tensor, convert to f32 numpy.
                t_stored = sf.get_tensor(vllm_name)
                if t_stored.dtype != torch.bfloat16:
                    print(f"[validate] FAIL: {vllm_name} dtype {t_stored.dtype} "
                          f"!= bf16")
                    ok = False
                    continue
                if tuple(t_stored.shape) != expected_shape:
                    print(f"[validate] FAIL: {vllm_name} shape "
                          f"{tuple(t_stored.shape)} != {expected_shape}")
                    ok = False
                    continue
                f32_stored = t_stored.float().numpy()

                # bf16 has ~7-bit mantissa. Q8_0 has fp16 scale (exact) +
                # int8 quant (exact within the amax/127 quantization). The
                # combined rel err is bounded by ~2^-7 (bf16) + ~1/127
                # (Q8_0) ~ 1.6%. We allow 3% head-room for accumulation.
                max_abs = float(np.max(np.abs(f32_ref))) + 1e-30
                rel = float(np.max(np.abs(f32_ref - f32_stored)) / max_abs)
                if rel > 3e-2:
                    print(f"[validate] FAIL: {vllm_name} rel err {rel:.4e}")
                    ok = False
                else:
                    print(f"[validate] OK {vllm_name}: rel err {rel:.4e} "
                          f"(max_abs={max_abs:.4f})")

        # Also check the lm_head.
        gguf_name = "output.weight"
        vllm_name = "head.weight"
        if "head.weight" not in wanted:
            return ok
        if vllm_name in keys and gguf_name in gguf.tensors_by_name:
            t = gguf.tensors_by_name[gguf_name]
            raw = gguf.read_tensor_raw(gguf_name)
            f32_ref = dequant_q8_0_to_f32(
                raw, int(np.prod(t.dims))).astype(np.float32).reshape(VOCAB, HIDDEN)
            t_stored = sf.get_tensor(vllm_name).float().numpy()
            max_abs = float(np.max(np.abs(f32_ref))) + 1e-30
            rel = float(np.max(np.abs(f32_ref - t_stored)) / max_abs)
            tag = "OK" if rel <= 3e-2 else "FAIL"
            if tag == "FAIL":
                ok = False
            print(f"[validate] {tag} {vllm_name}: rel err {rel:.4e} "
                  f"(max_abs={max_abs:.4f})")
    return ok


# ----------------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------------
def _progress(e: Dict, written: int, total: int, t0: float) -> None:
    elapsed = time.time() - t0
    rate = written / elapsed if elapsed > 0 else 0.0
    pct = 100.0 * written / total if total > 0 else 0.0
    print(f"[st]   wrote {written:>13,d} / {total:,d} B  "
          f"({pct:5.1f}% @ {rate / 1e6:6.1f} MB/s)  -- "
          f"{e['name']}  <- {e.get('_src', '')}")


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--gguf", default="/root/antirez/ds4/ds4flash.gguf",
                   help="Path to the antirez ds4flash GGUF.")
    p.add_argument("--out", required=True,
                   help="Output safetensors path (will be created/overwritten).")
    p.add_argument("--model-dir", default=None,
                   help="Directory containing model.safetensors.index.json + "
                        "config.json (default: same dir as --out).")
    p.add_argument("--limit-layers", type=int, default=None,
                   help="Convert only the first N layers (default: all 43).")
    p.add_argument("--targets", default="all",
                   choices=sorted(TARGET_SETS.keys()),
                   help="Which tensors to extract: 'all' (8 attn + shexp + "
                        "lm_head) or 'mla_qkv' (just wq_a/wkv/wq_b).")
    p.add_argument("--no-validate", action="store_true",
                   help="Skip the GGUF round-trip validation.")
    p.add_argument("--no-update-index", action="store_true",
                   help="Do not rewrite model.safetensors.index.json.")
    p.add_argument("--no-update-config", action="store_true",
                   help="Do not insert ignored_layers into config.json.")
    p.add_argument("--keep-scales", action="store_true",
                   help="Leave the .scale siblings in the index (default: drop).")
    p.add_argument("--print-ignored", action="store_true",
                   help="Print the ignored_layers list as JSON and exit.")
    if argv is None:
        argv = sys.argv[1:]
    args = p.parse_args(argv)

    if args.print_ignored:
        print(json.dumps(build_ignored_layers(), indent=2))
        return 0

    if not os.path.exists(args.gguf):
        sys.exit(f"[convert] GGUF not found: {args.gguf}")
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    model_dir = args.model_dir or out_dir

    # Disk-space pre-check.
    st = os.statvfs(out_dir)
    free_gb = (st.f_bavail * st.f_frsize) / 1e9
    # Each layer has 8 Q8_0 tensors. Approximate bf16 size as elems*2.
    # Per layer: ~107M elems; lm_head ~530M elems.
    n_layers = args.limit_layers if args.limit_layers else N_LAYERS
    # Quick estimate: 9.2 GB attn (43 layers) + 2.16 GB shexp + 1.06 GB lm_head
    # = 12.4 GB for all 43. Scale per layer.
    per_layer_gb = (9.2 + 2.16) / N_LAYERS
    est_gb = per_layer_gb * n_layers + 1.06
    print(f"[convert] planning {n_layers} layer(s) -> ~{est_gb:.2f} GB; "
          f"{free_gb:.1f} GB free on {out_dir}")
    if est_gb + 5 > free_gb:
        sys.exit(f"[convert] insufficient disk: need ~{est_gb:.1f} GB, "
                 f"have {free_gb:.1f} GB free; pass --limit-layers N to "
                 f"convert fewer layers.")

    gguf = GgufView(args.gguf)
    try:
        n_layers = args.limit_layers if args.limit_layers is not None else N_LAYERS
        entries = build_entries(gguf, n_layers=n_layers, target_set=args.targets)
        metadata = build_metadata(args.gguf)

        total_size = write_safetensors_streaming(
            args.out, entries, metadata, progress_cb=_progress)

        actual_size = os.path.getsize(args.out)
        print(f"[convert] DONE. wrote {actual_size:,d} B "
              f"({actual_size / 1e9:.2f} GB); header reported {total_size:,d} B; "
              f"match={actual_size == total_size}")
        if actual_size != total_size:
            print("[convert] FAIL: size mismatch")
            return 1

        if not args.no_validate:
            n_val = min(n_layers, 3)
            val_layers = list(range(n_val))
            if n_layers - 1 not in val_layers and n_layers > 0:
                val_layers.append(n_layers - 1)
            ok = validate_against_gguf(args.out, gguf, val_layers,
                                       target_set=args.targets)
            if not ok:
                print("[convert] VALIDATION FAILED")
                return 2
            print("[convert] validation passed (bf16 round-trip within tolerance)")

        bf16_keys = [e["name"] for e in entries]

        # Update the index.
        if not args.no_update_index:
            idx_path = os.path.join(model_dir, "model.safetensors.index.json")
            if os.path.exists(idx_path):
                update_index(
                    idx_path, os.path.basename(args.out), bf16_keys,
                    actual_size, drop_scales=not args.keep_scales,
                )
            else:
                print(f"[convert] index {idx_path} not found; skipping update")

        # Update config.json with ignored_layers.
        if not args.no_update_config:
            cfg_path = os.path.join(model_dir, "config.json")
            if os.path.exists(cfg_path):
                update_config_json(cfg_path, build_ignored_layers())
            else:
                print(f"[convert] config {cfg_path} not found; skipping update")
        return 0
    finally:
        gguf.close()


if __name__ == "__main__":
    raise SystemExit(main())
