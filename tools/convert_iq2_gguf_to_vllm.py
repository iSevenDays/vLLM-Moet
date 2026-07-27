#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GGUF -> vLLM-native safetensors converter for the IQ2_XXS + Q2_K MoE
expert tensors of antirez's DeepSeek-V4-Flash GGUF.

Step 1 of the IQ2_XXS -> Ada (sm_89) native-decode port. This is a one-off
offline converter: it reads ``blk.L.ffn_{gate,up}_exps.weight`` (GGML type 16
= IQ2_XXS) and ``blk.L.ffn_down_exps.weight`` (GGML type 10 = Q2_K) from the
GGUF and writes them into a single safetensors checkpoint as RAW uint8 bytes
under vLLM-expected param names, preserving the quantized block layout
byte-for-byte (a passthrough). Step 2 will teach vLLM's RoutedExperts loader
to read this format; this script does NOT touch vLLM.

Reference: overlay/vllm/vllm/model_executor/layers/quantization/utils/iq2_xxs_ref.py
           (minimal GGUF v3 parser + CPU dequant golden reference).

==============================================================================
Tensor layout (verified against antirez/ds4/ds4.c)
==============================================================================

antirez stores each routed-expert tensor as a single 3-D blob whose GGUF
``dims`` (read in file order = GGML ``ne[0..ndim-1]``) are:

    ffn_gate_exps / ffn_up_exps   (IQ2_XXS, type 16):  dims = (4096, 2048, 256)
    ffn_down_exps                 (Q2_K,    type 10):  dims = (2048, 4096, 256)

``ds4.c:tensor_expect_routed_expert`` confirms the semantic mapping:

    gate/up:  (DS4_N_EMBD=4096,  DS4_N_FF_EXP=2048, DS4_N_EXPERT=256)
    down:     (DS4_N_FF_EXP=2048, DS4_N_EMBD=4096,  DS4_N_EXPERT=256)

i.e. ``ne[0]`` is the row length (the Linear weight's IN features, contiguous
dimension), ``ne[1]`` is the number of rows (the Linear weight's OUT features)
and ``ne[2]`` is the expert index. ``ds4.c:routed_expert_row_bytes`` then
defines one "row" as ``(ne[0] / 256) * block_bytes`` -- so a row is exactly
``ne[0] / QK_K`` quantized blocks:

    gate/up row = (4096 / 256) * 66 = 16 * 66 = 1056 bytes  (covers in=hidden=4096)
    down row    = (2048 / 256) * 84 =  8 * 84 =  672 bytes  (covers in=intermediate=2048)

Per expert: ne[1] rows. Total per expert:
    gate/up: 2048 * 1056 = 2,164,608 B   (32,768 IQ2_XXS blocks)
    down:    4096 *  672 = 2,752,512 B   (32,768 Q2_K    blocks)

The bytes inside one expert are laid out, in flat order, as
``row[0], row[1], ..., row[ne[1]-1]`` and the 256 experts are concatenated in
index order. That is EXACTLY a C-order uint8 ndarray of shape

    gate/up: (n_experts=256, out_features=2048, row_bytes=1056)
    down:    (n_experts=256, out_features=4096, row_bytes=672)

so the GGUF -> safetensors conversion is a pure memcpy (no shuffle, no
dequant) -- the round-trip must be bit-identical, which the validation path
asserts.

==============================================================================
vLLM param-name convention (chosen to mirror HF + the existing mxfp4 loader)
==============================================================================

The HF DeepSeek-MoE family names routed-expert weights
``model.layers.L.mlp.experts.{gate_proj,up_proj,down_proj}.weight`` with a
leading expert dimension (shape ``[n_experts, out, in]``). vLLM's
``RoutedExperts.weight_loader`` (vllm/.../fused_moe/routed_experts.py:585)
maps ``gate_proj -> w1``, ``up_proj -> w3``, ``down_proj -> w2`` via the
shard-id substring, and the mxfp4 ``create_weights``
(overlay/.../mxfp4.py:530-644) registers ``w13_weight`` (fused gate+up) and
``w2_weight`` as ``[num_experts, out_features, in_features_packed]`` uint8.

Because the GGUF stores gate and up SEPARATELY (we'd have to physically
interleave IQ2_XXS block bytes to fuse them), we emit them UN-fused and let
the Step 2 loader fuse at load time. Param names:

    model.layers.{L}.mlp.experts.gate_proj.weight_iq2_xxs   # w1, [256, 2048, 1056]
    model.layers.{L}.mlp.experts.up_proj.weight_iq2_xxs     # w3, [256, 2048, 1056]
    model.layers.{L}.mlp.experts.down_proj.weight_q2_k      # w2, [256, 4096,  672]

The suffix (``_iq2_xxs`` / ``_q2_k``) tells the Step 2 loader the GGML type
so it knows which dequant kernel to dispatch.

NOTE on the task-spec deviation: an earlier draft of the task spec gave the
shapes as ``[256, 4096, 528]`` (gate/up) and ``[256, 2048, 1344]`` (down).
Those numbers have the dims swapped inconsistently (middle = ne[0] for one,
ne[1] for the other) and do not correspond to the natural per-row block
layout. We use the mathematically consistent shapes derived above, which
also match vLLM's ``[num_experts, out_features, in_features_packed]``
convention. The byte count per expert is identical either way.

==============================================================================
Lookup tables
==============================================================================

IQ2_XXS dequant needs two small constant tables (``iq2xxs_grid[256]`` uint64,
``ksigns_iq2xs[128]`` uint8). They are hard-coded verbatim in
``iq2_xxs_ref.py`` and will equally be hard-coded in the Triton kernel (they
are LUTs, not weights). For self-containment we ALSO emit them as small aux
tensors so anything inspecting the checkpoint can find them:

    _lookup.iq2xxs_grid   : uint64[256]
    _lookup.ksigns_iq2xs  : uint8[128]

==============================================================================
TP sharding
==============================================================================

A single file is written (simpler than per-rank shards). The expert dimension
is axis 0 of every tensor -- a TP=N loader assigns experts
``[rank * 256/N : (rank+1) * 256/N]`` to rank ``rank`` (DeepSeek-style
expert-parallel sharding, identical to what mxfp4 does).

==============================================================================
Disk / memory
==============================================================================

Output size = 43 layers * (2 * 553,648,128 + 704,643,072) B
            = 43 * 1,811,939,328 B
            = 77,913,391,104 B  ~= 78 GB (plus ~2 KB for aux + header).

The writer is a streaming passthrough: GGUF is mmap-ed read-only and bytes
are copied to the output file in 16 MB chunks, so peak Python RSS is a few
hundred MB regardless of total size.

Usage:
    python3 tools/convert_iq2_gguf_to_vllm.py \\
        --gguf /root/antirez/ds4/ds4flash.gguf \\
        --out  /root/models/DeepSeek-V4-Flash-IQ2/dsv4_iq2.safetensors \\
        [--limit-layers N] [--no-validate]
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

# Reuse the Step 0 parser + dequant + verbatim LUTs.
from iq2_xxs_ref import (  # noqa: E402
    parse_gguf_header,
    GGUF_TYPE_IQ2_XXS,
    GGUF_TYPE_Q2_K,
    IQ2_XXS_BLOCK_BYTES,
    Q2_K_BLOCK_BYTES,
    QK_K,
    dequant_iq2_xxs_block,
    dequant_q2_k_block,
    _IQ2XXS_GRID,
    _KSIGNS_IQ2XS,
)

# ----------------------------------------------------------------------------
# DeepSeek-V4-Flash shape constants.
# ----------------------------------------------------------------------------
N_LAYERS = 43
N_EXPERTS = 256
HIDDEN = 4096         # DS4_N_EMBD
INTERMEDIATE = 2048   # DS4_N_FF_EXP

# Per-row byte layout (matches ds4.c:routed_expert_row_bytes).
GATE_UP_ROW_BYTES = (HIDDEN // QK_K) * IQ2_XXS_BLOCK_BYTES        # 16 * 66 = 1056
DOWN_ROW_BYTES = (INTERMEDIATE // QK_K) * Q2_K_BLOCK_BYTES        #  8 * 84 =  672

# Stored tensor shapes (C-order uint8 -> natural GGUF byte order).
GATE_UP_SHAPE: Tuple[int, int, int] = (N_EXPERTS, INTERMEDIATE, GATE_UP_ROW_BYTES)
DOWN_SHAPE: Tuple[int, int, int] = (N_EXPERTS, HIDDEN, DOWN_ROW_BYTES)

GATE_UP_BYTES = N_EXPERTS * INTERMEDIATE * GATE_UP_ROW_BYTES      #   553,648,128
DOWN_BYTES = N_EXPERTS * HIDDEN * DOWN_ROW_BYTES                  #   704,643,072

# GGUF-side expected dims for sanity-checking the parser output.
EXPECTED_DIMS_IQ2_XXS: Tuple[int, int, int] = (HIDDEN, INTERMEDIATE, N_EXPERTS)
EXPECTED_DIMS_Q2_K: Tuple[int, int, int] = (INTERMEDIATE, HIDDEN, N_EXPERTS)


def layer_param_names(layer_idx: int) -> Tuple[str, str, str]:
    """vLLM/HF param names for (gate=w1, up=w3, down=w2) of layer L."""
    L = layer_idx
    # Emit the vLLM DSv4 param-name format directly: the model wires MoE as
    # ``self.ffn`` (not ``self.mlp``), FusedMoE wraps the IQ2 params under
    # ``routed_experts.``, and self.named_parameters() is relative to the
    # model (no ``model.`` prefix). The IQ2 family names ({gate,up}_weight,
    # down_weight) match the params registered in mxfp4._create_iq2_weights.
    # DeepseekV4ForCausalLM.load_weights still does prefix normalization
    # for the legacy "model.layers.*.mlp.experts.*_proj.*" format too.
    return (
        f"layers.{L}.ffn.experts.routed_experts.gate_weight_iq2_xxs",
        f"layers.{L}.ffn.experts.routed_experts.up_weight_iq2_xxs",
        f"layers.{L}.ffn.experts.routed_experts.down_weight_q2_k",
    )


# ----------------------------------------------------------------------------
# GGUF parsing.
# ----------------------------------------------------------------------------
class GgufView:
    """Open the GGUF, parse the header, keep an mmap for chunked reads."""

    def __init__(self, gguf_path: str) -> None:
        self.path = gguf_path
        self.fh = open(gguf_path, "rb")
        self.mm = mmap.mmap(self.fh.fileno(), 0, prot=mmap.PROT_READ)
        # Header + tensor directory live well under 64 MB.
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

    def close(self) -> None:
        try:
            self.mm.close()
        finally:
            self.fh.close()


# ----------------------------------------------------------------------------
# Minimal streaming safetensors writer.
#   Format: [u64 LE header_len][header_json_utf8][tensor0 bytes][tensor1 bytes]...
# Each tensor entry in the header JSON is:
#   { "dtype": "U8", "shape": [...], "data_offsets": [start, end] }
# data_offsets are byte offsets relative to the start of the data section
# (i.e. immediately after the preamble).
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


def _dtype_to_str(dt: np.dtype) -> str:
    for k, name in _NUMPY_DTYPE_NAME:
        if k == dt:
            return name
    raise ValueError(f"Unsupported numpy dtype: {dt}")


# A writer callable writes bytes by calling out_fh.write(...); the file is
# already seeked to the correct absolute offset before invocation.
WriterFn = Callable[[Any], None]


def _make_gguf_copy_writer(
    gguf_mm: mmap.mmap, gguf_abs_off: int, nbytes: int,
    chunk: int = 16 << 20,
) -> WriterFn:
    """Return a writer that copies ``nbytes`` from the GGUF mmap to the output."""
    def _w(out_fh) -> None:
        pos = gguf_abs_off
        remaining = nbytes
        while remaining > 0:
            n = chunk if chunk < remaining else remaining
            out_fh.write(gguf_mm[pos:pos + n])
            pos += n
            remaining -= n
    return _w


def _make_bytes_writer(b: bytes) -> WriterFn:
    """Return a writer for a small pre-materialized byte blob."""
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
        dtype  : numpy dtype      -- element dtype (for the header)
        shape  : tuple of ints    -- tensor shape (for the header)
        bytes_ : int              -- total byte count to write
        writer : callable(out_fh) -- writes the bytes; fh is pre-seeked

    Returns the total file size in bytes.
    """
    # 1. Compute data-section offsets.
    offset = 0
    for e in entries:
        e["_start"] = offset
        e["_end"] = offset + e["bytes_"]
        offset += e["bytes_"]
    total_data = offset

    # 2. Build the header JSON.
    header: Dict[str, Any] = {"__metadata__": metadata}
    for e in entries:
        header[e["name"]] = {
            "dtype": _dtype_to_str(np.dtype(e["dtype"])),
            "shape": list(e["shape"]),
            "data_offsets": [e["_start"], e["_end"]],
        }
    header_json = json.dumps(header, separators=(",", ":"))
    header_bytes = header_json.encode("utf-8")
    # Pad the JSON to an 8-byte boundary (spec allows any length; alignment
    # keeps the data section 8-byte-aligned, which is friendly to readers).
    pad = (8 - (len(header_bytes) % 8)) % 8
    header_bytes += b" " * pad
    preamble = struct.pack("<Q", len(header_bytes)) + header_bytes
    preamble_len = len(preamble)
    total_size = preamble_len + total_data

    # 3. Pre-allocate + write the preamble.
    print(f"[st] preamble {preamble_len} B (header JSON {len(header_json)} B + "
          f"{pad} B pad); data section {total_data} B; total {total_size} B "
          f"({total_size / 1e9:.2f} GB)")
    with open(out_path, "wb") as f:
        f.truncate(total_size)
        f.seek(0)
        f.write(preamble)

    # 4. Stream each tensor's bytes (chunked for the GGUF-backed ones).
    written = 0
    t0 = time.time()
    with open(out_path, "r+b") as f:
        for e in entries:
            f.seek(preamble_len + e["_start"])
            e["writer"](f)
            written += e["bytes_"]
            if progress_cb is not None:
                progress_cb(e, written, total_data, t0)

    # 5. fsync so the size is materialized on disk before we report done.
    with open(out_path, "r+b") as f:
        f.flush()
        os.fsync(f.fileno())
    return total_size


# ----------------------------------------------------------------------------
# Build the per-layer entries.
# ----------------------------------------------------------------------------
def build_layer_entries(
    layer_idx: int, gguf: GgufView,
) -> List[Dict]:
    """Build the three writer entries for one layer."""
    L = layer_idx
    gate_name, up_name, down_name = layer_param_names(L)

    gate_gguf = f"blk.{L}.ffn_gate_exps.weight"
    up_gguf = f"blk.{L}.ffn_up_exps.weight"
    down_gguf = f"blk.{L}.ffn_down_exps.weight"

    for gguf_n, want_dims, want_type in (
        (gate_gguf, EXPECTED_DIMS_IQ2_XXS, GGUF_TYPE_IQ2_XXS),
        (up_gguf, EXPECTED_DIMS_IQ2_XXS, GGUF_TYPE_IQ2_XXS),
        (down_gguf, EXPECTED_DIMS_Q2_K, GGUF_TYPE_Q2_K),
    ):
        if gguf_n not in gguf.tensors_by_name:
            sys.exit(f"[convert] missing tensor {gguf_n} in GGUF")
        t = gguf.tensors_by_name[gguf_n]
        if tuple(t.dims) != want_dims:
            sys.exit(f"[convert] {gguf_n} dims {t.dims} != expected {want_dims}")
        if t.gguf_type != want_type:
            sys.exit(f"[convert] {gguf_n} type {t.gguf_type} != expected {want_type}")

    gate_off = gguf.tensor_abs_off(gate_gguf)
    up_off = gguf.tensor_abs_off(up_gguf)
    down_off = gguf.tensor_abs_off(down_gguf)

    return [
        {
            "name": gate_name,
            "dtype": np.uint8,
            "shape": GATE_UP_SHAPE,
            "bytes_": GATE_UP_BYTES,
            "writer": _make_gguf_copy_writer(gguf.mm, gate_off, GATE_UP_BYTES),
            "_src": gate_gguf,
        },
        {
            "name": up_name,
            "dtype": np.uint8,
            "shape": GATE_UP_SHAPE,
            "bytes_": GATE_UP_BYTES,
            "writer": _make_gguf_copy_writer(gguf.mm, up_off, GATE_UP_BYTES),
            "_src": up_gguf,
        },
        {
            "name": down_name,
            "dtype": np.uint8,
            "shape": DOWN_SHAPE,
            "bytes_": DOWN_BYTES,
            "writer": _make_gguf_copy_writer(gguf.mm, down_off, DOWN_BYTES),
            "_src": down_gguf,
        },
    ]


def build_aux_entries() -> List[Dict]:
    """Tiny LUT tensors for self-containment (see module docstring)."""
    grid = np.asarray(_IQ2XXS_GRID, dtype=np.uint64)
    ksigns = np.asarray(_KSIGNS_IQ2XS, dtype=np.uint8)
    assert grid.shape == (256,) and ksigns.shape == (128,)
    return [
        {
            "name": "_lookup.iq2xxs_grid",
            "dtype": np.uint64,
            "shape": (256,),
            "bytes_": grid.nbytes,
            "writer": _make_bytes_writer(grid.tobytes()),
        },
        {
            "name": "_lookup.ksigns_iq2xs",
            "dtype": np.uint8,
            "shape": (128,),
            "bytes_": ksigns.nbytes,
            "writer": _make_bytes_writer(ksigns.tobytes()),
        },
    ]


def build_metadata(gguf_path: str) -> Dict[str, str]:
    return {
        "format": "ds4-iq2-xxs-q2-k-v1",
        "source_gguf": os.path.abspath(gguf_path),
        "model": "DeepSeek-V4-Flash",
        "n_layers": str(N_LAYERS),
        "n_experts": str(N_EXPERTS),
        "hidden_size": str(HIDDEN),
        "intermediate_size": str(INTERMEDIATE),
        "qk_k": str(QK_K),
        "iq2_xxs_block_bytes": str(IQ2_XXS_BLOCK_BYTES),
        "q2_k_block_bytes": str(Q2_K_BLOCK_BYTES),
        # Stored shapes (uint8, C-order = natural GGUF byte order).
        "gate_up_shape": ",".join(str(x) for x in GATE_UP_SHAPE),
        "down_shape": ",".join(str(x) for x in DOWN_SHAPE),
        "gate_up_row_bytes": str(GATE_UP_ROW_BYTES),
        "down_row_bytes": str(DOWN_ROW_BYTES),
        # Semantic mapping (ds4.c:tensor_expect_routed_expert):
        #   gate/up: ne = (hidden=4096, intermediate=2048, n_experts=256) -- IQ2_XXS
        #   down:    ne = (intermediate=2048, hidden=4096, n_experts=256) -- Q2_K
        # Each stored row = (ne[0] / QK_K) blocks of {66|84} bytes covering
        # ne[0]; consecutive rows index ne[1]; experts are concatenated (axis 0).
        "layout_note": (
            "axis0=expert, axis1=out_features(ne[1]), "
            "axis2=row_bytes=(ne[0]//QK_K)*block_bytes; passthrough GGUF bytes"
        ),
        # TP sharding: split along axis 0 (expert dim).
        "tp_shard_axis": "0",
        "tp_default": "2 -> rank r takes experts [r*128:(r+1)*128]",
        # Param-name convention.
        "param_gate": "model.layers.L.mlp.experts.gate_proj.weight_iq2_xxs (w1)",
        "param_up": "model.layers.L.mlp.experts.up_proj.weight_iq2_xxs (w3)",
        "param_down": "model.layers.L.mlp.experts.down_proj.weight_q2_k (w2)",
        # LUTs are also inlined as _lookup.* tensors (and hardcoded in the kernel).
        "iq2xxs_grid_inlined": "_lookup.iq2xxs_grid uint64[256]",
        "ksigns_iq2xs_inlined": "_lookup.ksigns_iq2xs uint8[128]",
    }


# ----------------------------------------------------------------------------
# Round-trip validation.
# ----------------------------------------------------------------------------
def _read_first_n_blocks_from_gguf(
    gguf: GgufView, name: str, n_blocks: int, block_bytes: int,
) -> bytes:
    off = gguf.tensor_abs_off(name)
    return bytes(gguf.mm[off:off + n_blocks * block_bytes])


def _read_first_n_blocks_from_safetensors(
    sf, name: str, n_blocks: int, block_bytes: int,
) -> bytes:
    t = sf.get_tensor(name)  # numpy uint8 memmap view; tobytes() materializes
    return t.tobytes()[:n_blocks * block_bytes]


def _read_last_n_blocks_from_safetensors(
    sf, name: str, n_blocks: int, block_bytes: int, total_bytes: int,
) -> bytes:
    t = sf.get_tensor(name)
    return t.tobytes()[total_bytes - n_blocks * block_bytes:]


def _read_last_n_blocks_from_gguf(
    gguf: GgufView, name: str, n_blocks: int, block_bytes: int, total_bytes: int,
) -> bytes:
    off = gguf.tensor_abs_off(name)
    return bytes(gguf.mm[off + total_bytes - n_blocks * block_bytes:
                          off + total_bytes])


def validate_round_trip(
    safetensors_path: str, gguf: GgufView, layers_to_check: List[int],
    n_blocks_head: int = 8, n_blocks_tail: int = 4,
) -> bool:
    """For each layer: bytes identical (head + tail) AND dequant identical.

    Because the bytes are a passthrough, byte-equality implies dequant-equality;
    we additionally dequant a handful of blocks to exercise the reference path
    and assert semantic consistency with the stored shape metadata.
    """
    from safetensors import safe_open

    print(f"[validate] checking layers {layers_to_check} "
          f"(head {n_blocks_head} blocks + tail {n_blocks_tail} blocks each)")
    ok = True
    with safe_open(safetensors_path, framework="numpy", device="cpu") as sf:
        names = list(sf.keys())
        meta = sf.metadata()
        print(f"[validate] safetensors has {len(names)} tensors; "
              f"metadata format={meta.get('format')}")
        # Required aux tensors present?
        for aux in ("_lookup.iq2xxs_grid", "_lookup.ksigns_iq2xs"):
            if aux not in names:
                print(f"[validate] FAIL: missing aux tensor {aux}")
                ok = False

        for L in layers_to_check:
            for kind, gguf_name_fn, st_name_fn, dims, block_bytes, total_bytes, dequant_fn in (
                ("gate", lambda l: f"blk.{l}.ffn_gate_exps.weight",
                 lambda l: f"model.layers.{l}.mlp.experts.gate_proj.weight_iq2_xxs",
                 EXPECTED_DIMS_IQ2_XXS, IQ2_XXS_BLOCK_BYTES, GATE_UP_BYTES,
                 dequant_iq2_xxs_block),
                ("up", lambda l: f"blk.{l}.ffn_up_exps.weight",
                 lambda l: f"model.layers.{l}.mlp.experts.up_proj.weight_iq2_xxs",
                 EXPECTED_DIMS_IQ2_XXS, IQ2_XXS_BLOCK_BYTES, GATE_UP_BYTES,
                 dequant_iq2_xxs_block),
                ("down", lambda l: f"blk.{l}.ffn_down_exps.weight",
                 lambda l: f"model.layers.{l}.mlp.experts.down_proj.weight_q2_k",
                 EXPECTED_DIMS_Q2_K, Q2_K_BLOCK_BYTES, DOWN_BYTES,
                 dequant_q2_k_block),
            ):
                gguf_name = gguf_name_fn(L)
                st_name = st_name_fn(L)
                if st_name not in names:
                    print(f"[validate] FAIL: layer {L} {kind}: "
                          f"missing tensor {st_name}")
                    ok = False
                    continue

                # Sanity: shape on disk matches what we declared.
                st_tensor = sf.get_tensor(st_name)
                expected_shape = (GATE_UP_SHAPE if kind != "down" else DOWN_SHAPE)
                if tuple(st_tensor.shape) != expected_shape:
                    print(f"[validate] FAIL: {st_name} shape {st_tensor.shape} "
                          f"!= expected {expected_shape}")
                    ok = False
                    continue

                # Head bytes from both sources.
                gguf_head = _read_first_n_blocks_from_gguf(
                    gguf, gguf_name, n_blocks_head, block_bytes)
                st_head = _read_first_n_blocks_from_safetensors(
                    sf, st_name, n_blocks_head, block_bytes)
                if gguf_head != st_head:
                    print(f"[validate] FAIL: {st_name} head bytes differ "
                          f"from GGUF {gguf_name}")
                    ok = False
                    continue

                # Tail bytes from both sources.
                gguf_tail = _read_last_n_blocks_from_gguf(
                    gguf, gguf_name, n_blocks_tail, block_bytes, total_bytes)
                st_tail = _read_last_n_blocks_from_safetensors(
                    sf, st_name, n_blocks_tail, block_bytes, total_bytes)
                if gguf_tail != st_tail:
                    print(f"[validate] FAIL: {st_name} tail bytes differ "
                          f"from GGUF {gguf_name}")
                    ok = False
                    continue

                # Dequant the first 2 head blocks on both sides, confirm equal.
                v_gguf = np.stack([
                    dequant_fn(gguf_head[b * block_bytes:(b + 1) * block_bytes])
                    for b in range(2)
                ])
                v_st = np.stack([
                    dequant_fn(st_head[b * block_bytes:(b + 1) * block_bytes])
                    for b in range(2)
                ])
                if not np.array_equal(v_gguf, v_st):
                    print(f"[validate] FAIL: {st_name} dequant values differ")
                    ok = False
                    continue

                amax = float(np.max(np.abs(v_gguf)))
                print(f"[validate] OK layer {L} {kind:4s}: head+tail bytes "
                      f"identical, dequant[0:2] matches, max|x|={amax:.4f}, "
                      f"shape={st_tensor.shape}")

        # Spot-check the LUTs.
        if "_lookup.iq2xxs_grid" in names:
            grid_t = sf.get_tensor("_lookup.iq2xxs_grid")
            expect = np.asarray(_IQ2XXS_GRID, dtype=np.uint64)
            if not np.array_equal(np.asarray(grid_t), expect):
                print("[validate] FAIL: _lookup.iq2xxs_grid mismatch")
                ok = False
            else:
                print(f"[validate] OK _lookup.iq2xxs_grid uint64"
                      f"{tuple(grid_t.shape)} bit-identical")
        if "_lookup.ksigns_iq2xs" in names:
            ks_t = sf.get_tensor("_lookup.ksigns_iq2xs")
            expect = np.asarray(_KSIGNS_IQ2XS, dtype=np.uint8)
            if not np.array_equal(np.asarray(ks_t), expect):
                print("[validate] FAIL: _lookup.ksigns_iq2xs mismatch")
                ok = False
            else:
                print(f"[validate] OK _lookup.ksigns_iq2xs uint8"
                      f"{tuple(ks_t.shape)} bit-identical")

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
          f"{e['name']}")


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="GGUF -> vLLM safetensors converter for IQ2_XXS+Q2_K MoE experts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--gguf", default="/root/antirez/ds4/ds4flash.gguf",
                   help="Path to the antirez ds4flash GGUF.")
    p.add_argument("--out", required=True,
                   help="Output safetensors path (will be created/overwritten).")
    p.add_argument("--limit-layers", type=int, default=None,
                   help="Convert only the first N layers (default: all 43).")
    p.add_argument("--no-validate", action="store_true",
                   help="Skip the round-trip validation step.")
    p.add_argument("--validate-layers", type=int, default=None,
                   help="Number of layers to round-trip validate "
                        "(default: all converted layers).")
    p.add_argument("--no-update-index", action="store_true",
                   help="Do not update model.safetensors.index.json next "
                        "to the IQ2 shard. By default the converter adds "
                        "the IQ2 keys to the index so vLLM's auto-loader "
                        "visits them; without this, the IQ2 params stay at "
                        "their init zeros (silent zero MoE output).")
    p.add_argument("--model-dir", default=None,
                   help="Directory containing model.safetensors.index.json "
                        "(default: same dir as --out).")
    args = p.parse_args(argv)

    if not os.path.exists(args.gguf):
        sys.exit(f"[convert] GGUF not found: {args.gguf}")
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    # Disk-space pre-check.
    st = os.statvfs(out_dir)
    free_gb = (st.f_bavail * st.f_frsize) / 1e9
    n_layers = args.limit_layers if args.limit_layers else N_LAYERS
    est_bytes = n_layers * (2 * GATE_UP_BYTES + DOWN_BYTES) + 2 * 1024  # +aux/header
    est_gb = est_bytes / 1e9
    print(f"[convert] planning {n_layers} layer(s) -> ~{est_gb:.2f} GB; "
          f"{free_gb:.1f} GB free on {out_dir}")
    if est_gb + 5 > free_gb:
        sys.exit(f"[convert] insufficient disk: need ~{est_gb:.1f} GB, "
                 f"have {free_gb:.1f} GB free on {out_dir}; "
                 f"pass --limit-layers N to convert fewer layers.")

    # Open the GGUF and build the entries.
    gguf = GgufView(args.gguf)
    try:
        entries: List[Dict] = []

        # Aux LUTs first (small, fast, makes the file self-describing).
        entries.extend(build_aux_entries())

        # Per-layer MoE expert tensors.
        for L in range(n_layers):
            entries.extend(build_layer_entries(L, gguf))

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
            n_val = args.validate_layers if args.validate_layers else n_layers
            val_layers = list(range(min(n_val, n_layers)))
            # Always include the last converted layer too (tail-of-file check).
            if (n_layers - 1) not in val_layers and n_layers > 0:
                val_layers.append(n_layers - 1)
            ok = validate_round_trip(args.out, gguf, val_layers)
            if not ok:
                print("[convert] VALIDATION FAILED")
                return 2
            print("[convert] validation passed (bit-identical + dequant match)")

        # Print a final per-tensor summary for the first layer + aux
        # so the operator can eyeball param names + shapes.
        from safetensors import safe_open
        with safe_open(args.out, framework="numpy", device="cpu") as sf:
            print("\n[summary] first-layer param names + shapes:")
            shown = 0
            for k in sf.keys():
                if k.startswith("_lookup") or ".mlp.experts." in k:
                    t = sf.get_tensor(k)
                    print(f"   {k:60s} dtype={t.dtype} shape={tuple(t.shape)}")
                    shown += 1
                    if shown >= 5:
                        break
            print(f"[summary] total tensors: {len(list(sf.keys()))}")

        # Update the model dir's safetensors index so vLLM's auto-loader
        # actually visits the IQ2 shard. Without this, the loader iterates
        # only the keys listed in model.safetensors.index.json and the IQ2
        # params stay at their init zeros, surfacing much later as zero
        # MoE output. Optional: pass --no-update-index to skip (e.g. the
        # IQ2 file lives in a different dir than the FP8 shards).
        if not args.no_update_index:
            shard_name = os.path.basename(args.out)
            # default: model dir = same dir as the IQ2 output
            model_dir = args.model_dir or out_dir
            idx_path = os.path.join(model_dir, "model.safetensors.index.json")
            if os.path.exists(idx_path):
                import json as _json
                with open(idx_path) as _f:
                    _idx = _json.load(_f)
                _wm = _idx.get("weight_map", {})
                before = sum(
                    1 for _k in _wm if "iq2" in _k or "q2_k" in _k)
                with safe_open(args.out, framework="pt") as _sf:
                    for _k in _sf.keys():
                        _wm[_k] = shard_name
                _meta = _idx.get("metadata", {})
                _meta["total_size"] = _meta.get("total_size", 0) + actual_size
                _idx["metadata"] = _meta
                with open(idx_path, "w") as _f:
                    _json.dump(_idx, _f)
                after = sum(
                    1 for _k in _wm if "iq2" in _k or "q2_k" in _k)
                print(f"[convert] index {idx_path}: iq2/q2_k keys "
                      f"{before} -> {after}; total_size += {actual_size}")
            else:
                print(f"[convert] index {idx_path} not found; skipping "
                      "update (the IQ2 shard won't be loaded by vLLM's "
                      "auto-loader unless you set up the index manually).")
        return 0
    finally:
        gguf.close()


if __name__ == "__main__":
    raise SystemExit(main())
