# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton port of DeepGEMM's ``fp8_paged_mqa_logits`` (DSv4 lightning indexer).

The DSA "lightning indexer" scores every cached context token against the
decode query and the top-k of those scores select the sparse-attention
candidate set. On Hopper/Blackwell vLLM computes the paged variant with
DeepGEMM's ``fp8_paged_mqa_logits``; DeepGEMM has no pre-SM90 kernel, so on
Ada (sm_89) ``vllm/utils/deep_gemm.py`` fell back to a dequantized torch
reference.

That reference was correct but **not CUDA-graph-capturable**: it read a
data-dependent context bound to the host (``int(lens[b].max().item())``)
to size a dynamic slice. With ``VLLM_USE_BREAKABLE_CUDAGRAPH`` the indexer
runs inside decode graph capture, where any host sync is illegal — the
engine died at ``profile_cudagraph_memory`` with
``cudaErrorStreamCaptureUnsupported`` (field-hit on the 4090, 2026-07-19,
AFTER ~10 min of weight load). The sibling sparse-MLA attention core had
already been made capture-safe; this indexer op is its sibling on the same
attention path and had been missed.

This module is the performant, capture-safe replacement:

* a Triton kernel (``_paged_mqa_logits_kernel``) that computes the logits
  directly from the paged FP8 indexer cache — no dequantize-to-a-temp, no
  host sync, static launch geometry, lengths consumed on-device;
* a pure-torch reference (``paged_mqa_logits_torch_ref``) — the capture-safe
  rewrite of the old fallback, semantics identical to DeepGEMM's and to the
  upstream reference ``_ref_fp8_fp4_paged_mqa_logits`` in
  ``tests/kernels/attention/test_deepgemm_attention.py``; it is the
  self-test oracle and the automatic degrade target;
* a reference cache packer, an on-device init self-test, and a module build
  id — mirroring ``triton_sparse_mla_dsv4.py``.

Semantics (FP8 indexer path; ``q_scale`` folded into ``weights``):

    logits[b*next_n + j, s] = sum_h relu(q[b,j,h,:] . k_deq[s,:]) * w[r,h]

for candidate token ``s`` up to the causal bound of draft row ``j``
(``s <= context_len[b] - next_n + j``), ``-inf`` elsewhere. ``k_deq`` is the
paged FP8-E4M3 cache row times its per-token f32 scale.

FP8 indexer cache layout (``[num_blocks, block_size, 1, D+4]`` uint8) is
SEGREGATED within each block, matching ``indexer_k_quant_and_cache_kernel``
(``csrc/.../cache_kernels.cu``): the first ``block_size*D`` bytes of a block
are FP8-E4M3 k values (k at ``block_base + pos*D + d``), the final
``block_size*4`` bytes are the little-endian f32 per-token dequant scales
(scale at ``block_base + block_size*D + pos*4``). It is NOT ``[D+4]`` per
token — reading it interleaved shifts k by ``4*pos`` bytes and bitcasts
neighbouring k bytes as the scale, yielding ~1e30 / NaN decode scores.

Safety model: the kernel is used ONLY after ``dsv4_paged_mqa_logits_self_test``
has validated it against the torch reference on the actual silicon (run once
at attention-layer init, BEFORE any capture). If the self-test does not run,
or fails, or Triton is unavailable, the op transparently uses the torch
reference (correct, capture-safe, slower). This keeps a boot correct even
where the kernel has never been proven.

Env knobs (read lazily):

* ``VLLM_DSV4_PAGED_MQA_LOGITS_FORCE_TORCH=1`` — never use the kernel; always
  the torch reference (triage / A-B oracle).
* ``VLLM_DSV4_PAGED_MQA_LOGITS_SELFTEST=0`` — skip the init self-test (the
  kernel then stays disabled, torch reference is used).
"""

import functools
import os

import torch

from vllm.logger import init_logger
from vllm.triton_utils import HAS_TRITON, tl, triton

logger = init_logger(__name__)

# Finite value never used as a sentinel here (the indexer wants true -inf for
# masked candidates so top-k excludes them); kept only for documentation
# parity with the sparse-MLA port.
_NEG_INF = float("-inf")

_IS_INTERPRET = os.getenv("TRITON_INTERPRET", "0") == "1"

# Set True by the on-device self-test once the kernel matches the reference on
# this silicon. Until then the op uses the torch reference.
_KERNEL_TRUSTED = False


def _force_torch() -> bool:
    return os.getenv("VLLM_DSV4_PAGED_MQA_LOGITS_FORCE_TORCH", "0") in (
        "1", "true", "yes", "on")


@functools.cache
def _module_build_id() -> str:
    """Short content hash of THIS file, logged by the self-test. 'Which
    build produced this log' burned a deploy cycle on the sibling port."""
    import hashlib
    with open(__file__, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


@triton.jit
def _decode_fp8_e4m3(b):
    """FP8-E4M3 byte (uint32 lane) -> f32, integer math.

    Portable to the Triton interpreter (numpy) so the CPU tier can run the
    kernel. The producers saturate to +/-448, so the NaN encodings 0x7F /
    0xFF never occur and are not handled."""
    sign = (b >> 7) & 1
    expo = ((b >> 3) & 0xF).to(tl.float32)
    mant = (b & 0x7).to(tl.float32)
    mag = tl.where(
        expo == 0.0,
        mant * 0.001953125,               # subnormal: mant/8 * 2^-6
        tl.exp2(expo - 7.0) * (1.0 + mant * 0.125),
    )
    return tl.where(sign == 1, -mag, mag)


@triton.jit
def _paged_mqa_logits_kernel(
    q_ptr,          # uint8 view of FP8 q, [num_rows, H, D]
    kv_ptr,         # uint8 [num_blocks, block_size, D+4]
    w_ptr,          # f32 [num_rows, H]
    bt_ptr,         # int32 [B, max_blocks]
    len_ptr,        # int32 [num_rows] per-row causal length (clamped >=0)
    out_ptr,        # f32 [num_rows, max_model_len]
    next_n,
    num_blocks,
    max_model_len,
    H,
    stride_q_row,
    stride_q_h,
    stride_kv_blk,
    stride_kv_pos,
    stride_w_row,
    stride_bt_b,
    stride_out_row,
    D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    DOT_BF16: tl.constexpr,
):
    r = tl.program_id(0)        # output row = b * next_n + j
    blk = tl.program_id(1)      # logical block index into block_tables
    b = r // next_n

    n_offs = tl.arange(0, BLOCK_N)
    n_ok = n_offs < BLOCK_SIZE
    d_offs = tl.arange(0, D)

    # physical page; clamp padding/invalid table slots in-bounds (their
    # candidates are length-masked out below, so the value is irrelevant).
    phys = tl.load(bt_ptr + b * stride_bt_b + blk)
    phys = tl.where((phys >= 0) & (phys < num_blocks), phys, 0).to(tl.int64)

    # Dequantize the K tile: FP8-E4M3 bytes * per-token f32 scale.
    # Segregated per-block layout (see module docstring): k at
    # block_base + pos*D, scale at block_base + block_size*D + pos*4.
    blk_base = phys * stride_kv_blk
    base = blk_base + n_offs[:, None] * D
    kb = tl.load(kv_ptr + base + d_offs[None, :], mask=n_ok[:, None],
                 other=0).to(tl.uint32)
    k = _decode_fp8_e4m3(kb)                                # [BLOCK_N, D]
    s_base = blk_base + BLOCK_SIZE * D + n_offs * 4
    c0 = tl.load(kv_ptr + s_base + 0, mask=n_ok, other=0).to(tl.uint32)
    c1 = tl.load(kv_ptr + s_base + 1, mask=n_ok, other=0).to(tl.uint32)
    c2 = tl.load(kv_ptr + s_base + 2, mask=n_ok, other=0).to(tl.uint32)
    c3 = tl.load(kv_ptr + s_base + 3, mask=n_ok, other=0).to(tl.uint32)
    scale = ((c3 << 24) | (c2 << 16) | (c1 << 8) | c0).to(
        tl.float32, bitcast=True)                           # [BLOCK_N]
    k = k * scale[:, None]
    if DOT_BF16:
        k = k.to(tl.bfloat16)

    # sum_h relu(q_h . k_n) * w_h, tiled over heads.
    acc = tl.zeros([BLOCK_N], tl.float32)
    for h0 in range(0, H, BLOCK_H):
        h_offs = h0 + tl.arange(0, BLOCK_H)
        h_ok = h_offs < H
        qb = tl.load(
            q_ptr + r * stride_q_row + h_offs[:, None] * stride_q_h
            + d_offs[None, :], mask=h_ok[:, None], other=0).to(tl.uint32)
        q = _decode_fp8_e4m3(qb)                            # [BLOCK_H, D]
        w = tl.load(w_ptr + r * stride_w_row + h_offs, mask=h_ok, other=0.0)
        if DOT_BF16:
            s = tl.dot(q.to(tl.bfloat16), tl.trans(k))
        else:
            # fp32 operands -> Ada tf32 MMA (10-bit mantissa), matching the
            # sparse-MLA attention kernel's DOT_BF16=False path.
            s = tl.dot(q, tl.trans(k), allow_tf32=True)     # [BLOCK_H, N]
        s = tl.maximum(s, 0.0) * w[:, None]
        s = tl.where(h_ok[:, None], s, 0.0)
        acc += tl.sum(s, axis=0)

    length = tl.load(len_ptr + r)
    n_pos = blk * BLOCK_SIZE + n_offs
    valid = n_ok & (n_pos < length) & (n_pos < max_model_len)
    out = tl.where(valid, acc, float("-inf"))
    tl.store(out_ptr + r * stride_out_row + n_pos, out,
             mask=n_pos < max_model_len)


def _per_row_lengths(context_lens: torch.Tensor, next_n: int,
                     bsz: int) -> torch.Tensor:
    """Per-(request, draft-row) causal length, capture-safe (no host sync).

    Row ``j`` of request ``b`` attends to tokens ``s <= context_len - next_n
    + j``; the equivalent exclusive length is ``context_len - next_n + 1 +
    j`` (== ``context_lens[b, j]`` when a 2-D per-row length is supplied).
    """
    dev = context_lens.device
    if context_lens.ndim == 2:
        lens2d = context_lens
    else:
        offs = torch.arange(next_n, device=dev, dtype=torch.int32)
        lens2d = context_lens[:, None] - (next_n - 1 - offs)[None, :]
    return lens2d.reshape(bsz * next_n).clamp_(min=0).to(torch.int32).contiguous()


def paged_mqa_logits_dsv4_triton(
    q_values: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    """Capture-safe Triton paged MQA logits (FP8 indexer path).

    q_values: FP8-E4M3 ``[B, next_n, H, D]`` (per-token q scale folded into
    ``weights``). kv_cache: uint8 ``[num_blocks, block_size, 1, D+4]``.
    weights: f32 ``[B*next_n, H]``. context_lens: int32 ``[B]`` or
    ``[B, next_n]``. block_tables: int32 ``[B, max_blocks]``. Returns f32
    ``[B*next_n, max_model_len]``.

    No host synchronization, no data-dependent shapes.
    """
    assert q_values.dtype == torch.float8_e4m3fn, q_values.dtype
    assert kv_cache.dtype == torch.uint8, kv_cache.dtype
    bsz, next_n, H, D = q_values.shape
    num_blocks, block_size = kv_cache.shape[0], kv_cache.shape[1]
    dev = q_values.device
    num_rows = bsz * next_n

    q_u8 = q_values.contiguous().view(torch.uint8)              # [B,next_n,H,D]
    q_u8 = q_u8.view(num_rows, H, D)
    # Segregated per-block layout: reshape to 2-D so stride(0) is the block
    # stride (block_size*(D+4)); the kernel addresses k/scale via the
    # constexpr D and BLOCK_SIZE (k at block_base+pos*D, scale at
    # block_base+BLOCK_SIZE*D+pos*4).
    kv = kv_cache.reshape(num_blocks, block_size * (D + 4))
    w = weights.contiguous()
    bt = block_tables.contiguous()
    lens_row = _per_row_lengths(context_lens.to(dev), next_n, bsz)
    out = torch.full((num_rows, max_model_len), float("-inf"),
                     dtype=torch.float32, device=dev)

    logger.info_once(
        "DSv4 paged-MQA-logits Triton port geometry: q%s %s kv_cache%s %s "
        "weights%s ctx_lens%s block_tables%s mml=%d (module build %s)",
        tuple(q_values.shape), q_values.dtype, tuple(kv_cache.shape),
        kv_cache.dtype, tuple(weights.shape), tuple(context_lens.shape),
        tuple(block_tables.shape), max_model_len, _module_build_id())

    if num_rows == 0:
        return out
    max_blocks = bt.shape[1]
    block_h = 32 if H >= 32 else 16
    grid = (num_rows, max_blocks)
    _paged_mqa_logits_kernel[grid](
        q_u8, kv, w, bt, lens_row, out,
        next_n, num_blocks, max_model_len, H,
        q_u8.stride(0), q_u8.stride(1),
        kv.stride(0), kv.stride(1),
        w.stride(0),
        bt.stride(0),
        out.stride(0),
        D=D,
        BLOCK_N=triton.next_power_of_2(block_size),
        BLOCK_SIZE=block_size,
        BLOCK_H=block_h,
        # Force the fp32/tf32 score dot -- the mirror of the sparse-MLA
        # attention fix (commit b15828066, 16K 0.948->0.966). The bf16 dot
        # rounded the dequantized K (fp8 value x full-f32 per-token scale)
        # and Q to 8-bit mantissa BEFORE scoring; the indexer's top-k
        # SELECTS which tokens sparse attention sees, so score rounding is
        # a discrete error source at long context (a needle key that slips
        # below the top-512 boundary is unrecoverable downstream). smem
        # fits fp32 operands here easily: D=128 (vs 512 in the attention
        # kernel), q [32,128] + k [BLOCK_N,128] f32 tiles.
        DOT_BF16=False,
        num_warps=4,
        num_stages=1,
    )
    return out


def paged_mqa_logits_torch_ref(
    q_values: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    """Pure-torch reference; capture-safe; the self-test oracle and the
    automatic degrade target. Semantics identical to DeepGEMM's op and to
    ``_ref_fp8_fp4_paged_mqa_logits`` (upstream diff gate 1e-3)."""
    qv = q_values
    bsz, next_n, H, d = qv.shape
    num_blocks, block_size = kv_cache.shape[0], kv_cache.shape[1]
    dev = qv.device
    if context_lens.ndim == 2:
        lens2d = context_lens.to(dev)
    else:
        offs = torch.arange(next_n, device=dev, dtype=torch.int32)
        lens2d = context_lens.to(dev)[:, None] - (next_n - 1 - offs)[None, :]
    lens2d = lens2d.clamp(min=0)
    # Segregated per-block layout: split each block into its k region
    # (block_size*d bytes) and its scale region (block_size*4 bytes).
    kv = kv_cache.reshape(num_blocks, block_size * (d + 4))
    kv_q = kv[:, :block_size * d].reshape(num_blocks, block_size, d)
    kv_s = kv[:, block_size * d:].reshape(num_blocks, block_size, 4)
    max_nblk = block_tables.shape[1]
    width = max_nblk * block_size
    w_cols = min(width, max_model_len)
    logits = torch.full((bsz * next_n, max_model_len), float("-inf"),
                        dtype=torch.float32, device=dev)
    wf = weights.float()
    pos = torch.arange(width, device=dev)
    for b in range(bsz):
        blocks = block_tables[b].long().clamp_(0, num_blocks - 1)
        kq = kv_q[blocks].reshape(width, d)
        ks = kv_s[blocks].reshape(width, 4)
        kf = (kq.contiguous().view(torch.float8_e4m3fn).float()
              * ks.contiguous().view(torch.float32).view(-1, 1))
        qf = qv[b].float()                                   # [next_n, H, d]
        s = torch.einsum("jhd,sd->jhs", qf, kf)
        w_b = wf[b * next_n:(b + 1) * next_n]                # [next_n, H]
        row = (torch.relu(s) * w_b[:, :, None]).sum(1)       # [next_n, width]
        row = row.masked_fill(pos[None, :] >= lens2d[b][:, None],
                              float("-inf"))
        logits[b * next_n:(b + 1) * next_n, :w_cols] = row[:, :w_cols]
    return logits


def pack_fp8_indexer_cache_reference(kv_rows: torch.Tensor,
                                     block_size: int) -> torch.Tensor:
    """Encode f32 rows ``[num_blocks*block_size, D]`` into the FP8 indexer
    cache byte layout ``[num_blocks, block_size, 1, D+4]`` uint8, SEGREGATED
    per block (block_size*D quantized k bytes, then block_size*4 f32 per-token
    scales) -- matching ``indexer_k_quant_and_cache_kernel``. Reference writer
    for the self-test and tests; saturating FP8-E4M3 so the kernel's integer
    decode is exact."""
    n, d = kv_rows.shape
    assert n % block_size == 0
    num_blocks = n // block_size
    amax = kv_rows.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = (amax / 448.0).to(torch.float32)                 # per-token f32
    quant = (kv_rows / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    # Segregated: all k bytes per block, then all scale bytes per block.
    qb = quant.view(torch.uint8).reshape(num_blocks, block_size * d)
    sb = scale.contiguous().view(torch.uint8).reshape(num_blocks, block_size * 4)
    blk_bytes = torch.cat([qb, sb], dim=-1)                  # [num_blocks, bs*(d+4)]
    return blk_bytes.reshape(num_blocks, block_size, 1, d + 4).contiguous()


def _worst_row_rel(got: torch.Tensor, want: torch.Tensor) -> float:
    """Worst per-row error relative to the row's magnitude, over the finite
    (unmasked) region. The kernel rounds q,k to bf16 before the dot while the
    reference uses f32, so near-zero logits carry ~1e-3 absolute noise; a
    row-relative metric (floor 1.0) scores that as ~1e-3 while a wrong decode
    / mask / index scores O(1). Masked positions must be -inf in both."""
    got = got.float()
    want = want.float()
    fin = torch.isfinite(want)
    if not bool(fin.any()):
        return 0.0
    row_ref = torch.where(fin, want.abs(), torch.zeros_like(want)) \
        .amax(dim=-1, keepdim=True).clamp_min(1.0)
    err = torch.where(fin, (got - want).abs() / row_ref,
                      torch.zeros_like(want))
    mask_mismatch = (torch.isfinite(got) != fin)
    penalty = 1.0e30 if bool(mask_mismatch.any()) else 0.0
    return max(err.max().item(), penalty)


def _self_test_case(device: torch.device) -> float:
    """Build and run the canonical self-test; return worst_row_rel.

    Keyword-only kernel/ref invocation and explicit dtypes throughout (the
    two field-hits the sibling port took at boot: positional drift and the
    loader's bf16 default-dtype ambient). Covers packed cache, next_n>1,
    len-0 rows, partial final block, and padded block-table slots."""
    torch.manual_seed(0)
    num_blocks, block_size, bsz, next_n, H, D = 8, 64, 4, 2, 32, 128
    max_model_len = 256
    kv_rows = torch.randn(num_blocks * block_size, D, device=device,
                          dtype=torch.float32) * 2.0
    kv_cache = pack_fp8_indexer_cache_reference(kv_rows, block_size)
    q = (torch.randn(bsz, next_n, H, D, device=device, dtype=torch.float32)
         ).clamp(-448, 448).to(torch.float8_e4m3fn)
    weights = torch.randn(bsz * next_n, H, device=device, dtype=torch.float32)
    context_lens = torch.tensor([[200, 201], [0, 1], [64, 65], [255, 256]],
                                device=device, dtype=torch.int32)
    max_blocks = -(-max_model_len // block_size)
    block_tables = torch.zeros(bsz, max_blocks, device=device,
                               dtype=torch.int32)
    perm = torch.randperm(num_blocks, device=device).to(torch.int32)
    for b in range(bsz):
        need = min(max_blocks, num_blocks)
        block_tables[b, :need] = perm[:need]
    kwargs = dict(q_values=q, kv_cache=kv_cache, weights=weights,
                  context_lens=context_lens, block_tables=block_tables,
                  max_model_len=max_model_len)
    got = paged_mqa_logits_dsv4_triton(**kwargs)
    want = paged_mqa_logits_torch_ref(**kwargs)
    return _worst_row_rel(got, want)


def _self_test_case_real_writer(device: torch.device) -> float:
    """Layout cross-check against vLLM's OWN cache writer.

    ``_self_test_case`` compares this port to a torch reference that reads the
    cache the same way the port does -- so it passes whatever the layout is.
    That is exactly how the segregated/interleaved mismatch survived: kernel,
    reference and reference-packer all agreed with each other and all
    disagreed with ``indexer_k_quant_and_cache``, which is what actually
    fills the cache at runtime. Here the bytes come from that real writer, so
    a layout drift fails instead of self-confirming.

    Returns the worst row-relative error, or -1.0 if the op is unavailable.
    """
    from vllm import _custom_ops as vllm_ops

    num_blocks, block_size, bsz, next_n, H, D = 4, 64, 2, 1, 32, 128
    max_model_len = 128
    n_tok = num_blocks * block_size
    kv_cache = torch.zeros(num_blocks, block_size, 1, D + 4,
                           dtype=torch.uint8, device=device)
    k_rows = (torch.randn(n_tok, D, device=device, dtype=torch.bfloat16) * 2.0)
    slot_mapping = torch.arange(n_tok, device=device, dtype=torch.int64)
    try:
        # quant_block_size == head_dim -> exactly one f32 scale per token,
        # which is the DSv4 indexer's configuration.
        vllm_ops.indexer_k_quant_and_cache(k_rows, kv_cache, slot_mapping, D,
                                           "fp8_e4m3")
    except Exception as exc:                                  # pragma: no cover
        logger.warning_once(
            "DSv4 paged-MQA-logits: real-writer layout cross-check skipped "
            "(%s: %s)", type(exc).__name__, exc)
        return -1.0
    q = (torch.randn(bsz, next_n, H, D, device=device, dtype=torch.float32)
         ).clamp(-448, 448).to(torch.float8_e4m3fn)
    weights = torch.randn(bsz * next_n, H, device=device, dtype=torch.float32)
    context_lens = torch.tensor([[100], [128]], device=device,
                                dtype=torch.int32)
    max_blocks = -(-max_model_len // block_size)
    block_tables = torch.zeros(bsz, max_blocks, device=device,
                               dtype=torch.int32)
    for b in range(bsz):
        # deliberately NOT identity: a layout bug that only shows up when the
        # logical and physical block index differ would otherwise hide.
        block_tables[b, :max_blocks] = torch.arange(
            max_blocks, device=device, dtype=torch.int32) + b
    got = paged_mqa_logits_dsv4_triton(
        q_values=q, kv_cache=kv_cache, weights=weights,
        context_lens=context_lens, block_tables=block_tables,
        max_model_len=max_model_len)

    # Ground truth from the PRE-quantization k, never touching the cache. The
    # only comparison that is independent of the layout: a reference that
    # reads the cache would share whatever assumption the kernel makes and
    # agree with it either way (which is the trap this test exists to avoid).
    width = max_blocks * block_size
    pos = torch.arange(width, device=device)
    want = torch.full((bsz * next_n, max_model_len), float("-inf"),
                      dtype=torch.float32, device=device)
    for b in range(bsz):
        phys = (block_tables[b].long()[pos // block_size] * block_size
                + pos % block_size).clamp_(0, n_tok - 1)
        kf = k_rows[phys].float()                            # [width, D]
        s = torch.einsum("jhd,sd->jhs", q[b].float(), kf)
        w_b = weights[b * next_n:(b + 1) * next_n].float()
        row = (torch.relu(s) * w_b[:, :, None]).sum(1)       # [next_n, width]
        row = row.masked_fill(
            pos[None, :] >= context_lens[b].to(device)[:, None].clamp(min=0),
            float("-inf"))
        cols = min(width, max_model_len)
        want[b * next_n:(b + 1) * next_n, :cols] = row[:, :cols]
    return _worst_row_rel(got, want)


@functools.cache
def dsv4_paged_mqa_logits_self_test() -> None:
    """One-shot on-device self-test; enables the kernel iff it matches the
    torch reference on this silicon. Runs at attention-layer init (before any
    graph capture). On failure the op keeps using the torch reference — a
    codegen/driver regression degrades performance, never correctness."""
    global _KERNEL_TRUSTED
    if _force_torch():
        logger.warning_once(
            "DSv4 paged-MQA-logits: VLLM_DSV4_PAGED_MQA_LOGITS_FORCE_TORCH=1 "
            "- using the torch reference, kernel disabled")
        return
    if os.getenv("VLLM_DSV4_PAGED_MQA_LOGITS_SELFTEST", "1") in (
            "0", "false", "no", "off"):
        logger.warning_once(
            "DSv4 paged-MQA-logits self-test SKIPPED - kernel stays disabled, "
            "torch reference in use (VLLM_DSV4_PAGED_MQA_LOGITS_SELFTEST=0)")
        return
    if not (HAS_TRITON and torch.cuda.is_available()):
        return
    device = torch.device("cuda", torch.cuda.current_device())
    worst = _self_test_case(device)
    cap = torch.cuda.get_device_capability(device)
    # Layout gate. Quantization noise only (k is fp8-e4m3 with a per-token
    # scale, q is fp8 on both sides), so a correct read lands near 1e-2 while
    # a layout mismatch is O(1) or worse.
    layout = _self_test_case_real_writer(device)
    # NaN-safe: a wrong layout reads bitcast FP8 bytes as the dequant scale
    # and the error comes back NaN, and `nan > 8e-2` is False. Gate on the
    # PASS condition so anything not provably small fails.
    if layout != -1.0 and not layout <= 8e-2:
        _KERNEL_TRUSTED = False
        logger.error(
            "DSv4 paged-MQA-logits LAYOUT CHECK FAILED on sm_%d%d: "
            "worst_row_rel=%.3e vs f32 ground truth over a cache filled by "
            "vLLM's own indexer_k_quant_and_cache (threshold 8e-2). The port "
            "is reading the indexer KV cache with the wrong byte layout - "
            "keeping the torch reference. Note the reference reads the cache "
            "the same way, so it is NOT known-good here either.",
            cap[0], cap[1], layout)
        return
    if layout >= 0.0:
        logger.info(
            "DSv4 paged-MQA-logits layout check vs vLLM's own cache writer: "
            "worst_row_rel=%.3e (f32 ground truth)", layout)
    if worst <= 3e-2:
        _KERNEL_TRUSTED = True
        logger.info(
            "DSv4 paged-MQA-logits Triton port self-test on sm_%d%d (%s): "
            "worst_row_rel=%.3e vs torch reference - kernel ENABLED "
            "(module build %s, torch %s, triton %s)",
            cap[0], cap[1], torch.cuda.get_device_name(device), worst,
            _module_build_id(), torch.__version__, triton.__version__)
    else:
        _KERNEL_TRUSTED = False
        logger.error(
            "DSv4 paged-MQA-logits Triton port self-test FAILED on sm_%d%d: "
            "worst_row_rel=%.3e vs torch reference (threshold 3e-2) - keeping "
            "the torch reference fallback (correct, slower). This is a "
            "codegen/driver mismatch; module build %s, torch %s, triton %s.",
            cap[0], cap[1], worst, _module_build_id(), torch.__version__,
            triton.__version__)


def maybe_paged_mqa_logits_dsv4(
    q_values: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor | None:
    """Return kernel logits iff the kernel has been trusted by the on-device
    self-test (and Triton is present and it isn't force-disabled); otherwise
    None so the caller uses the capture-safe torch reference. Performs no
    self-test and no host sync — safe to call inside CUDA-graph capture."""
    if not (_KERNEL_TRUSTED and HAS_TRITON and not _force_torch()):
        return None
    return paged_mqa_logits_dsv4_triton(q_values, kv_cache, weights,
                                        context_lens, block_tables,
                                        max_model_len)
