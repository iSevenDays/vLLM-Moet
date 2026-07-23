# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility wrapper for DeepGEMM API changes.

Users of vLLM should always import **only** these wrappers.
"""

import contextlib
import functools
import importlib
import os
from collections.abc import Callable
from enum import Enum
from typing import Any, NoReturn

import torch

import vllm.envs as envs
from vllm.logger import logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_deep_gemm
from vllm.utils.math_utils import cdiv

_DEEPGEMM_BLACKWELL_EXCLUDED_MODEL_TYPES: set[str] = {
    "qwen3_5_text",
    "qwen3_5_moe_text",
}


def should_auto_disable_deep_gemm(model_type: str | None) -> bool:
    """Check if DeepGemm should be auto-disabled for this model on Blackwell.

    Returns True if the model is known to have accuracy degradation with
    DeepGemm's E8M0 scale format on Blackwell GPUs (SM100+).
    """
    if model_type is None:
        return False
    if not (
        current_platform.is_device_capability_family(100)
        or current_platform.is_device_capability_family(120)
    ):
        return False
    return model_type in _DEEPGEMM_BLACKWELL_EXCLUDED_MODEL_TYPES


class DeepGemmQuantScaleFMT(Enum):
    # Float32 scales in Float32 tensor
    FLOAT32 = 0
    # Compute float32 scales and ceil the scales to UE8M0.
    # Keep the scales in Float32 tensor.
    FLOAT32_CEIL_UE8M0 = 1
    # Compute float32 scales and ceil the scales to UE8M0.
    # Pack the scales into a int32 tensor where each int32
    # element contains 4 scale values.
    UE8M0 = 2

    @classmethod
    def init_oracle_cache(cls) -> None:
        """Initialize the oracle decision and store it in the class cache"""
        cached = getattr(cls, "_oracle_cache", None)
        if cached is not None:
            return

        use_e8m0 = (
            envs.VLLM_USE_DEEP_GEMM_E8M0
            and is_deep_gemm_supported()
            and (_fp8_gemm_nt_impl is not None)
        )
        if not use_e8m0:
            cls._oracle_cache = cls.FLOAT32  # type: ignore
            return

        cls._oracle_cache = (  # type: ignore
            cls.UE8M0
            if (
                current_platform.is_device_capability_family(100)
                or current_platform.is_device_capability_family(120)
            )
            else cls.FLOAT32_CEIL_UE8M0
        )

    @classmethod
    def from_oracle(cls) -> "DeepGemmQuantScaleFMT":
        """Return the pre-initialized oracle decision"""
        cached = getattr(cls, "_oracle_cache", None)
        assert cached is not None, "DeepGemmQuantScaleFMT oracle cache not initialized"
        return cached


@functools.cache
def is_deep_gemm_supported() -> bool:
    """Return `True` if DeepGEMM is supported on the current platform.
    Currently, only Hopper and Blackwell GPUs are supported.
    """
    is_supported_arch = current_platform.support_deep_gemm()
    return envs.VLLM_USE_DEEP_GEMM and has_deep_gemm() and is_supported_arch


@functools.cache
def is_deep_gemm_e8m0_used() -> bool:
    """Return `True` if vLLM is configured to use DeepGEMM "
    "E8M0 scale on a Hopper or Blackwell-class GPU.
    """
    if not is_deep_gemm_supported():
        logger.debug_once(
            "DeepGEMM E8M0 disabled: DeepGEMM not supported on this system."
        )
        return False

    _lazy_init()

    if _fp8_gemm_nt_impl is None:
        logger.info_once("DeepGEMM E8M0 disabled: _fp8_gemm_nt_impl not found")
        return False

    if envs.VLLM_USE_DEEP_GEMM_E8M0:
        logger.info_once("DeepGEMM E8M0 enabled on current platform.")
        return True

    logger.info_once("DeepGEMM E8M0 disabled on current configuration.")
    return False


def _missing(*_: Any, **__: Any) -> NoReturn:
    """Placeholder for unavailable DeepGEMM backend."""
    raise RuntimeError(
        "DeepGEMM backend is unavailable in the current vLLM environment, "
        "or the available DeepGEMM package does not provide the required APIs "
        "for these kernels."
    )


_cublaslt_gemm_nt_impl: Callable[..., Any] | None = None
_fp8_gemm_nt_impl: Callable[..., Any] | None = None
_fp8_einsum_impl: Callable[..., Any] | None = None
_grouped_impl: Callable[..., Any] | None = None
_grouped_masked_impl: Callable[..., Any] | None = None
_grouped_fp4_impl: Callable[..., Any] | None = None
_fp8_fp4_mqa_logits_impl: Callable[..., Any] | None = None
_fp8_fp4_paged_mqa_logits_impl: Callable[..., Any] | None = None
_get_paged_mqa_logits_metadata_impl: Callable[..., Any] | None = None
_tf32_hc_prenorm_gemm_impl: Callable[..., Any] | None = None
_get_mn_major_tma_aligned_tensor_impl: Callable[..., Any] | None = None
_get_mk_alignment_for_contiguous_layout_impl: Callable[..., Any] | None = None
_get_theoretical_mk_alignment_for_contiguous_layout_impl: Callable[..., Any] | None = (
    None
)
_transform_sf_into_required_layout_impl: Callable[..., Any] | None = None
_pack_ue8m0_to_int_impl: Callable[..., Any] | None = None
_get_mn_major_tma_aligned_packed_ue8m0_tensor_impl: Callable[..., Any] | None = None
_get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor_impl: (
    Callable[..., Any] | None
) = None


@functools.cache
def _import_deep_gemm():
    """Import the deep_gemm module.

    Prefers an externally installed ``deep_gemm`` package (so users can
    pin a specific version), then falls back to the vendored copy bundled
    in the vLLM wheel.

    Returns ``None`` when neither source is usable.
    """
    # 1. Try the external (pip-installed) package first.
    try:
        module = importlib.import_module("deep_gemm")
        logger.debug_once("Imported deep_gemm module from site-packages")
        return module
    except ImportError:
        logger.info_once(
            "deep_gemm not found in site-packages, "
            "trying vendored vllm.third_party.deep_gemm"
        )

    # 2. Fall back to the vendored copy bundled in the vLLM wheel.
    try:
        module = importlib.import_module("vllm.third_party.deep_gemm")
        logger.debug_once("Imported deep_gemm module from vllm.third_party.deep_gemm")
        return module
    except ImportError:
        logger.info_once("Vendored deep_gemm not found either")
    except Exception as e:
        # The vendored module may raise RuntimeError during _C.init()
        # if JIT include files are missing (e.g. incomplete wheel).
        logger.warning_once("Failed to import vendored deep_gemm: %s", e)

    return None


def _apply_pdl(mod, enable: bool = True) -> None:
    mod_name = getattr(mod, "__name__", str(mod))
    try:
        set_pdl_fn = getattr(mod, "set_pdl", None)
        if set_pdl_fn is None:
            return
        set_pdl_fn(enable)
        logger.info_once(
            "DeepGEMM PDL %s on %s.",
            "enabled" if enable else "disabled",
            mod_name,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning_once("Failed to set DeepGEMM PDL on %s: %s", mod_name, e)


def _lazy_init() -> None:
    """Import deep_gemm and resolve symbols on first use."""
    global _cublaslt_gemm_nt_impl
    global _fp8_gemm_nt_impl, _fp8_einsum_impl
    global _grouped_impl, _grouped_masked_impl, _grouped_fp4_impl
    global _fp8_fp4_mqa_logits_impl, _fp8_fp4_paged_mqa_logits_impl
    global _get_paged_mqa_logits_metadata_impl
    global _tf32_hc_prenorm_gemm_impl
    global _get_mn_major_tma_aligned_tensor_impl
    global _get_mk_alignment_for_contiguous_layout_impl
    global _get_theoretical_mk_alignment_for_contiguous_layout_impl
    global _transform_sf_into_required_layout_impl
    global _pack_ue8m0_to_int_impl
    global _get_mn_major_tma_aligned_packed_ue8m0_tensor_impl
    global _get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor_impl
    # fast path
    if (
        _cublaslt_gemm_nt_impl is not None
        or _fp8_gemm_nt_impl is not None
        or _fp8_einsum_impl is not None
        or _grouped_impl is not None
        or _grouped_masked_impl is not None
        or _grouped_fp4_impl is not None
        or _fp8_fp4_mqa_logits_impl is not None
        or _fp8_fp4_paged_mqa_logits_impl is not None
        or _get_paged_mqa_logits_metadata_impl is not None
        or _tf32_hc_prenorm_gemm_impl is not None
        or _get_mk_alignment_for_contiguous_layout_impl is not None
        or _transform_sf_into_required_layout_impl is not None
        or _pack_ue8m0_to_int_impl is not None
        or _get_mn_major_tma_aligned_packed_ue8m0_tensor_impl is not None
        or _get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor_impl is not None
    ):
        return

    if not has_deep_gemm():
        return

    # Set up deep_gemm cache path
    DEEP_GEMM_JIT_CACHE_ENV_NAME = "DG_JIT_CACHE_DIR"
    if not os.environ.get(DEEP_GEMM_JIT_CACHE_ENV_NAME, None):
        os.environ[DEEP_GEMM_JIT_CACHE_ENV_NAME] = os.path.join(
            envs.VLLM_CACHE_ROOT, "deep_gemm"
        )

    _dg = _import_deep_gemm()
    if _dg is None:
        return

    # Enable PDL for DeepGEMM on architectures that support it (SM90+).
    if current_platform.is_arch_support_pdl():
        _apply_pdl(_dg, True)
    _cublaslt_gemm_nt_impl = getattr(_dg, "cublaslt_gemm_nt", None)
    _fp8_gemm_nt_impl = getattr(_dg, "fp8_gemm_nt", None)
    _fp8_einsum_impl = getattr(_dg, "fp8_einsum", None)
    _grouped_impl = getattr(_dg, "m_grouped_fp8_gemm_nt_contiguous", None)
    _grouped_masked_impl = getattr(_dg, "fp8_m_grouped_gemm_nt_masked", None)
    _grouped_fp4_impl = getattr(_dg, "m_grouped_fp8_fp4_gemm_nt_contiguous", None)
    # DeepGEMM exposes fp8_fp4_*_mqa_logits as the canonical symbols that
    # handle both the FP8 and FP4 Q/K paths via a tuple-typed `q`.
    _fp8_fp4_mqa_logits_impl = getattr(_dg, "fp8_fp4_mqa_logits", None)
    _fp8_fp4_paged_mqa_logits_impl = getattr(_dg, "fp8_fp4_paged_mqa_logits", None)
    _get_paged_mqa_logits_metadata_impl = getattr(
        _dg, "get_paged_mqa_logits_metadata", None
    )
    _tf32_hc_prenorm_gemm_impl = getattr(_dg, "tf32_hc_prenorm_gemm", None)
    _get_mn_major_tma_aligned_tensor_impl = getattr(
        _dg, "get_mn_major_tma_aligned_tensor", None
    )
    _get_mk_alignment_for_contiguous_layout_impl = getattr(
        _dg, "get_mk_alignment_for_contiguous_layout", None
    )
    _get_theoretical_mk_alignment_for_contiguous_layout_impl = getattr(
        _dg, "get_theoretical_mk_alignment_for_contiguous_layout", None
    )
    _transform_sf_into_required_layout_impl = getattr(
        _dg, "transform_sf_into_required_layout", None
    )
    _pack_ue8m0_to_int_impl = getattr(_dg, "pack_ue8m0_to_int", None)
    _get_mn_major_tma_aligned_packed_ue8m0_tensor_impl = getattr(
        _dg, "get_mn_major_tma_aligned_packed_ue8m0_tensor", None
    )
    _get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor_impl = getattr(
        _dg, "get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor", None
    )
    DeepGemmQuantScaleFMT.init_oracle_cache()


def get_num_sms() -> int:
    _lazy_init()
    dg = _import_deep_gemm()
    if dg is None:
        raise RuntimeError("DeepGEMM is not available")
    return int(dg.get_num_sms())


def set_num_sms(num_sms: int) -> None:
    _lazy_init()
    dg = _import_deep_gemm()
    if dg is None:
        raise RuntimeError("DeepGEMM is not available")
    dg.set_num_sms(num_sms)


def get_mk_alignment_for_contiguous_layout() -> list[int]:
    _lazy_init()
    if _get_mk_alignment_for_contiguous_layout_impl is None:
        return _missing()
    mk_align_size = _get_mk_alignment_for_contiguous_layout_impl()
    return [mk_align_size, mk_align_size]


def get_theoretical_mk_alignment_for_contiguous_layout(
    expected_m: int | None = None,
    num_groups: int | None = None,
) -> int:
    """Per-call optimal M alignment for grouped contiguous GEMMs.

    `expected_m` is the TOTAL routed tokens (sum across experts, typically
    M × num_topk). `num_groups` is the number of experts on this rank.
    The helper divides to recover per-expert em and picks an alignment based
    on data-driven thresholds (see deep_gemm runtime.hpp comments).

    Older callers that omit `num_groups` are interpreted as passing already
    per-expert em (legacy behaviour preserved for backward compat).
    """
    _lazy_init()
    if _get_theoretical_mk_alignment_for_contiguous_layout_impl is None:
        return _missing()
    if num_groups is None:
        return _get_theoretical_mk_alignment_for_contiguous_layout_impl(expected_m)
    if num_groups <= 0:
        raise ValueError(f"num_groups must be positive, got {num_groups}")
    try:
        return _get_theoretical_mk_alignment_for_contiguous_layout_impl(
            expected_m, num_groups
        )
    except TypeError:
        per_group_m = None if expected_m is None else cdiv(expected_m, num_groups)
        return _get_theoretical_mk_alignment_for_contiguous_layout_impl(per_group_m)


def set_mk_alignment_for_contiguous_layout(value: int) -> None:
    """Set DeepGEMM's BLOCK_M cap for grouped contiguous GEMMs.

    The DG heuristic constrains BLOCK_M ≤ this value when picking a kernel
    layout. Use this in concert with `compute_aligned_M_and_alignment`'s
    per-call alignment so the workspace's per-expert padding matches the
    kernel's BLOCK_M; a mismatch leads to the scheduler reading the wrong
    expert_id from `m_indices` at `m_block_idx * BLOCK_M` stride and
    OOB-indexing the B-weights tensor (manifests as IMA under CUDA-graph
    replay).
    """
    _lazy_init()
    dg = _import_deep_gemm()
    if dg is None:
        raise RuntimeError("DeepGEMM is not available")
    dg.set_mk_alignment_for_contiguous_layout(value)


@contextlib.contextmanager
def mk_alignment_scope(value: int):
    """Temporarily set DeepGEMM's BLOCK_M cap, restoring on exit.

    Use around a sequence of grouped-contiguous GEMM calls whose workspace
    is padded to `value` (typically the per_call_align returned by
    `compute_aligned_M_and_alignment`).
    """
    prev = get_mk_alignment_for_contiguous_layout()[0]
    set_mk_alignment_for_contiguous_layout(value)
    try:
        yield
    finally:
        set_mk_alignment_for_contiguous_layout(prev)


def get_col_major_tma_aligned_tensor(x: torch.Tensor) -> torch.Tensor:
    """Wrapper for DeepGEMM's get_mn_major_tma_aligned_tensor"""
    _lazy_init()
    if _get_mn_major_tma_aligned_tensor_impl is None:
        return _missing()
    return _get_mn_major_tma_aligned_tensor_impl(x)


def pack_ue8m0_to_int(x: torch.Tensor) -> torch.Tensor:
    """Pack 4 UE8M0 (uint8) scales into one int32.

    DeepGEMM's SM100/SM120 FP8/FP4 kernels accept either ``float32`` scales
    (legacy format, 4 B/scale) or ``int32`` packed UE8M0 scales (1 B/scale
    after 4:1 packing — 4× smaller than the legacy fp32 representation).
    """
    _lazy_init()
    if _pack_ue8m0_to_int_impl is None:
        return _missing()
    return _pack_ue8m0_to_int_impl(x)


def get_mn_major_tma_aligned_packed_ue8m0_tensor(x: torch.Tensor) -> torch.Tensor:
    """Pack UE8M0 (uint8) → int32 with the MN-major TMA-aligned layout the
    DeepGEMM kernels consume directly. 16× smaller than the fp32 legacy SF
    format. Use for non-grouped 2D scale tensors.
    """
    _lazy_init()
    if _get_mn_major_tma_aligned_packed_ue8m0_tensor_impl is None:
        return _missing()
    return _get_mn_major_tma_aligned_packed_ue8m0_tensor_impl(x)


def get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
    sf: torch.Tensor,
    ks_tensor: torch.Tensor,
    ks: list[int],
    gran_k: int,
) -> torch.Tensor:
    """Grouped (3D, expert-batched) variant of
    ``get_mn_major_tma_aligned_packed_ue8m0_tensor``. Use for MoE weight
    scale tensors of shape ``(num_experts, mn, k_scale)``.
    """
    _lazy_init()
    if _get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor_impl is None:
        return _missing()
    return _get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor_impl(
        sf, ks_tensor, ks, gran_k
    )


def cublaslt_gemm_nt(*args, **kwargs):
    _lazy_init()
    if _cublaslt_gemm_nt_impl is None:
        return _missing(*args, **kwargs)
    return _cublaslt_gemm_nt_impl(*args, **kwargs)


def fp8_gemm_nt(*args, **kwargs):
    _lazy_init()
    if _fp8_gemm_nt_impl is None:
        return _missing(*args, **kwargs)
    if "is_deep_gemm_e8m0_used" in kwargs:
        use_ue8m0 = kwargs["is_deep_gemm_e8m0_used"]
        del kwargs["is_deep_gemm_e8m0_used"]
    else:
        use_ue8m0 = is_deep_gemm_e8m0_used()
    return _fp8_gemm_nt_impl(*args, disable_ue8m0_cast=not use_ue8m0, **kwargs)


# ---------------------------------------------------------------------------
# Pre-SM90 torch fallbacks (Ada sm_89 and older): DeepGEMM kernels are
# SM90/SM100/family-120 only, but the DeepSeek-V4 model graph calls three of
# these entry points unconditionally on CUDA (o_proj fp8_einsum, the DSA
# indexer's mqa-logits pair). On unsupported arches we run dequantized torch
# reference math instead of crashing inside DeepGEMM's JIT. Semantics are
# lifted from the upstream-validated references in
# tests/kernels/attention/test_deepgemm_attention.py (diff gate 1e-3 there).
# Correctness-first: expect a real throughput cost on the indexer; a native
# sm_89 kernel port is the follow-up, not this fallback.

_fallback_warned: set[str] = set()


def _use_torch_fallback() -> bool:
    return (current_platform.is_cuda()
            and not current_platform.support_deep_gemm())


def _warn_fallback(name: str) -> None:
    if name not in _fallback_warned:
        _fallback_warned.add(name)
        logger.warning(
            "deep_gemm.%s: DeepGEMM does not support this GPU arch "
            "(pre-SM90) — using the dequantized torch REFERENCE fallback. "
            "Correct but slow; a native kernel port is the fix.", name)


def _scales_to_fp32(scales: torch.Tensor) -> torch.Tensor:
    """Scale tensor -> f32. UE8M0 (exponent-only, e.g. the weight
    scale_inv of scale_fmt=ue8m0 checkpoints - field-hit on the very
    first o_proj) decodes by exponent-field bitcast, exact for every
    byte; raw f32 passes through. The packed-int32 SM100 layouts are
    never produced on the fallback arches and stay rejected."""
    if scales.dtype == getattr(torch, "float8_e8m0fnu", None):
        return (scales.view(torch.uint8).to(torch.int32) << 23).view(
            torch.float32)
    assert scales.dtype == torch.float32, (
        f"torch fallback expects f32 or e8m0 block scales, "
        f"got {scales.dtype}")
    return scales


def _expand_block_scales(vals: torch.Tensor, scales: torch.Tensor):
    """values.float() * block-scales broadcast to the value shape."""
    x = vals.float()
    sc = _scales_to_fp32(scales)
    assert sc.ndim == x.ndim, (sc.shape, x.shape)
    for d in range(x.ndim):
        if sc.shape[d] != x.shape[d]:
            assert x.shape[d] % sc.shape[d] == 0, (
                "ragged block-scale dims are not supported by the torch "
                f"fallback: values {tuple(x.shape)} vs scales "
                f"{tuple(sc.shape)} at dim {d}")
            sc = sc.repeat_interleave(x.shape[d] // sc.shape[d], dim=d)
    return x * sc


_fallback_shapes_logged: set[str] = set()


def _log_fallback_shapes(name: str, detail: str) -> None:
    """One INFO line with the ACTUAL tensor geometry the first time each
    fallback runs — a shape mismatch then diagnoses from the log instead
    of costing a deploy cycle (the o_proj 2-D-weight einsum did exactly
    that before this line existed)."""
    if name not in _fallback_shapes_logged:
        _fallback_shapes_logged.add(name)
        logger.info("deep_gemm fallback %s: %s", name, detail)


def _einsum_operand(sub: str, tensor: torch.Tensor, dims: dict) -> torch.Tensor:
    """Reshape a dequantized operand to its subscript arity. DeepGEMM's
    einsum accepts operands whose LOGICAL rank exceeds their storage rank
    (o_proj passes wo_a's plain 2-D [h*d, r] linear weight for subscript
    'hdr'); torch.einsum does not. Split the storage using dims already
    pinned by the output/other operands; at most one letter may remain
    unknown (reshape -1 infers it)."""
    if tensor.ndim == len(sub):
        for ch, n in zip(sub, tensor.shape):
            dims.setdefault(ch, n)
        return tensor
    shape = [dims.get(ch, -1) for ch in sub]
    assert shape.count(-1) <= 1, (
        f"fallback einsum cannot infer '{sub}' from storage "
        f"{tuple(tensor.shape)} with known dims {dims}")
    view = tensor.reshape(shape)
    for ch, n in zip(sub, view.shape):
        dims.setdefault(ch, n)
    return view


def _torch_fp8_einsum(subscripts, a, b, out, **_kwargs) -> torch.Tensor:
    _warn_fallback("fp8_einsum")
    av = _expand_block_scales(*a) if isinstance(a, tuple) else a.float()
    bv = _expand_block_scales(*b) if isinstance(b, tuple) else b.float()
    lhs, out_sub = subscripts.replace(" ", "").split("->")
    sub_a, sub_b = lhs.split(",")
    dims: dict = {ch: n for ch, n in zip(out_sub, out.shape)}
    av = _einsum_operand(sub_a, av, dims)
    bv = _einsum_operand(sub_b, bv, dims)
    _log_fallback_shapes(
        "fp8_einsum",
        f"'{subscripts}' a{tuple(a[0].shape) if isinstance(a, tuple) else tuple(a.shape)}"
        f"->{tuple(av.shape)} b{tuple(b[0].shape) if isinstance(b, tuple) else tuple(b.shape)}"
        f"->{tuple(bv.shape)} out{tuple(out.shape)} {out.dtype}")
    out.copy_(
        torch.einsum(f"{sub_a},{sub_b}->{out_sub}", av, bv).to(out.dtype))
    return out


def _torch_fp8_mqa_logits(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke,
                          clean_logits: bool = False) -> torch.Tensor:
    """Reference lightning-indexer logits (prefill, unpaged):
    logits[m, n] = sum_h weights[m, h] * relu(q[m, h, :] . k[n, :]),
    -inf outside [ks[m], ke[m]). FP8 path only (q scale folded into
    weights; k dequantized by its per-row scale)."""
    _warn_fallback("fp8_fp4_mqa_logits")
    qv, qs = q
    if qs is not None:
        raise RuntimeError(
            "torch fallback supports the FP8 indexer path only (q_scale "
            "is not None -> FP4 Q); serve pre-SM90 with the FP8 indexer "
            "cache")
    k, k_scale = kv
    _log_fallback_shapes(
        "fp8_fp4_mqa_logits",
        f"q{tuple(qv.shape)} {qv.dtype} k{tuple(k.shape)} {k.dtype} "
        f"k_scale{tuple(k_scale.shape)} {k_scale.dtype} "
        f"weights{tuple(weights.shape)} ks/ke{tuple(cu_seqlen_ks.shape)}")
    kf = k.float() * _scales_to_fp32(k_scale).view(-1, 1)  # [N, D]
    qf = qv.float()                                       # [M, H, D]
    M, H, _ = qf.shape
    n = kf.shape[0]
    logits = torch.zeros(M, n, dtype=torch.float32, device=kf.device)
    wf = weights.float()
    for h in range(H):                       # bounds the [M, N] temporaries
        logits += torch.relu(qf[:, h, :] @ kf.T) * wf[:, h:h + 1]
    pos = torch.arange(n, device=kf.device)
    mask = ((pos[None, :] >= cu_seqlen_ks[:, None])
            & (pos[None, :] < cu_seqlen_ke[:, None]))
    return logits.masked_fill(~mask, float("-inf"))


def _torch_fp8_paged_mqa_logits(q, kv_cache, weights, context_lens,
                                block_tables, schedule_metadata,
                                max_model_len: int,
                                clean_logits: bool = False) -> torch.Tensor:
    """Reference paged variant (decode): FP8 indexer cache layout
    [num_blocks, block_size, 1, D+4] u8 — D fp8 bytes then a 4-byte f32
    per-position scale. context_lens [B] (per-request length; draft row j
    sees length-(next_n-1-j)) or [B, next_n] (explicit per-row lengths)."""
    _warn_fallback("fp8_fp4_paged_mqa_logits")
    qv, qs = q
    if qs is not None:
        raise RuntimeError(
            "torch fallback supports the FP8 indexer path only (FP4 Q is "
            "not wired); serve pre-SM90 with the FP8 indexer cache")
    bsz, next_n, _h, d = qv.shape
    block_size = kv_cache.shape[1]
    _log_fallback_shapes(
        "fp8_fp4_paged_mqa_logits",
        f"q{tuple(qv.shape)} {qv.dtype} kv_cache{tuple(kv_cache.shape)} "
        f"{kv_cache.dtype} weights{tuple(weights.shape)} "
        f"context_lens{tuple(context_lens.shape)} "
        f"block_tables{tuple(block_tables.shape)} mml={max_model_len}")
    dev = qv.device
    num_blocks = kv_cache.shape[0]
    if context_lens.ndim == 2:
        lens2d = context_lens.to(dev)
    else:
        offs = torch.arange(next_n, device=dev, dtype=torch.int32)
        lens2d = context_lens.to(dev)[:, None] - (next_n - 1 - offs)[None, :]
    # Capture-safe: no host sync, no data-dependent shapes/branches. Gather a
    # STATIC number of blocks (block_tables.shape[1]); padding slots are
    # clamped in-bounds and masked out, so the result is identical to the old
    # dynamic-slice version but the op graph is value-independent — required
    # inside CUDA-graph capture. The old int(lens2d[b].max().item()) synced
    # the host and crashed decode graph capture on Ada (2026-07-19).
    lens2d = lens2d.clamp(min=0)
    max_nblk = block_tables.shape[1]
    width = max_nblk * block_size
    w_cols = min(width, max_model_len)
    logits = torch.full((bsz * next_n, max_model_len), float("-inf"),
                        dtype=torch.float32, device=dev)
    wf = weights.float()
    pos = torch.arange(width, device=dev)
    for b in range(bsz):
        blocks = block_tables[b].long().clamp_(0, num_blocks - 1)
        kb = kv_cache[blocks].reshape(width, d + 4)
        kf = (kb[:, :d].view(torch.float8_e4m3fn).float()
              * kb[:, d:].contiguous().view(torch.float32).view(-1, 1))
        qf = qv[b].float()                                # [next_n, H, D]
        s = torch.einsum("jhd,sd->jhs", qf, kf)
        w_b = wf[b * next_n:(b + 1) * next_n]             # [next_n, H]
        row = (torch.relu(s) * w_b[:, :, None]).sum(1)    # [next_n, width]
        row = row.masked_fill(pos[None, :] >= lens2d[b][:, None],
                              float("-inf"))
        logits[b * next_n:(b + 1) * next_n, :w_cols] = row[:, :w_cols]
    return logits


def fp8_einsum(*args, **kwargs):
    if _use_torch_fallback():
        return _torch_fp8_einsum(*args, **kwargs)
    _lazy_init()
    if _fp8_einsum_impl is None:
        return _missing(*args, **kwargs)
    return _fp8_einsum_impl(*args, **kwargs)


def m_grouped_fp8_gemm_nt_contiguous(*args, **kwargs):
    _lazy_init()
    if _grouped_impl is None:
        return _missing(*args, **kwargs)
    return _grouped_impl(
        *args, disable_ue8m0_cast=not is_deep_gemm_e8m0_used(), **kwargs
    )


def m_grouped_fp8_fp4_gemm_nt_contiguous(*args, **kwargs):
    _lazy_init()
    if _grouped_fp4_impl is None:
        return _missing(*args, **kwargs)
    return _grouped_fp4_impl(
        *args, disable_ue8m0_cast=not is_deep_gemm_e8m0_used(), **kwargs
    )


def fp8_m_grouped_gemm_nt_masked(*args, **kwargs):
    _lazy_init()
    if _grouped_masked_impl is None:
        return _missing(*args, **kwargs)
    return _grouped_masked_impl(
        *args, disable_ue8m0_cast=not is_deep_gemm_e8m0_used(), **kwargs
    )


def transform_sf_into_required_layout(*args, **kwargs):
    _lazy_init()
    if _transform_sf_into_required_layout_impl is None:
        return _missing(*args, **kwargs)
    return _transform_sf_into_required_layout_impl(
        *args, disable_ue8m0_cast=not is_deep_gemm_e8m0_used(), **kwargs
    )


def fp8_fp4_mqa_logits(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool,
) -> torch.Tensor:
    """Compute MQA logits for a single sequence without KV paging.

    Unified FP8/FP4 dispatch — the underlying DeepGEMM kernel takes
    ``q = (values, scales_or_None)`` where ``scales`` is None for FP8 Q
    (per-token scale is folded into ``weights``) and a packed block-scale
    tensor for MXFP4 Q.

    Args:
        q: Tuple ``(q_values, q_scale)``. FP8 path: q_values is [M, H, D]
            float8_e4m3fn and q_scale is None (per-token scale is folded
            into ``weights``). FP4 path: q_values is packed uint8 and
            q_scale is the companion block-scale tensor.
        kv: Tuple `(k_packed, k_scales)` — FP8 layout is [N, D]
            float8_e4m3fn plus fp32 scales [N]; FP4 layout is packed uint8.
        weights: weights of shape [M, H], dtype `torch.float32`.
        cu_seqlen_ks: Start indices (inclusive) for valid K per query
            position, shape [M], dtype int32.
        cu_seqlen_ke: End indices (exclusive) for valid K per query
            position, shape [M], dtype int32.
        clean_logits: Whether to clean the unfilled logits into `-inf`.

    Returns:
        Logits tensor of shape [M, N], dtype `torch.float32`.
    """
    if _use_torch_fallback():
        return _torch_fp8_mqa_logits(q, kv, weights, cu_seqlen_ks,
                                      cu_seqlen_ke, clean_logits)
    _lazy_init()
    if _fp8_fp4_mqa_logits_impl is None:
        return _missing()
    return _fp8_fp4_mqa_logits_impl(
        q,
        kv,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        clean_logits=clean_logits,
    )


def get_paged_mqa_logits_metadata(
    context_lens: torch.Tensor, block_size: int, num_sms: int
) -> torch.Tensor:
    """Build scheduling metadata for paged MQA logits.

    Args:
        context_lens: Tensor of shape [B], dtype int32; effective context length
            per batch element.
        block_size: KV-cache block size in tokens (e.g., 64).
        num_sms: Number of SMs available. 132 for Hopper

    Returns:
        Backend-specific tensor consumed by `fp8_fp4_paged_mqa_logits` to
        schedule work across SMs.
    """
    if _use_torch_fallback():
        # the torch fallback ignores scheduling metadata; shape matches
        # the (num_sms + 1, 2) i32 buffer the indexer copies into
        return torch.zeros((num_sms + 1, 2), dtype=torch.int32,
                           device=context_lens.device)
    _lazy_init()
    if _get_paged_mqa_logits_metadata_impl is None:
        return _missing()
    return _get_paged_mqa_logits_metadata_impl(context_lens, block_size, num_sms)


def fp8_fp4_paged_mqa_logits(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor,
    max_model_len: int,
    clean_logits: bool,
) -> torch.Tensor:
    """Compute MQA logits using a paged KV-cache.

    Unified FP8/FP4 dispatch — the underlying DeepGEMM kernel takes
    ``q = (values, scales_or_None)``; pass ``(q_tensor, None)`` for the FP8
    path and ``(q_values, q_scale)`` for MXFP4.

    Args:
        q: Tuple ``(q_values, q_scale)``. FP8 path: q_values is
            [B, next_n, H, D] float8_e4m3fn and q_scale is None. FP4 path:
            q_values is packed uint8 and q_scale is the companion
            block-scale tensor.
        kv_cache: Paged KV-cache. FP8 layout is [num_blocks, block_size, 1,
            D+4], dtype `torch.uint8`, with the last 4 bytes per (block, pos)
            storing the float dequant scale.
        weights: Tensor of shape [B * next_n, H], dtype `torch.float32`.
        context_lens: Tensor of shape [B], dtype int32; effective context length
            for each batch element.
        block_tables: Tensor of shape [B, max_blocks], dtype int32; maps logical
            block indices to physical blocks in the paged cache.
        schedule_metadata: Returned by `get_paged_mqa_logits_metadata`;
            used to distribute work across SMs.
        max_model_len: Maximum sequence length used to size the logits output.
        clean_logits: Whether to clean the unfilled logits into `-inf`.

    Returns:
        Logits tensor of shape [B * next_n, max_model_len], dtype
        `torch.float32`.
    """
    if _use_torch_fallback():
        # FP8 indexer path: prefer the performant, capture-safe Triton port
        # (self-test-gated; maybe_* returns None until validated on this
        # silicon at layer init). Fall through to the capture-safe torch
        # reference otherwise. Both avoid the host sync that crashed
        # DeepGEMM's absent-kernel path inside decode graph capture.
        qv, qs = q
        if qs is None:
            from vllm.v1.attention.ops.triton_paged_mqa_logits_dsv4 import (
                maybe_paged_mqa_logits_dsv4)
            out = maybe_paged_mqa_logits_dsv4(qv, kv_cache, weights,
                                              context_lens, block_tables,
                                              max_model_len)
            if out is not None:
                return out
        return _torch_fp8_paged_mqa_logits(q, kv_cache, weights,
                                           context_lens, block_tables,
                                           schedule_metadata,
                                           max_model_len, clean_logits)
    _lazy_init()
    if _fp8_fp4_paged_mqa_logits_impl is None:
        return _missing()
    return _fp8_fp4_paged_mqa_logits_impl(
        q,
        kv_cache,
        weights,
        context_lens,
        block_tables,
        schedule_metadata,
        max_model_len,
        clean_logits=clean_logits,
    )


def tf32_hc_prenorm_gemm(
    x: torch.Tensor,
    fn: torch.Tensor,
    out: torch.Tensor,
    sqrsum: torch.Tensor,
    num_split: int,
) -> torch.Tensor:
    """
    Perform the following computation:
        out = x.float() @ fn.T
        sqrsum = x.float().square().sum(-1)

    See the caller function for shape requirement
    """
    _lazy_init()
    if _tf32_hc_prenorm_gemm_impl is None:
        return _missing()
    return _tf32_hc_prenorm_gemm_impl(
        x,
        fn,
        out,
        sqrsum,
        num_split,
    )


def _ceil_to_ue8m0(x: torch.Tensor):
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


def _align(x: int, y: int) -> int:
    return cdiv(x, y) * y


# Taken from https://github.com/deepseek-ai/DeepGEMM/blob/v2.1.1/csrc/utils/math.hpp#L19
def get_tma_aligned_size(x: int, element_size: int) -> int:
    return _align(x, 16 // element_size)


DEFAULT_BLOCK_SIZE = [128, 128]


# Taken from https://github.com/deepseek-ai/DeepGEMM/blob/dd6ed14acbc7445dcef224248a77ab4d22b5f240/deep_gemm/utils/math.py#L38
@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
def per_block_cast_to_fp8(
    x: torch.Tensor, block_size: list[int] = DEFAULT_BLOCK_SIZE, use_ue8m0: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_dtype = current_platform.fp8_dtype()
    assert x.dim() == 2
    m, n = x.shape
    block_m, block_n = block_size
    x_padded = torch.zeros(
        (_align(m, block_m), _align(n, block_n)), dtype=x.dtype, device=x.device
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, block_m, x_padded.size(1) // block_n, block_n)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    _, fp8_max = get_fp8_min_max()
    sf = x_amax / fp8_max
    sf = _ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = (x_view * (1.0 / sf)).to(fp8_dtype)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), sf.view(
        x_view.size(0), x_view.size(2)
    )


def calc_diff(x: torch.Tensor, y: torch.Tensor):
    """Return a global difference metric for unit tests.

    DeepGEMM kernels on Blackwell/B200 currently exhibit noticeable per-element
    error, causing `torch.testing.assert_close` to fail.  Instead of checking
    every element, we compute a cosine-style similarity over the whole tensor
    and report `1 - sim`.  Once kernel accuracy improves this helper can be
    removed.
    """

    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def should_use_deepgemm_for_fp8_linear(
    output_dtype: torch.dtype,
    weight_shape: tuple[int, int],
    supports_deep_gemm: bool | None = None,
):
    if supports_deep_gemm is None:
        supports_deep_gemm = is_deep_gemm_supported()

    # Verify DeepGEMM N/K dims requirements
    # NOTE: Also synchronized with test_w8a8_block_fp8_deep_gemm_matmul
    # test inside kernels/quantization/test_block_fp8.py
    N_MULTIPLE = 64
    K_MULTIPLE = 128

    return (
        supports_deep_gemm
        and output_dtype == torch.bfloat16
        and weight_shape[0] % N_MULTIPLE == 0
        and weight_shape[1] % K_MULTIPLE == 0
    )


__all__ = [
    "calc_diff",
    "DeepGemmQuantScaleFMT",
    "fp8_gemm_nt",
    "fp8_einsum",
    "m_grouped_fp8_gemm_nt_contiguous",
    "m_grouped_fp8_fp4_gemm_nt_contiguous",
    "fp8_m_grouped_gemm_nt_masked",
    "fp8_fp4_mqa_logits",
    "fp8_fp4_paged_mqa_logits",
    "get_paged_mqa_logits_metadata",
    "per_block_cast_to_fp8",
    "is_deep_gemm_e8m0_used",
    "is_deep_gemm_supported",
    "get_num_sms",
    "set_num_sms",
    "should_use_deepgemm_for_fp8_linear",
    "get_col_major_tma_aligned_tensor",
    "get_mk_alignment_for_contiguous_layout",
    "get_theoretical_mk_alignment_for_contiguous_layout",
    "pack_ue8m0_to_int",
    "get_mn_major_tma_aligned_packed_ue8m0_tensor",
    "get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor",
]
