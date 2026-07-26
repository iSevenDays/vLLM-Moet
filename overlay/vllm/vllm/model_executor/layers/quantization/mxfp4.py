# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    FusedMoEMethodBase,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutedExperts,
    SharedExperts,
)
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    TRITON_BACKENDS,
    Mxfp4MoeBackend,
    convert_gpt_oss_weight_to_mxfp4_moe_kernel_format,
    convert_weight_to_mxfp4_moe_kernel_format,
    make_mxfp4_moe_kernel,
    make_mxfp4_moe_quant_config,
    mxfp4_round_up_hidden_size_and_intermediate_size,
    select_deepseek_v4_mxfp4_moe_backend,
    select_mxfp4_moe_backend,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.utils import replace_parameter, set_weight_attrs

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# IQ2_XXS / Q2_K weight-loading path (Step 2 of the IQ2_XXS port).
#
# Opt-in via VLLM_MOE_W2_IQ2=1. When enabled, Mxfp4MoEMethod.create_weights
# allocates the raw IQ2_XXS / Q2_K block tensors emitted by
# tools/convert_iq2_gguf_to_vllm.py (Step 1) instead of the mxfp4 e2m1 packed
# tensors, and process_weights_after_loading skips the mxfp4 plane build (which
# would otherwise requantize to the {-4,-1,1,4} tsym4 alphabet). The Triton
# GEMM that consumes these raw blocks is Step 4b; apply() raises a clear
# NotImplementedError until that lands. Default (env unset) is unchanged: the
# production {-4,-1,1,4} mxfp4 path.
#
# Block layout (matches utils/iq2_xxs_ref.py Step 0 + the converter):
#   QK_K = 256 elements per block.
#   IQ2_XXS block = 66 bytes  -> row_bytes = (in_features / QK_K) * 66
#   Q2_K    block = 84 bytes  -> row_bytes = (in_features / QK_K) * 84
# Per-expert storage is C-order [out_features, row_bytes]; all experts
# concatenate on axis 0 (the TP-shard dim): TP=2 -> rank r takes
# experts [r*128:(r+1)*128].
# ---------------------------------------------------------------------------
_QK_K = 256
_IQ2_XXS_BLOCK_BYTES = 66
_Q2_K_BLOCK_BYTES = 84

# Process-wide cache for the two IQ2 LUTs that the converter emits as
# _lookup.iq2xxs_grid (uint64[256]) and _lookup.ksigns_iq2xs (uint8[128]).
# DeepseekV4ForCausalLM.load_weights stashes them here; the Step 4b Triton
# kernel will read them (or fall back to its inlined copies).
_IQ2_LUTS: dict[str, torch.Tensor] = {}


def _iq2_enabled() -> bool:
    """VLLM_MOE_W2_IQ2=1 gates the IQ2_XXS/Q2_K weight-loading path."""
    return os.getenv("VLLM_MOE_W2_IQ2", "0") == "1"


def _iq2_row_bytes(in_features: int, block_bytes: int) -> int:
    """Bytes required to store one row of `in_features` elements given a
    block layout of `block_bytes` bytes per QK_K elements."""
    if in_features % _QK_K != 0:
        raise ValueError(
            f"IQ2 in_features ({in_features}) must be a multiple of "
            f"QK_K ({_QK_K}); the block layout cannot span a partial "
            "block. Use a checkpoint whose expert dims are QK_K-aligned."
        )
    return (in_features // _QK_K) * block_bytes


def _iq2_expert_weight_loader(
    param: torch.nn.Parameter,
    loaded_weight: torch.Tensor,
    weight_name: str,
    shard_id: str,  # noqa: ARG001 - kept for signature parity
    expert_id: int,  # noqa: ARG001 - kept for signature parity
    return_success: bool = False,
) -> bool | None:
    """Weight loader for the fused-all-experts IQ2_XXS / Q2_K params.

    The checkpoint tensor packs every expert on axis 0 with the full
    per-expert byte layout intact (no GEMM-dim sharding — IQ2 blocks cannot
    be split without requantizing). TP shards axis 0 (experts): for TP=t and
    rank r, this loader copies loaded_weight[r*L:(r+1)*L] into param, where
    L = param.shape[0] is the local expert count the layer allocated.

    Args:
        param: [local_experts, out_features, row_bytes] uint8 parameter.
        loaded_weight: [global_experts, out_features, row_bytes] uint8 tensor
            straight from the safetensors checkpoint.
        weight_name: mapped param name (unused; signature parity).
        shard_id: w1/w2/w3 (unused; the direction is encoded in the param
            itself since gate/up/down are distinct params).
        expert_id: per-expert id from the mapping (unused; the IQ2 mapping
            emits a single fused entry so this is always 0).
        return_success: when True, return a bool instead of None.

    Returns:
        True (or None) after the copy.
    """
    # Imported lazily so static validation (no torch.distributed init) works.
    from vllm.distributed import (
        get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)

    g_experts = int(loaded_weight.shape[0])
    l_experts = int(param.shape[0])
    if l_experts == g_experts:
        # No expert sharding (TP=1 or experts-replicated TP).
        shard = loaded_weight
    elif l_experts > 0 and l_experts * get_tensor_model_parallel_world_size() \
            == g_experts:
        tp_rank = get_tensor_model_parallel_rank()
        shard = loaded_weight[tp_rank * l_experts:(tp_rank + 1) * l_experts]
    else:
        raise ValueError(
            f"IQ2 expert weight loader cannot satisfy param axis-0 "
            f"({l_experts}) from checkpoint axis-0 ({g_experts}) with "
            f"tp_size={get_tensor_model_parallel_world_size()}. The IQ2 "
            "path shards experts on axis 0; set EP or run TP=1.")

    if tuple(shard.shape) != tuple(param.shape):
        raise ValueError(
            f"IQ2 expert weight loader shape mismatch for {weight_name}: "
            f"param {tuple(param.shape)} vs shard {tuple(shard.shape)} "
            f"(shard_id={shard_id}).")
    param.data.copy_(shard)
    return True if return_success else None


class Mxfp4Config(QuantizationConfig):
    """Canonical base config for MXFP4 quantization.

    Subclasses override get_name() and override_quantization_method() to
    register themselves as the handler for a specific checkpoint format.
    """

    def __init__(self, ignored_layers: list[str] | None = None):
        super().__init__()
        self.ignored_layers = ignored_layers

    @classmethod
    def from_config(cls, config):
        return cls()

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "mxfp4"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    # TODO (zyongye) This is only temporaty fallback.
    # We should have `Mxfp4MoEMethod` after this migration is complete.
    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        if isinstance(layer, LinearBase):
            if self.ignored_layers and is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            logger.debug_once(
                "MXFP4 linear layer is not implemented - falling back to "
                "UnquantizedLinearMethod.",
            )
            return UnquantizedLinearMethod()
        elif isinstance(layer, RoutedExperts):
            return GptOssMxfp4MoEMethod(layer.moe_config)
        elif isinstance(layer, Attention):
            logger.debug_once(
                "MXFP4 attention layer is not implemented. "
                "Skipping quantization for this layer.",
            )
        return None

    def is_mxfp4_quant(self, prefix: str, layer: torch.nn.Module) -> bool:
        """MXFP4 config always uses MXFP4 quantization."""
        return True


class GptOssMxfp4Config(Mxfp4Config):
    """MXFP4 config for GPT-OSS checkpoints.

    Checkpoints carry ``"quant_method": "mxfp4"`` in their JSON config.
    override_quantization_method() maps that to the canonical internal name
    so that the rest of the loading path uses "gpt_oss_mxfp4" consistently.
    """

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "gpt_oss_mxfp4"

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant, hf_config=None
    ) -> QuantizationMethods | None:
        # Match both "mxfp4" (original checkpoint value) and "gpt_oss_mxfp4"
        # (already normalized by verify_and_update_model_config) so that
        # explicit --quantization mxfp4 from the user doesn't cause a mismatch.
        if not (
            isinstance(hf_quant_cfg, dict)
            and hf_quant_cfg.get("quant_method") in ("mxfp4", "gpt_oss_mxfp4")
        ):
            return None
        # Require explicit confirmation that this is a GPT-OSS model.
        # Do NOT fall back to returning the override when hf_config is None,
        # as that would silently claim all mxfp4 checkpoints.
        model_type = getattr(hf_config, "model_type", None)
        if model_type != "gpt_oss":
            return None
        return "gpt_oss_mxfp4"


class GptOssMxfp4MoEMethod(FusedMoEMethodBase):
    """MXFP4 MoE quantization method."""

    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)
        self.weight_dtype = "gpt_oss_mxfp4"
        self.mxfp4_backend, self.experts_cls = select_mxfp4_moe_backend(moe)

        self.max_capture_size = moe.max_capture_size

        self._cache_permute_indices: dict[torch.Size, torch.Tensor] = {}
        self.moe_kernel: mk.FusedMoEKernel | None = None

        # Used for triton kernel precision configs
        self.w13_precision_config = None
        self.w2_precision_config = None

    @property
    def skip_forward_padding(self) -> bool:
        # SM100_FI_MXFP4_MXFP8_TRTLLM supports padding with mxfp8 quant
        # so can skip the padding in the forward before applying the moe method
        return self.mxfp4_backend == Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8

    # TODO(bnell): move to MK/expert_class?
    @property
    def has_unpadded_output(self) -> bool:
        return self.mxfp4_backend in [
            Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8,
            Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16,
        ]

    def maybe_roundup_sizes(
        self,
        hidden_size: int,
        intermediate_size_per_partition: int,
        act_dtype: torch.dtype,
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> tuple[int, int]:
        hidden_size, intermediate_size_per_partition = super().maybe_roundup_sizes(
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            act_dtype=act_dtype,
            moe_parallel_config=moe_parallel_config,
        )
        return mxfp4_round_up_hidden_size_and_intermediate_size(
            self.mxfp4_backend, hidden_size, intermediate_size_per_partition
        )

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        self.num_experts = num_experts
        weight_dtype = torch.uint8
        scale_dtype = torch.uint8
        mxfp4_block = 32

        layer.params_dtype = params_dtype
        layer.num_experts = num_experts
        self.intermediate_size = intermediate_size_per_partition
        self.hidden_size = hidden_size

        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // 2,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // mxfp4_block,
                dtype=scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        w13_weight_scale.quant_method = "block"

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 2,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // mxfp4_block,
                dtype=scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)
        w2_weight_scale.quant_method = "block"

        if self.moe.has_bias:
            w13_bias = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    dtype=torch.bfloat16,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)

            w2_bias = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    hidden_size,
                    dtype=torch.bfloat16,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)

    def _setup_kernel(
        self,
        layer: RoutedExperts,
        w13: torch.Tensor,
        w2: torch.Tensor,
        w13_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        w13_bias: torch.Tensor | None = None,
        w2_bias: torch.Tensor | None = None,
    ) -> None:
        num_experts = self.num_experts
        intermediate_size = self.intermediate_size
        hidden_size = self.hidden_size
        sf_block_size = 32

        # Shape assertions
        assert (
            w13.dim() == 3
            and w13.shape[0] == num_experts
            and w13.shape[1] == intermediate_size * 2
            and w13.shape[2] == hidden_size // 2
        )
        assert (
            w13_scale.dim() == 3
            and w13_scale.shape[0] == num_experts
            and w13_scale.shape[1] == intermediate_size * 2
            and w13_scale.shape[2] == hidden_size // sf_block_size
        )
        assert (
            w2.dim() == 3
            and w2.shape[0] == num_experts
            and w2.shape[1] == hidden_size
            and w2.shape[2] == intermediate_size // 2
        )
        assert (
            w2_scale.dim() == 3
            and w2_scale.shape[1] == hidden_size
            and w2_scale.shape[2] == intermediate_size // sf_block_size
        )
        if w13_bias is not None:
            assert (
                w13_bias.dim() == 2
                and w13_bias.shape[0] == num_experts
                and w13_bias.shape[1] == intermediate_size * 2
            )
        if w2_bias is not None:
            assert (
                w2_bias.dim() == 2
                and w2_bias.shape[0] == num_experts
                and w2_bias.shape[1] == hidden_size
            )

        # Convert weights to kernel format
        w13, w2, w13_scale, w2_scale, w13_bias, w2_bias = (
            convert_gpt_oss_weight_to_mxfp4_moe_kernel_format(
                mxfp4_backend=self.mxfp4_backend,
                layer=layer,
                w13_weight=w13,
                w2_weight=w2,
                w13_weight_scale=w13_scale,
                w2_weight_scale=w2_scale,
                w13_bias=w13_bias,
                w2_bias=w2_bias,
                _cache_permute_indices=self._cache_permute_indices,
            )
        )

        # For TRITON backends, weights are wrapped tensors from triton_kernels
        # that don't support .detach(). Manually assign parameters.
        if self.mxfp4_backend not in TRITON_BACKENDS:
            replace_parameter(layer, "w13_weight", w13)
            replace_parameter(layer, "w2_weight", w2)
            replace_parameter(layer, "w13_weight_scale", w13_scale)
            replace_parameter(layer, "w2_weight_scale", w2_scale)
        else:
            layer.w13_weight = w13
            layer.w2_weight = w2
            self.w13_precision_config = w13_scale
            self.w2_precision_config = w2_scale

        # AITER backend requires weights to be marked as shuffled.
        if self.mxfp4_backend == Mxfp4MoeBackend.AITER_MXFP4_BF16:
            layer.w13_weight.is_shuffled = True
            layer.w2_weight.is_shuffled = True

        if w13_bias is not None and w2_bias is not None:
            replace_parameter(layer, "w13_bias", w13_bias)
            replace_parameter(layer, "w2_bias", w2_bias)

        # Build quant config
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)

        # Build kernel (modular or monolithic)
        if self.moe_quant_config is not None and self.experts_cls is not None:
            self.moe_kernel = make_mxfp4_moe_kernel(
                moe_quant_config=self.moe_quant_config,
                moe_config=self.moe,
                mxfp4_backend=self.mxfp4_backend,
                experts_cls=self.experts_cls,
                routing_tables=layer._expert_routing_tables(),
                layer=layer,
            )

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        w13 = layer.w13_weight
        w2 = layer.w2_weight
        w13_scale = layer.w13_weight_scale
        w2_scale = layer.w2_weight_scale
        w13_bias = getattr(layer, "w13_bias", None)
        w2_bias = getattr(layer, "w2_bias", None)

        if self.mxfp4_backend == Mxfp4MoeBackend.NONE:
            return

        self._setup_kernel(layer, w13, w2, w13_scale, w2_scale, w13_bias, w2_bias)

    def get_fused_moe_quant_config(
        self, layer: RoutedExperts
    ) -> FusedMoEQuantConfig | None:
        w1_bias = getattr(layer, "w13_bias", None)
        w2_bias = getattr(layer, "w2_bias", None)

        if self.mxfp4_backend in TRITON_BACKENDS:
            # TRITON backends free w13/w2_weight_scale after swizzling; the
            # swizzled scales live inside the precision configs instead.
            assert self.w13_precision_config is not None
            assert self.w2_precision_config is not None
            w1_scale = self.w13_precision_config
            w2_scale = self.w2_precision_config
        else:
            w1_scale = layer.w13_weight_scale
            w2_scale = layer.w2_weight_scale

        return make_mxfp4_moe_quant_config(
            mxfp4_backend=self.mxfp4_backend,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
            gemm1_alpha=1.702,
            gemm1_beta=1.0,
            swiglu_limit=7.0,
            layer=layer,
        )

    def select_gemm_impl(
        self,
        prepare_finalize: mk.FusedMoEPrepareAndFinalize,
        layer: RoutedExperts,
    ) -> mk.FusedMoEExpertsModular:
        raise ValueError(
            f"{self.__class__.__name__} uses the new modular kernel "
            "initialization logic. This function should not be called."
        )

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        assert not self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            expert_map=layer.expert_map,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply_monolithic(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            router_logits=router_logits,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
        )


class Mxfp4MoEMethod(FusedMoEMethodBase):
    """MXFP4 MoE quantization method."""

    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)
        self.weight_dtype = "mxfp4"
        self.mxfp4_backend, self.experts_cls = select_deepseek_v4_mxfp4_moe_backend(moe)

        self.max_capture_size = moe.max_capture_size

        self._cache_permute_indices: dict[torch.Size, torch.Tensor] = {}
        self.moe_kernel: mk.FusedMoEKernel | None = None

        # Used for triton kernel precision configs
        self.w13_precision_config = None
        self.w2_precision_config = None

        # VLLM_MOE_W2_IQ2=1 swaps the weight-loading path to the raw
        # IQ2_XXS / Q2_K block layout (Step 2 of the IQ2_XXS port). When
        # active, create_weights allocates the IQ2 params instead of the
        # mxfp4 e2m1 packed tensors, and process_weights_after_loading
        # skips the mxfp4 plane build (which would requantize to the
        # {-4,-1,1,4} tsym4 alphabet). The Triton GEMM that consumes these
        # raw blocks is Step 4b; apply() raises a clear NotImplementedError
        # until that lands. Default (env unset) is unchanged.
        self._iq2_active = _iq2_enabled()

    @property
    def supports_eplb(self) -> bool:
        return True

    @property
    def skip_forward_padding(self) -> bool:
        # SM100_FI_MXFP4_MXFP8_TRTLLM supports padding with mxfp8 quant
        # so can skip the padding in the forward before applying the moe method
        return self.mxfp4_backend == Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8

    # TODO(bnell): move to MK/expert_class?
    @property
    def has_unpadded_output(self) -> bool:
        return self.mxfp4_backend in [
            Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8,
            Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16,
        ]

    def maybe_roundup_sizes(
        self,
        hidden_size: int,
        intermediate_size_per_partition: int,
        act_dtype: torch.dtype,
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> tuple[int, int]:
        hidden_size, intermediate_size_per_partition = super().maybe_roundup_sizes(
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            act_dtype=act_dtype,
            moe_parallel_config=moe_parallel_config,
        )
        return mxfp4_round_up_hidden_size_and_intermediate_size(
            self.mxfp4_backend, hidden_size, intermediate_size_per_partition
        )

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        self.num_experts = num_experts
        weight_dtype = torch.uint8
        scale_dtype = torch.uint8
        mxfp4_block = 32

        layer.params_dtype = params_dtype
        layer.num_experts = num_experts
        self.intermediate_size = intermediate_size_per_partition
        self.hidden_size = hidden_size

        # VLLM_MOE_W2_IQ2=1: allocate the raw IQ2_XXS / Q2_K block params
        # and return early, SKIPPING the mxfp4 e2m1 / scale / bias alloc
        # below. The IQ2 block layout cannot be split on GEMM dims without
        # requantizing, so the IQ2 path shards EXPERTS across TP ranks
        # (128/card at TP=2; see _create_iq2_weights) and the runner's final
        # all-reduce sums the per-rank partial routed outputs. Skipping the
        # mxfp4 alloc here is VRAM-mandatory on a 48 GB card: the mxfp4
        # w13/w2 params would otherwise sit idle on top of the IQ2 params
        # the path actually uses. The host-RAM staging (VLLM_MOE_W2) below
        # is also skipped.
        if self._iq2_active:
            self._create_iq2_weights(
                layer=layer,
                num_experts=num_experts,
                extra_weight_attrs=extra_weight_attrs,
            )
            return

        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // 2,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // mxfp4_block,
                dtype=scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        w13_weight_scale.quant_method = "block"

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 2,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // mxfp4_block,
                dtype=scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)
        w2_weight_scale.quant_method = "block"

        if self.moe.has_bias:
            w13_bias = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    dtype=torch.bfloat16,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)

            w2_bias = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    hidden_size,
                    dtype=torch.bfloat16,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)
        # VLLM_MOE_W2: the raw checkpoint experts of all layers do not fit a
        # single GPU; stage them in host RAM until the 2-bit planes are built
        # in process_weights_after_loading.
        from vllm.model_executor.layers.quantization.utils import moe_w2_cubit
        if moe_w2_cubit.is_w2_layer(getattr(layer, "layer_name", "")):
            for pname in ("w13_weight", "w13_weight_scale", "w2_weight",
                          "w2_weight_scale"):
                p_ = getattr(layer, pname)
                attrs = {k: getattr(p_, k) for k in ("weight_loader",)
                         if hasattr(p_, k)}
                newp = torch.nn.Parameter(p_.data.cpu(), requires_grad=False)
                layer.register_parameter(pname, newp)
                set_weight_attrs(newp, attrs)
                if pname.endswith("_scale"):
                    newp.quant_method = "block"
            # stream-build: requant each layer the moment its last expert
            # tensor lands and drop its staging in place — peak host RAM
            # stays O(layers in flight) instead of the full checkpoint
            # (~140+ GB on DS4-Flash; the silent full-staging fallback
            # here OOM-killed three Ada deploy cycles with no traceback).
            if not moe_w2_cubit.arm_stream_build(layer):
                moe_w2_cubit.warn_full_staging(layer, "mxfp4 arm failed")

    def _create_iq2_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        extra_weight_attrs: dict,
    ) -> None:
        """Allocate raw IQ2_XXS / Q2_K block params for the Step 2 loader.

        Three params are registered on ``layer``, matching the safetensors
        checkpoint emitted by tools/convert_iq2_gguf_to_vllm.py (Step 1):
          - gate_weight_iq2_xxs  [E_local, intermediate, hidden/QK_K * 66]  u8
          - up_weight_iq2_xxs    [E_local, intermediate, hidden/QK_K * 66]  u8
          - down_weight_q2_k     [E_local, hidden, intermediate/QK_K * 84]  u8

        The IQ2_XXS/Q2_K block layout spans the FULL in_features dimension
        (QK_K=256 elements per block), so we must use the unsharded sizes
        from FusedMoEConfig even when the surrounding mxfp4 path would
        shard them. IQ2 blocks cannot be tensor-sharded on the GEMM dims
        (splitting a 256-element block would requantize), so the IQ2 path
        shards EXPERTS instead: under TP=2 each rank allocates
        ``num_experts // tp_size`` experts (128/card) and the per-rank
        partial routed outputs are summed by the runner's final all-reduce
        (``reduce_results=True`` default on FusedMoE; the reduce fires while
        ``tp_size > 1`` and ``_fused_output_is_reduced`` is False for this
        non-modular path).

        Expert-parallel self-sharding: vLLM defaults
        ``enable_expert_parallel=False``, so the production DSv4 TP=2 path
        is TP-replicated (``num_local_experts=256``, ``expert_map=None``).
        That would replicate all 256 experts per card and OOM a 48 GB GPU.
        When the layer has no expert_map and TP>1 divides num_experts, we
        therefore synthesize the linear map (rank r owns globals
        ``[r*L:(r+1)*L]``) and register it as the layer's ``_expert_map``
        buffer. ``_iq2_forward`` reads ``layer.expert_map`` to map global
        topk_ids to local expert ids; non-local experts contribute zero on
        this rank and the all-reduce completes the sum. The custom loader
        (``_iq2_expert_weight_loader``) shards axis 0 by TP rank to match.
        """
        hidden_full = self.moe.hidden_dim
        # self.moe.intermediate_size is the FULL intermediate; per-partition
        # is self.moe.intermediate_size_per_partition. We need full here.
        intermediate_full = self.moe.intermediate_size

        gate_row_bytes = _iq2_row_bytes(hidden_full, _IQ2_XXS_BLOCK_BYTES)
        up_row_bytes = _iq2_row_bytes(hidden_full, _IQ2_XXS_BLOCK_BYTES)
        down_row_bytes = _iq2_row_bytes(intermediate_full, _Q2_K_BLOCK_BYTES)

        # Expert-parallel self-sharding under TP>1 (see docstring).
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )
        tp_size = get_tensor_model_parallel_world_size()
        existing_map = getattr(layer, "expert_map", None)
        if existing_map is None and tp_size > 1 and num_experts % tp_size == 0:
            n_local = num_experts // tp_size
            tp_rank = get_tensor_model_parallel_rank()
            expert_map = torch.full((num_experts,), -1, dtype=torch.int32)
            start = tp_rank * n_local
            expert_map[start:start + n_local] = torch.arange(
                n_local, dtype=torch.int32)
            # Override the EP-off None buffer installed by
            # update_expert_map_info. Non-persistent: the map is derived
            # from rank/tp_size, not a checkpoint tensor. It still moves
            # to GPU with the module for the forward's global->local index.
            layer.register_buffer(
                "_expert_map", expert_map, persistent=False)
            logger.info(
                "IQ2 EP self-sharding: rank %s/%s owns %s/%s experts "
                "(global [%s:%s]); runner all-reduces the per-rank "
                "partials.", tp_rank, tp_size, n_local, num_experts, start,
                start + n_local)
        else:
            # EP already enabled at config (expert_map present) or TP=1:
            # this rank owns whatever num_experts was passed in.
            n_local = num_experts

        # Sanity: shapes must match the converter output bit-for-bit.
        # gate/up: [E_local, out=intermediate, row_bytes=hidden/256*66]
        # down:    [E_local, out=hidden,       row_bytes=intermediate/256*84]
        def _register(name: str, out_features: int, row_bytes: int) -> None:
            p = torch.nn.Parameter(
                torch.zeros(
                    n_local,
                    out_features,
                    row_bytes,
                    dtype=torch.uint8,
                ),
                requires_grad=False,
            )
            layer.register_parameter(name, p)
            # set_weight_attrs installs the default RoutedExperts weight
            # loader (and any other extras); we then override it with the
            # IQ2 fused-all-experts loader for axis-0 TP sharding.
            set_weight_attrs(p, extra_weight_attrs)
            p.weight_loader = _iq2_expert_weight_loader

        _register("gate_weight_iq2_xxs", intermediate_full, gate_row_bytes)
        _register("up_weight_iq2_xxs", intermediate_full, up_row_bytes)
        _register("down_weight_q2_k", hidden_full, down_row_bytes)

        # Mark the layer so process_weights_after_loading / apply can detect
        # the IQ2 path even when the method instance is rebound (defensive).
        layer._iq2_active = True  # type: ignore[attr-defined]  # type: ignore[attr-defined]

    def _setup_kernel(
        self,
        layer: RoutedExperts,
        w13: torch.Tensor,
        w2: torch.Tensor,
        w13_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        w13_bias: torch.Tensor | None = None,
        w2_bias: torch.Tensor | None = None,
    ) -> None:
        num_experts = self.num_experts
        intermediate_size = self.intermediate_size
        hidden_size = self.hidden_size
        sf_block_size = 32

        # Shape assertions
        assert (
            w13.dim() == 3
            and w13.shape[0] == num_experts
            and w13.shape[1] == intermediate_size * 2
            and w13.shape[2] == hidden_size // 2
        )
        assert (
            w13_scale.dim() == 3
            and w13_scale.shape[0] == num_experts
            and w13_scale.shape[1] == intermediate_size * 2
            and w13_scale.shape[2] == hidden_size // sf_block_size
        )
        assert (
            w2.dim() == 3
            and w2.shape[0] == num_experts
            and w2.shape[1] == hidden_size
            and w2.shape[2] == intermediate_size // 2
        )
        assert (
            w2_scale.dim() == 3
            and w2_scale.shape[1] == hidden_size
            and w2_scale.shape[2] == intermediate_size // sf_block_size
        )
        if w13_bias is not None:
            assert (
                w13_bias.dim() == 2
                and w13_bias.shape[0] == num_experts
                and w13_bias.shape[1] == intermediate_size * 2
            )
        if w2_bias is not None:
            assert (
                w2_bias.dim() == 2
                and w2_bias.shape[0] == num_experts
                and w2_bias.shape[1] == hidden_size
            )

        # Convert weights to kernel format
        w13, w2, w13_scale, w2_scale, w13_bias, w2_bias = (
            convert_weight_to_mxfp4_moe_kernel_format(
                mxfp4_backend=self.mxfp4_backend,
                layer=layer,
                w13_weight=w13,
                w2_weight=w2,
                w13_weight_scale=w13_scale,
                w2_weight_scale=w2_scale,
                w13_bias=w13_bias,
                w2_bias=w2_bias,
                _cache_permute_indices=self._cache_permute_indices,
            )
        )

        # For TRITON backends, weights are wrapped tensors from triton_kernels
        # that don't support .detach(). Manually assign parameters.
        if self.mxfp4_backend not in TRITON_BACKENDS:
            replace_parameter(layer, "w13_weight", w13)
            replace_parameter(layer, "w2_weight", w2)
            replace_parameter(layer, "w13_weight_scale", w13_scale)
            replace_parameter(layer, "w2_weight_scale", w2_scale)
        else:
            layer.w13_weight = w13
            layer.w2_weight = w2
            self.w13_precision_config = w13_scale
            self.w2_precision_config = w2_scale

        # AITER backend requires weights to be marked as shuffled.
        if self.mxfp4_backend == Mxfp4MoeBackend.AITER_MXFP4_BF16:
            layer.w13_weight.is_shuffled = True
            layer.w2_weight.is_shuffled = True

        if w13_bias is not None and w2_bias is not None:
            replace_parameter(layer, "w13_bias", w13_bias)
            replace_parameter(layer, "w2_bias", w2_bias)

        # Build quant config
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)

        # Build kernel (modular or monolithic)
        if self.moe_quant_config is not None and self.experts_cls is not None:
            self.moe_kernel = make_mxfp4_moe_kernel(
                moe_quant_config=self.moe_quant_config,
                moe_config=self.moe,
                mxfp4_backend=self.mxfp4_backend,
                experts_cls=self.experts_cls,
                routing_tables=layer._expert_routing_tables(),
                layer=layer,
            )

    def process_weights_after_loading(self, layer):
        # VLLM_MOE_W2_IQ2=1: skip the mxfp4 build_layer_planes step. That
        # step requants the loaded mxfp4 e2m1 weights into the
        # {-4,-1,1,4} tsym4 alphabet for the cubit moe_w2_mm kernel; the
        # IQ2 params hold raw IQ2_XXS / Q2_K blocks that must stay
        # bit-for-bit as loaded. Step 3 (the IQ2 plane builder) and Step 4b
        # (the Triton GEMM) will plug in here. For now we leave the raw
        # blocks as-is — the IQ2 kernel reads them directly.
        if self._iq2_active:
            return
        # VLLM_MOE_W2: build 2-bit tensor-sym planes; skip Marlin/other backends.
        from vllm.model_executor.layers.quantization.utils import moe_w2_cubit
        if moe_w2_cubit.is_w2_layer(getattr(layer, "layer_name", "")):
            key = getattr(layer, "_moe_w2_stream_key", None)
            if key is None:
                key = len(moe_w2_cubit._LAYERS)
            if not getattr(layer, "_moe_w2_stream_built", False):
                moe_w2_cubit.build_layer_planes(layer, key)
            layer._moe_w2_key = key
            return

        w13 = layer.w13_weight
        w2 = layer.w2_weight
        w13_scale = layer.w13_weight_scale
        w2_scale = layer.w2_weight_scale
        w13_bias = getattr(layer, "w13_bias", None)
        w2_bias = getattr(layer, "w2_bias", None)

        if self.mxfp4_backend == Mxfp4MoeBackend.NONE:
            return

        self._setup_kernel(layer, w13, w2, w13_scale, w2_scale, w13_bias, w2_bias)

    def get_fused_moe_quant_config(
        self,
        layer: RoutedExperts,
    ) -> FusedMoEQuantConfig | None:
        # VLLM_MOE_W2_IQ2=1: the IQ2 path runs a per-expert Python-loop
        # forward (_iq2_forward) and never builds the modular kernel, so
        # there is no FusedMoEQuantConfig to return. The mxfp4 w13/w2
        # params it would read here were not allocated (skipped in
        # create_weights), so accessing them raises AttributeError.
        # maybe_init_modular_kernel tolerates a None quant config.
        if self._iq2_active:
            return None
        w1_bias = getattr(layer, "w13_bias", None)
        w2_bias = getattr(layer, "w2_bias", None)
        swiglu_limit = getattr(layer, "swiglu_limit", None)

        if self.mxfp4_backend in TRITON_BACKENDS:
            # TRITON backends free w13/w2_weight_scale after swizzling; the
            # swizzled scales live inside the precision configs instead.
            assert self.w13_precision_config is not None
            assert self.w2_precision_config is not None
            w1_scale = self.w13_precision_config
            w2_scale = self.w2_precision_config
        else:
            w1_scale = layer.w13_weight_scale
            w2_scale = layer.w2_weight_scale

        return make_mxfp4_moe_quant_config(
            mxfp4_backend=self.mxfp4_backend,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
            swiglu_limit=swiglu_limit,
            layer=layer,
        )

    def select_gemm_impl(
        self,
        prepare_finalize: mk.FusedMoEPrepareAndFinalize,
        layer: RoutedExperts,
    ) -> mk.FusedMoEExpertsModular:
        raise ValueError(
            f"{self.__class__.__name__} uses the new modular kernel "
            "initialization logic. This function should not be called."
        )

    def _iq2_forward(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Per-expert Python-loop MoE forward for the IQ2_XXS / Q2_K path.

        Mirrors ``cpu_fused_moe_torch`` (the loop-based reference in
        ``fused_moe/cpu_fused_moe.py``): sort flattened (token, topk-slot)
        dispatches by expert, run the per-expert gate/up/down GEMMs, scatter
        back, then combine with the router weights. The per-expert GEMMs are
        the proven fused dequant+dot kernels from Step 4b/4c
        (``iq2_xxs_mm`` for gate/up, ``q2_k_mm`` for down) instead of the
        mxfp4 / moe_w2 kernels.

        Correctness-first: this is intentionally NOT a fused kernel. The
        proven GEMMs take ``[M, K]`` activations (M = tokens for one expert),
        which fits the per-expert dispatch directly. Peak throughput is later
        work.

        Contract: returns ONLY the routed-expert output (shared experts are
        orchestrated by the MoE runner via ``_maybe_apply_shared_experts``
        pre/post, matching the non-modular ``moe_w2_cubit`` path). Under
        expert-parallel TP each rank emits a partial output (non-local
        experts contribute zero here) and the runner all-reduces.

        Args:
            layer: RoutedExperts carrying ``gate_weight_iq2_xxs``,
                ``up_weight_iq2_xxs``, ``down_weight_q2_k`` (registered by
                ``_create_iq2_weights``), ``expert_map``, and
                ``apply_router_weight_on_input``.
            x: bf16 activations ``[num_tokens, hidden]``.
            topk_weights: router weights ``[num_tokens, topk]`` (any dtype;
                combined in fp32).
            topk_ids: global expert ids ``[num_tokens, topk]`` (int).

        Returns:
            bf16 ``[num_tokens, hidden]`` routed-expert output.
        """
        # Lazy imports keep static validation (no torch.distributed) working
        # and avoid importing Triton at module load.
        from vllm.model_executor.layers.quantization.utils import (
            iq2_xxs_mm_triton, q2_k_mm_triton)

        _iq2_nan_check = os.getenv("VLLM_MOE_W2_IQ2_NAN_CHECK", "0") == "1"

        num_tokens, hidden = x.shape
        device = x.device
        topk = topk_ids.shape[1]
        out = torch.zeros(
            num_tokens, hidden, dtype=torch.float32, device=device)

        if num_tokens == 0:
            return out.to(torch.bfloat16)

        # ---- LUTs + per-expert weight slabs -------------------------------
        # grid uint8[2048] + ksigns uint8[128], built from the constant
        # Step-0 tables (bit-exact proven), per-device cached. Self-contained:
        # does not depend on the checkpoint emitting ``_lookup.*`` tensors.
        grid, ksigns = iq2_xxs_mm_triton._get_luts(device)

        gate_w = layer.gate_weight_iq2_xxs  # [E_local, N=intermediate, rb]
        up_w = layer.up_weight_iq2_xxs      # [E_local, N=intermediate, rb]
        down_w = layer.down_weight_q2_k     # [E_local, N=hidden,      rb]

        e_local, gate_n, gate_rb = gate_w.shape
        _, up_n, up_rb = up_w.shape
        _, dn_n, dn_rb = down_w.shape

        # Logical K per family: K = (row_bytes / block_bytes) * QK_K.
        gate_k = (gate_rb // _IQ2_XXS_BLOCK_BYTES) * _QK_K  # = hidden
        up_k = (up_rb // _IQ2_XXS_BLOCK_BYTES) * _QK_K      # = hidden
        dn_k = (dn_rb // _Q2_K_BLOCK_BYTES) * _QK_K         # = intermediate

        if gate_k != hidden or up_k != hidden:
            raise RuntimeError(
                f"IQ2 gate/up K ({gate_k}/{up_k}) != hidden ({hidden}); "
                "checkpoint/layer dim mismatch.")
        if dn_n != hidden:
            raise RuntimeError(
                f"IQ2 down N ({dn_n}) != hidden ({hidden}).")
        if gate_n != up_n:
            raise RuntimeError(
                f"IQ2 gate N ({gate_n}) != up N ({up_n}); gate/up must "
                "share the intermediate dimension.")
        # gate/up: w_shape = (N=intermediate, K=hidden)
        # down:    w_shape = (N=hidden,       K=intermediate)
        gate_shape = (gate_n, gate_k)
        up_shape = (up_n, up_k)
        down_shape = (dn_n, dn_k)

        # ---- Dispatch: global topk_ids -> local expert ids ----------------
        expert_map = getattr(layer, "expert_map", None)
        flat_global = topk_ids.reshape(-1).to(torch.long)
        if expert_map is not None:
            # expert_map[g] = local_id, or -1 if the expert is not on this
            # rank (EP). Non-local dispatches contribute zero here; the
            # runner's all-reduce sums the per-rank partials.
            flat_local = expert_map[flat_global]
        else:
            flat_local = flat_global
            if e_local != layer.global_num_experts:
                raise RuntimeError(
                    f"IQ2 expert_map is None but E_local ({e_local}) != "
                    f"global_num_experts ({layer.global_num_experts}); "
                    "cannot map global topk_ids to local experts. Enable "
                    "expert-parallel or run TP=1.")

        valid_mask = flat_local >= 0
        if not bool(valid_mask.any()):
            return out.to(torch.bfloat16)

        local_ids_v = flat_local[valid_mask]
        slots_v = torch.nonzero(valid_mask, as_tuple=False).flatten()
        token_ids_v = slots_v // topk
        weights_v = topk_weights.reshape(-1).to(torch.float32)[slots_v]

        # Sort dispatches by local expert id so each expert's tokens are
        # contiguous (one segment per expert).
        order = local_ids_v.argsort()
        local_ids_s = local_ids_v[order]
        token_ids_s = token_ids_v[order]
        weights_s = weights_v[order]

        # Gather per-dispatch activations (a token routed to the same expert
        # in two slots appears twice -- mirrors cpu_fused_moe's
        # sorted_tokens expansion; the duplicate-slot weight sum falls out
        # correctly in the combine via index_add_).
        x_bf16 = x if x.dtype == torch.bfloat16 else x.to(torch.bfloat16)
        x_per_slot = x_bf16[token_ids_s]  # [num_dispatches, hidden]

        # If the router weight was already folded into x (topk=1 fast path),
        # don't multiply it in again during combine.
        apply_router_weight_on_input = getattr(
            layer, "apply_router_weight_on_input", False)

        # ---- Per-expert gate/up SwiGLU down + weighted combine ------------
        uniq, counts = torch.unique_consecutive(
            local_ids_s, return_counts=True)
        offsets = torch.cat([
            torch.zeros(1, dtype=torch.long, device=device),
            counts.cumsum(0).to(torch.long),
        ])

        for i in range(int(uniq.numel())):
            expert_id = int(uniq[i].item())
            s = int(offsets[i].item())
            e = int(offsets[i + 1].item())
            x_e = x_per_slot[s:e]
            if x_e.shape[0] == 0:
                continue
            tids_e = token_ids_s[s:e]

            # gate/up: IQ2_XXS fused dequant + dot.
            gate_out = iq2_xxs_mm_triton.iq2_xxs_mm(
                x_e, gate_w[expert_id], gate_shape, grid, ksigns)
            up_out = iq2_xxs_mm_triton.iq2_xxs_mm(
                x_e, up_w[expert_id], up_shape, grid, ksigns)
            # SwiGLU = SiLU(gate) * up (DSv4 routed experts use plain silu;
            # the clamped variant is shared-experts only).
            mid = torch.nn.functional.silu(gate_out) * up_out
            # down: Q2_K fused dequant + dot.
            expert_out = q2_k_mm_triton.q2_k_mm(
                mid, down_w[expert_id], down_shape)  # [M_e, hidden] bf16

            # VLLM_MOE_W2_IQ2_NAN_CHECK: diagnostic. A bad IQ2/Q2_K block can
            # make the per-expert GEMM emit NaN/Inf, which then propagates
            # through the residual into the next layer's sparse-MLA indexer
            # (whose topk over NaN scores yields invalid indices -> illegal
            # memory access, surfacing as an attention crash). When set, log
            # the offending local expert and clamp so the request completes
            # (output is corrupted but the crash is avoided, isolating the
            # cause to the IQ2 GEMM vs the attention path proper).
            if _iq2_nan_check:
                if bool((~torch.isfinite(expert_out)).any()):
                    # Distinguish Inf (overflow, e.g. down acc > bf16 max
                    # from a saturating mid) from NaN (0/0 or inf-inf).
                    out_isnan = bool(torch.isnan(expert_out).any())
                    out_isinf = bool(torch.isinf(expert_out).any())
                    logger.error(
                        "IQ2 non-finite expert_out: local_expert=%s M=%s "
                        "gate|up|mid|out finite=%s/%s/%s/%s "
                        "out_nan=%s out_inf=%s "
                        "max|gate|=%s max|up|=%s max|mid|=%s max|out|=%s",
                        expert_id, int(x_e.shape[0]),
                        bool(torch.isfinite(gate_out).all()),
                        bool(torch.isfinite(up_out).all()),
                        bool(torch.isfinite(mid).all()),
                        bool(torch.isfinite(expert_out).all()),
                        out_isnan, out_isinf,
                        float(gate_out.abs().max()),
                        float(up_out.abs().max()),
                        float(mid.abs().max()),
                        float(expert_out.abs().max()))
                    gate_out = torch.nan_to_num(gate_out)
                    up_out = torch.nan_to_num(up_out)
                    mid = torch.nan_to_num(mid)
                    expert_out = torch.nan_to_num(expert_out)

            if apply_router_weight_on_input:
                contrib = expert_out
            else:
                w_e = weights_s[s:e].to(expert_out.dtype)
                contrib = expert_out * w_e.unsqueeze(-1)
            # index_add_ handles the same token dispatched to this expert in
            # multiple slots (weights sum) and to different experts (partial
            # sums across the loop). fp32 accumulation matches the reference
            # combine precision.
            out.index_add_(0, tids_e, contrib.to(torch.float32))

        return out.to(torch.bfloat16)

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        # VLLM_MOE_W2_IQ2=1: IQ2_XXS / Q2_K routed-expert forward. The
        # proven Step 4b/4c Triton GEMMs are wired in as a per-expert
        # Python-loop MoE forward (correctness-first; the fused fast path is
        # later work). Shared experts stay orchestrated by the MoE runner.
        if self._iq2_active:
            return self._iq2_forward(layer, x, topk_weights, topk_ids)
        # VLLM_MOE_W2 routed-expert path (cubit moe_w2_mm, 2-bit planes).
        # Shared experts are orchestrated by the MoE runner (pre/post
        # _maybe_apply_shared_experts); the non-modular w2 path returns only the
        # routed-expert output, matching the other non-modular applies here.
        w2_key = getattr(layer, "_moe_w2_key", None)
        if w2_key is not None:
            from vllm.model_executor.layers.quantization.utils import (
                moe_w2_cubit)
            assert layer.expert_map is None and \
                not layer.apply_router_weight_on_input
            return moe_w2_cubit.moe_w2_forward(x, topk_weights, topk_ids,
                                               w2_key)

        assert not self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            expert_map=layer.expert_map,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply_monolithic(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            router_logits=router_logits,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
        )
