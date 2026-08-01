#!/usr/bin/env python3
"""CPU regression for identity-gated expert I/O skipping.

Proves the cached-layer predicate composes with vLLM's existing EP filter and,
more importantly, is consulted before ``safe_open.get_tensor``.
"""
import os

import torch

os.environ.setdefault("VLLM_MOE_W2", "1")
os.environ.setdefault("VLLM_MOE_W2_NUM_LAYERS", "43")
os.environ.setdefault("VLLM_MOE_W2_DELTA_GB", "0")

from vllm.model_executor.layers.quantization.utils import moe_w2_cubit
from vllm.model_executor.layers.quantization.utils import moe_w2_delta
from vllm.model_executor.layers.quantization.utils import moe_w2_store
from vllm.model_executor.model_loader import weight_utils


def expert(layer: int, eid: int, proj: str, suffix: str) -> str:
    return f"layers.{layer}.ffn.experts.{eid}.{proj}.{suffix}"


moe_w2_cubit._pack_io_skip_layers.clear()
moe_w2_cubit._install_pack_io_skip(0)
moe_w2_cubit._install_pack_io_skip(42)

for layer in (0, 42):
    for projection in ("w1", "w2", "w3"):
        for suffix in ("weight", "scale"):
            name = expert(layer, 255, projection, suffix)
            assert weight_utils.should_skip_weight(name, None), name

kept = [
    expert(1, 0, "w1", "weight"),
    "layers.0.ffn.shared_experts.w1.weight",
    "layers.0.ffn.experts.0.router.weight",
    "layers.0.attn.q_proj.weight",
    "mtp.0.ffn.experts.0.w1.weight",
]
for name in kept:
    assert not weight_utils.should_skip_weight(name, None), name

# Existing EP filtering still applies after the cache predicate declines.
assert weight_utils.should_skip_weight(expert(1, 3, "w1", "weight"), {2})
assert not weight_utils.should_skip_weight(expert(1, 2, "w1", "weight"), {2})

names = [
    expert(0, 0, "w1", "weight"),
    expert(0, 0, "w1", "scale"),
    expert(1, 0, "w1", "weight"),
    "layers.0.attn.q_proj.weight",
]
get_tensor_calls: list[str] = []


class FakeSafeOpen:
    def __init__(self, _path, framework):
        assert framework == "pt"

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def keys(self):
        return names

    def get_tensor(self, name):
        get_tensor_calls.append(name)
        if moe_w2_cubit._pack_io_should_skip(name):
            raise AssertionError(f"cached tensor was materialized: {name}")
        return torch.ones(1)


saved = {
    "safe_open": weight_utils.safe_open,
    "fs": weight_utils._get_fs_type,
    "size": weight_utils._get_checkpoints_size_bytes,
    "ram": weight_utils._get_available_ram_bytes,
}
try:
    weight_utils.safe_open = FakeSafeOpen
    weight_utils._get_fs_type = lambda _files: "zfs"
    weight_utils._get_checkpoints_size_bytes = lambda _files: 1
    weight_utils._get_available_ram_bytes = lambda: 1024
    yielded = [name for name, _ in weight_utils.safetensors_weights_iterator(
        ["synthetic.safetensors"], use_tqdm_on_load=False)]
finally:
    weight_utils.safe_open = saved["safe_open"]
    weight_utils._get_fs_type = saved["fs"]
    weight_utils._get_checkpoints_size_bytes = saved["size"]
    weight_utils._get_available_ram_bytes = saved["ram"]

assert yielded == names[2:], (yielded, names[2:])
assert get_tensor_calls == names[2:], get_tensor_calls

# Integration boundary: plan_pack_skip registers the actual transformer layer
# only after the identity/geometry pack probe succeeds.
class FakeParameter:
    def __init__(self, shape):
        self.data = torch.zeros(shape, dtype=torch.uint8)
        self.weight_loader = lambda *_args, **_kwargs: False

    @property
    def shape(self):
        return self.data.shape


class FakeLayer:
    layer_name = "model.layers.7.ffn.experts"
    w13_weight = FakeParameter((2, 4, 2))
    w13_weight_scale = FakeParameter((2, 4, 1))
    w2_weight = FakeParameter((2, 4, 2))
    w2_weight_scale = FakeParameter((2, 4, 1))


saved_pack_has_layer = moe_w2_store.pack_has_layer
saved_base_enabled = moe_w2_delta.base_enabled
saved_delta_enabled = moe_w2_delta.enabled
try:
    moe_w2_store.pack_has_layer = lambda *_args, **_kwargs: True
    moe_w2_delta.base_enabled = lambda: True
    moe_w2_delta.enabled = lambda: False
    moe_w2_cubit._n_created = 0
    fake_layer = FakeLayer()
    assert moe_w2_cubit.plan_pack_skip(fake_layer)
finally:
    moe_w2_store.pack_has_layer = saved_pack_has_layer
    moe_w2_delta.base_enabled = saved_base_enabled
    moe_w2_delta.enabled = saved_delta_enabled

assert moe_w2_cubit._pack_io_skip_layers == {7}
assert weight_utils.should_skip_weight(expert(7, 1, "w2", "weight"), None)
assert not weight_utils.should_skip_weight(expert(6, 1, "w2", "weight"), None)
for param_name in (
    "w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"
):
    param = getattr(fake_layer, param_name)
    assert param.data.numel() == 0
    assert param.weight_loader(return_success=True)
print(
    "PACK_IO_SKIP PASS cached=2 materialized=2 "
    f"layers={sorted(moe_w2_cubit._pack_io_skip_layers)}")
