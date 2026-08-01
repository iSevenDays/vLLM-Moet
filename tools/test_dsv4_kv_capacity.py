#!/usr/bin/env python3
"""CPU-only DSV4 packed-KV capacity and representation regression."""

import math
from types import SimpleNamespace as NS

import torch

from vllm.v1.core.kv_cache_utils import (
    generate_scheduler_kv_cache_config,
    get_kv_cache_capacity,
    get_kv_cache_groups,
    get_num_blocks_per_request_for_kv_cache_config,
)
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)


cfg = NS(
    scheduler_config=NS(
        disable_hybrid_kv_cache_manager=False,
        max_num_batched_tokens=1056,
    ),
    speculative_config=NS(use_eagle=lambda: False),
    model_config=NS(max_model_len=262144),
    parallel_config=NS(
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
    ),
)
register_all_kvcache_specs(cfg)
specs = {}
for i in range(43):
    ratio = 1 if i < 2 else (4 if i % 2 == 0 else 128)
    if ratio == 4:
        specs[f"L{i}.idx.k"] = MLAAttentionSpec(
            block_size=256, num_kv_heads=1, head_size=132,
            dtype=torch.uint8, compress_ratio=4, alignment=576)
        specs[f"L{i}.idx.state"] = SlidingWindowMLASpec(
            block_size=4, num_kv_heads=1, head_size=512,
            dtype=torch.float32, sliding_window=8, alignment=576)
    specs[f"L{i}.swa"] = SlidingWindowMLASpec(
        block_size=64, num_kv_heads=1, head_size=512,
        dtype=torch.uint8, sliding_window=128,
        cache_dtype_str="fp8_ds_mla", alignment=576,
        model_version="deepseek_v4")
    if ratio > 1:
        specs[f"L{i}.main"] = MLAAttentionSpec(
            block_size=256, num_kv_heads=1, head_size=512,
            dtype=torch.uint8, compress_ratio=ratio,
            cache_dtype_str="fp8_ds_mla", alignment=576,
            model_version="deepseek_v4")
        specs[f"L{i}.main.state"] = SlidingWindowMLASpec(
            block_size=4 if ratio == 4 else 8,
            num_kv_heads=1, head_size=2048 if ratio == 4 else 1024,
            dtype=torch.float32, sliding_window=8 if ratio == 4 else 128,
            alignment=576)

# DSpark contributes three uncompressed draft layers with SWA caches.
for i in range(43, 46):
    specs[f"L{i}.swa"] = SlidingWindowMLASpec(
        block_size=64, num_kv_heads=1, head_size=512,
        dtype=torch.uint8, sliding_window=128,
        cache_dtype_str="fp8_ds_mla", alignment=576,
        model_version="deepseek_v4")

groups = get_kv_cache_groups(cfg, specs)
worker = KVCacheConfig(
    num_blocks=4836, kv_cache_tensors=[], kv_cache_groups=groups)
worker_blocks = get_num_blocks_per_request_for_kv_cache_config(cfg, worker)
assert worker_blocks == [1024, 20, 20, 267, 149], worker_blocks

scheduler = generate_scheduler_kv_cache_config([worker])
scheduler_blocks = get_num_blocks_per_request_for_kv_cache_config(cfg, scheduler)
assert scheduler_blocks == worker_blocks
worker_capacity = get_kv_cache_capacity(cfg, worker)
scheduler_capacity = get_kv_cache_capacity(cfg, scheduler)
assert worker_capacity == scheduler_capacity
assert worker_capacity == (856573, 4836 / 1480), worker_capacity

# Uniform wrapper insertion order must not change scheduler unwrapping capacity.
for group in groups:
    if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
        object.__setattr__(
            group.kv_cache_spec,
            "kv_cache_specs",
            dict(reversed(group.kv_cache_spec.kv_cache_specs.items())),
        )
reversed_scheduler = generate_scheduler_kv_cache_config([worker])
assert get_num_blocks_per_request_for_kv_cache_config(
    cfg, reversed_scheduler) == worker_blocks
assert get_kv_cache_capacity(cfg, reversed_scheduler) == worker_capacity

strict_required = 3 * sum(worker_blocks) + 1  # one globally reserved null block
assert strict_required == 4441 < worker.num_blocks
headroom_blocks = worker.num_blocks - strict_required
packed_stride = 1_002_240
assert headroom_blocks == 395
assert math.isclose(headroom_blocks * packed_stride / 2**20,
                    377.545166015625)

print(
    "DSV4_KV_CAPACITY PASS "
    f"blocks={worker_blocks} pool={worker.num_blocks} "
    f"strict_3x={strict_required} headroom={headroom_blocks} "
    f"tokens={worker_capacity[0]} concurrency={worker_capacity[1]:.6f}")
