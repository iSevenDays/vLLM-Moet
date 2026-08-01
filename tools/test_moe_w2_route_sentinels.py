#!/usr/bin/env python3
"""CUDA regression for DSpark sentinel routes before MoE alignment."""

import threading
import torch

from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.quantization.utils import (
    moe_w2_cubit,
    moe_w2_delta,
)


assert torch.cuda.is_available(), "CUDA is required"
device = torch.device("cuda")
n_experts = 256
num_rows = 18
top_k = 6
numel = num_rows * top_k


def fresh_inputs():
    ids = (torch.arange(numel, device=device, dtype=torch.int64)
           % n_experts).view(num_rows, top_k)
    weights = torch.ones((num_rows, top_k), device=device,
                         dtype=torch.float32)
    return weights, ids


def validate(weights, ids, sentinel):
    ids[6] = sentinel
    safe_weights, safe_ids = moe_w2_cubit._sanitize_topk_routes(
        weights, ids, n_experts)
    torch.cuda.synchronize()

    assert bool((safe_ids[6] == n_experts).all())
    assert bool((safe_weights[6] == 0).all())
    assert bool((safe_ids[:6] == ids[:6]).all())
    assert bool((safe_weights[:6] == weights[:6]).all())

    sorted_ids, _expert_ids, num_post = moe_align_block_size(
        safe_ids, 4, n_experts)
    torch.cuda.synchronize()
    used = sorted_ids[:int(num_post.item())]
    real = used[used < numel].cpu()
    expected = torch.cat((torch.arange(36, dtype=torch.int32),
                          torch.arange(42, numel, dtype=torch.int32)))
    torch.testing.assert_close(real.sort().values, expected, rtol=0, atol=0)


for value in (-1, n_experts, n_experts + 997):
    validate(*fresh_inputs(), value)

# The static-shape sanitizer itself must replay safely after an input row is
# mutated from valid routes to DSpark's negative padding sentinel.
graph_weights, graph_ids = fresh_inputs()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    graph_safe_weights, graph_safe_ids = moe_w2_cubit._sanitize_topk_routes(
        graph_weights, graph_ids, n_experts)
graph_ids[6] = -1
graph.replay()
torch.cuda.synchronize()
assert bool((graph_safe_ids[6] == n_experts).all())
assert bool((graph_safe_weights[6] == 0).all())

# Eager prefill uses ensure_resident rather than a captured replay. Exercise a
# mixed valid/sentinel list with a minimal all-resident tier; this reaches the
# host mirror index that previously interpreted -1 as expert 255 and rejected E.
moe_w2_delta._COUNT = True
tier = object.__new__(moe_w2_delta.DeltaTier)
tier._store = {0: object()}
tier.E = n_experts
tier.dev = device
tier.seen = torch.zeros((1, n_experts), device=device, dtype=torch.int32)
tier._seen_host = torch.zeros((1, n_experts), device="cpu", dtype=torch.int32,
                              pin_memory=True)
tier._stream = torch.cuda.Stream(device)
tier._snap_lock = threading.Lock()
tier._lock = threading.Lock()
tier._layer_pins = set()
tier._mirror = torch.zeros((1, n_experts), dtype=torch.int32)
tier._owner_tick = torch.zeros(1, dtype=torch.int64)
tier._tick = 0
mixed = torch.tensor([-1, 0, 42, n_experts, n_experts + 997], device=device)
assert tier.ensure_resident(0, mixed) == 0
torch.cuda.synchronize()
assert int(tier.seen.sum().item()) == 2
assert int(tier.seen[0, 0].item()) == 1
assert int(tier.seen[0, 42].item()) == 1
assert int(tier.seen[0, n_experts - 1].item()) == 0

print("MOE_W2_ROUTE_SENTINELS PASS shape=18 topk=6 align=filtered "
      "graph=safe eager=filtered")
