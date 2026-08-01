#!/usr/bin/env python3
"""Focused CUDA gate for the bounded deterministic MoE unpermute."""

import time

import torch

from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.quantization.utils import moe_w2_cubit


assert torch.cuda.is_available(), "CUDA is required"
device = torch.device("cuda")
torch.manual_seed(20260801)
torch.cuda.manual_seed_all(20260801)

E = 256
TOP_K = 6


def make_case(tokens: int, hidden: int, block: int, sentinels: bool,
              top_k: int = TOP_K):
    route_ids = torch.randint(E, (tokens, top_k), device=device,
                              dtype=torch.int64)
    weights = torch.rand((tokens, top_k), device=device, dtype=torch.float32)
    if sentinels:
        # The serving sanitizer maps negative/out-of-range DSpark padding to E
        # and zeros its weight before moe_align.  Include several such holes.
        holes = torch.arange(0, tokens * top_k,
                             max(tokens * top_k // 7, 1), device=device)
        route_ids.view(-1)[holes] = E
        weights.view(-1)[holes] = 0
    sorted_ids, _expert_blocks, _num_post = moe_align_block_size(
        route_ids, block, E)
    slots = sorted_ids.numel()
    c2 = torch.randn((slots, hidden), device=device,
                     dtype=torch.bfloat16)
    route_slot = torch.empty(slots, device=device, dtype=torch.int32)
    out = torch.empty((tokens, hidden), device=device, dtype=torch.bfloat16)
    return route_ids, weights, sorted_ids, c2, route_slot, out


def reference(c2, sorted_ids, weights, miss_rows):
    tokens, top_k = weights.shape
    dump = tokens * top_k
    valid = (sorted_ids >= 0) & (sorted_ids < dump)
    safe = sorted_ids.clamp(min=0, max=dump - 1).long()
    slot_weights = weights.reshape(-1)[safe]
    slot_weights = torch.where(valid, slot_weights,
                               torch.zeros_like(slot_weights)).float()
    if miss_rows is not None:
        slot_weights *= miss_rows.float()
    dst = torch.where(valid, sorted_ids,
                      torch.full_like(sorted_ids, dump)).long()
    gathered = torch.zeros((dump + 1, c2.shape[1]), device=device,
                           dtype=torch.float32)
    gathered.index_copy_(0, dst, c2.float() * slot_weights.unsqueeze(1))
    return gathered[:dump].view(tokens, top_k, c2.shape[1]).sum(1).to(
        torch.bfloat16)


def run_case(tokens: int, hidden: int, block: int, sentinels: bool,
             with_misses: bool, top_k: int = TOP_K):
    route_ids, weights, sorted_ids, c2, route_slot, out = make_case(
        tokens, hidden, block, sentinels, top_k)
    miss_rows = None
    if with_misses:
        miss_rows = torch.rand(sorted_ids.numel(), device=device) > 0.2

    expected = reference(c2, sorted_ids, weights, miss_rows)
    actual = moe_w2_cubit._deterministic_unpermute(
        c2, sorted_ids, route_ids, weights, miss_rows, route_slot, out, E)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=1.6e-2, atol=1.6e-2)

    # The inverse must cover every non-sentinel flattened route exactly once.
    valid_routes = (route_ids.reshape(-1) >= 0) & (route_ids.reshape(-1) < E)
    flat_routes = torch.arange(tokens * top_k, device=device)[valid_routes]
    mapped_slots = route_slot[flat_routes.long()].long()
    assert bool((sorted_ids[mapped_slots] == flat_routes).all())

    stable = actual.clone()
    for _ in range(25):
        moe_w2_cubit._deterministic_unpermute(
            c2, sorted_ids, route_ids, weights, miss_rows, route_slot, out, E)
    torch.cuda.synchronize()
    assert torch.equal(stable, out), "fused output changed across identical runs"
    diff = actual.float() - expected.float()
    max_abs = float(diff.abs().max().item())
    mismatch = float((actual != expected).float().mean().item())
    rel_l2 = float((torch.linalg.vector_norm(diff) /
                    torch.linalg.vector_norm(expected.float()).clamp_min(
                        1e-20)).item())
    assert mismatch < 1e-3, mismatch
    assert rel_l2 < 1e-4, rel_l2
    return max_abs, mismatch, rel_l2


max_abs = 0.0
max_mismatch = 0.0
max_rel_l2 = 0.0
for args in (
    (1, 257, 4, False, False),
    (18, 4096, 4, True, True),
    (97, 257, 16, True, False),
    (3, 6144, 4, True, True, 8),
):
    case_abs, case_mismatch, case_rel_l2 = run_case(*args)
    max_abs = max(max_abs, case_abs)
    max_mismatch = max(max_mismatch, case_mismatch)
    max_rel_l2 = max(max_rel_l2, case_rel_l2)

# Cache-miss rows can contain stale non-finite values because their GEMMs do
# not write.  The fused kernel must mask before multiplication, not rely on
# IEEE 0*NaN.  Compare against a reference with those rows explicitly zeroed.
route_ids, weights, sorted_ids, c2, route_slot, out = make_case(
    18, 257, 4, True)
miss_rows = torch.rand(sorted_ids.numel(), device=device) > 0.35
clean = c2.clone()
clean[~miss_rows] = 0
c2[~miss_rows] = float("nan")
expected = reference(clean, sorted_ids, weights, None)
moe_w2_cubit._deterministic_unpermute(
    c2, sorted_ids, route_ids, weights, miss_rows, route_slot, out, E)
torch.cuda.synchronize()
assert bool(torch.isfinite(out).all())
torch.testing.assert_close(out, expected, rtol=1.6e-2, atol=1.6e-2)

# Capture after eager compilation, then mutate live inputs and replay.  This
# exercises the exact static-address path used by vLLM's FULL CUDA graphs.
route_ids, weights, sorted_ids, c2, route_slot, out = make_case(
    18, 4096, 4, True)
miss_rows = torch.ones(sorted_ids.numel(), device=device, dtype=torch.bool)
moe_w2_cubit._deterministic_unpermute(
    c2, sorted_ids, route_ids, weights, miss_rows, route_slot, out, E)
torch.cuda.synchronize()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    moe_w2_cubit._deterministic_unpermute(
        c2, sorted_ids, route_ids, weights, miss_rows, route_slot, out, E)
c2.mul_(0.5)
weights.mul_(0.75)
miss_rows[::11] = False
# Also invalidate one formerly valid route without clearing route_slot.  This
# models DSpark sentinel padding reusing the same captured workspace address.
route_ids.view(-1)[1] = E
weights.view(-1)[1] = 0
expected = reference(c2, sorted_ids, weights, miss_rows)
graph.replay()
torch.cuda.synchronize()
torch.testing.assert_close(out, expected, rtol=1.6e-2, atol=1.6e-2)
captured = out.clone()
for _ in range(25):
    graph.replay()
torch.cuda.synchronize()
assert torch.equal(captured, out), "CUDA graph replay is not bit-stable"

# Production-shape memory/latency gate.  Allocate the serving workspaces first,
# compile once, then measure only the fused epilogue.  The old implementation's
# analytic live intermediates are also reported without allocating them.
tokens, hidden, block = 1044, 4096, 16
route_ids, weights, sorted_ids, c2, route_slot, out = make_case(
    tokens, hidden, block, True)
miss_rows = torch.rand(sorted_ids.numel(), device=device) > 0.1
moe_w2_cubit._deterministic_unpermute(
    c2, sorted_ids, route_ids, weights, miss_rows, route_slot, out, E)
torch.cuda.synchronize()
baseline = torch.cuda.memory_allocated()
torch.cuda.reset_peak_memory_stats()
start = time.perf_counter()
for _ in range(20):
    moe_w2_cubit._deterministic_unpermute(
        c2, sorted_ids, route_ids, weights, miss_rows, route_slot, out, E)
torch.cuda.synchronize()
elapsed_ms = (time.perf_counter() - start) * 1000 / 20
peak_extra = torch.cuda.max_memory_allocated() - baseline
slots = sorted_ids.numel()
dump = tokens * TOP_K
old_bytes = (2 * slots * hidden * 4 + (dump + 1) * hidden * 4)
assert peak_extra < 16 * 2**20, peak_extra
assert old_bytes > 400 * 2**20, old_bytes

print(
    "MOE_W2_FUSED_UNPERMUTE PASS "
    f"max_abs={max_abs:.6g} mismatch={max_mismatch:.8%} "
    f"rel_l2={max_rel_l2:.3e} graph=stable "
    f"T={tokens} slots={slots} H={hidden} "
    f"peak_extra_mib={peak_extra / 2**20:.3f} "
    f"old_estimate_mib={old_bytes / 2**20:.3f} "
    f"latency_ms={elapsed_ms:.3f}"
)
