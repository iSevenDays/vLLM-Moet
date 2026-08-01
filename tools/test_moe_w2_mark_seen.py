#!/usr/bin/env python3
"""CUDA-graph regression for padded expert IDs in moe_w2 mark_seen."""

import torch

from vllm.model_executor.layers.quantization.utils import moe_w2_delta


assert torch.cuda.is_available(), "CUDA is required"
device = torch.device("cuda")
n_experts = 256


def expected(ids: torch.Tensor, *, count: bool) -> torch.Tensor:
    ids = ids.cpu()
    ids = ids[(ids >= 0) & (ids < n_experts)]
    out = torch.bincount(ids, minlength=n_experts).to(torch.int64)
    return out if count else out.clamp_max(1)


def check(mode_count: bool) -> None:
    moe_w2_delta._COUNT = mode_count
    dtype = torch.int32 if mode_count else torch.uint8

    # Eager boundary: negative, E, and large positive sentinels must not mark
    # either endpoint. Repeated valid IDs retain count/binary semantics.
    eager_ids = torch.tensor(
        [0, n_experts - 1, -1, n_experts, 42, -17, 999, 42],
        device=device,
        dtype=torch.int64,
    )
    eager_seen = torch.zeros(n_experts, device=device, dtype=dtype)
    moe_w2_delta.mark_seen(eager_seen, eager_ids)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        eager_seen.cpu().to(torch.int64),
        expected(eager_ids, count=mode_count),
        rtol=0,
        atol=0,
    )

    # Shape 18 under DSpark-5 produces 18 router rows x top-6 = 108 IDs.
    # Capture with stable storage, then mutate one complete top-6 row to the
    # two sentinel forms seen in scheduler/draft padding and replay the graph.
    graph_ids = torch.arange(108, device=device, dtype=torch.int64) % n_experts
    graph_seen = torch.zeros(n_experts, device=device, dtype=dtype)

    # Warm the PyTorch kernels before capture so compilation is not conflated
    # with graph correctness.
    moe_w2_delta.mark_seen(graph_seen, graph_ids)
    torch.cuda.synchronize()
    graph_seen.zero_()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        moe_w2_delta.mark_seen(graph_seen, graph_ids)

    for sentinel in (n_experts, -1, n_experts + 997):
        graph_seen.zero_()
        graph_ids.copy_(torch.arange(108, device=device) % n_experts)
        graph_ids[36:42] = sentinel
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            graph_seen.cpu().to(torch.int64),
            expected(graph_ids, count=mode_count),
            rtol=0,
            atol=0,
        )


check(True)
check(False)
print("MOE_W2_MARK_SEEN PASS shape=18 ids=108 modes=count,binary padding=ignored")
