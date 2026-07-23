# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.models.deepseek_v4.nvidia.ops.triton_ada_fp8_bmm import (
    ada_fp8_grouped_mm,
)


def _is_sm89() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 9)


pytestmark = pytest.mark.skipif(not _is_sm89(), reason="requires an SM89 GPU")


def _e8m0_to_fp32(scale: torch.Tensor) -> torch.Tensor:
    return (scale.view(torch.uint8).to(torch.int32) << 23).view(torch.float32)


def _inputs(
    m: int,
    *,
    h: int = 2,
    n: int = 256,
    k: int = 256,
    e8m0_scales: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    device = "cuda"

    # Match fused_inv_rope_fp8_quant's group-major backing storage and
    # non-contiguous [M, H, K] view.
    a = (
        torch.randn((h, m, k), device=device)
        .clamp_(-2.0, 2.0)
        .to(torch.float8_e4m3fn)
        .transpose(0, 1)
    )
    a_scale = torch.empty((h, k // 128, m), device=device).uniform_(0.01, 0.05)
    a_scale = a_scale.permute(2, 0, 1)

    b = torch.randn((h * n, k), device=device).clamp_(-2.0, 2.0).to(torch.float8_e4m3fn)
    scale_shape = (h * n // 128, k // 128)
    if e8m0_scales:
        b_scale = torch.randint(120, 124, scale_shape, device=device, dtype=torch.uint8)
        b_scale = b_scale.view(torch.float8_e8m0fnu)
    else:
        b_scale = torch.empty(scale_shape, device=device).uniform_(0.01, 0.05)
    return a, a_scale, b, b_scale


def _reference(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
) -> torch.Tensor:
    m, h, k = a.shape
    n = b.shape[0] // h
    bv = b.view(h, n, k).float()
    bs = _e8m0_to_fp32(b_scale) if b_scale.dtype == torch.float8_e8m0fnu else b_scale
    bs = bs.view(h, n // 128, k // 128)
    out = torch.zeros((m, h, n), dtype=torch.float32, device=a.device)
    for kb in range(k // 128):
        av = a[:, :, kb * 128 : (kb + 1) * 128].float()
        av = av * a_scale[:, :, kb, None]
        bkv = bv[:, :, kb * 128 : (kb + 1) * 128]
        bkv = bkv * bs[:, :, kb].repeat_interleave(128, dim=1)[:, :, None]
        out += torch.einsum("mhk,hnk->mhn", av, bkv)
    return out.to(torch.bfloat16)


@pytest.mark.parametrize("m", [1, 2, 8, 17, 65])
@pytest.mark.parametrize("e8m0_scales", [False, True])
def test_matches_dequantized_reference(m: int, e8m0_scales: bool) -> None:
    args = _inputs(m, e8m0_scales=e8m0_scales)
    actual = ada_fp8_grouped_mm(*args)
    expected = _reference(*args)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_cuda_graph_replay() -> None:
    args = _inputs(2, h=4, e8m0_scales=True)
    static_out = ada_fp8_grouped_mm(*args)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = ada_fp8_grouped_mm(*args)
    graph.replay()

    expected = _reference(*args)
    torch.testing.assert_close(static_out, expected, rtol=2e-2, atol=2e-2)
