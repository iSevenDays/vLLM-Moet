# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib

import pytest
import torch

dsv4_topk = importlib.import_module(
    "vllm.model_executor.layers.fused_moe.router.dsv4_topk"
)
fused_topk_bias_router = importlib.import_module(
    "vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router"
)


@pytest.fixture
def cuda_platform(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(dsv4_topk.current_platform, "is_cuda", lambda: True)


def test_can_use_dsv4_topk_accepts_flash_configuration(cuda_platform):
    gating_output = torch.zeros((2, 256), dtype=torch.float32)
    correction_bias = torch.zeros(256, dtype=torch.float32)

    assert dsv4_topk.can_use_dsv4_topk(
        gating_output,
        correction_bias,
        topk=6,
        renormalize=True,
        indices_dtype=torch.int32,
    )


@pytest.mark.parametrize(
    ("gating_output", "correction_bias", "topk", "renormalize", "indices_dtype"),
    [
        (
            torch.zeros((2, 128), dtype=torch.float32),
            torch.zeros(128),
            6,
            True,
            torch.int32,
        ),
        (
            torch.zeros((2, 256), dtype=torch.bfloat16),
            torch.zeros(256),
            6,
            True,
            torch.int32,
        ),
        (torch.zeros((2, 256)), None, 6, True, torch.int32),
        (torch.zeros((2, 256)), torch.zeros(256), 5, True, torch.int32),
        (torch.zeros((2, 256)), torch.zeros(256), 6, False, torch.int32),
        (torch.zeros((2, 256)), torch.zeros(256), 6, True, torch.uint8),
    ],
)
def test_can_use_dsv4_topk_rejects_unsupported_inputs(
    cuda_platform,
    gating_output,
    correction_bias,
    topk,
    renormalize,
    indices_dtype,
):
    assert not dsv4_topk.can_use_dsv4_topk(
        gating_output,
        correction_bias,
        topk,
        renormalize,
        indices_dtype,
    )


def test_can_use_dsv4_topk_requires_cuda(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(dsv4_topk.current_platform, "is_cuda", lambda: False)

    assert not dsv4_topk.can_use_dsv4_topk(
        torch.zeros((2, 256)),
        torch.zeros(256),
        topk=6,
        renormalize=True,
        indices_dtype=torch.int32,
    )


def test_fused_topk_bias_dispatches_to_dsv4_fast_path(
    monkeypatch: pytest.MonkeyPatch,
):
    expected_weights = torch.ones((2, 6), dtype=torch.float32)
    expected_ids = torch.zeros((2, 6), dtype=torch.int64)
    observed = {}

    monkeypatch.setattr(
        fused_topk_bias_router, "can_use_dsv4_topk", lambda *args: True
    )

    def fake_dsv4_topk(gating_output, correction_bias, indices_dtype, scale):
        observed["args"] = (
            gating_output,
            correction_bias,
            indices_dtype,
            scale,
        )
        return expected_weights, expected_ids

    monkeypatch.setattr(fused_topk_bias_router, "dsv4_topk", fake_dsv4_topk)

    hidden_states = torch.zeros((2, 16), dtype=torch.float32)
    gating_output = torch.zeros((2, 256), dtype=torch.float32)
    correction_bias = torch.zeros(256, dtype=torch.float32)
    weights, ids = fused_topk_bias_router.fused_topk_bias(
        hidden_states=hidden_states,
        gating_output=gating_output,
        scoring_func="sqrtsoftplus",
        e_score_correction_bias=correction_bias,
        topk=6,
        renormalize=True,
        indices_type=torch.int64,
        routed_scaling_factor=1.5,
    )

    assert weights is expected_weights
    assert ids is expected_ids
    assert observed["args"] == (
        gating_output,
        correction_bias,
        torch.int64,
        1.5,
    )
