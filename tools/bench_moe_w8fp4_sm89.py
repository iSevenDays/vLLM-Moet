#!/usr/bin/env python3
"""Bounded production-shape benchmark for Ada FP4-storage -> FP8 MMA."""

import argparse
import os

os.environ.setdefault("VLLM_MOE_W2_FP8_DELTA", "1")
os.environ.setdefault("VLLM_MOE_W2_FP8_DELTA_GB", "0.01")
os.environ.setdefault("VLLM_MOE_W2_FP8_STORE", "fp4")

import torch

from vllm.model_executor.layers.quantization.utils.moe_w2_sm89 import (
    _op_case_w8fp4,
    make_w8_launcher,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--m", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    args = parser.parse_args()

    assert torch.cuda.get_device_capability() == (8, 9)
    desc, buffers, reference = _op_case_w8fp4(args.k, args.n, args.m)
    launch = make_w8_launcher(args.k)
    for _ in range(args.warmup):
        launch(desc, args.n, 1)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.runs):
        launch(desc, args.n, 1)
    end.record()
    torch.cuda.synchronize()
    got = buffers[4].float().cpu()
    worst = (got - reference).abs().max().item() / reference.abs().max().item()
    assert torch.isfinite(got).all() and worst < 2.5e-2
    print(
        f"W8FP4_BENCH N={args.n} K={args.k} M={args.m} "
        f"ms={start.elapsed_time(end) / args.runs:.4f} "
        f"worst_rel={worst:.3e} runs={args.runs} warmup={args.warmup}"
    )


if __name__ == "__main__":
    main()
