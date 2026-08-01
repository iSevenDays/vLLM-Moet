#!/usr/bin/env python3
"""CPU-only quality audit of the 2-bit W2 base against one real MXFP4 expert.

This deliberately avoids the server, CUDA, generated kernels, and persistent
pack format.  It answers the first diagnostic question after a semantically
bad full-model run: how much error is introduced by the base representation
itself, before transport or kernel bugs are considered.
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)
CODE_LEVELS = torch.tensor([-4.0, -1.0, 1.0, 4.0], dtype=torch.float32)
NIBBLE_TO_CODE = torch.tensor(
    [2, 2, 2, 2, 2, 3, 3, 3, 1, 1, 1, 1, 1, 0, 0, 0],
    dtype=torch.uint8,
)
MAG_X2 = torch.tensor([0, 1, 2, 3, 4, 6, 8, 12], dtype=torch.int16)


def unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    packed = packed.view(torch.uint8)
    return torch.stack((packed & 0xF, packed >> 4), dim=-1).flatten(-2)


def original_dequant(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    nib = unpack_nibbles(packed)
    exponent = scale.view(torch.uint8).to(torch.float32) - 127.0
    return E2M1[nib.long()] * torch.exp2(exponent).repeat_interleave(32, -1)


def base_dequant(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Mirror mxfp4_refit_codes_scales plus reference dequantization."""
    nib = unpack_nibbles(packed)
    n, k = nib.shape
    scale_u8 = scale.view(torch.uint8)
    nib_blocks = nib.view(n, k // 32, 32)
    mag_x2 = MAG_X2[(nib_blocks & 7).long()]

    current_mag_x2 = torch.where(mag_x2 <= 4, 2, 8)
    current_error = ((current_mag_x2 - mag_x2) ** 2).sum(-1)
    lower_big = mag_x2 >= 3
    lower_mag_x2 = torch.where(lower_big, 4, 1)
    lower_error = ((lower_mag_x2 - mag_x2) ** 2).sum(-1)
    use_lower = (scale_u8 > 0) & (lower_error < current_error)

    current_codes = NIBBLE_TO_CODE[nib.long()].view(n, k // 32, 32)
    negative = nib_blocks >= 8
    lower_codes = torch.where(
        lower_big,
        torch.where(negative, 0, 3),
        torch.where(negative, 1, 2),
    ).to(torch.uint8)
    codes = torch.where(use_lower.unsqueeze(-1), lower_codes, current_codes)
    exponent = (scale_u8.to(torch.int16) - use_lower.to(torch.int16)).float()
    exponent -= 127.0
    return CODE_LEVELS[codes.long()].view(n, k) * torch.exp2(
        exponent).repeat_interleave(32, -1)


def metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    ref = reference.float().reshape(-1)
    got = candidate.float().reshape(-1)
    error = got - ref
    ref_rms = ref.square().mean().sqrt()
    return {
        "relative_rms": float(error.square().mean().sqrt() / ref_rms),
        "cosine": float(torch.nn.functional.cosine_similarity(ref, got, dim=0)),
        "max_abs_over_ref_rms": float(error.abs().max() / ref_rms),
        "zero_fraction": float((ref == 0).float().mean()),
    }


def load_tensor(model: Path, weight_map: dict, name: str) -> torch.Tensor:
    with safe_open(str(model / weight_map[name]), framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--vectors", type=int, default=4)
    args = parser.parse_args()

    index = json.loads((args.model / "model.safetensors.index.json").read_text())
    prefix = f"layers.{args.layer}.ffn.experts.{args.expert}"
    weights = {}
    for projection in ("w1", "w2", "w3"):
        packed = load_tensor(args.model, index["weight_map"],
                             f"{prefix}.{projection}.weight")
        scale = load_tensor(args.model, index["weight_map"],
                            f"{prefix}.{projection}.scale")
        original = original_dequant(packed, scale)
        base = base_dequant(packed, scale)
        weights[projection] = (original, base)
        print(json.dumps({
            "event": "projection_quality",
            "layer": args.layer,
            "expert": args.expert,
            "projection": projection,
            "shape": list(original.shape),
            **metrics(original, base),
        }, sort_keys=True))

    generator = torch.Generator().manual_seed(20260801)
    x = torch.randn(args.vectors, weights["w1"][0].shape[1],
                    generator=generator, dtype=torch.float32) * 0.3

    def expert_forward(which: int) -> torch.Tensor:
        w1, w2, w3 = (weights[name][which] for name in ("w1", "w2", "w3"))
        gate = x @ w1.T
        up = x @ w3.T
        hidden = torch.nn.functional.silu(gate) * up
        return hidden @ w2.T

    original_out = expert_forward(0)
    base_out = expert_forward(1)
    print(json.dumps({
        "event": "expert_forward_quality",
        "layer": args.layer,
        "expert": args.expert,
        "vectors": args.vectors,
        **metrics(original_out, base_out),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
