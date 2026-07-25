#!/usr/bin/env python3
"""CPU equivalence golden: batch[i] == single(i) for the three batched
pack/derivation twins introduced for the fast FP8/FP4 delta plane build.

The mxfp4 chunk loop in build_layer_planes was restructured to derive one
chunk's FP8 planes in two batched torch ops (mxfp4_fp8_unit_bytes_batch +
pack_fp8_fragment_major_batch) instead of ~E per-expert iterations, each
with its own D2H sync. This script proves the batched path reproduces the
SINGLE-expert golden bytes EXACTLY, for every expert of a few random
[E, N, K] shapes:

  pack_fp4_fragment_major_batch(nib)[i]            == pack_fp4_fragment_major(nib[i])
  pack_fp8_fragment_major_batch(w)[i]              == pack_fp8_fragment_major(w[i])
  mxfp4_fp8_unit_bytes_batch(w, ck, sv)[i]         == _mxfp4_fp8_unit_bytes(
                                                        mxfp4_to_nibbles(w[i]),
                                                        ck[i], sv[i])

The single-expert golden lives in moe_w2_cubit (_mxfp4_fp8_unit_bytes); the
packers and the batch twins live in moe_w2_planes. f32/int16 math is exact
for the e2m1 grid and UE8M0 byte deltas (integers in [-255,255] are exactly
representable in f32), so equality is byte-for-byte, not approximate.

Run: python3 kernels/gen/moe_w8_sm89_batch_check.py
"""
import os
import sys

import torch

# Do NOT write __pycache__ next to the overlay sources we import (keeps
# overlay/ clean -- gen_patches.py rejects stray .pyc under overlay/).
sys.dont_write_bytecode = True

# Import the overlay packers directly (pure-torch, no vLLM runtime needed).
_PLANE = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "overlay", "vllm", "vllm", "model_executor", "layers", "quantization",
    "utils")
sys.path.insert(0, _PLANE)

from moe_w2_planes import (  # noqa: E402
    mxfp4_to_nibbles,
    pack_fp4_fragment_major,
    pack_fp8_fragment_major,
    pack_fp4_fragment_major_batch,
    pack_fp8_fragment_major_batch,
    mxfp4_fp8_unit_bytes_batch,
)
# The single-expert FP8 derivation golden lives in the cubit module. It only
# needs torch + moe_w2_planes at call time, but importing the module pulls in
# vLLM logger -- stub it before import.
os.environ.setdefault("VLLM_USE_V1", "1")


def _mxfp4_fp8_unit_bytes(nib, ck_scale_bytes, serving_scale_bytes):
    """Verbatim copy of moe_w2_cubit._mxfp4_fp8_unit_bytes (the single-expert
    golden) -- replicated here so the check is hermetic (no vLLM import)."""
    from moe_w2_planes import _E2M1_VALS
    N, K = nib.shape
    w = _E2M1_VALS.to(nib.device)[nib.long()].view(N, K // 32, 32)
    ck = ck_scale_bytes.to(torch.float32).view(N, K // 32, 1)
    sv = serving_scale_bytes.to(torch.float32).view(N, K // 32, 1)
    u = w.to(torch.float32) * torch.exp2(ck - sv)
    return (u.reshape(N, K).clamp(-448.0, 448.0)
            .to(torch.float8_e4m3fn).view(torch.uint8))


torch.manual_seed(int(os.environ.get("SEED", "7")))
# N divisible by 16, K divisible by 64 (the packer asserts). Mix small and
# DS4-Flash-shaped (N13=2I, K13=H) geometries.
SHAPES = [
    (4, 16, 64),      # minimal
    (3, 32, 128),
    (8, 64, 512),     # DS4 w2-like
    (5, 128, 1024),   # DS4 w13-like
    (2, 48, 192),     # N=48: N%16==0, exercises a non-power-of-2 block count
]


def check(name, batch_fn, single_fn, build_args):
    """build_args(E, N, K) -> (batch_input_tuple, per_expert_input_callable)."""
    failures = 0
    for E, N, K in SHAPES:
        batch_args, single = build_args(E, N, K)
        batch_out = batch_fn(*batch_args)
        for i in range(E):
            single_out = single(i)
            bi = batch_out[i]
            if not torch.equal(bi, single_out):
                failures += 1
                print(f"[{name}] MISMATCH shape=({E},{N},{K}) expert={i}: "
                      f"batch {bi.shape} vs single {single_out.shape}; "
                      f"first diff at "
                      f"{(bi != single_out).nonzero()[0].tolist() if (bi != single_out).any() else 'none'}")  # noqa: E501
        print(f"[{name}] shape=({E},{N},{K}): {E}/{E} experts byte-identical "
              f"({batch_out.numel()} bytes/expert)")
    return failures


def main() -> int:
    total = 0

    # --- pack_fp4_fragment_major_batch : [E,N,K] nibbles -> [E, N*K/2] ----
    def args_fp4(E, N, K):
        nib = torch.randint(0, 16, (E, N, K), dtype=torch.uint8)
        return (nib,), lambda i: pack_fp4_fragment_major(nib[i])
    total += check("pack_fp4_fragment_major_batch", pack_fp4_fragment_major_batch,
                   None, args_fp4)

    # --- pack_fp8_fragment_major_batch : [E,N,K] e4m3 -> [E, N*K] ---------
    def args_fp8(E, N, K):
        w = torch.randint(0, 256, (E, N, K), dtype=torch.uint8)
        return (w,), lambda i: pack_fp8_fragment_major(w[i])
    total += check("pack_fp8_fragment_major_batch", pack_fp8_fragment_major_batch,
                   None, args_fp8)

    # --- mxfp4_fp8_unit_bytes_batch : [E,N,K/2]+ck+sv -> [E,N,K] ----------
    # w_packed [E,N,K/2], ck/sv [E,N,K/32] u8. ck >= sv in most blocks so the
    # exp2 delta is a positive power of 2 (typical refit decrements sv by 1),
    # but also seed some sv > ck to exercise negative deltas.
    def args_unit(E, N, K):
        w_packed = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8)
        ck = torch.randint(120, 136, (E, N, K // 32), dtype=torch.uint8)
        # sv in [ck-2, ck+1]: covers the refit -1/-2 paths AND sv>ck.
        sv = (ck.to(torch.int16) + torch.randint(-2, 2, ck.shape)).clamp(
            0, 255).to(torch.uint8)
        return (w_packed, ck, sv), lambda i: _mxfp4_fp8_unit_bytes(
            mxfp4_to_nibbles(w_packed[i]), ck[i], sv[i])
    total += check("mxfp4_fp8_unit_bytes_batch", mxfp4_fp8_unit_bytes_batch,
                   None, args_unit)

    if total:
        print(f"\nFAILED: {total} expert mismatches.")
        return 1
    print(f"\nOK: all batched twins byte-identical to the single-expert golden "
          f"across {len(SHAPES)} shapes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
