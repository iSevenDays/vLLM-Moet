#!/usr/bin/env python3
"""Small GPU regression for the host/base-cache W2 forward path.

This deliberately leaves the optional delta tier disabled.  That is the
production host-residency configuration which previously reached the shared
GEMM epilogue with ``use_w8`` unbound during vLLM's memory-profile dummy run.

Run inside the sm89 image with the current overlay mounted over the installed
module; the exact command is recorded in PLAN.md.
"""
import os

import torch

os.environ.setdefault("VLLM_MOE_W2", "1")
os.environ.setdefault("VLLM_MOE_W2_DELTA_GB", "0")
os.environ.setdefault("VLLM_MOE_W2_FP8_DELTA", "0")

from vllm.model_executor.layers.quantization.utils import moe_w2_cubit
from vllm.model_executor.layers.quantization.utils import moe_w2_delta
from vllm.model_executor.layers.quantization.utils.moe_w2_planes import (
    mxfp4_to_codes,
    pack_fragment_major,
    pack_scales,
)

assert moe_w2_cubit._ensure_ready(), "sm89 kernels unavailable"
assert not moe_w2_delta.fp8_enabled(), "test requires FP8 delta disabled"

torch.manual_seed(17)
dev = torch.device("cuda")
E, H, I, T, TOPK = 4, 4096, 2048, 1, 2
levels = torch.tensor([-4.0, -1.0, 1.0, 4.0], device=dev)

w13_pack = torch.randint(
    0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=dev)
s13 = torch.randint(
    118, 124, (E, 2 * I, H // 32), dtype=torch.uint8, device=dev)
w2_pack = torch.randint(
    0, 256, (E, H, I // 2), dtype=torch.uint8, device=dev)
s2 = torch.randint(
    118, 124, (E, H, I // 32), dtype=torch.uint8, device=dev)

planes13 = torch.stack([
    pack_fragment_major(mxfp4_to_codes(w13_pack[e])) for e in range(E)
])
sc13 = torch.stack([pack_scales(s13[e]) for e in range(E)])
planes2 = torch.stack([
    pack_fragment_major(mxfp4_to_codes(w2_pack[e])) for e in range(E)
])
sc2 = torch.stack([pack_scales(s2[e]) for e in range(E)])

c13_len, s13_len = planes13.shape[1], sc13.shape[1]
c2_len, s2_len = planes2.shape[1], sc2.shape[1]
slot_bytes = c13_len + s13_len + c2_len + s2_len
moe_w2_cubit._LAYERS.clear()
moe_w2_cubit._LAYERS[0] = {
    "N13": 2 * I,
    "K13": H,
    "N2": H,
    "K2": I,
    "E": E,
    "base": True,
    "off_s13": c13_len,
    "off_c2": c13_len + s13_len,
    "off_s2": c13_len + s13_len + c2_len,
}

base = moe_w2_delta.DeltaTier(
    1,
    E,
    dev,
    w13_bytes=c13_len + s13_len,
    w2_bytes=c2_len + s2_len,
    pool_gb=(E + 1) * slot_bytes / 2**30,
    policy="lru",
    tag="base-smoke",
)
base.miss_count = torch.zeros(1, dtype=torch.int32, device=dev)
base.add_layer_host_planes(
    0, torch.cat((planes13, sc13), dim=1),
    torch.cat((planes2, sc2), dim=1))
base.ensure_resident(0, torch.arange(E, device=dev))
moe_w2_delta._BASE_TIER = base
moe_w2_delta._TIER = None

x = (torch.randn(T, H, device=dev) * 0.3).to(torch.bfloat16)
topk_ids = torch.tensor([[0, 3]], dtype=torch.int32, device=dev)
topk_weights = torch.tensor([[0.65, 0.35]], device=dev)
got = moe_w2_cubit._moe_w2_forward(x, topk_weights, topk_ids, 0)
torch.cuda.synchronize()


def dequant(pack: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    codes = mxfp4_to_codes(pack)
    return levels[codes.long()] * torch.exp2(
        scales.float() - 127.0).repeat_interleave(32, -1)


a_deq = moe_w2_cubit.a32_dequant_ref(x, gemm=1)
ref = torch.zeros(T, H, device=dev)
for j in range(TOPK):
    expert = int(topk_ids[0, j])
    c13 = a_deq[0] @ dequant(w13_pack[expert], s13[expert]).T
    act = torch.nn.functional.silu(c13[:I]) * c13[I:]
    act_deq = moe_w2_cubit.a32_dequant_ref(
        act.to(torch.bfloat16).unsqueeze(0), gemm=2)
    ref[0] += float(topk_weights[0, j]) * (
        act_deq[0] @ dequant(w2_pack[expert], s2[expert]).T)

relative = (got.float() - ref).abs().max().item() / ref.abs().max().item()
cosine = torch.nn.functional.cosine_similarity(
    got.float().flatten(), ref.flatten(), dim=0).item()
assert torch.isfinite(got).all(), "host/base output is non-finite"
assert int(base.miss_count.item()) == 0, "unexpected base-cache miss"
assert relative < 0.06 and cosine > 0.999, (relative, cosine)
print(
    f"HOST_BASE_SMOKE PASS relative={relative:.3e} cosine={cosine:.6f} "
    f"slots={base.n_slots} fp8_delta={moe_w2_delta.fp8_enabled()}")
