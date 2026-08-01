#!/usr/bin/env python3
"""Reduced correctness gate for the mandatory exact FP4-storage cache.

Exercises real TP2 production shapes without loading a model: exact slot
geometry, pack-I/O skip selection, descriptor miss accounting, synchronous
fetch, pointer offsets, and the complete two-GEMM expert forward against a
dequantized checkpoint-FP4 reference.
"""
import os

import torch

os.environ.setdefault("VLLM_MOE_W2", "1")
os.environ.setdefault("VLLM_MOE_W2_EXACT_CACHE", "1")
os.environ.setdefault("VLLM_MOE_W2_FP8_DELTA", "1")
os.environ.setdefault("VLLM_MOE_W2_FP8_DELTA_GB", "0.05")
os.environ.setdefault("VLLM_MOE_W2_FP8_STORE", "fp4")
os.environ.setdefault("VLLM_MOE_W2_DELTA_GB", "0")
os.environ.setdefault("VLLM_MOE_W2_BASE_CACHE_GB", "0")
os.environ.setdefault("VLLM_MOE_W2_SCALE_REFIT", "0")
os.environ.setdefault("VLLM_MOE_W2_POOL_HEAT", "0")
os.environ.setdefault("VLLM_MOE_W2_DELTA_TICK_MS", "100000")

from vllm.model_executor.layers.quantization.utils import moe_w2_cubit
from vllm.model_executor.layers.quantization.utils import moe_w2_delta
from vllm.model_executor.layers.quantization.utils import moe_w2_store
from vllm.model_executor.layers.quantization.utils.moe_w2_planes import (
    _E2M1_VALS,
    mxfp4_to_nibbles,
)

assert moe_w2_cubit._ensure_ready(), "sm89 kernels unavailable"
assert moe_w2_delta.exact_cache_enabled()
assert moe_w2_delta.fp8_store_fp4()
assert moe_w2_delta.mandatory_tier() is None

torch.manual_seed(29)
dev = torch.device("cuda")
E, H, I, T, TOPK = 2, 4096, 1024, 1, 2  # exact TP2 checkpoint shapes
w13_pack = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8)
s13 = torch.randint(118, 124, (E, 2 * I, H // 32), dtype=torch.uint8)
w2_pack = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8)
s2 = torch.randint(118, 124, (E, H, I // 32), dtype=torch.uint8)


class ColdBuildLayer(torch.nn.Module):
    layer_name = "model.layers.1.ffn.experts"

    def __init__(self):
        super().__init__()
        for name, value in (
            ("w13_weight", w13_pack), ("w13_weight_scale", s13),
            ("w2_weight", w2_pack), ("w2_weight_scale", s2),
        ):
            self.register_parameter(
                name, torch.nn.Parameter(value, requires_grad=False))


# Exercise the production cold builder rather than manually staging its
# output.  This proves exact mode bypasses q2 construction and still emits a
# byte-correct slot: the descriptor/promotion/full-forward checks below read
# only the resulting tier.
cold_layer = ColdBuildLayer()
moe_w2_cubit._LAYERS.clear()
moe_w2_cubit.build_layer_planes(cold_layer, 0)
tier = moe_w2_delta.mandatory_tier()
assert tier is not None and tier._tag == "w8x"
assert moe_w2_cubit._LAYERS[0]["exact"]
for pname in ("w13_weight", "w13_weight_scale",
              "w2_weight", "w2_weight_scale"):
    assert getattr(cold_layer, pname).numel() == 0

f13_len, s13_len = 2 * I * H // 2, 2 * I * H // 32
f2_len, s2_len = H * I // 2, H * I // 32
slot_bytes = f13_len + s13_len + f2_len + s2_len
assert tier.slot_bytes == slot_bytes and tier.miss_count is not None
moe_w2_delta._BASE_TIER = None

# Descriptor-only miss -> fetch -> hit gate.  It catches counter/reset and
# section-pointer drift without involving the complete model forward.
eids = torch.tensor([1], dtype=torch.int32, device=dev)
npost = torch.tensor([4], dtype=torch.int32, device=dev)
desc = torch.zeros(4, 1, 6, dtype=torch.int64, device=dev)
a1 = torch.empty(4, H, dtype=torch.uint8, device=dev)
as1 = torch.empty(4, H // 32, dtype=torch.float32, device=dev)
c13 = torch.empty(4, 2 * I, dtype=torch.bfloat16, device=dev)
a2 = torch.empty(4, I, dtype=torch.uint8, device=dev)
as2 = torch.empty(4, I // 32, dtype=torch.float32, device=dev)
c2 = torch.empty(4, H, dtype=torch.bfloat16, device=dev)


def build_desc():
    moe_w2_cubit._desc_build_kernel_exactcache[(1,)](
        eids, npost, tier.slot_table[0], tier.miss_count, desc,
        a1.data_ptr(), as1.data_ptr(), c13.data_ptr(),
        a2.data_ptr(), as2.data_ptr(), c2.data_ptr(),
        tier.pool.data_ptr(), tier.slot_bytes,
        f13_len, f13_len + s13_len, f13_len + s13_len + f2_len,
        H, (H // 32) * 4, 4 * I, I, (I // 32) * 4, 2 * H,
        E, 1, 6, 4, BLOCK=256)
    torch.cuda.synchronize()


build_desc()
assert int(tier.miss_count.item()) == 1
assert int(desc[2, 0, 5].item()) == 0
tier.seen[0, 1] = 1
assert tier.force_promote(max_promote=None) == 1
tier.miss_count.zero_()
build_desc()
slot = int(tier.slot_table[0, 1].item())
base = tier.pool[slot].data_ptr()
assert int(tier.miss_count.item()) == 0
assert int(desc[2, 0, 2].item()) == base
assert int(desc[2, 0, 3].item()) == base + f13_len
assert int(desc[3, 0, 2].item()) == base + f13_len + s13_len
assert int(desc[3, 0, 3].item()) == base + f13_len + s13_len + f2_len
assert int(desc[2, 0, 5].item()) == 4

# Complete hit path versus original checkpoint FP4 (not versus a derivative
# q2 representation).  Eager forward ensures the second expert too.
x = (torch.randn(T, H, device=dev) * 0.3).to(torch.bfloat16)
topk_ids = torch.tensor([[0, 1]], dtype=torch.int32, device=dev)
topk_weights = torch.tensor([[0.55, 0.45]], device=dev)
got = moe_w2_cubit._moe_w2_forward(x, topk_weights, topk_ids, 0)
torch.cuda.synchronize()


def dequant(pack: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    nib = mxfp4_to_nibbles(pack)
    return _E2M1_VALS[nib.long()].float() * torch.exp2(
        scales.float() - 127.0).repeat_interleave(32, -1)


a_deq = moe_w2_cubit.a32_dequant_ref(x, gemm=1)
ref = torch.zeros(T, H, device=dev)
for j in range(TOPK):
    expert = int(topk_ids[0, j])
    c13_ref = a_deq[0] @ dequant(w13_pack[expert], s13[expert]).to(dev).T
    act = torch.nn.functional.silu(c13_ref[:I]) * c13_ref[I:]
    act_deq = moe_w2_cubit.a32_dequant_ref(
        act.to(torch.bfloat16).unsqueeze(0), gemm=2)
    ref[0] += float(topk_weights[0, j]) * (
        act_deq[0] @ dequant(w2_pack[expert], s2[expert]).to(dev).T)

relative = (got.float() - ref).abs().max().item() / ref.abs().max().item()
cosine = torch.nn.functional.cosine_similarity(
    got.float().flatten(), ref.flatten(), dim=0).item()
assert torch.isfinite(got).all()
assert int(tier.miss_count.item()) == 0
assert relative < 0.04 and cosine > 0.999, (relative, cosine)

# Warm-pack loader predicate must select the exact pack, not the resident
# planes cache or the obsolete 2-bit base pack.
class FakeParameter:
    def __init__(self, shape):
        self.data = torch.zeros(shape, dtype=torch.uint8)
        self.weight_loader = lambda *_args, **_kwargs: False

    @property
    def shape(self):
        return self.data.shape


class FakeLayer:
    layer_name = "model.layers.7.ffn.experts"
    w13_weight = FakeParameter((2, 4, 2))
    w13_weight_scale = FakeParameter((2, 4, 1))
    w2_weight = FakeParameter((2, 4, 2))
    w2_weight_scale = FakeParameter((2, 4, 1))


seen_tags = []
saved = moe_w2_store.pack_has_layer
try:
    moe_w2_store.pack_has_layer = lambda tag, *_a, **_k: seen_tags.append(tag) or True
    moe_w2_cubit._n_created = 0
    assert moe_w2_cubit.plan_pack_skip(FakeLayer())
finally:
    moe_w2_store.pack_has_layer = saved
assert seen_tags == ["w8x"], seen_tags

print(
    f"EXACT_CACHE_SMOKE PASS relative={relative:.3e} cosine={cosine:.6f} "
    f"slot={slot_bytes / 2**20:.3f}MiB miss_fetch_hit=1 pack={seen_tags[0]}")
