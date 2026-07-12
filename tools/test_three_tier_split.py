#!/usr/bin/env python3
"""GPU unit test: split-FP4 OVER THE BASE CACHE (residency coupling).

Three-tier layout with VLLM_MOE_W2_DELTA_SPLIT: host 2-bit base -> GPU
base pool ([codes13|sc13|codes2|sc2] slots) -> FP4 need-pool holding
2-bit REFINEMENT planes (no private scales). moe_w4s_mm reads base codes
+ scales from the BASE slot and the refinement from the FP4 slot, so a
pair is split-served only when the expert is resident in BOTH tables.

Checks:
 1. mixed dispatch vs reference: base-only experts at 2-bit dequant,
    both-resident experts at split-FP4 dequant (shared UE8M0 scales);
 2. the coupling transient: an FP4-mapped expert whose base slot is gone
    contributes exactly ZERO and bumps the miss counter (fetch+replay
    contract), never garbage;
 3. eviction coupling: base _take_slots_batch NEVER victimizes an expert
    mapped in the coupled FP4 tier (hard exclusion — these are the gate's
    quality-critical promotions and a base-evicted pair zeroes inside gate
    replays, which never refetch);
 4. under full-pool pressure the take serves only the uncoupled slots and
    leaves every coupled mapping intact and consistent (the base pool must
    be sized with the FP4 pool's coupled floor in mind).

Run (inside the vllm image): python3 test_three_tier_split.py
"""
import os
import sys

import torch

os.environ.setdefault("VLLM_MOE_W2", "1")
os.environ["VLLM_MOE_W2_DELTA_SPLIT"] = "1"

from vllm.model_executor.layers.quantization.utils import moe_w2_cubit  # noqa: E402
from vllm.model_executor.layers.quantization.utils import moe_w2_delta  # noqa: E402
from vllm.model_executor.layers.quantization.utils.moe_w2_planes import (  # noqa: E402
    mxfp4_to_codes, mxfp4_to_nibbles, nibbles_to_refinement,
    pack_fragment_major, pack_scales, split_fp4_dequant,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (  # noqa: E402
    per_token_group_quant_fp8,
)

assert moe_w2_cubit._ensure_ready(), "cubins not found"
assert moe_w2_delta.split_enabled()
dev = torch.device("cuda")
torch.manual_seed(11)

E = int(os.environ.get("E", "32"))
H = int(os.environ.get("H", "4096"))
I = int(os.environ.get("I", "2048"))
T = int(os.environ.get("T", "9"))
TOPK = 6
LEVELS = torch.tensor([-4.0, -1.0, 1.0, 4.0], device=dev)

w13_pack = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=dev)
s13 = torch.randint(118, 124, (E, 2 * I, H // 32), dtype=torch.uint8, device=dev)
w2_pack = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, device=dev)
s2 = torch.randint(118, 124, (E, H, I // 32), dtype=torch.uint8, device=dev)

planes13 = torch.stack([pack_fragment_major(mxfp4_to_codes(w13_pack[e])) for e in range(E)])
sc13p = torch.stack([pack_scales(s13[e]) for e in range(E)])
planes2 = torch.stack([pack_fragment_major(mxfp4_to_codes(w2_pack[e])) for e in range(E)])
sc2p = torch.stack([pack_scales(s2[e]) for e in range(E)])
c13len, s13len = planes13.shape[1], sc13p.shape[1]
c2len, s2len = planes2.shape[1], sc2p.shape[1]

# base-cache layer state, exactly _finish_layer's base branch
moe_w2_cubit._LAYERS[0] = dict(
    N13=2 * I, K13=H, N2=H, K2=I, E=E, base=True,
    off_s13=c13len, off_c2=c13len + s13len,
    off_s2=c13len + s13len + c2len,
    off4_s13=2 * c13len, off4_c2=2 * c13len + s13len,
    off4_s2=2 * c13len + s13len + 2 * c2len,
)

# BASE tier: small pool (forces eviction pressure in check 3)
moe_w2_delta._BASE_GB = 0.25         # base_enabled() -> True
btier = moe_w2_delta.DeltaTier(1, E, dev,
                               w13_bytes=c13len + s13len,
                               w2_bytes=c2len + s2len,
                               pool_gb=0.25, policy="lru", tag="base")
btier.miss_count = torch.zeros(1, dtype=torch.int32, device=dev)
moe_w2_delta._BASE_TIER = btier
btier.add_layer_host_planes(0, torch.cat((planes13, sc13p), dim=1),
                            torch.cat((planes2, sc2p), dim=1))

# FP4 need-pool in SPLIT mode: refinement planes, no scale sections
tier = moe_w2_delta.DeltaTier(1, E, dev,
                              w13_bytes=2 * I * H // 4,
                              w2_bytes=H * I // 4,
                              pool_gb=0.1, policy="freq", tag="fp4",
                              host_pinned=True)
moe_w2_delta._TIER = tier
btier._coupled_fp4 = tier            # the residency coupling under test
rf13 = torch.stack([pack_fragment_major(
    nibbles_to_refinement(mxfp4_to_nibbles(w13_pack[e]))) for e in range(E)])
rf2 = torch.stack([pack_fragment_major(
    nibbles_to_refinement(mxfp4_to_nibbles(w2_pack[e]))) for e in range(E)])
tier.add_layer_host_planes(0, rf13, rf2)

# make every expert BASE-resident, half of them FP4-resident
btier.ensure_resident(0, torch.arange(E, device=dev))
promoted = list(range(0, E, 2))
with tier._lock:
    for e in promoted:
        slot = tier._take_slots_batch(1)[0]
        tier._promote(0, e, slot)
torch.cuda.synchronize()
assert all(int(btier._mirror[0, e]) >= 0 for e in range(E))

x = (torch.randn(T, H, device=dev) * 0.3).to(torch.bfloat16)
topk_ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(T)]).to(torch.int32)
topk_w = torch.rand(T, TOPK, device=dev) * 0.5

E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6] * 2, device=dev)
E2M1[8:] *= -1


def dequant2(pack, sc):
    codes = mxfp4_to_codes(pack)
    return LEVELS[codes.long()] * torch.exp2(sc.float() - 127.0).repeat_interleave(32, -1)


def dequant_split(pack, sc):
    nib = mxfp4_to_nibbles(pack)
    return split_fp4_dequant(nib) * torch.exp2(sc.float() - 127.0).repeat_interleave(32, -1)


def reference(zero_experts=()):
    a8, as8 = per_token_group_quant_fp8(x, 128)
    a_deq = a8.float() * as8.repeat_interleave(128, 1)
    ref = torch.zeros(T, H, device=dev)
    for t in range(T):
        for j in range(TOPK):
            e = int(topk_ids[t, j])
            if e in zero_experts:
                continue
            dq = dequant_split if e in promoted else dequant2
            w13d = dq(w13_pack[e], s13[e])
            c13 = a_deq[t] @ w13d.T
            act = torch.nn.functional.silu(c13[:I]) * c13[I:]
            q2, qs2 = per_token_group_quant_fp8(act.to(torch.bfloat16).unsqueeze(0), 128)
            act_deq = q2.float() * qs2.repeat_interleave(128, 1)
            w2d = dq(w2_pack[e], s2[e])
            ref[t] += float(topk_w[t, j]) * (act_deq[0] @ w2d.T)
    return ref


# ---- 1. mixed dispatch --------------------------------------------------
got = moe_w2_cubit._moe_w2_forward(x, topk_w, topk_ids, 0)
ref = reference()
rel = (got.float() - ref).abs().max().item() / ref.abs().max().item()
cos = torch.nn.functional.cosine_similarity(
    got.float().flatten(), ref.flatten(), dim=0).item()
print(f"three-tier SPLIT mixed ({len(promoted)}/{E} FP4): max_rel={rel:.3e} "
      f"cos={cos:.6f}")
ok = rel < 0.06 and cos > 0.999
assert int(btier.miss_count.item()) == 0, "unexpected misses in check 1"

# ---- 2. coupling transient: FP4-mapped, base gone -> ZERO + miss --------
victim = promoted[1]
vb = int(btier._mirror[0, victim])
btier.slot_table[0, victim] = -1
btier._mirror[0, victim] = -1
got2 = moe_w2_cubit._moe_w2_forward(x, topk_w, topk_ids, 0)
miss = int(btier.miss_count.item())
ref2 = reference(zero_experts={victim})
rel2 = (got2.float() - ref2).abs().max().item() / ref2.abs().max().item()
routed = int((topk_ids == victim).sum())
print(f"transient (expert {victim} base-unmapped, routed {routed}x): "
      f"miss_count={miss} max_rel={rel2:.3e}")
ok = ok and miss > 0 and rel2 < 0.06
btier.slot_table[0, victim] = vb     # restore
btier._mirror[0, victim] = vb

# ---- 3. eviction coupling: FP4-mapped spared while others exist ---------
btier.step_begin()                   # new step: releases pin scopes
btier.seen.zero_()
btier._seen_host.zero_()
btier._tick += 10                    # everything cold
n_uncoupled = int((btier._mirror[0] >= 0).sum()) - len(promoted)
with btier._lock:
    taken = btier._take_slots_batch(n_uncoupled, emergency=True)
still = [e for e in promoted if int(btier._mirror[0, e]) >= 0]
print(f"eviction pressure (k={n_uncoupled} = all uncoupled): took "
      f"{len(taken)} slots; FP4-mapped intact {len(still)}/{len(promoted)}")
ok = ok and len(still) == len(promoted) and len(taken) == n_uncoupled

# ---- 4. full-pool pressure: coupled slots survive, tables consistent ----
# the check-3 slots are re-owned by this step's fetches (step-pinned,
# hard) — the only takeable victims left are still-mapped UNCOUPLED slots
uncoupled_left = int((btier._mirror[0] >= 0).sum()) - len(promoted)
k4 = len(promoted)
with btier._lock:
    for s in taken:
        btier._own(s, 0, int(btier._owner_ei[s]))
        btier._step_pins.add(s)
    taken2 = btier._take_slots_batch(k4, emergency=True)
still2 = [e for e in promoted if int(btier._mirror[0, e]) >= 0]
consistent = all(
    int(btier._mirror[0, e]) < 0 or int(btier.slot_table[0, e]) >= 0
    for e in range(E))
print(f"full-pool pressure: took {len(taken2)}/{k4} (uncoupled left: "
      f"{uncoupled_left}), FP4-mapped intact {len(still2)}/{len(promoted)}, "
      f"tables consistent: {consistent}")
ok = (ok and len(taken2) == uncoupled_left
      and len(still2) == len(promoted) and consistent)

print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
