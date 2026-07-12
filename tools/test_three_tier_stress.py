#!/usr/bin/env python3
"""Stress test for the three-tier FP4 promote/dispatch path (gate idiom).

Reproduces the serving pattern of a gate storm (tau -> fire every step):
mark_seen -> decode forward -> force_promote -> manager passes -> eviction
pressure, and after EVERY iteration verifies the invariants that serving
quality rests on:

  I1. every FP4-mapped slot's POOL BYTES equal the host-store row
      (promotion/eviction must never leave a mapped slot half-written or
      rewritten under a live mapping);
  I2. slot_table == mirror, and owner tensors agree with the mirror;
  I3. the forward output matches a dequant reference built from the
      CURRENT mirror state (dispatch reads the tier the tables say).

Env: ITERS (default 40), SEED, E/H/I/T as usual.
Run (inside the vllm image): python3 test_three_tier_stress.py
"""
import os
import sys

import torch

os.environ.setdefault("VLLM_MOE_W2", "1")

from vllm.model_executor.layers.quantization.utils import moe_w2_cubit  # noqa: E402
from vllm.model_executor.layers.quantization.utils import moe_w2_delta  # noqa: E402
from vllm.model_executor.layers.quantization.utils.moe_w2_planes import (  # noqa: E402
    mxfp4_to_codes, mxfp4_to_nibbles, pack_fp4_fragment_major,
    pack_fragment_major, pack_scales,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (  # noqa: E402
    per_token_group_quant_fp8,
)

assert moe_w2_cubit._ensure_ready(), "cubins not found"
dev = torch.device("cuda")
torch.manual_seed(int(os.environ.get("SEED", "11")))

E = int(os.environ.get("E", "96"))
H = int(os.environ.get("H", "4096"))
I = int(os.environ.get("I", "2048"))
T = int(os.environ.get("T", "6"))
ITERS = int(os.environ.get("ITERS", "40"))
TOPK = int(os.environ.get("TOPK", "4"))
LEVELS = torch.tensor([-4.0, -1.0, 1.0, 4.0], device=dev)
E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6] * 2, device=dev)
E2M1[8:] *= -1

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

moe_w2_cubit._LAYERS[0] = dict(
    N13=2 * I, K13=H, N2=H, K2=I, E=E, base=True,
    off_s13=c13len, off_c2=c13len + s13len,
    off_s2=c13len + s13len + c2len,
    off4_s13=2 * c13len, off4_c2=2 * c13len + s13len,
    off4_s2=2 * c13len + s13len + 2 * c2len,
)

moe_w2_delta._BASE_GB = 0.25
# base pool ~56 slots << E=96: per-step working set (<=48) FITS, but
# consecutive steps churn the pool -- the live regime where the emergency
# eviction must always cover a replay's fetch batch
bslots = int(os.environ.get("BSLOTS", "72"))
moe_w2_delta._BASE_GB = bslots * (c13len + s13len + c2len + s2len) / 2**30
btier = moe_w2_delta.DeltaTier(1, E, dev,
                               w13_bytes=c13len + s13len,
                               w2_bytes=c2len + s2len,
                               pool_gb=moe_w2_delta._BASE_GB,
                               policy="freq", tag="base")
btier.miss_count = torch.zeros(1, dtype=torch.int32, device=dev)
moe_w2_delta._BASE_TIER = btier
base_host13 = torch.cat((planes13, sc13p), dim=1)
base_host2 = torch.cat((planes2, sc2p), dim=1)
btier.add_layer_host_planes(0, base_host13, base_host2)

# FP4 need-pool, NON-split ([fp4_13|sc13|fp4_2|sc2] slots), tight pool so
# gate promotions must evict (churn = the corruption surface)
fslots_budget = 10 * ((2 * I * H // 2 + s13len) + (H * I // 2 + s2len))
tier = moe_w2_delta.DeltaTier(1, E, dev,
                              w13_bytes=2 * I * H // 2 + s13len,
                              w2_bytes=H * I // 2 + s2len,
                              pool_gb=fslots_budget / 2**30,
                              policy="need", tag="fp4", host_pinned=True)
moe_w2_delta._TIER = tier
fp13 = torch.stack([pack_fp4_fragment_major(mxfp4_to_nibbles(w13_pack[e]))
                    for e in range(E)])
fp2 = torch.stack([pack_fp4_fragment_major(mxfp4_to_nibbles(w2_pack[e]))
                   for e in range(E)])
fp4_host13 = torch.cat((fp13, sc13p), dim=1)
fp4_host2 = torch.cat((fp2, sc2p), dim=1)
tier.add_layer_host_sections(0, (fp13, sc13p), (fp2, sc2p))

btier.ensure_resident(0, torch.arange(int(os.environ.get('PRELOAD', '32')), device=dev))
torch.cuda.synchronize()

fp4_row = torch.cat((fp4_host13, fp4_host2), dim=1)      # expected slot bytes
base_row = torch.cat((base_host13, base_host2), dim=1)


def dequant2(pack, sc):
    codes = mxfp4_to_codes(pack)
    return LEVELS[codes.long()] * torch.exp2(sc.float() - 127.0).repeat_interleave(32, -1)


def dequant4(pack, sc):
    nib = mxfp4_to_nibbles(pack)
    return E2M1[nib.long()] * torch.exp2(sc.float() - 127.0).repeat_interleave(32, -1)


def check_invariants(it, tier_, expected_row, label):
    mapped = [(int(li), int(ei), int(tier_._mirror[li, ei]))
              for li in range(1) for ei in range(E)
              if int(tier_._mirror[li, ei]) >= 0]
    for li, ei, s in mapped:
        # I2: tables agree
        st_ = int(tier_.slot_table[li, ei].item())
        assert st_ == s, (label, it, "slot_table!=mirror", li, ei, st_, s)
        oli = int(tier_._owner_li[s])
        oei = int(tier_._owner_ei[s])
        assert (oli, oei) == (li, ei), (
            label, it, "owner!=mirror", li, ei, "owner:", oli, oei, "slot", s)
        # I1: pool bytes == host row
        got = tier_.pool[s]
        exp = expected_row[ei].to(dev)
        if not torch.equal(got, exp):
            bad = (got != exp).nonzero().flatten()
            raise AssertionError(
                f"{label} iter {it}: POOL BYTES CORRUPT for (li={li},ei={ei})"
                f" slot {s}: {bad.numel()} bytes differ, first at "
                f"{int(bad[0])}")
    return mapped


x = (torch.randn(T, H, device=dev) * 0.3).to(torch.bfloat16)
a8, as8 = per_token_group_quant_fp8(x, 128)
a_deq = a8.float() * as8.repeat_interleave(128, 1)
w2cache = {e: dequant2(w13_pack[e], s13[e]) for e in range(E)}
w2cache2 = {e: dequant2(w2_pack[e], s2[e]) for e in range(E)}
w4cache = {e: dequant4(w13_pack[e], s13[e]) for e in range(E)}
w4cache2 = {e: dequant4(w2_pack[e], s2[e]) for e in range(E)}


def reference(topk_ids, topk_w, fp4_set):
    ref = torch.zeros(T, H, device=dev)
    for t in range(T):
        for j in range(TOPK):
            e = int(topk_ids[t, j])
            wa = w4cache[e] if e in fp4_set else w2cache[e]
            wb = w4cache2[e] if e in fp4_set else w2cache2[e]
            c13 = a_deq[t] @ wa.T
            act = torch.nn.functional.silu(c13[:I]) * c13[I:]
            q2, qs2 = per_token_group_quant_fp8(act.to(torch.bfloat16).unsqueeze(0), 128)
            act_deq = q2.float() * qs2.repeat_interleave(128, 1)
            ref[t] += float(topk_w[t, j]) * (act_deq[0] @ wb.T)
    return ref


if os.environ.get("PIN_TRACE"):
    _orig_fp = moe_w2_delta.DeltaTier.force_promote
    def _fp_traced(self, layers=None, max_promote=None):
        p0 = len(self._step_pins)
        n0 = self._n_promoted
        n = _orig_fp(self, layers=layers, max_promote=max_promote)
        print(f"[fp:{self._tag}] pins {p0}->{len(self._step_pins)} "
              f"promoted {self._n_promoted - n0} ret {n} "
              f"seen_snap={int((self._seen_host > 0).sum())}")
        return n
    moe_w2_delta.DeltaTier.force_promote = _fp_traced

worst = 0.0
for it in range(ITERS):
    topk_ids = torch.stack([torch.randperm(E, device=dev)[:TOPK]
                            for _ in range(T)]).to(torch.int32)
    topk_w = torch.rand(T, TOPK, device=dev) * 0.5
    btier.step_begin()
    tier.step_begin()

    if it == 0 and os.environ.get("PIN_TRACE"):
        print(f"[trace] after step_begin: pins={len(btier._step_pins)} "
              f"free={len(btier._free)}")
    # decode forward + the runner's miss loop: fetch + replay until
    # miss-free (the emergency eviction MUST cover the batch -- that is
    # the exact contract the live gate-storm regression violates)
    got = moe_w2_cubit._moe_w2_forward(x, topk_w, topk_ids, 0)
    torch.cuda.synchronize()
    if it == 0 and os.environ.get("PIN_TRACE"):
        print(f"[trace] after forward: pins={len(btier._step_pins)} "
              f"miss={int(btier.miss_count.item())}")
    for _pass in range(4):
        miss = int(btier.miss_count.item())
        if miss == 0:
            break
        btier.force_promote(max_promote=None)
        if it == 0 and os.environ.get("PIN_TRACE"):
            print(f"[trace] pass {_pass} after force_promote: "
                  f"pins={len(btier._step_pins)} free={len(btier._free)} "
                  f"unfixed={btier._kpi_unfixed}")
        btier.miss_count.zero_()
        got = moe_w2_cubit._moe_w2_forward(x, topk_w, topk_ids, 0)
        torch.cuda.synchronize()
    unres = btier._kpi_unfixed
    if unres:
        raise AssertionError(
            f"iter {it}: base emergency eviction STARVED -- {unres} "
            f"experts unrestored with pool {btier.n_slots} slots, "
            f"free {len(btier._free)}, pins {len(btier._step_pins)}, "
            f"seen {int((btier._seen_host > 0).sum())}")
    fp4_set = {e for e in range(E) if int(tier._mirror[0, e]) >= 0}
    base_set = {e for e in range(E) if int(btier._mirror[0, e]) >= 0}
    routed = set(topk_ids.flatten().tolist())
    not_served = routed - base_set - fp4_set
    if not_served:
        raise AssertionError(
            f"iter {it}: routed experts UNSERVED after replay loop: "
            f"{sorted(not_served)}")
    ref = reference(topk_ids, topk_w, fp4_set)
    rel = (got.float() - ref).abs().max().item() / ref.abs().max().item()
    worst = max(worst, rel)
    if rel > 0.06:
        raise AssertionError(
            f"iter {it}: OUTPUT diverged (rel {rel:.3e}) with fp4_set="
            f"{sorted(fp4_set)}")

    # gate idiom: this step was low-confidence -> force_promote the routed
    n = tier.force_promote(max_promote=8)
    # manager passes (deterministic, in-thread)
    btier._last_capture = 0.0
    tier._last_capture = 0.0
    btier._tick_once()
    tier._tick_once()
    torch.cuda.synchronize()

    check_invariants(it, tier, fp4_row, "fp4")
    check_invariants(it, btier, base_row, "base")

print(f"ITERS={ITERS} worst_rel={worst:.3e} "
      f"fp4_mapped={int((tier._mirror >= 0).sum())}/{E} "
      f"promoted_total={tier._n_promoted} evicted_total={tier._n_evicted}")
print("RESULT: PASS")
