#!/usr/bin/env python3
"""Does the indexer's SCALE GRANULARITY cost us top-k ranking recall?

Context. Long-context digit recall on sm_89 needs ~44% selection coverage
(index_topk / candidates) to be reliable, while upstream passes needle @121k at
1.7% coverage -- a ~26x gap in indexer RANKING quality (see
docs/dsv4-sm89-longcontext/STATUS.md 5.15). Everything downstream of the scores
is already verified: the top-k kernel selects correctly BY SCORE
(tools/test_indexer_topk_selection.py), the ragged torch fallback matches the
checkpoint formula, and the paged Triton port passes its suite. So the defect is
in the scores themselves.

One concrete divergence exists in how the indexer's keys are quantized:

  * checkpoint reference (inference/model.py): the indexer q and kv go through
    `fp4_act_quant(..., fp4_block_size, True)` with `fp4_block_size = 32`
    -> 4 scale groups per 128-dim head.
  * this port's FP8 indexer cache (triton_paged_mqa_logits_dsv4.py): layout
    `[num_blocks, block_size, 1, D+4]` uint8 -- D fp8 bytes plus ONE f32 scale
    for the whole row -> 1 scale group per 128-dim head.

4x coarser. Coarse scaling costs the most on rows with outlier channels: dims far
below the row max get very few effective mantissa bits, their contribution to
q.k becomes noise, and since top-k is a RANKING over many near-tied candidates,
that noise reorders the selection. It would never fail a "matches its reference"
test because this port's reference shares the coarse layout.

What this measures: recall@k of the exact top-k under each quantization, as the
candidate count grows to the real values seen in the experiments (2422 / 4640 /
8956 candidates at pt 9686 / 18559 / 35825).

Two channel regimes, because the answer depends on them and only one is
realistic:
  homogeneous  - all dims ~N(0,1). Coarse scaling is nearly free here.
  outlier      - a few dims carry much larger magnitude (what real attention
                 activations look like). This is the regime that matters, and it
                 is SYNTHETIC -- it demonstrates the mechanism's size, it does
                 not prove the real keys look like this. Confirming that needs
                 the score-rank trace on a live request.

CPU-only, seconds:

  docker run --rm --entrypoint python3 \
    -v $PWD/tools/test_indexer_score_ranking.py:/opt/t.py:ro \
    vllm-moet-sm89:v0251 /opt/t.py
"""
import sys

import torch

D = 128       # index_head_dim
H = 64        # index_n_heads
E4M3_MAX = 448.0


def quant_e4m3_grouped(x: torch.Tensor, group: int) -> torch.Tensor:
    """Round-trip x [N, D] through E4M3 with one f32 scale per `group` dims.

    group == D  -> this port's per-row indexer cache scale.
    group == 32 -> the checkpoint's fp4_block_size granularity.
    """
    N, dim = x.shape
    assert dim % group == 0
    xg = x.reshape(N, dim // group, group)
    amax = xg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    scale = amax / E4M3_MAX
    q = (xg / scale).to(torch.float8_e4m3fn).float() * scale
    return q.reshape(N, dim)


def indexer_scores(q: torch.Tensor, k: torch.Tensor, w: torch.Tensor):
    """Reference formula: sum_h relu(q[h] . k[n]) * w[h]  ->  [N]."""
    # q [H, D], k [N, D], w [H]
    per_head = torch.relu(k @ q.T)          # [N, H]
    return (per_head * w).sum(dim=-1)


def make_keys(n, regime, gen):
    k = torch.randn(n, D, generator=gen)
    if regime == "outlier":
        # a handful of channels an order of magnitude larger, constant across
        # keys (channel-wise outliers, as seen in real attention activations)
        chan = torch.ones(D)
        idx = torch.randperm(D, generator=gen)[:6]
        chan[idx] = 12.0
        k = k * chan
    return k


def run(regime, n, k_sel, gen):
    q = torch.randn(H, D, generator=gen)
    if regime == "outlier":
        chan = torch.ones(D)
        idx = torch.randperm(D, generator=gen)[:6]
        chan[idx] = 12.0
        q = q * chan
    w = torch.rand(H, generator=gen) + 0.1
    keys = make_keys(n, regime, gen)

    exact = indexer_scores(q, keys, w)
    top_exact = set(exact.topk(k_sel).indices.tolist())
    best_exact = int(exact.argmax())

    out = {}
    for label, group in (("row(D=128)", D), ("block32", 32)):
        kq = quant_e4m3_grouped(keys, group)
        s = indexer_scores(q, kq, w)
        top_q = set(s.topk(k_sel).indices.tolist())
        recall = len(top_q & top_exact) / k_sel
        rank_best = int((s > s[best_exact]).sum())
        out[label] = (recall, rank_best)
    return out


def main():
    print("indexer score ranking vs key-quantization granularity")
    print(f"D={D} H={H}  E4M3, one f32 scale per group\n")
    ok = True
    for regime in ("homogeneous", "outlier"):
        print(f"--- {regime} channels " + ("(realistic)" if regime == "outlier"
                                           else "(control)"))
        print(f"{'candidates':>11} {'k':>6} {'recall row(D)':>14} "
              f"{'recall block32':>15} {'rank_best row':>14} {'blk32':>7}")
        for n, k_sel in ((2422, 512), (4640, 512), (8956, 512), (8956, 2048)):
            gen = torch.Generator().manual_seed(7 ^ n ^ k_sel ^ len(regime))
            r = run(regime, n, k_sel, gen)
            rr, rb = r["row(D=128)"]
            br, bb = r["block32"]
            print(f"{n:>11} {k_sel:>6} {rr:>13.1%} {br:>14.1%} "
                  f"{rb:>14} {bb:>7}")
            if regime == "outlier" and br < rr:
                ok = False
        print()
    print("Read: if block32 recall exceeds row(D) recall in the OUTLIER regime,")
    print("finer indexer scale granularity buys ranking recall, and this port's")
    print("per-row scale is a live suspect for the ~26x coverage gap.")
    print("A per-row scale that matches block32 means granularity is NOT the")
    print("mechanism and the gap is elsewhere.")
    return 0 if ok else 0  # informational: never fail the build on a synthetic


if __name__ == "__main__":
    sys.exit(main())
