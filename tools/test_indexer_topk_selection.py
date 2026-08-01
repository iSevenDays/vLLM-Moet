#!/usr/bin/env python3
"""Does the Lightning-Indexer top-k SELECT BY SCORE, or just take the first k?

Cheap sm89 probe for the long-context needle-retrieval regression. Motivation:
end-to-end needle results pass iff the needle's ratio-4 compressed entry index
is < index_topk (512) -- i.e. within the FIRST 512 entries -- and fail beyond,
independent of total context length. That is the signature of a top-k that has
degenerated to positional order.

DeepSeek's reference (checkpoint inference/model.py) is unambiguous:
    index_score = einsum(q, kv_cache[:, :end_pos // ratio]).relu_() * w -> sum
    topk_idxs   = index_score.topk(min(index_topk, end_pos // ratio))[1]
so selection MUST be by score. Only compress_ratio==4 layers get an Indexer;
ratio-128 layers use positional order by design (all causal entries).

The adversarial case is the one that matters: put every high score BEYOND
index k. A score-ranked kernel returns those positions; a positional kernel
returns 0..k-1 and scores 0 of them. Run under the serving image:

  docker run --rm --gpus '"device=0"' --entrypoint python3 \
    -v $PWD/tools/test_indexer_topk_selection.py:/opt/t.py:ro \
    vllm-moet-sm89:v0251 /opt/t.py
"""
import sys

import torch

WORKSPACE = 1024 * 1024
NEG = -1.0e4  # the indexer masks invalid columns with a large negative


def _expected(logits, seq_len, k):
    """Reference: torch.topk over the causal prefix, as a set of indices."""
    valid = min(seq_len, k)
    return set(logits[0, :seq_len].topk(valid).indices.tolist())


def _report(name, got, want, k, seq_len):
    got_valid = [i for i in got if i >= 0]
    hit = len(set(got_valid) & want)
    first_k = set(range(min(k, seq_len)))
    positional = len(set(got_valid) & first_k)
    verdict = "PASS" if set(got_valid) == want else "FAIL"
    print(f"  {name:<22} {verdict}  recall={hit}/{len(want)}  "
          f"returned={len(got_valid)}  overlap_with_first_{k}={positional}")
    if verdict == "FAIL":
        print(f"    want(sample)={sorted(want)[:8]}")
        print(f"    got (sample)={sorted(got_valid)[:8]}")
        if positional > hit:
            print("    ^ returns the FIRST k, not the top-k BY SCORE "
                  "-> positional degeneration")
    return verdict == "PASS"


def make_case(seq_len, k, hot_positions, device):
    """logits[0, p] high for p in hot_positions, low elsewhere."""
    logits = torch.full((1, seq_len), NEG, dtype=torch.float32, device=device)
    logits[0, :seq_len] = torch.linspace(-50.0, -40.0, seq_len, device=device)
    for rank, p in enumerate(hot_positions):
        logits[0, p] = 100.0 - rank * 0.01   # distinct, clearly highest
    return logits


def run_op(op_name, logits, seq_len, k, device):
    R, S = logits.shape
    idx = torch.full((R, k), -1, dtype=torch.int32, device=device)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
    if op_name == "persistent_topk":
        ws = torch.empty(WORKSPACE, dtype=torch.uint8, device=device)
        torch.ops._C.persistent_topk(logits, seq_lens, idx, ws, k, S)
    elif op_name == "cooperative_topk":
        ws = torch.empty(WORKSPACE, dtype=torch.uint8, device=device)
        torch.ops._C.cooperative_topk(logits, seq_lens, idx, ws, k, S)
    elif op_name == "top_k_per_row_decode":
        from vllm import _custom_ops as ops
        ops.top_k_per_row_decode(logits, 1, seq_lens, idx, R,
                                 logits.stride(0), logits.stride(1), k)
    else:
        raise ValueError(op_name)
    torch.cuda.synchronize()
    return idx[0].tolist()


def main():
    dev = "cuda"
    cap = torch.cuda.get_device_capability()
    print(f"device={torch.cuda.get_device_name()} capability=sm_{cap[0]}{cap[1]}")

    # CRITICAL: vLLM's C extension registers torch.ops._C lazily on import.
    # Probing torch.ops._C BEFORE this import reports every op as missing.
    try:
        from vllm import _custom_ops as ops
    except Exception as e:
        print(f"  (vllm._custom_ops unavailable: {e})")
        ops = None

    available = []
    for name in ("persistent_topk", "cooperative_topk"):
        if not hasattr(torch.ops._C, name):
            continue
        # cooperative_topk uses a thread-block cluster launch and has no
        # sm_89 kernel image; calling it raises AND poisons the CUDA context
        # for every later case. The serving path gates it behind
        # has_device_capability(90), so skipping here matches production.
        if name == "cooperative_topk" and cap < (9, 0):
            print(f"  (skipping {name}: SM90+ only, gated off in serving path)")
            continue
        available.append(name)
    if ops is not None and hasattr(ops, "top_k_per_row_decode"):
        available.append("top_k_per_row_decode")
    print(f"ops available: {available}\n")
    if not available:
        print("NO TOPK OPS FOUND — cannot test")
        return 2

    k = 512
    cases = [
        # (label, seq_len, hot_positions) -- the bug shows in ADVERSARIAL_*
        ("SANITY_first_k", 2500, list(range(0, k))),
        ("ADVERSARIAL_beyond_k", 2500, list(range(1000, 1000 + k))),
        ("ADVERSARIAL_tail", 2500, list(range(2500 - k, 2500))),
        ("ADVERSARIAL_scattered", 2500, None),
        ("SHORT_seq_lt_k", 400, list(range(0, 400))),
    ]

    g = torch.Generator().manual_seed(1234)
    all_ok = True
    for op in available:
        print(f"[{op}] k={k}")
        for label, seq_len, hot in cases:
            if hot is None:
                hot = torch.randperm(seq_len, generator=g)[:k].tolist()
            logits = make_case(seq_len, k, hot, dev)
            want = _expected(logits, seq_len, k)
            try:
                got = run_op(op, logits, seq_len, k, dev)
            except Exception as e:
                print(f"  {label:<22} ERROR {type(e).__name__}: {e}")
                all_ok = False
                continue
            all_ok &= _report(label, got, want, k, seq_len)
        print()

    print("RESULT:", "ALL PASS" if all_ok else "FAILURES PRESENT")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
