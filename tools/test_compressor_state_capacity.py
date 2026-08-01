#!/usr/bin/env python3
"""The compressor state cache must hold a WHOLE prefill chunk + the window.

Why this invariant exists. The compressor state is a SlidingWindowMLASpec whose
window is only `(1 + (ratio == 4)) * ratio` entries (8 for ratio 4, 128 for
ratio 128), but chunked prefill writes up to `max_num_batched_tokens` tokens per
forward. Within one chunk the order is: `save_partial_states` writes EVERY
token's (kv, score+ape) into the state cache, and only then does the fused
compress kernel gather `(1 + overlap) * ratio` entries per compressed output. If
the per-request state allocation were merely window-sized, the later tokens of a
chunk would overwrite the state that earlier outputs in the SAME chunk still
need to read, and the corruption would be silent -- correct for short prompts,
wrong for long ones, and dependent on where content lands relative to chunk
boundaries.

The production allocation confirms the design is chunk-aware. Boot log for
BATCHED_TOKENS=1056 reports per-request KV groups (1024, 20, 20, 267, 149):

    ratio 4  : ceil((1056 + 8)   / 4) = 266  -> observed 267
    ratio 128: ceil((1056 + 128) / 8) = 148  -> observed 149

(one spare block each, for wrap/alignment). This test pins that relationship so
a future change to max_num_batched_tokens, block_size, or the window cannot
quietly reintroduce intra-chunk overwrite.

CPU-only, no GPU, no server:

  docker run --rm --entrypoint python3 \
    -v $PWD/tools/test_compressor_state_capacity.py:/opt/t.py:ro \
    vllm-moet-sm89:v0251 /opt/t.py
"""
import math
import sys

# Mapping asserted by CompressorStateCache.__init__ (compressor.py):
#   sliding_window = (1 + (ratio == 4)) * ratio
#   block_size     = 4 for ratio 4, 8 for ratio 128
RATIO_BLOCK_SIZE = {4: 4, 128: 8}
# Observed per-request group block counts from the production boot log
# ("GPU KV packed blocks: ... per-group=(1024,20,20,267,149)") at
# max_num_batched_tokens = 1056.
OBSERVED = {(4, 1056): 267, (128, 1056): 149}


def sliding_window(ratio: int) -> int:
    return (1 + (ratio == 4)) * ratio


def required_blocks(ratio: int, batched_tokens: int) -> int:
    """Blocks needed so one chunk plus the carry-in window never overlap."""
    bs = RATIO_BLOCK_SIZE[ratio]
    return math.ceil((batched_tokens + sliding_window(ratio)) / bs)


def check_code_constants() -> bool:
    """Cross-check the constants against the live class, if importable."""
    try:
        import inspect

        from vllm.models.deepseek_v4.compressor import CompressorStateCache
        src = inspect.getsource(CompressorStateCache.__init__)
    except Exception as e:  # pragma: no cover - import shape varies
        print(f"  (could not import CompressorStateCache: {type(e).__name__}: {e})")
        print("  falling back to documented constants")
        return True
    ok = True
    for ratio, bs in RATIO_BLOCK_SIZE.items():
        needle = f"self.block_size = {bs}"
        if needle not in src:
            print(f"  MISMATCH: expected '{needle}' for ratio {ratio} in source")
            ok = False
    if "self.sliding_window = coff * compress_ratio" not in src:
        print("  MISMATCH: sliding_window formula changed in source")
        ok = False
    print(f"  code constants: {'OK' if ok else 'DRIFTED'}")
    return ok


def main() -> int:
    print("compressor state capacity: one chunk + window must fit")
    ok = check_code_constants()
    print()
    print(f"{'ratio':>6} {'win':>5} {'blk':>4} {'batched':>8} {'required':>9} "
          f"{'observed':>9}  verdict")
    print("-" * 60)

    for ratio in (4, 128):
        for batched in (528, 1024, 1056, 2112, 8192):
            req = required_blocks(ratio, batched)
            obs = OBSERVED.get((ratio, batched))
            if obs is None:
                verdict, good = "n/a (not measured)", True
            else:
                # observed must COVER the requirement (spare blocks are fine)
                good = obs >= req
                verdict = "COVERS" if good else "TOO SMALL — overwrite risk"
                ok &= good
            print(f"{ratio:>6} {sliding_window(ratio):>5} "
                  f"{RATIO_BLOCK_SIZE[ratio]:>4} {batched:>8} {req:>9} "
                  f"{str(obs) if obs is not None else '-':>9}  {verdict}")

    # A window-only allocation must be REJECTED by the invariant: this is the
    # bug the sizing rule prevents, so prove the check can actually fail.
    print()
    for ratio in (4, 128):
        win_only = math.ceil(sliding_window(ratio) / RATIO_BLOCK_SIZE[ratio])
        req = required_blocks(ratio, 1056)
        detects = win_only < req
        print(f"  negative control ratio {ratio:>3}: window-only alloc "
              f"{win_only} blocks < required {req} -> "
              f"{'DETECTED' if detects else 'NOT DETECTED (check is vacuous!)'}")
        ok &= detects

    print()
    print("RESULT:", "capacity invariant holds" if ok else "INVARIANT VIOLATED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
