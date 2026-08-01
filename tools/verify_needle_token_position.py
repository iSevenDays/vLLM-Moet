#!/usr/bin/env python3
"""Where is the needle, in TOKENS? (The check that would have saved a whole trace.)

`needle_digits_probe.py` inserts the needle at a BYTE fraction of the filler but
reports its position as `int(prompt_tokens * depth)` -- a TOKEN fraction. Those
differ whenever the filler's halves tokenize at different densities, which is the
case here (markdown + Python is denser in the first half). A rank trace aimed at
the reported estimate therefore watches the wrong entry entirely.

This reconstructs the deterministic prompt byte-exact using the probe's OWN
builder, tokenizes it with the real tokenizer, and reports the measured token
position of the needle and of its digits, plus the compressed-entry columns those
map to. CPU only, seconds. Run it before aiming any position-addressed
instrument.

  docker run --rm --entrypoint python3 \
    -v $PWD:/w:ro -v /root/models/DeepSeek-V4-Flash-0731:/model:ro \
    vllm-moet-sm89:v0251 /w/tools/verify_needle_token_position.py
"""
import sys

sys.path.insert(0, "/w/tools")
sys.path.insert(0, "tools")

from transformers import AutoTokenizer  # noqa: E402

import needle_digits_probe as P  # noqa: E402

MODEL_DIR = "/model"
RATIOS = (4, 128)


def main():
    L, depth, variant, run = 8192, 0.5, "ask", 0
    bpt = 3.6

    blob = P.filler_blob(P.DEFAULT_FILLER, print)
    lines, mdepths, question, expected, label = P.build_spec(variant, L, run)
    depths = mdepths if mdepths is not None else [depth]
    ctx = P.build_context(max(1, int(L * bpt)), lines, depths, blob, question)
    code = P.needle_code(L, run)
    word, digits = code.split("-")
    needle_line = lines[0]

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    ids = tok(ctx, add_special_tokens=False)["input_ids"]
    n_tok = len(ids)
    n_byte = len(ctx.encode())

    # byte position of the needle line and of the digits within it
    b_needle = ctx.index(needle_line)
    b_digits = ctx.index(digits, b_needle)

    # token position = number of tokens in the prefix before that byte offset
    def tok_at(byte_off):
        return len(tok(ctx[:byte_off], add_special_tokens=False)["input_ids"])

    t_needle = tok_at(b_needle)
    t_digits = tok_at(b_digits)

    print(f"code={code}  context: {n_byte:,} bytes / {n_tok:,} tokens")
    print()
    print(f"{'item':<26} {'byte':>9} {'byte%':>7} {'token':>8} {'token%':>7}")
    print("-" * 62)
    for name, b, t in (("needle line start", b_needle, t_needle),
                       (f"digits '{digits}'", b_digits, t_digits)):
        print(f"{name:<26} {b:>9,} {b / n_byte:>6.1%} {t:>8,} {t / n_tok:>6.1%}")
    print()

    est = int(n_tok * depth)
    print(f"probe's reported estimate  int(pt x depth) = {est:,}")
    print(f"measured digits position                   = {t_digits:,}")
    print(f"ERROR                                      = "
          f"{est - t_digits:+,} tokens")
    print()
    print(f"{'ratio':>6} {'true col (digits)':>18} {'col from estimate':>18} "
          f"{'off by':>8}")
    for r in RATIOS:
        true_col, est_col = t_digits // r, est // r
        print(f"{r:>6} {true_col:>18} {est_col:>18} {est_col - true_col:>8}")
    print()
    print("Aim any position-addressed trace at the MEASURED value.")
    print(f"  VLLM_DSV4_INDEXER_TRACE_POS={t_digits}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
