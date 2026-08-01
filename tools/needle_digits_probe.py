#!/usr/bin/env python3
"""Needle probe that reports WHAT was lost, not just pass/fail.

Motivation — the old metric caused a multi-hour misdiagnosis. `bench/runner/
probes.py --probe needle_sweep` reports a bare correct/incorrect plus a mean
answer-token confidence. Under that metric this model looks like it "loses
long-context retrieval from ~8K". It does not. Decoding the answers against the
deterministic needle codes shows the needle's WORD is recovered at every length
tested (2K-48K) and only the numeric DIGITS are lost:

    true PELICAN-1605  ->  "PELICAN PELICAN PELICAN..."
    true LANTERN-2037  ->  "LANTERN-9-9-9-9-9..."
    true CYPRESS-0527  ->  "CYPHER"

"Retrieval is broken" and "the attended value's fine detail is unreadable" lead
to completely different investigations. So this probe always splits the verdict
into word_present / digits_present, and captures the top-N logprobs at every
answer position, because the shape of the digit distribution is the diagnostic:

    4K PASS: digit token emitted at logprob -0.000 (p~1.000), runner-up -14.4
             => the model is READING the digits
    8K FAIL: emitted at -0.765 (p~0.47) over a diffuse spread of unrelated
             3-digit priors, true token ABSENT from the top-20
             => at THIS answer position the value is unreadable. NOTE: it is
             NOT globally unavailable -- see `digitsonly` below, which recovers
             the same digits exactly from the same prompt. The loss depends on
             the digits' POSITION IN THE ANSWER (STATUS 5.17).

Question variants exist to separate retrieval from readback:

  ask      the baseline task
  yesno    "Is the code <TRUE>?" -- a 1-token answer, so a correct "yes" shows
           the information is present even when it cannot be generated. ALWAYS
           pair with yesno_neg; "yes" alone is meaningless (yes-bias).
  yesno_neg  same with a deliberately WRONG code; a truthful model says "no".
  spell    "spell it one character at a time" -- isolates multi-token decode.
  words    code is three salient WORDS, no digits -- if this passes, what is
           lost is specifically low-salience content.
  repeat   the code appears 3x at spread-out depths -- if redundancy rescues it,
           the failure is SNR-limited, not structural.
  verbose  digits also written out ("one six zero five").
  digitsonly / firstword  ask for one component only, so it lands at answer
           position ~0. digitsonly returns 1605 EXACTLY on the same 8K prompt
           where `ask` returns 1234 -- proof the information is present and the
           failure is decode-side (STATUS 5.17).
  digitspad  the same digits-only task with a forced multi-token prefix that
           pushes them to answer position ~5+. Isolates ANSWER POSITION as the
           only variable.

Findings this probe produced (see PLAN.md 5.10): errors become NEAR-MISSES once
redundancy or salience is added (1606 for 1605, 7542 for 7544, ZEPHYR-ORYX-CORYX
for ...-CYPRESS) while the plain task falls back to the generic prior 1234;
results are deterministic per input (bit-identical on repeat) and NON-monotonic
in length (pt 8316 passes while 5100 and 6650 fail).

Filler is the same realistic repo text needle_sweep uses. CAUTION: the repo
elsewhere claims random-WORD filler "does not reproduce the failure" -- that is
WRONG at >=10K tokens, where tools/needle_probe.py reproduces it plainly and
more severely at depth 0.1 than 0.5 (STATUS 5.16). `--abs-pos` places the needle at an absolute
token position instead of a depth fraction, which is how the "fixed position"
and "fixed length" hypotheses were both refuted.

Usage:
  python3 tools/needle_digits_probe.py --endpoint http://127.0.0.1:8011 \
      --lengths 3072,5120,8192 --variant ask --out /tmp/needle.json
  python3 tools/needle_digits_probe.py --lengths 8192 \
      --variant ask --abs-pos 194,1695,4843,9492
"""
import argparse
import json
import os
import random
import re
import sys
import time
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_FILLER = [
    "CONCEPTS.md",
    "docs/ada-sm89-port.md",
    "kernels/MANIFEST.md",
    "AGENTS.md",
    "overlay/vllm/vllm/model_executor/layers/quantization/utils/moe_w2_cubit.py",
    "/root/autostart/CLAUDE.md",
]
# Must match bench/runner/probes.py so codes/prompts are comparable run-to-run.
CODE_WORDS = [
    "ZEPHYR", "ORYX", "GLACIER", "NIMBUS", "COBALT", "FALCON", "LANTERN",
    "CYPRESS", "OBSIDIAN", "PELICAN", "GRANITE", "MARLIN", "PUMICE", "SAFFRON",
    "TUNDRA", "VECTOR", "WILLOW", "EMBER",
]
QUESTION = "What is the project access code? Reply with only the code."
DIGIT_WORDS = "zero one two three four five six seven eight nine".split()
VARIANTS = ("ask", "yesno", "yesno_neg", "spell", "words", "repeat", "verbose",
            "digitsonly", "firstword", "digitspad")


def needle_code(L, run=0):
    """Deterministic code for (length, run) — identical to needle_sweep's."""
    rng = random.Random(0x4E4C ^ L ^ (run * 0x10001))
    return f"{rng.choice(CODE_WORDS)}-{rng.randint(0, 9999):04d}"


def filler_blob(files, log):
    parts = []
    for f in files:
        p = f if os.path.isabs(f) else os.path.join(REPO, f)
        try:
            parts.append(open(p, encoding="utf-8", errors="ignore").read())
        except OSError as e:
            log(f"[needle] filler missing, skipping: {f} ({e})")
    if not parts:
        raise SystemExit("no filler files readable")
    return "\n\n---\n\n".join(parts)


def build_spec(variant, L, run):
    """-> (needle_lines, depths_or_None, question, expected, label)"""
    c = needle_code(L, run)
    word, digits = c.split("-")
    line = f"IMPORTANT NOTE: The project access code is {c}. Remember this."
    if variant == "ask":
        return [line], None, QUESTION, c, c
    if variant == "yesno":
        return [line], None, f"Is the project access code {c}? Answer yes or no.", "yes", c
    if variant == "yesno_neg":
        wrong = f"{word}-{(int(digits) + 4321) % 10000:04d}"
        return ([line], None,
                f"Is the project access code {wrong}? Answer yes or no.", "no", wrong)
    if variant == "spell":
        return ([line], None,
                "Spell the project access code one character at a time, "
                "separated by spaces.", c, c)
    if variant == "words":
        rng = random.Random(0xBEEF ^ L)
        trio = "-".join(rng.sample([w for w in CODE_WORDS if w != word], 3))
        return ([f"IMPORTANT NOTE: The project access code is {trio}. "
                 f"Remember this."], None, QUESTION, trio, trio)
    if variant == "repeat":
        return [line] * 3, [0.35, 0.5, 0.65], QUESTION, c, c
    if variant == "verbose":
        spoken = " ".join(DIGIT_WORDS[int(d)] for d in digits)
        return ([f"IMPORTANT NOTE: The project access code is {c}, that is "
                 f"{word} dash {spoken}. Remember this."], None, QUESTION, c, c)
    # The two position-isolating variants. Every observed failure loses the TAIL
    # of the copied span (GLACIER-7741-ORYX -> GLACIER-7741 -> GLACIER;
    # PELICAN-1605 -> PELICAN), while a 1-token answer (yesno/yesno_neg) is
    # correct. If the loss is about POSITION IN THE ANSWER rather than the
    # information being unavailable, then asking for the digits FIRST should
    # recover them, and asking for only the word should always succeed.
    if variant == "digitsonly":
        return ([line], None,
                "What are the four digits at the end of the project access "
                "code? Reply with only those four digits.", digits, digits)
    if variant == "digitspad":
        # Identical retrieval task to `digitsonly`, but a forced multi-token
        # prefix pushes the digits from answer position ~0 to ~5+. Everything
        # else -- prompt, needle, depth, what must be recalled -- is unchanged,
        # so a PASS->FAIL flip isolates ANSWER POSITION as the variable and
        # implicates the decode path (e.g. the speculative block boundary)
        # rather than anything in attention.
        return ([line], None,
                "Reply with exactly this and nothing else: "
                "THE FINAL FOUR DIGITS ARE <the four digits at the end of the "
                "project access code>", digits, digits)
    if variant == "firstword":
        return ([line], None,
                "What is the word at the start of the project access code? "
                "Reply with only that word.", word, word)
    raise SystemExit(f"unknown variant: {variant}")


def build_context(target_bytes, lines, depths, blob, question):
    overhead = sum(len(l.encode()) for l in lines) + len(question.encode()) + 64
    fill = max(1, target_bytes - overhead)
    enc = blob.encode("utf-8", "ignore")
    text = (enc * ((fill // len(enc)) + 1))[:fill].decode("utf-8", "ignore")
    out = text
    # deepest-first so earlier byte offsets stay valid as we insert
    for line, depth in sorted(zip(lines, depths), key=lambda x: -x[1]):
        at = int(len(text) * depth)
        out = out[:at] + "\n\n" + line + "\n\n" + out[at:]
    return out + "\n\n" + question


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def one(base, model, L, variant, run, depth, blob, bpt, max_tok, top_n,
        timeout, log):
    lines, mdepths, question, expected, label = build_spec(variant, L, run)
    depths = mdepths if mdepths is not None else [depth]
    ctx = build_context(max(1, int(L * bpt)), lines, depths, blob, question)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": ctx}],
        "max_tokens": max_tok, "temperature": 0.0,
        "logprobs": True, "top_logprobs": top_n,
        "chat_template_kwargs": {"thinking": False},
    }
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        d = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    except Exception as e:
        log(f"[needle] L={L} {variant} run{run}: ERROR {type(e).__name__}: {e}")
        return {"length": L, "variant": variant, "run": run,
                "error": f"{type(e).__name__}: {e}"}
    dt = time.perf_counter() - t0

    ch = d["choices"][0]
    ans = ch["message"].get("content") or ""
    pt = d.get("usage", {}).get("prompt_tokens", 0)
    lp = ((ch.get("logprobs") or {}).get("content")) or []
    tokens = [{"tok": p.get("token"), "lp": round(p.get("logprob", 0.0), 4),
               "top": [(t.get("token"), round(t.get("logprob", 0.0), 3))
                       for t in (p.get("top_logprobs") or [])[:top_n]]}
              for p in lp]

    # THE decomposition: correct is not one bit, it is word vs digits.
    code = needle_code(L, run)
    word, digits = code.split("-")
    word_present = word.lower() in ans.lower()
    digits_present = digits in ans
    correct = norm(expected) in norm(ans)
    rec = {
        "length": L, "variant": variant, "run": run, "depth": depths,
        "actual_prompt_tokens": pt, "needle_abs_pos_est": [int(pt * x) for x in depths],
        "true_code": code, "expected": expected, "answer": ans,
        "correct": correct, "word_present": word_present,
        "digits_present": digits_present,
        "lost": ("nothing" if correct else
                 "digits_only" if word_present else "word_and_digits"),
        "wall_s": round(dt, 2), "tokens": tokens,
    }
    log(f"[needle] L={L:>6} {variant:<9} pt={pt:<6} "
        f"{'PASS' if correct else 'FAIL':<4} word={str(word_present):<5} "
        f"digits={str(digits_present):<5} lost={rec['lost']:<15} "
        f"({dt:.0f}s) want={label!r} got={ans[:44]!r}")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:8011")
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--lengths", default="3072,5120,8192")
    ap.add_argument("--variant", default="ask", choices=VARIANTS)
    ap.add_argument("--depth", type=float, default=0.5)
    ap.add_argument("--abs-pos", default=None,
                    help="comma-separated ABSOLUTE token positions; overrides "
                         "--depth (depth = pos / measured prompt_tokens)")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--bpt", type=float, default=3.6,
                    help="bytes/token for sizing; 3.6 matches needle_sweep so "
                         "prompts are byte-identical to its runs")
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--top-logprobs", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--filler-files", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    log = print
    files = a.filler_files.split(":") if a.filler_files else DEFAULT_FILLER
    blob = filler_blob(files, log)
    log(f"[needle] filler {len(blob.encode()):,} bytes; variant={a.variant}")

    lengths = [int(x) for x in a.lengths.split(",") if x]
    results = []
    for L in lengths:
        # An absolute position needs prompt_tokens, which is only known after a
        # request; bpt*L is a good enough estimate to convert, and the record
        # reports the measured position so the estimate is auditable.
        if a.abs_pos:
            est_pt = L * a.bpt / a.bpt  # == L; refined by actual_prompt_tokens
            depths = [min(0.999, max(0.0, int(p) / max(1, est_pt)))
                      for p in a.abs_pos.split(",")]
        else:
            depths = [a.depth]
        for depth in depths:
            for run in range(a.runs):
                results.append(one(a.endpoint, a.model, L, a.variant, run,
                                   depth, blob, a.bpt, a.max_tokens,
                                   a.top_logprobs, a.timeout, log))

    ok = [r for r in results if r.get("correct")]
    digits_only = [r for r in results if r.get("lost") == "digits_only"]
    log("")
    log(f"[needle] {len(ok)}/{len(results)} fully correct; "
        f"{len(digits_only)} lost DIGITS ONLY (word retrieved). "
        f"Word-retrieved-but-digits-lost is NOT a retrieval failure — see "
        f"PLAN.md 5.10 before concluding anything about attention.")
    if a.out:
        json.dump(results, open(a.out, "w"), indent=1)
        log(f"[needle] wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
