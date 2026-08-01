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
more severely at depth 0.1 than 0.5 (STATUS 5.16). `--abs-pos N` places the
needle's anchor (the digits) at an ABSOLUTE TOKEN position N -- tokenize, insert,
re-tokenize, verify within +-2 -- so it is independent of the filler's byte/token
density. (It previously converted N to a byte fraction, which is how the §2
wrong-column error arose.) Every record reports the MEASURED `needle_token_pos`.

Usage:
  python3 tools/needle_digits_probe.py --endpoint http://127.0.0.1:8011 \
      --lengths 3072,5120,8192 --variant ask --out /tmp/needle.json
  python3 tools/needle_digits_probe.py --lengths 8192 \
      --variant ask --abs-pos 1000,3871,8000
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


def _filler_text(target_bytes, lines, question, blob):
    """The repeated filler truncated to target_bytes minus prompt overhead.

    Byte-for-byte identical to the original inline build_context math, so prompts
    stay comparable to historical needle_sweep runs. Shared so the measured/token
    placement path cannot drift from it."""
    overhead = sum(len(l.encode()) for l in lines) + len(question.encode()) + 64
    fill = max(1, target_bytes - overhead)
    enc = blob.encode("utf-8", "ignore")
    return (enc * ((fill // len(enc)) + 1))[:fill].decode("utf-8", "ignore")


def build_context(target_bytes, lines, depths, blob, question):
    """Insert needles at BYTE fractions (`depths`) of the filler.

    Kept as the entry point verify_needle_token_position.py imports. New runs
    that need a MEASURED token position (or token-based placement) use
    build_context_measured, which shares this filler math.
    """
    text = _filler_text(target_bytes, lines, question, blob)
    out = text
    # deepest-first so earlier byte offsets stay valid as we insert
    for line, depth in sorted(zip(lines, depths), key=lambda x: -x[1]):
        at = int(len(text) * depth)
        out = out[:at] + "\n\n" + line + "\n\n" + out[at:]
    return out + "\n\n" + question


# ---------------------------------------------------------------------------
# Tokenization + MEASURED needle positions
#
# The needle is inserted at a BYTE fraction of the filler but reported
# historically as int(prompt_tokens * depth) -- a TOKEN fraction. Those differ
# whenever the filler's halves tokenize at different densities (markdown +
# Python is denser up front), which pointed the §5.19 rank trace 243 ratio-4
# columns away from the needle (README §2). Below we load the model's own
# tokenizer.json on the CPU and locate the needle's ANCHOR -- the digits when
# the code has them (the column a trace watches), else the line start -- by
# token. `tokenizers` is preferred (loads tokenizer.json directly; on this
# model add_bos/eos=false so it is byte-identical to AutoTokenizer's
# add_special_tokens=False path, verified to reproduce 9682/3871/967).
# ---------------------------------------------------------------------------

DEFAULT_TOKENIZER_DIR = "/root/models/DeepSeek-V4-Flash-0731"


class _TokCount:
    """Adapter: count_ids(str) -> number of tokens, no specials added."""

    def __init__(self, count_fn, source):
        self._count = count_fn
        self.source = source

    def count_ids(self, text):
        return self._count(text)


def load_tokenizer(tokenizer_dir):
    """Return (_TokCount | None, source_or_reason).

    Tries the standalone `tokenizers` lib (tokenizer.json), then
    `transformers.AutoTokenizer`; returns (None, reason) if neither imports so
    callers can degrade loudly instead of silently re-emitting the bad estimate.
    """
    import os

    json_path = tokenizer_dir
    if os.path.isdir(tokenizer_dir):
        json_path = os.path.join(tokenizer_dir, "tokenizer.json")
    try:
        from tokenizers import Tokenizer  # type: ignore

        tk = Tokenizer.from_file(json_path)
        return _TokCount(lambda s: len(tk.encode(s).ids),
                         f"tokenizers:{json_path}"), "ok"
    except Exception as e1:  # noqa: BLE001
        pass
    try:
        from transformers import AutoTokenizer  # type: ignore

        tk = AutoTokenizer.from_pretrained(tokenizer_dir,
                                           trust_remote_code=True)
        return _TokCount(
            lambda s: len(tk(s, add_special_tokens=False)["input_ids"]),
            f"transformers:{tokenizer_dir}"), "ok"
    except Exception as e2:  # noqa: BLE001
        return None, f"tokenizers import failed ({e1!r}); transformers failed ({e2!r})"


def _anchor_in_line(line, digits):
    """Byte offset within `line` of the position a trace should watch: the
    digits when the code has them, else the line start (0)."""
    if digits and digits in line:
        return line.index(digits)
    return 0


def _nth_find(haystack, needle, n):
    """Byte offset of the (0-indexed) n-th occurrence of `needle`, else -1."""
    start, idx = 0, -1
    for _ in range(n + 1):
        idx = haystack.find(needle, start)
        if idx < 0:
            return -1
        start = idx + 1
    return idx


def _find_byte_for_token(out, line, anchor_off, target, tok):
    """Byte offset `at` in `out` at which inserting `line` puts its anchor (at
    line-relative byte `anchor_off`) at ~`target` tokens.

    The anchor's prefix in the final ctx is `out[:at] + "\\n\\n" + line[:anchor_off]`;
    we tokenize exactly that, so the result matches how the position is later
    measured. Token count is monotone non-decreasing in `at`, so binary search.
    """
    pad = "\n\n"
    pre = line[:anchor_off]

    def prefix_tok(at):
        return tok.count_ids(out[:at] + pad + pre)

    n = len(out)
    if prefix_tok(n) <= target:
        return n
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if prefix_tok(mid) < target:
            lo = mid + 1
        else:
            hi = mid
    cands = [x for x in (lo - 1, lo) if 0 <= x <= n]
    return min(cands, key=lambda x: abs(prefix_tok(x) - target))


def build_context_measured(target_bytes, lines, placements, blob, question,
                           digits, tok):
    """Build the context and MEASURE each needle's anchor token position.

    placements: list parallel to `lines`; each is ("depth", frac) or ("tok", N).
      - ("depth", frac): insert at byte fraction `frac` of the filler (legacy).
      - ("tok", N): insert so the line's anchor (digits if present else start)
        lands at token position N. Processed shallowest-first so each anchor's
        prefix already contains every shallower needle; requires `tok`.
    digits: the digits substring used as anchor when present (may be None).
    tok: _TokCount or None (depth mode still works without it).

    Returns (ctx, total_tokens_or_None, positions) where positions is a list
    parallel to `lines` of {line_byte, line_token, anchor_byte, anchor_token,
    placed_target, placed_ok}. anchor_* are the digits position when digits are
    in the line, else the line-start position.
    """
    text = _filler_text(target_bytes, lines, question, blob)
    n_lines = len(lines)
    tok_mode = any(m == "tok" for m, _ in placements)
    if tok_mode and tok is None:
        raise SystemExit(
            "--abs-pos needs a tokenizer to place by token; install `tokenizers`"
            " (pip) or pass --tokenizer <model-dir>.")

    out = text
    if tok_mode:
        for i in sorted(range(n_lines), key=lambda i: placements[i][1]):
            line = lines[i]
            at = _find_byte_for_token(out, line,
                                      _anchor_in_line(line, digits),
                                      placements[i][1], tok)
            out = out[:at] + "\n\n" + line + "\n\n" + out[at:]
    else:
        for line, (m, frac) in sorted(zip(lines, placements),
                                      key=lambda x: -x[1][1]):
            at = int(len(text) * frac)
            out = out[:at] + "\n\n" + line + "\n\n" + out[at:]
    ctx = out + "\n\n" + question

    total = tok.count_ids(ctx) if tok is not None else None
    positions = []
    for i, line in enumerate(lines):
        lb = _nth_find(ctx, line, i)
        anchor_off = _anchor_in_line(line, digits)
        ab = lb + anchor_off if lb >= 0 else -1
        line_tok = tok.count_ids(ctx[:lb]) if (tok is not None and lb >= 0) else None
        anchor_tok = (tok.count_ids(ctx[:ab])
                      if (tok is not None and lb >= 0) else None)
        placed_target = placements[i][1] if placements[i][0] == "tok" else None
        placed_ok = (abs(anchor_tok - placed_target) <= 2
                     if (placed_target is not None and anchor_tok is not None)
                     else None)
        positions.append({
            "line_byte": lb, "line_token": line_tok,
            "anchor_byte": ab, "anchor_token": anchor_tok,
            "placed_target": placed_target, "placed_ok": placed_ok,
        })
    return ctx, total, positions


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def one(base, model, L, variant, run, place, blob, bpt, tok, max_tok, top_n,
        timeout, log):
    """Run one request. `place` is ("depth", frac) or ("tok", abs_token_pos);
    a token placement puts the needle's anchor (digits if present) at that
    absolute token position and requires `tok`."""
    lines, mdepths, question, expected, label = build_spec(variant, L, run)
    if place[0] == "tok":
        if len(lines) != 1:
            raise SystemExit(
                f"--abs-pos is single-needle, but variant {variant!r} inserts "
                f"{len(lines)} needle lines")
        placements = [place]
        req_depths = [None]
    else:
        req_depths = mdepths if mdepths is not None else [place[1]]
        placements = [("depth", d) for d in req_depths]

    code = needle_code(L, run)
    word, digits = code.split("-", 1)
    ctx, total_tok, positions = build_context_measured(
        max(1, int(L * bpt)), lines, placements, blob, question, digits, tok)

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
                "place_mode": place[0],
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
    word_present = word.lower() in ans.lower()
    digits_present = digits in ans
    correct = norm(expected) in norm(ans)

    pos0 = positions[0] if positions else {}
    rec = {
        "length": L, "variant": variant, "run": run,
        "place_mode": place[0], "depth": req_depths,
        "actual_prompt_tokens": pt,
        "context_tokens": total_tok,
        "position_measured": total_tok is not None,
        # The anchor = digits (the ratio-4 column a trace watches) when the
        # code has them, else the needle line start. These REPLACE the unsound
        # needle_abs_pos_est = int(pt * depth) that pointed traces 243 columns
        # away from the needle (README §2).
        "needle_token_pos": [p["anchor_token"] for p in positions],
        "needle_token_frac": ([round(p["anchor_token"] / total_tok, 4)
                               for p in positions] if total_tok
                              else [None] * len(positions)),
        "needle_byte_pos": [p["anchor_byte"] for p in positions],
        "needle_byte_frac": [round(p["anchor_byte"] / len(ctx), 4)
                             for p in positions],
        "needle_line_token_pos": [p["line_token"] for p in positions],
        "placed_target": pos0.get("placed_target"),
        "placed_ok": pos0.get("placed_ok"),
        "true_code": code, "expected": expected, "answer": ans,
        "correct": correct, "word_present": word_present,
        "digits_present": digits_present,
        "lost": ("nothing" if correct else
                 "digits_only" if word_present else "word_and_digits"),
        "wall_s": round(dt, 2), "tokens": tokens,
    }
    ptag = (f" place_tok={pos0.get('placed_target')} ok={pos0.get('placed_ok')}"
            if place[0] == "tok" else "")
    log(f"[needle] L={L:>6} {variant:<9} pt={pt:<6} "
        f"{'PASS' if correct else 'FAIL':<4} word={str(word_present):<5} "
        f"digits={str(digits_present):<5} lost={rec['lost']:<15} "
        f"ntok@anchor={pos0.get('anchor_token')}{ptag} "
        f"({dt:.0f}s) want={label!r} got={ans[:40]!r}")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:8011")
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--lengths", default="3072,5120,8192")
    ap.add_argument("--variant", default="ask", choices=VARIANTS)
    ap.add_argument("--depth", type=float, default=0.5)
    ap.add_argument("--abs-pos", default=None,
                    help="comma-separated ABSOLUTE TOKEN positions for the "
                         "needle's anchor (digits if present, else line start); "
                         "one request per position. Places BY TOKEN (tokenize, "
                         "insert, verify within +-2) -- NOT by byte fraction. "
                         "Overrides --depth; single-needle variants only.")
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER_DIR,
                    help="model dir (with tokenizer.json) or tokenizer.json "
                         "path used to MEASURE needle token positions and to "
                         "place --abs-pos. Pass '' to disable (positions become "
                         "UNMEASURED and --abs-pos errors).")
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

    if a.tokenizer:
        tok, tok_status = load_tokenizer(a.tokenizer)
        if tok is None:
            log(f"[needle] WARNING: tokenizer unavailable ({tok_status}); "
                f"needle_token_pos will be UNMEASURED and --abs-pos will error. "
                f"Install `tokenizers` (pip) or fix --tokenizer.")
        else:
            log(f"[needle] tokenizer: {tok.source}")
    else:
        tok = None
        log("[needle] tokenizer disabled (--tokenizer ''); positions UNMEASURED")

    abs_positions = [int(x) for x in a.abs_pos.split(",")] if a.abs_pos else []
    lengths = [int(x) for x in a.lengths.split(",") if x]
    results = []
    for L in lengths:
        places = ([("tok", n) for n in abs_positions]
                  if abs_positions else [("depth", a.depth)])
        for place in places:
            for run in range(a.runs):
                results.append(one(a.endpoint, a.model, L, a.variant, run,
                                   place, blob, a.bpt, tok, a.max_tokens,
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
