"""The measurement probes. Same methodology as the historical tools/ scripts
(bench_tok, prefill_probe, needle_probe, probe_quant_quality, decode_ab), but
returning structured dicts instead of prints.

All greedy (temperature 0). Raw samples are kept in the result — aggregates
are derived, never the only record."""

import hashlib
import json
import math
import os
import random
import re
import statistics
import threading
import time
import urllib.request

WORDS = ("alpha quantum river matrix ember glacier syntax violet nimbus cobalt "
         "tangent fjord lantern zephyr cipher marble thunder willow plasma onyx "
         "harbor crimson vector lattice meadow falcon pixel saffron tundra orbit"
         ).split()

NEEDLE_SECRET = "GLACIER-7741-ORYX"

DECODE_PROMPT = '''Refactor this function to use pathlib and add type hints:

import os

def find_files(directory, extension):
    results = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.endswith(extension):
                results.append(os.path.join(root, f))
    return results

Refactored version:
'''

COHERENCE_PROMPTS = [
    "Q: What is the capital of Australia?\nA:",
    "Q: A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left?\nA:",
    "Write a Python function that returns the n-th Fibonacci number iteratively.\n\n```python\n",
    "Translate to French: 'The weather is beautiful today, let's go for a walk in the park.'\nFrench:",
    "Q: If all bloops are razzies and all razzies are lazzies, are all bloops definitely lazzies? Explain step by step.\nA:",
    ("Summarize the following paragraph in one sentence:\n\n"
     "The industrial revolution, which began in Britain in the late eighteenth century, "
     "transformed economies that had been based on agriculture and handicrafts into economies "
     "based on large-scale industry, mechanized manufacturing, and the factory system. New "
     "machines, new power sources, and new ways of organizing work made existing industries "
     "more productive and efficient.\n\nSummary:"),
    "Q: Which planet in our solar system has the most moons?\nA:",
    "Q: What is 847 + 256? Show your work.\nA:",
    "Write a one-line Python list comprehension that squares the even numbers in a list called xs.\n\n```python\n",
    "Explain in two sentences why the sky is blue.\nAnswer:",
    "Q: What is the next number in the sequence 2, 6, 18, 54?\nA:",
    "The three primary colors of light are",
]

ARITH = [
    ("What is 347 * 28? Reply with only the final number.", 9716),
    ("What is 86 * 74? Reply with only the final number.", 6364),
    ("What is 512 * 943? Reply with only the final number.", 482816),
    ("What is 38 + 277 + 4019 + 86? Reply with only the final number.", 4420),
    ("What is 1234 + 5678 + 9101 + 234 + 87? Reply with only the final number.", 16334),
]


def _post(url, payload, timeout):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _completion(base, model, prompt, max_tokens, timeout=900, ignore_eos=True):
    t0 = time.perf_counter()
    d = _post(base + "/v1/completions", {
        "model": model, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0.0, "ignore_eos": ignore_eos}, timeout)
    return d, time.perf_counter() - t0


def warmup(base, model, n, log=print):
    for i in range(n):
        random.seed(0xBEEF + i)
        p = " ".join(random.choice(WORDS) for _ in range(24))
        _completion(base, model, p, 128)
        log(f"[warmup] {i + 1}/{n}")


def decode(base, model, runs=5, max_tokens=512, log=print, **_):
    _completion(base, model, DECODE_PROMPT, max_tokens)   # warmup, dropped
    samples, texts = [], []
    for i in range(runs):
        d, dt = _completion(base, model, DECODE_PROMPT, max_tokens)
        ct = d["usage"]["completion_tokens"]
        text = d["choices"][0]["text"]
        samples.append(round(ct / dt, 1))
        texts.append(hashlib.sha1(text.encode()).hexdigest()[:10])
        log(f"[decode] run {i + 1}/{runs}: {samples[-1]} tok/s")
    s = sorted(samples)
    return {
        "samples_tok_s": samples,
        "median_tok_s": round(statistics.median(s), 1),
        "min_tok_s": s[0], "max_tok_s": s[-1],
        "spread_pct": round((s[-1] - s[0]) / statistics.median(s) * 100, 1),
        "distinct_outputs": f"{len(set(texts))}/{len(texts)}",
        "max_tokens": max_tokens,
    }


def batch_decode(base, model, concurrency=(1, 4, 8), max_tokens=384, runs=3,
                 log=print, **_):
    def stream_prompt(i):
        random.seed(0xD00D ^ i)
        return " ".join(random.choice(WORDS) for _ in range(32))

    levels = {}
    for n in concurrency:
        aggs = []
        for r in range(runs):
            done = [None] * n
            def worker(i):
                d, _dt = _completion(base, model, stream_prompt(i + r * 100),
                                     max_tokens, timeout=1800)
                done[i] = d["usage"]["completion_tokens"]
            t0 = time.perf_counter()
            ths = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
            for t in ths:
                t.start()
            for t in ths:
                t.join()
            wall = time.perf_counter() - t0
            aggs.append(round(sum(done) / wall, 1))
        med = statistics.median(sorted(aggs))
        levels[str(n)] = {
            "aggregate_tok_s": round(med, 1),
            "per_stream_tok_s": round(med / n, 1),
            "samples": aggs,
        }
        log(f"[batch] {n} streams: {med:.1f} tok/s aggregate")
    return {"levels": levels, "max_tokens": max_tokens}


def prefill(base, model, words=8000, runs=3, log=print, **_):
    samples = []
    for i in range(runs):
        random.seed(0xC0FFEE ^ (words + i))
        p = " ".join(random.choice(WORDS) for _ in range(words))
        d, dt = _completion(base, model, p, 1, timeout=3600, ignore_eos=False)
        pt = d["usage"]["prompt_tokens"]
        samples.append({"prompt_tokens": pt, "wall_s": round(dt, 2),
                        "tok_s": round(pt / dt, 0)})
        log(f"[prefill] run {i + 1}/{runs}: {pt} tok in {dt:.1f}s "
            f"= {pt / dt:,.0f} tok/s")
    med = statistics.median(sorted(s["tok_s"] for s in samples))
    return {"words": words, "samples": samples, "median_tok_s": med}


def needle(base, model, sizes_words=None, sizes_frac=None, context=None,
           depth=0.5, max_tokens=512, log=print, **_):
    if not sizes_words:
        # conservative words-per-token guess; the result records the real count
        sizes_words = [max(1000, int(f * (context or 0) / 1.35 / 1000) * 1000)
                       for f in (sizes_frac or [0.03, 0.5])]
    cases = []
    for nwords in sizes_words:
        random.seed(7)
        filler = [random.choice(WORDS) for _ in range(nwords)]
        at = int(len(filler) * depth)
        needle_txt = (f"IMPORTANT FACT: the vault passphrase is "
                      f"{NEEDLE_SECRET}. Remember it exactly.")
        ctx = (" ".join(filler[:at]) + "\n\n" + needle_txt + "\n\n"
               + " ".join(filler[at:]))
        user = (ctx + "\n\nQuestion: What is the vault passphrase? "
                "Reply with ONLY the passphrase, nothing else.")
        t0 = time.perf_counter()
        try:
            r = _post(base + "/v1/chat/completions", {
                "model": model,
                "messages": [{"role": "user", "content": user}],
                "max_tokens": max_tokens, "temperature": 0,
                "chat_template_kwargs": {"thinking": False},
            }, timeout=3600)
        except Exception as e:  # noqa: BLE001 — a FAIL case, not a crash
            cases.append({"words": nwords, "error": f"{type(e).__name__}: {e}",
                          "pass": False})
            log(f"[needle] {nwords} words: ERROR {e}")
            continue
        dt = time.perf_counter() - t0
        m = r["choices"][0]["message"]
        msg = ((m.get("content") or "") + " "
               + (m.get("reasoning") or m.get("reasoning_content") or ""))
        ok = NEEDLE_SECRET in msg
        pt = r.get("usage", {}).get("prompt_tokens", 0)
        cases.append({"words": nwords, "prompt_tokens": pt, "depth": depth,
                      "ttft_gen_s": round(dt, 1), "pass": ok})
        log(f"[needle] {pt} tok @ depth {depth}: "
            f"{'PASS' if ok else 'FAIL'} ({dt:.0f}s)")
    return {"cases": cases, "all_pass": all(c["pass"] for c in cases)}


def arithmetic(base, model, log=print, **_):
    rows = []
    for q, gold in ARITH:
        r = _post(base + "/v1/chat/completions", {
            "model": model, "messages": [{"role": "user", "content": q}],
            "max_tokens": 4096, "temperature": 0}, timeout=900)
        content = (r["choices"][0]["message"].get("content") or "").strip()
        nums = re.findall(r"-?[\d,]*\d", content.replace(",", ""))
        got = int(nums[-1]) if nums else None
        rows.append({"gold": gold, "got": got, "ok": got == gold})
    score = sum(r["ok"] for r in rows)
    log(f"[arith] {score}/{len(rows)}")
    return {"score": score, "of": len(rows), "rows": rows}


def _degenerate(text):
    """Cheap loop detector: a short chunk repeating many times back-to-back."""
    if len(text) < 64:
        return False
    for w in (4, 8, 16):
        chunk = text[-w:]
        if chunk.strip() and text.endswith(chunk * min(8, len(text) // w)):
            return True
    return False


def coherence(base, model, log=print, **_):
    texts, degen = [], 0
    for i, p in enumerate(COHERENCE_PROMPTS):
        d, _dt = _completion(base, model, p, 128, ignore_eos=False)
        t = d["choices"][0]["text"]
        texts.append(t)
        if _degenerate(t):
            degen += 1
            log(f"[coherence] prompt {i}: DEGENERATE tail")
    log(f"[coherence] degenerate {degen}/{len(texts)}")
    return {"degenerate": degen, "of": len(texts), "pass": degen == 0,
            "texts": texts}


def quality(base, model, log=print, *, profile, runs=200, concurrency=2,
            max_tokens=6000, baseline=None, request_overrides=None,
            tool=None, artifacts_dir=None, artifact_tag=None, **_):
    """Dataset-eval probe (GSM8K / GPQA-diamond / …) via llm-inference-bench.

    Runs the pinned external tool against the recipe's server and keeps BOTH
    representations: the raw tool JSON (append-only artifact next to the
    result — flips and per-item data stay reviewable) and compact aggregates
    in the result itself. `baseline` names an entry in bench/baselines/
    (a NATIVE reference measured by the same tool): the tool then computes
    the paired comparison (accuracy flips, completion-token inflation) that
    is THE quality KPI of this project — parity with native, not absolute
    scores. `request_overrides` is merged into every request payload (e.g.
    {"chat_template_kwargs": {"thinking": true}, "temperature": 1.0} for
    think-mode evals)."""
    import json as _json
    import subprocess
    import sys as _sys

    tool = tool or os.environ.get(
        "LLM_BENCH", "/root/workspace/llm-inference-bench/llm_decode_bench.py")
    if not os.path.exists(tool):
        raise RuntimeError(
            f"quality probe needs llm-inference-bench (looked at {tool}; "
            "set `quality_tool` in the box yaml or LLM_BENCH)")
    port = base.rsplit(":", 1)[1].split("/", 1)[0]
    tag = artifact_tag or profile
    out_json = os.path.join(artifacts_dir or "/tmp",
                            f"quality__{tag}.json")
    cmd = [_sys.executable, tool, "--port", port, "--model", model,
           "--test-profile", profile,
           "--profile-runs", str(runs),
           "--profile-concurrency", str(concurrency),
           "--max-tokens", str(max_tokens),
           "--display-mode", "plain", "--no-hw-monitor",
           "--output", out_json]
    from common import baseline_path  # late import: probes stays standalone
    if baseline:
        cmd += ["--compare-baseline", baseline_path(baseline)]
    if request_overrides:
        cmd += ["--request-overrides-json", _json.dumps(request_overrides)]
    log(f"[quality] {profile} runs={runs} c={concurrency}"
        + (f" baseline={baseline}" if baseline else "")
        + (" +overrides" if request_overrides else ""))
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=8 * 3600)
    if r.returncode != 0 or not os.path.exists(out_json):
        raise RuntimeError(
            f"tool exit {r.returncode}: {(r.stderr or r.stdout)[-400:]}")
    with open(out_json) as f:
        data = _json.load(f)
    acc = (data.get("accuracy") or {})
    toks = [x.get("completion_tokens") or 0 for x in data.get("runs", [])
            if x.get("phase") == "profile"]
    toks.sort()
    res = {
        "profile": profile, "runs": runs, "concurrency": concurrency,
        "max_tokens": max_tokens,
        "request_overrides": request_overrides or {},
        "accuracy_pct": round(100.0 * acc.get("accuracy", 0.0), 2),
        "correct": acc.get("correct"), "of": acc.get("scored"),
        "truncated_no_answer": acc.get("truncated_no_answer"),
        "hit_max_tokens": acc.get("hit_max_tokens"),
        "tokens_avg": round(sum(toks) / len(toks), 1) if toks else None,
        "tokens_p50": toks[len(toks) // 2] if toks else None,
        "tokens_p90": toks[int(0.9 * len(toks))] if toks else None,
        "tool_sha": _tool_sha(tool),
        "duration_s": round(time.time() - t0, 1),
        "artifact": os.path.basename(out_json),
    }
    if baseline:
        res["baseline"] = baseline
        cmp_ = data.get("comparison") or {}
        ct = cmp_.get("completion_tokens") or {}
        if cmp_:
            bm, cm = ct.get("baseline_mean"), ct.get("candidate_mean")
            res["vs_baseline"] = {
                "acc_delta_pp": round(cmp_["delta_pp"], 2)
                if cmp_.get("delta_pp") is not None else None,
                "flips_only_baseline": cmp_.get("flips_baseline_only_correct"),
                "flips_only_candidate": cmp_.get("flips_candidate_only_correct"),
                "mcnemar_p": cmp_.get("mcnemar_exact_p"),
                "token_inflation_pct": round((cm - bm) / bm * 100, 1)
                if bm and cm else None,
            }
    log(f"[quality] {profile}: {res['accuracy_pct']}% "
        f"({res['correct']}/{res['of']}), tokens avg {res['tokens_avg']}"
        + (f", vs {baseline}: {res.get('vs_baseline')}" if baseline else ""))
    return res


def _tool_sha(tool):
    try:
        import subprocess
        d = os.path.dirname(os.path.abspath(tool))
        sha = subprocess.run(["git", "-C", d, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", d, "status", "--short"],
                               capture_output=True, text=True).stdout.strip()
        return sha + ("+dirty" if dirty else "") if sha else None
    except Exception:  # noqa: BLE001
        return None


# --- derailment probe ---------------------------------------------------------
#
# Pure client-side probe that measures output degeneration vs prompt context
# length. The context is N random words from WORDS (random.seed(L) => every
# run at a given L is byte-identical) with the instruction APPENDED AFTER the
# filler, so prefill processes the full context before the real turn. Greedy
# throughout. Two modes per length:
#   overgen — max_tokens=512 (won't-stop / repetition; chat has no ignore_eos,
#             so finish_reason=="length" + large out_tokens IS the signal)
#   stop    — max_tokens=64  (does it stop after OK?)
#
# ALL metrics come from /v1/chat/completions (the PRIMARY and ONLY generation
# path). The model is chat-tuned (deepseek_v4 tokenizer + tool/reasoning
# parsers): RAW /v1/completions confabulates on this model even with no filler
# ("Reply with the single word OK." -> "This is a JSON only API..."), so the
# completions path is not used at all. Chat also returns the FULL top_logprobs
# (20) per token, which is the KEY next-token-distribution metric.
#
# IMPORTANT D-score note: on this model a CLEAN confident "OK" already has
# near-collapsed entropy (median_entropy ~0.02, frac maxp>0.9 == 1.0) because
# the model is simply certain. So entropy collapse ALONE is not derailment —
# it only counts when the output is long enough for repetition to manifest.
# The frac_maxprob / out_tokens/max_tokens terms are therefore gated on
# out_tokens >= DERAILMENT_MIN_LEN. A short confident answer scores D≈0.
# 262144 is opt-in (one ~262K prefill is slow) — the default caps at 131072.

DERAILMENT_INSTRUCTION = "Reply with the single word OK."
DERAILMENT_LENGTHS = [1024, 4096, 8192, 16384, 32768, 65536, 131072]
DERAILMENT_LENGTHS_FULL = DERAILMENT_LENGTHS + [262144]
DERAILMENT_TOP_LOGPROBS = 20
# below this output length, confidence and won't-stop don't count as derail
# (a short confident answer is clean, not collapsed). Repetition needs length.
DERAILMENT_MIN_LEN = 16


def _ngram_repetition(text, n):
    """1 - distinct/total over word-level n-grams (0 = no repetition)."""
    words = text.split()
    if len(words) < n:
        return 0.0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams) if grams else 0.0


def _dist_stats(top_lp):
    """(max_prob, entropy, n) over a {token: logprob} dict via softmax over the
    returned alternatives."""
    if not top_lp:
        return None
    vals = list(top_lp.values())
    m = max(vals)
    exps = [math.exp(v - m) for v in vals]
    z = sum(exps)
    probs = [e / z for e in exps]
    ent = -sum(p * math.log(p) for p in probs if p > 0.0)
    return max(probs), ent, len(vals)


def _chat_logprob_stats(content):
    """Per-token distribution stats from a chat logprobs.content list (each
    item: {token, logprob, top_logprobs:[{token, logprob}, ...]}). The first
    generated token = the post-prefill turn boundary."""
    empty = {"per_token": [], "tokens": [], "median_entropy": None,
             "frac_maxprob_gt_0_9": None, "turn_boundary": None,
             "n_top_median": None}
    if not content:
        return empty
    per, tokens = [], []
    for tok in content:
        tokens.append(tok.get("token", ""))
        tl = tok.get("top_logprobs") or []
        mapping = {x.get("token", "?"): x.get("logprob", 0.0) for x in tl}
        per.append(_dist_stats(mapping))
    ents = [s[1] for s in per if s]
    maxps = [s[0] for s in per if s]
    ns = [s[2] for s in per if s]
    return {
        "per_token": per, "tokens": tokens,
        "median_entropy": round(statistics.median(ents), 4) if ents else None,
        "frac_maxprob_gt_0_9": round(sum(1 for m in maxps if m > 0.9) / len(maxps), 4)
                                 if maxps else None,
        "turn_boundary": per[0] if per else None,
        "n_top_median": statistics.median(ns) if ns else None,
    }


def _timed_post(url, payload, timeout, log, what=""):
    """_post with wall-time and retry-once on transient errors. Returns
    (data_or_None, wall_s, error_or_None) so one failed request cannot abort
    the sweep."""
    last = None
    for attempt in range(2):
        try:
            t0 = time.perf_counter()
            d = _post(url, payload, timeout)
            return d, time.perf_counter() - t0, None
        except Exception as e:  # noqa: BLE001 — record, don't crash the sweep
            last = f"{type(e).__name__}: {e}"
            if attempt == 0:
                log(f"  [derailment] transient{what}: {last}; retrying once")
                continue
    return None, 0.0, last


def _calibrate_wpt(base, model, log):
    """One tiny chat request to measure tokens-per-word for this tokenizer, so
    the filler guess lands near the target length (the real prompt_tokens is
    still recorded per request)."""
    nwords = 500
    random.seed(0x1BEEF)
    sample = " ".join(random.choice(WORDS) for _ in range(nwords))
    try:
        t0 = time.perf_counter()
        d = _post(base + "/v1/chat/completions", {
            "model": model,
            "messages": [{"role": "user", "content": sample}],
            "max_tokens": 1, "temperature": 0.0}, 300)
        dt = time.perf_counter() - t0
        pt = d["usage"]["prompt_tokens"]
        return {"sample_words": nwords, "prompt_tokens": pt,
                "tok_per_word": round(pt / nwords, 4),
                "wall_s": round(dt, 2)}
    except Exception as e:  # noqa: BLE001
        log(f"[derailment] calibration failed ({e}); defaulting to 1.0 tok/word")
        return {"sample_words": nwords, "tok_per_word": 1.0,
                "error": f"{type(e).__name__}: {e}"}


def _derailment_score(rec, mode, max_tok):
    """D in [0,1]. Ungated terms: rep_8, 1 - token_type_ratio, and (stop mode
    hitting length). Length-gated terms (only when out_tokens >= MIN_LEN, i.e.
    when there is enough output for repetition to be meaningful):
    frac_maxprob_gt_0_9 (collapse on repeating tokens) and out_tokens/max_tok
    (won't-stop). A short confident answer therefore scores D≈0."""
    candidates = [
        rec.get("rep_8", 0.0) or 0.0,
        1.0 - (rec.get("token_type_ratio", 1.0) or 1.0),
    ]
    out = rec.get("out_tokens") or 0
    if out >= DERAILMENT_MIN_LEN:
        fmp = rec.get("frac_maxprob_gt_0_9")
        if fmp is not None:
            candidates.append(fmp)
        if max_tok:
            candidates.append(round(min(out / max_tok, 1.0), 4))
    if mode == "stop" and rec.get("stop_reason") == "length":
        candidates.append(1.0)
    return round(max(candidates), 4)


def _derailment_one(base, model, prompt, L, mode, run, max_tok, timeout, log):
    """One greedy /v1/chat/completions request; all metrics derived from it."""
    payload = {"model": model,
               "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tok, "temperature": 0.0,
               "logprobs": True, "top_logprobs": DERAILMENT_TOP_LOGPROBS}
    d, dt, err = _timed_post(base + "/v1/chat/completions", payload, timeout,
                             log, what=f" L={L} {mode}")
    rec = {"length": L, "mode": mode, "run": run, "target_length": L,
           "max_tokens": max_tok}
    if err:
        log(f"[derailment] L={L:>6} {mode:7} run {run}: ERROR {err}")
        rec["error"] = err
        rec["D"] = None
        return rec
    ch = (d.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    text = msg.get("content") or ""
    finish = ch.get("finish_reason")
    content = ((ch.get("logprobs") or {}).get("content")) or []
    usage = d.get("usage", {})
    pt = usage.get("prompt_tokens", 0)
    ct = usage.get("completion_tokens", 0)
    instr = bool(re.match(r"^\s*OK\b", text[:12]))
    rep3 = round(_ngram_repetition(text, 3), 4)
    rep8 = round(_ngram_repetition(text, 8), 4)
    rep16 = round(_ngram_repetition(text, 16), 4)
    lp = _chat_logprob_stats(content)
    gen_toks = lp["tokens"] or text.split()
    ttr = round(len(set(gen_toks)) / len(gen_toks), 4) if gen_toks else 1.0
    tb = lp["turn_boundary"]
    rec.update({
        "prompt_tokens": pt, "out_tokens": ct, "stop_reason": finish,
        "instruction_followed": instr,
        "rep_3": rep3, "rep_8": rep8, "rep_16": rep16,
        "token_type_ratio": ttr,
        "median_entropy": lp["median_entropy"],
        "frac_maxprob_gt_0_9": lp["frac_maxprob_gt_0_9"],
        "turn_boundary_entropy": round(tb[1], 4) if tb else None,
        "turn_boundary_maxprob": round(tb[0], 4) if tb else None,
        "n_top_median": lp["n_top_median"],
        "wall_s": round(dt, 2),
        "out_tok_s": round(ct / dt, 1) if dt > 0 else None,
        "prefill_tps": round(pt / dt, 0) if dt > 0 else None,
        "text_sha1": hashlib.sha1(text.encode()).hexdigest()[:12],
        "text_head": text[:60],
    })
    rec["D"] = _derailment_score(rec, mode, max_tok)
    log(f"[derailment] L={L:>6} {mode:7} run {run}: pt={pt} out={ct} "
        f"stop={finish} instr={instr} rep8={rep8:.2f} ttr={ttr:.2f} "
        f"H={rec['median_entropy']} tb_maxp={rec['turn_boundary_maxprob']} "
        f"D={rec['D']:.2f}")
    return rec


def _med(rs, key, fallback=None):
    vals = [r.get(key) for r in rs if r.get(key) is not None]
    if not vals and fallback:
        vals = [r.get(fallback) for r in rs if r.get(fallback) is not None]
    return round(statistics.median(vals), 4) if vals else None


def _derailment_summary(raw, lengths, modes, runs):
    per_length, d_table, entropy_curve, onset = {}, {}, {}, {}
    for mode in modes:
        d_table[mode] = {}
        entropy_curve[mode] = {}
        onset_L = None
        for L in lengths:
            rs = [r for r in raw if r.get("length") == L
                  and r.get("mode") == mode and "error" not in r]
            if not rs:
                continue
            d_runs = [r["D"] for r in rs if r.get("D") is not None]
            d_med = round(statistics.median(d_runs), 4) if d_runs else None
            d_table[mode][L] = d_med
            entropy_curve[mode][L] = _med(rs, "median_entropy")
            majority = (sum(1 for v in d_runs if v >= 0.5)
                        >= max(1, (runs + 1) // 2)) if d_runs else False
            if majority and onset_L is None:
                onset_L = L
            per_length.setdefault(str(L), {})[mode] = {
                "D_runs": d_runs, "D_median": d_med,
                "median_entropy": entropy_curve[mode][L],
                "frac_maxprob_gt_0_9": _med(rs, "frac_maxprob_gt_0_9"),
                "rep_8": _med(rs, "rep_8"),
                "token_type_ratio": _med(rs, "token_type_ratio"),
                "out_tokens": _med(rs, "out_tokens"),
                "prompt_tokens": _med(rs, "prompt_tokens"),
                "turn_boundary_maxprob": _med(rs, "turn_boundary_maxprob"),
                "stop_reasons": [r.get("stop_reason") for r in rs],
                "instruction_followed_frac": round(
                    sum(1 for r in rs if r.get("instruction_followed")) / len(rs), 3),
                "prefill_tps": _med(rs, "prefill_tps"),
            }
        onset[mode] = onset_L
    return {"per_request": raw, "per_length": per_length,
            "D_table": d_table, "onset_L": onset, "entropy_curve": entropy_curve}


def _derailment_print(summary, log):
    log("")
    log("=== derailment (chat-only) ===")
    hdr = (f"{'length':>7} {'mode':7} {'D':>5} {'H_med':>7} "
           f"{'tb_maxp':>7} {'rep_8':>6} {'out_tok':>7} "
           f"{'stop':>6} {'instr':>5} {'pf_tps':>7}")
    log(hdr)
    log("-" * len(hdr))
    for mode in summary["modes"]:
        for L in sorted(summary["D_table"].get(mode, {})):
            agg = summary["per_length"][str(L)][mode]
            H = agg["median_entropy"]
            tb = agg["turn_boundary_maxprob"]
            log(f"{L:>7} {mode:7} {(agg['D_median'] or 0):>5.2f} "
                f"{(H if H is not None else float('nan')):>7.3f} "
                f"{(tb if tb is not None else float('nan')):>7.3f} "
                f"{(agg['rep_8'] or 0):>6.2f} "
                f"{(agg['out_tokens'] or 0):>7.0f} "
                f"{(agg['stop_reasons'][0] if agg['stop_reasons'] else '-'):>6} "
                f"{agg['instruction_followed_frac']:>5.2f} "
                f"{(agg['prefill_tps'] or 0):>7.0f}")
    log("")
    for mode, L in summary["onset_L"].items():
        log(f"onset L* ({mode}): {L}")
    log(f"endpoint=/v1/chat/completions "
        f"top_logprobs={summary['top_logprobs_requested']} "
        f"(n_top_median observed={summary.get('n_top_median_observed')}); "
        f"D_min_len_gate={DERAILMENT_MIN_LEN}")


def derailment(base, model, lengths=None, runs=3, modes=("overgen", "stop"),
               timeout=900, log=print, **_):
    """Output-degeneration vs prompt context length. Pure client-side; hits
    /v1/chat/completions only (the model is chat-tuned; raw /v1/completions
    confabulates). Returns a dict with raw per-request records, a D(L) table
    per mode, the onset length L*, and the median_entropy / turn-boundary
    curves. See the block docstring above for the D-score gating rationale."""
    lengths = list(lengths or DERAILMENT_LENGTHS)
    modes = tuple(modes)
    cal = _calibrate_wpt(base, model, log)
    wpt = cal.get("tok_per_word", 1.0) or 1.0
    raw = []
    n_top_observed = None
    for L in lengths:
        random.seed(L)
        filler = " ".join(random.choice(WORDS)
                          for _ in range(max(1, int(L / wpt))))
        prompt = filler + "\n\n" + DERAILMENT_INSTRUCTION
        for mode in modes:
            max_tok = 512 if mode == "overgen" else 64
            for run in range(runs):
                rec = _derailment_one(base, model, prompt, L, mode, run,
                                      max_tok, timeout, log)
                raw.append(rec)
                if rec.get("n_top_median") is not None:
                    n_top_observed = rec["n_top_median"]
    summary = _derailment_summary(raw, lengths, modes, runs)
    summary.update({
        "lengths": lengths, "modes": list(modes), "runs": runs,
        "calibration": cal,
        "endpoint": "chat/completions",
        "top_logprobs_requested": DERAILMENT_TOP_LOGPROBS,
        "n_top_median_observed": n_top_observed,
        "d_score_min_len_gate": DERAILMENT_MIN_LEN,
        "instruction": DERAILMENT_INSTRUCTION,
    })
    _derailment_print(summary, log)
    return summary


# --- needle_sweep probe --------------------------------------------------------
#
# Mid-context needle length-sweep: retrieval quality vs prompt length. This is
# the before/after metric for the FP8-delta kernel fix. A unique access code is
# placed at a configurable depth (default 0.5 = mid-context) inside realistic
# filler, then the model is asked to return ONLY the code. Greedy (temp 0) so a
# single run is deterministic; the sweep records correctness + the answer-token
# confidence (mean top-token softmax prob over the first ~6 answer tokens) at
# each length. CONFIRMED failure to reproduce: mid-context needle at temp 0 is
# CORRECT at 2K ("ZEPHYR-4429") and WRONG/GARBLED at 131K ("ZEPHYRTHALOSE...").
#
# FILLER MUST BE REALISTIC. Random-WORDS filler did NOT reproduce the failure
# for an easy end-of-context retrieval task; concatenated REAL repo text does.
# Default source = the fixed file list below (deterministic given the set),
# overridable via NEEDLE_FILLER_FILES (colon-separated) or --filler-files. The
# blob is byte-repeated and truncated to the target length.
#
# ALL metrics come from /v1/chat/completions (the model is chat-tuned; raw
# /v1/completions confabulates). thinking:false so the tight max_tokens budget
# is spent on the answer, not a reasoning trace. 262144 is opt-in (one ~262K
# prefill is slow: ~512 tok/s real prefill -> 131K ~= 256s, 262K ~= 512s).

NEEDLE_DEFAULT_LENGTHS = [2048, 16384, 32768, 65536, 131072]
NEEDLE_LENGTHS_FULL = NEEDLE_DEFAULT_LENGTHS + [262144]
NEEDLE_TOP_LOGPROBS = 20
NEEDLE_ANSWER_TOKENS = 6        # max-prob averaged over first N answer tokens
NEEDLE_MAX_TOKENS = 24
NEEDLE_QUESTION = "What is the project access code? Reply with only the code."

# Distinctive, pronounceable code words (kept out of the common-filler vocabulary
# so the code is unambiguous to retrieve and easy to confabulate-away when the
# kernel degrades at length).
NEEDLE_CODE_WORDS = [
    "ZEPHYR", "ORYX", "GLACIER", "NIMBUS", "COBALT", "FALCON", "LANTERN",
    "CYPRESS", "OBSIDIAN", "PELICAN", "GRANITE", "MARLIN", "PUMICE", "SAFFRON",
    "TUNDRA", "VECTOR", "WILLOW", "EMBER",
]

NEEDLE_DEFAULT_FILLER_FILES = [
    "CONCEPTS.md",
    "docs/ada-sm89-port.md",
    "kernels/MANIFEST.md",
    "AGENTS.md",
    "overlay/vllm/vllm/model_executor/layers/quantization/utils/moe_w2_cubit.py",
    "/root/autostart/CLAUDE.md",
]


def _needle_code(L, run):
    """Deterministic unique code for a (length, run) pair: WORD-DDDD."""
    rng = random.Random(0x4E4C ^ L ^ (run * 0x10001))   # 'NL' = Needle Length
    word = rng.choice(NEEDLE_CODE_WORDS)
    digits = f"{rng.randint(0, 9999):04d}"
    return f"{word}-{digits}"


def _needle_filler_blob(log, files):
    """Concatenate the filler source files (repo-relative OR absolute) into one
    UTF-8 text blob. Missing files are skipped with a warning so a relocated
    tree degrades gracefully instead of aborting the sweep."""
    # probes.py is in <repo>/bench/runner/ -> repo root is three dirnames up
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parts, used = [], []
    for f in files:
        p = f if os.path.isabs(f) else os.path.join(repo, f)
        try:
            with open(p, encoding="utf-8", errors="ignore") as fh:
                parts.append(fh.read())
            used.append(f)
        except OSError as e:
            log(f"[needle_sweep] filler file missing, skipping: {f} ({e})")
    if not parts:
        # last-resort deterministic fallback so the probe still runs
        random.seed(0xF11)
        parts.append(" ".join(random.choice(WORDS) for _ in range(4000)))
        used.append("<WORDS fallback>")
    blob = "\n\n---\n\n".join(parts)
    return blob, used


def _needle_build_context(target_bytes, depth, code, blob):
    """Realistic filler truncated to ~target_bytes, with the needle inserted at
    `depth` of the filler and the question appended at the very end (so prefill
    processes the full context before the real turn)."""
    needle_line = (f"IMPORTANT NOTE: The project access code is {code}. "
                   f"Remember this.")
    fill_bytes = max(1, target_bytes - len(needle_line.encode())
                     - len(NEEDLE_QUESTION.encode()) - 64)
    enc = blob.encode("utf-8", "ignore")
    if not enc:
        filler_text = ""
    else:
        reps = (fill_bytes // len(enc)) + 1
        filler_text = (enc * reps)[:fill_bytes].decode("utf-8", "ignore")
    at = int(len(filler_text) * depth)
    return (filler_text[:at] + "\n\n" + needle_line + "\n\n"
            + filler_text[at:] + "\n\n" + NEEDLE_QUESTION)


def _calibrate_bpt(base, model, blob, log):
    """Bytes-per-token for the REALISTIC filler (markdown + python has a
    different bytes/token ratio than random WORDS). One tiny chat request on a
    ~12 KB sample so the length targets land near the requested prompt_tokens;
    the real usage.prompt_tokens is still recorded per request. Mirrors
    _calibrate_wpt from the derailment probe."""
    sample = blob[:12000]
    sample_bytes = len(sample.encode("utf-8", "ignore"))
    try:
        t0 = time.perf_counter()
        d = _post(base + "/v1/chat/completions", {
            "model": model,
            "messages": [{"role": "user", "content": sample}],
            "max_tokens": 1, "temperature": 0.0,
            "chat_template_kwargs": {"thinking": False}}, 300)
        dt = time.perf_counter() - t0
        pt = d["usage"]["prompt_tokens"]
        bpt = round(sample_bytes / max(1, pt), 4)
        log(f"[needle_sweep] calibration: {sample_bytes} bytes -> {pt} tok "
            f"=> {bpt} bytes/tok ({dt:.1f}s)")
        return {"sample_bytes": sample_bytes, "prompt_tokens": pt,
                "bytes_per_token": bpt, "wall_s": round(dt, 2)}
    except Exception as e:  # noqa: BLE001
        log(f"[needle_sweep] calibration failed ({e}); defaulting to 3.6 b/t")
        return {"sample_bytes": sample_bytes, "bytes_per_token": 3.6,
                "error": f"{type(e).__name__}: {e}"}


def _answer_maxprob(content, n=NEEDLE_ANSWER_TOKENS):
    """Mean of the per-token top-token softmax prob over the first n answer
    tokens (from chat logprobs.content). High = confident retrieval; a garbled
    answer depresses this too."""
    if not content:
        return None
    probs = []
    for tok in content[:n]:
        tl = tok.get("top_logprobs") or []
        mapping = {x.get("token", "?"): x.get("logprob", 0.0) for x in tl}
        s = _dist_stats(mapping)
        if s:
            probs.append(s[0])
    return round(sum(probs) / len(probs), 4) if probs else None


def _answer_garbled(answer, correct):
    """Non-code / degenerate answer that isn't a clean refusal or honest miss."""
    if correct or not answer.strip():
        return False
    a = answer.strip()
    if _degenerate(a):
        return True
    # clean refusal / honest miss -> not garbled
    if re.match(r"^(i\b|i'm|i’m|there\b|the\b|no\b|sorry|none\b|n/?a\b|"
                r"unable|cannot|can't|what\b|which\b)", a, re.I):
        return False
    # the code always contains a 4-digit run; no 3+ digit run => confabulation
    if not re.search(r"\d{3}", a) and len(a) >= 5:
        return True
    return False


def _needle_one(base, model, ctx, L, depth, run, code, max_tok, timeout, log):
    payload = {"model": model,
               "messages": [{"role": "user", "content": ctx}],
               "max_tokens": max_tok, "temperature": 0.0,
               "logprobs": True, "top_logprobs": NEEDLE_TOP_LOGPROBS,
               "chat_template_kwargs": {"thinking": False}}
    d, dt, err = _timed_post(base + "/v1/chat/completions", payload, timeout,
                             log, what=f" L={L} run={run}")
    rec = {"length": L, "target_length": L, "depth": depth, "run": run,
           "needle_code": code, "max_tokens": max_tok}
    if err:
        log(f"[needle_sweep] L={L:>6} run {run}: ERROR {err}")
        rec["error"] = err
        rec["correct"] = False
        rec["degenerate"] = None
        return rec
    ch = (d.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    answer = msg.get("content") or ""
    finish = ch.get("finish_reason")
    content = ((ch.get("logprobs") or {}).get("content")) or []
    usage = d.get("usage", {})
    pt = usage.get("prompt_tokens", 0)
    ct = usage.get("completion_tokens", 0)
    code_norm = re.sub(r"[^a-z0-9]", "", code.lower())
    ans_norm = re.sub(r"[^a-z0-9]", "", answer.lower())
    ok = (code.lower() in answer.lower()) or (bool(code_norm)
                                              and code_norm in ans_norm)
    repetitive = _degenerate(answer)
    garbled = _answer_garbled(answer, ok)
    maxprob = _answer_maxprob(content)
    rec.update({
        "prompt_tokens": pt, "out_tokens": ct, "stop_reason": finish,
        "correct": ok, "answer_head": answer[:80],
        "answer_token_maxprob": maxprob,
        "repetitive": repetitive, "garbled": garbled,
        "degenerate": repetitive or garbled,
        "wall_s": round(dt, 2),
        "prefill_tps": round(pt / dt, 0) if dt > 0 and pt else None,
    })
    log(f"[needle_sweep] L={L:>6} run {run}: pt={pt} "
        f"{'PASS' if ok else 'FAIL'} maxp={maxprob} "
        f"deg={rec['degenerate']} ({dt:.0f}s) head={answer[:40]!r}")
    return rec


def _needle_sweep_print(summary, log):
    log("")
    log("=== needle_sweep (mid-context retrieval, chat-only) ===")
    hdr = (f"{'length':>7} {'pt':>7} {'correct':>7} {'maxp':>6} "
           f"{'deg':>4} {'wall_s':>7}  answer_head")
    log(hdr)
    log("-" * 72)
    for L in summary["lengths"]:
        cases = [c for c in summary["per_request"]
                 if c.get("length") == L and "error" not in c]
        if not cases:
            err = [c for c in summary["per_request"] if c.get("length") == L]
            head = (err[0].get("error", "") if err else "")[:40]
            log(f"{L:>7} {'-':>7} {'ERR':>7} {'-':>6} {'-':>4} {'-':>7}  {head}")
            continue
        c = cases[0]
        correct = "OK" if c.get("correct") else "FAIL"
        mp = c.get("answer_token_maxprob")
        mp_s = f"{mp:.3f}" if isinstance(mp, float) else "-"
        deg = "Y" if c.get("degenerate") else "-"
        pt = c.get("prompt_tokens", "-")
        log(f"{L:>7} {pt:>7} {correct:>7} {mp_s:>6} {deg:>4} "
            f"{c.get('wall_s', '-'):>7}  {c.get('answer_head', '')[:40]!r}")
    log("")
    log(f"onset L* (first FAIL): {summary.get('onset_L')}")
    log(f"endpoint=/v1/chat/completions top_logprobs={NEEDLE_TOP_LOGPROBS} "
        f"max_tokens={summary['max_tokens']} depth={summary['depth']} "
        f"thinking=false")
    log(f"filler_source: {summary['filler_source']}")


def needle_sweep(base, model, lengths=None, runs=1, depth=0.5,
                 max_tokens=NEEDLE_MAX_TOKENS, timeout=900,
                 filler_files=None, log=print, **_):
    """Mid-context needle retrieval vs prompt length. Pure client-side against
    /v1/chat/completions. Returns per-request records, a per-length pass/fail
    table, the onset length L* (smallest length that FAILS), and the filler
    source. See the block docstring above for why realistic filler is required
    and why chat-only is the only path used."""
    lengths = list(lengths or NEEDLE_DEFAULT_LENGTHS)
    env_files = os.environ.get("NEEDLE_FILLER_FILES", "")
    files = ([f for f in filler_files.split(":") if f]
             if filler_files
             else ([f for f in env_files.split(":") if f] or list(NEEDLE_DEFAULT_FILLER_FILES)))
    blob, used = _needle_filler_blob(log, files)
    cal = _calibrate_bpt(base, model, blob, log)
    bpt = cal.get("bytes_per_token", 3.6) or 3.6
    raw = []
    for L in lengths:
        target_bytes = max(1, int(L * bpt))   # L is a TARGET prompt_tokens count
        for run in range(runs):
            code = _needle_code(L, run)
            ctx = _needle_build_context(target_bytes, depth, code, blob)
            raw.append(_needle_one(base, model, ctx, L, depth, run, code,
                                   max_tokens, timeout, log))
    ok_lens = {L for L in lengths
               if all((c.get("correct") and "error" not in c)
                      for c in raw if c.get("length") == L)
               and any(c.get("length") == L for c in raw)}
    fail_lens = {L for L in lengths if L not in ok_lens
                 and any(c.get("length") == L for c in raw)}
    onset = min(fail_lens) if fail_lens else None
    summary = {
        "per_request": raw, "lengths": lengths, "runs": runs, "depth": depth,
        "max_tokens": max_tokens, "timeout": timeout,
        "endpoint": "chat/completions",
        "top_logprobs": NEEDLE_TOP_LOGPROBS,
        "filler_source": used,
        "filler_base_bytes": len(blob.encode("utf-8", "ignore")),
        "calibration": cal,
        "length_units": "prompt_tokens (target; filler sized via bytes_per_token)",
        "onset_L": onset,
        "correct_lengths": sorted(ok_lens),
        "failed_lengths": sorted(fail_lens),
        "all_correct": not fail_lens,
    }
    _needle_sweep_print(summary, log)
    return summary


PROBES = {
    "decode": decode,
    "batch_decode": batch_decode,
    "prefill": prefill,
    "needle": needle,
    "arithmetic": arithmetic,
    "coherence": coherence,
    "quality": quality,
    "derailment": derailment,
    "needle_sweep": needle_sweep,
}


# --- server-log scraping ------------------------------------------------------

def scrape_server_log(log_path):
    """Spec-decode acceptance + cache-tier hit-rate lines from the serve log."""
    out = {"spec_acceptance_length": None, "draft_acceptance_pct": None,
           "pool_lines": [], "planes_cache_lines": []}
    try:
        with open(log_path, errors="ignore") as f:
            lines = f.readlines()
    except OSError:
        return out
    acc_len = acc_rate = None
    for ln in lines:
        if "Mean acceptance length:" in ln:
            m = re.search(r"Mean acceptance length:\s*([0-9.]+)", ln)
            acc_len = float(m.group(1)) if m else acc_len
        if "Avg Draft acceptance rate:" in ln:
            m = re.search(r"Avg Draft acceptance rate:\s*([0-9.]+)", ln)
            acc_rate = float(m.group(1)) if m else acc_rate
        if "SpecDecoding metrics" in ln:
            m = re.search(r"Mean acceptance length:\s*([0-9.]+)", ln)
            if m:
                acc_len = float(m.group(1))
        if "hit-rate" in ln:
            out["pool_lines"].append(ln.strip()[-300:])
        if "planes cache" in ln or "planes from cache" in ln:
            out["planes_cache_lines"].append(ln.strip()[-300:])
    out["spec_acceptance_length"] = acc_len
    out["draft_acceptance_pct"] = acc_rate
    out["pool_lines"] = out["pool_lines"][-8:]
    out["planes_cache_lines"] = out["planes_cache_lines"][-4:]
    return out


# --- standalone CLI (live-server probe) ---------------------------------------

def _main():
    """Run a standalone length-sweep probe against an already-running vLLM
    server (default the live moet container on :8001). No server boot, no
    docker, no restart. --probe selects between the two chat-only sweeps:

      derailment (DEFAULT — preserves the documented bare-invocation commands):
        python3 bench/runner/probes.py --lengths 1024,4096 --runs 1   # smoke
        python3 bench/runner/probes.py                                # full (<=131072)
        python3 bench/runner/probes.py --out /tmp/derailment_before.json
        python3 bench/runner/probes.py --lengths 1024,4096,8192,16384,32768,65536,131072,262144

      needle_sweep (mid-context retrieval; the FP8-delta before/after metric):
        python3 bench/runner/probes.py --probe needle_sweep --lengths 2048,16384 --runs 1  # smoke
        python3 bench/runner/probes.py --probe needle_sweep --out /tmp/needle_before.json  # baseline
        # opt-in the slow ~262K prefill:
        python3 bench/runner/probes.py --probe needle_sweep \
            --lengths 2048,16384,32768,65536,131072,262144
    """
    import argparse
    ap = argparse.ArgumentParser(description=_main.__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", default="derailment",
                    choices=["derailment", "needle_sweep"],
                    help="which length-sweep probe to run (default: derailment)")
    ap.add_argument("--endpoint", "--base", dest="base",
                    default=os.environ.get("DERAILMENT_BASE",
                                           "http://localhost:8001"))
    ap.add_argument("--model", default=os.environ.get("DERAILMENT_MODEL",
                                                      "deepseek-v4-flash"))
    ap.add_argument("--lengths", default=None,
                    help="comma list of target lengths. derailment: prompt-"
                         "token lengths (default caps at 131072; add 262144 "
                         "opt-in). needle_sweep: target bytes (default "
                         "2048..131072; add 262144 opt-in)")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=900,
                    help="per-request timeout (seconds; >=900 fits a 262K prefill)")
    ap.add_argument("--out", default=None,
                    help="write the full result JSON here (default: stdout only)")
    # derailment-only
    ap.add_argument("--modes", default="overgen,stop",
                    help="(derailment only) comma list of modes")
    # needle_sweep-only
    ap.add_argument("--depth", type=float, default=0.5,
                    help="(needle_sweep only) needle position 0..1 (default 0.5 = mid-context)")
    ap.add_argument("--max-tokens", dest="max_tokens", type=int,
                    default=NEEDLE_MAX_TOKENS,
                    help="(needle_sweep only) answer max_tokens (default 24)")
    ap.add_argument("--filler-files", dest="filler_files", default=None,
                    help="(needle_sweep only) colon-separated filler source files "
                         "(default: realistic repo text; also via NEEDLE_FILLER_FILES)")
    args = ap.parse_args()
    lengths = ([int(x) for x in args.lengths.split(",") if x.strip()]
               if args.lengths else None)
    if args.probe == "needle_sweep":
        res = needle_sweep(args.base, args.model,
                           lengths=lengths or NEEDLE_DEFAULT_LENGTHS,
                           runs=args.runs, depth=args.depth,
                           max_tokens=args.max_tokens, timeout=args.timeout,
                           filler_files=args.filler_files, log=print)
    else:
        res = derailment(args.base, args.model, lengths=lengths or DERAILMENT_LENGTHS,
                         runs=args.runs,
                         modes=tuple(m for m in args.modes.split(",") if m),
                         timeout=args.timeout, log=print)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1, ensure_ascii=False)
            f.write("\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    _main()
