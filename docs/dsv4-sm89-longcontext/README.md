# DeepSeek-V4-Flash on Ada sm_89 — long-context investigation

Canonical home for the DSv4-Flash-0731 / 2×RTX 4090 D work. Everything that was
previously scattered under `/root/ktransformers` now lives here, next to the code
it describes.

| file | what it is |
|------|------------|
| [`STATUS.md`](STATUS.md) | **Authoritative status, measurements, decisions, NEXT queue.** Start here. |
| [`HANDOFF.md`](HANDOFF.md) | Raw state + primary sources for a fresh agent. Read after STATUS. |
| [`runlogs/`](runlogs/) | Raw artifacts backing every claim in STATUS. |
| [`../../tools/README.md`](../../tools/README.md) | The test harnesses, what each one proves, and its runtime. |

## The one thing to read before touching this

**"Needle retrieval is broken above ~8K" is WRONG, and believing it costs hours.**
The model recovers the needle's **word** at every length tested (2K–48K); only the
numeric **digits** are lost:

```
true PELICAN-1605  ->  "PELICAN PELICAN PELICAN..."
true LANTERN-2037  ->  "LANTERN-9-9-9-9-9..."
```

Retrieval works. The attended value's fine detail is unreadable. Those are
different bugs with different fixes. Use
[`tools/needle_digits_probe.py`](../../tools/needle_digits_probe.py), which
reports `word_present` / `digits_present` separately — the old
`bench/runner/probes.py --probe needle_sweep` metric collapses both into one
bit and is what produced the misdiagnosis.

## Root cause found (2026-08-01): `index_topk=512` selection is too marginal

`--hf-overrides '{"index_topk":2048}'` takes the needle battery from **2/6 to
6/6, all digits exact**, including the pt 9686 case that failed under every
other configuration (STATUS §5.15).

Top-512 selection over the ratio-4 compressed entries was operating near ties,
so the digit-bearing entry frequently fell outside the selected set. That
explains every earlier observation:

- the **word** always survived — the 20 `compress_ratio=128` layers see the whole
  context (~75 entries at 9.7K, far below `index_topk`) and never select at all;
- digits failed **non-monotonically** in length and were deterministic per input
  — whether a given needle's entry makes the cut depends on its score
  distribution, not on length;
- halving `--max-num-batched-tokens` reshuffled *which* cases passed without
  changing the count (2/6 either way, a different 2) — a marginal selection is
  perturbed by any numerically-transparent change (STATUS §5.14);
- better **storage** precision could not help (`VLLM_DSV4_KV_INT8=1`: 6/6
  identical) because the entry was never attended in the first place.

### …but `index_topk` is a diagnostic, not the fix

Extending to longer contexts finds the ceiling, and one variable explains every
result: **selection coverage** = `index_topk / (prompt_tokens / 4)`. PASS while
coverage ≳44%, FAIL at ≤40%, coin-flip between. Rule of thumb: **reliable
context ≈ 10 × index_topk** (512 → ~5K, 2048 → ~20K; 35825 tokens at topk 2048
= 23% coverage, and it FAILS).

**The regression is real, but quote it qualitatively — not as a ratio.** An
earlier "~26× ranking-quality gap" figure was withdrawn (STATUS §5.16): it
divided our hard-needle coverage requirement by upstream's, and upstream's
recipe (`needle: sizes_words: [8000, 90000]`) runs a *different, easier* probe
(random-word filler, default depth 0.1). Like-for-like, on upstream's own probe
and settings, is still damning: at 10,815 tokens we return `GLACIER` for secret
`GLACIER-7741-ORYX`, while upstream records PASS for both 8000 and 90000 words
(≈121K tokens).

Use that easy needle as the regression gate — it fails in **~65 s** versus
100–650 s per hard-needle point, and it is exactly what upstream validates:

```bash
python3 tools/needle_probe.py 8011 8000 0.1     # FAIL today: "GLACIER"
```

⚠️ `bench/suites/needle_sweep.yaml` and the `needle_sweep` docstring claim
random-word filler "did NOT reproduce the failure". **That is wrong** at this
length — it reproduces plainly, and more severely at depth 0.1 than 0.5.

Note the failure shape is filler-independent: the needle's leading component
survives and the **tail** is lost (`GLACIER-7741-ORYX` → `GLACIER-7741` →
`GLACIER`). Whatever the mechanism, it truncates verbatim copy length rather
than failing to locate the needle.

So:
- `index_topk=2048` is a usable mitigation **up to ~20K context** and must still
  be quality-validated (GSM8K/GPQA) since it deviates from the trained 512.
- It **cannot** reach 262K — that needs ~26,000, i.e. ~40% of all entries, which
  is nearly dense attention and defeats the sparse design.
- **The real fix is the sm89 indexer's score quality.** Everything downstream is
  verified, so the defect is in the scores or their inputs. Indexer RoPE is
  already eliminated; remaining suspects and their cheap checks are in
  STATUS §5.15.

## Eliminated — do not re-litigate (each by measurement, not argument)

| cause | how it was ruled out |
|-------|----------------------|
| FP8 KV dtype | upstream passes `needle @121k` with the same flag on native Blackwell kernels |
| NoPE storage precision | `VLLM_DSV4_KV_INT8=1` A/B: **6/6 identical** despite 3–4× better readback |
| top-k selection kernels | `tools/test_indexer_topk_selection.py`: 512/512 correct with all high scores beyond index 512 |
| indexer scoring (ragged + paged) | torch fallback matches the checkpoint reference; paged Triton suite passes |
| per-token scalars | top-k is per query **row**, so `q_scale`/`softmax_scale`/`head_scale` are ranking-neutral |
| compressor semantics | `tools/test_compressor_vs_checkpoint_ref.py`: agrees with an independent checkpoint transcription at ~2.6e-3 (bf16 floor) |
| compressed-attention KV store/gather | 11 previously-blocked tests now pass after the arch-gate fix |
| intra-chunk state overwrite | `tools/test_compressor_state_capacity.py`: allocation is chunk-aware (267/149 blocks vs 266/148 required) |
| fixed needle position (~2048) | passes at abs 2721, fails at abs 1695 |
| fixed total-length cutoff (~8192) | non-monotonic: pt 8316 passes, 5100/6650 fail |

## Ground truth — prefer these over any doc here

1. **`/root/models/DeepSeek-V4-Flash-0731/inference/model.py`** — DeepSeek's own
   implementation for this exact checkpoint. Authoritative for the ring-buffer
   SWA cache, the compressor, and the indexer (`Indexer` exists **only** on
   `compress_ratio == 4` layers; ratio-128 layers use positional order).
2. **`~/llama.cpp`** — a DFlash+DSpark build that passes the needle test
   (`src/models/dflash.cpp`).
3. **`upstream/main`** (`kacper-daftcode/vLLM-Moet`) — passes needle @121K on
   Blackwell. Note it contains **none** of the sm_89 Triton kernels, so it is a
   behavioural reference, not a code-diff target.

## Working rule

Deploys and end-to-end probes are expensive (6–10 min boots, 30–190 s per
probe); unit tests are cheap (seconds, often CPU-only). Every harness in
[`tools/README.md`](../../tools/README.md) was written to replace a server round
trip. Add to them before reaching for another boot.
