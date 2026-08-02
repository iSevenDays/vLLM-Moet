# DeepSeek-V4-Flash on Ada sm_89 — long-context digit loss

**Single source of truth.** Last updated 2026-08-01. Everything for this
investigation lives here, in the repo whose code it describes.

| path | what it is |
|------|------------|
| **this file** | **Knowledge**: system, what is established, what is eliminated and why. Read §0, §2, §4 first. |
| [`PLAN.md`](PLAN.md) | **Execution**: ordered task queue T1–T8 with exact commands, expected output and decision tables. **If you are here to do work, start there.** |
| [`runlogs/`](runlogs/) | Raw artifacts. [`ALL_NEEDLE_RESULTS.csv`](runlogs/ALL_NEEDLE_RESULTS.csv) = all 60 needle measurements, machine-readable. |
| [`../../tools/README.md`](../../tools/README.md) | Test harnesses: what each proves, what it costs, how to run it. |

Nothing else. The former `STATUS.md` / `BRIEFING.md` / `HANDOFF.md` / archive are
**deleted** — fully absorbed here. If you need the original §5.1–5.19 narrative
for provenance, run `git log --follow --all -- 'docs/dsv4-sm89-longcontext/*'` and
read `git show 3678f3b00:docs/dsv4-sm89-longcontext/archive/STATUS.md`.

---

> ## ✅ RESOLVED 2026-08-02 — root cause found and fixed
>
> The long-context digit loss was **not** an architectural `index_topk` limit
> (§5/§7's pre-resolution framing, now refuted). It was two bugs in the sm89
> decode-indexer path. The bugs were ported from
> [`the-crypt-keeper/vLLM-sm89`](https://github.com/the-crypt-keeper/vLLM-sm89)
> (`sm89-ds4-work`) and re-validated on the RTX 4090 D:
>
> - **Bug #2 — long-context digit loss.** The paged indexer KV cache was read
>   **interleaved** (`[D+4]` per token) while `indexer_k_quant_and_cache_kernel`
>   writes it **segregated** (all k bytes, then all scale bytes, per block) →
>   garbage/NaN decode candidate scores past L=2048. The §5.19 rank trace looked
>   clean because it watched the *prefill* path (correct). The bug was in the
>   *decode* path, which §2 later showed was never instrumented. Fix: the three
>   readers → segregated + a real-writer self-test. Commit `93bd9b7d5`.
> - **Bug #1 — short-prompt repetition.** The radix decode top-k got an
>   uncompressed scan bound + an unclamped `k_select=512`. With <512 valid
>   candidates (absolute context <2048) it emitted NaN indices → token-salad
>   (e.g. a 15-token "capital of Germany?" returned word-salad repetition). Fix:
>   route decode to `top_k_per_row_decode` + compress the scan bound. Commit
>   `115b8b955`.
> - **Tokenizer.** The reasoning-effort ladder was broken (`high` was a silent
>   no-op, real `max` unreachable). Commit `02fe69a84`.
>
> **Validated:** `ask` @ 8K / 32K / 64K all PASS exact → `index_topk=512` is
> sufficient once the cache is read correctly. So the `=2048` workaround in §7
> is **obsolete**. Short-prompt repetition gone; paged-MQA layout self-test
> `3.642e-02` at boot. `docker/serve_sm89_ds4.sh` bind-mounts the six fixed
> overlay files by default (`MOUNT_LAYOUT_FIX=1`, fatal if missing) until the
> image is rebuilt.
>
> The retraction chain in §4 and the raw measurements in §3 are retained for
> provenance. §2's instrument correction (the trace watched the wrong column)
> stood. §5 H1/H2/H3 were superseded by the fix, not by their falsifiers.

---

## 0. Read this first

Two facts will cost you hours if you do not know them.

**(a) "Needle retrieval is broken above ~8K" is wrong.** The model recovers the
needle's leading component at every length tested (2K–48K) and loses the
**trailing** component:

```
true PELICAN-1605       -> "PELICAN-1234"   /  "PELICAN PELICAN PELICAN..."
true LANTERN-2037       -> "LANTERN-9-9-9-9-9..."
true GLACIER-7741-ORYX  -> "GLACIER-7741"  (depth 0.5)  /  "GLACIER"  (depth 0.1)
```

Retrieval locates the needle. Exact multi-token copy is what fails. Use
[`tools/needle_digits_probe.py`](../../tools/needle_digits_probe.py), which
reports `word_present` / `digits_present` separately. The old
`bench/runner/probes.py --probe needle_sweep` collapses both into one bit. That
old probe produced the original misdiagnosis.

**(b) Seven interpretations have been retracted here (§4). The raw
request/response measurements have never been wrong — every failure was in what
an instrument was believed to point at.** Before you trust any diagnostic,
verify its addressing on a known input. The prompts are deterministic. The
tokenizer runs on CPU in seconds. That check would have caught the worst error
(§2) immediately.

### Standing decisions

- **`/root/models/DeepSeek-V4-Flash-0731` is the only checkpoint.** The older
  `/root/models/DeepSeek-V4-Flash` was **deleted 2026-08-01** (operator-confirmed,
  146 GB reclaimed). It was verifiably the old one: 44 `compress_ratios` entries
  and **no** dspark keys, vs 0731's 46 entries plus `dspark_block_size 5`,
  `dspark_target_layer_ids [40,41,42]`, `dspark_markov_rank 256`. Any result
  quoted from it is not comparable. Re-measure or drop such results.
  Two stale `MODEL` defaults that pointed at it (`docker/serve_sm89_ds4.sh:142`,
  `start.sh:23`) were repointed to `-0731` before the delete. So nothing can
  resolve to a missing path.
- **~1 TB reclaimed 2026-08-01** (403 GB → 1.4 TB free). Beyond the checkpoint:
  its unreferenced conversions `-RAWINT4` / `-AMXINT4` / `-AMXINT4-NUMA1`
  (388 GB, 0 mounts / 0 refs), and the dead cache tree
  `moet-cache/{packs,planes,planes-qp}` (537 GB) — verifiably keyed to the
  deleted checkpoint (`ckpt_id cc19494e…`, `n_layers 44`, lossy `base` tag, vs
  the live exact `w8x` packs in `moet-cache-0731-exact`). **`moet-cache/jit` was
  preserved** — it is the shared compiler cache mounted as `/root/.cache`.
  Still present, 0 mounts, reclaim candidates if wanted: `moet-cache-0731`
  (53 GB), `moet-cache-0731-host` (133 GB), `DeepSeek-V4-Flash-0731-AMXINT4`
  (125 GB — AMXINT4 is a rejected direction, ~10 tok/s).
- `vllm-moet-sm89:v0251` is the canonical living tag. Preserve baselines as
  suffixed tags. Never default to a candidate tag.
- **Never `--enforce-eager`** on this rig.
- Production: container `moet-0731-dspark-exact`, port **8011**, sole container,
  indexer trace off by default.

---

## 1. System

| property | value |
|---|---|
| GPUs | 2 × RTX 4090 D, 48 GiB, **Ada sm_89**; Proxmox LXC, 630 GiB RAM, no swap |
| Model | DSv4-Flash-0731, 43 layers, 256 experts / 6 active, FP4 E2M1 experts, bf16 attention |
| Serving | local fork `vLLM-Moet`, image `vllm-moet-sm89:v0251`, TP=2 |
| Launch | `--kv-cache-dtype fp8 --block-size 256 --max-model-len 262144 --max-num-batched-tokens 1056 --max-num-seqs 3 --no-enable-prefix-caching`, DSpark k=5, graphs [1,2,4,6,8,12,18] |
| Attention | FlashInfer has **no sm_89 DSv4 sparse-MLA kernel** → decode *and* prefill route to a local Triton port (`triton_sparse_mla_dsv4.py`), boot self-test `worst_row_rel 6.6e-3` |
| Indexer | ragged/prefill logits fall back to a **torch reference** pre-SM90; paged/decode uses a local Triton port. FP8 indexer cache (MXFP4 is sm_10x-only). |

### Architecture, from the checkpoint's own reference

`/root/models/DeepSeek-V4-Flash-0731/inference/model.py` ships with the weights
and is authoritative — ~600 lines, read it directly.

- KV cache is a **ring buffer of only `window_size` tokens**
  (`kv_cache[:bsz, start_pos % win] = kv`), sized
  `window_size + max_seq_len // compress_ratio` (line 479). Older tokens are
  **never retained exactly** — only as learned gated-pooling compressed entries.
- Selection: `topk_idxs = index_score.topk(min(index_topk, end_pos // ratio))`,
  `index_score = (einsum(q, kv_cache[:, :end_pos//ratio]).relu_() * weights).sum(dim=2)`.
- An `Indexer` exists **only on `compress_ratio == 4` layers**. Ratio-128 layers
  use positional order (all causal entries) by design.
- Attend set = `cat([always_included_window_idxs, compress_topk_idxs])`, two
  segments sharing ONE softmax (line ~520).
- The indexer *simulates FP4* on q/kv (`fp4_act_quant(..., fp4_block_size=32)`).

Config: `index_topk 512`, `index_n_heads 64`, `index_head_dim 128`,
`sliding_window 128`, `rope_theta 10000`, `compress_rope_theta 160000`,
yarn factor 16 / `original_max_position_embeddings 65536`.
`compress_ratios` = 21 layers ratio 4, 20 ratio 128, 5 zeros; indexed
`compress_ratios[layer_id]` (`overlay/.../deepseek_v4/attention.py:201`).

### Serving state — throughput, prefill envelope, KV capacity

These results are settled and are **not** part of the open problem. They are
here so nobody re-measures them. All on the §1 production config.

| gate | result |
|---|---|
| single warmed request, 256 tok | median **~62 decode tok/s** (target ≥40 — passes decisively); both GPUs 98–99 % util |
| 3 × 256 concurrent, graph-18 | median **~33 tok/s/stream**, ~73 tok/s aggregate, 9/9 coherent, no assertion |
| DSpark acceptance | ~36–37 % of proposed tokens |
| first request after any restart | **always excluded** — it warms DSpark/Triton/Inductor caches and can launch `cc1plus`. It is a correctness gate, never a throughput sample |

**Prefill envelope ≈ 64–80K tokens.** Fresh-boot sweep: 16K/32K/48K/64K all pass;
80K OOMs **at the same `nvidia-smi` peak (48,508 MiB) where 64K passed**. So the
constraint is not total memory. It is a specific deep-context allocation plus
PyTorch cache fragmentation. The ceiling is therefore *allocator-state
dependent*, not a clean function of prompt length. A dirty allocator (after
several decode benchmarks) fails earlier than a fresh boot.

**KV capacity (boot log, authoritative).** `pool=4836` packed blocks,
`allocatable=4835` (one null block reserved), `max-request=1480`,
`per-group=(1024,20,20,267,149)`, GPU KV cache 856,573 tokens, max concurrency
**3.27×** at 262,144 tokens/request. The two compressor groups (267/149) are
sized *per prefill chunk plus the sliding window*. That sizing is what makes
chunked prefill safe — see §4's capacity row.

⚠️ **`--kv-cache-memory-bytes` BYPASSES `--gpu-memory-utilization`.** Boot log,
verbatim: *"reserved 4.51 GiB … skipped memory profiling. This does not respect
the gpu_memory_utilization config."* Lowering `UTIL` alone is a **no-op**.
Headroom must come from reducing `--kv-cache-memory-bytes` or
`--max-num-batched-tokens`.

---

## 2. THE CORRECTION: the §5.19 rank trace measured the wrong column

The now-deleted `STATUS.md` §5.19 / `BRIEFING.md` §5 presented an indexer rank
trace as "the single most informative measurement". **It was pointed at filler
prose.** (Recoverable via git; see the file table above.)

The trace ran with `VLLM_DSV4_INDEXER_TRACE_POS=4843`, taken from the probe's
`needle_abs_pos_est = int(prompt_tokens × depth)` = `int(9686 × 0.5)`
(`needle_digits_probe.py:232`) → ratio-4 column `4843 // 4 = 1210`.

The prompt is deterministic, so the true position is checkable on CPU in seconds.
Reconstructed byte-exact and tokenized with the real tokenizer:

| item | value |
|---|---|
| context | 29,433 bytes / 9,682 tokens (probe reports pt 9,686) |
| needle inserted at | byte 14,515 of 29,076 = **49.9 % by byte** |
| needle first token `'IM'` | abs **3,858** = **39.8 % by token** |
| `'160'`, `'5'` — **the digits** | abs 3,873–3,874 → ratio-4 column **968** |
| **column actually traced** | **1210** — abs 4,836–4,843 |

Token dump of the traced column: `[' in',' `','gen','/','`',' as',' the',' validated']`
— filler prose, **~243 columns from the needle**.

**Independently re-verified** with `tools/verify_needle_token_position.py`
(CPU, seconds — run this before you aim any position-addressed instrument):

| | measured | probe's estimate | off by |
|---|---:|---:|---:|
| digits token position | **3,871** | 4,841 | **+970 tokens** |
| ratio-4 column | **967** | 1210 | **243** |
| ratio-128 column | **30** | 37 | 7 |

⚠️ The exact ratio-4 column is **967–968**. It depends on whether you index the
`"1605"` string start or the `'160'` token, and on the chat template's prefix.
For a trace, that ±1 matters: **sweep a small column range, not a single value.**

**Root cause of the error:** the probe inserts the needle at a **byte** fraction
of the filler but reports its position as a **token** fraction. The filler's
first half (markdown + Python) tokenizes denser than its second half. So
byte-50 % lands at token-39.8 %.

**Invalidated — do not quote:** the rank table (112/592/523/1003/693/184/125/492);
"selected in roughly half the ratio-4 layers"; "rank varies 5.4 %→48 % across
layers" (that is the layer-to-layer spread of *a filler token*, unremarkable);
and the derived open question "why does the needle's entry rank vary?" — not a
real observation.

**Survives:** "the top-k boundary is exactly 512" (rank 492 selected, 523 not) —
independent of which column was watched, and separately corroborated by
`tools/test_indexer_topk_selection.py` and the paged suite. All 60 end-to-end
measurements (§3) — black-box request/response, unaffected. The
`index_topk=2048` result — unaffected.

**MEASURED (T4, 2026-08-01):** the digit-bearing entry **DOES survive the
k=512 cut at question-time** — selected in 20/21 ratio-4 layers (rank 0 in one)
on the FAILING `ask`; `digitsonly` (passing) is 21/21. So selection is
**EXONERATED (H2 confirmed)** and the §5.18 selection story is retracted. Full
data + the question-chunk isolation: [`runlogs/T4_analysis.md`](runlogs/T4_analysis.md).

### Two further instrument defects
1. `_trace_indexer_rank` was called **only from the prefill branch**, so
   selection during generation had never been observed (and the symptom is a
   generation failure). **Fixed 2026-08-01 (T3):** the trace is now labeled by
   **layer** (resolved `k_cache_prefix`), emitted from **TP rank 0 only** (both
   ranks ran it, so every line was doubled), sweeps a **±`SPAN`** window of
   ratio-4 columns (`VLLM_DSV4_INDEXER_TRACE_SPAN`, default 2 — the exact column
   is 967–968), and **also fires on the decode branch** after
   `persistent_topk`/`top_k_per_row_decode` (own cap
   `VLLM_DSV4_INDEXER_TRACE_DECODE_MAX`). The `sel` (selected) flag is the robust
   headline signal. *Observation* of decode selection is still pending — T4.
2. The archived `BRIEFING.md` §3b "needle absolute token pos" column
   (194/1695/2324/4843/9492) comes from the same estimate. `--abs-pos N` was
   converted to `depth = N/L` and applied as a byte fraction. The pass/fail
   pattern still refutes "fixed position" and "fixed length" (those need only
   *some* monotone axis). But **the axis values are wrong** — recompute before
   you fit any threshold. **FIXED 2026-08-01 (T1):** `needle_digits_probe.py`
   now reports the measured `needle_token_pos` (anchor = the digits, the column
   a trace watches) and `needle_token_frac`, plus `position_measured`. The
   unsound `needle_abs_pos_est = int(pt × depth)` field is removed. `--abs-pos N`
   now places by **token** (tokenize → insert → re-tokenize → verify ±2, recorded
   as `placed_ok`). Verified: `ask/8192/0.5` → digits @ token **3,871**, ratio-4
   col **967** (was 4,843 / 1,210). Item 3 below (recompute the §3b axis) is still
   open.

---

## 3. What is established

### 3a. Configuration sweep — plain question (`ask`, depth 0.5)

`pt` = actual prompt tokens; topk = `index_topk`; chunk = `max_num_batched_tokens`.

| pt | topk512 ch1056 | topk512 ch528 | topk2048 ch528 | INT8-NoPE | true code |
|---:|---|---|---|---|---|
| 2864 | PASS | PASS | PASS | PASS | TUNDRA-3711 |
| 5100 | FAIL `FALCON` | FAIL `FALCON` | **PASS** | FAIL | FALCON-1042 |
| 6650 | FAIL `7541` | **PASS** | **PASS** | FAIL `7542` | PUMICE-7544 |
| 8316 | **PASS** | FAIL `1206` | **PASS** | PASS | SAFFRON-7888 |
| 9376 | FAIL `0120` | FAIL `CYAN-42` | **PASS** | FAIL `1234` | CYPRESS-0074 |
| 9686 | FAIL `1234` | FAIL `1234` | **PASS** | FAIL `1234` | PELICAN-1605 |
| 13988 | — | — | **PASS** | — | LANTERN-7358 |
| 18559 | — | — | **PASS** | — | PELICAN-3738 |
| 35825 | — | — | FAIL `CYBERTRON` | — | CYPRESS-0527 |
| ~53000 | — | — | TIMEOUT (1800 s, no data) | — | — |

Two facts worth holding onto: **chunk size flips individual points in opposite
directions** (2/6 either way, a *different* 2) — chunk size should not change
scores, so a pure coverage story does not explain this. And **`index_topk=2048`
is the only change ever measured to convert failures into passes**.

### 3b. Question form, all at pt ≈ 9685, topk512 ch1056

| variant | answer | result |
|---|---|---|
| `ask` "What is the project access code?" | `PELICAN-1234` | FAIL |
| `digitsonly` "the four **digits** at the end" | `1605` | **PASS** |
| `digitspad` forced prefix, digits at answer pos ~6 | `THE FINAL FOUR DIGITS ARE 1605` | **PASS** |
| `firstword` | `pelican` | PASS |
| `yesno` / `yesno_neg` (1 token, true / wrong code) | `yes` / `no` | PASS — real discrimination |
| `spell` | `P E L I C A N \n P E L I C A N …` | FAIL |
| `words` (3 salient words, no digits) | `ZEPHYR-ORYX-CORYX` | FAIL (3rd blended) |
| `repeat` (code 3× at .35/.5/.65) | `PELICAN-305` | FAIL |
| `verbose` (digits also spelled out) | `PELICAN-1606` | FAIL (off by one) |

`digitsonly` passing is the key datum: **the digits are retrievable at 8K on the
stock config.** `digitspad` passing kills any "answer position" explanation.

### 3c. Logprobs at the digit position (raw)

```
pt 3909 PASS, position 5:  '538' logprob -0.000 (p~1.000); runners-up -14.4, -15.4
pt 9686 FAIL, position 5:  '123' logprob -0.765 (p~0.47); runners-up 301 -1.5,
                           927 -2.5, 627 -2.6, 294 -4.0 — true token '160' ABSENT
                           from the top-20
positions 0-4 in BOTH cases: all ~ -0.000
```

### 3d. Reference points

**llama.cpp passes the needle test on this same hardware** (operator-verified
2026-08-01) — **but on the OLD (superseded) checkpoint, not 0731; NOT
like-for-like** (T6, 2026-08-01). So the prior shift this datum motivated
("architectural limit of `index_topk=512`" → "port defect") is **not warranted**.
The sm_89-port-suspicion stories lose their strongest support. Original confounds
(now partly resolved):
1. **Which weights — RESOLVED (old checkpoint).** The GGUF llama.cpp used has
   `compress_ratios` length **44** and **no dspark keys** = the OLD checkpoint
   (0731 is 46 + dspark; file dated Jul 15, pre-0731). The 0731-derived
   `DeepSeek-V4-Flash-DSpark-support.gguf` (5.99 GB, has `dspark.*`) was **not**
   used. Not like-for-like. (T6; `local_blocks` also absent → T2 confirmed.)
2. **Which probe / length / depth.** 8K random-word depth 0.1 and 10.8K
   realistic-filler depth 0.5 are different claims. Record pt, filler, depth, reply.
3. **Which quantization.** That GGUF is IQ2-XXS experts + Q8 projections; ours is
   FP4 E2M1. A *pass* under IQ2 is evidence **against** an expert-precision story.

**Upstream (`kacper-daftcode/vLLM-Moet`, Blackwell)** records
`needle @121k tokens PASS` with the same `--kv-cache-dtype fp8`. But its recipe
is `needle: sizes_words: [8000, 90000]`, which drives the **easy random-word
probe at default depth 0.1** — not the realistic-filler needle. Running *that
same probe* on our production config: 8000 words → pt 10,815 → `GLACIER` (FAIL);
at depth 0.5 → `GLACIER-7741` (FAIL). Upstream contains **none** of the sm_89
Triton kernels, so it is a behavioural reference only. Note our launch also
differs on `max_num_batched_tokens` (1056 vs 4096), speculation (`dspark k=5` vs
`deepseek_mtp k=2`), `max-model-len` (262144 vs 131072) and util (0.98 vs 0.92) —
see H4.

⚠️ `bench/suites/needle_sweep.yaml` and the `needle_sweep` docstring claim
random-word filler "did NOT reproduce the failure". **That is wrong** at ≥10K —
it reproduces plainly, and more severely at depth 0.1 than 0.5.

---

## 4. Eliminated and retracted

### Verified clean (evidence attached, so it can be disputed)

| component | evidence |
|---|---|
| top-k selection kernel | k=512, all high scores beyond index 512 → `persistent_topk` returns **512/512** correct, zero overlap with first 512 (`tools/test_indexer_topk_selection.py`) |
| ragged indexer logits | sm_89 torch fallback = `sum_h relu(q·kᵀ)·w`, correct per-row k dequant and `[ks,ke)` −inf masking; matches checkpoint (`overlay/.../deep_gemm.py:588`) |
| paged indexer logits | own suite 9 passed / 5 skipped |
| compressor semantics | vs an **independent transcription of the checkpoint's `Compressor`**: ~2.6e-3 (bf16 floor) across the ratio-4 overlapping window, softmax axis, block-0 padding, APE indexing, RoPE position, ratio-128 branch (`tools/test_compressor_vs_checkpoint_ref.py`) — **within a single chunk only**, see H3 |
| compressor state capacity | chunk-aware, not window-sized: required 266/148 blocks, observed **267/149** at chunk 1056; 134/82 → **135/83** at chunk 528 |
| compressed-attn KV store/gather | 11 previously-blocked tests pass after the arch-gate fix; suite now 30 passed / 6 skipped / 0 failed |
| NoPE KV storage precision | E4M3 readback err 2.48e-2 vs INT8 8.0e-3, yet the INT8 A/B is **6/6 identical** → storage precision is not limiting |
| indexer key-quant granularity | per-row(D=128) vs checkpoint's block-32: recall@512 within ~1 %, top entry never dislodged |
| indexer RoPE | `attention.py:262` shares the attention rope; built with `compress_ratio` → `compress_rope_theta`, matching the reference |
| per-token scalars | top-k is per query **row**, so `q_scale`/`softmax_scale`/`head_scale` are ranking-neutral by construction |
| `sqrtsoftplus`, `noaux_tc`, `num_hash_layers`, `hc_*` | **all implemented and matching the checkpoint** — `overlay/.../fused_moe/router/fused_topk_bias_router.py:243`, `overlay/.../deepseek_v4/nvidia/model.py:599`, `overlay/.../nvidia/model.py:584`/`:716` (raises rather than degrading), Sinkhorn HC at `:866–979`. Also **mis-scoped**: the first two govern MoE expert routing, `hc_*` governs residual mixing; neither touches `index_score`. Remaining gap: the CUDA kernels are unverified numerically (task V4). |

### Retracted interpretations — do not revive without reading why

1. ~~FP8 KV cache is the cause~~ (inherited) — upstream passes with the same flag.
2. ~~Fails above a fixed absolute needle position (~2048)~~ — passes at est. 2721, fails at est. 1695. (Axis values also wrong, §2.)
3. ~~Fails above a fixed total length (~8192)~~ — non-monotonic: pt 8316 passes while 5100 and 6650 fail.
4. ~~"~26× indexer ranking-quality gap"~~ — **unsound arithmetic**: divided our hard-needle coverage requirement by upstream's *easier* probe's.
5. ~~The failure is decode-side, at the speculative block boundary~~ — `num_speculative_tokens=5` and the collapse at answer position 5 looked compelling; killed by `digitspad`.
6. ~~sm_89 produces bad indexer scores~~ — retired on the strength of the §5.19 trace, which is now void (§2), so this is **unsupported in both directions**. Kernel-level suites still pass.
7. ~~The §5.19 rank trace is the decisive measurement~~ — §2. Wrong column.

### A real bug found and fixed (unrelated to the symptom)
`has_cutedsl()` = `_has_module("cutlass")` — a **package** check with no device
check. The package is installed, so `dequantize_and_gather_k_cache` dispatched to
a CuTeDSL kernel with no pre-SM90 lowering. Now gated on
`has_device_capability(90)`. It had been hiding the only sm_89 validation of the
compressed-attention K-cache path.

---

## 5. Hypotheses, ranked by posterior × cheapness to falsify

**H1 — Selection policy differs from the reference: missing always-included
local/recent compressed blocks.** ~~Highest-value structural lead.~~
**REFUTED 2026-08-01 (T2, CPU).** All three implementations assemble the attend
set as `[sliding window] + [top-k compressed]`; none always-includes recent
compressed blocks. The trained checkpoint itself has no `local_blocks`
(`model.py:520` = `cat([window, topk])`; `Indexer` is pure top-k, `:433`); the
port matches it (`combine_topk_swa_indices` kernel writes `topk_len + swa_len`);
and `indexer.local_blocks` in llama.cpp **defaults to 0** and is consumed only by
`minimax-m3.cpp` — `dflash.cpp` does not reference it at all. The lead was an
artifact of a llama-arch key that exists for MiniMax-M3. Full side-by-side + the
reproduce commands: [`runlogs/T2_local_blocks_parity.md`](runlogs/T2_local_blocks_parity.md).

**H2 — The digit entry is in fact selected and the mechanism story is wrong.**
**CONFIRMED (T4, 2026-08-01).** At question-time the digit column is selected in
**20/21 ratio-4 layers** (rank 0 in one) on the FAILING `ask`; `digitsonly`
(passing) is 21/21 — i.e. `digitsonly` does **not** select the digit more.
Selection is exonerated; the value/decode path reopens. §5.18's "query content
raises the digit entry's score" is retracted (it was inferred from
`digitsonly`/`digitspad` passing, never measured; the one direct measurement
watched the wrong column). Details: [`runlogs/T4_analysis.md`](runlogs/T4_analysis.md).

**H3 — Chunked-prefill boundary corrupts the compressor's overlapping window.**
Chunk size flips individual points (§3a), which a pure coverage story cannot
explain — chunk size should not change scores. The ratio-4 compressor is
overlapping and carries `kv_state`/`score_state` across calls. The reference
builds them in one `start_pos == 0` pass, and our validation was single-chunk.
*Falsifier:* V2b, CPU, no boot.

**H4 — Config divergence from upstream's validated recipe, not sm_89.** Our
launch differs on chunk, speculation method/width, ctx and util (§3d); the sm_89
attribution holds none of them fixed. *Falsifier:* V5, one boot.

**H5 — `index_topk=512` is genuinely marginal for this task on any hardware.**
**Weakened at 8K (T4):** the digit column IS inside the k=512 cut at
question-time on the failing `ask` (20/21 layers), so plain coverage is not the
8K mechanism. NOT fully eliminated: `index_topk=2048` still rescues 6/6 to 18.5K
(§7) — so at *longer* contexts (where the digit likely ranks worse) coverage may
contribute. And 2048 may help at 8K via a non-coverage path (attention-weight
distribution / context entries). The 8K and >8K mechanisms may differ.

---

## 6. Verification queue — cheapest first

> **Superseded by [`PLAN.md`](PLAN.md)**, which specifies each of these as an
> executable task (T1–T8) with exact commands, expected output and decision
> tables. The summaries below are kept for rationale. **Follow PLAN.md for the
> order and the mechanics.**

**V0 — Measurement hygiene. Do this before any new sweep.** *(CPU, ~30 min)*
Otherwise every future position claim inherits the §2 error.
0. `tools/verify_needle_token_position.py` already exists and reports the
   measured position for any (length, variant, depth) — run it first. Note it
   needs `/root/autostart/CLAUDE.md` mounted to reproduce the filler blob exactly
   at prompt sizes above ~29 KB (below that the file falls beyond truncation and
   is irrelevant).
1. ~~`needle_digits_probe.py` must report the **measured** token position of the
   needle, not `int(pt × depth)`.~~ **DONE (T1, 2026-08-01).** Records now carry
   `needle_token_pos` / `needle_token_frac` (anchor = digits) + `position_measured`.
2. ~~`--abs-pos` must place by **token** position (tokenize, insert, verify), not
   by converting to a byte fraction.~~ **DONE (T1, 2026-08-01).** `--abs-pos N`
   binary-searches the byte offset that puts the anchor at token N and records
   `placed_ok` (within ±2). It needs the model tokenizer on the host (`pip install
   tokenizers`; loads `tokenizer.json`, byte-identical to AutoTokenizer
   `add_special_tokens=False`). Without it, positions are `UNMEASURED` and
   `--abs-pos` errors loudly.
3. Recompute the archived §3b position axis and restate the two refutations
   against it.

**V1 — Retrace with the correct column.** *(one boot, ~15 min; decisive)*
The measurement §5.19 was meant to be. Re-derive the digit token position first
(CPU, seconds — the prompt is deterministic), then:
```bash
-e VLLM_DSV4_INDEXER_TRACE=1 -e VLLM_DSV4_INDEXER_TRACE_POS=3871 \   # verified; sweep +/-8 tokens
-e VLLM_DSV4_INDEXER_TRACE_MIN_N=2000 -e VLLM_DSV4_INDEXER_TRACE_MAX=64
python3 tools/needle_digits_probe.py --lengths 8192 --variant ask        # FAILS
python3 tools/needle_digits_probe.py --lengths 8192 --variant digitsonly # PASSES
```
Trace **both** variants — §5.18's mechanism is now a directly testable
prediction. Prerequisites, both required for the output to be readable:
label emissions **by layer id** (currently counter-labeled,
`sparse_attn_indexer.py:106`) and **gate on TP rank 0** (both ranks emit today).
*Outcomes:* digit column outside the cut on `ask` / inside on `digitsonly` →
§5.18 confirmed, fix is coverage. Digit column **inside** the cut on failing
`ask` → **H2**, selection exonerated, reopen the decode/value path.

**V2 — Trace the decode path.** *(code change, fold into V1's boot)*
Add `_trace_indexer_rank` to the decode branch after
`persistent_topk`/`top_k_per_row_decode`. Generation-time selection has never
been observed and the symptom is a generation failure.

**V2b — Cross-chunk compressor parity.** *(CPU, no boot)*
Extend `tools/test_compressor_vs_checkpoint_ref.py` to drive the port's
compressor in N chunks and compare against the reference's single-pass output at
the boundary columns. Tests H3.

**V3 — `indexer.local_blocks` parity.** *(CPU, ~1 h reading)*
Compare attend-set assembly across three implementations: checkpoint
`model.py:513–520`, this port (`sparse_attn_indexer.py` + the sparse-MLA
backend's index handling), and llama.cpp `src/models/dflash.cpp` +
`LLM_KV_ATTENTION_INDEXER_LOCAL_BLOCKS`. One question: **does any of them
always-include recent compressed blocks that the port drops?** Yes → H1
confirmed, fix is a selection-policy patch and `index_topk` can return to 512.

**V4 — Numerically verify the CUDA routing kernels.** *(unit, minutes)*
Production runs `ops.topk_hash_softplus_sqrt`; the torch fallbacks are XPU/CPU
only. Assert kernel ≡ torch fallback ≡ checkpoint `Gate.forward`, including the
DSv4-Flash bias regime (all ≈ 8.08). Low prior for this symptom but closes §4's
last gap. Add an assert on the `input_tokens is not None` fallback while there.

**V5 — Upstream recipe on sm_89.** *(one boot + ~2 min)*
Run `pro6000x2-tp2.yaml`'s serve args verbatim (chunk 4096, `deepseek_mtp` k=2,
ctx 131072, util 0.92) and upstream's own cheap gate
`python3 tools/needle_probe.py 8011 8000 0.1` (~65 s). Pass → the gap is
configuration (H4), not architecture. Note the `moe_w2` strict guard may trip
when speculation changes: vary `num_speculative_tokens` rather than removing
speculation; **never relax the guard**.

**V6 — Close the llama.cpp confounds.** *(minutes)* §3d's three items. This is
the cheapest way to firm up the single most informative new datum. Record in
`runlogs/`.

---

## 7. Fix strategy

> ✅ **RESOLVED 2026-08-02 — see the banner at the top of this file.** The cause
> was the decode-indexer bugs (layout + top-k), not coverage. The fix is in, so
> `index_topk=512` is sufficient (64K retrieval passes). The `=2048` interim
> workaround below is **obsolete**. This section is retained as the
> pre-resolution strategy of record.

**Now — interim production setting.** `index_topk=2048` is the only change
measured to convert failures into passes (6/6 exact to 18.5K):
```bash
--hf-overrides '{"index_topk": 2048}'
```
Costs compute, not VRAM. Ship it **if** long-context exactness matters more than
throughput, and record the throughput delta. Caveats that must stay attached: its
mechanism is now unconfirmed (§2); it fails at 35.8K; it deviates from the
trained 512 so it needs GSM8K/GPQA validation; and the `reliable ≈ 10 ×
index_topk` extrapolation implies ~26,000 at 262K — effectively dense, not a
principled setting. **`index_topk = 0` does not mean unlimited** — it selects
nothing and leaves only the 128-token window.

**Next — fix the cause.** Follow whichever of H1/H2/H3 the queue confirms:
H1 → patch attend-set assembly to always-include local blocks (likely small, and
`index_topk` returns to 512). H2 → selection exonerated, reopen the decode/value
path and re-examine `digitspad` under the corrected instrument. H3 → fix the
compressor's cross-chunk state, which would also explain the chunk-size
sensitivity. H4 → adopt upstream's serve args, no kernel work.

**Do not:** quote the §5.19 rank table, the "5 %→48 % spread", or the "~26×
gap"; run another end-to-end sweep before V0; relax the `moe_w2` strict
miss-replay guard to unblock a DSpark A/B; use `--enforce-eager`.

---

## 8. Reproduction

```bash
docker start moet-0731-dspark-exact          # sole container, 6-10 min boot
until curl -fsS http://127.0.0.1:8011/health >/dev/null; do sleep 20; done

python3 tools/needle_digits_probe.py --lengths 8192 --variant ask         # FAIL
python3 tools/needle_digits_probe.py --lengths 8192 --variant digitsonly  # PASS
python3 tools/needle_probe.py 8011 8000 0.1                              # upstream's probe, FAIL
```
CPU harnesses run in seconds and are safe while serving — see
[`tools/README.md`](../../tools/README.md). Costs: boot 6–10 min; 8K needle point
~100 s; 32K ~650 s.

⚠️ `EXTRA_MOUNTS` is **not** a knob in `docker/serve_sm89_ds4.sh` — passing it is
silently ignored and your modified file never reaches the container. Smuggle
mounts through `EXTRA_DOCKER_ENV`, which is spliced raw into `docker run`.

## 9. Primary sources

- `/root/models/DeepSeek-V4-Flash-0731/inference/model.py` — **the authority.**
  `Compressor` (284), `Indexer` (~393), `Attention.forward` (~490), attend-set
  assembly (513–520), `Gate.forward` (576).
- `inference/kernel.py` — `act_quant`, `fp4_act_quant`.
- `overlay/vllm/vllm/model_executor/layers/sparse_attn_indexer.py` — selection.
- `overlay/vllm/vllm/v1/attention/ops/triton_sparse_mla_dsv4.py` — sm_89
  attention port (packed layout contract in the docstring).
- `overlay/vllm/vllm/models/deepseek_v4/compressor.py` +
  `common/ops/fused_compress_quant_cache.py`.
- `~/llama.cpp` — passes the needle test; `src/models/dflash.cpp`, indexer params
  in `src/llama-arch.cpp:255–262`.
