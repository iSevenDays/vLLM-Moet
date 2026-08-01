> # ⚠️ SUPERSEDED — DO NOT QUOTE
> Consolidated into [`../README.md`](../README.md) (2026-08-01), which is now the
> single source of truth. This file is kept for provenance only and contains
> **withdrawn claims** — notably the §5.19 indexer rank trace, which measured the
> wrong column (README §2). Verify anything here against README before using it.

# BRIEFING for an external analyst — raw data first

You are being handed this because the investigating agent's **interpretations have
been wrong three times**, each caught only by a control built to falsify them.
The raw measurements are reliable; the conclusions drawn from them repeatedly
were not. **Form your own conclusions. Where this file states an interpretation
it is labelled as such, and you should treat it as a hypothesis to test, not a
finding.**

Nothing here needs you to edit files. Everything is observation, configuration,
or a pointer to a primary source you can read.

---

## 1. The system, factually

| property | value |
|---|---|
| GPUs | 2 × RTX 4090 D, 48 GiB each, **Ada sm_89** |
| Host | Proxmox LXC, 630 GiB RAM, no swap |
| Model | DeepSeek-V4-Flash-0731, `/root/models/DeepSeek-V4-Flash-0731` (~146 GiB) |
| Arch | `deepseek_v4`, 43 layers, 256 routed experts / 6 active, FP4 E2M1 expert weights, bf16 attention |
| Serving | locally-built vLLM fork `vLLM-Moet`, image `vllm-moet-sm89:v0251`, TP=2 |
| Launch | `--kv-cache-dtype fp8 --block-size 256 --max-model-len 262144 --max-num-batched-tokens 1056 --max-num-seqs 3 --no-enable-prefix-caching`, DSpark spec-decode k=5, CUDA graph sizes [1,2,4,6,8,12,18] |
| Attention path | FlashInfer has **no sm_89 kernel** for DSv4 sparse MLA, so decode AND prefill route to a **local Triton port** (`triton_sparse_mla_dsv4.py`). Boot self-test `worst_row_rel=6.6e-3` vs its torch reference. |
| Indexer scoring | `fp8_fp4_mqa_logits` (ragged/prefill) falls back to a **torch reference** on pre-SM90; paged/decode uses a local Triton port |
| Indexer cache | FP8 (`use_fp4_indexer_cache=False`; MXFP4 is sm_10x-only) |

### Architecture facts from the checkpoint's OWN reference

`/root/models/DeepSeek-V4-Flash-0731/inference/model.py` ships with the weights
and is authoritative. **Read it directly — it is ~600 lines and settles most
questions.** Key mechanics:

- The KV cache is a **ring buffer of only `window_size` tokens**:
  `self.kv_cache[:bsz, start_pos % win] = kv`, sized
  `kv_cache_size = window_size + max_seq_len // compress_ratio` (line 479).
  Tokens older than the window are **never retained exactly** — only as
  learned gated-pooling compressed entries.
- Selection: `topk_idxs = index_score.topk(min(index_topk, end_pos // ratio))`,
  where `index_score = (einsum(q, kv_cache[:, :end_pos // ratio]).relu_() *
  weights.unsqueeze(-1)).sum(dim=2)`.
- An `Indexer` exists **only on `compress_ratio == 4` layers**; ratio-128 layers
  use positional order (all causal entries) by design.
- The final attend set is `cat([always_included_window_idxs, compress_topk_idxs])`,
  two segments sharing ONE softmax.
- The indexer *simulates FP4* on its q/kv (`fp4_act_quant(..., fp4_block_size=32)`).

### Config values (verbatim)

```
index_topk 512 | index_n_heads 64 | index_head_dim 128 | sliding_window 128
num_hidden_layers 43 | num_nextn_predict_layers 1
rope_theta 10000 | compress_rope_theta 160000
rope_scaling: yarn, factor 16, original_max_position_embeddings 65536
scoring_func sqrtsoftplus | topk_method noaux_tc
num_hash_layers 3 | hc_eps 1e-06 | hc_mult 4 | hc_sinkhorn_iters 20
dspark_block_size 5 | dspark_target_layer_ids [40,41,42] | dspark_markov_rank 256
dspark_noise_token_id 128799
compress_ratios (46 entries for 43 layers):
[0,0,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,
 4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,0,0,0]
```
Counts: 21 layers ratio 4, 20 layers ratio 128, 5 zeros. Code indexes it as
`compress_ratios[layer_id]` (`overlay/.../deepseek_v4/attention.py:201`).
**Never investigated by the agent:** `num_hash_layers`, `hc_*` (sinkhorn),
`scoring_func=sqrtsoftplus`, `topk_method=noaux_tc`.

---

## 2. The symptom, stated without interpretation

A needle (a code like `PELICAN-1605`) is placed mid-context in realistic filler;
the model is asked to return it. **The model returns the needle's leading
component and loses the trailing component.**

```
true PELICAN-1605       -> "PELICAN-1234"  /  "PELICAN PELICAN PELICAN..."
true LANTERN-2037       -> "LANTERN-9-9-9-9-9..."
true CYPRESS-0527       -> "CYPHER"
true GLACIER-7741-ORYX  -> "GLACIER-7741"   (depth 0.5)
                        -> "GLACIER"        (depth 0.1)
true ZEPHYR-ORYX-CYPRESS-> "ZEPHYR-ORYX-CORYX"
```

All HTTP 200. Deterministic: the same request repeated gives byte-identical
output.

---

## 3. RAW DATA — all of it

| artifact | contents |
|---|---|
| `runlogs/ALL_NEEDLE_RESULTS.csv` | **60 needle measurements** across 6 configurations, one row each: config, target/actual prompt tokens, variant, depth, true code, answer, correct/word_present/digits_present, wall time |
| `runlogs/needle_*.json` | the same runs with **full top-20 logprobs at every answer token position** — the raw distributions, not summaries |
| `runlogs/indexer_rank_trace_20260801.log` | live indexer score-RANK trace (see §5) |
| `runlogs/*.log` | console transcripts |

### 3a. Configuration sweep, plain question (`ask`, depth 0.5)

`pt` = actual prompt tokens. topk = `index_topk`. chunk = `max_num_batched_tokens`.

| pt | topk512 chunk1056 | topk512 chunk528 | topk2048 chunk528 | INT8-NoPE topk512 | true code |
|---:|---|---|---|---|---|
| 2864 | PASS | PASS | PASS | PASS | TUNDRA-3711 |
| 5100 | FAIL `FALCON` | FAIL `FALCON` | **PASS** | FAIL `FALCON` | FALCON-1042 |
| 6650 | FAIL `7541` | **PASS** | **PASS** | FAIL `7542` | PUMICE-7544 |
| 8316 | **PASS** | FAIL `1206` | **PASS** | PASS | SAFFRON-7888 |
| 9376 | FAIL `0120` | FAIL `CYAN-42` | **PASS** | FAIL `1234` | CYPRESS-0074 |
| 9686 | FAIL `1234` | FAIL `1234` | **PASS** | FAIL `1234` | PELICAN-1605 |
| 13988 | — | — | **PASS** | — | LANTERN-7358 |
| 18559 | — | — | **PASS** | — | PELICAN-3738 |
| 35825 | — | — | FAIL `CYBERTRON` | — | CYPRESS-0527 |
| ~53000 | — | — | TIMEOUT (1800 s, no data) | — | — |

### 3b. Needle position, at fixed total length

| total pt | needle absolute token pos | result |
|---:|---:|---|
| 9686 | 194 | PASS |
| 2864 | 1146 / 1432 / 1662 / 1890 / 2148 / 2434 / 2721 | PASS (7/7) |
| 3909 | 1955 | PASS |
| 9685 | 1695 | FAIL |
| 9685 | 2324 | FAIL |
| 9686 | 4843 | FAIL |
| 9686 | 9492 | FAIL |

### 3c. Question-form variants, ALL at pt ≈ 9685, topk512 chunk1056

| variant | question | answer | result |
|---|---|---|---|
| `ask` | "What is the project access code?" | `PELICAN-1234` | FAIL |
| `digitsonly` | "the four **digits** at the end" | `1605` | **PASS** |
| `digitspad` | forced prefix then digits (digits at answer pos ~6) | `THE FINAL FOUR DIGITS ARE 1605` | **PASS** |
| `firstword` | "the word at the start" | `pelican` | PASS |
| `yesno` | "Is the code PELICAN-1605?" | `yes` | PASS |
| `yesno_neg` | "Is the code PELICAN-5926?" (wrong) | `no` | PASS (real discrimination) |
| `spell` | "spell one character at a time" | `P E L I C A N \n P E L I C A N ...` | FAIL |
| `words` | code = 3 salient WORDS, no digits | `ZEPHYR-ORYX-CORYX` | FAIL (3rd word blended) |
| `repeat` | code appears 3× at depths .35/.5/.65 | `PELICAN-305` | FAIL |
| `verbose` | digits also spelled out in words | `PELICAN-1606` | FAIL (off by one) |

### 3d. Logprobs at the digit position (raw)

```
pt 3909 PASS, position 5:  emitted '538' logprob -0.000 (p~1.000)
                           runners-up: 539 -14.375, 53 -15.375, 584 -16.125
pt 9686 FAIL, position 5:  emitted '123' logprob -0.765 (p~0.47)
                           runners-up: 301 -1.515, 927 -2.515, 627 -2.640,
                                       294 -4.015, <EOS> -4.515, 901 -4.640
                           the true token '160' is ABSENT from the top-20
positions 0-4 in BOTH cases: all at logprob ~ -0.000
```

### 3e. Upstream comparison (`kacper-daftcode/vLLM-Moet`, Blackwell)

Recipe `bench/recipes/deepseek-v4-flash/pro6000x2-tp2.yaml`:
```
# Quality-validated 2026-07-13: GPQA Diamond 72.2%; GSM8K-200
# (paired McNemar p=1, flips 0<->1), needle @121k tokens PASS.
  - --kv-cache-dtype fp8
  needle:
    sizes_words: [8000, 90000]
```
`sizes_words` drives the OLD `needle` probe = **random-word filler, default
depth 0.1**, secret `GLACIER-7741-ORYX`. Upstream contains **none** of the sm_89
Triton kernels (they are 100% local to this fork), so it is a behavioural
reference only. We are 127 commits ahead / 11 behind; the 11 are docs/recipes.

Running that **same probe, same secret, same depth** on our production config:

| words | pt | reply | verdict |
|---:|---:|---|---|
| 8000 | 10815 | `GLACIER` | FAIL |
| 8000 (depth 0.5) | 10815 | `GLACIER-7741` | FAIL |

Upstream records PASS at both 8000 and 90000 words (≈121K tokens).

---

## 4. Verified-clean components (with the evidence, so you can dispute it)

| component | evidence | where |
|---|---|---|
| top-k selection kernel | k=512, all high scores placed beyond index 512 → `persistent_topk` returns **512/512** correct, zero overlap with first 512 | `tools/test_indexer_topk_selection.py` |
| top-k, again, live | trace shows rank 492 → selected, rank 523 → NOT selected; boundary exactly 512 | §5 |
| ragged indexer logits | sm_89 torch fallback implements `sum_h relu(q·kᵀ)·w` with correct per-row k dequant and `[ks,ke)` −inf masking; matches checkpoint formula | `overlay/.../utils/deep_gemm.py:588` |
| paged indexer logits | own suite 9 passed / 5 skipped | `overlay/vllm/tests/kernels/attention/test_triton_paged_mqa_logits_dsv4.py` |
| compressor semantics | vs an **independent transcription of the checkpoint's `Compressor`**: agrees ~2.6e-3 (shared bf16 store rounding floor) across the ratio-4 overlapping window, softmax axis, block-0 −inf/0 padding, APE indexing, RoPE position, and the ratio-128 branch | `tools/test_compressor_vs_checkpoint_ref.py` |
| compressor state capacity | allocation is chunk-aware, not window-sized: required 266/148 blocks, observed **267/149** at chunk 1056; 134/82 → **135/83** at chunk 528 | `tools/test_compressor_state_capacity.py` |
| compressed-attn KV store/gather | 11 previously-blocked tests pass after an arch-gate fix; suite now 30 passed / 6 skipped / 0 failed on sm_89 | `vllm/tests/kernels/test_compressor_kv_cache.py` |
| NoPE KV storage precision | E4M3 readback err 2.48e-2 vs INT8 8.0e-3 (flat in key count), yet the INT8 A/B is **6/6 identical** → storage precision is not limiting | `tools/test_nope_needle_recall.py` |
| indexer key-quant granularity | per-row(D=128) vs checkpoint's block-32: recall@512 within ~1%, top entry never dislodged | `tools/test_indexer_score_ranking.py` |
| indexer RoPE | `attention.py:262` shares the attention rope with the indexer; that rope is built with `compress_ratio` → `compress_rope_theta`, matching the reference | source read |
| per-token scalars | top-k is per query ROW, so `q_scale`/`softmax_scale`/`head_scale` are ranking-neutral by construction | reasoning |

### A real bug found and fixed (unrelated to the symptom)
`has_cutedsl()` = `_has_module("cutlass")` — a **package** check with no device
check. The package is installed, so `dequantize_and_gather_k_cache` dispatched to
a CuTeDSL kernel with no pre-SM90 lowering. Now gated on
`has_device_capability(90)`. This had been hiding the only sm_89 validation of
the compressed-attention K-cache path.

---

## 5. The single most informative measurement

Live trace of the needle's compressed-entry score RANK, on the FAILING `ask`
request (pt 9686, needle abs token 4843 → ratio-4 column 1210, 2088 candidates,
k=512, production config). Each line is one traced ratio-4 layer:

| rank / 2088 | pct | selected |
|---:|---:|:---:|
| 112 | 5.4% | **True** |
| 592 | 28.4% | False |
| 523 | 25.0% | False |
| 1003 | 48.0% | False |
| 693 | 33.2% | False |
| 184 | 8.8% | **True** |
| 125 | 6.0% | **True** |
| 492 | 23.6% | **True** |

The needle's entry is selected in roughly **half** the traced ratio-4 layers; its
rank varies **5.4% → 48%** across layers while the cut sits at 512/2088 = 24.5%.

Reproduce (opt-in, zero cost when off):
```
VLLM_DSV4_INDEXER_TRACE=1  VLLM_DSV4_INDEXER_TRACE_POS=<abs needle token>
VLLM_DSV4_INDEXER_TRACE_MIN_N=2000  VLLM_DSV4_INDEXER_TRACE_MAX=16
```
Limitations: both TP ranks emit (lines duplicate); emissions are labelled by
counter, **not by layer id** — so the mapping from trace line to layer is
unknown, and per-layer attribution requires adding the layer prefix.

---

## 6. What the agent concluded, and RETRACTED

Treat all four as cautionary, not as findings.

1. ~~"FP8 KV cache is the cause"~~ — inherited hypothesis. Refuted: upstream
   passes with the same flag.
2. ~~"Fails above a fixed absolute needle position (~2048)"~~ — refuted by §3b
   (passes at abs 2721, fails at abs 1695).
3. ~~"Fails above a fixed total length (~8192)"~~ — refuted: response is
   non-monotonic (pt 8316 passes while 5100 and 6650 fail).
4. ~~"~26× indexer ranking-quality gap vs native"~~ — **withdrawn as unsound**:
   it divided our hard-needle coverage requirement by upstream's, but upstream
   runs a *different, easier* probe (§3e).
5. ~~"The failure is decode-side, at the speculative block boundary"~~ —
   `num_speculative_tokens=5` and the collapse is at answer position 5, which
   looked compelling. **Refuted by `digitspad`** (§3c): digits at answer
   position ~6 pass.
6. ~~"The sm_89 indexer produces bad scores"~~ — not supported by §5: the needle
   frequently ranks top 5–10%.

### The agent's current interpretation (a hypothesis, not a finding)
Selection coverage (`index_topk / candidates`) is the operative variable, and
what gets selected depends on **query content** — a query naming "digits" raises
the digit-bearing entry's score. The digits are always physically present and
readable; whether they are *selected* depends on how much of the candidate set
the top-k admits and whether the query makes them salient.

---

## 7. Open questions the agent could not answer

1. **Is `index_topk=512` sufficient for exact multi-token verbatim recall at
   ~10K tokens under a generic query on a CORRECT implementation?** i.e. is this
   architectural, or a port defect? Upstream has only validated the easy
   random-word needle, so there is no like-for-like reference point.
2. **Why does the needle's entry rank vary 5%→48% across layers?** Is that
   expected for DSA, or does it indicate a per-layer defect (wrong weights,
   wrong rope, a layer-index mapping error)? The trace cannot currently attribute
   lines to specific layers.
3. **`scoring_func: sqrtsoftplus` and `topk_method: noaux_tc`** — never checked
   whether this port implements these, or falls back to softmax/plain top-k.
4. **`num_hash_layers: 3`, `hc_eps`, `hc_mult`, `hc_sinkhorn_iters: 20`** — never
   investigated at all. What are the "hash layers" / Sinkhorn iterations for, and
   does this port implement them?
5. Why does the failure always lose the **trailing** component of the needle
   (`GLACIER-7741-ORYX` → `GLACIER-7741` → `GLACIER`)? A copy-length truncation
   was hypothesised and refuted (§6.5); what else produces that shape?
6. `index_topk=2048` fixes it to ~20K but the rule `reliable ≈ 10 × index_topk`
   implies ~26,000 for 262K — ~40% of all entries, effectively dense. Is there a
   principled setting, or is a higher `index_topk` simply wrong?
7. A clean DSpark A/B is **blocked**: removing `SPECULATIVE_CONFIG` boots
   (`speculative_config=None`) but the first long request dies with
   `moe_w2 exact cache failed to converge after 8 replay passes (1 routed expert
   pairs still missing)` — removing speculation shifts the expert access pattern
   and trips the exact path's strict miss-replay guard. Is there a safe way to
   isolate DSpark's contribution?

---

## 8. Primary sources to read yourself

- `/root/models/DeepSeek-V4-Flash-0731/inference/model.py` — **the authority.**
  `Compressor` (line 284), `Indexer` (≈393), `Attention.forward` (≈490).
- `/root/models/DeepSeek-V4-Flash-0731/inference/kernel.py` — `act_quant`,
  `fp4_act_quant`.
- `overlay/vllm/vllm/model_executor/layers/sparse_attn_indexer.py` — selection.
  Prefill top-k at the `ops.top_k_per_row_prefill` call; decode at
  `ops.top_k_per_row_decode` / `torch.ops._C.persistent_topk`.
- `overlay/vllm/vllm/v1/attention/ops/triton_sparse_mla_dsv4.py` — the sm_89
  attention port (packed layout contract in the module docstring).
- `overlay/vllm/vllm/v1/attention/ops/triton_paged_mqa_logits_dsv4.py` — paged
  indexer logits.
- `overlay/vllm/vllm/models/deepseek_v4/compressor.py` +
  `common/ops/fused_compress_quant_cache.py` — the compressor.
- `~/llama.cpp` — a DFlash+DSpark build that reportedly passes the needle test;
  `src/models/dflash.cpp`, indexer params in `src/llama-arch.cpp` (note it models
  `indexer.block_size` and `indexer.local_blocks`, which this port does not
  obviously have).
- `STATUS.md` §§5.10–5.19 — the full chain with retractions marked inline.

## 9. Reproduction

```bash
# production server (sole container, ~6-10 min boot)
docker start moet-0731-dspark-exact
until curl -fsS http://127.0.0.1:8011/health >/dev/null; do sleep 20; done

# the failing case, and the two that expose it
python3 tools/needle_digits_probe.py --lengths 8192 --variant ask
python3 tools/needle_digits_probe.py --lengths 8192 --variant digitsonly
python3 tools/needle_probe.py 8011 8000 0.1     # upstream's own probe

# cheap CPU harnesses (seconds, no GPU, safe while serving) — tools/README.md
```

Costs, so you can weigh what to ask for: a boot is 6–10 min; an 8K needle point
~100 s; 32K ~650 s; the CPU harnesses are seconds.
