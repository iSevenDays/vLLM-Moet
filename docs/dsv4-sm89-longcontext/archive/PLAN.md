> # ⚠️ SUPERSEDED — DO NOT QUOTE
> Consolidated into [`../README.md`](../README.md) (2026-08-01), which is now the
> single source of truth. This file is kept for provenance only and contains
> **withdrawn claims** — notably the §5.19 indexer rank trace, which measured the
> wrong column (README §2). Verify anything here against README before using it.

# PLAN — DSv4-Flash long-context digit loss on sm_89

**Written 2026-08-01.** Successor to `BRIEFING.md` (raw data) and `STATUS.md`
(§§5.10–5.19, the chain with retractions marked inline). Read this first; it
corrects a measurement those two present as decisive.

**Audience: the next agent.** Boots are 6–10 min and needle points 100–650 s, so
every task below states its cost and its falsifier. Cheap CPU checks come first
on purpose — three of the four items in §2 need no GPU at all.

---

## 0. Standing decisions

- **`/root/models/DeepSeek-V4-Flash-0731` is the only checkpoint.** The older
  `/root/models/DeepSeek-V4-Flash` (146 GB) is superseded and must not be used
  for any new measurement. Configs differ materially — 0731 adds the DSpark
  stage (`dspark_block_size 5`, `dspark_target_layer_ids [40,41,42]`,
  `dspark_markov_rank 256`) and carries 46 `compress_ratios` entries vs 44.
  Any result quoted from the old checkpoint is not comparable and should be
  re-measured or dropped. *Deletion of the old tree is approved in principle but
  had not been executed at the time of writing — confirm before reclaiming.*
- **`vllm-moet-sm89:v0251` is the canonical living tag.** Preserve baselines as
  suffixed tags; never introduce candidate-tag defaults.
- **Never `--enforce-eager`** on this rig.
- Production container: `moet-0731-dspark-exact`, port **8011**, sole container,
  trace off by default.

---

## 1. CORRECTION: §5.19's rank trace did not measure the needle

**This is the most important thing in this document.** `STATUS.md` §5.19 and
`BRIEFING.md` §5 present the indexer rank trace as "the single most informative
measurement" and draw three conclusions from it. The trace was pointed at the
wrong column.

### What was traced

The trace ran with `VLLM_DSV4_INDEXER_TRACE_POS=4843`, taken from the probe's
`needle_abs_pos_est` field, which is computed as `int(prompt_tokens * depth)` —
`int(9686 × 0.5) = 4843` (`tools/needle_digits_probe.py:232`). Every line in
`runlogs/indexer_rank_trace_20260801.log` therefore reports ratio-4 column
`4843 // 4 = 1210`.

### Where the needle actually is

The `ask`/8192/depth-0.5 prompt is deterministic (`needle_code` seeds
`random.Random(0x4E4C ^ 8192 ^ 0)` → `PELICAN-1605`). Reconstructed byte-exact
on the host and tokenized with the real tokenizer inside the container
(CPU only, seconds):

| item | value |
|---|---|
| context | 29,433 bytes / 9,682 tokens (+2 template prefix ⇒ ≈9,684; probe reports pt 9,686) |
| needle inserted at | byte 14,515 of 29,076 (**49.9 % by byte**) |
| needle line first token `'IM'` | **abs 3,858** (**39.8 % by token**) |
| `' P' 'EL' 'IC' 'AN'` | abs 3,868–3,871 → **ratio-4 column 967** |
| `'160'`, `'5'` (THE DIGITS) | abs 3,873, 3,874 → **ratio-4 column 968** |
| **column actually traced** | **1210** — abs tokens 4,840–4,843 |

Token dump of the traced column, from the live tokenizer:

```
col 1210 (abs 4836..4843): [' in', ' `', 'gen', '/', '`', ' as', ' the', ' validated']
```

That is filler prose. The needle's digit-bearing entry sits **242 ratio-4
columns away** and was never observed. The r128 figure is wrong the same way
(traced col 37; true col `3873 // 128 = 30`).

The root cause of the error is that the probe inserts the needle at a **byte**
fraction of the filler but reports its position as a **token** fraction. The
filler's first half (markdown + Python) tokenizes denser than its second half,
so byte-50 % lands at token-39.8 %.

### What this invalidates, and what survives

**Invalidated — do not quote:**
- The rank table (112 / 592 / 523 / 1003 / 693 / 184 / 125 / 492) describes an
  arbitrary filler entry, not the needle.
- "The needle's entry is selected in roughly half the traced ratio-4 layers."
- "Its rank varies 5.4 % → 48 % across layers." That is the layer-to-layer rank
  spread of *a filler token*, which is unremarkable.
- Open question #2 in `BRIEFING.md` §7 ("why does the needle's entry rank vary
  5 %→48 %?") is not a real observation and should be struck.

**Survives:**
- "The top-k boundary is exactly 512" (rank 492 selected, 523 not). Selection
  correctness does not depend on which column was watched. Still corroborated by
  `tools/test_indexer_topk_selection.py` and the paged suite.
- Everything in `BRIEFING.md` §3 (the 60 end-to-end measurements). Those are
  black-box request/response pairs and are unaffected.
- The `index_topk=2048` result (§5.15): 6/6 pass to 18.5K. Unaffected.

**Newly re-opened:** whether the digit-bearing entry survives the k=512 cut is
now **unmeasured**. The central mechanism claim of §5.18–5.19 rests on it.

### Second defect in the same instrument

`_trace_indexer_rank` is called only from the **prefill** branch
(`sparse_attn_indexer.py:663`). The decode branch (`:688` onward,
`persistent_topk` / `top_k_per_row_decode`) has no trace at all. Selection
during generation has never been observed.

### Also mislabeled: BRIEFING §3b

The "needle absolute token pos" column (194 / 1695 / 2324 / 4843 / 9492) comes
from the same estimate: `--abs-pos N` is converted to `depth = N / L` and then
applied as a byte fraction. The true token positions are all lower. The
pass/fail pattern still refutes the "fixed position" and "fixed length"
hypotheses (those only need *some* monotone position axis), but the axis values
are wrong and must be recomputed before anyone fits a threshold to them.

---

## 2. CLOSED: the four "never investigated" config keys

`BRIEFING.md` flagged these as the most likely place a fresh reader finds
something missed. All four are implemented. Two were also mis-scoped: they are
MoE/residual parameters with no path into indexer selection, so they could not
have explained the needle symptom.

| key | status | evidence |
|---|---|---|
| `scoring_func: sqrtsoftplus` | **implemented**, matches checkpoint | `fused_topk_bias_router.py:243` dispatches to `vllm_topk_softplus_sqrt` → CUDA `ops.topk_hash_softplus_sqrt`; torch fallback at `:76` is `sqrt(softplus(x))`, identical to checkpoint `Gate.forward` (`model.py:576`, `F.softplus(scores).sqrt()`). Not a softmax fallback. |
| `topk_method: noaux_tc` | **implemented**, matches checkpoint | `nvidia/model.py:599–603` allocates `e_score_correction_bias` exactly when `topk_method == "noaux_tc"`. Bias-for-selection-only semantics preserved: `fused_topk_bias_router.py:78–84` adds bias to `scores_for_choice` but gathers weights from unbiased `scores` — same as checkpoint's `original_scores.gather` (`model.py:585`). The port comments the DSv4-Flash-specific reason (all biases ≈ 8.08 would flatten the weights). |
| `num_hash_layers: 3` | **implemented**, fails loudly not silently | `nvidia/model.py:584` `is_hash_moe = extract_layer_index(prefix) < config.num_hash_layers`, matching checkpoint `Gate.hash = layer_id < args.n_hash_layers`. `tid2eid` lookup table allocated at `:590`. **`:716` raises `ValueError` if `input_ids` is absent** rather than degrading to score routing. |
| `hc_*` (`hc_mult 4`, `hc_sinkhorn_iters 20`, `hc_eps`) | **implemented** | Hyper-Connections: the residual stream carries `hc_mult` copies; `hc_pre` reduces 4→1 via Sinkhorn-normalized weights, `hc_post` expands 1→4 (checkpoint `Block.hc_pre`/`hc_post`, `model.py:680–693`). Port: `nvidia/model.py:866–979` threads `hc_sinkhorn_iters`; MTP path at `nvidia/mtp.py:107–156`. |

**Scoping note.** `scoring_func` and `topk_method` govern **MoE expert routing**;
`hc_*` governs **residual mixing**. Neither touches `index_score` or the top-k
over compressed entries. A defect in them would show up as broad quality
degradation, not as selective loss of a needle's trailing tokens. Deprioritize
them for this symptom — but note the port's torch fallbacks are only reached on
XPU/CPU, so the **CUDA kernels** `topk_hash_softplus_sqrt` and the Sinkhorn path
remain unverified against the checkpoint numerically (see task V4).

One latent risk worth a cheap guard: the hash-MoE torch fallback
(`fused_topk_bias_router.py:88`) honors the table only when
`input_tokens is not None`, silently falling back to score routing. The CUDA
path is reached in production and `nvidia/model.py:716` guards it, so this is
not live — but the fallback should assert rather than degrade.

---

## 3. The reference point that changed: llama.cpp passes

**Operator-verified manually (2026-08-01): the needle test passes under
llama.cpp on this same hardware.** This is the like-for-like reference that
`BRIEFING.md` §7 Q1 said did not exist, and it shifts the prior substantially:
"architectural limit of `index_topk=512`" becomes much less likely, "port
defect" much more likely.

Before treating it as decisive, three confounds must be closed — all cheap:

1. **Which weights.** Shell history shows llama.cpp serving
   `/root/antirez/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`.
   Confirm whether that GGUF was converted from the **0731** checkpoint or the
   superseded one (§0). Different weights ⇒ not like-for-like.
2. **Which probe, which length, which depth.** A pass at 8K random-word depth
   0.1 and a pass at 10.8K realistic-filler depth 0.5 are different claims.
   Record prompt tokens, filler type, depth, and the exact reply.
3. **Which quantization.** That GGUF is IQ2-XXS experts with Q8 attention
   projections; our path is FP4 E2M1 experts. Expert precision differs, and
   `dsv4-longcontext-derail-rootcause` already implicates 2-bit expert weight
   fidelity in prefill — so a *pass* under IQ2 is evidence **against** the expert
   -precision story and **for** something in the vLLM attention/selection path.

**Structural lead from the same source.** `llama.cpp/src/llama-arch.cpp:260–261`
defines `%s.attention.indexer.block_size` and
`%s.attention.indexer.local_blocks`. `local_blocks` has no obvious counterpart in
this port. If llama.cpp unconditionally admits the most recent N compressed
blocks *in addition to* the top-k, that is a selection-policy difference that
would directly produce our symptom shape. **This is the highest-value structural
lead in the document** — see task V3.

---

## 4. Hypotheses, ranked

Ranked by (posterior after §§1–3) × (cheapness to falsify). H1 and H2 both
became more likely once llama.cpp passed.

**H1 — Selection policy differs from the reference: missing always-included
local/recent compressed blocks.**
The checkpoint concatenates `always_included_window_idxs` (the 128-token raw
window) with `compress_topk_idxs`, sharing one softmax (`model.py:520`).
llama.cpp additionally models `indexer.local_blocks`. If the port admits *only*
the top-k from the compressed segment, entries that are recent-but-not-top-k are
dropped where the reference keeps them.
*Falsifier:* read the port's index assembly and compare against `model.py:513–520`
and llama.cpp's dflash graph. CPU, no boot.

**H2 — The digit entry is in fact selected, and the mechanism story is wrong.**
§5.18's "query content raises the digit entry's score" was inferred from
`digitsonly`/`digitspad` passing, never measured — and §1 shows the one direct
measurement watched the wrong column. If column 968 turns out to be selected in
every layer on the *failing* `ask` request, the failure is not selection at all
and the investigation returns to the value/decode path with §5.17's framing
un-retracted.
*Falsifier:* task V1. One boot.

**H3 — Chunked-prefill boundary corrupts the compressor's overlapping window.**
§5.14 found digit fidelity is chunk-size dependent (`1056` vs `528` flips
individual points), which is unexplained by a pure top-k coverage story — chunk
size should not change *scores*. The ratio-4 compressor is overlapping
(`Compressor.overlap_transform`) and carries `kv_state`/`score_state` across
calls; the reference builds them in one `start_pos == 0` pass.
`tools/test_compressor_vs_checkpoint_ref.py` validated semantics but
**within a single chunk**.
*Falsifier:* extend that harness to drive the port's compressor in N chunks and
compare against the reference's single-pass output at the boundary columns. CPU,
no boot.

**H4 — Config divergence from upstream's validated recipe, not sm_89.**
Our launch differs from `pro6000x2-tp2.yaml` on more than hardware:
`max_num_batched_tokens` 1056 vs 4096, speculative `dspark k=5` vs
`deepseek_mtp k=2`, `max-model-len` 262144 vs 131072, `gpu-memory-utilization`
0.98 vs 0.92. The comparison in §3e attributes the whole gap to sm_89 while
holding none of these fixed.
*Falsifier:* task V5. One boot.

**H5 — `index_topk=512` is genuinely marginal for this task on any hardware.**
The prior standing explanation. Now weakened by §3 (llama.cpp passes) and by §1
(its direct evidence evaporated), but not eliminated — llama.cpp may simply be
running an easier probe.
*Falsifier:* V1 measuring the digit column comfortably inside the cut on a
*failing* request would kill it; the llama.cpp confounds in §3 closing cleanly
would too.

**Retired.** DSpark block boundary (killed by `digitspad`); fp8 KV (upstream
passes with it; INT8 A/B 6/6 identical); fixed needle position; fixed total
length; indexer key-quant granularity (~1 %, top entry never dislodged);
"sm_89 produces bad indexer scores" — note this was retired on the strength of
the §5.19 trace and is now **unsupported in both directions**, though the
kernel-level suites still pass.

---

## 5. Verification queue

Ordered: everything that needs no GPU first.

### V1 — Retrace with the correct column *(one boot, ~15 min; the decisive one)*

The measurement §5.19 was meant to be. Trace the **digit** token, not the
estimate.

```bash
docker run ... -e VLLM_DSV4_INDEXER_TRACE=1 \
               -e VLLM_DSV4_INDEXER_TRACE_POS=3873 \
               -e VLLM_DSV4_INDEXER_TRACE_MIN_N=2000 \
               -e VLLM_DSV4_INDEXER_TRACE_MAX=64 ...
python3 tools/needle_digits_probe.py --lengths 8192 --variant ask   # FAILS
python3 tools/needle_digits_probe.py --lengths 8192 --variant digitsonly  # PASSES
```

Before trusting `3873`, re-derive it — the prompt is deterministic, so this is a
CPU check of seconds (method in §1; `/tmp/needle_ctx.txt` reconstruction script
pattern is in the session log). Trace **both** variants: §5.18's whole mechanism
is that the digit-naming query selects the digit entry in more layers. That is
now a *directly testable* prediction.

Prerequisites, both cheap and both required for the result to be readable:
- **Label emissions by layer id.** Currently counter-labeled, so per-layer
  attribution is impossible (`sparse_attn_indexer.py:106`).
- **Suppress the duplicate rank.** Both TP ranks emit; gate on rank 0.

Outcomes:
- Digit column outside the cut in several layers on `ask`, inside on
  `digitsonly` → §5.18 confirmed, H5 back in play, fix is coverage (§6).
- Digit column **inside** the cut on the failing `ask` → **H2**: selection is not
  the mechanism; reopen the value/decode path.

### V2 — Trace the decode path *(code change + the same boot as V1)*

Add the `_trace_indexer_rank` call to the decode branch (~`:828`, after the
`persistent_topk` / `top_k_per_row_decode` selection). Generation-time selection
has never been observed, and the symptom is a *generation* failure. Fold into
V1's boot.

### V3 — `indexer.local_blocks` parity *(CPU, ~1 h reading)*

Compare three implementations of the attend-set assembly:
- checkpoint `model.py:513–520` (`get_window_topk_idxs` ⧺ `compress_topk_idxs`)
- this port (`sparse_attn_indexer.py` + the sparse-MLA backend's index handling)
- llama.cpp `src/models/dflash.cpp` + `LLM_KV_ATTENTION_INDEXER_LOCAL_BLOCKS`

Answer one question: **does any of them always-include recent compressed blocks
that the port drops?** If yes, H1 is confirmed and the fix is a selection-policy
patch, not a bigger `index_topk`.

### V4 — Numerically verify the CUDA routing kernel *(CPU/GPU unit, minutes)*

The torch fallbacks in `fused_topk_bias_router.py` are XPU/CPU-only; production
runs `ops.topk_hash_softplus_sqrt`. Assert kernel ≡ `_topk_softplus_sqrt_torch`
≡ checkpoint `Gate.forward` on random gating outputs including the
DSv4-Flash bias regime (all ≈ 8.08). Low prior for this symptom (§2 scoping),
but it is minutes and closes the last unverified piece of §2. Add an assert on
the `input_tokens is not None` fallback while there.

### V5 — Upstream recipe on sm_89 *(one boot + ~2 min probe)*

Run `bench/recipes/deepseek-v4-flash/pro6000x2-tp2.yaml`'s serve args verbatim
(chunk 4096, `deepseek_mtp` k=2, ctx 131072, util 0.92) and run **upstream's own
probe**, which is the cheap gate (~65 s at 10.8K):

```bash
python3 tools/needle_probe.py 8011 8000 0.1
```

A pass isolates the gap to configuration, not architecture, and collapses H4
into a one-line fix. A fail makes the sm_89 attribution much stronger. Note the
`moe_w2 exact cache` guard may trip when speculation changes (§5.18) — vary
`num_speculative_tokens` rather than removing speculation, or raise the replay
budget; **never relax the strict guard**.

### V6 — Close the llama.cpp confounds *(minutes)*

Per §3: which checkpoint the GGUF came from, which probe/length/depth passed,
and the exact reply text. Record in `runlogs/`. This is the cheapest way to
firm up the single most informative new datum.

### V7 — Measurement hygiene *(CPU, ~30 min; do before any new sweep)*

1. `needle_digits_probe.py` must report the **measured** token position of the
   needle, not `int(pt × depth)`. It already has the tokenizer available via the
   server; simplest correct fix is to locate the needle's token span from the
   returned prompt or to tokenize locally and report both byte- and token-depth.
2. `--abs-pos` should place by **token** position (tokenize, insert, verify)
   rather than converting to a byte fraction.
3. Recompute BRIEFING §3b's position column and restate the two refutations
   against the corrected axis.

Without this, every future position-dependent claim inherits the §1 error.

---

## 6. Fix strategy

### Now — restore correctness in production

`index_topk=2048` is the only change measured to convert failures to passes:
6/6 exact to 18.5K (§5.15), still failing at 35.8K.

```bash
--hf-overrides '{"index_topk": 2048}'
```

Costs compute, not VRAM. Ship it as the interim production setting **if** long-
context exactness matters more than throughput, and record the throughput delta.
Caveats to keep attached to it: it is a coverage workaround whose mechanism is
now unconfirmed (§1); it does not hold at 32K+; and the `reliable ≈ 10 ×
index_topk` extrapolation implies ~26,000 at 262K, i.e. effectively dense, which
is not a principled setting. `index_topk = 0` does **not** mean unlimited —
it selects nothing and leaves only the 128-token window (STATUS.md, "`index_topk
= 0` — asked").

### Next — fix the cause, not the coverage

Follow whichever of H1/H2/H3 V1–V3 confirms:
- **H1 confirmed** → patch the attend-set assembly to match the reference /
  llama.cpp (always-include local blocks). Likely small, and would let
  `index_topk` return to 512.
- **H2 confirmed** → selection is exonerated; reopen the decode/value path with
  §5.17's evidence, which was retracted for a reason that no longer holds
  cleanly. Re-examine the `digitspad` control under the corrected instrument.
- **H3 confirmed** → fix the compressor's cross-chunk state; would also explain
  the otherwise-anomalous chunk-size sensitivity (§5.14).
- **H4 confirmed (V5 passes)** → adopt upstream's serve args; no kernel work.

### Do not

- Do not quote the §5.19 rank table, the "5 %→48 % spread", or the "~26×
  ranking gap" (already withdrawn).
- Do not run another end-to-end sweep before V7 lands.
- Do not relax the `moe_w2` strict miss-replay guard to unblock a DSpark A/B.
- Do not use `--enforce-eager`.

---

## 7. Process note

Six interpretations have now been retracted in this investigation, and this
document adds a seventh — the §5.19 trace, which was itself built to end the
guessing. The pattern is consistent: **the raw request/response measurements have
never been wrong; every failure has been in what an instrument was believed to be
pointing at.** §5.18's own closing note said "the cheap falsifier is always worth
writing before the expensive confirmation" — the missing step here was cheaper
still, and would have caught it: *tokenize the prompt and confirm the instrument
is aimed at the token you care about.* The prompt is deterministic and the
tokenizer runs on CPU in seconds.

Before trusting any new diagnostic, verify its addressing on a known input.
