# PLAN — execution guide for the next agent

**Written 2026-08-01 by Claude (Opus 5), who may be unavailable for ~1 week.**

You are good at executing tasks. This document assumes you do **not** have the
accumulated context of this investigation, so every task below is fully
specified: exact commands, expected output, and a decision table telling you what
to conclude and where to go next. Do not improvise the order.

- **Background/knowledge → [`README.md`](README.md).** Read §0, §2, §4 before
  starting. It tells you what is already ruled out, so you don't redo it.
- **This file → what to do.** Tasks are `T1…T8`, ordered. Cheapest and most
  informative first. **T1, T2, T3, T5, T6, T8 need no GPU boot at all.**

---

## 0. Prime directives

1. **Do not run an end-to-end needle sweep before T1 lands.** Every
   position-dependent number produced before T1 is suspect (README §2).
2. **Verify an instrument's addressing before trusting its output.** The single
   worst error in this investigation was a diagnostic aimed at the wrong data
   (README §2). A CPU check of seconds would have caught it.
3. **One GPU container at a time.** The model fills both GPUs (~48 GiB each).
   Two containers = OOM and a wasted 10-minute boot.
4. **Record every result in `runlogs/` and update `README.md`.** If you conclude
   something, write down the evidence and the command that produced it.
5. **When a result contradicts this plan, believe the result.** Seven
   interpretations have already been retracted here. Write the contradiction
   down; don't force it into the existing story.
6. **Prefer CPU harnesses over boots.** Boot 6–10 min; 8K needle point ~100 s;
   32K ~650 s; CPU harnesses seconds. See `tools/README.md`.

---

## 1. Environment — exact facts

| item | value |
|---|---|
| Repo (work here) | `/root/autostart/vllm-ds4/vLLM-Moet` |
| Docs | `docs/dsv4-sm89-longcontext/` (README = knowledge, PLAN = this file) |
| Model | `/root/models/DeepSeek-V4-Flash-0731` — **the only valid checkpoint** |
| Image | `vllm-moet-sm89:v0251` |
| Production container | `moet-0731-dspark-exact`, port **8011** |
| GPUs | 2 × RTX 4090 D 48 GiB, **sm_89** (Ada) |

### Start the production server

```bash
docker ps --format '{{.Names}}' | grep -E '^moet' || echo "(none running - good)"
docker start moet-0731-dspark-exact
# wait for readiness (6-10 min). NEVER assume it is up.
until curl -fsS --max-time 5 http://127.0.0.1:8011/health >/dev/null 2>&1; do
  docker ps --filter name=moet-0731-dspark-exact --filter status=running -q | grep -q . \
    || { echo "CONTAINER DIED"; docker logs --tail 40 moet-0731-dspark-exact; break; }
  sleep 20
done
curl -fsS http://127.0.0.1:8011/v1/models
```

### Launch a *modified* config (needed for T4, T7)

The launcher is `./docker/serve_sm89_ds4.sh`, driven entirely by env vars. This
is the **production baseline** — copy it and change only what a task says to:

```bash
cd /root/autostart/vllm-ds4/vLLM-Moet
MODEL=/root/models/DeepSeek-V4-Flash-0731 \
CACHE=/root/models/moet-cache-0731-exact \
JIT_CACHE=/root/models/moet-cache/jit \
STORE=/root/models/moet-cache-0731-exact/packs \
NAME=<pick-a-name> PORT=8011 NETWORK=host RESTART=no \
RESIDENCY=exact EXACT_GB=30 ARENA_GB=40 MEM_GB=428 SCALE_REFIT=0 \
TP=2 GPUS='"device=0,1"' CUSTOM_ALL_REDUCE=0 \
MAXLEN=262144 UTIL=0.98 BATCHED_TOKENS=1056 NUM_SEQS=3 \
CUDAGRAPH_SIZES=1,2,4,6,8,12,18 PREFIX_CACHING=0 MTP_TOKENS=0 \
SPECULATIVE_CONFIG='{"method":"dspark","num_speculative_tokens":5,"dspark_scheduler":false}' \
EXTRA_DOCKER_ENV='-e VLLM_MOE_W2_DELTA_TRACE=1' \
EXTRA_ARGS='--cudagraph-metrics --kv-cache-memory-bytes 4846832640' \
READY_TIMEOUT_S=1800 ./docker/serve_sm89_ds4.sh
```

### Always restore production when done

```bash
docker stop <your-container> && docker rm <your-container>
docker start moet-0731-dspark-exact     # then wait for health as above
```

---

## 2. Invariants — never do these

| never | why |
|---|---|
| `--enforce-eager` | operator directive; it also masks graph-path bugs |
| Relax the `moe_w2` strict miss-replay guard | it exists so no output is emitted from a residual cache miss. If it trips, vary `num_speculative_tokens` instead of removing speculation |
| Use the old checkpoint for any measurement | it is **deleted** (2026-08-01, operator-confirmed). Nothing to use |
| Quote the §5.19 rank table, the "5 %→48 % spread", or the "~26× ranking gap" | all withdrawn (README §2, §4) |
| Hand-edit `patches/` | generated. Edit `overlay/` and run `tools/gen_patches.py` |
| Edit files under `vllm/` (the pinned v0.25.1 baseline) | copy to `overlay/vllm/<same path>` and edit there |

### Committing an `overlay/` change (mandatory sequence)

```bash
python3 tools/gen_patches.py            # regenerate
python3 tools/gen_patches.py --verify   # must print OK: N/N
python3 tools/gen_patches.py --check    # must print OK: N/N
git add overlay/<file> patches/ && git commit
```
If `--verify` or `--check` is not `N/N`, **do not commit**. Currently N = 85.
End commit messages with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.

---

## 3. Traps that have already cost time — recognize these

| symptom | cause | fix |
|---|---|---|
| Your edited file has no effect in the container | **`EXTRA_MOUNTS` is not a knob** in `serve_sm89_ds4.sh`; it is silently ignored | pass `-v src:dst:ro` through `EXTRA_DOCKER_ENV`, which is spliced raw into `docker run`. Verify with `docker inspect <name> --format '{{range .Mounts}}{{.Destination}}{{println}}{{end}}'` |
| `hasattr(torch.ops._C, "x")` is False for an op that exists | `torch.ops._C` registers **lazily on `import vllm`** | `import vllm` (or `from vllm import _custom_ops`) *first*, then probe |
| Every test after the first fails with weird CUDA errors | `cooperative_topk` has no sm_89 kernel image and **poisons the CUDA context** | skip it below sm_90 (serving gates it on `has_device_capability(90)`) |
| `docker rm -f` printed nothing and the old container is still up | `rm -f` can fail silently; two containers then fight for VRAM | always `docker stop X; docker rm X`, then `docker ps` to confirm |
| Boot dies at graph capture with `Triton Error [CUDA]: out of memory` | only ~440 MiB headroom in the default config; a new Triton variant compiles during capture | reduce `--kv-cache-memory-bytes` (2 GiB = `2147483648` still holds one 262K request; min ≈1481 blocks ≈1.484 GiB) |
| Boot dies with `operation not permitted when stream is capturing` | a Triton kernel variant absent from the JIT cache is compiling **inside** CUDA-graph capture | pre-warm that variant, or avoid changing kernel constexprs |
| Tests fail with CuTeDSL `cubin-chip=sm_89` compile errors | a CuTeDSL kernel with no pre-SM90 lowering | already fixed for `dequantize_and_gather_k_cache`; if a new site appears, gate on `current_platform.has_device_capability(90)` |
| Root filesystem full | happened once; the operator resized it | `df -h /` before long runs |

---

## 4. Task queue

### T1 — Fix the probe's position reporting *(CPU, ~45 min, NO boot)* — **do first**

**Why.** `tools/needle_digits_probe.py` inserts the needle at a **byte** fraction
of the filler but reports its position as `int(prompt_tokens × depth)`, a **token**
fraction. At `ask`/8192 that is off by **+970 tokens = 243 ratio-4 columns**. This
is what invalidated the rank trace (README §2). Until fixed, every
position-dependent claim is unreliable.

**Steps.**
1. Confirm the bug and get ground truth:
   ```bash
   docker run --rm --entrypoint python3 -v "$PWD:/w:ro" \
     -v /root/models/DeepSeek-V4-Flash-0731:/model:ro \
     -v /root/autostart/CLAUDE.md:/root/autostart/CLAUDE.md:ro \
     vllm-moet-sm89:v0251 /w/tools/verify_needle_token_position.py
   ```
   Expect: context `29,433 bytes / 9,682 tokens`, digits at token **3,871**,
   ratio-4 column **967**, estimate 4,841 → column 1210, "off by 243".
2. Edit `tools/needle_digits_probe.py`:
   - Replace the `needle_abs_pos_est` field (`int(pt * depth)`) with a
     **measured** `needle_token_pos`. Reuse the approach in
     `tools/verify_needle_token_position.py` (tokenize the prefix up to the
     needle's byte offset). Keep the byte offset too, as
     `needle_byte_pos`/`needle_byte_frac`.
   - Make `--abs-pos N` place the needle by **token** position: tokenize, insert,
     re-tokenize, verify the needle's token index is within ±2 of N; log the
     achieved position. Do **not** convert N to a byte fraction.
   - Emit both `needle_token_pos` and `needle_token_frac` in the JSON record.
3. Re-run the verifier and confirm probe-reported == verifier-measured.

**Done when.** For `ask`/8192/depth 0.5 the probe reports `needle_token_pos`
≈ 3871 (not 4843), and `--abs-pos 3871` actually places it there.

**Then.** Commit (`tools/` only, no patch regeneration needed). Go to **T2**.

---

### T2 — `indexer.local_blocks` parity *(CPU, ~1–2 h reading, NO boot)* — **highest value**

**Why.** This tests **H1**, the top structural hypothesis. The checkpoint builds
its attend set as `cat([always_included_window_idxs, compress_topk_idxs])`
(`/root/models/DeepSeek-V4-Flash-0731/inference/model.py:513–520`). llama.cpp —
which **passes** this needle test on the same hardware — additionally models
`indexer.local_blocks` (`~/llama.cpp/src/llama-arch.cpp:260–261`). If the
reference/llama.cpp unconditionally admit the most recent N *compressed* blocks
**in addition to** the top-k, and this port admits only the top-k, then recent
entries the reference keeps are being dropped here. That alone would produce the
observed symptom.

**Read these three, answer one question.**
1. Checkpoint: `inference/model.py`, `Attention.forward` around lines 505–520
   (`get_window_topk_idxs`, `get_compress_topk_idxs`, the `torch.cat`).
2. This port: `vllm/vllm/models/deepseek_v4/common/ops/cache_utils.py:519`
   `combine_topk_swa_indices` (+ its Triton kernel at `:568`), and how
   `overlay/vllm/vllm/model_executor/layers/sparse_attn_indexer.py` feeds it.
3. llama.cpp: `~/llama.cpp/src/models/dflash.cpp`, plus
   `LLM_KV_ATTENTION_INDEXER_LOCAL_BLOCKS` / `..._BLOCK_SIZE` in
   `src/llama-arch.cpp:255–262`, and where `hparams` consumes them.

**The question: does any implementation always-include recent compressed blocks
that this port drops?** Also check whether the GGUF metadata actually sets
`local_blocks` for this model (`gguf-py` or `llama-gguf` dump on the GGUF in T6),
because an unset key means llama.cpp isn't using it either.

| finding | conclusion | next |
|---|---|---|
| Reference/llama.cpp always-include local compressed blocks, port does not | **H1 CONFIRMED** — this is likely the root cause | write it up, then design the patch to `combine_topk_swa_indices`; `index_topk` may return to 512. Escalate to operator before landing a semantics change |
| All three assemble the set the same way | H1 refuted | record it, go to **T3** |
| `local_blocks` unset in the GGUF metadata | H1 weakened, not settled | note it, go to **T3** |

**Deliverable.** A section in `README.md` with the three code excerpts side by
side and an explicit verdict.

---

### T3 — Make the rank trace usable *(code, ~30 min, NO boot)*

**Why.** T4 depends on this. The existing trace has three defects that make its
output unreadable or absent.

**File:** `overlay/vllm/vllm/model_executor/layers/sparse_attn_indexer.py`
(the trace helper is at `:79`, the prefill call site at `:663`).

1. **Label by layer, not by counter.** Currently `logger.info` prints
   `[n/max]` (`:106`). Per-layer attribution is impossible, and the whole point
   is comparing layers. Thread the layer identity in — `sparse_attn_indexer()` is
   a free function, so pass the caller's `prefix`/layer index down from
   `SparseAttnIndexer` (it has `self.prefix`-style context; check the call site in
   `overlay/vllm/vllm/models/deepseek_v4/attention.py`), and include it in the
   log line.
2. **Emit from TP rank 0 only.** Both ranks log today, so every line appears
   twice. Gate with
   `from vllm.distributed import get_tensor_model_parallel_rank` (guard against
   it being uninitialized).
3. **Add a decode-branch trace.** The helper is called only from prefill. The
   symptom is a *generation* failure and decode selection has **never been
   observed**. Add a call after the decode selection — after
   `torch.ops._C.persistent_topk` (`:810`) and `ops.top_k_per_row_decode`
   (`:819`). Note decode `logits` are shaped `[num_padded_tokens, max_seq_len]`
   and there are no `cu_seqlen_ks/ke`; use `seq_lens` for the valid range, and
   write a small separate helper rather than forcing the prefill one to fit.
4. **Accept a column range.** Add `VLLM_DSV4_INDEXER_TRACE_SPAN` (default 2) and
   report every ratio-4 column in `[pos//4 - span, pos//4 + span]`. Needed
   because the true column is **967–968** depending on tokenization details
   (README §2).

**Verify without a boot:** `python3 -c "import ast;ast.parse(open('<file>').read())"`,
then the mandatory patch sequence in §2. Do **not** boot yet.

**Then.** Go to **T4**.

---

### T4 — Retrace at the correct column *(one boot, ~30 min) — THE DECISIVE TASK*

**Why.** This is the measurement §5.19 was meant to be. It discriminates **H2**
from **H5**, i.e. whether selection is the mechanism at all.

**Steps.**
1. Re-derive the position (do not trust a stale number):
   run the T1 verifier; note `needle_token_pos` (expect ≈3871).
2. Stop production, launch with the trace on. Use the §1 baseline plus:
   ```bash
   NAME=moet-t4 \
   EXTRA_DOCKER_ENV="-e VLLM_MOE_W2_DELTA_TRACE=1 \
     -e VLLM_DSV4_INDEXER_TRACE=1 \
     -e VLLM_DSV4_INDEXER_TRACE_POS=3871 \
     -e VLLM_DSV4_INDEXER_TRACE_SPAN=2 \
     -e VLLM_DSV4_INDEXER_TRACE_MIN_N=2000 \
     -e VLLM_DSV4_INDEXER_TRACE_MAX=128 \
     -v $PWD/overlay/vllm/vllm/model_executor/layers/sparse_attn_indexer.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer.py:ro"
   ```
   **Confirm the mount landed** (§3) before waiting 10 minutes.
3. Run **both** variants and capture the trace after each:
   ```bash
   python3 tools/needle_digits_probe.py --lengths 8192 --variant ask \
     --out runlogs/T4_ask.json                       # EXPECTED: FAIL, PELICAN-1234
   docker logs moet-t4 2>&1 | grep "indexer rank trace" > runlogs/T4_trace_ask.log

   python3 tools/needle_digits_probe.py --lengths 8192 --variant digitsonly \
     --out runlogs/T4_digitsonly.json                # EXPECTED: PASS, 1605
   docker logs moet-t4 2>&1 | grep "indexer rank trace" > runlogs/T4_trace_digitsonly.log
   ```
   Tracing **both** is the point: README §4's mechanism claim is that the
   digit-naming query selects the digit entry in *more* layers. That is now a
   directly testable prediction.
4. Restore production (§1).

**Decision table.**

| observation on the FAILING `ask` | conclusion | next |
|---|---|---|
| digit column **outside** the k=512 cut in several ratio-4 layers, **and inside** in most layers on the passing `digitsonly` | selection **is** the mechanism; README §4's story confirmed. H5 back in play | go to **T5**; fix strategy = coverage (README §7) or H1's patch if T2 confirmed it |
| digit column **inside** the cut in essentially every layer on the failing `ask` | **H2 CONFIRMED** — selection is exonerated. The information is selected and still not produced | **stop and re-plan.** Reopen the decode/value path: re-examine the `digitspad` control (README §3b) with the corrected instrument, and use T3's new decode trace. Report to the operator before spending more boots |
| digit column *inside* on `ask` but the **decode** trace shows it dropped during generation | the failure is generation-time selection | this is a new finding — write it up prominently; it was never observable before T3 |
| trace emits nothing | `MIN_N` too high, or the mount didn't land, or `POS` out of range | check `docker inspect` mounts; lower `MIN_N`; confirm `POS` < prompt tokens |

**Record.** Put the raw trace logs in `runlogs/` and a summary table in
`README.md` §2, replacing the "unmeasured" note.

---

### T5 — Cross-chunk compressor parity *(CPU, ~1 h, NO boot)*

**Why.** Tests **H3**. Digit fidelity is **chunk-size dependent** — halving
`max_num_batched_tokens` 1056→528 flips individual points in *both* directions
(README §3a). A pure top-k coverage story cannot explain that: chunk size should
not change scores. The ratio-4 compressor is *overlapping* and carries
`kv_state`/`score_state` across calls, whereas the checkpoint builds them in one
`start_pos == 0` pass. `tools/test_compressor_vs_checkpoint_ref.py` validated the
semantics **within a single chunk only**.

**Steps.** Extend that harness with a chunk-split invariance case: drive the
port's compressor over the same token sequence (a) as one chunk and (b) split
into N chunks (include a split that lands **mid-`compress_ratio`**, i.e. not a
multiple of 4), and compare the compressed entries at and around the boundary
against the reference's single-pass output.

| result | conclusion |
|---|---|
| entries differ at boundaries | **H3 CONFIRMED** — fix the cross-chunk state; also explains the chunk-size sensitivity |
| byte-identical across splits | H3 refuted; the chunk-size sensitivity remains unexplained — record that explicitly as an open anomaly |

---

### T6 — Close the llama.cpp confounds *(~30 min, no GPU)*

**Why.** "llama.cpp passes the needle test on this hardware" is the single most
important new datum — it is the like-for-like reference that shifts the prior from
"architectural limit" to "port defect". Three confounds must be closed or it
proves nothing.

1. **Which weights.** The GGUF in shell history is
   `/root/antirez/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`.
   Determine whether it was converted from the **0731** checkpoint or the
   superseded one (dump GGUF metadata: `general.name`, any source/config keys,
   and whether DSpark keys are present). Different weights ⇒ not like-for-like.
2. **Which probe / length / depth.** Re-run llama.cpp's pass and record: prompt
   tokens, filler type (random-word vs realistic), depth, the exact reply text,
   and the command. A pass at 8K random-word depth 0.1 and a pass at 10.8K
   realistic-filler depth 0.5 are *different claims*. Ideally run our own
   `tools/needle_probe.py 8011 8000 0.1` equivalent against llama.cpp's endpoint.
3. **While you have the metadata**, check `attention.indexer.local_blocks` and
   `attention.indexer.block_size` for T2.

**Record** in `runlogs/T6_llamacpp_confounds.md`. If the GGUF is from the old
checkpoint, downgrade the datum in `README.md` §3d and tell the operator.

---

### T7 — Upstream recipe on sm_89 *(one boot + ~2 min probe)*

**Why.** Tests **H4**. Our launch differs from upstream's validated
`bench/recipes/deepseek-v4-flash/pro6000x2-tp2.yaml` on more than hardware:
`max_num_batched_tokens` 1056 vs 4096, speculation `dspark k=5` vs
`deepseek_mtp k=2`, `max-model-len` 262144 vs 131072, util 0.98 vs 0.92. The
sm_89 attribution holds none of these fixed.

**Steps.** Read that recipe (`git show upstream/main:bench/recipes/deepseek-v4-flash/pro6000x2-tp2.yaml`;
`upstream` remote already exists), launch with its serve args as closely as the
hardware allows, then run the **cheap** gate (~65 s):

```bash
python3 tools/needle_probe.py 8011 8000 0.1     # currently FAILS: "GLACIER"
```

| result | conclusion |
|---|---|
| PASS | the gap is **configuration, not architecture** (H4). Bisect the four differences one at a time — this is the cheapest possible win |
| FAIL | the sm_89 attribution strengthens considerably |

⚠️ Changing speculation may trip the `moe_w2` strict guard (§2). Vary
`num_speculative_tokens` rather than removing speculation. `MTP_TOKENS`/
`SPECULATIVE_CONFIG` are the knobs; expect a Triton recompile and possible
capture-time OOM (§3) — reduce `--kv-cache-memory-bytes` if so.

---

### T8 — Numerically verify the CUDA routing kernels *(unit test, ~30 min)*

**Why.** Low prior for this symptom but it closes the last unverified item in
README §4. Production runs `ops.topk_hash_softplus_sqrt`; the torch fallbacks in
`overlay/.../fused_moe/router/fused_topk_bias_router.py` are only reached on XPU/CPU, so the **CUDA** path has
never been checked numerically against the checkpoint.

**Steps.** New `tools/test_topk_router_vs_checkpoint.py` asserting
kernel ≡ `_topk_softplus_sqrt_torch` (`overlay/.../fused_moe/router/fused_topk_bias_router.py:76`) ≡ checkpoint
`Gate.forward` (`inference/model.py:576`, `F.softplus(scores).sqrt()`) on random
gating outputs, **including the DSv4-Flash bias regime where all
`e_score_correction_bias` ≈ 8.08**. Also add an assert to the hash-MoE torch
fallback (`overlay/.../fused_moe/router/fused_topk_bias_router.py:88`) so it raises instead of silently
falling back to score routing when `input_tokens is None`.

---

## 5. Reporting protocol

After **every** task:
1. Raw artifacts → `runlogs/` with a `T<n>_` prefix.
2. Update `README.md`: the finding, the command that produced it, and move any
   settled hypothesis into §4's eliminated table (with its evidence) or promote
   it in §5.
3. Update this file: mark the task done with a one-line result.
4. Commit. Overlay changes require the §2 sequence.

**Escalate to the operator (don't decide alone) when:**
- T2 confirms H1 and a selection-semantics patch is needed (changes model behaviour).
- T4 returns the H2 branch (the whole framing changes).
- Anything requires deleting data, relaxing the `moe_w2` guard, or shipping
  `index_topk=2048` to production (it deviates from the trained 512 and needs
  GSM8K/GPQA validation first).

---

## 6. If you get stuck

- **Ground truth is `/root/models/DeepSeek-V4-Flash-0731/inference/model.py`.**
  It ships with the weights and is authoritative over any doc here, including
  this one.
- `README.md` §4 lists what is already ruled out **with the evidence**, so you can
  dispute it rather than redo it.
- `tools/README.md` lists every harness with its cost and what it proves.
- The old `STATUS.md`/`BRIEFING.md`/`HANDOFF.md` were deleted (absorbed into
  `README.md`). If you need the original §5.1–5.19 narrative:
  `git show 3678f3b00:docs/dsv4-sm89-longcontext/archive/STATUS.md`. It contains
  **withdrawn claims** — verify against `README.md` first.
- The current best mitigation, if the operator needs long-context exactness
  working *now*: `--hf-overrides '{"index_topk": 2048}'` (6/6 exact to 18.5K,
  fails at 35.8K, unvalidated for quality). **`index_topk = 0` does not mean
  unlimited — it selects nothing.**

## 7. Task status

| task | status | result |
|---|---|---|
| T1 probe position hygiene | DONE | probe reports measured `needle_token_pos` (digits @ 3871 @ ask/8192/0.5, ratio-4 col 967; not 4843/1210); `--abs-pos` places by token (`placed_ok`, ±2 verified 200/2000/3871/6000); `needle_abs_pos_est` removed. Live smoke reproduces FAIL/digits_only. Needs host `pip install tokenizers`. |
| T2 `local_blocks` parity (H1) | DONE | **H1 REFUTED** (CPU). All three = `[window]+[top-k compressed]`; none has local_blocks. Checkpoint `model.py:520` cat([win,topk]), Indexer pure topk (:433); port kernel writes `topk_len+swa_len`; llama.cpp `indexer_local_blocks`=0 default, read ONLY in minimax-m3.cpp — dflash.cpp doesn't reference it. Lead was a MiniMax-M3 llama-arch key. → runlogs/T2_local_blocks_parity.md. Go T3. |
| T3 make trace usable | DONE | trace now: labeled by LAYER (k_cache_prefix), TP-rank-0 only (was doubled), DECODE branch added after persistent_topk/top_k_per_row_decode (own cap `..._DECODE_MAX`), and `..._TRACE_SPAN` (default 2) sweeps ±ratio-4 columns. ast.parse + patches verify 85/85 + check 85/85. No boot. The `sel` flag is the robust headline signal. → unblocks T4. |
| T4 retrace correct column (decisive) | DONE | **H2 CONFIRMED — selection exonerated.** At question-time (the 2421-candidate chunk) the digit col 967 is selected in **20/21 ratio-4 layers** (rank 0 in one) on the FAILING `ask`; digitsonly (passing) 21/21 — digitsonly does NOT select it more. The digit IS attended where it matters; loss is downstream (value/decode). §5.18/H5 selection story retracted at 8K. Image has no trace; used pre-T3 overlay mount (T3 decode-trace mount crashes boot). → runlogs/T4_analysis.md. Prod restored. **H2 branch = re-plan: reopen decode/value path.** |
| T5 cross-chunk compressor (H3) | TODO | |
| T6 llama.cpp confounds | DONE (item 2 deferred) | **llama.cpp pass was on the OLD checkpoint** (GGUF `compress_ratios`=44, no dspark keys, Jul 15) NOT 0731 → not like-for-like; "port defect" prior downgraded, H5 regains weight. `local_blocks` ABSENT in GGUF (T2 confirmed on metadata). Command recorded (ngram-mod spec, q8_0 KV); item 2 (re-run on 0731 GGUF) deferred (needs GPU). → runlogs/T6_llamacpp_confounds.md |
| T7 upstream recipe (H4) | TODO | |
| T8 CUDA routing kernels | TODO | |
