> # ⚠️ SUPERSEDED — DO NOT QUOTE
> Consolidated into [`../README.md`](../README.md) (2026-08-01), which is now the
> single source of truth. This file is kept for provenance only and contains
> **withdrawn claims** — notably the §5.19 indexer rank trace, which measured the
> wrong column (README §2). Verify anything here against README before using it.

# HANDOFF — DeepSeek-V4-Flash-0731 long-context server (read me first)

> **Moved 2026-08-01.** This file previously lived at
> `/root/ktransformers/HANDOFF.md`. It is now maintained here, in the repo whose
> code it describes. Index: [`README.md`](README.md).
> Legacy artifacts NOT migrated (they predate this investigation and are still
> referenced by older docs): the pre-needle throughput/prefill runlogs
> (`20260801T0*.jsonl`) and the numbered SGLang-era patches `0001`-`0023` remain
> under `/root/ktransformers/{runlogs,patches}/`. Everything this investigation
> produced is in `runlogs/` beside this file.


> Paste this whole file (plus `PLAN.md`) to the next agent. It is raw state +
> data + references, not a finished conclusion. The next agent is expected to
> form its own conclusions from the primary sources below. Last updated
> 2026-08-01 ~13:40 UTC. Authoritative continuation file: `PLAN.md` (v3.9).

## 0. What this is, in one paragraph

Two RTX 4090 D (48 GiB, Ada sm89) serve **DeepSeek-V4-Flash-0731** via a
**locally-built vLLM fork (`vLLM-Moet`)** using **exact FP4 expert storage**
expanded in-register to FP8 MMA, **DSpark-5** speculative decoding, **FP8 KV
cache**, TP=2 across both GPUs. The single-stream throughput target (>=40 tok/s)
is met (~62 tok/s). Two open problems: prefill is capped at ~64-80K tokens
(capacity, §5.8), and long-context **exact-value recall** is unreliable.
⚠️ **The old "needle retrieval is broken from ~8K" framing is WRONG** — see §5.
Retrieval WORKS: the model recovers the needle's WORD at every length tested
(2K-48K) and only loses the numeric DIGITS, non-monotonically in length and
deterministically per input. Every numerical path has since been eliminated by
test (FP8 KV dtype, NoPE storage precision, top-k selection, indexer scoring,
compressor semantics, compressed-attention KV store/gather); the current lead is
the **chunked-prefill emulation** of the checkpoint's one-shot semantics. The
user's real working context is **262144 (262K)**, so both problems must be fixed.

## 1. System and runtime (verified 2026-08-01)

- HW: 2× RTX 4090 D, 48 GiB each (49140 MiB total/card; CUDA-visible ~47.37
  GiB = 48521 MiB). Proxmox LXC, 630 GiB RAM, no swap. nvidia-smi works in-LXC.
- Model: `/root/models/DeepSeek-V4-Flash-0731` (~146 GiB; deepseek_v4 arch, 256
  routed experts, 6 active/token, 43 layers, FP4 E2M1 expert weights, bf16
  attention, max_position_embeddings 1,048,576).
- Exact pack/cache: `/root/models/moet-cache-0731-exact` (keyed host `w8x`
  packs, 43/43 layers; GPU holds an expert slot pool, host RAM holds planes).
- Shared JIT cache: `/root/models/moet-cache/jit`.
- Image: `vllm-moet-sm89:v0251`, ID
  `sha256:ab93aeacbffbde3cd30c2c61452ad36f298b1c26eef057546d39a557fab8c183`
  (short `ab93aeacbffb`). Built locally; NOT from a registry.
- vLLM-Moet source HEAD: `55d0a2b50` ("Fuse deterministic MoE unpermute without
  route temporaries") at `/root/autostart/vllm-ds4/vLLM-Moet`.
- Active container: `moet-0731-dspark-exact`, host network, port **8011**,
  RestartPolicy=no (does NOT auto-restart; an OOM kills it until `docker start`).
- Resolved argv (from `docker inspect`), abridged:
  `vllm serve --model /model --served-model-name deepseek-v4-flash auto
  --trust-remote-code --kv-cache-dtype fp8 --block-size 256 --max-model-len
  262144 --gpu-memory-utilization 0.98 --max-num-batched-tokens 1056
  --max-num-seqs 3 --tokenizer-mode deepseek_v4 --no-scheduler-reserve-full-isl
  --tool-call-parser deepseek_v4 --enable-auto-tool-choice
  --reasoning-parser deepseek_v4 --tensor-parallel-size 2
  --disable-custom-all-reduce --no-enable-prefix-caching --port 8001
  [--kv-cache-memory-bytes 4846832640] [--cudagraph-metrics]
  [SPECULATIVE_CONFIG={"method":"dspark","num_speculative_tokens":5,
  "dspark_scheduler":false}]
  [CUDAGRAPH_SIZES=1,2,4,6,8,12,18]`.
- **KV budget (boot log, authoritative):** `GPU KV packed blocks: pool=4836,
  allocatable=4835 (one null block reserved), max-request=1480, per-group=
  (1024,20,20,267,149); GPU KV cache size 856,573 tokens; Maximum concurrency
  for 262,144 tokens/request: 3.27x`.
- ⚠️ **`--kv-cache-memory-bytes 4846832640` BYPASSES `--gpu-memory-utilization`**
  (boot log verbatim: "reserved 4.51 GiB ... skipped memory profiling. This does
  not respect the gpu_memory_utilization config"). Lowering UTIL alone is a
  no-op; headroom must come from reducing `--kv-cache-memory-bytes` or
  `--max-num-batched-tokens`.

## 2. File map (places of interest)

- `docs/dsv4-sm89-longcontext/STATUS.md` — **authoritative status/NEXT/decisions**.
- `docs/dsv4-sm89-longcontext/HANDOFF.md` — this file.
- `/root/ktransformers/tools/vllm_dsv4_chat_benchmark.py` — throughput/
  prefill/capacity probe. **Patched this session** to emit `gpu_telemetry`
  (per-GPU `memory_used_mib` max) on FAILURE and survive EngineCore-death.
- `/root/ktransformers/runlogs/` — all raw artifacts (see §6).
- `/root/autostart/vllm-ds4/vLLM-Moet/` — the vLLM fork source. HEAD `55d0a2b50`.
- `/root/autostart/vllm-ds4/vLLM-Moet/tools/needle_probe.py` — quick needle
  (RANDOM-WORDS filler; does NOT reproduce the failure — see its docstring).
- `/root/autostart/vllm-ds4/vLLM-Moet/tools/needle_full.py` — needle + reasoning.
- `/root/autostart/vllm-ds4/vLLM-Moet/bench/runner/probes.py` — **the faithful
  probe runner**. `--probe needle_sweep` uses REALISTIC filler (concatenated
  repo text) which DOES reproduce the failure. Also has `derailment`,
  `decode`, `prefill`, `coherence`, `quality` probes.
- `/root/autostart/vllm-ds4/vLLM-Moet/bench/suites/needle_sweep.yaml` — suite
  doc; calls needle_sweep "the before/after metric for the FP8-delta kernel
  fix". Confirmed symptom (its words): "CORRECT at 2K, WRONG/GARBLED at 131K".
- Exact kernel sources to inspect for the retrieval root cause:
  - `…/vllm/model_executor/layers/quantization/utils/moe_w2_cubit.py`
    (w8 FP8-delta expert tier; self-tests clean — worst_rel 2.517e-3).
  - `…/vllm/model_executor/layers/quantization/utils/moe_w2_delta.py`
    (w8x tiered store / host planes).
  - The MLA / sparse-attention path + the FP8 KV cache scaling (the launch
    sets `--kv-cache-dtype fp8`; check whether a KV scale other than 1.0 is
    computed/needed for DSv4).
- `/root/.config/hf.env`, `/root/autostart/vllm-ds4/.../docker/serve_sm89_ds4.sh`
  — launch env + orchestrator (Mode A for the 0731 model).
- `/root/autostart/CLAUDE.md` — broader ops notes (the qwen/gemma production
  stack). Note: its fp8-KV-scaling guidance ("leave at 1.0") is for Qwen, not
  necessarily DSv4.

## 3. Commit / SHA graph

Preservation commits on branch `fix/kt-kernel-install-pep668-venv`
(`/root/ktransformers`), most recent first:
- `24a432d` [docs](plan): record fused image launch state
- `a9b4763` [fix]: preserve bounded deterministic MoE unpermute  ← patch 0023
- `15c9c01` [fix]: preserve packed KV logging fix                ← patch 0022
- `b2063f1` [fix]: preserve graph-18 source patches
- `f0e2df8` [fix]: preserve graph-18 sentinel safety fixes       ← 0020/0021
- `b642bba` [docs]: condense handoff and preserve vLLM fixes
- `aed02e8` [docs]: record active exact cache cold build
- `afee6a2` [docs]: preserve exact FP4 cache implementation      ← 0016/0017

vLLM-Moet source commits (at `/root/autostart/vllm-ds4/vLLM-Moet`):
- `55d0a2b50` Fuse deterministic MoE unpermute without route temporaries
- `2d53905de` Fix hashability of packed KV once-log arguments
- `23ee20ecd` Filter speculative sentinel routes before MoE alignment
- `a4a11205b` (mark-seen), `729729ff7` (exact w8x heat + packed-KV capacity),
  `b6a1750eb` (exact-cache impl), `b62d58639` (experiment passthrough).

Preserved patches in `/root/ktransformers/patches/`: 0016 (exact fp4 cache),
0017 (launch banner), 0018 (CLI passthrough), 0019 (exact heat + packed-KV
capacity fix), 0020 (graph-safe heat sentinel), 0021 (MoE alignment/eager
sentinel), 0022 (packed-KV once-log hashability), **0023 (bounded deterministic
fused unpermute; SHA-256
`fb2832ccedee391ed981bd7dd7c5a17a1369d7ee910e839d618ed8a38562487f`)**.

> This session's commit (PLAN v3.6 + probe patch + sweep/needle runlogs + this
> file) is the new HEAD of `fix/kt-kernel-install-pep668-venv`. See git log.

## 4. What is PROVEN (don't re-litigate)

1. **Fused-unpermute fix (`55d0a2b50`) works.** The §5.6 T=1044 transient OOM
   (413 MiB epilogue) is gone: a 100K prefill ran **677 s past the old death
   point with zero assertions** before a *different* OOM. Reduced gates pass;
   0 MiB allocator growth vs 413 MiB old; max BF16 diff one ULP; bit-identical
   repeat.
2. **Single-stream throughput PASSES.** median **62.1 decode tok/s** @256
   tokens, both GPUs ~98-99% util, DSpark ~36-37% acceptance.
3. **3×256 graph-18 concurrent PASSES, no crash.** median **32.9 tok/s/stream**,
   aggregate 73 tok/s, 9/9 coherent. The earlier `dstIndex<dstAddDimSize`
   crash (negative DSpark padding IDs) is fixed by `a4a11205b`+`23ee20ecd`.
4. **Long-prefill ceiling is allocator-state, not prompt size.** Fresh-boot
   prefill sweep: 16/32/48/64K all PASS; **80K OOMs at the SAME nvidia-smi
   peak (48,508 MiB) where 64K passed** → the constraint is a specific
   deep-context allocation + PyTorch cache fragmentation, not the memory
   total. Envelope ≈ 64-80K; the prior 100K OOM was aggravated by prior
   decode benchmarks dirtying the allocator.

## 5. The OPEN problem — REFRAMED 2026-08-01: retrieval works, DIGITS are lost

**Do not carry forward the old "retrieval breaks at 8K" framing.** Decoding the
answers against the deterministic needle codes shows the needle is found at
EVERY length; only its fine-grained numeric content is lost:

| target | true code | model answered | word | digits |
|-------:|-----------|----------------|:----:|:------:|
| 8192   | `PELICAN-1605` | `PELICAN PELICAN PELICAN…` | ✅ | ❌ |
| 16384  | `PELICAN-3738` | `PELICAN`                   | ✅ | ❌ |
| 32768  | `CYPRESS-0527` | `CYPHER`                    | ~  | ❌ |
| 49152  | `LANTERN-2037` | `LANTERN-9-9-9-9-9…`        | ✅ | ❌ |

Top-20 logprobs at the digit position (the decisive evidence):
- 4K PASS: `538` at logprob **-0.000** (p≈1.000); runner-up **-14.4**. Certain.
- 8K FAIL: `123` at **-0.765** (p≈0.47); a diffuse spread of unrelated 3-digit
  priors, and the true token `160` is **absent from the top-20**.
- WORD tokens sit at p≈1.000 at BOTH lengths.

Error *character* is the key signal: adding redundancy/salience turns failures
into NEAR-MISSES, not priors — `verbose` → `1606` vs `1605`; `repeat` → `305`;
`words` → `ZEPHYR-ORYX-CORYX` vs `…-CYPRESS`. Plain digits fall back to the
generic prior `1234`. Deterministic per input (bit-identical on repeat).

### REFUTED — do not re-litigate (all measured, not argued)
1. **FP8 KV cache is not the cause.** Upstream's `pro6000x2-tp2` recipe passes
   `needle @121k tokens` with the SAME `--kv-cache-dtype fp8`; it differs by
   running native Blackwell kernels. The defect is in the sm89 port.
2. **Fixed absolute needle position (~2048).** At total pt 2864 the needle
   passes at abs positions 1146→2721 (7/7); at pt 9685 it FAILS at abs 1695.
3. **Fixed total length (~8192).** NON-monotonic: pt 8316 **PASSES** while
   pt 5100 and 6650 FAIL. There is no clean cutoff.
4. **Top-k selection kernels.** `tools/test_indexer_topk_selection.py`: with
   all high scores beyond index 512, `persistent_topk` (the op sm89 decode
   actually calls) returns 512/512 correct, zero overlap with the first 512.
5. **Ragged indexer-logits fallback.** `_torch_fp8_mqa_logits` implements
   `sum_h relu(q·kᵀ)·w` with correct k dequant and `[ks,ke)` masking.
6. **Paged indexer-logits Triton port**: its suite passes (9 passed, 5 skipped).
7. **Per-token scalars** (`q_scale`/`softmax_scale`/`head_scale`) are
   ranking-neutral — top-k is per query ROW.

### Quantified (CPU-only, `tools/test_nope_needle_recall.py`, seconds/run)
| encoding | needle value readback rel-err | needle still argmax |
|----------|------------------------------:|:-------------------:|
| FP8-E4M3 | **2.48e-2** (flat in N)       | yes |
| INT8     | **8.0e-3** (flat in N)        | yes |

Worst attention-output rel-err E4M3 3.22e-2 vs INT8 8.02e-3 → **4.0× better at
identical bytes/token**. So E4M3 imposes a constant ~2.5% readback floor that
does NOT itself create a length threshold — matching the non-monotonic
behaviour. A constant ~2.5% error sits at the edge of resolving a specific
4-digit token, so pass/fail turns on the individual needle.

### `VLLM_DSV4_KV_INT8=1`: TESTED and ELIMINATED
The knob purpose-built for this bug ("the higher-fidelity swap for the
long-context needle-retrieval regression") does **nothing** for it. Boot
confirmed active (`nope_int8=True`); identical seeds → byte-identical prompts:

| actual pt | FP8-E4M3 | INT8 | true code |
|----------:|----------|------|-----------|
| 2864 | PASS `TUNDRA-3711`  | PASS `TUNDRA-3711`  | TUNDRA-3711 |
| 5100 | FAIL `FALCON`       | FAIL `FALCON`       | FALCON-1042 |
| 6650 | FAIL `PUMICE-7541`  | FAIL `PUMICE-7542`  | PUMICE-7544 |
| 8316 | PASS `SAFFRON-7888` | PASS `SAFFRON-7888` | SAFFRON-7888 |
| 9376 | FAIL `CYPRESS-0120` | FAIL `CYPRESS-1234` | CYPRESS-0074 |
| 9686 | FAIL `PELICAN-1234` | FAIL `PELICAN-1234` | PELICAN-1605 |

**6/6 identical.** A 3-4× readback-precision gain fixed nothing ⇒ **NoPE KV
storage precision is not the limiting factor.** This also de-prioritizes bf16
KV (2× density cost for the same quantity INT8 already improved 4× for free).

⚠️ Operational: INT8 is currently **incompatible with CUDA graphs** here.
Toggling the constexpr creates a Triton variant absent from the JIT cache, so
it compiles INSIDE capture → `CUDA error: operation not permitted when stream
is capturing`. With the default 4.51 GiB KV it died even earlier with
`Triton Error [CUDA]: out of memory` during capture (~440 MiB headroom). The
A/B above ran `--enforce-eager` + `--kv-cache-memory-bytes 2147483648`.
Also learned: a **2 GiB KV pool boots and still holds a full 262K request**
(min ≈1481 blocks ≈1.484 GiB) — useful for the §5.8 headroom work.

### Compressor: TESTED and ELIMINATED; arch gate FIXED (shipped)
`tools/test_compressor_vs_checkpoint_ref.py` (CPU-only, seconds) compares our
compressor against a **third, independent transcription of the checkpoint's own
`Compressor`** — necessary because the existing suite validates the fused kernel
against a reference in the SAME file, so a shared misreading would pass. It
checks the ratio-4 overlapping window (4 entries from the PREVIOUS block via the
first half of the `2*head_dim` projection, then 4 from the current block via the
second half), the softmax axis, block 0's `-inf`/`0` padding, APE indexing by
within-block position, and the RoPE position index (`b*ratio`).
**Result: AGREE at ~2.6e-3 rel** (shared bf16-store rounding floor) across
ratio 4 and the ratio-128 non-overlap branch. The compressor is not the defect.

**Fixed a real bug while getting there.** `has_cutedsl()` is a package check
only and `cutlass` IS installed, so `dequantize_and_gather_k_cache` dispatched to
`DequantGatherKCacheKernel`, which has no pre-SM90 lowering. Now gated on
`has_device_capability(90)` with the arch-portable Triton fallback
(`overlay/vllm/vllm/models/deepseek_v4/common/ops/cache_utils.py`, commit
`8377e2db1`; patches `--verify 85/85`, `--check 85/85`).

That gate had been hiding **the only sm89 validation of the compressed-attention
K-cache store/gather path** — the path holding the values retrieval reads back.
Before: 19 passed / 17 failed, all CuTeDSL sm89 compile errors. After: 11 of the
17 pass, including `test_deepseek_v4_attention_quant_cache_roundtrip` 8/8 and
`test_dequantize_and_gather_k_cache` 2/2 → **that path is numerically correct on
sm89, demonstrated rather than assumed.** The remaining 6
(`test_fused_kv_insert_indexer[use_fp4=True]`) are the datacenter-Blackwell-only
MXFP4 indexer cache (ptxas rejects its PTX for sm_89, exit 255); now skipped with
that reason via an overlay of the test. **Suite on sm89: 30 passed, 6 skipped,
0 failed.**

### CURRENT leading hypothesis: CHUNKED-PREFILL emulation
Everything numerical has now been eliminated: FP8 KV dtype, NoPE storage
precision, top-k selection, indexer scoring (ragged + paged), per-token scalars,
compressor semantics, and the compressed-attention KV store/gather. What has
NOT been examined is where vLLM necessarily **departs** from the checkpoint's
reference: the reference prefills a whole sequence in one shot
(`start_pos == 0`, with `kv` held exactly and `kv_state`/`score_state` carrying
the `seqlen % ratio` remainder), whereas we run **chunked prefill at
`--max-num-batched-tokens 1056`**. Suspects:
- the ring-buffer `window_size` write (`kv_cache[:, start_pos % win]`) across
  chunk boundaries,
- the `remainder` carry (a chunk boundary that is not a multiple of
  `compress_ratio` leaves partial state that the next chunk must consume),
- the SWA + compressed two-segment split that shares ONE softmax.
A boundary-sensitive bug here would make digit survival depend on where the
needle falls relative to chunk/compression boundaries — exactly the observed
content-dependent, deterministic, NON-monotonic pattern (pt 8316 passes while
5100 and 6650 fail). Cheap test first: drive the compressor/state path over
multiple chunk splits of the SAME sequence and require chunk-split invariance
(one 8K chunk vs 8×1056 chunks must produce identical compressed entries).

### Latent arch bug found (not this root cause, but it will bite)
`has_cutedsl()` (`vllm/vllm/utils/import_utils.py:547`) is
`_has_module("cutlass")` — a PACKAGE check with **no architecture check**. The
package IS installed, so `dequantize_and_gather_k_cache`
(`…/deepseek_v4/common/ops/cache_utils.py:403`) dispatches to a CuTeDSL kernel
that cannot compile for `cubin-chip=sm_89`. That is why 17/36 of
`vllm/tests/kernels/test_compressor_kv_cache.py` fail on sm89 (all CuTeDSL
compile errors). Serving does not reach that path today (it would hard-crash);
the gate should also test device capability.

### Architecture ground truth (checkpoint's OWN reference)
`/root/models/DeepSeek-V4-Flash-0731/inference/model.py` is authoritative: the
KV cache is a RING BUFFER of only `window_size` tokens; older tokens are
reachable ONLY via learned gated-pooling compressed entries. Selection is
`topk(min(index_topk, end_pos // ratio))` over COMPRESSED entries, and an
Indexer exists ONLY on `compress_ratio == 4` layers — ratio-128 layers use
positional order (all causal entries) by design. This checkpoint: **21 layers
ratio 4, 20 ratio 128, 5 ratio 0**. Hence top-512 is a no-op below 2048 tokens
on ratio-4 layers and below 65536 on ratio-128 layers — which is why the coarse
WORD always survives (ratio-128 layers see the whole context) but exact digits
do not.

## 6. Raw artifacts (read these)

- Prefill sweep: `runlogs/sweep-prefill-{16000,32000,48000,64000,80000}-*.jsonl`
  (each has per-run gpu_telemetry with memory max).
- Needle sweep (2K-48K): `docs/dsv4-sm89-longcontext/runlogs/needle_sweep_20260801T103931Z.{json,log}`.
- Needle onset (4K/8K/12K): `docs/dsv4-sm89-longcontext/runlogs/needle_onset_20260801T110302Z.{json,log}`.
- Prior §5.5 fixed-image serving evidence: `runlogs/20260801T0730Z-…-3x64.jsonl`,
  `…0731Z-…-3x256-shapewarm.jsonl`, `…0732Z-…-3x256-warm3.jsonl`,
  `…0734Z-…-single256-warm3.jsonl`, `…0736Z-…-prefill100k-shapewarm.jsonl`.
- 100K OOM evidence (raw engine traceback, `EngineDeadError`, the
  `CUDA out of memory: Tried to allocate 72.00 MiB ... GPU0 ... 72.88 MiB free
  ... 47.16 GiB in use` line): `docker logs moet-0731-dspark-exact` around
  `08-01 09:46:58` (and the §5.7 100K attempt around `08-01 07:3x`).

## 7. Reproducible commands

```bash
# Status
docker ps -a --filter name=moet-0731-dspark-exact
curl -fsS http://127.0.0.1:8011/health     # 200 = up
curl -fsS http://127.0.0.1:8011/v1/models
docker logs --tail 200 moet-0731-dspark-exact

# Restart after an OOM (config baked into the container; reproduces exact KV4836/exact30)
docker start moet-0731-dspark-exact   # ~4.5 min boot (weight load + graph capture)

# Wait for ready
for i in $(seq 1 90); do curl -fsS --max-time 5 http://127.0.0.1:8011/health \
  >/dev/null 2>&1 && { echo READY; break; }; sleep 10; done

# Serving gates
cd /root/ktransformers
python3 tools/vllm_dsv4_chat_benchmark.py single    --warmups 0 --runs 3 --max-tokens 256
python3 tools/vllm_dsv4_chat_benchmark.py concurrent --warmups 0 --runs 3 --streams 3 --max-tokens 256

# Prefill sweep (capacity envelope)
for S in 16000 32000 48000 64000 80000; do
  python3 tools/vllm_dsv4_chat_benchmark.py prefill \
    --warmups 0 --runs 1 --streams 1 --target-prompt-tokens $S --max-tokens 1 --timeout 3600
done   # 80K will OOM-kill the engine; docker start to continue

# Needle retrieval (the correctness metric) — REALISTIC filler, depth 0.5
cd /root/autostart/vllm-ds4/vLLM-Moet
python3 -u bench/runner/probes.py --probe needle_sweep \
  --endpoint http://127.0.0.1:8011 --model deepseek-v4-flash \
  --lengths 2048,4096,8192,12288,16384 --runs 1 --timeout 1800 \
  --out docs/dsv4-sm89-longcontext/runlogs/needle_<stamp>.json
# NOTE: do NOT exceed ~48K-64K on a single sweep or a mid-sweep OOM kills the
# engine and wastes the remaining points. Cap lengths inside the envelope.

# DECISIVE retrieval test (UNRUN): bf16 KV restart + 8K needle.
# Requires re-launch via serve_sm89_ds4.sh with KV_CACHE_DTYPE overridden to bf16
# (or --kv-cache-dtype bf16), keeping exact30 + the 4836-block budget intent.
```

## 8. Discipline / invariants (from PLAN.md §9)

- First request after any restart is excluded (cold compile / `cc1plus`); never
  a throughput sample.
- Cache misses on the exact path are STRICT (fetch, replay, converge) — never
  emit from a residual miss or the base fallback.
- Reuse keyed packs + JIT caches; never restamp a cache from a different
  checkpoint/TP/quantizer/shape.
- Runtime/package edits must be committed in vLLM-Moet AND exported to
  `/root/ktransformers/patches/` (reinstall overwrites live changes).
- Stop after a decisive failure; prefer reduced probes before a full restart.
- Do not delete caches or unrelated dirty/untracked files.

## 9. Suggested first moves for the next agent

**Work the cheap loop first.** Boots cost ~6-10 min and each long-context probe
~30-110 s; the two new CPU/1-GPU harnesses in `tools/` answer numerical
questions in seconds. Prefer adding to them over another restart.

1. **Test chunk-split invariance of the compressor/state path** — the current
   lead (§5). Require that compressing the SAME sequence as one 8K chunk and as
   8×1056 chunks yields byte-identical compressed entries, and that a chunk
   boundary landing mid-`compress_ratio` (i.e. not a multiple of 4) carries the
   remainder correctly. Pattern to copy:
   `tools/test_compressor_vs_checkpoint_ref.py` (CPU, seconds, no server).
   This is the one place vLLM must depart from the checkpoint's one-shot
   reference, and every numerical path has already been eliminated.
2. **Get direct observability instead of inferring from answers.** Add an
   env-gated trace in `overlay/vllm/vllm/model_executor/layers/
   sparse_attn_indexer.py` dumping, per ratio-4 layer for one request:
   `end_pos // ratio` (candidate count), `k_select`, whether selection was a
   no-op, and the score RANK of the compressed entry holding a known needle
   position. Every remaining question is currently answered by 100 s black-box
   probes; this converts them into direct measurements.
3. **`--hf-overrides '{"index_topk": 2048}'`** is available and untried: raises
   ratio-4 selectivity coverage 4×. Costs attention compute/bandwidth, not VRAM.
4. **Fix the `has_cutedsl()` arch gate** (below) — a latent sm89 hard-crash, and
   it currently blocks 17/36 compressor tests you will want for step 1.
5. **Fix INT8-under-CUDA-graphs** if the knob is ever needed again: pre-warm the
   Triton variant before capture (see §5).
6. The ~64-80K prefill ceiling still needs a headroom cut
   (`--kv-cache-memory-bytes` or `--max-num-batched-tokens`); SECONDARY to digit
   fidelity. The INT8 runs showed a 2 GiB KV pool boots and still holds a full
   262K request.
7. Read `PLAN.md` end-to-end (esp. new §§5.10-5.12) for the full decision
   history and rejected directions (CPU AMXINT4, hybrid offload, SGLang
   MXFP4/Marlin — none support sm89).

### Reference implementations available locally (use them as ground truth)
- **`/root/models/DeepSeek-V4-Flash-0731/inference/model.py`** — DeepSeek's own
  reference for this exact checkpoint. Authoritative for indexer/compressor/
  ring-buffer semantics.
- **`~/llama.cpp`** — a DFlash+DSpark-capable build that reportedly does not
  degenerate and passes the needle test (`src/models/dflash.cpp`; indexer
  params incl. `indexer.block_size` / `indexer.local_blocks` in
  `src/llama-arch.cpp`). Good for cross-checking algorithm shape.
- **`upstream/main`** (`kacper-daftcode/vLLM-Moet`) — passes needle @121K on
  Blackwell with `--kv-cache-dtype fp8`. NOTE: it contains **none** of the sm89
  Triton kernels (they are 100% local), so it is a behavioural reference, not a
  code diff target. We are 127 ahead / 11 behind; the 11 are docs/recipes only.
