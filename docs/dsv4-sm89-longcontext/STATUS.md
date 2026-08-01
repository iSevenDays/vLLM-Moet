# DeepSeek V4 Flash 0731 performance handoff

> **Moved 2026-08-01.** This file previously lived at
> `/root/ktransformers/PLAN.md`. It is now maintained here, in the repo whose
> code it describes. Index: [`README.md`](README.md).
> Legacy artifacts NOT migrated (they predate this investigation and are still
> referenced by older docs): the pre-needle throughput/prefill runlogs
> (`20260801T0*.jsonl`) and the numbered SGLang-era patches `0001`-`0023` remain
> under `/root/ktransformers/{runlogs,patches}/`. Everything this investigation
> produced is in `runlogs/` beside this file.


Last updated: 2026-08-01 13:40 UTC

Version: 3.9

Status: serving gates PASS (single 62 tok/s; 3x256 graph-18 33 tok/s/stream).
Prefill envelope ~64-80K (5.8: ceiling is allocator-state, not prompt size).
**NEEDLE "RETRIEVAL" IS REFRAMED (5.10 — supersedes 5.9's framing).** The model
retrieves the needle's WORD at EVERY length tested (2K-48K); what it loses is
the DIGITS. Top-20 logprobs: at 4K the digit token is emitted at p≈1.000 with
the runner-up 14 nats down; at 8K the distribution collapses to p≈0.47 over
unrelated 3-digit priors and the true token is absent from the top-20. Errors
become NEAR-MISSES (1606 for 1605) once redundancy/salience is added → a
borderline numerical-PRECISION signature, deterministic per input.
REFUTED and not to be re-litigated: FP8 KV as the cause (upstream passes needle
@121K with the same flag, on native Blackwell kernels); a fixed absolute needle
position (~2048); a fixed total-length cutoff (~8192 — response is
NON-monotonic: pt 8316 passes while 5100 and 6650 fail); the top-k selection
kernels (unit-tested correct); the indexer scoring path; and — measured in 5.11 —
**`VLLM_DSV4_KV_INT8=1`, which changes nothing (6/6 identical), so NoPE KV
storage precision is NOT the limiting factor either.**
Also eliminated (5.13): the **ratio-4 compressor semantics** (differential-tested
against a third, independent transcription of the checkpoint's own Compressor)
and the **compressed-attention K-cache store/gather path** (now covered by
passing sm89 tests after fixing an arch-blind CuTeDSL gate — a real shipped bug).
CURRENT LEAD (5.13): the **chunked-prefill emulation** of the checkpoint's
one-shot semantics — the ring-buffer `window_size` handling and the `remainder`
carry across `--max-num-batched-tokens 1056` chunk boundaries — i.e. where vLLM
necessarily departs from the whole-sequence reference.
Prefer the new second-scale `tools/` harnesses over 6-10 min server boots.

This is the authoritative continuation file. Keep it concise: record only the
current system, reproducible commands, measurements, decisions, invariants, and
NEXT queue. Detailed history remains in `README_LOCAL.md`,
`CONTEXT_FOR_RESEARCH_V2.md`, `RESEARCH_FINDINGS_DUAL_GPU.md`, git history, and
`runlogs/`.

## 1. Goal and acceptance criteria

- Use both RTX 4090 D 48-GiB GPUs effectively.
- DeepSeek-V4-Flash-0731 with original checkpoint FP4 expert information,
  converted in-register to native Ada E4M3/FP8 MMA. Do not silently substitute
  the lossy IQ2/base path for the exact path.
- DSpark speculative decoding enabled where it improves end-to-end throughput.
- At least 40 output tok/s for one warmed request. This is already exceeded.
- Eventually support three concurrent streams, each up to about 262K context,
  with FP8 KV and more than 100K-token prefill. Capacity must be proved from the
  scheduler-visible KV pool before attempting the destructive long-context test.
- Exclude the first request after every restart. It warms DSpark/Triton/Inductor
  caches and can launch `cc1plus`; it is a correctness/compile gate, never a
  throughput sample.

## 2. Hardware and runtime

- GPUs: 2 x RTX 4090 D, 48 GiB each, Ada SM89.
- Interconnect: PCIe/P2P-capable host; current TP=2 path deliberately uses
  PyNCCL (`CUSTOM_ALL_REDUCE=0`) to avoid relying on vLLM custom all-reduce.
- Model: `/root/models/DeepSeek-V4-Flash-0731`.
- Main workspace: `/root/ktransformers`.
- vLLM-Moet workspace: `/root/autostart/vllm-ds4/vLLM-Moet`.
- Exact pack/cache: `/root/models/moet-cache-0731-exact`.
- Shared JIT cache: `/root/models/moet-cache/jit`.
- Image: `vllm-moet-sm89:v0251`, image ID `ab93aeacbffb...`.
- vLLM-Moet source: `55d0a2b50`; exact-cache implementation: `b6a1750eb`.
- Main preservation commits: `afee6a2`, `aed02e8`, `b642bba`.
- Active launch (2026-08-01 13:20 UTC): container `moet-0731-dspark-exact`,
  port 8011, the §4 FP8 baseline config (KV 4,846,832,640 bytes, graph-18,
  DSpark-5, exact30). Restored and restarted after the 5.11 INT8 A/B; readiness
  verified by `curl :8011/health`. `RestartPolicy=no` — an OOM leaves it down
  until `docker start`. The INT8 test containers were removed; the baseline
  container is the only one for this model.

## 3. Exact FP4-storage design already validated

- Original checkpoint E2M1 nibbles and UE8M0 scales are persisted in a keyed
  host pack (`w8x`). GPU cache slots retain FP4 storage, 6.375 MiB/expert slot.
- The SM89 kernel expands FP4 to E4M3 in registers and executes native Ada FP8
  tensor-core MMA. This answers the “why not FP8?” question: arithmetic is FP8;
  compact FP4 storage preserves checkpoint fidelity and increases residency.
- Cache misses are strict: fetch, replay, and require convergence. No output may
  be emitted from a residual miss or the lower-quality base fallback.
- Reduced exact smoke: relative error 2.411e-2, cosine 0.999819.
- Host regression: relative error 1.540e-2, cosine 0.999912.
- 43/43 layer packs exist and are reused. Do not rerun checkpoint conversion
  unless cache metadata/hash/shape invalidation rejects a pack.
- Preserved patches:
  - `patches/0016-vllm-moet-exact-fp4-cache.patch`
  - `patches/0017-vllm-moet-exact-launch-banner.patch`
- Exact `w8x` heat dumping plus metadata/shape invalidation is fixed and
  CPU-tested in commit `729729ff7`; it is live in the rebuilt image. The pack
  itself is already persistent and reusable.

## 4. Current exact/DSpark configuration

The current launch uses:

```bash
cd /root/autostart/vllm-ds4/vLLM-Moet
MODEL=/root/models/DeepSeek-V4-Flash-0731 \
CACHE=/root/models/moet-cache-0731-exact \
JIT_CACHE=/root/models/moet-cache/jit \
STORE=/root/models/moet-cache-0731-exact/packs \
NAME=moet-0731-dspark-exact PORT=8011 NETWORK=host RESTART=no \
RESIDENCY=exact EXACT_GB=30 ARENA_GB=40 MEM_GB=428 SCALE_REFIT=0 \
TP=2 GPUS='"device=0,1"' CUSTOM_ALL_REDUCE=0 \
MAXLEN=262144 UTIL=0.98 BATCHED_TOKENS=1056 NUM_SEQS=3 \
CUDAGRAPH_SIZES=1,2,4,6,8,12,18 PREFIX_CACHING=0 MTP_TOKENS=0 \
SPECULATIVE_CONFIG='{"method":"dspark","num_speculative_tokens":5,"dspark_scheduler":false}' \
EXTRA_DOCKER_ENV='-e VLLM_MOE_W2_DELTA_TRACE=1 -e VLLM_MOE_W2_KPI_EVERY=100' \
EXTRA_ARGS='--cudagraph-metrics --kv-cache-memory-bytes 4846832640' \
READY_TIMEOUT_S=1800 ./docker/serve_sm89_ds4.sh
```

Resolved properties:

- TP=2; both GPUs are active and each owns one rank.
- Exact GPU pool: 4,818 slots/rank, approximately 29.99 GiB/rank.
- Host arena: 6,425 slots/rank, 40 GiB/rank.
- Exact DSpark-5 conservative one-pass floor for three sequences: 4,644 slots;
  only 174 GPU slots remain beyond the floor.
- Graph-18 A/B captures scheduled-token sizes 1, 2, 4, 6, 8, 12, and 18.
- Prefix caching disabled for correctness isolation.
- Boot completed, CUDA graphs captured, HTTP health passed, and both cards used
  about 47.9 GiB. Corrected boot captured PIECEWISE 7/7, target FULL 3/3,
  and DSpark FULL 3/3 in 9 seconds / 0.56 GiB per rank.
- Never combine SGLang-only `--swa-full-tokens-ratio 0.1` with this vLLM launch.
  Retain ratio 0.1 for any future SGLang DSV4 comparison.

## 5. Verified performance

### 5.1 Single warmed request: PASS

Probe:

```bash
cd /root/ktransformers
python3 tools/vllm_dsv4_chat_benchmark.py single \
  --warmups 0 --runs 4 --max-tokens 256 --timeout 3600
```

- Excluded cold/compile request: coherent, 67.16 decode tok/s, TTFT 6.74 s;
  `cc1plus`/runtime-kernel compilation observed. It is not a benchmark sample.
- Five valid warmed rates including the semantic gate:
  64.387, 67.640, 57.508, 65.636, 62.303 decode tok/s.
- Median: 64.387 decode tok/s; 53.804 end-to-end tok/s; TTFT 0.717 s.
- Both GPUs were active: warmed-four utilization means 89.6% / 86.2%, active
  p50 about 98% / 99%.
- DSpark over the four repeated samples: 1,855 proposed, 655 accepted; 35.31%
  token acceptance. Fifth proposal position is rarely accepted.
- Artifacts:
  - `runlogs/20260801T060649Z-vllm-exact-dspark-warm-gate.jsonl`
  - `runlogs/20260801T061145Z-vllm-exact-dspark-warm4.jsonl`

Conclusion: the user’s >=40 tok/s single-request target is decisively passed.

### 5.2 Three concurrent warmed requests: REJECT current graph set

Commands:

```bash
python3 tools/vllm_dsv4_chat_benchmark.py concurrent \
  --warmups 0 --runs 3 --streams 3 --max-tokens 256 --timeout 3600

python3 tools/vllm_dsv4_chat_benchmark.py concurrent \
  --warmups 0 --runs 1 --streams 3 --max-tokens 64 --timeout 3600
```

- First 256-token trio: 3.233 / 3.290 / 3.458 decode tok/s per stream. Further
  repetitions were interrupted because the configuration was already rejected.
- Complete 64-token telemetry trio: 2.830 / 2.938 / 3.257 per stream;
  aggregate completion rate 7.546 tok/s; TTFT 3.989 s.
- GPUs were underutilized, not compute/HBM saturated: GPU0 mean/p50/p95
  41.4/39/73%; GPU1 54.6/57/89%; low memory-traffic utilization and about
  100/110 W.
- Exact cache was full at 4,818/4,818. Per trace window there were roughly
  100-175 promotions/evictions/rank; RAM fetches cost about 4.5-7.9 ms and
  strict replay amplifies the synchronous tail despite 96-99% reported hits.
- Artifacts:
  - `runlogs/20260801T061230Z-vllm-exact-dspark-concurrent3-rejected.jsonl`
  - `runlogs/20260801T061600Z-vllm-exact-dspark-concurrent3-telemetry.jsonl`

Most important discriminator: vLLM chooses CUDA graphs using the total number
of scheduled tokens, not request count. DSpark-5 schedules up to six tokens per
request. One request fits graph size 8; three can require 18, above the largest
captured graph, forcing eager/piecewise execution. This directly matches the
low concurrent GPU utilization. Test graph 18 before changing quantization,
pool size, or DSpark width.

### 5.3 Graph 18 A/B: performance recovery, then fatal repeat failure

- Restart with graphs `[1,2,4,6,8,12,18]` succeeded at exact30/UTIL 0.98.
  Capture took 13 s and 0.53 GiB/rank versus 0.31 GiB with `[1,2,4,8]`.
  Both packs were reused; no conversion. Both GPUs used 48,114 MiB at ready.
- Excluded first 64-token request: coherent; TTFT 24.35 s and multiple first-
  shape JIT warnings. Measured short single: 74.82 decode tok/s, coherent.
- Three x 64: aggregate 33.62 tok/s versus 7.55 baseline (4.45x); median
  per-request decode 32.40 tok/s. One stream lagged at 17.09 tok/s.
- First three x 256: coherent; 71.92 aggregate tok/s; per-request decode
  26.42/32.39/33.03; both GPU utilization p50 99%. This sample also compiled
  `_ada_fp8_grouped_mm_kernel`, so it is a shape warm-up, not a steady result.
- The next three x 256 streamed coherent partial text, then rank 0 hit CUDA
  `Indexing.cu:indexFuncLargeIndex` assertion `dstIndex < dstAddDimSize`.
  Final usage/finish frames were lost; the next admission returned HTTP 500;
  EngineCore and API exited. Crash scheduler dump: three requests, six
  scheduled tokens each, total 18; DSpark placeholders were all `-1`.
- Root cause is confirmed. Threads 36-41 were one top-6 row of negative DSpark
  padding IDs. Raw `-1` first reached vLLM's MoE alignment CUDA kernel, whose
  guard rejects only IDs `>= E`; it could atomically write shared memory at a
  negative offset. The same IDs then reached exact heat `index_add_`, producing
  the observed `dstIndex < dstAddDimSize` assertion. `_promote` was only the
  later asynchronous synchronization point.
- Commit `a4a11205b` masks invalid heat IDs without dynamic-shape indexing;
  count and binary CUDA-graph replays pass at 108 IDs. Commit `23ee20ecd` maps
  invalid routes to alignment's positive `E` skip sentinel, zeros their weights,
  and filters eager `ensure_resident` IDs. Alignment, capture replay, and mixed
  eager sentinel tests pass at shape 18 x top-6.
- Focused tests run against the old image with the changed source mounted:

  ```bash
  docker run --rm --gpus '"device=0"' --entrypoint python3 \
    -v "$PWD/overlay/vllm/vllm/model_executor/layers/quantization/utils/moe_w2_delta.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/moe_w2_delta.py:ro" \
    -v "$PWD/tools/test_moe_w2_mark_seen.py:/opt/test.py:ro" \
    vllm-moet-sm89:v0251 /opt/test.py
  docker run --rm --gpus '"device=0"' --entrypoint python3 \
    -v "$PWD/overlay/vllm/vllm/model_executor/layers/quantization/utils/moe_w2_cubit.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/moe_w2_cubit.py:ro" \
    -v "$PWD/overlay/vllm/vllm/model_executor/layers/quantization/utils/moe_w2_delta.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/moe_w2_delta.py:ro" \
    -v "$PWD/tools/test_moe_w2_route_sentinels.py:/opt/test.py:ro" \
    vllm-moet-sm89:v0251 /opt/test.py
  ```
- Evidence: `runlogs/20260801T0644Z-vllm-exact-dspark-graph18.jsonl`.

Conclusion: graph 18 is the correct performance direction, but current code is
accepted for short-context serving: the rebuilt image survived the warm-up plus
one 3x64, one 3x256 shape warm-up, and three consecutive measured 3x256 trios.

### 5.4 First fixed-image boot: isolated KV log failure, fixed

- Image `d86872...` reused both packs, staged all 43 expert layers, loaded
  40.42 GiB/rank, passed native/FP8 boot guards, and reached 4.51 GiB available
  KV memory. It did not OOM and never reached graph capture or inference.
- New packed-KV `logger.info_once` passed a Python list; its once-cache hashes
  arguments and raised `TypeError: unhashable type: 'list'`.
- Commit `2d53905de` passes a tuple and adds a real `info_once` regression. The
  capacity test prints `(1024, 20, 20, 267, 149)` and passes. Rebuilt image
  `2d49f8...` passed all four pre-load tests again.

### 5.5 Corrected graph-18 serving results

- Excluded 64-token process warm-up: coherent, TTFT 6.74 s, 78.34 decode tok/s.
- 3x64 gate: 30.32 aggregate tok/s; 35.11/33.10/15.62 per request; coherent.
- Excluded 3x256 shape warm-up: 69.87 aggregate; 27.77/35.91/33.98 per request.
- Three measured 3x256 trios (nine requests): all HTTP 200/coherent, median
  per-request decode 32.37 tok/s, aggregate 63.62 tok/s across sequential trios,
  median TTFT 2.54 s. No assertion/error/abort; health remained 200.
- Measured single 256-token runs: 67.04/59.77/58.32 decode tok/s; median 59.77,
  median TTFT 0.83 s. This passes the requested ~50 tok/s single-stream target.
- DSpark after 16 successful requests: 2,167/5,860 accepted = 36.98%.

### 5.6 First 100K prefill: epilogue OOM, reduced fix passed

- The excluded 100K shape warm-up tokenized 99,993 tokens and entered a
  1,044-token prefill chunk, then both ranks OOMed in the deterministic MoE
  unpermute at `gath.index_copy_`. The SSE response closed empty and EngineCore
  died. Artifact: `runlogs/20260801T0736Z-vllm-exact-dspark-graph18-fixed-prefill100k-shapewarm.jsonl`.
- At `T=1044`, top-k 6, H=4096, and 10,104 padded slots, the torch epilogue's
  FP32 slot conversion/product plus FP32 route gather peak near 413.64 MiB per
  rank. The failed allocation requested 158 MiB with only 46.88 MiB free.
- Commit `55d0a2b50` replaces those route-sized FP32 tensors with a persistent
  int32 route inverse and a fixed-j-order Triton reduction. There are no
  atomics; invalid DSpark routes are checked before the inverse is read; cache
  misses are masked before multiplication so stale NaN/Inf cannot contribute.
- Reduced gates passed on both GPUs, including eager repeat stability, CUDA
  graph replay after live input/sentinel changes, mixed misses, H=257/4096/6144,
  top-k 6/8, and the production failure shape. At T=1044: fused allocator peak
  growth 0 MiB with preallocated output versus 413.64 MiB old estimate;
  0.039-0.116 ms; maximum BF16 difference one ULP, worst mismatch 0.006782%,
  relative L2 `5.979e-06`; repeated outputs are bit-identical.
- The keyed expert packs and shared JIT cache remain valid. Runtime compilation
  (`cc1plus`) on the first request is a cold-start phase and is excluded from
  every throughput sample.
- Rebuilt image `ab93aeacbffb...` with the following command; the mark-seen,
  route-sentinel, pool-heat, fused-unpermute, and packed-KV tests all pass from
  the installed image across GPU0/GPU1:

  ```bash
  cd /root/autostart/vllm-ds4/vLLM-Moet
  DOCKER_BUILDKIT=1 docker build -f Dockerfile.sm89-v0251 \
    -t vllm-moet-sm89:v0251 .
  ```

### 5.7 Fused-fix 100K retry: original transient gone, new capacity OOM

- Restart on rebuilt image `ab93aeacbffb...` was clean: KV pool 4836/allocatable
  4835, exact30, graph set [1,2,4,6,8,12,18], DSpark-5, both GPUs 48,060 MiB at
  ready, /health 200. Serving gates on this image: single-256 median 62.12
  decode tok/s (3/3 coherent); 3x256 graph-18 median 32.93 tok/s/stream,
  aggregate 73.1 tok/s, 9/9 coherent, no assertion. Serving path confirmed.
- 100K prefill retry (same command as 5.6, 99,993 tokenized tokens): the request
  ran 677 s with NO assertion and NO `index_copy` failure — the fused unpermute
  eliminated the T=1044 transient and is validated for that scope. Deep in the
  prefill a worker threw `CUDA out of memory: Tried to allocate 72.00 MiB; GPU0
  47.37 GiB total, 72.88 MiB free, 47.16 GiB in use`. EngineCore raised
  EngineDeadError, APIServer shut down, :8011 refused connections. Container
  Exited(0), RestartPolicy=no (will not auto-restart).
- This is a DISTINCT failure from 5.6. 5.6 was a 413 MiB epilogue spike at
  T=1044; the fused fix reduced it to ~0. The new OOM is per-step activation
  pressure at long prefill: exact30 weights (~40.4 GiB) + the full 4836-block KV
  pool (4.51 GiB) leave only ~440 MiB headroom at idle, and the ~100K prefill
  activation footprint (~300 MiB growth observed) exceeds it. Section 6's
  377.5 MiB/card margin counted decode KV only, not prefill activations.
- IMPORTANT config gotcha: this launch sets `--kv-cache-memory-bytes 4846832640`,
  so vLLM SKIPS memory profiling and the `--gpu-memory-utilization 0.98` flag is
  NOT respected for KV sizing (verbatim from boot log: "reserved 4.51 GiB ...
  skipped memory profiling. This does not respect the gpu_memory_utilization
  config"). Therefore lowering UTIL alone is a no-op; headroom must come from
  reducing `--kv-cache-memory-bytes` (shrinks KV pool) or `BATCHED_TOKENS`
  (shrinks per-step activation peak).
- Tiered-store telemetry during the run: pool 4818/4818 covering 43.8% of 11008
  experts, ~212K NVMe fetches/rank, ~1.32 TiB moved, miss p50 ~60 ms — the
  expected exact-path long-prefill cost (correctness, not throughput).
- Observability gap exposed: the probe emitted NO runlog and NO GPU telemetry
  summary for the failed run (summary is success-only). Next run must use a
  background nvidia-smi sampler and/or a sub-100K sweep to capture the curve.

### 5.8 Prefill-size sweep: ceiling is allocator-state, not prompt size

- Patched the benchmark probe (`tools/vllm_dsv4_chat_benchmark.py`) to
  emit `gpu_telemetry` (per-GPU `memory_used_mib` max) on FAILURE too and to
  survive a hard EngineCore-death exception instead of swallowing the data.
  Syntax-checked; first use confirmed telemetry on a passing run.
- Fresh-boot prefill sweep (exact30, KV 4836, graph-18, DSpark-5 unchanged),
  `--max-tokens 1`, nvidia-smi peak per point:
  - 16K -> 48,424 MiB  PASS (152 s)
  - 32K -> 48,484 MiB  PASS (268 s)
  - 48K -> 48,504 MiB  PASS (400 s)
  - 64K -> 48,508 MiB  PASS (544 s)
  - 80K -> 48,508 MiB  FAIL (OOM, EngineCore died, exit 1)
- Decisive: 80K OOM'd at the IDENTICAL peak (48,508) where 64K passed. The
  nvidia-smi total is therefore NOT the constraint — most of it is PyTorch
  reusable cache. The OOM is a specific deep-context allocation that cannot be
  satisfied from cache+free at that moment. Peak growth decelerates (+60, +20,
  +4 MiB per step) and has plateaued at ~48,508 by 48-64K.
- The original 100K OOM (5.7) ran on a server that had already served 3
  single-256 + 9 concurrent-3x256 requests (dirty/fragmented allocator); this
  fresh boot reached 64K. So the ceiling is allocator-fragmentation-sensitive,
  not a clean function of prompt length. Envelope at this config: ~64-80K.
- Implication for the 262K goal: it is ~4x beyond the current envelope. The
  headroom fix (reduce `--kv-cache-memory-bytes`, since UTIL is bypassed by
  `kv_cache_memory_bytes`; or reduce `BATCHED_TOKENS`) is a prerequisite to
  testing 131K/262K retrieval, deferred until the needle sweep characterizes
  correctness within the current envelope.

### 5.9 Needle retrieval: CORRECT at 2K, FAILS from 16K (onset L*=16384)

- Ran `bench/runner/probes.py --probe needle_sweep` against :8011 (realistic
  filler: CONCEPTS.md, docs/ada-sm89-port.md, kernels/MANIFEST.md, AGENTS.md,
  overlay/.../moe_w2_cubit.py, /root/autostart/CLAUDE.md), depth 0.5,
  thinking:false, max_tokens 24. All four points within the capacity envelope
  (no OOM; server stayed HTTP 200):
  - 2K  (pt 2019)   PASS  maxp 1.000  answer `LANTERN-7356` (exact)
  - 16K (pt 19713)  FAIL  maxp 0.881  answer `PELICAN` (wrong, degenerate)
  - 32K (pt 38222)  FAIL  maxp 0.778  answer `CYPHER` (wrong, degenerate)
  - 48K (pt 59166)  FAIL  maxp 0.654  answer `LANTERN-9-9-9-9-9-9-9-9-9-9-9` (garbage)
  - coarse onset L* (first FAIL) = 16384.
- This is the documented long-context derailment (suite comment: "CORRECT at
  2K, WRONG/GARBLED at 131K"), now localized: it begins between 2K and 16K, not
  at 131K. Quality DEGRADES with length: perfect -> wrong-but-confident ->
  repetitive garbage. A retrieval-quality failure, not a crash (all 200).
- Onset REFINEMENT (4096/8192/12288, same protocol, run 20260801T110302Z):
  - 4K  (pt 4192)   PASS  maxp 0.860  `ZEPHYR-5384` (exact)
  - 8K  (pt 10374)  FAIL  maxp 0.881  `PELICAN PELICAN PELICAN...` (degenerate)
  - 12K (pt 14902)  FAIL  maxp 0.769  garbled confabulation
  - refined onset L* = 8192 (~4-10K actual prompt tokens). Retrieval is reliable
    only up to ~4K.
- Leading hypothesis: this config uses `--kv-cache-dtype fp8` (section 4). FP8
  KV precision loss at mid-context is a plausible cause of the retrieval decay —
  distinct from the w8 expert FP8-delta tier (which IS active and self-tests
  clean). A bf16-KV restart + 8K-needle retest would confirm/refute; not yet run
  (bf16 KV halves KV density, tightening the already-narrow capacity margin).
- INDEPENDENT of the capacity ceiling (5.8): retrieval fails at 16K, far inside
  the ~64-80K prefill envelope. So the headroom fix is necessary but NOT
  sufficient for the 262K goal — even with capacity, retrieval is broken past
  ~16K.
- The build has the w8 FP8-delta tier active (boot self-test worst_rel 2.517e-3)
  — the fix this suite is the before/after metric for — yet retrieval still
  derails at 16K. Retrieval correctness is now the PRIMARY blocker for long
  context; capacity is secondary.
- Artifact: `docs/dsv4-sm89-longcontext/runlogs/needle_sweep_20260801T103931Z.json` (+ `.log`).

### 5.10 Needle "retrieval" REFRAMED: retrieval works; the DIGITS are lost

The 5.9 framing ("retrieval is broken from ~8K") is WRONG and must not be
carried forward. Decoding the actual answers against the deterministic needle
codes shows the model finds the needle at EVERY length tested:

| target | true code | model answered | word | digits |
|-------:|-----------|----------------|:----:|:------:|
| 8192   | `PELICAN-1605` | `PELICAN PELICAN PELICAN…` | ✅ | ❌ |
| 16384  | `PELICAN-3738` | `PELICAN`                   | ✅ | ❌ |
| 32768  | `CYPRESS-0527` | `CYPHER`                    | ~  | ❌ |
| 49152  | `LANTERN-2037` | `LANTERN-9-9-9-9-9…`        | ✅ | ❌ |

So this is NOT a retrieval/attention-location failure. The needle is attended;
the *fine-grained numeric content* is what degrades. Confirmed by top-20
logprobs at the digit position (probe: `patches/probes/needle_logprobs.py`
lineage, runlogs below):

- 4K PASS: emits `538` at logprob **-0.000** (p≈1.000); runner-up `539` at
  **-14.4**. Certain — it is READING the digits.
- 8K FAIL: emits `123` at logprob **-0.765** (p≈0.47); runners-up `301`,
  `927`, `627`, `294` — a diffuse spread of unrelated 3-digit priors, and the
  true token `160` is **absent from the top-20 entirely**.
- The WORD tokens stay at p≈1.000 at BOTH lengths.

Character of the errors (this is the key diagnostic signal): with redundancy or
salience added, answers become NEAR-MISSES rather than priors —
`verbose` (digits also spelled out) → `1606` vs `1605`; `repeat` (code 3×) →
`305` vs `1605`; `words` (3 salient words) → `ZEPHYR-ORYX-CORYX` vs
`…-CYPRESS` (first two exact, third blended). Plain digits fall back to the
generic prior `1234`. Near-misses under improved SNR = a borderline NUMERICAL
PRECISION path, not absent attention and not a structural cap.

**Two structural hypotheses were tested and REFUTED — do not re-litigate:**

1. *"Fails above a fixed absolute needle position (~2048 = index_topk×ratio4)."*
   REFUTED: at total pt 2864 the needle PASSES at absolute positions 1146,
   1432, 1662, 1890, 2148, 2434 and 2721 (7/7, all depths 0.40→0.95) — well
   above 2048; while at total pt 9685 it FAILS at absolute position 1695 —
   below 2048. Needle position is not the controlling variable.
2. *"Fails above a fixed total context length (~8192 = 2048 entries × ratio 4)."*
   REFUTED: the length response is NON-MONOTONIC. pt 8316 **PASSES** (exact
   `SAFFRON-7888`) while pt 5100 and pt 6650 FAIL. There is no clean cutoff.

What IS established: reliability decreases with length on average (2K/2.9K/4K
reliable; ≥5K unreliable but not uniformly), is deterministic per input
(bit-identical on repeat, so not run-to-run noise), improves with redundancy,
and a needle in the first ~200 tokens of a 9.7K context still passes.

**Eliminations (each backed by a cheap test, not by reasoning alone):**

- **FP8 KV cache is NOT the cause.** Upstream's `pro6000x2-tp2` recipe passes
  `needle @121k tokens` with the SAME `--kv-cache-dtype fp8`
  (`git show upstream/main:bench/recipes/deepseek-v4-flash/pro6000x2-tp2.yaml`).
  Upstream differs by running native Blackwell kernels, so the defect is in the
  sm89 port, not in the precision *format*.
- **The top-k selection kernels are CORRECT.** New unit test
  (vLLM-Moet `tools/test_indexer_topk_selection.py`, commit `005129b44`):
  with k=512 and all high scores placed BEYOND index 512, `persistent_topk`
  (the op the sm89 decode path actually calls) returns **512/512** of the
  high-scoring indices with **zero** overlap with the first 512.
  `top_k_per_row_decode` also passes. Positional degeneration is ruled out.
- **The ragged indexer-logits fallback is faithful.** sm89 routes
  `fp8_fp4_mqa_logits` to `_torch_fp8_mqa_logits`
  (`overlay/vllm/vllm/utils/deep_gemm.py:588`); it implements
  `sum_h relu(q[:,h]·kᵀ)·w[:,h]` with correct per-row k dequant and correct
  `[ks,ke)` -inf masking — matching the checkpoint reference.
- **The paged indexer-logits Triton port passes** its own suite
  (`overlay/vllm/tests/kernels/attention/test_triton_paged_mqa_logits_dsv4.py`:
  9 passed, 5 skipped).
- **Per-token scalars cannot be the cause.** top-k is per QUERY ROW, so
  `q_scale`, `softmax_scale` and `head_scale` (all per-token/global) are
  ranking-neutral. Only per-head `weights`, per-key `k_scale`, or the RoPE
  angles can change ranking across keys.
- **The sparse-MLA reader's decode math is correct**: E4M3 (bias 7, subnormal
  `mant*2^-9`), two's-complement INT8, and UE8M0 `2^(s-127)` all verified by
  hand against the documented packed layout.

**Architecture note (from the checkpoint's OWN reference,
`/root/models/DeepSeek-V4-Flash-0731/inference/model.py` — authoritative):**
the KV cache is a RING BUFFER of only `window_size` tokens
(`self.kv_cache[:bsz, start_pos % win] = kv`); everything older is reachable
ONLY through learned gated-pooling compressed entries. Selection is
`topk(min(index_topk, end_pos // ratio))` over COMPRESSED entries, and the
indexer exists only on `compress_ratio == 4` layers — ratio-128 layers use
positional order (all causal entries) by design. Per-layer counts for this
checkpoint: 21 layers ratio 4, 20 layers ratio 128, 5 layers ratio 0. So
top-512 is a no-op below 2048 tokens on ratio-4 layers and below 65536 on
ratio-128 layers. This explains why the coarse WORD always survives (ratio-128
layers see the whole context) while exact digits do not.

**Latent arch bug found (not this root cause, but it will bite):**
`has_cutedsl()` (`vllm/vllm/utils/import_utils.py:547`) returns
`_has_module("cutlass")` — a PACKAGE check with NO architecture check. The
`cutlass` package IS installed in the image, so
`dequantize_and_gather_k_cache` (`vllm/vllm/models/deepseek_v4/common/ops/
cache_utils.py:403`) dispatches to a CuTeDSL kernel that CANNOT compile for
`cubin-chip=sm_89`. This is why 17/36 of
`vllm/tests/kernels/test_compressor_kv_cache.py` fail on sm89 (all CuTeDSL
compile errors). Serving does not currently reach that path (it would hard-
crash), but the gate should also test device capability.

### 5.11 `VLLM_DSV4_KV_INT8=1` A/B: ELIMINATED (no effect on digit fidelity)

Ran the knob purpose-built for this bug. Boot confirmed it active
(`nope_int8=True`, `int8 selected=True`). Identical seeds/filler, so the prompts
are byte-identical to the baseline:

| actual pt | FP8-E4M3 baseline | INT8 NoPE | true code |
|----------:|-------------------|-----------|-----------|
| 2864 | PASS `TUNDRA-3711`  | PASS `TUNDRA-3711`  | TUNDRA-3711 |
| 5100 | FAIL `FALCON`       | FAIL `FALCON`       | FALCON-1042 |
| 6650 | FAIL `PUMICE-7541`  | FAIL `PUMICE-7542`  | PUMICE-7544 |
| 8316 | PASS `SAFFRON-7888` | PASS `SAFFRON-7888` | SAFFRON-7888 |
| 9376 | FAIL `CYPRESS-0120` | FAIL `CYPRESS-1234` | CYPRESS-0074 |
| 9686 | FAIL `PELICAN-1234` | FAIL `PELICAN-1234` | PELICAN-1605 |

**6/6 identical pass/fail.** A measured 3-4× improvement in NoPE value readback
(2.48e-2 → 8.0e-3, §5.10) moved one wrong digit by one and fixed nothing.
**Conclusion: NoPE KV storage precision is NOT the limiting factor.** Do not
spend more time on KV value precision (this also further de-prioritizes bf16 KV,
which costs 2× density for the same quantity INT8 already improved 4× for free).

Two operational findings from these runs:
- `VLLM_DSV4_KV_INT8=1` is **currently unusable with CUDA graphs** in this
  image. Toggling the constexpr creates a Triton variant absent from the JIT
  cache, so Triton compiles/loads it INSIDE graph capture →
  `CUDA error: operation not permitted when stream is capturing`. First attempt
  (default 4.51 GiB KV) died earlier still with `Triton Error [CUDA]: out of
  memory` during capture (~440 MiB headroom). The A/B above therefore ran with
  `--enforce-eager` (correctness is unaffected; only decode speed is). Proper
  fix: pre-warm the INT8 kernel variant before capture, or persist its cubins.
- A **2 GiB KV pool boots fine** and still holds a full 262K request
  (min ≈1481 blocks ≈1.484 GiB) — useful for the §5.8 headroom work.

### 5.12 Ratio-4 COMPRESSOR hypothesis: RAISED, then ELIMINATED (see 5.13)

With storage precision, selection kernels, and indexer scoring all eliminated,
the remaining path carrying old-token detail is the compressed representation
itself. Per the checkpoint reference, tokens outside the `window_size` ring
buffer are reachable ONLY through `Compressor` — "learned gated pooling over
`compress_ratio` consecutive tokens", and crucially:

```python
# reference Compressor: overlap=True when ratio==4
overlap = 1 if compress_ratio == 4 else 0
total   = (1 + overlap) * compress_ratio      # gathers 8 entries for ratio 4
```

i.e. the ratio-4 compressor uses **overlapping windows** "for smoother
compression boundaries". If our sm89 implementation
(`overlay/vllm/vllm/models/deepseek_v4/compressor.py`,
`…/common/ops/fused_compress_quant_cache.py`) mishandles that overlap or the
gating, then whether a needle's digits survive depends on where it falls
relative to 4-token compression-block boundaries — which would produce exactly
the observed behaviour: content-dependent, deterministic per input, and
NON-monotonic in both length and position (pt 8316 passes, 5100/6650 fail).
This also explains why the coarse WORD always survives: the 20 ratio-128 layers
see the whole context and carry the gist.

Test it CHEAPLY and first, before any further server boot: differential-test our
compressor against the checkpoint's reference `Compressor` on CPU, the same way
`tools/test_nope_needle_recall.py` does for the encoding. Note
`vllm/tests/kernels/test_compressor_kv_cache.py` cannot serve as-is — 17/36 of
its cases fail on sm89 purely from the arch-blind `has_cutedsl()` gate (§5.10).

- Artifacts: `docs/dsv4-sm89-longcontext/runlogs/needle_logprobs_8k_20260801T112338Z.json`,
  `docs/dsv4-sm89-longcontext/runlogs/needle_logprobs_ctrl_20260801T113740Z.json`,
  `docs/dsv4-sm89-longcontext/runlogs/needle_variants_fp8_20260801T114419Z.json`,
  `docs/dsv4-sm89-longcontext/runlogs/needle_boundary_20260801T122658Z.json`,
  `docs/dsv4-sm89-longcontext/runlogs/needle_totallen_20260801T123553Z.json`,
  `docs/dsv4-sm89-longcontext/runlogs/needle_int8_eager_20260801T130145Z.json`.

### 5.13 Compressor ELIMINATED; arch gate fixed; compressor suite now green

**Compressor semantics are correct.** New CPU-only differential test
(vLLM-Moet `tools/test_compressor_vs_checkpoint_ref.py`, commit `8377e2db1`)
compares against a THIRD, independent transcription of the checkpoint's own
`Compressor` — necessary because `vllm/tests/kernels/test_compressor_kv_cache.py`
validates the fused kernel against a reference in the SAME file, so a shared
misreading of the architecture would pass silently. Checked: the ratio-4
overlapping window (4 entries from the PREVIOUS block via the first half of the
`2*head_dim` projection, then 4 from the current block via the second half), the
softmax axis, block 0's `-inf`/`0` padding, APE indexing by within-block
position, and the per-output RoPE position index (`b*ratio`).
Result: **AGREE at ~2.6e-3 rel** (the shared bf16-store rounding floor) across
ratio 4 and the ratio-128 non-overlap branch, for 1/2/5/9 blocks and state block
sizes 4/8/16/64. So §5.12's hypothesis is dead — the compressor is not the
defect.

**Arch gate fixed (a real bug, shipped).** `has_cutedsl()` is a package check
only, and the `cutlass` package IS installed in the sm89 image, so
`dequantize_and_gather_k_cache` dispatched to `DequantGatherKCacheKernel`, which
has no pre-SM90 lowering (`cute-to-nvvm{… cubin-chip=sm_89}` fails). The fix
requires `has_device_capability(90)` too and falls back to the arch-portable
Triton implementation. Overlay:
`overlay/vllm/vllm/models/deepseek_v4/common/ops/cache_utils.py`;
patches regenerated (`--verify 85/85`, `--check 85/85`).

**This unblocked the only sm89 validation of the compressed-attention K-cache
store/gather path** — the very path that holds the values long-context retrieval
reads back. Before: 19 passed / 17 failed, every failure a CuTeDSL sm89 compile
error, so nothing was known about numerical correctness. After the fix, 11 of
the 17 pass, including `test_deepseek_v4_attention_quant_cache_roundtrip` 8/8,
`test_dequantize_and_gather_k_cache` 2/2 and
`test_deepseek_v4_quant_magnitude_range`. **That path is numerically correct on
sm89 — now demonstrated, not assumed.**

The remaining 6 (`test_fused_kv_insert_indexer[use_fp4=True]`) are the MXFP4
indexer cache: datacenter-Blackwell only (`indexer.py` hard-asserts
`is_device_capability_family(100)`; `use_fp4_indexer_cache` defaults to False;
on sm89 `ptxas --gpu-name=sm_89` rejects the kernel's PTX with exit 255 because
the E2M1 conversion instructions do not exist pre-sm_100). Now skipped with that
reason via an overlay of the test, because 6 permanently-red Blackwell-only
cases hide regressions in the paths sm89 does use.
**Suite on sm89: 30 passed, 6 skipped, 0 failed.**

Running tally of eliminated causes: FP8 KV dtype, NoPE storage precision
(INT8 A/B), top-k selection kernels, indexer scoring (ragged + paged),
per-token scalars, compressor semantics, and the compressed-attention KV
store/gather path.

### 5.14 Digit fidelity is CHUNK-SIZE DEPENDENT → it sits at the noise floor

Two tests, in order.

**(a) Intra-chunk overwrite ruled out.** The compressor state is a
`SlidingWindowMLASpec` with a window of only 8 entries (ratio 4) / 128
(ratio 128), yet chunked prefill writes up to `max_num_batched_tokens` tokens
per forward *before* the compress kernel gathers. A window-sized allocation
would silently corrupt. It is not window-sized — the allocation is chunk-aware:

| ratio | window | block | required at 1056 | observed | at 528 | observed |
|------:|-------:|------:|-----------------:|---------:|-------:|---------:|
| 4     | 8      | 4     | 266              | **267**  | 134    | **135**  |
| 128   | 128    | 8     | 148              | **149**  | 82     | **83**   |

(one spare block each). Pinned by
`tools/test_compressor_state_capacity.py` (CPU-only, commit `913b4e0d7`), which
also cross-checks the ratio→block_size / sliding_window constants against
`CompressorStateCache.__init__` and carries a negative control proving a
window-only allocation would be rejected.

**(b) But halving the chunk size CHANGES THE ANSWERS.** Identical prompts,
identical weights, only `--max-num-batched-tokens` 1056 → 528:

| actual pt | chunk 1056 | chunk 528 | true code |
|----------:|------------|-----------|-----------|
| 2864 | PASS `TUNDRA-3711`  | PASS `TUNDRA-3711`  | TUNDRA-3711 |
| 5100 | FAIL `FALCON`       | FAIL `FALCON`       | FALCON-1042 |
| 6650 | FAIL `PUMICE-7541`  | **PASS `PUMICE-7544`** | PUMICE-7544 |
| 8316 | **PASS `SAFFRON-7888`** | **FAIL `SAFFRON-1206`** | SAFFRON-7888 |
| 9376 | FAIL `CYPRESS-0120` | FAIL `CYAN-42` (word lost too) | CYPRESS-0074 |
| 9686 | FAIL `PELICAN-1234` | FAIL `PELICAN-1234` | PELICAN-1605 |

**2/6 correct either way — but a DIFFERENT 2.** Two points flipped in OPPOSITE
directions. Interpretation matters here: chunked prefill is not expected to be
bitwise invariant in any model (reduction order changes), so bit differences are
normal. What is diagnostic is that such differences FLIP retrieval answers, and
that they flip *both ways* — a systematic chunking bug would degrade
consistently. This is a **noise floor**, not a discrete defect.

That reframes the whole hunt and fits every prior observation: the word always
survives (the 20 ratio-128 layers see the whole context — only ~75 entries at
9.7K, far under `index_topk`, so no selection happens there at all); digits are
lost non-monotonically and deterministically per input; redundancy produces
near-misses; and 4× better NoPE storage precision (§5.11) changed nothing
because storage was never the bottleneck. The most likely limiting quantity is
**which ratio-4 compressed entries get SELECTED** — top-512 over a diffuse score
distribution operating near ties, so any numerically-transparent perturbation
reshuffles the selected set and the digit-bearing entry sometimes drops out.

Sharp prediction, now under test: raising `index_topk` should make selection far
less marginal and digit recall correspondingly robust.
`--hf-overrides '{"index_topk":2048}'` works (2048 is one of the k values
`use_persistent_topk` accepts, and the boot log already captures `K_extra=2048`
geometries). Run against the chunk-528 / topk-512 row above as a clean
single-variable baseline. Artifacts: `docs/dsv4-sm89-longcontext/runlogs/needle_chunk528_*.json`,
`docs/dsv4-sm89-longcontext/runlogs/needle_topk2048_*.json`.

## 6. KV capacity: runtime fits three full contexts; printed metrics were wrong

- Startup reported 912,691 tokens/3.482x; `/metrics` reported
  624,189/2.381x. Neither was authoritative. Both used a byte-weighted capacity
  estimator on different worker/scheduler spec representations, while runtime
  allocation consumes shared packed block IDs per group.
- The actual per-request maxima for DSV4 plus three DSpark draft layers are
  `[1024, 20, 20, 267, 149] = 1,480` packed blocks.
- Pool: 4,836 blocks. One null block is reserved. Strict three-stream demand is
  `3 * 1,480 + 1 = 4,441`, leaving 395 blocks or 377.545 MiB/GPU at the
  1,002,240-byte packed stride.
- Correct nominal capacity: 856,573 tokens / 3.267568x at max length 262,144.
  Prefix caching is off, scheduler watermark is zero, and DSpark lookahead is
  already covered by the configured 1,056 batched-token bound.
- A CPU-only regression proves worker/scheduler and inner-dictionary-order
  invariance. It passes with the exact five groups above.
- vLLM-Moet commits `729729ff7` and `2d53905de` fix the calculation/log and emit pool,
  allocatable, per-request, and per-group blocks. Preserved patch 0019 carries
  this fix plus exact `w8x` heat persistence. The live graph-18 server still
  uses the prior image, so its old printed metrics remain wrong; runtime block
  allocation itself is unchanged and correct.
- Capacity is no longer a blocker to staged long-context validation, but the
  377.5-MiB/card margin is narrow. Run short concurrency, then 100K/131K
  prefill, before the final 3 x 262K allocation test.

## 7. Decisions and rejected directions

- CPU-only AMXINT4: about 10 tok/s; DDR4 expert bandwidth ceiling.
- Hybrid TP1/TP2 CPU+GPU expert offload: 7-9 tok/s; synchronization dominates.
- SGLang GPU1 draft plus target expert N8/static/hot experiments: correct final
  hot version was only 7.793 tok/s; rejected. History is in git/old docs.
- Expanded E4M3 GPU storage wastes residency. Keep checkpoint FP4 storage and
  perform E4M3 expansion in registers for FP8 MMA.
- FlashInfer packed-FP4 and SGLang MXFP4 Marlin routes did not support SM89 in
  the tested builds. Do not assume a brand name implies an Ada kernel exists.
- PP=2 remains a later alternative, but current TP2 exact/DSpark has already
  passed the single-stream target. First fix batching/cache locality.
- Multi-GPU is not deferred: the current successful path already uses both
  GPUs. After GPU0/rank-local execution is efficient, continue using GPU1 for
  TP, KV capacity, and aggregate throughput; the full DSV4 model cannot fit in
  96 GiB as fully expanded FP8.

## 8. NEXT queue (execute in order)

1. Let the current exact30/TP2/DSpark-5/graph18 launch reach readiness. A
   monitoring sub-agent is checking load/capture/failure logs without sending
   inference; once logs report listening, verify `/health` and `/v1/models`.

2. Exclude one short process/compile warm-up, then require one coherent warmed
   single-256 and one coherent 3x256 graph-18 request. Confirm both ranks/GPU
   utilization, exact misses/replays, DSpark acceptance, and HTTP health.

3. DONE (see 5.7-5.8): 100K retried — fused fix held 677 s, then a DISTINCT
   capacity OOM; fresh-boot sweep maps the envelope at ~64-80K (16/32/48/64K
   PASS, 80K OOMs at the same peak as 64K). Ceiling is allocator-state, not
   prompt size. Do NOT blind-retry 100K/131K/262K — they are beyond the envelope
   until the headroom fix lands. The command below is the retried one (kept for
   reference):

   ```bash
   cd /root/ktransformers
   python3 tools/vllm_dsv4_chat_benchmark.py prefill \
     --warmups 0 --runs 1 --streams 1 --target-prompt-tokens 100000 \
     --max-tokens 1 --timeout 3600
   ```

4. DONE (see 5.10): needle behaviour characterized within the envelope and
   REFRAMED — retrieval works, digit fidelity is the defect; three structural
   hypotheses refuted by measurement; top-k kernels and indexer-logits
   fallbacks unit-tested correct.

4a. ACTIVE: finish the `VLLM_DSV4_KV_INT8=1` A/B. Launch exactly as §4 plus
   `-e VLLM_DSV4_KV_INT8=1` and `--kv-cache-memory-bytes 2147483648` (the
   reduced pool is REQUIRED: toggling the constexpr forces a one-time Triton
   recompile that OOMs the ~440 MiB-headroom default during graph capture; 2 GiB
   still holds a full 262K request, min ≈1481 blocks ≈1.484 GiB). Re-run the
   same needle battery with identical seeds/filler and compare digit fidelity.

4b. Then: add an env-gated indexer trace (candidate count, `k_select`, whether
   selection was a no-op, and the score RANK of the entry holding a known needle
   position) so selection is MEASURED, not inferred. Cheaper and more conclusive
   than further end-to-end sweeps.

4c. Untried levers: `--hf-overrides '{"index_topk": 2048}'` (4× ratio-4
   selectivity coverage, costs compute not VRAM); differential-test the ratio-4
   compressor against the checkpoint reference. Fix the arch-blind
   `has_cutedsl()` gate (5.10) — a latent sm89 hard-crash.

4d. Feedback-loop rule going forward: debugging/deploying is expensive (6-10 min
   boots, 30-110 s probes) while unit tests are cheap. Two harnesses now exist —
   `tools/test_indexer_topk_selection.py` (kernel correctness) and
   `tools/test_nope_needle_recall.py` (CPU-only, quantifies encoding precision
   in seconds). Extend these before reaching for another restart.

5. If per-stream throughput must improve beyond ~32 tok/s, implement an
   opt-in cache-local decode scheduler quantum. Minimal reviewed design: keep
   scheduler FCFS order,
   schedule one resident decode owner for 16 completed scheduler steps, rotate
   fairly, leave prefills eligible, preserve inactive DSpark drafts, and refuse
   async/PP overlap. Test q=8/16/32. Do not weaken the conservative pool-floor
   guard in the first patch.

6. After long-context correctness, A/B DSpark width 4 vs 5 and scheduler quantum
   q=8/16/32; fifth-position acceptance is low, while q may reduce the observed
   cache/fairness stalls without changing model quality.

## 9. Safety and observability invariants

- First request after process start is always excluded; `cc1plus` proves cold
  compilation, not steady-state slowness.
- Prefer reduced kernel/config/scheduler probes before a full model restart.
- Reuse keyed pack and compiler caches. Never restamp a cache from a different
  checkpoint, TP size, residency mode, quantizer, or shape.
- On every launch record resolved argv/env, source/image IDs, GPU ownership,
  model-load completion, exact pool/arena slots, scheduler KV tokens, graph
  capture sizes, and DSpark acceptance.
- On every benchmark record per-request TTFT/decode/end-to-end rates, exact
  miss/replay KPI, graph dispatch mode, both-GPU utilization/power/memory, and
  semantic output. High cache hit percentage alone is insufficient; synchronous
  replay percentage and tail latency predict throughput.
- Stop after a decisive failure; do not spend three full repetitions proving an
  obvious regression.
- Do not delete caches or unrelated dirty/untracked files. Disk expansion is
  complete and space is not currently a blocker.
- Runtime/package edits must be committed in their source repository and
  exported into `/root/ktransformers/patches`; package reinstall can overwrite
  live changes.

## 10. Quick status commands

```bash
docker ps --filter name=moet-0731-dspark-exact
docker logs --tail 200 moet-0731-dspark-exact
curl -fsS http://127.0.0.1:8011/health
curl -fsS http://127.0.0.1:8011/metrics | \
  grep -E 'kv_cache|num_gpu_blocks|cudagraph|spec_decode'
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw \
  --format=csv,noheader,nounits
```

## 11. Relevant preserved artifacts

- Exact implementation: patches 0016 and 0017 above.
- Experiment CLI passthrough: patch 0018, vLLM-Moet commit `b62d58639`.
- Exact heat persistence and packed-KV capacity fix: patch 0019, vLLM-Moet
  commit `729729ff7`.
- Graph-safe exact heat sentinel fix: patch 0020, commit `a4a11205b`.
- MoE alignment/eager sentinel boundary: patch 0021, commit `23ee20ecd`.
- Packed-KV once-log hashability: patch 0022, commit `2d53905de`.
- Bounded deterministic fused unpermute: patch 0023, commit `55d0a2b50`;
  SHA-256 `fb2832ccedee391ed981bd7dd7c5a17a1369d7ee910e839d618ed8a38562487f`.
- Corrected serving artifacts: `runlogs/20260801T0729Z-*` through
  `runlogs/20260801T0734Z-*`; first 100K OOM evidence:
  `runlogs/20260801T0736Z-*`.
- Benchmark client: `tools/vllm_dsv4_chat_benchmark.py`.
- Warm single evidence: `runlogs/20260801T060649Z-*` and
  `runlogs/20260801T061145Z-*`.
- Rejected concurrent evidence: `runlogs/20260801T061230Z-*` and
  `runlogs/20260801T061600Z-*`.
- Older SGLang patches 0001-0013 and their focused probes remain preserved but
  are not the active runtime direction.

Update this file after every meaningful decision or reproducible state change,
not after every shell inspection.
