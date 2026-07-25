# PROGRESS — DeepSeek-V4-Flash "endless generation" in Claude Code

Tracking doc for the investigation + fix of the Claude Code runaway-generation
issue on the local rig. Living doc — update as work progresses.

## Symptom

`claude-local` (Claude Code → `http://llm.local:8080`) produces a **wall of
text / endless generation** for trivial prompts (e.g. `hey`) and erratic
behavior during tool-calling agent loops.

## Diagnosis (evidence-led, 2026-07-24)

Two compounding causes. **Root cause #1 (operational) is dominant; quantization
(#2) is a secondary contributor that is not yet isolated.**

### #1 — Stale Mode-B sticky router vs a deepseek-only moet backend (DOMINANT)

`vllm-sticky-router.service` (pid 574, enabled, restarted 06:56 today) is
carrying the **old Mode-B (qwen) routing config** while the live stack is
**Mode A: a single moet container serving `deepseek-v4-flash` + `auto` on
`:8001`** (host net). Concretely, the running router env:

```
ROUTER_BACKENDS=:8001, :8002, :8009      ← only :8001 (moet) is up; :8002 & :8009 REFUSED
ROUTER_BACKEND_0_MODELS=qwen-27b auto …  ← thinks :8001 serves qwen-27b (it serves deepseek)
ROUTER_RECAP_OFFLOAD=1 → claude-haiku → :8009   (DOWN)
ROUTER_FALLBACK_PRIMARY_MODEL=qwen-27b   (not served)
ROUTER_PROTECT_MODEL=qwen-27b            (not served)
ROUTER_OVERFLOW_THRESHOLD=10, REJECT_ON_SATURATION=1
```

Evidence from the user's own `claude-cli/2.1.150` traffic in
`/var/log/vllm/sticky-router.log` — every agent turn (~107K-token bodies,
363–400 KB):

```
ctx-narrow est=107273+32000 dropped=[1] kept=[0]
messages idx=0 (sticky) … model='auto'
retry-recovery: target 0 connection-refused, served by idx=1   ← moet refused (transient saturation; 200 in 2 ms when idle) → rerouted to DEAD :8002
```

Recap/title-gen turns were rewritten to `claude-haiku` and sent to `idx=2` =
`:8009` (gemma sidecar, **also dead**); the overflow guard fires a flood of
**429**s. The watchdog logs **832 consecutive "full qwen outage"** lines.

**Config flow:** systemd `vllm-sticky-router.service` →
`/root/autostart/run-host-helper.sh sticky-router` → sources `/root/.config/hf.env`
→ calls `setup_router_runtime_env` (defined in `vllm-qwen27b-dflash.sh`, the
Mode-B launcher) → derives backends from `GPU_DEVICE_ARRAY` (one per GPU),
`BACKEND_PRESET_0/1` (qwen presets), and the gemma sidecar, then **forces**
`ROUTER_BACKENDS=$ROUTER_BACKENDS_COMPUTED`. Moet is TP=2 = ONE backend on
`:8001`, so this Mode-B computation cannot describe it. (Note: `STICKY_SESSIONS=0`
in hf.env, yet the unit runs anyway.)

### #2 — 2-bit MoE quantization KV/logit flattening at long context (SECONDARY, not yet isolated)

When a request *does* reach deepseek, it is a **~107K-token prefill on 2-bit MoE
experts**. The repo's own code (`moe_w2_cubit.py:89-101`) documents the
mechanism:

> *"the KV built from 2-bit prefill flattens decode logits cumulatively with
> context length"* … +8–11% completion-token inflation vs native.

The known remediation (prefill via the FP4 delta tier) **cannot run on Ada/sm89**
— `serve_sm89_ds4.sh` ships `VLLM_MOE_W2_DELTA_GB=0` and `moe_w2_sm89.py:44-46`
states the FP4 delta tier "is NOT ported yet" on Ada. Direct API tests were
**clean up to ~14 K tokens** at temp 0; real Claude Code traffic is ~7× larger
(107 K), in the regime where this flips into degeneration. Consistent with, but
**not yet isolated from**, cause #1.

## Could not extract the literal "hey"

`ROUTER_CAPTURE_CONVOS=1` is set, but `/var/log/vllm/router-convos/` is **stale
since Jul 17**, `router-stats.jsonl` stopped Jul 17, and dashboard history is
empty — the running router isn't writing captures (likely because broken traffic
dead-ends before the completion/capture point). Only metadata survived in
`sticky-router.log`. **Re-arming capture is required before the next bad "hey".**

## Plan

### Interim router fix (NOW) — isolated, does NOT touch the Mode-B launcher

Operator's direction: `vllm-qwen27b-dflash.sh` was used for qwen and probably
will be again; **if DeepSeek-V4-Flash proves viable on this hardware, the
moet/deepseek mode will be routed INTO that launcher properly later.** So the
interim fix stays out of the launcher and is forward-compatible (remove the
override once the launcher hosts deepseek).

- [x] Step 1 — this PROGRESS.md
- [x] Add a `ROUTER_LAYOUT=moet` branch to `run-host-helper.sh` `sticky-router)`
      that honors an explicit operator-declared single backend and skips the
      Mode-B `setup_router_runtime_env` computation.
- [x] Declare the moet layout in `/root/.config/hf.env`:
      `ROUTER_BACKENDS=http://localhost:8001`,
      `ROUTER_BACKEND_0_MODELS="deepseek-v4-flash auto"`,
      `ROUTER_BACKEND_0_MAX_CTX=262144`, `ROUTER_RECAP_OFFLOAD=0`,
      clear `ROUTER_FALLBACK_PRIMARY_MODEL`, `ROUTER_PROTECT_MODEL=deepseek-v4-flash`.
- [x] Re-arm observability: `ROUTER_META_ENABLED=1`, keep
      `ROUTER_CAPTURE_CONVOS=1` (verify the dir receives writes after restart).
- [x] `systemctl restart vllm-sticky-router`; verify env loaded, `/metrics`
      shows the single deepseek backend, capture dir gets a fresh write on a
      clean test request.

### Verification (2026-07-24 23:06)

- New router env live: `ROUTER_BACKENDS=http://localhost:8001`,
  `seed={0:['deepseek-v4-flash','auto']}`, ctx auto-detected 262144 from
  `/v1/models`. Startup banner: `backends: ['http://localhost:8001']`.
- `/metrics`: only `backend="0"`; no more `:8002`/`:8009` polls (the dead
  backends are gone from the new process).
- Request **through `:8004`** (Claude Code path), `model:auto` "hey" →
  `stop:end_turn`, "Hello! How can I help you today?", 10 tok. Routes cleanly.
- Observability re-armed: both `/var/log/vllm/router-convos/` and
  `router-stats.jsonl` mtime jumped 2026-07-17 → 2026-07-24 23:06; the test
  "hey" was captured to `001784934361157-…json` (keys: summary, messages,
  response_text, response_truncated). Extraction path confirmed.

### Then (operator-driven)

- [ ] Operator submits a bad `hey` in Claude Code.
- [ ] Extract the captured request + response from
      `/var/log/vllm/router-convos/` (newest `<epoch_ms>-<req_id>.json`).
- [x] Captured bad "hey #3" repro → `repro/endless-gen-hey3-2026-07-24.md` (routing exonerated; confirms long-context derailment).

### Step 2 — server-side capture routing (2026-07-24 23:24)

Discovery: the operator's Claude Code traffic does NOT use the sticky router.
`claude-local` (Mac) → `llm.local:8080` (Mac `/etc/hosts`: `llm.local`=
`192.168.0.24`) → nginx `qwen-lb` → `vllm_backends` → moet `:8001`. The router
`:8004` was bypassed, so re-arming its capture didn't help. (Switching the
client to `:8004` is forbidden by the ansible repo's CLAUDE.md — loses the
`:8080` front-door features.) The bad `hey` repro confirmed via nginx access
log: `claude-cli` POST `/v1/messages`, **1 MB (~250K tok) request, 322 s,
~1 MB degenerate response**, via `:8002(dead,502)→:8001(200)`.

Fix: nginx `:8080` now routes the chat endpoints through the router —
`location = /v1/messages` and `= /v1/chat/completions` →
`host.docker.internal:8004`; all other locations unchanged. Verified
(`nginx -t`, `docker restart qwen-lb`, inode reload confirmed, `/v1/models`
and `/dashboard/` still 200, a `/v1/messages` through `:8080` produced a
capture with matching req id across HTTP/router-log/convo-file).
Backup: `/root/.config/qwen-lb/nginx.conf.bak-router-capture`.
Revert: `cp …bak-router-capture …/nginx.conf && docker restart qwen-lb`.
CAVEAT: edits the LIVE nginx.conf, regenerated by the launcher's
`create_nginx_config` on the next launcher run — non-persistent; to make
durable, encode in `create_nginx_config`.

### Later (contingent on the bad-"hey" + quant isolation)

- [ ] Decisive quant test: ~107 K-token "hey" **directly to `:8001`** (bypass
      router) at temp 0 and 1. Degeneration ⇒ #2 confirmed; clean ⇒ router was
      the whole problem.
- [ ] If DeepSeek-V4-Flash is viable here, route the moet/deepseek mode into
      `vllm-qwen27b-dflash.sh` (extend `setup_router_runtime_env` for the TP2
      single-backend layout) and remove the interim `ROUTER_LAYOUT=moet` override.

---

## Streamed full-precision prefill quality mode (2026-07-25)

Implementation of pieces 1–6 of `internal/STREAMED_PREFILL_PLAN.md` on branch
`feat/moe-w2-sm89-native-decode`. Key changes (overlay files; not committed):

### Piece 1 — Deferred pool allocation (build-time OOM fix, single boot)

`DeltaTier.__init__` gains `defer_pool=True` kwarg: when set, `_auto_pending=True`
regardless of `pool_gb`, and `_explicit_gb` records the cap for `finalize_auto`.
`get_tier`'s FP8 branch passes `defer_pool=True`; the pool is allocated post-KV by
the existing `finalize_auto` hook in `gpu_worker.py` (line ~769), never during
`build_layer_planes`. A deferred parity gate (`w8_pool_gate_deferred` in
`moe_w2_cubit.py`) runs right after `finalize_auto` to fill the gap the build-time
gate had to skip (n_slots was 0). **The two-boot cache workaround is obsolete**:
a single boot now builds (or cache-loads) planes and then sizes the pool post-KV.

### Piece 2 — Surgical ensure-mode (frozen w8 background tick + emergency-only eviction)

`_tick_once` re-gains the `w8` freeze guard (removed in f85fbdf): when the w8 free
list is empty the background manager returns immediately. `_take_slots_batch` gains
the complementary `emergency=False` guard so the background's lazy-promote path
respects the freeze, while `force_promote` and `ensure_resident` (both pass
`emergency=True`) continue to evict-and-pin. This eliminates the background-race
surface without losing per-chunk prefill coverage.

### Piece 3 — FP4-storage kernel + tier (`VLLM_MOE_W2_FP8_STORE=fp4`)

New `moe_w8fp4_mm_sm89` Triton kernel: stores the checkpoint e2m1 nibble plane
(6 MiB/slot at TP2 vs 12 MiB for fp8-store) and decodes each nibble to its exact
e4m3 bit pattern in registers before the native fp8 `tl.dot`. Halves pool slot,
pinned host arena (~65 GiB/rank vs ~129 GiB/rank for fp8-store), and per-chunk H2D
traffic. `SCALE_REFIT=0` is required and enforced at build (a refit-decremented
scale needs u = 2×e2m1, off the e2m1 grid). `PLANES_SUBDIR=planes-qp` keeps the
refit=0 plane cache separate so toggling the mode does not wipe the refit=1 cache.

### Piece 4 — Per-chunk H2D promotion: verification (no code changes)

The streamed ensure-mode path already exists on this branch. Verified assertions:

1. **Dispatch** (`moe_w2_cubit.py:3029–3043`): when `use_w8` and
   `_PREFILL_FP4_ENSURE` (env `VLLM_MOE_W2_PREFILL_FP4=ensure`) and not capturing,
   `tier.ensure_resident(layer_key, topk_ids.view(-1))` is called at line 3042
   BEFORE `slot_row = tier.slot_table[layer_key]` at line 3043 — the pool is
   populated before the desc build reads it.

2. **Stream ordering** (`moe_w2_delta.py::ensure_resident`, lines 1400–1461):
   line 1415 does `self._stream.wait_stream(main)` ordering all pool `copy_`s
   after every GEMM already enqueued on the forward stream; line 1419 `ev.synchronize()`
   blocks the host until bytes land; `slot_table[li, ei] = slot` at line 1458 is
   written on the forward thread after synchronize, before the desc build launches.
   `rows_for` at line 1447 passes `scan=True` (protects the store's decode hot set).

3. **CUDA-graph safety** (`moe_w2_cubit.py:3008–3009`): `use_w8` requires
   `prefill` (T > `_PREFILL_T`=96, defined at line 145 — never a captured shape);
   the capturing branch at line 3031 calls `tier.notify_capture()` instead of
   ensure. Quality mode chunk is 4096 ≫ 96, confirming no graph capture overlap.

4. **Layer-pin rotation** (`moe_w2_delta.py:1423`): `_layer_pins.clear()` at the
   top of each `ensure_resident` call releases the previous layer's pins; with
   piece 2's emergency bypass, pass 3 of `_take_slots_batch` rotates the whole
   pool every layer.

### Piece 5 — Chunk size + H2D cost management (numbers, no code)

- `BATCHED_TOKENS=4096` (chunk); MTP must be off (>1024 OOMs at startup with MTP).
- Pool: `FP8_DELTA_GB=1.5` = 256 × 6 MiB FP4-store slots = one full layer union @ TP2.
  Deferred allocation clips to post-KV free VRAM minus 3 GiB reserve.
- `UTIL=0.90`: post-KV free ≈ total×(1−UTIL) ≈ 4.8 GiB/card; must cover pool
  1.5 GiB + 3 GiB reserve. At UTIL=0.98 only ~1 GiB remains and pool clips to ~0.
- H2D cost: 43 MoE layers × 256 × 6 MiB ≈ 64.5 GiB/chunk ÷ ~20 GB/s ≈ 3.2 s/chunk;
  131K/4096 = 32 chunks → ~1.5–3 min/prefill. fp4-store is the perf and RAM choice
  (fp8-store ~368 GiB/rank would not fit the 428 GiB container cap).
- Pinned host arena: ~65 GiB/rank at TP2 fp4-store (~129 GiB total).

### Piece 6 — Serve config (`docker/serve_sm89_ds4.sh`)

`QUALITY_PREFILL=1` preset block inserted after `set -euo pipefail`, before all
default variable assignments (load order: preset uses `${VAR:-...}` so explicit
user env wins; later defaults also use `${VAR:-...}` so they keep preset values).
Wire-up: `mkdir -p "$CACHE/$PLANES_SUBDIR"`, `-v "$CACHE/$PLANES_SUBDIR":/plane-cache`,
`$QP_ENV \` added to docker run env args next to `$DELTA_ENV`, start banner extended
with `quality-prefill=$QUALITY_PREFILL`, and quality-mode healthy-boot markers
added. FP8_DELTA_GB comment notes the deferred post-KV allocation. TECHNICAL NOTES
section gains a STREAMED QUALITY PREFILL paragraph with UTIL rationale, H2D cost,
arena sizing, planes-qp cache isolation, and the obsolescence of the two-boot flow.
