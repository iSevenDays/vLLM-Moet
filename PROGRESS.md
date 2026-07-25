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
