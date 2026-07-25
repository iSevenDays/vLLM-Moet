# Handover: validate the streamed quality-prefill mode (QUALITY_PREFILL=1)

Audience: the AI agent operating on the serving box (2× 48 GiB Ada, TP2,
DeepSeek-V4-Flash mxfp4). Everything below is runnable from the repo root on
that host. Branch: `feat/moe-w2-sm89-native-decode`. The change set is the
two commits titled "Streamed full-precision prefill ..." (code) and
"serve_sm89_ds4: QUALITY_PREFILL preset ..." (ops); full changelog in
PROGRESS.md ("Streamed full-precision prefill quality mode (2026-07-25)").

## What this mode is (one paragraph)

Long prefills activate ~all 256 experts/layer, so no resident FP8 pool can
cover them (measured: 128-slot pool = ~1% coverage, needle onset unchanged at
16K). The quality mode instead streams: for EACH prefill chunk and EACH MoE
layer, the routed experts' full-precision planes are promoted host→GPU
(`ensure_resident`, synchronous, emergency-evicting the previous layer's
slots) before the GEMMs run — so the whole prefill computes at FP8 precision
from FP4-STORAGE slots (checkpoint e2m1 nibbles, decoded to exact e4m3 bit
patterns in registers by `moe_w8fp4_mm_sm89`). Decode is untouched (2-bit
base + native cubin). The GPU pool (1.5 GiB = one 256-expert layer union at
TP2) is allocated AFTER KV-cache sizing (deferred; the old build-time OOM and
the two-boot workaround are gone). MTP is off in this mode; prefill costs
~1.5-2.5 min per 131K (H2D-bound: 43 MoE layers × 256 × 6 MiB ≈ 64.5 GiB
worst case per 4096-token chunk).

## Step 0 — sync + rebuild the image (the patches are baked in)

```bash
git fetch && git checkout feat/moe-w2-sm89-native-decode && git pull
python3 tools/gen_patches.py --verify && python3 tools/gen_patches.py --check
DOCKER_BUILDKIT=1 docker build -f Dockerfile.sm89-v0251 -t vllm-moet-sm89:v0251 .
```

If `--verify`/`--check` fail: the vllm/ baseline checkout must be at the
v0.25.1 tag (`git -C vllm checkout v0.25.1`) — do NOT regenerate patches
against any other baseline.

## Step 1 — CPU goldens (no GPU needed; run in any torch env / the image)

```bash
docker run --rm -v "$PWD/kernels:/k" --entrypoint python3 vllm-moet-sm89:v0251 \
  /k/gen/moe_w8fp4_sm89_cpu_check.py       # NEW: fp4-storage addressing+decode
docker run --rm -v "$PWD/kernels:/k" --entrypoint python3 vllm-moet-sm89:v0251 \
  /k/gen/moe_w8_sm89_cpu_check.py          # regression: fp8-store still golden
```

Expected: per-K `PASS` lines, `layout_mismatches=0`, `distinct=1`,
`RESULT: PASS`; the fp4 golden additionally prints a 16-nibble decode-table
proof (patterns 0x00/0x30/0x38/0x3c/0x40/0x44/0x48/0x4c ± sign). Any FAIL =
STOP, report the full output; do not boot the quality mode.

## Step 2 — quality-mode boot

```bash
QUALITY_PREFILL=1 EXTRA_DOCKER_ENV="-e VLLM_MOE_W2_FP8_TRACE=1" \
  ./docker/serve_sm89_ds4.sh
docker logs -f moet 2>&1 | grep -E 'moe_w2|o_proj'
```

Notes:
- FIRST quality boot re-quantizes (SCALE_REFIT=0 changes plane bytes) into a
  SEPARATE cache dir (`planes-qp`) — 15-20 min, once. Later boots stream from
  cache. The refit=1 cache in `planes/` is untouched.
- The preset pins MAXLEN=131072, UTIL=0.90, BATCHED_TOKENS=4096, MTP_TOKENS=0,
  SCALE_REFIT=0, FP8_DELTA_GB=1.5, VLLM_MOE_W2_FP8_STORE=fp4,
  VLLM_MOE_W2_PREFILL_FP4=ensure. Any explicitly-exported env still wins.

## Step 3 — boot log markers, in order (all must appear)

1. `moe_w2 delta tier: auto-sizing deferred until after KV-cache allocation`
   (the w8 tier at build — pool NOT allocated during the plane build)
2. `moe_w8_sm89 self-test` line with `worst_rel=...` passing (this is the
   fp4-storage kernel op gate when FP8_STORE=fp4 — same log site)
3. ONE warning `w8 POOL parity gate could NOT run at build ... auto-deferred
   cold pool` — EXPECTED in this mode, not a failure
4. `moe_w2: FP8 BOOT GUARD: delta tier staged (43 layers); pool allocation
   DEFERRED to post-KV finalize_auto` (info, at the profile forward)
5. `moe_w2 delta tier AUTO: 256 slots x 6.0 MiB (1.50 GiB pool; ...)`
   — if the slot count is well below 256 or the "clipping to the budget"
   warning appears, post-KV free VRAM is short: lower UTIL (0.88) or MAXLEN
6. `moe_w2_cubit: sm_89 FP8 POOL parity OK ... through pool slot 255`
   (the deferred gate; a RAISE here aborts boot = slot-sizing bug, report it)
7. decode path unchanged: the usual `sm_89 NATIVE decode cubin ACTIVE` parity
   lines must still appear
8. on the first real prefill: `moe_w2 prefill-FP4/FP8 ensure mode: eager
   chunk working sets fetched...` and (FP8_TRACE=1) up to 8 lines
   `moe_w2 w8 prefill: layer .. pool_slots=256 resident_this_layer=.. T=4096`
   — `resident_this_layer` should be ≈ the layer's routed-expert count
   (approaches 256 deep into a long prompt); persistent low values = the
   ensure path is not firing (check VLLM_MOE_W2_PREFILL_FP4=ensure landed:
   `docker exec moet env | grep MOE_W2`)

## Step 4 — functional validation

```bash
curl -s http://localhost:8001/v1/models | head -3        # engine up
# short-prompt sanity (decode path):
curl -s http://localhost:8001/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"2+2?"}],"max_tokens":16}'
```

Then the decisive test — the long-context needle at the KNOWN failure onset
(bare 2-bit prefill derails from 16K): run the existing harness
(`tools/needle_probe.py` / `tools/needle_full.py` — check `--help`; prior
sessions ran the 16K and 131K variants) at 16K first, then 32K/64K/131K.
Success = correct retrieval at 16K where the 2-bit baseline failed, and
coherent long-context behaviour at 131K. Record prefill wall-time per length
(expect ~1.5-2.5 min at 131K; the first chunks are faster).

Decode regression check: a short-prompt decode tok/s sample should match the
MTP-off baseline (~38 tok/s single-stream) — decode never touches the w8 path.

## Triage

| Symptom | Meaning | Action |
|---|---|---|
| boot RAISE `VLLM_MOE_W2_FP8_STORE=fp4 requires VLLM_MOE_W2_SCALE_REFIT=0` | refit leaked in (env override) | unset SCALE_REFIT / keep preset |
| `w8 POOL parity FAILED` (deferred gate) | slot/stride sizing bug — real | STOP, report worst_rel + log context |
| AUTO line `pool disabled (0 slots ...)` | post-KV budget exhausted | UTIL=0.88, or MAXLEN down, or FP8_DELTA_GB=auto |
| `moe_w8_sm89 self-test FAILED` | Triton codegen/addressing on this driver | run the fp4 CPU golden; report triton/torch/driver versions |
| warning `pool too small for one prefill layer (N unfetched)` | pool < layer union (clipped) | raise budget (UTIL down) — quality silently degrades otherwise |
| needle still derails at 16K with markers all green | coverage is live but insufficient/mis-hypothesized | capture FP8_TRACE lines + `[w8]` KPI lines, report |
| boot OOM at profile/capture | 4096-chunk workspaces + reserve too tight | confirm MTP_TOKENS=0 landed; try CUDAGRAPH_SIZES=1,2,4 |

## Rollback

`./docker/serve_sm89_ds4.sh` without QUALITY_PREFILL is byte-identical to the
pre-change production path (FP8 off by default, planes/ cache untouched,
BATCHED_TOKENS=1024, MTP on). No state to clean up other than the optional
`planes-qp` dir.

## Report back (checklist)

- [ ] CPU goldens: both RESULT lines
- [ ] Boot markers 1-8: present/absent, with the AUTO slot count and parity worst_rel
- [ ] First-boot requant duration; warm-boot duration from planes-qp
- [ ] Needle: pass/fail per length (16K/32K/64K/131K) + prefill wall-times
- [ ] Decode tok/s vs MTP-off baseline
- [ ] `nvidia-smi` per-card VRAM at steady state
- [ ] Any warning lines matching `moe_w2` not listed above
