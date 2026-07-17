# Ada (sm_89) port — 2-bit MoE experts without QMMA

The SM120 story compresses routed experts to 2-bit codebooks + UE8M0
block-32 scales and executes them with hand-written QMMA.SF SASS. Ada
(RTX 4090 / RTX 2000-6000 Ada) has FP8 e4m3 tensor cores but **no QMMA,
no QMMA.SF, no FP4/NVFP4** — so the port is an *emulation*, not a
retarget: the 2-bit codes decode to BF16 in registers and a standard
BF16 MMA runs with both scales folded as explicit multiplies. What the
compression buys — experts read from HBM at 2 bits/elem instead of
BF16's 16 — is a property of the *planes*, not of QMMA, and survives
unchanged. Only the tensor-core instruction changes.

## What ships

| piece | where |
|---|---|
| Triton kernel (decode + prefill, one kernel) | `kernels/triton/moe_w2_sm89.py`, mirror of `vllm/.../utils/moe_w2_sm89.py` on the fork branch |
| loader arch branch (`cap == (8, 9)`) | `moe_w2_cubit._ensure_ready` (fork branch → generated patch) |
| arch-blind dispatch | `moe_w2_cubit._launch`: callable → Triton, `c_void_p` → cubin |
| serving image | `Dockerfile.sm89-v024` |
| CPU golden (runs anywhere) | `kernels/gen/moe_w2_sm89_cpu_check.py` |
| on-silicon op gate | `kernels/gen/moe_w2_check_sm89.py` (same math + verdict as `moe_w2_check.py`) |

Everything else in the moe_w2 path — the three MoE method hooks, the
forward op, the desc-build Triton kernels, the workspace allocator, the
deterministic unpermute, the host packers — is arch-agnostic and
untouched.

## The kernel in one paragraph

Grid keeps the cubin geometry: one program per (16-row N-tile, pair);
each pair's 6×u64 descriptor `{a, as, b, bs, c, m_rows}` is read on
device (`m_rows == 0` → early EXIT, exactly like the cubin). The K loop
walks one 32-group per iteration: 8 fragment-major code bytes per N-row
expand to 32 codes, A is gathered *through the same byte-major k
permutation* so every code byte is read once; codes map to {−4,−1,1,4}
and a8 e4m3 casts to BF16 — both exact in bf16 — so the
[16,32]×[32,16] BF16 dot produces exact products in the FP32
accumulator. The per-32 scales, `a_s[m]` (f32) and `2^(sb[n]−127)`
(exponent-field bitcast, exact for every byte), fold as one [M,N] outer
product per group. Output rows store BF16 at the desc `c` pointer,
masked to `m_rows`. No atomics, fixed schedule → byte-deterministic
run-to-run (the 4-run half of the op gate).

Decode (mblock 4) and prefill MC4 (mblock 16) differ only in `m_rows`,
so ONE kernel registers under both mandatory `_fns` keys (`w2`,
`w2mc4`). AFRAG is a QMMA A-fragment LDG.128 trick — load-issue-bound on
SM120, meaningless here — prefill falls back to `w2mc4`. The w4/w4q FP4
delta tiers are not ported: **serve with `VLLM_MOE_W2_DELTA=0`**
(`_require_kernels` fails loudly otherwise).

## Ada serving constraints

- `--kv-cache-dtype fp8` — the NVFP4 KV cache is Blackwell-only.
- No sparse-MLA SM120 attention: DS4/GLM attention routes through stock
  FlashInfer/FlashAttention (the family-120 gates make the SM120 fixes
  inert on Ada).
- `VLLM_MOE_W2_DELTA=0` (see above). `VLLM_MOE_W2_AFRAG` is ignored.
- Triton JIT-compiles per K on first use; vLLM's warmup covers this
  before cudagraph capture.

## Observability — read this before your second deploy cycle

Deployment cycles cost more than log lines, so the Ada path now reports
its own health at startup. What a HEALTHY boot logs (EngineCore
process):

1. `moe_w2_cubit: sm_89 Triton emulation ready on <device> (triton X.Y):
   self-test worst_rel=…e-03 (gate 2.5e-2)` — a 1-pair op-level GEMM ran
   against a torch reference **on your silicon** at engine init. If
   Triton miscompiles on this device/driver, the boot fails here with an
   attributed message instead of serving garbage.
   (`VLLM_MOE_W2_SM89_SELFTEST=0` to skip.)
2. `moe_w2 planes: X.XX GiB/layer x N layers = Y GiB …` — the plane
   budget, projected at the FIRST plane build. GPU-resident mode on a
   card that cannot possibly hold the planes fails **here** with the
   remedy, instead of ~40 layers later as a bare CUDA OOM. This line is
   also your "moe_w2 is ACTIVE" beacon — the stock backend banner
   (e.g. `Using 'MARLIN' Mxfp4 MoE backend`) still prints and does NOT
   mean the moe_w2 path is off.
3. `moe_w2: env VLLM_MOE_W2_… is not read by any moe_w2 module and does
   NOTHING — did you mean …?` — typo guard over every `VLLM_MOE_W2*`
   env. The moe_w2 knobs are read with `os.getenv`, NOT registered in
   `vllm/envs.py` — so vllm's own `Unknown vLLM environment variable`
   warning fires even for **correct** names. Ignore that one; trust this
   one.

Triage recipe when the engine dies at startup:

```bash
docker logs <name> 2>&1 | grep -B2 -A30 -E "EngineCore.*(Error|Traceback|CRITICAL)|moe_w2"
docker inspect <name> --format '{{.State.ExitCode}} {{.State.OOMKilled}}'   # host OOM kill?
```

The APIServer traceback ending in `wait_for_engine_startup` only says
"the engine died" — the real error is in the `(EngineCore pid=…)` lines
above it. Decode its last line:

- `Failed core proc(s): {'EngineCore_0': …}` — the engine raised a
  Python exception; its traceback IS in the log, scroll up.
- `Failed core proc(s): {}` (**empty set**) — the engine died with NO
  Python exception; there is no traceback to find. Two suspects:
  host OOM-killer (`sudo dmesg -T | grep -iE "oom|killed process" |
  tail -5`, plus `free -g`; `docker inspect --format '{{.State.OOMKilled}}'`
  only catches container-limit kills, not host ones) or a native crash —
  the image bakes `PYTHONFAULTHANDLER=1`, so segfault/abort stacks land
  in `docker logs`.

Before reading any log, confirm WHICH build produced it: `docker exec
<name> cat /opt/moet-checks/SOURCE.txt` prints the vllm fork SHA the
image's patch was generated from (file absent = pre-observability
image — rebuild). A relaunch on a stale image looks identical in the
log until the first moe_w2 line, and has already burned a debug cycle.

VRAM reality on Ada: DS4-Flash planes are ~1.73 GiB/layer × 43 layers ≈
**74 GiB** — GPU-resident planes need the 96 GB SM120 board. On 24–48 GB
Ada cards you MUST run the base cache: `VLLM_MOE_W2_BASE_CACHE_GB=<pool
GiB>` keeps planes in pinned host RAM and streams a GPU slot pool
(start around 4–8 GiB on a 4090 and watch the miss/replay counters), or
shard with TP, or serve a smaller model. The same budget line covers the
host side: when the pinned projection exceeds 90% of `MemAvailable` the
boot warns (a pinned arena that outgrows host RAM dies as a bare
cudaHostAlloc failure — or the OOM-killer takes the engine with no
traceback at all); cap it with `VLLM_MOE_W2_BASE_RAM_GB` and give the
spill an NVMe home via `VLLM_MOE_W2_STORE_DIR`. A corrected, annotated
launcher for exactly this configuration ships as
`docker/serve_sm89_ds4.sh`.

Env cheat sheet (spellings that have already burned a deploy cycle):
`VLLM_MOE_W2_PLANES_CACHE` (plural — persists built planes across
boots), `VLLM_MOE_W2_DELTA_GB=0` (valid way to disable the delta tier;
`VLLM_MOE_W2_DELTA=0` also works), `VLLM_MOE_W2_BASE_CACHE_GB` (the base
cache), `VLLM_MOE_W2_STORE_DIR` (expert pack store). First boot without
a planes cache re-quantizes the whole checkpoint — expect a long load;
mount `VLLM_MOE_W2_PLANES_CACHE` at a persistent path to pay it once.

In-container op gate on the real card (bake ships at /opt/moet-checks):

```bash
docker run --rm --gpus all --entrypoint bash vllm-moet-sm89:v024 -c \
  'for K in 512 1024 2048 4096 6144 7168; do \
     K=$K python3 /opt/moet-checks/moe_w2_check_sm89.py || exit 1; done'
```

## Validation state (be honest with yourself)

CPU golden **PASS** on this repo for all K ∈ {512, 1024, 2048, 4096,
6144, 7168}: byte-exact fragment-major addressing against
`pack_fragment_major`/`pack_scales` (0 mismatches over every row/k) and
worst_rel 2.5–3.2e-3 vs the f32 reference — the same band the SM120
cubins measure — 4-run identical. `tools/test_moe_w2_planes.py` (host
packing) still ALL PASS.

Pending an actual sm_89 card (none on the dev box that produced this
port): `kernels/gen/moe_w2_check_sm89.py` for every K (gate:
worst_rel < 2.5e-2 AND 4 runs byte-identical), then
`tools/test_moe_w2_forward.py` (rel < 0.06, cos > 0.999) and
`tools/test_moe_w2_prefill_fp4.py` with `VLLM_MOE_W2_DELTA=0`, then a
`tools/bench_tok.py` decode benchmark vs stock FP8 to quantify the
HBM-read win on memory-bound shapes.
