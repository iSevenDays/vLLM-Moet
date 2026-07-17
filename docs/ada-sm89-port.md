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
