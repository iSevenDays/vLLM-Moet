---
title: "Native kernel fast path with parity gate and tier-restricted dispatch"
date: 2026-07-24
category: architecture-patterns
module: moe_w2_decode_kernel
problem_type: architecture_pattern
component: tooling
severity: high
applies_when:
  - "Replacing a Triton/generic kernel on a GPU-architecture-specific hot path with a hand-written native CUDA kernel (e.g. Ada sm_89 e4m3 tensor cores, reachable from plain CUDA C++ without hand-written SASS)"
  - "The native kernel is validated for only a bounded input envelope (specific (K, N) shapes, m_rows <= 4) and silently writes nothing outside it"
  - "A faster kernel must ship alongside an existing correct-but-slower implementation without risking silent output corruption in production"
  - "Dispatch must key off an exact tier+shape registry match because a sibling tier (prefill) carries out-of-envelope shapes through the same dispatcher"
  - "The kernel launch must stay CUDA-graph-capture-safe"
related_components:
  - "kernels/cuda/moe_w2_sm89_decode.cu"
  - "overlay/vllm/vllm/model_executor/layers/quantization/utils/moe_w2_cubit.py"
  - "overlay/vllm/vllm/model_executor/layers/quantization/utils/moe_w2_sm89.py"
tags: [cuda-kernel, sm89-ada, tensor-core-mma, fp8-e4m3, moe-w2-gemm, native-kernel-fallback, parity-gate, tier-restricted-dispatch]
---

# Native kernel fast path with parity gate and tier-restricted dispatch

## Context

vLLM-Moet serves DeepSeek-V4-Flash on 2× RTX 4090 D (sm_89/Ada, TP2, 262144
max context) using the 2-bit "W2" MoE expert planes. The project's fast path
is hand-written SASS cubins, but those exist only for sm_120 (`QMMA.SF`); Ada
ran a Triton emulation of the same GEMM. This left throughput on the table:
Ada's e4m3 tensor core **is** reachable from plain CUDA C++ via inline PTX
(`mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`, compiling to SASS
`QMMA.16832.F32.E4M3.E4M3`) — no SASS assembler needed. A hand-written CUDA
decode kernel was integrated behind the existing dispatcher, keeping the
Triton emulation as a permanent fallback. Kernel facts, shapes, and speedups
are catalogued in `kernels/MANIFEST.md` ("Ada (sm_89) NATIVE decode cubin");
this doc captures the reusable integration pattern and its pitfalls.

## Guidance

**Pattern: native cubin fast path = opportunistic loader + boot parity gate
+ tier-restricted exact-shape dispatch + emulation fallback.**

1. **Write the narrowest kernel that is true to its call site, and encode
   the envelope as a hard guard.** The decode tier guarantees `m_rows <= 4`;
   the kernel enforces it:

   ```c
   const int m_rows = static_cast<int>(descriptor[5]);
   if (m_rows < 1 || m_rows > 4) {
     return;   // out-of-envelope: writes NOTHING (see dispatch hazard below)
   }
   ```

2. **Statically gate the build artifact.** `nvcc` silently falling back to
   FMA math would still be "correct", just slow. The build script greps the
   SASS and fails the build if the tensor-core instruction is absent:

   ```bash
   if ! grep -E '(Q|H)MMA\..*(E4M3|FP8)' "${sass_file}" | head -n 24; then
     echo "error: no E4M3/FP8 tensor-core instruction found in cubin" >&2
     exit 1
   fi
   ```

3. **Load opportunistically, never fatally.** Missing cubin, disabled env
   knob, driver refusing the module, missing symbol, failed parity — all
   degrade to the emulation with a clear log line, never a dead engine:

   ```python
   except Exception as e:  # noqa: BLE001
       _native_fns.clear()
       logger.warning("moe_w2_cubit: sm_89 native decode cubin unavailable "
                      "(%s); decode stays on the Triton emulation", e)
   ```

4. **Parity-gate activation at every boot, per shape, on the deployed
   silicon.** A build-time check cannot catch driver/runtime regressions or
   a stale cubin that no longer matches the plane layout. The loader runs a
   random op-level case against a torch reference for each registered shape
   (gate 2.5e-2) before the shape enters the dispatch registry
   (`moe_w2_sm89.native_self_test`).

5. **Dispatch on tier identity AND exact shape — never shape alone.** This
   is the load-bearing correctness gate. The kernel's `m_rows <= 4`
   precondition holds only on the decode tier ("w2", 4-token pair blocks);
   the prefill tier ("w2mc4") pushes `m_rows` up to 16 through the *same*
   dispatcher, and the kernel would return without writing — leaving stale
   workspace values in the output rows, i.e. **silent corruption**, not a
   crash. `n_rows` also doubles as the C row stride, so both K and N must
   match the registry key exactly:

   ```python
   def _launch(tier, K, desc, n_rows, pairs, stream):
       if _native_fns and tier == "w2":
           nfn = _native_fns.get((K, n_rows))
           if nfn is not None:
               _launch_sm89_native(nfn, desc, n_rows, pairs, stream)
               return
       fn = _fns[(tier, K)]   # everything else: Triton emulation
   ```

6. **Reuse the proven launch mechanics.** ctypes `cuLaunchKernel` on the
   *current torch stream* is intercepted by CUDA stream capture, so full
   cudagraph decode keeps working with zero special-casing (same launch
   class the sm_120 cubin path already used). Verify explicitly with a
   `torch.cuda.CUDAGraph` capture+replay test rather than assuming.

### Validation ladder (run all of it before trusting a new kernel)

Static SASS inspection (instruction present, registers, zero spills) →
standalone driver-API sweep (`kernels/gen/moe_w2_sm89_native_check.py`: all
M × pairs × seeds × shapes × GPUs, bit-determinism probe) → in-image
integration test (mixed live/dead descriptor rows, sentinel checks on
unwritten rows, three-way parity native/emulation/torch, graph
capture+replay) → per-boot parity gate → end-to-end A/B benchmark. "The op
benchmark looks good" alone is how corruption reaches production.

### Pitfalls encountered

- **bf16 sentinel rounding.** A masked-write test filled a bf16 buffer with
  sentinel `777.0`; bf16 (8 mantissa bits) silently rounds it to `776.0`,
  so "row still equals sentinel" failed spuriously. Pick a sentinel exactly
  representable in the buffer's dtype (`768.0` = 1.5 × 2⁹).
- **Generated artifacts inside `overlay/`.** Running `python3 -m py_compile`
  on overlay files left `overlay/.../__pycache__/*.pyc`, which
  `tools/gen_patches.py` swept up as bogus patches and failed `--verify`.
  Sweep `find overlay -name __pycache__ -exec rm -rf {} +` and regenerate;
  never leave generated files in `overlay/`.
- **Mirror discipline.** `kernels/triton/moe_w2_sm89.py` must stay
  byte-identical to the overlay copy — `cp` + `cmp` after every overlay
  edit.
- **Projection vs measurement.** The packed-KV backport projected −8% page
  stride from layout math; the booted server measured −3.6%. Only logged
  numbers from a real boot count.

## Why This Matters

A quantized-MoE stack on non-primary-target hardware always has a gap
between the reference SASS kernel and a portable emulation. Hand-written
PTX closes most of it (measured here: 3.1–4.7× at the op level; end-to-end
+27% single-stream decode, +66% at concurrency 4, +21% at 55k context,
greedy output bit-deterministic) — but only defensively integrated: the
emulation stays the correctness backstop, and the dispatcher must be unable
to hand the kernel a shape or tier it wasn't written for. The failure mode
being guarded is *silent wrong tokens*, not a crash, which no amount of
post-hoc monitoring reliably catches.

## When to Apply

- Porting a datacenter-GPU tensor-core kernel (QMMA.SF/FP4-class) to a
  consumer GPU with a different but native tensor-core instruction.
- Adding any fast path with a narrower precondition (fixed M, fixed K/N)
  to a dispatcher that also serves wider-precondition tiers.
- Any masked/partial-write kernel tested via sentinel-filled buffers in a
  reduced-precision dtype — verify the sentinel round-trips exactly first.
- Any `overlay/` + generated-patch repo workflow — sweep `__pycache__`
  before regenerating patches.

## Examples

- Kernel: `kernels/cuda/moe_w2_sm89_decode.cu` (PRMT-LUT in-register 2-bit →
  e4m3 decode, weights as MMA A operand, FP32 per-32-group scale fold; two
  fixed shapes = the DS4-Flash TP2 decode GEMMs).
- Build + static gate: `kernels/cuda/build_moe_w2_sm89_decode.sh`; committed
  artifact `kernels/cubins-sm89/moe_w2_sm89_decode.cubin`.
- Loader/dispatcher: `moe_w2_cubit.py` — `_load_sm89_native`,
  `_launch_sm89_native`, `_native_fns`, fast path in `_launch`. Kill switch
  `VLLM_MOE_W2_SM89_NATIVE=0` (launcher knob `SM89_NATIVE`).
- Reference + self-tests: `moe_w2_sm89.py` — `_op_case`, `self_test`,
  `native_self_test`.
- Boot marker: `sm_89 NATIVE decode cubin ACTIVE for (K, N)=[(1024, 4096),
  (4096, 2048)]`; fallback logs "decode stays on the Triton emulation".

## Related

- `kernels/MANIFEST.md` § "Ada (sm_89) NATIVE decode cubin" — canonical
  kernel facts, shapes, measured speedups (do not duplicate; cite).
- `kernels/MANIFEST.md` § "Ada (sm_89) Triton emulation" — the fallback.
- `docs/ada-sm89-port.md` — the Ada port record (refreshed 2026-07-24 to
  cover the native decode path and its boot markers).
- Rejected alternatives on this hot path (measured, see NEXT.md /
  benchmarks): wider Triton BLOCK_N 32/64 (17–41% slower), Triton FP8-MMA
  variant (0–3%), W4Q split-FP4 (slower + incompatible with scale refit),
  torch.compile (unsupported on this path). MTP K=2 wins single-stream but
  loses concurrency — kept K=1.
