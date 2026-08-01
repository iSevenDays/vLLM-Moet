# Dig — decode-specific path (agent 4)

**Leading sm_89-specific mechanism found: decode indexer logits run in TF32,
prefill runs in fp32.** This is the only decode-path attribute that differs
between passing sm_120 and failing sm_89, and it has a **zero-code, one-boot
falsifier.**

Source: read-only sub-agent over the decode kernels, 2026-08-01.

## PRIMARY (sm_89-exclusive): decode logits = TF32, prefill = fp32
`overlay/vllm/vllm/v1/attention/ops/triton_paged_mqa_logits_dsv4.py:185`
```python
s = tl.dot(q, tl.trans(k), allow_tf32=True)   # decode indexer score
```
- Q/K dequantized to **fp32** (FP8-E4M3 integer decode, exact) then dotted in
  **TF32** (Ada's only fp32-input MMA = TF32, 10-bit mantissa; the comment at
  `:183-184` admits this).
- **Prefill** scores in **pure fp32** (`_torch_fp8_mqa_logits`, `deep_gemm.py:587`,
  fp32 einsum) because on sm_89 `_use_torch_fallback()` is True.
- **sm_120** decode uses native DeepGEMM (`_use_torch_fallback()` False) and
  **never enters this Triton port** — so the fp32/TF32 asymmetry is sm_89-only.
- Boot self-test `worst_row_rel = 4.691e-4` IS the TF32 rounding.

**Why it matches every datum:**
- T4: digit selected at PREFILL (fp32, 20/21 layers). At DECODE, TF32 can push a
  borderline digit (fp32 rank ~510) past the `index_topk=512` cut → its KV never
  loaded during the digit-emitting steps → clean prior-fallback (§3c: true token
  absent from top-20).
- `index_topk=2048` rescues 6/6 (§3a): a TF32-perturbed rank 510→513 is still
  inside 2048. k=512 loses it. Exactly the observed pattern.

## FALSIFIER (no code change, one boot)
`VLLM_DSV4_PAGED_MQA_LOGITS_FORCE_TORCH=1` (`triton_paged_mqa_logits_dsv4.py:84-86`)
disables the Triton port → decode logits use the **fp32** `paged_mqa_logits_torch_ref`.
Run the 8K `ask` needle: **if it PASSES (or the digit rank stays ≤512 at decode),
TF32 is confirmed as the mechanism.** This is the cheapest possible decisive test.

## FIX (if confirmed)
`allow_tf32=False` at `triton_paged_mqa_logits_dsv4.py:185` (and the sibling
sparse-MLA dots at `:282/:291/:316/:325`) on sm_89 — full fp32 dot, matching
prefill. Throughput cost TBD (decode logits are a small fraction of step time).

## SECONDARY (amplifier, not sm_89-specific): `persistent_topk` tie-break
Decode top-k (`persistent_topk.cuh:574-577` threshold-bin fill via `atomicAdd`;
`:848-855` radix extras silently dropped `if (pos < TopK)`) is **non-deterministic**
and differs from prefill's stable insertion sort (`sampler.cu:510`). Both sm_89 and
sm_120 take this branch, so it isn't the differentiator — but once TF32 pushes the
digit into the threshold bin, this tie-break can drop it. The §4 "top-k verified"
test used high-contrast scores, never exercising a borderline rank-512 tie.

## Decode-trace crash root-caused (explains T3)
`_trace_indexer_rank_decode` (`sparse_attn_indexer.py:176-177`) does `.item()` host
syncs; under `VLLM_USE_BREAKABLE_CUDAGRAPH` decode runs inside CUDA-graph capture
where host sync is illegal → context-poisoning → boot crash (the `int(lens.max())`
class the port's own docstring warns about). **Fix:** on-device row selection +
deferred host-emit (stash into a device ring buffer during capture, drain after
replay) — same capture-safety discipline the paged kernel already uses.

## Other findings
- **Value readout is innocent:** decode sparse-MLA attention (`_dsv4_gather_k`)
  reads compressed KV at the SAME precision as prefill (same kernel, fp32 dequant).
  The digit's value is read fine **iff its index is selected** → mechanism is
  decode SELECTION, not value.
- **Decode≠prefill selection (Finding 5):** decode re-runs the full indexer+topk
  every step; **no carry-over** of the prefill candidate set. So T4 (prefill) does
  not determine decode selection — the digit can be prefill-selected yet
  decode-dropped. This is why H2 (prefill exonerated) doesn't close the case.
- **Coverage gap:** self-test only exercises `block_size=64`; production uses
  `block_size=256` (BLOCK_N=256, ~128 KiB K tile vs Ada's 99 KiB/SM smem ceiling
  the sparse-MLA port avoids). Precision is scale-invariant (per-block), so the
  4.691e-4 is representative, but BLOCK_N=256 occupancy/spill on sm_89 is unvalidated.

## Net
This **refines H2**: prefill selection is fine, but **decode selection (TF32-scoring)
drops the digit** — a selection mechanism, just the never-observed decode one. It
reconciles T4 (prefill exonerated) with the index_topk=2048 rescue and the
generation-failure symptom. The `FORCE_TORCH=1` boot is the decisive next test.
