# Dig — post-attention MoE expert value path (agent 3)

**T8 (routing kernel) EXONERATED — arch-independent, numerically faithful. The
only surviving suspect in the MoE scope is the sm_89-native decode W2 cubin
(hand-written SASS, full op-gate never run), with a zero-code falsifier.**

Source: read-only sub-agent over the MoE routing + FP4 expert path, 2026-08-01.

## FINDING 1 — routing kernel `topk_hash_softplus_sqrt` (T8): CLEARED
`vllm/csrc/libtorch_stable/moe/topk_softplus_sqrt_kernels.cu`. float32 throughout;
softplus+sqrt (`__logf(1+__expf)/beta`, beta=1, threshold=20) ≡ checkpoint
`F.softplus(x).sqrt()` (`model.py:576`); bias added for selection then
**subtracted before storing the weight** (`:384-386`) → unbiased `sqrt(softplus)`
weights ≡ checkpoint. The only `__CUDA_ARCH__` guards are `>=900` for PDL
**scheduling**, not numerics. Selection dispatch (`dsv4_topk`) has **no arch
check** → identical on sm_89 and sm_120. **A bug here would manifest identically
on sm_120.** T8 is now numerically verified.

## FINDING 2 — FP4 expert W2: faithful on inspection; native decode cubin is the one sm_89-unique, under-validated binary
- Triton emulation `moe_w2_mm_sm89`: fp8→bf16 (exact), {-4,-1,1,4} levels→bf16
  (exact), **fp32 accumulator**, per-32-group scale fold (bit-faithful to ref),
  bf16 output. The 2.6e-3 self-test = bf16 output rounding.
- **sm_89 native decode cubin** (`moe_w2_sm89_decode.cubin`, hand-written SASS,
  `mma.sync...e4m3.e4m3.f32` + fp32 fmaf): serves BOTH decode expert shapes at
  TP2. Numerically faithful on inspection (fp32 fmaf, exact fp8 products, correct
  UE8M0 scale fold, sb=0 subnormal guarded). **BUT its own header has a TODO:
  "Run kernels/gen/moe_w2_sm89_native_check.py ... before integrating" — the full
  op-gate was apparently never run**, only a single-seed boot self-test (loose
  2.5e-2 gate). Could harbor a systematic bias on real digit-critical activations
  (e.g. outlier channels) that a single random case misses.

## FINDING 3 — final norm + LM head (FP8 block-scaled Triton): low
fp32 accumulation, exact fp8 products, bf16 output — arch-comparable to sm_120's
Cutlass/DeepGEMM path to ~bf16 rounding. Would perturb all tokens uniformly, not
selectively digits.

## Suspect + falsifier
Most-suspect sm_89 spot in scope: the **native decode W2 cubin**.
**Zero-code one-boot A/B:** `VLLM_MOE_W2_SM89_NATIVE=0` (`moe_w2_cubit.py:420`)
→ decode expert GEMM falls back to the self-tested Triton emulation. If the 8K
`ask` digits recover → the cubin is corrupting decode expert output; confirm with
the full op-gate + a cubin-vs-emulation diff on real digit-position activations.

## Net
Routing (T8) and LM-head exonerated. The MoE path's only sm_89-unique,
under-validated piece is the native decode W2 cubin — testable with
`VLLM_MOE_W2_SM89_NATIVE=0`. This is a VALUE mechanism (expert output bias),
orthogonal to agent 4's decode-SELECTION mechanism (TF32 decode indexer logits).
