# Dig — prefill value-readout + F32-scoring (agent 2)

**Headline: the attention softmax WEIGHTS are recomputed (not inherited from the
indexer) and the recomputation is effectively lossless on sm_89 — so the
"F32-scoring" hypothesis (as framed) is invalid. The one open precision question
is the decode INDEXER logits, which is agent 4's territory and is discriminated
by `FORCE_TORCH=1`.**

Source: read-only sub-agent over the sparse-MLA attention + indexer, 2026-08-01.

## IMPORTANT correction to the data-flow model
The sparse-MLA kernel **RECOMPUTES** Q·K from bf16 Q + freshly-dequantized KV; it
does **not** consume the indexer's FP8 scores. Evidence: the kernel signature
(`triton_sparse_mla_dsv4.py:463-481`) takes `sparse_indices` (positions only), no
score tensor; the body does `s = tl.dot(q, tl.trans(k)) * qk_scale` (`:280-282`).
So the indexer's FP8 scores drive **selection only** (the top-k cut, which T4
exonerated at prefill), never the softmax weights. ⇒ "score the indexer in F32"
is moot; llama.cpp's F32-vs-our-FP8 indexer comparison does not bear on attention
weights.

## Finding 1 — sparse-MLA dots are TF32 but effectively lossless
`triton_sparse_mla_dsv4.py:282/316` (Q·K), `:291/325` (P·V), `allow_tf32=True`,
`DOT_BF16=False` (`:649`).
- Q is bf16 (7 mantissa bits); K dequantized from FP8-E4M3 (3 bits) or INT8 (7) by
  exact integer math + power-of-two scale → both ≤ TF32's 10 mantissa bits ⇒ TF32
  truncation discards only trailing zeros.
- **sm_120 runs FP8 MMA (3-bit operands) — LOWER precision than sm_89's TF32.** If
  dot precision were the failure mechanism, sm_89 should pass where sm_120 fails:
  the opposite of the symptom.
- The 6.6e-3 self-test residual is the **bf16 output cast** (`:344`), shared with
  sm_120 — not the dots.
- **Verdict: LOW (attention-weight precision exonerated).**

## Findings 2-4 — value readout exonerated
- Compressed-KV dequant/gather (`_dsv4_gather_k`) recovers the exact quantized
  value in fp32 (FP8-E4M3 integer decode, exact). RoPE bf16 reassembly lossless.
- INT8 A/B was 6/6 identical AND `needle_int8_eager.log` shows the SAME failures
  with INT8 → dequant/readout precision is not the bottleneck.
- §3c clean prior-fallback ⇒ weak digit signal in the residual stream, NOT value
  corruption (which would be garbage).

## The one open precision question (agent 4's, not mine)
The **decode INDEXER logits** (`triton_paged_mqa_logits_dsv4.py:185`, TF32) feed
decode **selection** — and selection is the one thing that could still drop the
digit at decode (agent 4: decode re-selects every step, no prefill carry-over).
My losslessness argument applies to the attention dots; whether the 4.691e-4
decode-indexer Triton-vs-torch residual (TF32 + accumulation order) flips a
borderline rank-512 entry is exactly what `VLLM_DSV4_PAGED_MQA_LOGITS_FORCE_TORCH=1`
settles. **That boot is the discriminator between agent 4 (TF32 decode selection
drops it) and the kernel-numerics-exonerated view.**

## If FORCE_TORCH=1 is NULL (needle still fails)
Kernel numerics are then fully exonerated. Redirect to:
- **H3 (chunked-compressor boundary):** §3a's "chunk size flips individual points
  in OPPOSITE directions" is strong evidence for a correctness/state bug
  (cross-chunk `kv_state`/`score_state`), not precision. Extend
  `tools/test_compressor_vs_checkpoint_ref.py` to N chunks (T5).
- **H4 (config divergence):** chunk 1056 vs upstream 4096, DSpark k=5 vs
  deepseek_mtp k=2, util 0.98 vs 0.92.

## Minimal A/B for the attention dots (only if FORCE_TORCH points back here)
Thread `ALLOW_TF32: tl.constexpr` through the four dots (`:282/:291/:316/:325`),
gated by `VLLM_DSV4_SPARSE_MLA_FP32_DOT=1` at `:649` (mirrors the `DOT_BF16`
pattern). Expected: NULL (decisive negative control). Perf ~10× slower (fp32 on
CUDA cores, Ada has no fp32 tensor core) — correctness probe only.
