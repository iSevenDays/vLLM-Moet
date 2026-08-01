# Hypothesis (b) — structural decode-path port difference: REFUTED

Read-only diff of our sm_89 decode-indexer/selection path vs upstream
(`upstream/main:patch/vllm-moet-v0.24.0.patch`, vLLM v0.24.0) and the checkpoint.
**All five structural candidates match. There is no port difference in this path.**

1. **Indexer cache dtype — MATCH.** FP4 indexer cache is **sm_100 (B200/GB200)
   only** (`overlay/.../mla/indexer.py:282-289`, baseline v0.24.0 — blocks sm_120
   too). **sm_89 AND sm_120 both use FP8-E4M3 indexer cache.** Identical.
2. **Causal candidate bound — MATCH.** `[b,j] = compressed_seq_len - next_n + 1 + j`
   (`indexer.py:537-543`), baseline, shared; Triton port + torch ref + native
   DeepGEMM all consume the same 2D bound. No off-by-one.
3. **Score-formula ordering — MATCH.** All three ports (torch prefill, torch paged
   decode, Triton paged decode) = `(relu(q·k) * w).sum(h)`, identical to the
   checkpoint `model.py:427-430`.
4. **Indexer Q quant — MATCH.** Both arches run the FP8 `fused_indexer_q_rope_quant`
   path (MXFP4 Q is sm_100-only). (Bonus: the overlay's
   `sparse_attn_indexer.py:393` `fused_indexer_q_rope_quant` is DEAD CODE — never
   imported; the model uses `common.ops.fused_indexer_q`. Cleanup, not a bug.)
5. **Paging/block-table — MATCH.** Production indexer block = 64 (= block_size//
   compress_ratio = 256//4), which IS the self-test geometry (the prior "BLOCK_N=256"
   note was wrong); `FORCE_TORCH=1` already exonerated any Triton edge.

## Implication (decisive)
The decode-indexer path is structurally faithful to both upstream and the
checkpoint. The ONLY sm_89-exclusive attribute (the TF32 score dot) is already
exonerated by direct test (`allow_tf32=False` and `FORCE_TORCH=1` both reproduced
`PELICAN-1234`). So:

- **There is no port/kernel defect to fix.** Hypothesis (b) is refuted.
- Because sm_89 and sm_120 use the **identical FP8 indexer cache + identical
  formula**, sm_120 @ `index_topk=512` would compute the **same decode scores**
  and drop the digit the same way → **sm_120 would ALSO fail the hard needle.**
  The "sm_120 passes" premise is not just unverified, it is very likely FALSE
  (upstream only ever ran the easy random-word probe, which sm_89 also fails).

## Net
The digit-loss is an **inherent property of the trained model's decode indexer
scoring under the `index_topk=512` budget** on the ratio-4 compressed cache —
hardware-independent, not an sm_89 defect. No principled "port fix" exists; the
only lever is `index_topk` (a symptom-level mitigation: 2048, validated 6/6 to
18.5K, with the README §7 caveats). The one remaining unmeasured confirmation is
the digit's actual decode rank (capture-safe decode trace; `dig_decode_path.md`).
