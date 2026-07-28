# antirez/ds4 reference source

Vendored **read-only reference** for differential analysis: comparing antirez's
DSv4-Flash inference (which PASSES the long-context needle with IQ2_XXS) against
our vLLM IQ2_XXS port (which FAILS ~8-10K). Not built, not shipped — source only.

- **Source:** Salvatore Sanfilippo's `ds4` (antirez), public repo.
- **License:** see upstream (`LICENSE` in the antirez/ds4 repo).
- **Files:** `ds4.c` (CPU / orchestration), `ds4_cuda.cu` (CUDA kernels).

Key functions for the precision differential:
- `dequant_q8_0_to_f32_kernel` (ds4_cuda.cu:643) + `cuda_q8_f32_ptr` (:1398) —
  antirez dequants Q8_0 weights to an **fp32** cache (fp32 compute).
- `cuda_block_iq2_xxs` (ds4_cuda.cu:70-78) — the IQ2_XXS 2-bit block.
- The MoE forward + residual/attention compute dtype (search ds4_cuda.cu).
