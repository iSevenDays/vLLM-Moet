# Cross-codebase diff: llama.cpp dflash vs checkpoint vs our port

Source: read-only Explore agent over `~/llama.cpp` + checkpoint `inference/model.py`
+ our `overlay/.../deepseek_v4/`. All claims below **verified** against the files
2026-08-01 (the investigation has been burned by un-verified analysis before —
README §2 — so each surprising claim was checked).

## Verified-identical (do NOT re-chase)
- **Attend-set assembly (Q1):** all three attend to `[sliding-window raw KV] + [top-k compressed KV]` under ONE softmax. No extra recent-completed segment, no sinks diff, no second window. (`deepseek4.cpp:649-708` `build_csa_lid_attention`; `model.py:513-520`; our `cache_utils.py:526-644` `combine_topk_swa_indices`.)
- **Causal mask (Q3):** a query at `pos` sees compressed entries `0..(pos+1)//ratio - 1` in all three. (`llama-graph.cpp:662-668`; `model.py:431-432`; `cache_utils.py:616` `topk_len = (pos+1)//COMPRESS_RATIO`.)
- **Compressor algorithm (Q4) + cross-chunk state semantics:** ratio-4 overlap (`[prev|cur]` softmax-pool → RMSNorm → RoPE at `pos+1-ratio`) matches; state carried in a `2*ratio` ring/paged buffer across chunks.
- **No always-include-recent mechanism (Q6):** none beyond top-k + raw SWA window in ANY codebase. `local_blocks` confirmed MiniMax-only (T2).

## Verified divergences (actionable)

### D1 [HIGHEST] Compressed-cache buffer ZEROING — our port does NOT zero it
llama.cpp explicitly zeroes `kv_csa`/`kv_lid`/compressor-state at construction, on
`clear()`, and on `seq_rm`, with the comment (verified `llama-kv-cache-dsv4.cpp:1127-1130`):
> "uninitialized buffer contents would otherwise leak in (instance-specific
> garbage) and corrupt recall. Zero all compressed buffers up front..."

and `:1249` "DSV4 compressed buffers must never expose stale/uninit rows".
Our port allocates the compressed KV cache with `torch.empty` (verified: no
`zero_`/`fill_(0)`/`memset` on the long-range compressed cache in `attention.py`
or `compressor.py`). **If any read reaches a row past the written prefix** (an
off-by-one in `topk_len`/`n_visible`, or a block-table entry to an uninit page),
garbage leaks into the softmax. NB: this is CONTINGENT (normal reads are bounded
by `topk_len`), and the clean "word kept, digits → prior 1234" symptom looks more
like non-selection than garbage — so D1 is a real latent bug but probably not the
primary symptom. Still worth fixing.

### D2 [HIGH] Indexer scoring PRECISION — FP8/MXFP4 vs F32
Our port quantizes indexer Q/K to FP8/MXFP4 before `fp8_fp4_mqa_logits`
(verified `sparse_attn_indexer.py:318-330`, q_fp8/fp8_min/fp8_max; README §1
"indexer simulates FP4"). llama.cpp scores with **F32 Q + F32 accumulation**
(`lightning-indexer.cu:407,440` assert `q->type == GGML_TYPE_F32`). The checkpoint
trains with FP4 *simulation* (`model.py:422`). Quantization noise on a borderline
needle block near rank 512 can demote it — consistent with short-context PASS
(comfy margin) vs 8K FAIL (margin shrinks). **Directly supports H5.**

### D3 [HIGH] sm_89 top-k kernel + the team's OWN coverage comment
On sm_89, `use_cooperative_topk` is gated off (`has_device_capability(90)` fails),
so decode uses `persistent_topk`; prefill uses `top_k_per_row_prefill`. llama.cpp
uses arch-independent `ggml_top_k`. And the team's own comment (verified
`sparse_attn_indexer.py:55-56`, verbatim):
> "Long-context exact recall on sm_89 needs far more selection coverage
>  (index_topk / candidates) than the architecture targets..."

Combined with the known `index_topk=2048` mitigation (6/6 pass to 18.5K, README
§7), **H5 (marginal index_topk=512 on sm_89) is the leading explanation.**

### D4 [MEDIUM] QAT vs on-device emulation
If our MXFP4/FP8 emulation on Ada diverges from the checkpoint's training-time
`fp4_act_quant` (block size / rounding / scale format), scores shift. Diff
against `inference/kernel.py::fp4_act_quant`. Lower priority (arch nominally
trained for low precision).

## Recent llama.cpp DSv4 commits (last ~3 months) — flagged ones
- `13f2b28b0` (Jul 11) + `7f575c39d` (Jul 14): compressed-cache clearing & seq_rm
  fixes — directly in the D1 "uninit rows" area.
- `33a75f41c` (Jul 15): overlap-compressor index-gather refactor (semantics unchanged).
- `2ed3c1abb` (Jul 10): attend-set cleanup (removed a zero-attention-bias + dead `raw_k` repeat).
- `67b9b0e7f` (Jul 22): APE tensor op fix (APE = compressor additive pos emb).
None is a single obvious "this is why llama.cpp passes" smoking gun; the clearing
fixes (D1) are the closest.

## Implication for T4 / the hypotheses
- T4 (does digit col 967 survive the k=512 cut on the failing `ask`?) is still the
  decisive measurement: H5 predicts it is OUTSIDE the cut; H2 predicts INSIDE.
- D1 (zeroing) is orthogonal to T4's selection signal — it would need its own
  check (does any read reach past the written prefix?).
- D2/D3 + the `:55` comment make H5 the prior. **The cheap confirmation (already
  known): raise `index_topk` (1024/2048) and the digits return (README §7).**
