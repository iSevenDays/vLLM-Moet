# T2 — `indexer.local_blocks` parity (H1)

**Verdict: H1 REFUTED.** No implementation always-includes recent *compressed*
blocks beyond the top-k. All three assemble the attend set as
**`[sliding window] + [top-k compressed]`**. There is no `local_blocks` mechanism
to be missing.

Produced 2026-08-01 (CPU, no boot). Task spec: `PLAN.md` T2.

## The one question

> Does any implementation always-include recent compressed blocks that this
> port drops? — No.

## The three, side by side

### 1. Checkpoint (the authority) — `/root/models/DeepSeek-V4-Flash-0731/inference/model.py`

Attend-set assembly (`Attention.forward`, ~513–520):
```python
topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos)   # raw 128-token window
if self.compress_ratio:
    ...
    if self.indexer is not None:
        compress_topk_idxs = self.indexer(x, qr, start_pos, offset).int()
    else:
        compress_topk_idxs = get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset)
    topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
```
The `Indexer` returns pure top-k (`Indexer.__call__`, ~433):
```python
topk_idxs = index_score.topk(min(self.index_topk, end_pos // ratio), dim=-1)[1]
```
`index_topk = 512` (ModelArgs:77). **No `local_blocks`; no always-included recent
compressed entries.** "Recent" coverage comes only from the raw sliding window.

### 2. This port — `overlay/vllm/vllm/models/deepseek_v4/common/ops/cache_utils.py`

`combine_topk_swa_indices` (526) → `_combine_topk_swa_indices_kernel`. Per token:
```python
topk_len = tl.minimum((pos + 1) // COMPRESS_RATIO, TOP_K)
swa_len  = tl.minimum(pos + 1, WINDOW_SIZE)
...                                  # write topk_len compressed indices, then
...                                  # swa_len window indices into the buffer
combined_len = topk_len + swa_len
```
i.e. `combined = [topk compressed] + [window]`, combined width `topk + window_size`
(aligned). **No `local_blocks`.** A grep of `overlay/` and `vendor/` for
`local_blocks`/`always_included` finds only `dcp_local_block_size`
(`mla_attention.py:1557`) — an unrelated DCP / kv-cache-interleave concept, not the
indexer attend set. Matches the checkpoint.

### 3. llama.cpp — `~/llama.cpp`

`indexer_local_blocks` **defaults to 0** (`llama-hparams.h:232`):
```cpp
uint32_t indexer_local_blocks = 0;
```
The KV key is defined (`llama-arch.cpp:261`) and saved (`llama-model-saver.cpp:285`),
but it is **read and consumed only in `models/minimax-m3.cpp:25–26`**:
```cpp
ml.get_key(LLM_KV_ATTENTION_INDEXER_LOCAL_BLOCKS, hparams.indexer_local_blocks);
msa_p = { (int) hparams.indexer_block_size, (int) hparams.indexer_top_k, (int) hparams.indexer_local_blocks };
```
…which drives `llm_graph_input_msa_local` ("local window always wins a slot",
`minimax-m3.cpp:90–104`). That is a **MiniMax-M3** selection feature.

`models/dflash.cpp` (the DeepSeek-V4-Flash path) **does not reference `local_blocks`
at all** and reads no `LLM_KV_ATTENTION_INDEXER_*` keys — only
`LLM_KV_ATTENTION_SLIDING_WINDOW` (`dflash.cpp:25`) plus the DSpark
`dflash.block_size` metadata. It routes layers to SWA vs dense sub-caches
(`dflash.cpp:268`); it has no always-included-recent-compressed mechanism.

## Conclusion

The "missing `local_blocks`" lead came from `llama-arch.cpp:260–261` advertizing a
key that exists in the llama.cpp *codebase* but belongs to MiniMax-M3, not DFlash.
The trained DSv4-Flash checkpoint itself has no such mechanism, so the port matching
`[window + topk]` is faithful, not defective. Patching `combine_topk_swa_indices` to
force-include recent compressed blocks would be a **behaviour change versus the
trained checkpoint** with no reference backing — do not pursue.

Go to **T3**. The decisive measurement remains **T4** (selection-vs-value, at the
correct column). The chunk-size sensitivity (H3, T5) and the llama.cpp confounds
(T6) are unaffected.

## Reproduce (CPU, seconds)
```bash
grep -nE "local_blocks|always_included" overlay/ vendor/                   # port: none relevant
grep -nE "indexer_local_blocks" ~/llama.cpp/src/llama-hparams.h            # = 0 default
grep -nE "LOCAL_BLOCKS|local_block" ~/llama.cpp/src/models/dflash.cpp      # (no output)
grep -nE "LOCAL_BLOCKS" ~/llama.cpp/src/models/minimax-m3.cpp              # the only reader
sed -n '513,520p' /root/models/DeepSeek-V4-Flash-0731/inference/model.py   # checkpoint cat([win, topk])
```
