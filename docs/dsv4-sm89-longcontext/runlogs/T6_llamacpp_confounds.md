# T6 — Close the llama.cpp confounds

**Headline: llama.cpp's "pass" used the OLD (superseded) checkpoint, NOT 0731.
It is NOT like-for-like with our port — the strongest "port defect" prior is
DOWNGRADED.** `local_blocks` is also absent from the GGUF (closes T2's deferred
metadata question; H1 stays refuted).

Produced 2026-08-01 (CPU, no GPU). Task spec: `PLAN.md` T6.

## Item 1 — Which weights? → OLD checkpoint (confound #1 CONFIRMED)

llama.cpp's recorded pass (shell history) used:
`/root/antirez/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`

That GGUF's metadata (dumped via `~/llama.cpp/gguf-py`):

| key | value | meaning |
|-----|-------|---------|
| `general.name` | `DeepSeek V4 Flash` | ambiguous (no `0731` marker) |
| `deepseek4.attention.compress_ratios` | length **44** (3 zeros, 21×4, 20×128) | **OLD-checkpoint signature** |
| dspark.* / dflash.* | **none** | OLD-checkpoint signature |
| file mtime | Jul 15 | **before** the 0731 checkpoint (Jul 31) |

README §0 discriminator: **OLD = 44 `compress_ratios` entries + no dspark keys**;
**0731 = 46 entries + `dspark_block_size 5`, `dspark_target_layer_ids [40,41,42]`,
`dspark_markov_rank 256`**. This GGUF matches OLD on all three signals.

The sibling `DeepSeek-V4-Flash-DSpark-support.gguf` (5.99 GB) DOES carry 0731's
dspark keys (`block_size=5`, `target_layer_ids=[40,41,42]`, `markov_rank=256`,
`n_layers=3`, `stage_count=3`) — i.e. it is the 0731-derived GGUF — but llama.cpp's
needle-pass did **not** use it.

**Conclusion:** the "llama.cpp passes on this same hardware" datum (README §3d)
was measured on the **old weights**, not the 0731 weights our port serves. It is
therefore not like-for-like. The prior shift it motivated ("architectural limit"
→ "port defect") is **not warranted by this datum**.

## Item 3 — `local_blocks` in the GGUF? → ABSENT (closes T2)

The GGUF carries only:
`deepseek4.attention.indexer.head_count = 64`, `key_length = 128`, `top_k = 512`.
There is **no** `attention.indexer.local_blocks` and no `indexer.block_size`.
llama.cpp's default is `indexer_local_blocks = 0`, and `dflash.cpp` does not read
the key anyway (only `minimax-m3.cpp` does). So even llama.cpp uses **no**
always-included-recent-compressed-blocks mechanism for this model — exactly as T2
found for the checkpoint and the port. **H1 stays refuted on the metadata front.**

## Item 2 — Which probe / length / depth? → command recorded, reply not in history

The llama-server launch (shell history):
```
llama-server --model ...IQ2XXS...gguf -ngl 99 --split-mode layer -ts 43,43 \
  -b 1024 -ub 1024 -fa on -c 1048576 --cache-type-k q8_0 --cache-type-v q8_0 \
  --port 8002 --parallel 4 --jinja --reasoning on --reasoning-preserve \
  --spec-type ngram-mod --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 \
  --spec-ngram-mod-n-max 64 --no-mmap --prefetch-weights 1 \
  --chat-template-kwargs '{"reasoning_effort":"max"}'
```
(One variant adds `--ctx-checkpoints 64 --cache-ram 32768`, another uses `-c 1310720`.)

Notable: llama.cpp used **n-gram** speculation (not DSpark) and **q8_0** KV — both
different from our port (DSpark k=5, fp8 KV). The exact reply text (the needle
answer) is not in shell history; re-running llama.cpp against the **0731**
(`DSpark-support`) GGUF with our probe would close this, but that needs a GPU
(not available while moet-t4 boots) and the 5.99 GB GGUF may be too aggressively
quantized to be a clean reference. **Deferred.**

## Impact on the hypotheses

- The llama.cpp pass no longer supports "port defect" over "architectural / 0731
  weight behaviour." H5 ("index_topk=512 marginal on any hardware") and H4 (config
  divergence) regain weight relative to the sm_89-port-suspicion stories.
- The decisive measurement remains **T4** (does the digit column survive the
  k=512 cut on the FAILING request?) — read against the OLD-weight caveat, i.e.
  even a "selection is fine" result no longer implicates the sm_89 kernels alone.

## Reproduce (CPU, seconds)
```bash
python3 - <<'PY'
import sys; sys.path.insert(0, "/root/llama.cpp/gguf-py")
import gguf
r = gguf.GGUFReader("/root/antirez/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf")
cr = list(r.fields["deepseek4.attention.compress_ratios"].contents().tolist() \
        if hasattr(r.fields["deepseek4.attention.compress_ratios"].contents(),'tolist') \
        else r.fields["deepseek4.attention.compress_ratios"].contents())
print("len", len(cr), cr)   # 44 -> OLD checkpoint
print("local_blocks?", "deepseek4.attention.indexer.local_blocks" in r.fields)  # False
print("dspark?", any("dspark" in k for k in r.fields))  # False
PY
```
