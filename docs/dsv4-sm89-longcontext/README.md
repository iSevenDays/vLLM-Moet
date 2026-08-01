# DeepSeek-V4-Flash on Ada sm_89 — long-context investigation

Canonical home for the DSv4-Flash-0731 / 2×RTX 4090 D work. Everything that was
previously scattered under `/root/ktransformers` now lives here, next to the code
it describes.

| file | what it is |
|------|------------|
| [`STATUS.md`](STATUS.md) | **Authoritative status, measurements, decisions, NEXT queue.** Start here. |
| [`HANDOFF.md`](HANDOFF.md) | Raw state + primary sources for a fresh agent. Read after STATUS. |
| [`runlogs/`](runlogs/) | Raw artifacts backing every claim in STATUS. |
| [`../../tools/README.md`](../../tools/README.md) | The test harnesses, what each one proves, and its runtime. |

## The one thing to read before touching this

**"Needle retrieval is broken above ~8K" is WRONG, and believing it costs hours.**
The model recovers the needle's **word** at every length tested (2K–48K); only the
numeric **digits** are lost:

```
true PELICAN-1605  ->  "PELICAN PELICAN PELICAN..."
true LANTERN-2037  ->  "LANTERN-9-9-9-9-9..."
```

Retrieval works. The attended value's fine detail is unreadable. Those are
different bugs with different fixes. Use
[`tools/needle_digits_probe.py`](../../tools/needle_digits_probe.py), which
reports `word_present` / `digits_present` separately — the old
`bench/runner/probes.py --probe needle_sweep` metric collapses both into one
bit and is what produced the misdiagnosis.

## Current understanding (2026-08-01)

Digit fidelity sits at a **numerical noise floor**, not behind a discrete defect:

- Halving `--max-num-batched-tokens` (1056 → 528) keeps the score at 2/6 but
  flips **a different 2** — two cases move in *opposite* directions
  (STATUS §5.14). A systematic bug degrades consistently; this reshuffles.
- The failure is deterministic per input (bit-identical on repeat), yet
  **non-monotonic in length** (pt 8316 passes while 5100 and 6650 fail).
- Errors are **near-misses** once redundancy or salience is added (`1606` for
  `1605`, `7542` for `7544`) but fall back to the generic prior `1234` on the
  plain task.
- The word always survives because the 20 `compress_ratio=128` layers see the
  whole context (~75 entries at 9.7K, far below `index_topk`, so no selection
  happens there at all).

Leading quantity: **which ratio-4 compressed entries get selected**. Raising
`index_topk` is the active experiment (STATUS §5.14).

## Eliminated — do not re-litigate (each by measurement, not argument)

| cause | how it was ruled out |
|-------|----------------------|
| FP8 KV dtype | upstream passes `needle @121k` with the same flag on native Blackwell kernels |
| NoPE storage precision | `VLLM_DSV4_KV_INT8=1` A/B: **6/6 identical** despite 3–4× better readback |
| top-k selection kernels | `tools/test_indexer_topk_selection.py`: 512/512 correct with all high scores beyond index 512 |
| indexer scoring (ragged + paged) | torch fallback matches the checkpoint reference; paged Triton suite passes |
| per-token scalars | top-k is per query **row**, so `q_scale`/`softmax_scale`/`head_scale` are ranking-neutral |
| compressor semantics | `tools/test_compressor_vs_checkpoint_ref.py`: agrees with an independent checkpoint transcription at ~2.6e-3 (bf16 floor) |
| compressed-attention KV store/gather | 11 previously-blocked tests now pass after the arch-gate fix |
| intra-chunk state overwrite | `tools/test_compressor_state_capacity.py`: allocation is chunk-aware (267/149 blocks vs 266/148 required) |
| fixed needle position (~2048) | passes at abs 2721, fails at abs 1695 |
| fixed total-length cutoff (~8192) | non-monotonic: pt 8316 passes, 5100/6650 fail |

## Ground truth — prefer these over any doc here

1. **`/root/models/DeepSeek-V4-Flash-0731/inference/model.py`** — DeepSeek's own
   implementation for this exact checkpoint. Authoritative for the ring-buffer
   SWA cache, the compressor, and the indexer (`Indexer` exists **only** on
   `compress_ratio == 4` layers; ratio-128 layers use positional order).
2. **`~/llama.cpp`** — a DFlash+DSpark build that passes the needle test
   (`src/models/dflash.cpp`).
3. **`upstream/main`** (`kacper-daftcode/vLLM-Moet`) — passes needle @121K on
   Blackwell. Note it contains **none** of the sm_89 Triton kernels, so it is a
   behavioural reference, not a code-diff target.

## Working rule

Deploys and end-to-end probes are expensive (6–10 min boots, 30–190 s per
probe); unit tests are cheap (seconds, often CPU-only). Every harness in
[`tools/README.md`](../../tools/README.md) was written to replace a server round
trip. Add to them before reaching for another boot.
