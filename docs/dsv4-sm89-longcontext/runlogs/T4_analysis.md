# T4 — Retrace at the correct column (THE DECISIVE MEASUREMENT)

**Verdict: H2 CONFIRMED — selection is EXONERATED.** The digit's compressed
entry IS selected in essentially every ratio-4 layer at question-time on the
FAILING `ask` (rank 0 in one layer). The information is attended but the digits
are still not produced. The failure is **downstream of selection** (value
readout / decode / how the digit token is produced from attended KV).

This **overturns the H5 / §5.18 selection story** (the prior favorite, and what
the team's own `sparse_attn_indexer.py:55` comment leans toward). Measurement >
inference: the §2 lesson holds.

Produced 2026-08-01. One boot (`moet-t4`, pre-T3 trace mount — the image has no
trace and the T3 decode-trace mount crashes boot; see T3). Correct column from
T1: `VLLM_DSV4_INDEXER_TRACE_POS=3871` → ratio-4 col **967** (was 4843/1210).

## Setup
- `ask/8192` (FAIL → "PELICAN", digits lost) and `digitsonly/8192` (PASS → 1605).
- Pre-T3 prefill trace, both TP ranks (no rank0 gate), counter-labeled, single
  column. 126 lines/variant = 21 ratio-4 layers × 3 prefill chunks × 2 ranks;
  **TP0-only deduped → 63 emits/variant** (both ranks emit identical values).

## THE result — isolate the QUESTION chunk

The prefill is chunked (1056 tok). The trace fires on 3 chunks per layer with
candidate counts **{2088, 2349, 2421}**; `2421 = 9686//4` is the FULL context =
the chunk that contains the **question tokens**. Averaging across chunks (the
naive read) is misleading: the 2088/2349 chunks are filler-token prefills where
the needle is causally irrelevant. **Only the 2421 (question) chunk matters.**

| chunk (candidates) | ask selected / 21 | digitsonly selected / 21 |
|---|---|---|
| 2088 (filler) | 1 | 4 |
| 2349 (filler) | 5 | 3 |
| **2421 (QUESTION)** | **20 / 21** (rank 0..917) | **21 / 21** (rank 3..381) |

At question-time, in essentially every ratio-4 layer, **the digit column is
selected on the failing `ask`** — including **rank 0** (the top-scored entry) in
one layer. `digitsonly` selects it in 21/21 with a slightly better rank envelope,
but the failing `ask` is at 20/21. That is not "the digit is dropped."

(TP0 question-chunk emits, ask: ranks 8, 0, 39 in the last three layers;
digitsonly: 5, 13, 22. ask tops at rank 0.)

## What this rules in / out
- **Selection coverage (H5 / §5.18) is NOT the mechanism at 8K.** The digit IS
  attended at the layers/time that produce the answer. `index_topk=512` is not
  dropping the digit here.
- **H2 stands:** the value/decode path reopens. The attended KV contains the
  digit but the model emits `PELICAN` / `1234`. Next: the **decode** selection
  (never observed — T3's decode trace crashed boot and is parked), the **value
  readout** precision (NoPE/FP8 KV — README §4 found E4M3 vs INT8 storage didn't
  matter, but the *readout* path is separate), and re-examine `digitspad` under
  the corrected instrument.
- **Tension with `index_topk=2048`** (6/6 pass to 18.5K, README §7): at 8K the
  digit is already selected at topk=512, so 2048 must help via a *different*
  mechanism (attention-weight distribution / context entries), OR the rescue
  applies at longer contexts where the digit ranks worse. The 8K and >8K
  mechanisms may differ.

## Raw data
- `runlogs/T4_ask.json`, `runlogs/T4_digitsonly.json` (probe records).
- `runlogs/T4_trace_ask.log`, `runlogs/T4_trace_digitsonly.log` (126 lines each;
  grep `r4:col=967`).

## Reproduce (needs a boot; pre-T3 trace mount)
```bash
# launch moet-t4 with pre-T3 sparse_attn_indexer.py mounted + INDEXER_TRACE_POS=3871
python3 tools/needle_digits_probe.py --lengths 8192 --variant ask         # FAIL
python3 tools/needle_digits_probe.py --lengths 8192 --variant digitsonly  # PASS
docker logs moet-t4 --since <ts> 2>&1 | grep "indexer rank trace"
# group by candidates= : the 2421 (question) chunk shows 20/21 & 21/21 selected.
```
