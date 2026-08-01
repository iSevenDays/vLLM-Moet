# Root cause (convergent determination) — decode-side index_topk=512 COVERAGE

**Every kernel-numerics suspect is EXONERATED by direct test. The digit-loss is a
model/indexer-scoring + tight-budget property, NOT an sm_89 kernel defect. The
fix is coverage (`index_topk=2048`, already validated).**

## The final test — `allow_tf32=False` (suspect 1, decode TF32)
`moet-fix`: prod config + `triton_paged_mqa_logits_dsv4.py:185` `allow_tf32=False`
(true fp32 decode indexer dot, 2 GiB KV for the fp32 headroom). `ask/8192`:
- answer = **`PELICAN-1234`** (word kept, digits lost) — **identical to baseline**.
- → **decode TF32 is NOT the mechanism.** Reverted (`allow_tf32=True`, patches 85/85).

(Aside: this also settles the Monte-Carlo vs GPU-flip-test disagreement — the
Monte-Carlo's "TF32 could flip ~2-13%" was real arithmetically, but the digit's
decode rank evidently does not ride the 512 cut, so TF32 never gets the chance.)

## Full exonerated set (all by direct test)
| suspect | test | verdict |
|---|---|---|
| H1 local_blocks | T2 code diff | REFUTED (none in ckpt/port/dflash) |
| H3 compressor chunk-split | T5 unit test | REFUTED (0.0 diff, chunk-invariant) |
| suspect 2 native W2 cubin | GPU op-gate | EXONERATED (faithful to ref at bf16 floor) |
| prefill attention weights | dig (agent 2) | lossless; sm89 TF32 > sm120 FP8 precision |
| value readout / KV dequant | dig (agent 2) | exact; INT8 A/B identical |
| routing kernel T8 | dig (agent 3) | arch-independent, faithful to ckpt |
| **suspect 1 decode TF32** | **`allow_tf32=False` boot** | **EXONERATED (identical failure)** |

## The convergent root cause
1. **T4:** the digit's compressed entry **IS selected at PREFILL** (20/21 ratio-4
   layers at question-time, rank 0 in one) at `index_topk=512` — yet the model
   fails. So the failure is **downstream of prefill selection**.
2. **Decode re-selects every step with NO carry-over** (dig, agent 4) — so prefill
   selection does not determine what is attended when the digit tokens are emitted.
3. **`index_topk=2048` is the only change that converts failures to passes**
   (6/6 to 18.5K, README §3a/§7) → **coverage is the lever**.
4. **`allow_tf32=False` does NOT fix it** → the decode drop is **not a precision
   artifact**; it is the model's actual decode indexer scoring under the tight
   512 budget.
5. `digitsonly`/`firstword`/`yesno` pass; `ask`/`spell`/`verbose` fail (§3b) → the
   query content determines whether the digit's decode score clears 512.

⇒ **At the digit-generating decode steps, the `ask` query scores the digit's
compressed entry below the `index_topk=512` cut, so its KV is not attended and
the model falls back to its digit prior (`1234`).** This is a property of the
trained indexer's decode scoring + the aggressive 512 budget on a heavily-
compressed (ratio-4) cache — **not an sm_89 port/kernel defect** (all kernel
numerics exonerated; sm89's TF32 is even higher precision than sm120's FP8 MMA).

## Why this is consistent with "assume sm120 passes"
If sm120 passes the HARD realistic-filler needle (unverified — upstream's recipe
uses the EASY random-word probe, README §3d), the difference would not be kernel
precision (exonerated) but that sm120's native DeepGEMM decode-indexer path + the
upstream serve recipe (chunk 4096, deepseek_mtp k=2) change the decode-selection
dynamics enough to clear 512. That remains the one untested comparison (the 0731
checkpoint is DSpark-bound, so `deepseek_mtp` is untestable here — T7).

## Fix
`--hf-overrides '{"index_topk": 2048}'` (README §7). Costs compute not VRAM.
Caveats that stay attached: fails at 35.8K; deviates from the trained 512 (needs
GSM8K/GPQA validation); ~26,000 extrapolated at 262K (effectively dense).

## The one unmeasured confirmation (optional)
The digit's actual **decode** rank (the decode trace, `_trace_indexer_rank_decode`,
T3). It crashes boot (`.item()` host-sync under cudagraph); agent 4 gave the
capture-safe fix (on-device row-select + deferred emit). Measuring it would show
the digit's `sel` flag flipping to 0 at the digit-emitting decode steps on `ask`.
The convergent evidence above is strong without it.
