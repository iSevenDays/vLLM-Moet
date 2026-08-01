# FINAL — decode trace direct observation: decode-coverage REFUTED

The capture-safe decode trace finally worked (simple direct-log form under
`--enforce-eager`). It directly observed the digit column (c967/c968) rank/sel
at the digit-generating decode steps. Verdict: **decode selection is NOT the
pass/fail discriminator.**

## Measured (ask vs digitsonly, both /8192, POS=3871, under --enforce-eager)
| variant | result | r4 c967/c968 at DECODE |
|---|---|---|
| `ask` (FAIL → PELICAN-1234) | digits lost | ~71% sel=0 (median rank ~841) |
| `digitsonly` (PASS → 1605) | digits kept | ~73% sel=0 (same regime) |

Both probes drop the digit ~72% at decode; digitsonly still recovers it. So
the pass/fail difference is NOT decode-selection coverage.

## vs prefill (T4)
T4: the digit was selected in 20/21 ratio-4 layers (95%) at the question prefill
chunk. So decode (~28% selected) << prefill (95%) — the digit IS attended less
at decode — **but identically so for pass and fail.**

## The actual mechanism (directly observed)
The digit's compressed entry is **weakly attended at decode (~28% selected)** in
both cases. Whether the model produces the digits depends on the **query/output
framing** interacting with that weak signal:
- `digitsonly` ("the four digits…") cues digits → extracts the weak signal → PASS.
- `ask` ("what is the access code?") cues the code → outputs the salient word,
  then can't copy the digits from the weakly-attended compressed entry → prior
  fallback `1234`.
- `index_topk=2048` widens the cut → the digit's decode-attention strengthens
  → enough signal for `ask` too (6/6 to 18.5K, README §7).

## What this rules out (all by direct test)
- Decode-coverage as the discriminator (this trace).
- Cudagraph/capture bug (`--enforce-eager` reproduces the failure — KEY A).
- Kernel precision: TF32→fp32 (`allow_tf32=False`), INT8 KV, native-vs-emulation
  W2, all identical; prefill attention/value/routing exonerated.
- Structural port-vs-upstream difference (v0.24.0 patch diff; v0.26.0 diff).
- A working upstream sm_89 reference (stock v0.26.0 gates out capability-major
  8 for DSv4 — it cannot run here at all).

## Conclusion
The digit-loss is a **model-capability/budget property**: the trained indexer
scores the digit's compressed entry weakly at decode (~28% selected) under a
non-digit-cuing query at `index_topk=512`, and the model can't copy exact digits
from that weak signal. It is NOT an sm_89 port/kernel/graph defect. The
mitigation is `index_topk=2048` (strengthens the signal; symptom-level, with
the README §7 caveats: fails at 35.8K, deviates from trained 512, needs quality
validation). A principled fix would require a different trained indexer or a
higher budget — there is no port bug to backport.
