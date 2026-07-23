# Quality validation

This document defines the quality claims for the Moet expert compression.
It also defines the limits of those claims.

## Checkpoint precision

The official DeepSeek-V4-Flash checkpoint uses MXFP4 for the routed expert
weights. It uses one UE8M0 scale for each block of 32 weights. The dense and
attention weights use FP8.

The W2 path changes the routed expert weights. It maps each MXFP4 value to the
sign-symmetric codebook `{-4, -1, 1, 4}`. The normal conversion keeps the
checkpoint block scale. The optional scale refit can decrease a block scale by
one exponent when this change decreases the exact block SSE.
Base-only W2 enables scale refit by default. Set `SCALE_REFIT=0` in the Ada
launcher, or set `VLLM_MOE_W2_SCALE_REFIT=0` directly, to disable it.
Do not combine scale refit with the FP4 delta tier. The two tiers currently
share scale planes, and the runtime rejects this configuration.

## Two different serving configurations

Do not use results from one configuration as proof for the other
configuration.

| Configuration | Main settings | Status |
|---|---|---|
| Base-only W2 | `VLLM_MOE_W2_DELTA_GB=0` | The small probe found coherent short answers. It did not validate long coding-agent prompts. |
| Maximum quality | 34 GiB FP4 delta, confidence gate, and FP4 prefill | The benchmark campaign matched the native checkpoint on GSM8K and GPQA. |

The maximum-quality recipe is
`bench/recipes/deepseek-v4-flash/pro6000x2-tp2-maxq.yaml`. It uses two RTX PRO
6000 GPUs. It does not describe the Ada base-only configuration.

The recorded maximum-quality results are in
`bench/results/v2026.07.17-quality/rtx-pro6000x4/deepseek-v4-flash__pro6000x2-tp2-maxq.json`.
The results include these values:

| Probe | W2 with FP4 recovery | Native comparison |
|---|---:|---:|
| GSM8K, run 1 | 97.5% | 97.0% |
| GSM8K, run 2 | 97.5% | 97.0% |
| GPQA without thinking | 74.75% | 74.24% |
| GPQA with high thinking | 88.38% | 86.87% |
| Needle | Pass at 121,152 prompt tokens | Not applicable |

These results validate the FP4 recovery configuration. They do not validate
the base-only W2 configuration.

## Base-only probe limits

The early `QUANT_PROBE` study reported these values for the sign-symmetric
base:

- Mean accepted draft length was 2.73. The FP4 control value was 2.68.
- The draft acceptance rate was 86.3%. The FP4 control value was 84.1%.
- The short coherence set passed 12 of 12 prompts.
- The arithmetic set passed 2 of 5 prompts.

This study is an ablation result. It is not a production parity result. The
set had only 12 short coherence prompts. It did not contain 50,000-token to
100,000-token coding-agent sessions.

MTP acceptance is not a direct W2 fidelity measurement in this repository.
The W2 layer selection excludes the MTP drafter. Thus, the drafter uses the
stock expert path and the target uses W2. A low acceptance value can show a
target and drafter mismatch, but it cannot locate the cause of that mismatch.

The historical codebook ablation used the same small probe:

| Codebook | Bits | Mean accepted length | Draft acceptance | Arithmetic | Coherence |
|---|---:|---:|---:|---:|---:|
| Official FP4 | 4.0 | 2.68 | 84.1% | 3 of 5 | 12 of 12 |
| Symmetric K=8 | 3.0 | 2.60 | 79.8% | 3 of 5 | 12 of 12 |
| Symmetric K=6 | 2.58 | 2.68 | 84.0% | 2 of 5 | 11 of 12 |
| Asymmetric K=4 | 2.0 | 1.00 | 0.0% | 0 of 5 | 0 of 12 |
| Sign-symmetric K=4 | 2.0 | 2.73 | 86.3% | 2 of 5 | 12 of 12 |

The asymmetric K=4 result produced degenerate output. The sign-symmetric
codebook removed that failure in the small probe. This does not prove that the
sign-symmetric base matches FP4 on long prompts.

## Ada base-only audit

The Ada launcher must use `VLLM_MOE_W2_DELTA_GB=0`. The FP4 delta kernels are
not available on Ada. Therefore, the maximum-quality result does not apply to
the current two-card Ada service.

A 2026-07-21 audit used direct production-geometry tests. The tests found no
large error in these paths:

- The native SM89 block-scaled FP8 output projection had relative L2 error
  from 0.000675 to 0.000722.
- The packed sparse MLA cache writer passed short and 262,143-position tests.
- The sparse MLA reader used the production page stride and had a worst row
  relative error of 0.0078125.

The same audit sampled 18 real expert matrices before model load. The current
base-only conversion had mean relative L2 error 0.381501 and mean row cosine
0.927426. This is a large weight error even when the kernel is bit-correct.

The scale-refit candidate reduced the sample values to 0.354995 relative L2
error and 0.935017 row cosine. It did not change storage size or the kernel.
This is a weight-level result. A long-context model evaluation is still
required.

## Required Ada comparison

Start with MTP and prefix caching disabled. Use the same long agent prompt for
each run.

```bash
RESIDENCY=gpu TP=2 GPUS='"device=0,1"' \
MAXLEN=262144 NUM_SEQS=4 MTP_TOKENS=0 PREFIX_CACHING=0 \
SCALE_REFIT=1 ./docker/serve_sm89_ds4.sh
```

Then run these controlled comparisons:

1. Set `SCALE_REFIT=0`. Compare answer correctness with the refit result.
2. Keep the better W2 setting. Set `PREFIX_CACHING=1`. Repeat the exact prompt.
3. Keep the better cache setting. Set `MTP_TOKENS=1`. Measure correctness and
   generated tokens per second.

Do not enable MTP only because its acceptance counter is nonzero. Enable it
only when the end-to-end generated-token rate increases and the answer stays
correct.

## FP4 recovery instrumentation

The FP4 delta manager can report its coverage and replacement activity. These
settings only apply on a GPU architecture that has the FP4 delta kernels.

```bash
VLLM_MOE_W2_DELTA_TRACE=1        # Select summary output.
# VLLM_MOE_W2_DELTA_TRACE=2      # Or, select output for each event.
VLLM_MOE_W2_DELTA_TRACE_EVERY=64
VLLM_MOE_W2_DELTA_DUMP=/tmp/delta.json
```

Trace level 1 reports coverage, promotions, replacements, and an FP4 expert
count for each layer. Trace level 2 also reports each promote and replace
event. `DeltaTier.precision_map()` returns the same precision map to code.
Use `tools/delta_trace_demo.py` to test this instrumentation without a model
load.

## Cache identity

Persistent expert data depends on the checkpoint, TP size, residency, and
quantizer settings. The cache metadata now includes the checkpoint shard file
identity and the scale-refit setting. The host-pack loader-skip probe also
checks the checkpoint identity, layer count, and quantizer identity.

Remove old caches after an unclean disk or file-system event. Cache metadata
and file sizes cannot prove that all data bytes are free from storage errors.

## Test commands

```bash
python3 tools/test_moe_w2_planes.py
python3 tools/test_moe_w2_store_identity.py
python3 tools/probe_quant_quality.py <port>
python3 tools/needle_probe.py <port> 120000 0.5
```

Use a same-GPU native control for each quality claim. Record the complete
recipe with the result.
