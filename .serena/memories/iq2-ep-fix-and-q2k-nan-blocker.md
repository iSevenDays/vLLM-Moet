# IQ2 EP self-sharding fix + q2_k_mm NaN blocker

Branch: `feat/moe-w2-sm89-native-decode` (commit 9bc0900aa, pushed).

## What was fixed (the EP / OOM fix)
The IQ2_XXS port (Steps 1-5, merged to main as 65e074511) OOM'd at boot:
`_create_iq2_weights` allocated all 256 experts per rank under TP=2 (~77 GB
MoE/card on a 48 GB card). Root cause: vLLM defaults
`enable_expert_parallel=False`, so production DSv4 TP=2 is TP-replicated
(`num_local_experts=256`, `expert_map=None`).

The fix is SELF-CONTAINED in `overlay/vllm/vllm/model_executor/layers/quantization/mxfp4.py`,
gated on `VLLM_MOE_W2_IQ2=1` (no config-level EP needed):
- `_create_iq2_weights`: when `expert_map is None` and TP>1 divides
  `num_experts`, synthesize the linear expert_map (rank r owns globals
  `[r*L:(r+1)*L]`), register it as the layer's `_expert_map` buffer, and
  allocate `n_local = num_experts // tp_size` experts (128/card at TP=2).
  The runner's final all-reduce (`reduce_results=True` default on FusedMoE;
  fires while `tp_size>1` and `_fused_output_is_reduced=False`) sums the
  per-rank partials. `_iq2_expert_weight_loader` already shards axis-0 by
  TP rank to match.
- `create_weights`: skip the mxfp4 e2m1/scale/bias alloc on the IQ2 path
  (VRAM-mandatory; those params would sit idle).
- `get_fused_moe_quant_config`: return None on IQ2 (avoids AttributeError
  on the unallocated `w13_weight_scale` during `maybe_init_modular_kernel`).

Loader guard in `overlay/vllm/vllm/models/deepseek_v4/nvidia/model.py`:
skip expert tensors whose mapped name isn't in `params_dict` (the non-IQ2
`gate_proj.weight[_scale]` from the sharded checkpoint map to the
now-unallocated mxfp4 params; the IQ2 params load via their own entries).

Diagnostic: `VLLM_MOE_W2_IQ2_NAN_CHECK=1` logs+clamps per-expert non-finite
output.

## Boot requirement
ENFORCE_EAGER=1 is mandatory: the IQ2 forward is a per-expert Python loop
with `.item()` syncs, so it can't be cudagraph-captured (cudaErrorStreamCaptureUnsupported).

## Boot result (VERIFIED working)
On 2x RTX 4090 D, TP=2, ENFORCE_EAGER=1:
- OOM GONE. "IQ2 EP self-sharding: rank 0/2 owns 128/256 experts" on both ranks.
- Model loads 41.56 GiB/card; reaches `Application startup complete`.
- Decode + small-prefill produce COHERENT output.

## The blocker: q2_k_mm NaN (NOT the EP fix's scope)
The 2K needle fails (HTTP 500). Decisive diagnosis via
VLLM_MOE_W2_IQ2_NAN_CHECK=1: the **`q2_k_mm` Triton kernel (Step 4c) emits
NaN for certain Q2_K expert down-weight blocks** — per-expert log shows
`gate|up|mid finite=True/True/True` but `out finite=False` (max=nan),
input-independent (same experts NaN at M=3 and M=338). Local experts 1, 34,
35 first, then NaN cascades through the residual so later layers see
all-False gate/up/mid/out.

The NaN propagates via the residual into the next layer's sparse-MLA
indexer (`compute_global_topk_indices_and_lens`), whose topk over NaN
scores yields invalid indices -> attention "illegal memory access"
(`cudaErrorStreamCaptureUnsupported`/illegal memory access). So the crash
surfaces in the ATTENTION path but originates in `q2_k_mm`.

Root cause is in the `q2_k_mm` Triton kernel (or the IQ2 converter's Q2_K
scales for those expert blocks), NOT the EP sharding. Next step: audit
`q2_k_mm_triton.py` dequant for the offending experts (likely a bad block
scale -> Inf/NaN), or re-check the Step-1 converter's Q2_K scale output
for experts 1/34/35 (+ their rank-1 mirrors 129/162/163).

## Serve command that worked (boot)
```
MAXLEN=32768 UTIL=0.98 MTP_TOKENS=0 BATCHED_TOKENS=512 ENFORCE_EAGER=1 \
  EXTRA_DOCKER_ENV="-e VLLM_MOE_W2_IQ2=1 -e VLLM_MOE_W2=0" \
  IMG=vllm-moet-sm89:v0251 ./docker/serve_sm89_ds4.sh
```
(MTP_TOKENS not MTP; script uses MTP_TOKENS. ENFORCE_EAGER=1 required.)
