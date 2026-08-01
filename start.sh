#!/usr/bin/env bash
# vLLM-Moet launcher for DeepSeek-V4-Flash on Ada / sm_89 (RTX 4090, RTX 2000-6000 Ada).
# The 2-bit MoE expert GEMMs run through the Triton EMULATION path (moe_w2_sm89.py),
# NOT the SM120 QMMA.SF cubins — Ada has FP8 tensor cores but no QMMA / QMMA.SF / FP4.
# The patch — including the sm_89 Triton kernel and the arch-blind moe_w2_cubit
# dispatcher — is BAKED INTO vllm-moet-sm89:v024, so no overlay bind-mounts are needed
# (overlaying moe_w2_cubit.py from an older copy would clobber the sm_89 dispatch branch).
#
# Ada serving constraints (see docs/ada-sm89-port.md):
#   VLLM_MOE_W2_DELTA_GB=0   the FP4 delta / w4 tiers are NOT ported to Ada; the loader's
#                            _require_kernels fails loudly if the delta tier is enabled
#   --kv-cache-dtype fp8     the NVFP4 KV cache is Blackwell-only
#   attention                stock FlashInfer/FlashAttention (no sparse-MLA SM120 path)
#   VLLM_MOE_W2_AFRAG        ignored on Ada (prefill falls back to w2mc4); harmless
#   Triton JIT               moe_w2_sm89 compiles per-K on first use; warmup covers it
#
# Other differences vs the upstream README command, and why:
#   plane-cache overlay      persists 2-bit planes; warm starts skip staging+quantize
#   /root/.cache mount       persists torch.compile/FlashInfer/DeepGEMM/Triton JIT caches
#   no --rm, restart policy  keep the container; `docker stop/start moet` beats cold runs
#   NETWORK=none default     pass NETWORK=host only when you've decided to expose it
set -euo pipefail
MODEL="/root/models/DeepSeek-V4-Flash-0731"   # 146 GB checkpoint (46 shards), deepseek_v4 arch
CACHE="/root/models/moet-cache"          # plane-cache + JIT caches live here (on /, ~79 GB free)
NETWORK=${NETWORK:-none}
MAXLEN=${MAXLEN:-16384}
UTIL=${UTIL:-0.95}
NAME=${NAME:-moet}


mkdir -p "$CACHE/planes" "$CACHE/jit" "$CACHE/tilelang"
docker rm -f "$NAME" 2>/dev/null || true
docker run -d --name "$NAME" --gpus '"device=0"' --network "$NETWORK" --ipc host --shm-size 64g \
  -v "$MODEL":/model:ro \
  -v "$CACHE/planes":/plane-cache \
  -v "$CACHE/jit":/root/.cache \
  -v "$CACHE/tilelang":/root/.tilelang \
  -e VLLM_MOE_W2=1 -e VLLM_MOE_W2_DELTA_GB=0 \
  -e VLLM_MOE_W2_PLANE_CACHE=/plane-cache \
  -e DG_JIT_CACHE_DIR=/root/.cache/deep_gemm \
  -e TRITON_CACHE_DIR=/root/.cache/triton \
  -e TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor \
  vllm-moet-sm89:v024 \
  --model /model --served-model-name deepseek-v4-flash auto --trust-remote-code \
  --kv-cache-dtype fp8 --block-size 256 --max-model-len "$MAXLEN" \
  --gpu-memory-utilization "$UTIL" --max-num-batched-tokens 2048 --max-num-seqs 4 \
  --async-scheduling \
  --tokenizer-mode deepseek_v4 --no-scheduler-reserve-full-isl \
  --speculative-config '{"method": "deepseek_mtp", "num_speculative_tokens": 2}' \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"],"cudagraph_capture_sizes":[1,2,4,8,12,16,24]}' \
  --port 8000
echo "started $NAME (sm_89, network=$NETWORK, max-model-len=$MAXLEN, util=$UTIL); follow: docker logs -f $NAME"
