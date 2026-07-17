#!/usr/bin/env bash
# vLLM-Moet launcher — DeepSeek-V4-Flash on Ada / sm_89 (RTX 4090, RTX 2000-6000 Ada).
#
# The corrected version of the launcher that burned the first Ada deploy
# cycle; every non-obvious line documents which failure it prevents. The
# 2-bit MoE expert GEMMs run through the Triton emulation (moe_w2_sm89.py,
# baked into vllm-moet-sm89:v024); docs/ada-sm89-port.md § Observability
# lists the healthy-boot log lines to expect and the triage recipe.
#
# The three env lines that are NOT optional on Ada + DS4-Flash:
#   VLLM_MOE_W2_BASE_CACHE_GB   DS4-Flash planes are ~1.73 GiB/layer x 43
#                               = ~74 GiB. GPU-RESIDENT planes fit only the
#                               96 GB SM120 board — on 24-48 GB Ada the
#                               base cache is mandatory: planes live in
#                               pinned host RAM, the GPU streams this many
#                               GiB of expert slots. Start 4-8 on a 4090;
#                               the boot now fails EARLY with this remedy
#                               if you drop it (plane-budget preflight).
#   VLLM_MOE_W2_DELTA_GB=0      the w4/w4q FP4 delta tiers are not ported
#                               to Ada (_require_kernels names this fix).
#   VLLM_MOE_W2_PLANES_CACHE    PLANES, plural — the singular spelling is
#                               a silent no-op (the env-typo guard now
#                               warns, with a did-you-mean). Persists the
#                               quantized planes: first boot re-quantizes
#                               the whole checkpoint (long), later boots
#                               skip it.
#
# Host RAM: the base cache pins the ~74 GiB of planes in host memory. On a
# smaller host, cap the pinned arena (VLLM_MOE_W2_BASE_RAM_GB=<GiB>) and
# give the spill an NVMe home (VLLM_MOE_W2_STORE_DIR) — the budget
# preflight warns at boot when MemAvailable cannot hold the projection.
set -euo pipefail
MODEL=${MODEL:-/root/models/DeepSeek-V4-Flash}
CACHE=${CACHE:-/root/models/moet-cache}
NETWORK=${NETWORK:-none}
MAXLEN=${MAXLEN:-16384}
UTIL=${UTIL:-0.90}
BASE_GB=${BASE_GB:-6}
NAME=${NAME:-moet}

mkdir -p "$CACHE/planes" "$CACHE/jit"
docker rm -f "$NAME" 2>/dev/null || true
docker run -d --name "$NAME" --gpus '"device=0"' --network "$NETWORK" --ipc host --shm-size 64g \
  -v "$MODEL":/model:ro \
  -v "$CACHE/planes":/plane-cache \
  -v "$CACHE/jit":/root/.cache \
  -e VLLM_MOE_W2=1 \
  -e VLLM_MOE_W2_DELTA_GB=0 \
  -e VLLM_MOE_W2_BASE_CACHE_GB="$BASE_GB" \
  -e VLLM_MOE_W2_PLANES_CACHE=/plane-cache \
  -e TRITON_CACHE_DIR=/root/.cache/triton \
  -e TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor \
  vllm-moet-sm89:v024 \
  --model /model --served-model-name deepseek-v4-flash --trust-remote-code \
  --kv-cache-dtype fp8 --block-size 256 --max-model-len "$MAXLEN" \
  --gpu-memory-utilization "$UTIL" --max-num-batched-tokens 2048 --max-num-seqs 4 \
  --tokenizer-mode deepseek_v4 --no-scheduler-reserve-full-isl \
  --speculative-config '{"method": "deepseek_mtp", "num_speculative_tokens": 2}' \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"],"cudagraph_capture_sizes":[1,2,4,8,12,16,24]}' \
  --port 8000
echo "started $NAME (sm_89, base-cache ${BASE_GB} GiB, network=$NETWORK, max-model-len=$MAXLEN, util=$UTIL)"
echo "healthy-boot markers:  docker logs -f $NAME 2>&1 | grep -E 'moe_w2'"
echo "  1) 'sm_89 Triton emulation ready on <GPU> ... self-test worst_rel=...'"
echo "  2) 'moe_w2 planes: ... -> PINNED HOST RAM (base cache ...)'"
echo "on failure:            docker logs $NAME 2>&1 | grep -B2 -A30 -E 'EngineCore.*(Error|Traceback)|moe_w2'"
echo "                       docker inspect $NAME --format '{{.State.ExitCode}} {{.State.OOMKilled}}'"
