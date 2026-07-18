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
# network=none is a deliberate air-gap: the first boot re-quantizes the
# whole checkpoint and should not be reachable half-warm. It also means
# the API port ($PORT) is UNREACHABLE until you relaunch with NETWORK=host
# (or a bridge + -p mapping) once the boot markers look healthy.
NETWORK=${NETWORK:-none}
MAXLEN=${MAXLEN:-16384}
UTIL=${UTIL:-0.90}
PORT=${PORT:-8001}
# Both 48 GB cards via tensor parallelism (TP2); GPU1 was idle on TP1. BASE_GB is
# the GPU-resident expert pool PER RANK (per card) — 20 GiB fits a 48 GB card with
# room for the TP-sharded dense weights + KV. Fall back to one card with:
#   TP=1 GPUS='"device=0"' ./docker/serve_sm89_ds4.sh   (then bump ARENA_GB to ~20)
GPUS=${GPUS:-'"device=0,1"'}
TP=${TP:-2}
BASE_GB=${BASE_GB:-20}
NAME=${NAME:-moet}
IMG=${IMG:-vllm-moet-sm89:v024}
# --- small-host RAM tier (REQUIRED on this box: ~38 GiB host RAM << the ~80 GiB the
#     all-pinned 2-bit base needs). STORE_DIR moves the base out of pinned RAM into
#     on-disk per-rank packs: read-mostly files (mmap/page-cache), NOT swap — so ZFS
#     is safe for them (swap-on-ZFS deadlocks; a pack file does not). The packs
#     double as a persistent quant cache (later boots skip dequant->requant).
#     BASE_RAM_GB pins a small MRU arena over each pack. NOTE: the arena is PER RANK,
#     so under TP2 the host holds 2 x ARENA_GB — kept small here (2 x 10 = 20 GiB) to
#     fit this 38 GiB box. STORE needs a REAL fs (bind mount, NOT overlayfs), ~80 GB
#     free for both per-rank packs combined.
STORE=${STORE:-$CACHE/packs}
ARENA_GB=${ARENA_GB:-10}

mkdir -p "$CACHE/planes" "$CACHE/jit" "$STORE"
docker rm -f "$NAME" 2>/dev/null || true
# TP>1 needs --disable-custom-all-reduce on this stack (README § TP).
TPARGS=""
if [ "$TP" -gt 1 ]; then TPARGS="--tensor-parallel-size $TP --disable-custom-all-reduce"; fi
docker run -d --name "$NAME" --gpus "$GPUS" --network "$NETWORK" --ipc host --shm-size 64g \
  -v "$MODEL":/model:ro \
  -v "$CACHE/planes":/plane-cache \
  -v "$CACHE/jit":/root/.cache \
  -v "$STORE":/packs \
  -e VLLM_MOE_W2=1 \
  -e VLLM_MOE_W2_DELTA_GB=0 \
  -e VLLM_MOE_W2_BASE_CACHE_GB="$BASE_GB" \
  -e VLLM_MOE_W2_PLANES_CACHE=/plane-cache \
  -e VLLM_MOE_W2_STORE_DIR=/packs \
  -e VLLM_MOE_W2_BASE_RAM_GB="$ARENA_GB" \
  -e TRITON_CACHE_DIR=/root/.cache/triton \
  -e TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor \
  "$IMG" \
  --model /model --served-model-name deepseek-v4-flash --trust-remote-code \
  --kv-cache-dtype fp8 --block-size 256 --max-model-len "$MAXLEN" \
  --gpu-memory-utilization "$UTIL" --max-num-batched-tokens 2048 --max-num-seqs 4 \
  --tokenizer-mode deepseek_v4 --no-scheduler-reserve-full-isl \
  $TPARGS \
  --speculative-config '{"method": "deepseek_mtp", "num_speculative_tokens": 2}' \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"],"cudagraph_capture_sizes":[1,2,4,8,12,16,24]}' \
  --port "$PORT"
BUILD=$(docker exec "$NAME" cat /opt/moet-checks/SOURCE.txt 2>/dev/null | grep -v '^#' | head -1 || true)
echo "started $NAME (sm_89, gpus=${GPUS} tp=${TP}, base-cache ${BASE_GB} GiB/rank, arena ${ARENA_GB} GiB/rank, store ${STORE}->/packs, port ${PORT}, network=$NETWORK, max-model-len=$MAXLEN, util=$UTIL)"
echo "image build (vllm fork SHA): ${BUILD:-UNKNOWN - pre-observability image, REBUILD from current main}"
echo "healthy-boot markers:  docker logs -f $NAME 2>&1 | grep -E 'moe_w2'"
echo "  1) 'moe_w2: env ... does NOTHING — did you mean ...' (only if you typoed a knob)"
echo "  2) 'sm_89 Triton emulation ready on <GPU> ... self-test worst_rel=...'"
echo "  3) 'moe_w2 planes: ... -> PINNED HOST RAM (base cache ...)'"
echo "on failure:            docker logs $NAME 2>&1 | grep -B2 -A30 -E 'EngineCore.*(Error|Traceback|CRITICAL)|moe_w2'"
echo "                       docker inspect $NAME --format '{{.State.ExitCode}} {{.State.OOMKilled}}'"
echo "  'Failed core proc(s): {}' (empty set) = EngineCore died with NO Python exception:"
echo "    host OOM-killer  -> sudo dmesg -T | grep -iE 'oom|killed process' | tail -5   (and: free -g)"
echo "    native crash     -> faulthandler stacks are in docker logs (PYTHONFAULTHANDLER=1 is baked in)"
