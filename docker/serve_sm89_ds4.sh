#!/usr/bin/env bash
# ========================== WHAT THIS IS  (ELI5) ============================
# Starts DeepSeek-V4-Flash on Ada GPUs (RTX 4090 / RTX 2000-6000 Ada) with
# the 2-bit "Moet" expert compression. The FIRST run quantizes the model
# into a reusable on-disk "pack" (slow); every later run boots from that
# pack in ~10 minutes. Watch progress:  docker logs -f moet
#
# The three things you will actually change:
#
# 1) PRODUCTION / make the API reachable.
#    The default network=none is a deliberate air-gap for the first
#    (quantizing) boot. Once boots come up warm and healthy, run:
#
#        NETWORK=host RESTART=unless-stopped ./docker/serve_sm89_ds4.sh
#
#    Clients then use  http://<this-host>:8001/v1  (OpenAI-compatible).
#    Quick test:       curl http://localhost:8001/v1/models
#
# 2) TWO GPUS (bigger expert pool -> faster decode).
#
#        TP=2 GPUS='"device=0,1"' ./docker/serve_sm89_ds4.sh
#
#    WARNING: TP2 needs roughly 2x the host RAM (two pinned arenas plus
#    ~26 GiB of /dev/shm). On this 38 GiB host a TP2 attempt once hard-hung
#    the whole box - raise host RAM / the LXC memory cap FIRST. The MEM_GB
#    hard cap below contains a failed attempt (container dies, host lives).
#
# 3) IMAGE VERSION.
#
#        IMG=vllm-moet-sm89:v0251 ./docker/serve_sm89_ds4.sh
#
#    v024 = vLLM 0.24 lineage, v0251 = vLLM 0.25.1 lineage. Build them with
#    Dockerfile.sm89-v024 / Dockerfile.sm89-v0251 from the repo root.
#
# Every knob below has a one-line comment. The deep WHY (what broke when a
# default was different, exact RAM math, ZFS notes) is collected in the
# TECHNICAL NOTES block at the BOTTOM of this file - read it when something
# misbehaves, not to launch. Healthy-boot log markers are echoed at start.
# ============================================================================
set -euo pipefail
MODEL=${MODEL:-/root/models/DeepSeek-V4-Flash}   # checkpoint dir (read-only)
CACHE=${CACHE:-/root/models/moet-cache}          # quant caches; ~90 GB free
NETWORK=${NETWORK:-none}     # 'none' = unreachable (first boot); 'host' = production
RESTART=${RESTART:-no}       # production: unless-stopped (survives crashes/reboots)
MAXLEN=${MAXLEN:-16384}      # context length
UTIL=${UTIL:-0.90}           # fraction of VRAM vLLM may use
PORT=${PORT:-8001}           # API port (reachable only with NETWORK=host)
GPUS=${GPUS:-'"device=0"'}   # which GPUs; two cards: '"device=0,1"' + TP=2
TP=${TP:-1}                  # tensor parallelism = number of GPUs used
BASE_GB=${BASE_GB:-20}       # GPU expert-pool GiB per rank - THE speed knob
STORE=${STORE:-$CACHE/packs} # on-disk quant pack (real fs, NOT overlayfs)
ARENA_GB=${ARENA_GB:-14}     # pinned host-RAM cache over the pack, per rank
MEM_GB=${MEM_GB:-30}         # HARD container RAM cap - protects the host
NAME=${NAME:-moet}
IMG=${IMG:-vllm-moet-sm89:v024}

mkdir -p "$CACHE/planes" "$CACHE/jit" "$STORE"
docker rm -f "$NAME" 2>/dev/null || true
# TP>1 needs host IPC + big /dev/shm for inter-worker tensors; TP1 must NOT
# pay that (see TECHNICAL NOTES: the ~26 GiB shm helped sink a TP2 first run).
if [ "$TP" -gt 1 ]; then
  TPARGS="--tensor-parallel-size $TP --disable-custom-all-reduce"
  IPCARGS="--ipc host --shm-size 64g"
else
  TPARGS=""
  IPCARGS="--shm-size 8g"
fi
docker run -d --name "$NAME" --restart "$RESTART" --gpus "$GPUS" --network "$NETWORK" $IPCARGS \
  --memory "${MEM_GB}g" --memory-swap "$((MEM_GB + 2))g" \
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
echo "started $NAME (sm_89, gpus=${GPUS} tp=${TP}, memcap=${MEM_GB}g, base-cache ${BASE_GB} GiB/rank, arena ${ARENA_GB} GiB/rank, store ${STORE}->/packs, port ${PORT}, network=$NETWORK, restart=$RESTART, max-model-len=$MAXLEN, util=$UTIL)"
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
exit 0

# ========================= TECHNICAL NOTES ==================================
# (Nothing below runs - the WHY behind each default, kept for debugging.
#  Fuller story: docs/ada-sm89-port.md, § Observability + Troubleshooting.)
#
# WHY the moe_w2 envs are NOT optional on Ada + DS4-Flash:
#   VLLM_MOE_W2_BASE_CACHE_GB   DS4-Flash 2-bit planes are ~1.69 GiB/layer x
#                               43 layers = ~73 GiB. GPU-RESIDENT planes fit
#                               only the 96 GB SM120 board - on 24-48 GB Ada
#                               the base cache is mandatory: planes live
#                               host-side, the GPU streams BASE_GB worth of
#                               expert slots. Pool coverage is the dominant
#                               perf knob (DS4 measured: 15%->19% coverage
#                               = +33% decode) - raise BASE_GB when VRAM
#                               allows and watch the '[base] KPI' log line.
#                               The plane-budget preflight fails EARLY with
#                               this remedy if the env is dropped.
#   VLLM_MOE_W2_DELTA_GB=0      the w4/w4q FP4 delta tiers are not ported
#                               to Ada (_require_kernels names this fix).
#   VLLM_MOE_W2_PLANES_CACHE    PLANES, plural - the singular spelling is a
#                               silent no-op (the env-typo guard warns with
#                               a did-you-mean). Persists quantized planes:
#                               without it every boot re-quantizes.
#
# SMALL-HOST RAM TIER (required on this 38 GiB host; all-pinned base needs
# ~80 GiB): VLLM_MOE_W2_STORE_DIR moves the 2-bit base out of pinned RAM
# into an on-disk pack - a read-mostly mmap/page-cache file, NOT swap, so
# ZFS is safe for it (swap-on-ZFS deadlocks; a pack file does not). The
# pack doubles as the persistent quant cache (boot-from-pack skips requant
# per layer). VLLM_MOE_W2_BASE_RAM_GB pins an MRU arena over the pack, per
# rank. STORE needs a REAL filesystem (bind mount, NOT overlayfs) and ~80 GB
# free.
#
# NETWORK=none default: the first boot writes the pack; a half-quantized
# server should not be reachable. It also means PORT is unreachable until
# a NETWORK=host relaunch - that is the point.
#
# TP2 ON A SMALL HOST: two pinned arenas + ~26 GiB /dev/shm for inter-worker
# tensor passing did not fit 38 GiB - the first TP2 attempt swap-stormed the
# host into a hard hang (even sshd stopped). Hence: TP1 keeps shm at 8g and
# skips --ipc host entirely; TP2 requires more host RAM first. TP>1 also
# needs --disable-custom-all-reduce on these boards.
#
# MEM_GB HARD CAP: belt-and-suspenders against the above. If a load
# overshoots, the cgroup OOM-killer takes the CONTAINER (fast, contained)
# instead of the host swap-storming slow USB swap into a hang. Sized for
# TP1 (RSS ~20-26 GiB) + margin; only +2 GiB swap so it cannot lean on USB
# swap. Also cap the LXC itself: pct set 100 -memory 32000 -swap 4000.
#
# IMAGE IDENTITY + TRIAGE: the start banner prints the vllm fork SHA baked
# into the image (/opt/moet-checks/SOURCE.txt; 'UNKNOWN' = stale image,
# rebuild). 'Failed core proc(s): {}' in a crash means NO Python exception
# existed - check dmesg for the OOM-killer or the faulthandler stacks in
# docker logs (PYTHONFAULTHANDLER=1 is baked into the image).
# ============================================================================
