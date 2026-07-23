#!/usr/bin/env bash
# ========================== WHAT THIS IS  (ELI5) ============================
# Starts DeepSeek-V4-Flash on Ada GPUs (RTX 4090 / RTX 2000-6000 Ada) with
# the 2-bit "Moet" expert compression. The FIRST run quantizes the model
# into a reusable on-disk "pack" (slow); every later run boots from that
# pack in ~10 minutes. Watch progress:  docker logs -f moet
#
# The four things you will actually change:
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
# 2) TWO GPUS. Select the TP size and the expert residency. These are two
#    independent choices.
#
#      # gpu residency. Use this mode when the host has much VRAM and little RAM
#      # (for example, 2 x 48 GB cards and 20 to 30 GB RAM). The 2-bit base shards
#      # across the cards. The base stays on the GPUs. There is no host base cache.
#      # The RAM use is low. This mode is the fastest.
#        RESIDENCY=gpu TP=2 GPUS='"device=0,1"' UTIL=0.98 ./docker/serve_sm89_ds4.sh
#
#      # host residency (default). Use this mode when the host has much RAM and
#      # little VRAM. The planes stay in pinned host RAM. The GPU streams a pool.
#      # host residency at TP2 needs approximately 2 x the host RAM (two pinned
#      # arenas and approximately 26 GiB of /dev/shm). Increase the host RAM or the
#      # LXC cap before you use host residency at TP2.
#
#  ############################################################################
#  ##  THE CACHE IS SPECIFIC TO TP SIZE, RESIDENCY, AND QUANTIZER SETTINGS. ##
#  ##  To serve on N GPUs, do the first-run quantization at TP=N with the    ##
#  ##  same residency. host residency writes base.rank<i>of<N>.pack.         ##
#  ##  gpu residency writes the plane cache. A cache from TP1 (or host) does ##
#  ##  not apply to a TP2 (or gpu) run. SCALE_REFIT also changes the cache.  ##
#  ##  new quantization. A new quantization needs 15 to 20 minutes.          ##
#  ##  Select the TP size and the residency before the first boot.           ##
#  ##  A first-run quantization is long. Keep READY_TIMEOUT_S high. A small  ##
#  ##  value stops the quantization (see below).                             ##
#  ############################################################################
#
#    The MEM_GB hard cap contains a failed attempt. The container stops. The
#    host continues to operate.
#
# 3) IMAGE VERSION.
#
#        IMG=vllm-moet-sm89:v0251 ./docker/serve_sm89_ds4.sh
#
#    v0251 is the current vLLM 0.25.1 lineage. Build it with
#    Dockerfile.sm89-v0251 from the repository root.
#
# 4) QUALITY, SPEED, AND MEMORY CONTROLS.
#
#    SCALE_REFIT=1 is the normal Ada setting. During quantization, it tests a
#    second scale for each block of 32 expert weights. It uses the second scale
#    only when the exact block error is smaller. It does not use more VRAM and
#    it does not change the serving kernel. Set SCALE_REFIT=0 only for a
#    comparison or a rollback. A change to SCALE_REFIT causes a new 15 to 20
#    minute quantization because the cache content changes.
#
#    PREFIX_CACHING=0 is the correctness baseline. Prefix caching saves the KV
#    data for repeated prompt prefixes. It can make a repeated long prompt load
#    faster, but it does not make normal token generation faster. Set it to 1
#    only after the saved long agent prompt gives the correct answer with 0.
#    If the answer changes with 1, set it back to 0.
#
#    MTP_TOKENS=0 disables speculative decoding. MTP uses a small draft head to
#    propose tokens. Accepted draft tokens can increase generation speed. A low
#    acceptance value adds work and can reduce speed. The measured Ada run
#    accepted approximately one half token for each step, so 0 is the default.
#    Test MTP_TOKENS=1 only after the answer is correct. Keep 1 only when the
#    generated-token rate increases and the answer does not change.
#
#    NUM_SEQS=4 lets the scheduler run four requests. It does not reserve four
#    complete 262K contexts. The measured KV pool holds approximately 286K
#    tokens in total. Thus, it holds one 262K request or several shorter
#    requests. Set NUM_SEQS=1 or 2 when long requests cause memory pressure.
#
#    BATCHED_TOKENS controls the number of prompt tokens in one prefill step.
#    A larger value can load prompts faster but uses more workspace. Reduce it
#    after an out-of-memory error during prompt load. CUDAGRAPH_SIZES controls
#    the request shapes that vLLM captures. A shorter list saves some VRAM.
#    These controls do not reduce the model weight allocation.
#
#    UTIL controls the vLLM VRAM budget. Keep 0.98 on the two 48 GiB cards.
#    Reduce it only when another process needs VRAM. A smaller value can leave
#    no memory for KV blocks because the model weights use approximately
#    43 GiB on each card.
#
#    The API accepts the names `deepseek-v4-flash` and `auto`. Keep both names
#    when a client selects `auto`. The first name is the name in API responses.
#    The DeepSeek tool-call and reasoning parsers convert model text to the
#    structured data that Claude Code and other agents need. Keep these parser
#    options for agent use. They are not needed for a plain text-only client.
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
MAXLEN=${MAXLEN:-262144}     # maximum context length; lower for the first boot test
UTIL=${UTIL:-0.98}           # fraction of VRAM vLLM may use (raised from 0.96; 2x48GiB
                             # at TP2 RESIDENCY=gpu leaves ~5 GiB/card for everything-not-
                             # weights, so every basis point matters. 0.98 + trimmed graphs
                             # + smaller batch is the working budget for 131K sparse MLA).
BATCHED_TOKENS=${BATCHED_TOKENS:-1024}  # max-num-batched-tokens; prefill uses most workspace
NUM_SEQS=${NUM_SEQS:-4}      # request scheduler limit, not four full-length KV allocations
CUDAGRAPH_SIZES=${CUDAGRAPH_SIZES:-1,2,4,8}  # cudagraph_capture_sizes, comma-sep (trimmed
                             # from [1,2,4,8,12,16,24] to reduce captured buffers and
                             # workspaces. Graph capture does not copy the model weights).
MTP_TOKENS=${MTP_TOKENS:-0}  # speculative tokens; read header section 4 before you enable
PREFIX_CACHING=${PREFIX_CACHING:-0}  # reuse repeated prompt KV; read header section 4
SCALE_REFIT=${SCALE_REFIT:-1}  # normal W2 conversion; 0 is for comparison or rollback
PORT=${PORT:-8001}           # API port (reachable only with NETWORK=host)
GPUS=${GPUS:-'"device=0"'}   # which GPUs; two cards: '"device=0,1"' + TP=2
TP=${TP:-1}                  # tensor parallelism = number of GPUs used
RESIDENCY=${RESIDENCY:-host} # 'host' = 2-bit base in pinned RAM + GPU pool (RAM-heavy);
                             # 'gpu'  = base sharded ONTO the GPUs, no host cache (VRAM-heavy,
                             # low RAM). TP, RESIDENCY, and SCALE_REFIT identify the quant cache.
FORCE_RESIDENT=${FORCE_RESIDENT:-0}  # gpu residency: set 1 to bypass the boot-guard VRAM-budget
                             # check (VLLM_MOE_W2_FORCE_RESIDENT). The guard refuses knife-edge
                             # configs that DO serve on >=48 GiB cards; set 1 to consent past the
                             # refusal. No effect under RESIDENCY=host.
READY_TIMEOUT_S=${READY_TIMEOUT_S:-1800}  # engine-ready wait (vLLM default 600); a long
                             # first-run quant is KILLED at 600s -> raise it.
BASE_GB=${BASE_GB:-20}       # host residency: GPU expert-pool GiB/rank (THE speed knob).
                             # gpu residency forces BASE_CACHE_GB=0 (base lives on the GPUs).
STORE=${STORE:-$CACHE/packs} # host residency only: on-disk quant pack (real fs, NOT overlayfs)
ARENA_GB=${ARENA_GB:-14}     # host residency only: pinned host-RAM cache over the pack, per rank
MEM_GB=${MEM_GB:-30}         # HARD container RAM cap - protects the host
NAME=${NAME:-moet}
IMG=${IMG:-vllm-moet-sm89:v0251}

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
if [ "$MTP_TOKENS" -gt 0 ]; then
  MTPARGS="--speculative-config {\"method\":\"deepseek_mtp\",\"num_speculative_tokens\":$MTP_TOKENS}"
else
  MTPARGS=""
fi
if [ "$PREFIX_CACHING" = 1 ]; then
  PREFIXARGS=""
else
  PREFIXARGS="--no-enable-prefix-caching"
fi
# Expert residency. Refer to header section 2 and the TECHNICAL NOTES.
# 'gpu' sets BASE_CACHE_GB=0. The 2-bit base then stays on the GPUs. TP shards it.
# There is no host pack or arena. 'host' uses a pinned-RAM base, an on-disk pack, and a
# GPU pool. Both modes keep the plane cache. A change to the residency or the TP size
# makes the quant cache invalid. The engine then does a new quantization.
if [ "$RESIDENCY" = gpu ]; then
  RESVOL=""
  RESENV="-e VLLM_MOE_W2_BASE_CACHE_GB=0 -e VLLM_MOE_W2_PLANES_CACHE=/plane-cache -e VLLM_MOE_W2_FORCE_RESIDENT=$FORCE_RESIDENT"
else
  RESVOL="-v $STORE:/packs"
  RESENV="-e VLLM_MOE_W2_BASE_CACHE_GB=$BASE_GB -e VLLM_MOE_W2_PLANES_CACHE=/plane-cache -e VLLM_MOE_W2_STORE_DIR=/packs -e VLLM_MOE_W2_BASE_RAM_GB=$ARENA_GB"
fi
docker run -d --name "$NAME" --restart "$RESTART" --gpus "$GPUS" --network "$NETWORK" $IPCARGS \
  --memory "${MEM_GB}g" --memory-swap "$((MEM_GB + 2))g" \
  -v "$MODEL":/model:ro \
  -v "$CACHE/planes":/plane-cache \
  -v "$CACHE/jit":/root/.cache \
  $RESVOL \
  -e VLLM_MOE_W2=1 \
  -e VLLM_MOE_W2_DELTA_GB=0 \
  -e VLLM_MOE_W2_SCALE_REFIT="$SCALE_REFIT" \
  $RESENV \
  -e VLLM_ENGINE_READY_TIMEOUT_S="$READY_TIMEOUT_S" \
  -e TRITON_CACHE_DIR=/root/.cache/triton \
  -e TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor \
  "$IMG" \
  --model /model --served-model-name deepseek-v4-flash auto --trust-remote-code \
  --kv-cache-dtype fp8 --block-size 256 --max-model-len "$MAXLEN" \
  --gpu-memory-utilization "$UTIL" --max-num-batched-tokens "$BATCHED_TOKENS" --max-num-seqs "$NUM_SEQS" \
  --tokenizer-mode deepseek_v4 --no-scheduler-reserve-full-isl \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  $TPARGS $PREFIXARGS $MTPARGS \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"],"cudagraph_capture_sizes":['"$CUDAGRAPH_SIZES"']}' \
  --port "$PORT"
BUILD=$(docker exec "$NAME" cat /opt/moet-checks/SOURCE.txt 2>/dev/null | grep -v '^#' | head -1 || true)
echo "started $NAME (sm_89, gpus=${GPUS} tp=${TP} residency=${RESIDENCY}, memcap=${MEM_GB}g, ready-timeout=${READY_TIMEOUT_S}s, port ${PORT}, network=$NETWORK, restart=$RESTART, max-model-len=$MAXLEN, util=$UTIL, batched=$BATCHED_TOKENS, seqs=$NUM_SEQS, mtp=$MTP_TOKENS, prefix-cache=$PREFIX_CACHING, scale-refit=$SCALE_REFIT, graphs=[$CUDAGRAPH_SIZES])"
if [ "$RESIDENCY" = gpu ]; then
  echo "  residency=gpu: 2-bit base GPU-RESIDENT (BASE_CACHE_GB=0), sharded across ${TP} rank(s); no host pack/arena. FORCE_RESIDENT=${FORCE_RESIDENT} (1 = bypass the boot-guard VRAM-budget refusal on >=48 GiB cards). Watch for: 'moe_w2 planes: ... GPU-RESIDENT' and ~37 GiB/card VRAM."
else
  echo "  residency=host: base-cache ${BASE_GB} GiB/rank + arena ${ARENA_GB} GiB/rank + pack ${STORE}->/packs."
fi
echo "image build (upstream v0.25.1 SHA): ${BUILD:-UNKNOWN - pre-observability image, REBUILD from current main}"
echo "healthy-boot markers:  docker logs -f $NAME 2>&1 | grep -E 'moe_w2|o_proj'"
echo "  1) 'moe_w2: env ... does NOTHING — did you mean ...' (only if you typoed a knob)"
echo "  2) 'sm_89 Triton emulation ready on <GPU> ... self-test worst_rel=...'"
echo "  3) 'moe_w2 planes: ... -> GPU-RESIDENT' or '-> PINNED HOST RAM'"
echo "  4) 'DeepSeek V4 o_proj: using native SM89 block-scaled FP8 grouped matmul'"
echo "on failure:            docker logs $NAME 2>&1 | grep -B2 -A30 -E 'EngineCore.*(Error|Traceback|CRITICAL)|moe_w2|o_proj'"
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
#   VLLM_MOE_W2_BASE_CACHE_GB   DS4-Flash 2-bit planes use approximately
#                               73 GiB. TP2 GPU residency shards them to
#                               approximately 36 GiB per card. Use this mode
#                               on two 48 GiB cards with a small host. Host
#                               residency moves the base to host RAM and uses
#                               BASE_CACHE_GB as the GPU expert-pool size.
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
# TP2 ON A SMALL HOST (host residency): two pinned arenas + ~26 GiB /dev/shm for
# inter-worker tensor passing did not fit 38 GiB - the first TP2 attempt swap-stormed
# the host into a hard hang (even sshd stopped). A later base-cache TP2 run OOM-killed
# a worker at the 30 GB container cap (OOMKilled=true, 2026-07-19). Hence: TP1 keeps
# shm at 8g and skips --ipc host; host-residency TP2 needs more host RAM first. TP>1
# also needs --disable-custom-all-reduce on these boards.
#
# EXPERT RESIDENCY. The quantization cache is specific to (TP size, residency).
#   host (default): the 2-bit base stays in pinned host RAM. It uses an on-disk pack
#     and an MRU arena. The GPU keeps a BASE_GB pool. Use this mode for a host with much
#     RAM and little VRAM. host residency at TP2 uses much RAM (see above).
#   gpu (BASE_CACHE_GB=0): the base shards onto the GPUs. TP splits approximately 73 GiB
#     into approximately 36 GiB per rank. This fits two 48 GB cards. There is no host
#     pack or arena. Use this mode for a host with much VRAM and little RAM (this box).
#     This mode prevents the host-RAM out-of-memory condition.
#   The cache depends on the TP size and the residency. host writes
#     base.rank<i>of<N>.pack. gpu writes the plane cache. A cache from TP1 or host does
#     not apply to a TP2 or gpu run. In that condition the engine does a new
#     quantization (15 to 20 minutes). Select the TP size and the residency before the
#     first boot. To serve on N GPUs, do the quantization at TP=N.
#
# READY_TIMEOUT_S. VLLM_ENGINE_READY_TIMEOUT_S has a default of 600 seconds. A first-run
# quantization needs 15 to 20 minutes. This time is more than the default. Thus the
# engine-ready wait stops the workers. It then shows "Engine core initialization failed
# ... Failed core proc(s): {}". This launcher sets 1800 seconds. The quantization can
# then complete.
#
# MEM_GB HARD CAP: belt-and-suspenders against the above. If a load
# overshoots, the cgroup OOM-killer takes the CONTAINER (fast, contained)
# instead of the host swap-storming slow USB swap into a hang. Sized for
# TP1 (RSS ~20-26 GiB) + margin; only +2 GiB swap so it cannot lean on USB
# swap. Also cap the LXC itself: pct set 100 -memory 32000 -swap 4000.
#
# VRAM-BUDGET TUNING (UTIL / BATCHED_TOKENS / NUM_SEQS / CUDAGRAPH_SIZES /
#   MTP_TOKENS): at TP=2 RESIDENCY=gpu on 2x48 GiB cards, the 2-bit base
#   (~73 GiB) shards to ~36 GiB/rank, and weights + norms + embeddings take
#   the budget to ~43 GiB/card. That leaves ~5 GiB/card for CUDA runtime,
#   activations, workspace, and KV cache. Sparse MLA's KV cost is tiny
#   (~584 B/token; 262K x 1 seq = 146 MiB), but vLLM reserves a num_blocks
#   budget up front; without trimming the other levers it ends up negative.
#   Defaults are the correctness-first configuration for this box:
#     UTIL=0.98            (raised from 0.96; +0.97 GiB/card vs 0.94)
#     BATCHED_TOKENS=1024  (prefill is the workspace spike)
#     NUM_SEQS=4           (four short requests; full 262K requests share KV)
#     CUDAGRAPH_SIZES=1,2,4,8  (was 1,2,4,8,12,16,24; ~80-100 MiB each)
#     MTP_TOKENS=0         (the live draft accepted only about one half token)
#     PREFIX_CACHING=0     (isolate long-context correctness before reuse)
#     SCALE_REFIT=1        (same-size W2 conversion with lower block SSE)
#   Override any of them via env to test smaller-first smoke runs (e.g.
#   MAXLEN=8192 UTIL=0.94 ./docker/serve_sm89_ds4.sh).
#
# IMAGE IDENTITY + TRIAGE: the start banner prints the upstream v0.25.1 SHA
# the patches/ set was diffed against (/opt/moet-checks/SOURCE.txt;
# 'UNKNOWN' = stale image, rebuild). 'Failed core proc(s): {}' in a crash
# means NO Python exception
# existed - check dmesg for the OOM-killer or the faulthandler stacks in
# docker logs (PYTHONFAULTHANDLER=1 is baked into the image).
# ============================================================================
