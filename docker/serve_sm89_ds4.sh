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
#    Defaults target this host: 2 x RTX 4090 D 48 GiB, the patched P2P
#    driver, TP2 GPU residency, and the existing refit plane cache. Run:
#
#        ./docker/serve_sm89_ds4.sh
#
#    Clients then use  http://<this-host>:8001/v1  (OpenAI-compatible).
#    Quick test:       curl http://localhost:8001/v1/models
#    To isolate a first quantizing boot explicitly, pass NETWORK=none.
#
# 2) TWO GPUS. Select the TP size and the expert residency. These are two
#    independent choices.
#
#    TP>1 uses vLLM's custom P2P all-reduce. On GeForce RTX 4090 D that needs a
#    patched open-gpu-kernel-modules driver (stock drivers block P2P):
#      https://github.com/Duanyll/open-gpu-kernel-modules/tree/595.71.05-p2p-48g
#    WITHOUT that patch pass CUSTOM_ALL_REDUCE=0 (NCCL all-reduce) or TP may fail
#    to init / silently fall back.
#
#      # gpu residency. Use this mode when the host has much VRAM and little RAM
#      # (for example, 2 x 48 GB cards and 20 to 30 GB RAM). The 2-bit base shards
#      # across the cards. The base stays on the GPUs. There is no host base cache.
#      # The RAM use is low. This mode is the fastest.
#        RESIDENCY=gpu TP=2 GPUS='"device=0,1"' UTIL=0.98 ./docker/serve_sm89_ds4.sh
#
#      # host residency. Use this mode when the host has much RAM and
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
#    v0251 is the canonical living tag (vLLM 0.25.1 lineage) and the launcher
#    default (see the IMG line below). Build it with Dockerfile.sm89-v0251
#    from the repository root. Roll back to the pre-native baseline with
#    IMG=vllm-moet-sm89:v0251-pre-native.
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
#    MTP_TOKENS=1 enables speculative decoding. A small draft head proposes the
#    next tokens. Accepted proposals increase the output speed. A benchmark on two
#    RTX 4090 D cards (sm_89, RESIDENCY=gpu, TP=2) measured the output rate:
#    MTP_TOKENS=1 gives 56 tokens/s in decode and 104 tokens/s with four parallel
#    requests. With MTP_TOKENS=0, the output rate decreases to 38 tokens/s and
#    73 tokens/s. The draft acceptance is 0.5, but MTP still increases the speed.
#    Keep MTP_TOKENS=1.
#
#    NUM_SEQS=4 lets the scheduler run four requests. It does not reserve four
#    complete 262K contexts. The measured KV pool holds approximately 286K
#    tokens in total. Thus, it holds one 262K request or several shorter
#    requests. Set NUM_SEQS=1 or 2 when long requests cause memory pressure.
#
#    BATCHED_TOKENS sets the number of prompt tokens in one prefill step. For this
#    model, the prefill speed does not depend on this number. A benchmark put a
#    55,000-token prompt at 883 tokens/s with 1024 and at 879 tokens/s with 2048.
#    A larger value does not increase the prefill speed, but it uses more workspace.
#    With MTP_TOKENS=1, a value above 1024 causes an out-of-memory error during
#    startup. Keep BATCHED_TOKENS=1024. CUDAGRAPH_SIZES controls the request shapes
#    that vLLM captures. A shorter list saves VRAM. These controls do not decrease
#    the model weight allocation.
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
# QUALITY_PREFILL=1: streamed full-precision prefill (long-context quality mode).
# Per-chunk ensure-mode: each prefill chunk's routed experts are promoted from
# the pinned host arena into a deferred GPU pool and the w13/w2 GEMMs run at
# FP8 precision from FP4-STORAGE slots (exact e2m1->e4m3 decode); decode stays
# on the 2-bit base + native cubin. Costs: ~1.5-2.5 min per 131K prefill (H2D-
# bound), MTP off, ~65 GiB/rank pinned host arena, SCALE_REFIT=0 (a DIFFERENT
# plane cache -> first boot re-quantizes into planes-qp, 15-20 min).
QUALITY_PREFILL=${QUALITY_PREFILL:-0}
if [ "$QUALITY_PREFILL" = 1 ]; then
  MAXLEN=${MAXLEN:-131072}       # KV floor scales with max-model-len
  UTIL=${UTIL:-0.90}             # leave ~4.8 GiB/card post-KV: the deferred
                                 # pool (1.5) + workspace reserve (3.0) live there
  BATCHED_TOKENS=${BATCHED_TOKENS:-4096}  # chunk size = the H2D amortizer
  MTP_TOKENS=${MTP_TOKENS:-0}    # quality mode is prefill-bound; >1024 OOMs with MTP
  SCALE_REFIT=${SCALE_REFIT:-0}  # REQUIRED by fp4-storage (build enforces)
  FP8_DELTA_GB=${FP8_DELTA_GB:-1.5}  # 256 x 6 MiB = one full layer union @ TP2
  QP_ENV="-e VLLM_MOE_W2_FP8_STORE=fp4 -e VLLM_MOE_W2_PREFILL_FP4=ensure"
  PLANES_SUBDIR=planes-qp        # refit=0 changes plane bytes; a shared dir
                                 # would WIPE the refit=1 cache on every flip
else
  QP_ENV=""
  PLANES_SUBDIR=planes
fi
MODEL=${MODEL:-/root/models/DeepSeek-V4-Flash-0731}   # checkpoint dir (read-only)
CACHE=${CACHE:-/root/models/moet-cache}          # quant caches; ~90 GB free
JIT_CACHE=${JIT_CACHE:-$CACHE/jit}               # compiler cache may be shared
                                                  # across weight-cache namespaces
NETWORK=${NETWORK:-host}     # this host serves directly; use 'none' to isolate quantization
RESTART=${RESTART:-no}       # production: unless-stopped (survives crashes/reboots)
MAXLEN=${MAXLEN:-262144}     # maximum context length; lower for the first boot test
UTIL=${UTIL:-0.98}           # fraction of VRAM vLLM may use (raised from 0.96; 2x48GiB
                             # at TP2 RESIDENCY=gpu leaves ~5 GiB/card for everything-not-
                             # weights, so every basis point matters. 0.98 + trimmed graphs
                             # + smaller batch is the working budget for 131K sparse MLA).
BATCHED_TOKENS=${BATCHED_TOKENS:-1024}  # max-num-batched-tokens; keep 1024 with MTP on (see section 4)
NUM_SEQS=${NUM_SEQS:-4}      # request scheduler limit, not four full-length KV allocations
CUDAGRAPH_SIZES=${CUDAGRAPH_SIZES:-1,2,4,8}  # cudagraph_capture_sizes, comma-sep (trimmed
                             # from [1,2,4,8,12,16,24] to reduce captured buffers and
                             # workspaces. Graph capture does not copy the model weights).
BREAKABLE_CUDAGRAPH=${BREAKABLE_CUDAGRAPH:-auto}  # auto = leave vLLM default; 0 = keep
                             # torch.compile/Inductor enabled instead of auto breakable graphs.
MTP_TOKENS=${MTP_TOKENS:-1}  # speculative tokens; keep 1 (see header section 4)
SPECULATIVE_CONFIG=${SPECULATIVE_CONFIG:-}  # compact JSON override for embedded
                             # speculative heads (for example 0731 DSpark).
                             # Empty preserves the MTP_TOKENS behavior above.
PREFIX_CACHING=${PREFIX_CACHING:-1}  # reuse repeated prompt KV; read header section 4
SCALE_REFIT=${SCALE_REFIT:-1}  # normal W2 conversion; 0 is for comparison or rollback
FP8_DELTA_GB=${FP8_DELTA_GB:-0}  # FP8-e4m3 delta PREFILL tier (Ada native FP8 MMA).
                                 # >0 enables it: FP8-resident prefill pairs divert to
                                 # the w8 Triton kernel (higher precision than the bare
                                 # 2-bit prefill + native fast MMA + no SCALE_REFIT
                                 # conflict). FP4 delta tiers are mutually exclusive and
                                 # pinned OFF. 0 (default) = inert (2-bit base only).
                                 # QUALITY_PREFILL=1 sets this to 1.5; the pool is
                                 # allocated POST-KV (deferred; VLLM_MOE_W2_FP8_DELTA_GB
                                 # caps it) — no build-time OOM, single boot.
SM89_NATIVE=${SM89_NATIVE:-}   # native e4m3 QMMA decode cubin: empty = image
                               # default (ON when /cubit-share has the cubin,
                               # parity-gated at boot); 0 = force Triton-only
# Removed experiment knobs (2026-07-24), all measured failures or unported
# paths on Ada -- kept only in documentation: DELTA_GB / DELTA_SPLIT /
# PREFILL_FP4 (FP4 delta tiers: not ported to Ada; w4q measured slower and
# incompatible with SCALE_REFIT=1), TOPP / TOPP_MIN / TOPP_RENORM (adaptive
# expert top-p: major regression), SM89_WARPS / SM89_IMPL / SM89_BLOCK_N
# (Triton tuning: warps 1/4 and BLOCK_N 32/64 regress, fp8mma gained <=3%),
# SM89_W4Q (rejected). Evidence: NEXT.md, kernels/MANIFEST.md,
# docs/solutions/architecture-patterns/. The runtime env vars still exist --
# pass them via EXTRA_DOCKER_ENV for a one-off experiment.
EXTRA_DOCKER_ENV=${EXTRA_DOCKER_ENV:-}  # optional raw docker env args, e.g. "-e FOO=1"
EXTRA_ARGS=${EXTRA_ARGS:-}       # optional raw vLLM CLI args, e.g. "--cudagraph-metrics"
PORT=${PORT:-8001}           # API port (reachable only with NETWORK=host)
GPUS=${GPUS:-'"device=0,1"'} # this host's two 48 GiB RTX 4090 D cards
TP=${TP:-2}                  # tensor parallelism = number of GPUs used
CUSTOM_ALL_REDUCE=${CUSTOM_ALL_REDUCE:-1}  # 1 = keep vLLM's P2P custom all-reduce; 0 = disable
                             # (-> --disable-custom-all-reduce, NCCL fallback). Custom all-reduce
                             # needs the open-gpu-kernel-modules P2P patch on RTX 4090 D - see
                             # header section 2. Set 0 if your driver lacks that patch.
RESIDENCY=${RESIDENCY:-gpu}  # 'host' = 2-bit base in pinned RAM + GPU pool (RAM-heavy);
                             # 'exact' = checkpoint FP4 in host RAM + mandatory GPU
                             # cache; native FP8 MMA for BOTH prefill and decode;
                             # 'gpu'  = base sharded ONTO the GPUs, no host cache (VRAM-heavy,
                             # low RAM). TP, RESIDENCY, and SCALE_REFIT identify the quant cache.
FORCE_RESIDENT=${FORCE_RESIDENT:-1}  # gpu residency: set 1 to bypass the boot-guard VRAM-budget
                             # check (VLLM_MOE_W2_FORCE_RESIDENT). The guard refuses knife-edge
                             # configs that DO serve on >=48 GiB cards; set 1 to consent past the
                             # refusal. No effect under RESIDENCY=host.
READY_TIMEOUT_S=${READY_TIMEOUT_S:-1800}  # engine-ready wait (vLLM default 600); a long
                             # first-run quant is KILLED at 600s -> raise it.
BASE_GB=${BASE_GB:-20}       # host residency: GPU expert-pool GiB/rank (THE speed knob).
                             # gpu residency forces BASE_CACHE_GB=0 (base lives on the GPUs).
EXACT_GB=${EXACT_GB:-30}     # exact residency: FP4-storage expert pool GiB/rank.
STORE=${STORE:-$CACHE/packs} # host residency only: on-disk quant pack (real fs, NOT overlayfs)
ARENA_GB=${ARENA_GB:-14}     # host residency only: pinned host-RAM cache over the pack, per rank
MEM_GB=${MEM_GB:-428}        # current host's HARD container RAM cap
NAME=${NAME:-moet}
IMG=${IMG:-vllm-moet-sm89:v0251}  # canonical living tag (vLLM 0.25.1 lineage),
                             # rebuilt in place by:
                             #   DOCKER_BUILDKIT=1 docker build -f Dockerfile.sm89-v0251 \
                             #     -t vllm-moet-sm89:v0251 .
                             # current content adds the native SM89 W2 decode cubin, packed
                             # KV-group allocator, and DSv4 specialized router (validated
                             # 2026-07-24: +27% short decode, +66% c4 aggregate). Pre-native
                             # rollback image: IMG=vllm-moet-sm89:v0251-pre-native.

if [ "$RESIDENCY" = exact ]; then
  if [ "$SCALE_REFIT" != 0 ]; then
    echo "FATAL: RESIDENCY=exact requires SCALE_REFIT=0 (checkpoint FP4 scales)" >&2
    exit 1
  fi
  PLANES_SUBDIR=planes-exact
  FP8_DELTA_GB=$EXACT_GB
fi

mkdir -p "$CACHE/$PLANES_SUBDIR" "$JIT_CACHE" "$STORE"
docker rm -f "$NAME" 2>/dev/null || true
# TP>1 needs host IPC + big /dev/shm for inter-worker tensors; TP1 must NOT
# pay that (see TECHNICAL NOTES: the ~26 GiB shm helped sink a TP2 first run).
if [ "$TP" -gt 1 ]; then
  TPARGS="--tensor-parallel-size $TP"
  # vLLM's custom (P2P) all-reduce needs the open-gpu-kernel-modules P2P patch on
  # GeForce RTX 4090 D (stock drivers block cudaDeviceCanAccessPeer). Patch:
  #   https://github.com/Duanyll/open-gpu-kernel-modules/tree/595.71.05-p2p-48g
  # Without that patch set CUSTOM_ALL_REDUCE=0 (-> NCCL all-reduce).
  if [ "$CUSTOM_ALL_REDUCE" = 0 ]; then
    TPARGS="$TPARGS --disable-custom-all-reduce"
  fi
  IPCARGS="--ipc host --shm-size 64g"
else
  TPARGS=""
  IPCARGS="--shm-size 8g"
fi
if [ -n "$SPECULATIVE_CONFIG" ]; then
  SPECARGS=(--speculative-config "$SPECULATIVE_CONFIG")
elif [ "$MTP_TOKENS" -gt 0 ]; then
  SPECARGS=(--speculative-config "{\"method\":\"deepseek_mtp\",\"num_speculative_tokens\":$MTP_TOKENS}")
else
  SPECARGS=()
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
elif [ "$RESIDENCY" = exact ]; then
  RESVOL="-v $STORE:/packs"
  RESENV="-e VLLM_MOE_W2_BASE_CACHE_GB=0 -e VLLM_MOE_W2_PLANES_CACHE=/plane-cache -e VLLM_MOE_W2_STORE_DIR=/packs -e VLLM_MOE_W2_BASE_RAM_GB=$ARENA_GB"
else
  RESVOL="-v $STORE:/packs"
  RESENV="-e VLLM_MOE_W2_BASE_CACHE_GB=$BASE_GB -e VLLM_MOE_W2_PLANES_CACHE=/plane-cache -e VLLM_MOE_W2_STORE_DIR=/packs -e VLLM_MOE_W2_BASE_RAM_GB=$ARENA_GB"
fi
# Bind-mount the sm89 fixes over the image's baked-in copies, until
# vllm-moet-sm89:v0251 is rebuilt with them. Three ported fixes:
#  * layout (Bug #2): the paged indexer KV cache is read INTERLEAVED while
#    indexer_k_quant_and_cache_kernel writes SEGREGATED -> garbage/NaN decode
#    scores past L=2048 (the long-context "digit loss").
#  * decode top-k (Bug #1): the radix selectors got an UNCOMPRESSED scan bound
#    and an unclamped k_select=512; when the compressed candidate count < 512
#    (absolute context < 2048) they emit stale/NaN indices -> token salad.
#  * tokenizer reasoning-effort ladder: vLLM 0.25.1 maps most reasoning_effort
#    values to "high" then ignores "high" (silent no-op); only "max" emitted a
#    prefix, and that prefix was the official "high" text. Real "max" was
#    unreachable. Fix: the official low/high/max ladder (encoding_dsv4.py).
# Default ON so every launch picks up the fixes; MOUNT_LAYOUT_FIX=0 disables
# (e.g. once a rebuilt image carries them natively, or for an A/B).
MOUNT_LAYOUT_FIX=${MOUNT_LAYOUT_FIX:-1}
LAYOUT_FIX_VOLS=""
if [ "$MOUNT_LAYOUT_FIX" = "1" ]; then
  REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
  _VLP=/usr/local/lib/python3.12/dist-packages
  for _rel in \
    vllm/v1/attention/ops/triton_paged_mqa_logits_dsv4.py \
    vllm/utils/deep_gemm.py \
    vllm/v1/attention/backends/mla/indexer.py \
    vllm/model_executor/layers/sparse_attn_indexer.py \
    vllm/tokenizers/deepseek_v4.py \
    vllm/tokenizers/deepseek_v4_encoding.py ; do
    if [ -f "$REPO/overlay/vllm/$_rel" ]; then
      LAYOUT_FIX_VOLS="$LAYOUT_FIX_VOLS -v $REPO/overlay/vllm/$_rel:$_VLP/$_rel:ro"
    fi
  done
  if [ -z "$LAYOUT_FIX_VOLS" ]; then
    echo "FATAL: MOUNT_LAYOUT_FIX=1 (default) but no sm89-fix overlay files found under $REPO. " \
      "Refusing to launch: the image ships the buggy sm89 code (Bug #1 + Bug #2 + tokenizer). " \
      "Either run from the repo checkout, or set MOUNT_LAYOUT_FIX=0 to serve the image as-is." >&2
    exit 1
  fi
  unset REPO _VLP _rel
fi
if [ "$BREAKABLE_CUDAGRAPH" = auto ]; then
  BREAKABLE_ENV=""
else
  BREAKABLE_ENV="-e VLLM_USE_BREAKABLE_CUDAGRAPH=$BREAKABLE_CUDAGRAPH"
fi
if [ -n "$SM89_NATIVE" ]; then
  SM89_ENV="-e VLLM_MOE_W2_SM89_NATIVE=$SM89_NATIVE"
else
  SM89_ENV=""
fi
# FP4 delta tiers are not ported to Ada -- pinned off explicitly so a stray
# inherited env can never enable them (see the removed-knobs note above).
# When FP8_DELTA_GB>0, the FP8-e4m3 delta prefill tier is enabled instead
# (mutually exclusive with FP4; SCALE_REFIT stays on — FP8 has no refit conflict).
# Float-safe (awk, not integer `[ -gt ]` — the old test silently treated
# "0.5" as non-numeric and fell through to the FP8-off branch, invalidating
# the 0.5 GiB test) and LOUD on garbage.
if awk -v g="$FP8_DELTA_GB" 'BEGIN { if (g+0 != g) exit 2; exit !(g+0 > 0) }'; then
  DELTA_ENV="-e VLLM_MOE_W2_FP8_DELTA=1 -e VLLM_MOE_W2_FP8_DELTA_GB=$FP8_DELTA_GB -e VLLM_MOE_W2_DELTA_GB=0 -e VLLM_MOE_W2_DELTA_SPLIT=0"
  if [ "$RESIDENCY" = exact ]; then
    DELTA_ENV="$DELTA_ENV -e VLLM_MOE_W2_EXACT_CACHE=1 -e VLLM_MOE_W2_FP8_STORE=fp4"
  fi
elif [ $? -eq 2 ]; then
  echo "FATAL: FP8_DELTA_GB='$FP8_DELTA_GB' is not numeric" >&2; exit 1
else
  DELTA_ENV="-e VLLM_MOE_W2_DELTA_GB=0 -e VLLM_MOE_W2_DELTA_SPLIT=0"
fi
docker run -d --name "$NAME" --restart "$RESTART" --gpus "$GPUS" --network "$NETWORK" $IPCARGS \
  --memory "${MEM_GB}g" --memory-swap "$((MEM_GB + 2))g" \
  -v "$MODEL":/model:ro \
  -v /root/models/DeepSeek-V4-Flash-IQ2:/root/models/DeepSeek-V4-Flash-IQ2:ro \
  -v "$CACHE/$PLANES_SUBDIR":/plane-cache \
  -v "$JIT_CACHE":/root/.cache \
  $RESVOL \
  $LAYOUT_FIX_VOLS \
  -e VLLM_MOE_W2=1 \
  -e VLLM_DSV4_DEFAULT_REASONING_EFFORT="${REASONING_EFFORT:-max}" \
  $DELTA_ENV \
  $QP_ENV \
  -e VLLM_MOE_W2_SCALE_REFIT="$SCALE_REFIT" \
  $RESENV \
  -e VLLM_ENGINE_READY_TIMEOUT_S="$READY_TIMEOUT_S" \
  $BREAKABLE_ENV \
  $SM89_ENV \
  $EXTRA_DOCKER_ENV \
  -e TRITON_CACHE_DIR=/root/.cache/triton \
  -e TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor \
  "$IMG" \
  --model /model --served-model-name deepseek-v4-flash auto --trust-remote-code \
  ${DTYPE:+--dtype $DTYPE} \
  ${ATTN_BACKEND:+--attention-backend $ATTN_BACKEND} \
  --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8}" --block-size 256 --max-model-len "$MAXLEN" \
  --gpu-memory-utilization "$UTIL" --max-num-batched-tokens "$BATCHED_TOKENS" --max-num-seqs "$NUM_SEQS" \
  --tokenizer-mode deepseek_v4 --no-scheduler-reserve-full-isl \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  $TPARGS $PREFIXARGS "${SPECARGS[@]}" \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"],"cudagraph_capture_sizes":['"$CUDAGRAPH_SIZES"']}' \
  ${ENFORCE_EAGER:+--enforce-eager} \
  $EXTRA_ARGS \
  --port "$PORT"
BUILD=$(docker exec "$NAME" cat /opt/moet-checks/SOURCE.txt 2>/dev/null | grep -v '^#' | head -1 || true)
echo "started $NAME (sm_89, gpus=${GPUS} tp=${TP} residency=${RESIDENCY}, memcap=${MEM_GB}g, ready-timeout=${READY_TIMEOUT_S}s, port ${PORT}, network=$NETWORK, restart=$RESTART, max-model-len=$MAXLEN, util=$UTIL, batched=$BATCHED_TOKENS, seqs=$NUM_SEQS, speculative=${SPECULATIVE_CONFIG:-mtp:$MTP_TOKENS}, prefix-cache=$PREFIX_CACHING, scale-refit=$SCALE_REFIT, sm89-native=${SM89_NATIVE:-default-on}, graphs=[$CUDAGRAPH_SIZES], breakable-cudagraph=$BREAKABLE_CUDAGRAPH, quality-prefill=$QUALITY_PREFILL)"
if [ "$RESIDENCY" = gpu ]; then
  echo "  residency=gpu: 2-bit base GPU-RESIDENT (BASE_CACHE_GB=0), sharded across ${TP} rank(s); no host pack/arena. FORCE_RESIDENT=${FORCE_RESIDENT} (1 = bypass the boot-guard VRAM-budget refusal on >=48 GiB cards). Watch for: 'moe_w2 planes: ... GPU-RESIDENT' and ~37 GiB/card VRAM."
elif [ "$RESIDENCY" = exact ]; then
  echo "  residency=exact: checkpoint FP4 cache ${EXACT_GB} GiB/rank + arena ${ARENA_GB} GiB/rank + w8x pack ${STORE}->/packs; native FP8 MMA for prefill/decode, strict miss replay."
else
  echo "  residency=host: base-cache ${BASE_GB} GiB/rank + arena ${ARENA_GB} GiB/rank + pack ${STORE}->/packs."
fi
echo "image build (upstream v0.25.1 SHA): ${BUILD:-UNKNOWN - pre-observability image, REBUILD from current main}"
echo "healthy-boot markers:  docker logs -f $NAME 2>&1 | grep -E 'moe_w2|o_proj'"
echo "  1) 'moe_w2: env ... does NOTHING — did you mean ...' (only if you typoed a knob)"
echo "  2) 'sm_89 Triton emulation ready on <GPU> ... self-test worst_rel=...'"
echo "  2b) 'sm_89 NATIVE decode cubin ACTIVE for (K, N)=...' after two per-shape parity lines (absent cubin / SM89_NATIVE=0 logs 'decode stays on the Triton emulation' instead - emulation-only, still correct)"
echo "  3) 'moe_w2 planes: ... -> GPU-RESIDENT' or '-> PINNED HOST RAM'"
echo "  4) 'DeepSeek V4 o_proj: using native SM89 block-scaled FP8 grouped matmul'"
if [ "$QUALITY_PREFILL" = 1 ]; then
  echo "quality-prefill boot markers (in addition to the above):"
  echo "  1) 'moe_w2 delta tier: auto-sizing deferred until after KV-cache allocation' (w8, at build)"
  echo "  5) 'moe_w2 delta tier AUTO: 256 slots x 6.0 MiB (1.50 GiB pool; ...)' (post-KV pool alloc)"
  echo "  6) 'moe_w2_cubit: sm_89 FP8 POOL parity OK ... through pool slot 255' (deferred gate)"
  echo "  7) first prefill: 'moe_w2 prefill-FP4/FP8 ensure mode: eager chunk working sets fetched...'"
fi
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
# NETWORK=host is the production default on this dedicated host. Pass
# NETWORK=none explicitly when a first quantizing boot must be isolated.
#
# TP2 ON A SMALL HOST (host residency): two pinned arenas + ~26 GiB /dev/shm for
# inter-worker tensor passing did not fit 38 GiB - the first TP2 attempt swap-stormed
# the host into a hard hang (even sshd stopped). A later base-cache TP2 run OOM-killed
# a worker at the 30 GB container cap (OOMKilled=true, 2026-07-19). Hence: TP1 keeps
# shm at 8g and skips --ipc host; host-residency TP2 needs more host RAM first. TP>1
# custom all-reduce is ON by default now (CUSTOM_ALL_REDUCE=1); it needs the P2P
# driver patch linked in the header - without that patch set CUSTOM_ALL_REDUCE=0.
#
# EXPERT RESIDENCY. The quantization cache is specific to (TP size, residency).
#   host: the 2-bit base stays in pinned host RAM. It uses an on-disk pack
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
# MEM_GB HARD CAP: belt-and-suspenders against a failed load. This host's
# container cap is 428 GiB; the launcher permits only another 2 GiB of swap.
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
#     MTP_TOKENS=1         (speculative decoding; measured +40% decode and +30% concurrency)
#     PREFIX_CACHING=1     (reuse repeated prompt KV; enabled by default)
#     SCALE_REFIT=1        (same-size W2 conversion with lower block SSE)
#   Override any of them via env to test smaller-first smoke runs (e.g.
#   MAXLEN=8192 UTIL=0.94 ./docker/serve_sm89_ds4.sh).
#
# STREAMED QUALITY PREFILL (QUALITY_PREFILL=1): enables per-chunk ensure-mode
# FP8-delta prefill. Each prefill chunk's routed experts are synchronously
# promoted from the pinned host arena into a deferred GPU pool (FP4-storage
# slots, 6 MiB each @ TP2) before the descriptor-kernel GEMM reads slot_table.
# Key parameters (piece-5 arithmetic):
#   BATCHED_TOKENS=4096  chunk size; MTP must be OFF (>1024 OOMs with MTP on;
#                        quality mode is H2D-bound anyway)
#   UTIL=0.90            post-KV free VRAM ≈ total×(1−UTIL) ≈ 4.8 GiB/card;
#                        must cover pool 1.5 GiB + 3 GiB workspace reserve —
#                        at UTIL=0.98 only ~1 GiB remains and the pool clips to ~0
#   FP8_DELTA_GB=1.5     256 × 6 MiB FP4-store slots = one full layer union @ TP2
#                        (deferred: allocated post-KV by finalize_auto, never at build)
#   SCALE_REFIT=0        REQUIRED: fp4-storage slots hold checkpoint e2m1 nibbles
#                        verbatim; a refit-decremented serving scale would need
#                        u = 2×e2m1 (off the e2m1 grid) — build enforces this
#   PLANES_SUBDIR=planes-qp  separate cache dir: SCALE_REFIT=0 changes plane
#                        bytes; sharing a dir with refit=1 would wipe that cache
# Cost: ~1.5-2.5 min per 131K prefill (43 MoE layers × 256 × 6 MiB ≈ 64.5 GiB
#   worst-case H2D/chunk ÷ ~20 GB/s pinned bandwidth ≈ 3.2 s + GEMMs);
#   131K/4096 = 32 chunks. fp4-store halves H2D vs fp8-store and keeps the
#   pinned host arena at ~65 GiB/rank (~129 GiB total TP2 — comfortable under
#   the 428 GiB cap; fp8-store's ~129 GiB/rank ≈ 258 GiB total fits too but
#   doubles the per-chunk H2D time — fp4-store is the perf choice, not just RAM).
# Two-boot cache workaround: OBSOLETE. With deferred pool allocation the build
#   allocates no GPU pool, so a single boot builds (or cache-loads) the planes
#   and then sizes the pool post-KV. Any prior two-boot operational notes are
#   superseded.
#
# IMAGE IDENTITY + TRIAGE: the start banner prints the upstream v0.25.1 SHA
# the patches/ set was diffed against (/opt/moet-checks/SOURCE.txt;
# 'UNKNOWN' = stale image, rebuild). 'Failed core proc(s): {}' in a crash
# means NO Python exception
# existed - check dmesg for the OOM-killer or the faulthandler stacks in
# docker logs (PYTHONFAULTHANDLER=1 is baked into the image).
# ============================================================================
