# Ada (SM89) port

This document describes the current Ada path for DeepSeek-V4-Flash.
The active image uses official vLLM v0.25.1 at commit `752a3a504`.
The `overlay/` directory is the source of the vLLM changes.
The `patches/` directory contains generated per-file patches.
There is no vLLM fork branch.

## Scope

Ada GPUs support FP8 E4M3 tensor-core operations. They do not support
QMMA, QMMA.SF, FP4, or NVFP4 tensor-core operations.

The Ada port has two separate compute paths:

1. The MoE path reads 2-bit expert planes. It decodes the planes to BF16
   values in registers. It then uses BF16 tensor-core operations.
2. The attention output projection reads FP8 activations and FP8 weights.
   It uses native FP8 tensor-core operations on SM89.

Do not confuse these paths. The MoE kernel is an emulation of the SM120
QMMA path. The output-projection kernel is a native Ada FP8 path.

## Components

| Component | Source |
|---|---|
| 2-bit MoE kernel | `overlay/vllm/vllm/model_executor/layers/quantization/utils/moe_w2_sm89.py` |
| Published MoE mirror | `kernels/triton/moe_w2_sm89.py` |
| Native FP8 output-projection kernel | `overlay/vllm/vllm/models/deepseek_v4/nvidia/ops/triton_ada_fp8_bmm.py` |
| Output-projection dispatch | `overlay/vllm/vllm/models/deepseek_v4/nvidia/ops/o_proj.py` |
| Sparse-MLA Triton port | `overlay/vllm/vllm/v1/attention/ops/triton_sparse_mla_dsv4.py` |
| Ada launcher | `docker/serve_sm89_ds4.sh` |
| Ada image | `Dockerfile.sm89-v0251` |
| Native FP8 tests | `overlay/vllm/tests/kernels/test_triton_ada_fp8_bmm.py` |

## 2-bit MoE path

The MoE kernel uses the same packed expert planes as the SM120 kernel.
Each plane contains 2-bit codes and block-32 UE8M0 scales.
The kernel maps the codes to `{-4, -1, 1, 4}`.
The kernel decodes the values in registers.
The kernel uses BF16 matrix operations and FP32 accumulation.

The kernel supports decode and prefill. It registers the `w2` and
`w2mc4` operation keys. Ada does not support the FP4 delta kernels.
Set `VLLM_MOE_W2_DELTA_GB=0` on Ada.

## Native FP8 output projection

DeepSeek-V4 applies inverse RoPE before the output projection.
The fused quantizer produces these tensors:

- FP8 activations with shape `[M, H, K]`.
- FP32 activation scales with shape `[M, H, K / 128]`.
- FP8 weights with storage shape `[H * N, K]`.
- FP32 or UE8M0 weight scales with shape `[H * N / 128, K / 128]`.

The activation tensor can have non-contiguous strides.
The native kernel reads those strides directly.
It computes `bhr,hdr->bhd` in one grouped launch.
It applies each scale after its block-128 FP8 dot product.
It accumulates the result in FP32.
It writes a BF16 output.

The kernel reads UE8M0 weight scales through a zero-copy `uint8` view.
It decodes each exponent in registers.
It does not materialize FP32 weights or FP32 scale tensors.

The dispatch uses this kernel only when the device capability is SM89.
Other architectures continue to use the existing DeepGEMM path.
The operation is registered as `torch.ops.vllm.ada_fp8_grouped_mm`.
The registration supports CUDA graph capture and `torch.compile`.

Look for this log entry:

```text
DeepSeek V4 o_proj: using native SM89 block-scaled FP8 grouped matmul
```

If this entry is absent on SM89, verify the image source stamp.

## Sparse-MLA KV cache

The launcher passes `--kv-cache-dtype fp8`.
The sparse SM120 call surface accepts this value on Ada.
The MLA setup then converts the value to `fp8_ds_mla`.
This conversion occurs in `_canonicalize_sparse_mla_kv_cache_dtype`.
Thus the SWA cache uses the 576-byte alignment rule.

The logical packed row uses 584 bytes for each token:

- 448 bytes for FP8 NoPE data.
- 128 bytes for BF16 RoPE data.
- 8 bytes for UE8M0 scales and padding.

K and V use the same 512-dimension MLA cache row.
A 262,144-token sequence uses 146 MiB before allocator padding.
Four such sequences use 584 MiB.

The allocator can add padding between pages.
The resulting cache view can be non-contiguous.
The sparse-MLA kernel reads `kv_cache.stride(0)` as the page stride.
Do not replace this stride with `pbs * 584`.

A negative KV-cache budget does not mean that one sequence needs many
gigabytes of KV memory. The memory profiler first subtracts weights,
graphs, activations, and runtime allocations from the configured GPU
budget. A negative value means that these allocations used the complete
budget before vLLM created the KV-cache blocks.

## TP2 on a small host

Use GPU residency when the system has two 48 GiB Ada cards and no more
than approximately 30 GiB of available host RAM.

```bash
RESIDENCY=gpu TP=2 GPUS='"device=0,1"' \
  NETWORK=host RESTART=unless-stopped \
  ./docker/serve_sm89_ds4.sh
```

GPU residency shards the approximately 73 GiB 2-bit base across both
cards. Each card holds approximately 36 GiB of expert planes. Dense
weights and other model data increase the measured load to approximately
43 GiB on each card.

Do not use TP2 host residency on a 30 GiB host. Each rank needs a pinned
arena. TP2 also needs a large shared-memory allocation. This configuration
caused host memory exhaustion during field tests.

The launcher uses these memory controls for TP2 GPU residency:

| Control | Default | Purpose |
|---|---:|---|
| `UTIL` | `0.98` | Increase the vLLM GPU memory budget. |
| `BATCHED_TOKENS` | `1024` | Limit the prefill workspace. |
| `NUM_SEQS` | `4` | Limit scheduler concurrency. This does not reserve four full contexts. |
| `CUDAGRAPH_SIZES` | `1,2,4,8` | Limit captured graph shapes. |
| `MTP_TOKENS` | `0` | Disable MTP for the correctness baseline. |
| `PREFIX_CACHING` | `0` | Disable prefix reuse for the correctness baseline. |
| `SCALE_REFIT` | `1` | Decrease a W2 block scale only when exact SSE decreases. |

The measured live cache had 286,615 token slots. Thus, it can hold one full
262,144-token request. It cannot hold four requests at that length. The
`NUM_SEQS=4` setting permits four shorter requests that share the token slots.

Smaller CUDA graph lists reduce captured buffers and workspaces.
They do not reduce the model weight allocation.
Therefore, use this control as a secondary memory control.

The quantization cache depends on the TP size and the residency.
A TP1 cache is not valid for TP2.
A host-resident cache is not valid for GPU residency.
Select both values before the first quantization run.
Keep `READY_TIMEOUT_S=1800` for a new quantization run.

The cache also depends on the quantizer settings. A change to `SCALE_REFIT`
must cause a cache miss. The cache metadata includes this setting. The host
pack probe checks the checkpoint identity before it skips checkpoint tensors.

## Long-agent correctness test

The FP4 recovery quality result does not apply to Ada. Ada uses the base-only
W2 path. Start with MTP and prefix caching disabled. This configuration makes
the first comparison easy to interpret.

```bash
RESIDENCY=gpu TP=2 GPUS='"device=0,1"' \
  MAXLEN=262144 NUM_SEQS=4 MTP_TOKENS=0 PREFIX_CACHING=0 SCALE_REFIT=1 \
  NETWORK=host ./docker/serve_sm89_ds4.sh
```

Use one saved long coding-agent prompt. First, compare `SCALE_REFIT=1` with
`SCALE_REFIT=0`. Then enable `PREFIX_CACHING=1` and repeat the same prompt.
Enable `MTP_TOKENS=1` only after the answer is correct. Keep MTP enabled only
when the generated-token rate increases.

The W2 layer selection excludes the MTP drafter. Thus, the drafter and target
do not use the same expert path. MTP acceptance is not a direct W2 quality
measurement.

## Build and start

Build the current Ada image from the repository root:

```bash
docker build -f Dockerfile.sm89-v0251 -t vllm-moet-sm89:v0251 .
```

Start the service with `docker/serve_sm89_ds4.sh`.
The script removes an old container with the selected name.
Do not run the script while another operation owns that container.

Check the image source stamp after the container starts:

```bash
docker exec moet cat /opt/moet-checks/SOURCE.txt
```

## Validation on RTX 4090 D

The native FP8 output-projection path was tested on an RTX 4090 D with
Triton 3.6.0.

- The CUDA test suite passed: `11 passed`.
- The suite covered FP32 and UE8M0 scales.
- The suite covered `M=1,2,8,17,65`.
- CUDA graph capture and replay passed.
- An end-to-end projection smoke test had a maximum BF16 difference of
  `0.015625` from the dequantized reference.
- SM89 compilation emitted native
  `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` instructions.

The production-shape microbenchmark used `H=4`, `K=4096`, `N=1024`, and
`M=1,2,4,8`.

| Path | Time |
|---|---:|
| Previous dequantized fallback | 0.443 to 0.465 ms |
| Native SM89 FP8 kernel | 0.0351 to 0.0361 ms |

This result is a kernel microbenchmark. It is not a server throughput
result. Measure server throughput after you rebuild and start the image.
The kernel does not change MTP acceptance.

## Startup checks

Check these log entries after a new image starts:

1. Check for `sm_89 Triton emulation ready`.
2. Check for `moe_w2 STREAM-BUILD armed` during a new quantization.
3. Check for `moe_w2 planes:` and the selected residency.
4. Check for `native SM89 block-scaled FP8 grouped matmul`.
5. Check that `Available KV cache memory` is positive.

Use these commands for a startup failure:

```bash
docker logs moet 2>&1 | grep -B2 -A30 -E \
  'EngineCore.*(Error|Traceback|CRITICAL)|moe_w2|o_proj'
docker inspect moet --format '{{.State.ExitCode}} {{.State.OOMKilled}}'
```

If `OOMKilled` is true, inspect the host memory limit.
If the KV-cache budget is negative, reduce captured graph sizes and
workspace limits. Increase `UTIL` only when the card has sufficient free
runtime memory.

If TP2 throughput is low, check the PCIe link width for each GPU:

```bash
nvidia-smi topo -m
nvidia-smi --query-gpu=index,pcie.link.gen.current,pcie.link.width.current \
  --format=csv
```

A PCIe x1 link can limit TP2 communication.

## Source update procedure

Edit vLLM files only under `overlay/vllm/`.
Then generate and verify the per-file patches:

```bash
python3 tools/gen_patches.py
python3 tools/gen_patches.py --verify
python3 tools/gen_patches.py --check
python3 tools/check_patch_files.py
```

Commit the overlay file and its generated patch in the same commit.
Do not edit a generated patch directly.
