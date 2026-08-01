# `tools/` — standalone harnesses

Scripts here run **outside** pytest, against the serving image, so they can
exercise real kernels without the vLLM test-suite scaffolding. Convention:
`test_*.py` = a correctness harness with a pass/fail verdict; everything else is
a probe, benchmark, or converter.

**Why these exist.** Deploying this stack costs 6–10 minutes per boot and each
long-context probe costs 30–190 s. Every harness below was written to answer a
question that would otherwise need a server round trip. Extend them before
reaching for another boot. Background:
[`docs/dsv4-sm89-longcontext/README.md`](../docs/dsv4-sm89-longcontext/README.md).

## Long-context / DSv4 sm_89 investigation

| harness | needs | runtime | what it proves |
|---------|-------|---------|----------------|
| `test_nope_needle_recall.py` | CPU | ~1 min | Quantifies what the packed KV encoding costs exact-value recall. E4M3 **2.48e-2** vs INT8 **8.0e-3** readback error, flat in attended-key count; needle never loses argmax. |
| `test_compressor_vs_checkpoint_ref.py` | CPU + `/kernels` mount | seconds | Compressor semantics vs an **independent transcription of the checkpoint's own `Compressor`** — the in-repo reference and the kernel were written together, so a shared misreading would pass. Agrees at ~2.6e-3 (bf16 floor). |
| `test_compressor_state_capacity.py` | CPU | seconds | The state cache holds a **whole prefill chunk + the window**, so `save_partial_states` cannot overwrite state a later compress output still needs. Carries a negative control. |
| `test_indexer_topk_selection.py` | 1 GPU | ~1 min | The Lightning-Indexer top-k selects **by score**, not positionally: with every high score beyond index 512, `persistent_topk` returns 512/512 correct. |
| `needle_digits_probe.py` | server | 30–190 s/point | Needle probe reporting `word_present` / `digits_present` separately plus top-N logprobs. **Use this, not `bench/runner/probes.py --probe needle_sweep`**, whose single-bit verdict caused a multi-hour misdiagnosis. |
| `vllm_dsv4_chat_benchmark.py` | server | varies | Throughput / prefill / capacity probe (`single`, `concurrent`, `prefill`). Emits `gpu_telemetry` on failure too and survives EngineCore death. |
| `needle_probe.py` | server | ~1 min | Quick needle with RANDOM-WORD filler. **Does not reproduce the failure** — realistic filler is required. Kept for contrast. |
| `needle_full.py` | server | varies | Needle + reasoning trace. |

### Running the containerised harnesses

CPU-only (safe while the server is up — no GPU contention):

```bash
docker run --rm --entrypoint python3 \
  -v $PWD/tools/test_nope_needle_recall.py:/opt/t.py:ro \
  vllm-moet-sm89:v0251 /opt/t.py
```

Needs the repo test tree mounted (for the in-repo reference):

```bash
docker run --rm --entrypoint python3 \
  -v $PWD/tools/test_compressor_vs_checkpoint_ref.py:/opt/t.py:ro \
  -v $PWD/vllm/tests/kernels:/kernels:ro \
  vllm-moet-sm89:v0251 /opt/t.py
```

Single GPU — **check free VRAM first**; a loaded server leaves only ~1 GiB:

```bash
docker run --rm --gpus '"device=0"' --entrypoint python3 \
  -v $PWD/tools/test_indexer_topk_selection.py:/opt/t.py:ro \
  vllm-moet-sm89:v0251 /opt/t.py
```

Repo pytest suites, with an overlay file layered over the installed package:

```bash
docker run --rm --gpus '"device=0"' \
  -v $PWD/overlay/vllm/vllm/models/deepseek_v4/common/ops/cache_utils.py:/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/common/ops/cache_utils.py:ro \
  -v $PWD/vllm/tests/kernels:/base:ro \
  -v $PWD/overlay/vllm/tests/kernels/test_compressor_kv_cache.py:/ovl.py:ro \
  --entrypoint bash vllm-moet-sm89:v0251 -c '
    mkdir -p /w/kernels && cp -r /base/. /w/kernels/ && cp /ovl.py /w/kernels/test_compressor_kv_cache.py
    touch /w/kernels/__init__.py
    cd /w && python3 -m pytest -q -p no:cacheprovider kernels/test_compressor_kv_cache.py'
```

Expected on sm_89: **30 passed, 6 skipped, 0 failed**. The 6 skips are the
datacenter-Blackwell-only MXFP4 indexer cache.

## Two traps that cost real time

1. **`torch.ops._C` is registered lazily on `import vllm`.** Probing it before
   that import reports every custom op as missing — a silent false negative.
   Import vLLM first, then check.
2. **`cooperative_topk` has no sm_89 kernel image** and *poisons the CUDA
   context* when called, so every later case in the same process fails with
   confusing errors. Serving gates it behind `has_device_capability(90)`;
   harnesses must skip it below sm_90 to match.

## Other harnesses

`test_moe_w2_mark_seen.py`, `test_moe_w2_route_sentinels.py`,
`test_moe_w2_pool_heat.py`, `test_bf12.py` — MoE exact-cache / route-sentinel
tests from the expert-path work (see `docs/ada-sm89-port.md`).
`bench_*`, `probe_*`, `repro_*`, `convert_*`, `scan_*` — benchmarks, quality
probes, minimal reproducers, and GGUF/safetensors converters.
`gen_patches.py` — the overlay→patches generator; run `--verify` and `--check`
before every commit that touches `overlay/` (see `AGENTS.md`).
