# The v0.24.0 port

The project targets **official vLLM v0.24.0**, which ships DeepSeek‑V4 + SM120 natively
(`vllm/models/deepseek_v4/`, FlashInfer SM120 sparse‑MLA, GLM‑5.x `GlmMoeDsaForCausalLM`).
Our overlay is a **3.7k‑line patch**: the 2‑bit expert planes, the FP4 delta cache, the
confidence gate, the cubit dispatch — plus the SM120 fixes below.

## Apply

```bash
git clone --branch v0.24.0 https://github.com/vllm-project/vllm && cd vllm
git apply /path/to/vLLM-Moet/patch/vllm-moet-v0.24.0.patch

# python-only overlay: reuse the official precompiled wheel for the C/CUDA artifacts
VLLM_USE_PRECOMPILED=1 pip install -e . --no-deps --no-build-isolation
```

Environment pins that go with the patch (both required on SM120):

| dep | version | why |
|---|---|---|
| DeepGEMM | nv‑dev **`a6b593d2`** (build from source, ~2 min) | the release pin `891d57b4` has no family‑120 host paths — "Unknown SF transformation" (linear), `t.dim()==N` (o_proj einsum), "Unsupported architecture" (indexer paged‑MQA metadata). Same as vLLM issues #47130/#47436; vLLM main already moved its pin. |
| flashinfer‑python | **0.6.14** (+ `flashinfer-jit-cache==0.6.14+cu130`) | the official 0.6.12 pin predates the kwargs (`swa_topk_lens`, `extra_sparse_*`) that v0.24's SM120 DS4 attention passes to `trtllm_batch_decode_sparse_mla_dsv4`. |

## SM120 fixes carried in the patch (base was broken-as-released)

1. `sparse_attn_indexer.py` — don't select `cooperative_topk` on SM12x (thread‑block **cluster
   launch** is SM90/SM100‑only; consumer Blackwell rejects it with "invalid argument").
2. `models/deepseek_v4/nvidia/ops/o_proj.py` — SM12x einsum recipe = SM90‑style
   `(1,128,128)` with **raw row‑major f32 block scales** (matches DeepGEMM nv‑dev's own SM120
   test convention; the SM100 packed/TMA‑aligned int32 layout produces NaN).
3. `fp8_utils.py` — skip the SM100 weight‑scale pre‑packing for the `is_bmm` (einsum) weights
   on family 120.
4. `capture_error_mode="thread_local"` on **all four** CUDA‑graph capture paths
   (`compilation/cuda_graph.py`, `compilation/breakable_cudagraph.py`,
   `v1/worker/gpu/cudagraph_utils.py`, `v1/worker/gpu_ubatch_wrapper.py`) — the FP4 delta
   cache promotes experts from a background thread; without thread_local its side‑stream work
   invalidates capture (`CUDA_ERROR_STREAM_CAPTURE_INVALIDATED`).

## Our hooks

- `mxfp4.py` (`Mxfp4MoEMethod`) — FP4‑checkpoint path (DeepSeek‑V4‑Flash): host‑stage experts
  at `create_weights`, build 2‑bit planes at `process_weights_after_loading`, `moe_w2_forward`
  in `apply`.
- `fp8.py` (`Fp8MoEMethod`) — FP8 block‑quant checkpoint path (DS4‑Flash‑Base,
  **GLM‑5.2‑FP8**): same three hooks; the loader re‑quantizes fp8+f32‑block‑128 to the
  sign‑symmetric 2‑bit codebook at load (`build_layer_planes_fp8`, float64 math, golden‑tested
  against the GLM‑5.2 sweep reference).
- `moe_w2_*` utils are shape‑generic now: layer cutoff from `num_hidden_layers` (43 DS4 / 78
  GLM‑5.2), cubins probed for K∈{6144,4096,2048,1024,512}, workspaces sized from the model's
  hidden size (GLM‑5.x H=6144 supported; `kernels/` ships the K=6144 family).
- `modelopt.py` (`ModelOptNvFp4FusedMoE`) — NVFP4 checkpoint path (**GLM‑5.2‑NVFP4**): same
  three hooks; the loader dequantizes modelopt NVFP4 (e2m1 codes + f8e4m3 block‑16 scales +
  per‑tensor `weight_scale_2`) to f64 — exact, all three factors representable — and
  re‑quantizes to the sign‑symmetric codebook (`nvfp4_to_codes_scales`; the UE8M0 block‑32
  output scales absorb `scale_2`). Golden‑tested EXACT on real checkpoint shards; forward
  op‑validated through the K=6144/K=2048 cubins on real weights.
- **BASE cache** (`VLLM_MOE_W2_BASE_CACHE_GB=N`, the "159B on one 5090" path): the packed
  2‑bit base planes (codes + UE8M0 scales, four sections per expert slot) live in pinned host
  RAM; the GPU pool caches hot experts through the same slot‑table/manager/eviction machinery
  as the delta tier. Decode misses zero the pair's contribution, bump an in‑graph miss
  counter, and the runner fetches all missing routed experts synchronously (batched pinned
  H2D, 51.6 GiB/s measured) and replays the step's graph once — replay bit‑identical to a
  resident forward (unit‑tested, `internal` test_base_cache). Prefill prefetches per layer
  via `ensure_resident`. TP OR‑reduces the miss decision; PP unsupported; mutually exclusive
  with the FP4 delta tier. Measured on 1× RTX 5090: 22.9 GiB GPU / 72.7 GiB pinned host,
  ~32 tok/s steady (96% pool hit at 19% coverage), 10–21 tok/s under working‑set shift.
- **Deterministic unpermute**: the MoE output scatter used atomic `index_add_`, so identical
  runs wobbled (~1.6e‑2 on prefill) and greedy decode was not reproducible (surfaced by the
  PP determinism investigation; never PP‑specific). Valid `sorted_ids` form a permutation of
  `token*top_k+j`, so a bijective `index_copy_` + fixed‑order `sum(dim=1)` replaces it —
  6/6 bit‑identical repeats on prefill and decode, capture‑safe.
- MTP under **pipeline parallelism** (ported from the fork, inert off‑PP/off‑spec): draft‑token
  broadcast to rank 0 under async scheduling, `output_token_ids` trim on all ranks, and the
  drafter `embed_tokens` share across PP ranks (the NVFP4/DS4 MTP head ships no embedding of
  its own; upstream's share is gated to `pp_world_size == 1`). Validated on DS4‑Flash PP4
  (4× RTX 5090): acceptance to 2.81, 184 vs 93 tok/s (~2× MTP speedup).
- **NVFP4 KV cache** (`--kv-cache-dtype nvfp4`) for the SM120 sparse‑MLA path — a packed
  **352 B/token** layout (512× E2M1 + 32× E4M3 block‑16 scales at a fixed 2⁻⁶ global scale +
  64× FP8 rope) replacing the 656 B `fp8_ds_mla` layout, **1.86× less KV/token**. The write
  kernel is a standalone SM120 torch extension (`csrc/nvfp4_ds_mla/`); FlashInfer's sparse‑MLA
  JIT sources are patched with `ModelType::GLM_NSA_NVFP4` (`tools/nvfp4_flashinfer_sm120/`) so
  the packed bulk expands in place before QK and the FP8‑MMA pipeline is unchanged. Live on
  GLM‑5.2 TP4 (128K ctx): KV pool **+38%** (415K → 571K tokens), decode parity (104 tok/s —
  sparse reads only top‑2048), needle PASS to 126K, arithmetic + coherence intact. (Follow‑up:
  move the write kernel into `vllm._C`; FlashInfer 0.6.14 ships an AOT `sparse_mla_sm120.so`
  that must be removed so the patched JIT sources rebuild.)
- **Deterministic MoE unpermute** — the routed‑expert scatter‑add uses a bijective
  `index_copy` instead of `index_add`, removing the atomic‑accumulation non‑determinism in
  free‑running decode (matters under PP where physical KV‑block assignment varies run to run).
- **BASE cache (inverted delta)** — an opt‑in mirror of the delta tier: the 2‑bit base lives in
  host RAM and a GPU pool caches the hot experts, so a model whose 2‑bit base does not fit VRAM
  (e.g. GLM‑5.2 on 2 cards) can still serve at cache‑hit speed. GLM routing is concentrated
  enough (≈89% of token→expert routings served from ~20% of experts) to make this practical.
- **AFRAG prefill** — fragment‑major activation repack (single‑pass Triton into dedicated
  buffers) so each QMMA A‑fragment loads in one `LDG.128`; ~1.3× on the prefill GEMM,
  bit‑identical to the mc4 path, default‑on where the `mc4afrag` cubins ship.

Everything stays **opt‑in** (`VLLM_MOE_W2=1` etc.); with the knobs off the only behavioural
delta vs stock v0.24.0 are the SM120 fixes above.

The **confidence gate is fully wired** (2026‑07‑08 evening): `VLLM_MOE_W2_GATE=1` arms the
FP4 re‑forward — low‑confidence decode steps force‑promote their routed experts to FP4 and
replay the step once (inline on TP/single‑GPU incl. MTP verify steps; a worker‑driven
full‑pipeline replay under PP, pure‑decode only). τ is runtime‑tunable via
`VLLM_MOE_W2_GATE_TAU(_FILE)`. Live‑validated on the official checkpoint (1× PRO 6000, MTP
k=2, graphs): fires/promotes/replays per τ, coherent output; arming the gate costs ~10%
single‑stream (per‑step confidence sync) and the replays at τ=0.60 were throughput‑neutral
on top of that (FP4 re‑decides lift MTP acceptance enough to pay for themselves); τ=0.75
costs ~5% more.

## Benchmarks (2026‑07‑08)

Official FP4 checkpoint, `VLLM_MOE_W2=1`, FP4 delta 1 GiB, MTP k=2, cudagraphs
FULL_AND_PIECEWISE, fp8 KV, block 256, `max_num_seqs` 4, mnbt 1024; PRO 6000 runs at
24576 ctx, 5090 runs at 16384 ctx. Tools: `tools/bench_tok.py` (single‑stream decode,
512 tok, median of 5) and a unique‑prefix prefill probe (8k tokens, median of ≥3; unique
prefixes defeat the prefix cache). MTP acceptance ~2.6 tok/step in every config.

| config | decode | prefill 8k | decode conc‑3 (aggregate) |
|---|---:|---:|---:|
| 1× RTX PRO 6000 | **161.2 tok/s** | **4 847 tok/s** | ~289 tok/s |
| 2× RTX PRO 6000 TP2 | **209.6 tok/s** | **5 791 tok/s** | ~380 tok/s |
| 4× RTX 5090 TP4 | **214.4 tok/s** | **5 561 tok/s** | ~430 tok/s |

**2026‑07‑09 — AFRAG prefill kernels ship and default on** (fragment‑major activations;
bit‑identical outputs, 1.30×/1.27× on the K=4096/K=2048 prefill GEMMs). Same 8k‑unique‑prompt
probe, median of 5: 1× PRO 6000 **4 777 → 5 340 tok/s (+11.8%)**; 4× 5090 TP4 (median of 3)
5 987 → 6 101 tok/s (+1.9% — the 1024‑token chunks shard per‑rank GEMM work too thin for the
full kernel win). TP2 has **not** been re‑measured with AFRAG yet — the tables carry its
2026‑07‑08 pre‑AFRAG figure (5 791), which is conservative. Opt out with
`VLLM_MOE_W2_AFRAG=0`.

Prefill rides upstream's FlashInfer SM120 sparse‑MLA path — which also makes a custom cubit
MLA‑prefill kernel unnecessary on this base.

**Batch scaling** (same knobs but `--max-num-seqs 32 --max-num-batched-tokens 2048`;
N identical‑length greedy requests, 384 tok each, aggregate decode tok/s, median of 5):

| concurrency | 1 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|
| 1× RTX PRO 6000 | 156 | 290 | 493 | 659 | **933** |
| 4× RTX 5090 (TP4) | 198 | 460 | 762 | 1 006 | **1 560** |

6–8× aggregate from batch 1→32 — the 2‑bit expert reads amortize well across the batch
(decode stays HBM‑bound; the per‑step expert working set grows sublinearly with batch).
At 32 streams each request still gets ~29 tok/s (PRO 6000) / ~49 tok/s (TP4).

**Long context (512K) on one 96 GB card** — validated live (`tools/needle_probe.py`, unique
secret embedded in filler, greedy): PASS at **102 238 / 256 294 / 453 286 prompt tokens**
(depth 0.1) and at 453K with the needle mid‑context (depth 0.5); cold TTFT 27 s / 64 s /
~2 min. Server config for the 512K window on 1× PRO 6000: `--max-model-len 524288
--gpu-memory-utilization 0.97 --max-num-batched-tokens 2048 --max-num-seqs 1` with
`VLLM_MOE_W2=1 VLLM_MOE_W2_DELTA_GB=0` (the FP4 delta pool trades against KV headroom at
extreme context; with delta 1 GiB use ≤256K). The KV fit comes from DS4's compressed KV +
upstream's FP8 Lightning‑Indexer cache; vLLM reports 947K cached tokens in this config.

**Delta pool auto‑sizing.** `VLLM_MOE_W2_DELTA_GB=auto` resolves the delta‑vs‑KV trade
automatically: the pool allocation is deferred until after the KV cache is allocated
(and before cudagraph capture — the graphs bake the pool pointer), then sized as
`free VRAM − VLLM_MOE_W2_DELTA_RESERVE_GB` (default 3, capture/workspace headroom),
optionally capped by `VLLM_MOE_W2_DELTA_MAX_GB`. At extreme context it lands at 0 slots
(pure 2‑bit, pinned host store released — the manual `DELTA_GB=0` rule, without the manual
step); at 24K ctx / util 0.95 it recovers a ~1.6 GiB pool (133 slots) and benches at
166 tok/s single‑stream (1× PRO 6000, MTP k=2, graphs).
