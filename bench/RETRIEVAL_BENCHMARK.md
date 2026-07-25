# Mid-Context Retrieval Benchmark (needle_sweep)

A chat-only mid-context needle probe for measuring long-context retrieval quality
in vLLM-served LLMs. Tests whether the model can retrieve a specific "needle"
(a unique access code) placed at 50% depth in realistic filler text, at increasing
context lengths.

## Quick start

```bash
# Smoke test (2K + 16K, ~30s)
python3 bench/runner/probes.py --probe needle_sweep --lengths 2048,16384 --runs 1

# Full sweep (2K through 131K)
python3 bench/runner/probes.py --probe needle_sweep --lengths 2048,4096,8192,16384,32768,65536,131072 --runs 1

# Via the suite YAML
python3 -m bench.runner.bench --suite bench/suites/needle_sweep.yaml
```

## What it measures

For each context length L:
- **correct**: did the model return the exact needle code? (substring match)
- **maxp**: the answer-token top-logprob max probability (confidence — high=exact, ~0.5=guessing)
- **deg**: degeneration flag (garbled/repetitive output)
- **onset L***: the first length where retrieval fails

## Config

- `--lengths`: comma-separated target token counts (default: 2K-131K)
- `--runs`: repetitions per length (default: 1, greedy temp=0)
- `--out`: output JSON path
- `--filler-files`: override the realistic filler source files
- Endpoint: `http://localhost:8001/v1/chat/completions` (configurable)

The probe uses **realistic filler** (concatenated repo docs, not random words) to
isolate context length as the sole variable. The needle is a unique code like
`PELICAN-7356` placed at 50% depth. The instruction asks for the code only.

## Interpretation

- **PASS (maxp > 0.9)**: exact retrieval — the model found and decoded the needle precisely.
- **FAIL (maxp 0.5-0.8)**: approximate retrieval — the model found the needle region
  but lost exact digits (precision degradation).
- **FAIL + deg=True**: garbled output — the model lost the needle entirely.

Gradual maxp decay across lengths → compounding quantization noise.
Sudden cliff → threshold/config boundary (e.g., sparse attention engagement).
