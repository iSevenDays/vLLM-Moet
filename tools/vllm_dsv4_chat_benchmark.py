"""Observable vLLM DeepSeek-V4 chat benchmark for this 2x4090D host.

The first generation request is always an excluded compile/cache warm-up.
Long prompts are sized through vLLM's own ``/tokenize`` endpoint, so reported
prompt-token counts use the exact live tokenizer/chat template.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import requests


SHORT_PROMPT = (
    "Explain how speculative decoding accelerates a mixture-of-experts model. "
    "Cover drafting, target verification, acceptance, expert-cache behavior, "
    "and hardware bottlenecks. Write at least 350 words."
)

# Realistic, deterministic filler behaves more like a long agent transcript
# than repeated one-token padding and exercises broad expert routing.
FILLER_UNIT = """
Repository review note: the request enters an asynchronous scheduler, obtains
KV-cache blocks, executes sparse attention, routes six experts per MoE layer,
and records cache hit, miss, replay, and latency counters. The implementation
must preserve numerical correctness across tensor-parallel ranks. Diagnostics
include memory ownership, communication backend, kernel selection, request
timings, and deterministic identifiers for every artifact.

def validate_runtime(configuration, observations):
    required = ["coherent_output", "bounded_memory", "native_kernel"]
    missing = [name for name in required if not observations.get(name)]
    if missing:
        raise RuntimeError({"missing": missing, "config": configuration})
    return {"status": "accepted", "evidence": observations}

Operationally, a cold first request may compile kernels and populate caches;
it must never be counted as steady-state throughput. Subsequent measurements
use the same protocol while preserving complete per-GPU telemetry. Long
contexts also need retrieval-quality checks, not merely successful allocation.
""".strip()

LONG_SUFFIX = (
    "\n\nBased on the preceding engineering record, explain the three most "
    "important performance risks and propose concrete mitigations."
)


def emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(int((len(ordered) - 1) * q), len(ordered) - 1)]


@dataclass
class GPUSampler:
    interval_s: float = 0.5
    samples: list[dict[str, float | int]] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    QUERY = (
        "index,memory.used,utilization.gpu,utilization.memory,power.draw,"
        "clocks.sm,clocks.mem,pcie.link.gen.current,pcie.link.width.current"
    )
    KEYS = (
        "index",
        "memory_used_mib",
        "gpu_util_pct",
        "memory_util_pct",
        "power_w",
        "sm_clock_mhz",
        "memory_clock_mhz",
        "pcie_gen",
        "pcie_width",
    )

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_s * 3))

    def _run(self) -> None:
        start = time.perf_counter()
        while not self._stop.is_set():
            try:
                proc = subprocess.run(
                    [
                        "nvidia-smi",
                        f"--query-gpu={self.QUERY}",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                now = time.perf_counter() - start
                for line in proc.stdout.splitlines():
                    vals = [item.strip() for item in line.split(",")]
                    if len(vals) != len(self.KEYS):
                        continue
                    row: dict[str, float | int] = {"elapsed_s": now}
                    for key, raw in zip(self.KEYS, vals):
                        row[key] = int(raw) if key == "index" else float(raw)
                    self.samples.append(row)
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            self._stop.wait(self.interval_s)

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {"sample_count": len(self.samples), "gpus": {}}
        indices = sorted({int(row["index"]) for row in self.samples})
        for index in indices:
            rows = [row for row in self.samples if row["index"] == index]
            gpu: dict[str, Any] = {}
            for key in self.KEYS[1:]:
                vals = [float(row[key]) for row in rows]
                gpu[key] = {
                    "mean": statistics.fmean(vals),
                    "p50": percentile(vals, 0.50),
                    "p95": percentile(vals, 0.95),
                    "max": max(vals),
                }
            out["gpus"][str(index)] = gpu
        return out


def post_json(url: str, payload: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    response = requests.post(url, json=payload, timeout=(15, timeout_s))
    response.raise_for_status()
    return response.json()


def tokenize_count(base: str, model: str, content: str, timeout_s: int) -> int:
    result = post_json(
        f"{base}/tokenize",
        {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "add_generation_prompt": True,
        },
        timeout_s,
    )
    return int(result["count"])


def size_long_prompt(
    base: str,
    model: str,
    target_tokens: int,
    stream_index: int,
    timeout_s: int,
    tolerance: int = 128,
) -> tuple[str, int, int]:
    """Proportionally size, then locally refine, a realistic chat prompt."""
    prefix = f"Long-context stream {stream_index}.\n\n"
    sample = prefix + FILLER_UNIT * 32 + LONG_SUFFIX
    sample_tokens = tokenize_count(base, model, sample, timeout_s)
    chars_per_token = len(sample) / max(sample_tokens, 1)
    target_chars = max(1, int(target_tokens * chars_per_token))

    cache: dict[int, int] = {}

    def materialize(n_chars: int) -> str:
        repeats = n_chars // len(FILLER_UNIT) + 1
        return prefix + (FILLER_UNIT * repeats)[:n_chars] + LONG_SUFFIX

    def count(n_chars: int) -> int:
        if n_chars not in cache:
            cache[n_chars] = tokenize_count(
                base, model, materialize(n_chars), timeout_s
            )
        return cache[n_chars]

    # Three proportional corrections usually land within 0.1%; the bounded
    # binary refinement avoids dozens of full-262K tokenize responses.
    for _ in range(3):
        observed = count(target_chars)
        if abs(observed - target_tokens) <= tolerance:
            return materialize(target_chars), observed, len(cache) + 1
        target_chars = max(1, int(target_chars * target_tokens / observed))

    observed = count(target_chars)
    if observed < target_tokens:
        lo, hi = target_chars, int(target_chars * 1.02) + 1024
        while count(hi) < target_tokens:
            lo, hi = hi, int(hi * 1.05) + 1024
    else:
        lo, hi = max(1, int(target_chars * 0.98) - 1024), target_chars
        while count(lo) > target_tokens and lo > 1:
            hi, lo = lo, max(1, int(lo * 0.95) - 1024)

    best_chars, best_tokens = lo, count(lo)
    for _ in range(8):
        mid = (lo + hi) // 2
        got = count(mid)
        if got <= target_tokens and got > best_tokens:
            best_chars, best_tokens = mid, got
        if abs(got - target_tokens) <= tolerance:
            return materialize(mid), got, len(cache) + 1
        if got < target_tokens:
            lo = mid + 1
        else:
            hi = mid - 1
    return materialize(best_chars), best_tokens, len(cache) + 1


def _delta_text(delta: dict[str, Any]) -> str:
    return str(delta.get("content") or delta.get("reasoning_content") or "")


def stream_chat(
    base: str,
    model: str,
    content: str,
    label: str,
    max_tokens: int,
    timeout_s: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "min_tokens": max_tokens,
        "ignore_eos": True,
        "reasoning_effort": "none",
        "include_reasoning": False,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.perf_counter_ns()
    first = last = None
    usage: dict[str, Any] = {}
    finish_reason = None
    fragments: list[str] = []
    events = 0
    with requests.post(
        f"{base}/v1/chat/completions",
        json=payload,
        stream=True,
        timeout=(15, timeout_s),
    ) as response:
        status = response.status_code
        response.raise_for_status()
        for raw in response.iter_lines(chunk_size=1, decode_unicode=True):
            if not raw:
                continue
            line = raw.removeprefix("data: ")
            if line == "[DONE]":
                continue
            event = json.loads(line)
            events += 1
            if event.get("usage"):
                usage = event["usage"]
            choices = event.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
            text = _delta_text(choice.get("delta") or {})
            if text:
                now = time.perf_counter_ns()
                first = now if first is None else first
                last = now
                fragments.append(text)
    completed = time.perf_counter_ns()
    text = "".join(fragments)
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    ttft_s = (first - started) / 1e9 if first is not None else None
    decode_s = (last - first) / 1e9 if first is not None and last is not None else 0.0
    elapsed_s = (completed - started) / 1e9
    words = text.split()
    trigrams = Counter(zip(words, words[1:], words[2:]))
    repeated = trigrams.most_common(1)[0] if trigrams else ((), 0)
    failures = []
    if status != 200:
        failures.append(f"http={status}")
    if completion_tokens != max_tokens:
        failures.append(f"completion_tokens={completion_tokens}, expected={max_tokens}")
    if finish_reason != "length":
        failures.append(f"finish_reason={finish_reason!r}, expected='length'")
    if not text.strip():
        failures.append("empty output")
    if completion_tokens >= 64 and len(set(words)) < 16:
        failures.append(f"unique_words={len(set(words))}, expected >=16")
    if repeated[1] >= 16:
        failures.append(f"repeated trigram {repeated[0]!r} x{repeated[1]}")
    result = {
        "event": "request_result",
        "label": label,
        "valid": not failures,
        "failures": failures,
        "status_code": status,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "ttft_s": ttft_s,
        "decode_s": decode_s,
        "elapsed_s": elapsed_s,
        "decode_tok_s": (
            (completion_tokens - 1) / decode_s
            if completion_tokens > 1 and decode_s > 0
            else None
        ),
        "e2e_tok_s": completion_tokens / elapsed_s,
        "stream_events": events,
        "output_words": len(words),
        "unique_words": len(set(words)),
        "most_repeated_trigram": list(repeated[0]),
        "most_repeated_trigram_count": repeated[1],
        "prompt_sha256": sha256_text(content),
        "text_sha256": sha256_text(text),
        "text_head": text[:240],
    }
    emit(result)
    return result


def mode_defaults(mode: str) -> tuple[int, int, int, int]:
    # warmups, measured runs, streams, target prompt tokens
    return {
        "single": (1, 5, 1, 0),
        "concurrent": (1, 3, 3, 0),
        "prefill": (1, 1, 1, 131072),
        # 262000 prompt + 128 generation = 262128, leaving 16 tokens beneath
        # the 262144 model limit while proving three almost-full live contexts.
        "capacity": (1, 1, 3, 262000),
    }[mode]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["single", "concurrent", "prefill", "capacity"])
    parser.add_argument("--base", default="http://127.0.0.1:8011")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--warmups", type=int)
    parser.add_argument("--runs", type=int)
    parser.add_argument("--streams", type=int)
    parser.add_argument("--target-prompt-tokens", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--gpu-sample-interval", type=float, default=0.5)
    args = parser.parse_args()

    warmups, runs, streams, target_tokens = mode_defaults(args.mode)
    warmups = args.warmups if args.warmups is not None else warmups
    runs = args.runs if args.runs is not None else runs
    streams = args.streams if args.streams is not None else streams
    target_tokens = (
        args.target_prompt_tokens
        if args.target_prompt_tokens is not None
        else target_tokens
    )
    max_tokens = args.max_tokens
    if max_tokens is None:
        max_tokens = 1 if args.mode == "prefill" else (128 if args.mode == "capacity" else 256)

    prompts: list[str] = []
    sizing: list[dict[str, Any]] = []
    if target_tokens:
        for stream_index in range(streams):
            prompt, actual, tokenize_calls = size_long_prompt(
                args.base,
                args.model,
                target_tokens,
                stream_index,
                args.timeout,
            )
            prompts.append(prompt)
            sizing.append(
                {
                    "stream": stream_index,
                    "target_prompt_tokens": target_tokens,
                    "tokenized_prompt_tokens": actual,
                    "tokenize_calls": tokenize_calls,
                    "prompt_chars": len(prompt),
                    "prompt_sha256": sha256_text(prompt),
                }
            )
    else:
        prompts = [f"Stream {i}. {SHORT_PROMPT}" for i in range(streams)]

    emit(
        {
            "event": "benchmark_config",
            "mode": args.mode,
            "base": args.base,
            "model": args.model,
            "warmups": warmups,
            "runs": runs,
            "streams": streams,
            "max_tokens": max_tokens,
            "sizing": sizing,
            "first_generation_is_excluded_warmup": bool(warmups),
            "decode_rate_definition": (
                "(completion_tokens-1)/(last_nonempty_delta-first_nonempty_delta)"
            ),
        }
    )

    results: list[dict[str, Any]] = []
    for index in range(warmups):
        result = stream_chat(
            args.base,
            args.model,
            f"Excluded compile/cache warm-up {index + 1}. {SHORT_PROMPT}",
            f"excluded_warmup_{index + 1}",
            max_tokens,
            args.timeout,
        )
        if not result["valid"]:
            raise SystemExit("warm-up semantic gate failed")

    sampler = GPUSampler(args.gpu_sample_interval)
    sampler.start()
    started = time.perf_counter()
    failed_exception: BaseException | None = None
    try:
        for run_index in range(runs):
            if streams == 1:
                batch = [
                    stream_chat(
                        args.base,
                        args.model,
                        prompts[0],
                        f"measured_{run_index + 1}_stream_0",
                        max_tokens,
                        args.timeout,
                    )
                ]
            else:
                with ThreadPoolExecutor(max_workers=streams) as pool:
                    futures = [
                        pool.submit(
                            stream_chat,
                            args.base,
                            args.model,
                            prompts[stream_index],
                            f"measured_{run_index + 1}_stream_{stream_index}",
                            max_tokens,
                            args.timeout,
                        )
                        for stream_index in range(streams)
                    ]
                    batch = [future.result() for future in futures]
            results.extend(batch)
    except Exception as exc:
        # Engine death / connection reset mid-stream: still emit telemetry below.
        failed_exception = exc
    finally:
        elapsed = time.perf_counter() - started
        sampler.stop()

    valid = [result for result in results if result["valid"]]
    if failed_exception is not None or len(valid) != len(results):
        emit(
            {
                "event": "benchmark_summary",
                "mode": args.mode,
                "valid": False,
                "request_count": len(results),
                "wall_s": elapsed,
                "exception": repr(failed_exception) if failed_exception else None,
                "failures": [
                    {
                        "label": result["label"],
                        "failures": result["failures"],
                        "status_code": result["status_code"],
                    }
                    for result in results
                    if not result["valid"]
                ],
                "prompt_tokens": [result["prompt_tokens"] for result in results],
                "completion_tokens": [result["completion_tokens"] for result in results],
                "gpu_telemetry": sampler.summary(),
            }
        )
        raise SystemExit(
            "benchmark failed; see failure summary (with gpu_telemetry) above"
        )
    total_tokens = sum(result["completion_tokens"] for result in results)
    decode_rates = [
        result["decode_tok_s"]
        for result in results
        if result["decode_tok_s"] is not None
    ]
    ttfts = [result["ttft_s"] for result in results if result["ttft_s"] is not None]
    emit(
        {
            "event": "benchmark_summary",
            "mode": args.mode,
            "valid": True,
            "request_count": len(results),
            "wall_s": elapsed,
            "aggregate_completion_tok_s": total_tokens / elapsed,
            "per_request_decode_tok_s": decode_rates,
            "median_request_decode_tok_s": statistics.median(decode_rates)
            if decode_rates
            else None,
            "median_ttft_s": statistics.median(ttfts) if ttfts else None,
            "prompt_tokens": [result["prompt_tokens"] for result in results],
            "completion_tokens": [result["completion_tokens"] for result in results],
            "gpu_telemetry": sampler.summary(),
        }
    )


if __name__ == "__main__":
    main()
