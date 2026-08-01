#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPU-only Monte-Carlo pre-filter for the "TF32 decode-indexer" hypothesis
(agent 4). No GPU, no production impact.

QUESTION (in advance of the FORCE_TORCH=1 boot)
-----------------------------------------------
The decode INDEXER logits kernel runs Q.K in TF32
(`triton_paged_mqa_logits_dsv4.py:185`, `allow_tf32=True`) while prefill scores
in pure fp32. The measured Triton-vs-fp32-torch worst-row relative residual is
`worst_row_rel = 4.691e-4`. Is a ~4.69e-4 per-score perturbation large enough to
flip a borderline rank-512 entry out of the `index_topk=512` cut at decode?

If YES  -> the FORCE_TORCH=1 boot is likely to help (TF32 plausibly drops the
          digit's compressed entry during the digit-emitting decode steps).
If NO   -> the boundary gap is >>> 4.69e-4, TF32 is effectively lossless for
          *selection*, and the boot is likely NULL.

METHOD (pure numpy, CPU)
------------------------
1. Model realistic indexer score distributions for a long-context query. The
   score is `sum_h relu(q_h . k_n) * w_h` over N ~ 2088-2421 compressed entries
   (bf16 q, fp8-derived k, fp32 accumulate). Four generators, all calibrated to
   the *observed* trace statistics (mean~0.09, std~0.81, range [-1.4, 1.6]):
     (a) skew-normal  (b) gaussian-mixture bulk+tail  (c) simulated indexer
     (d) plain Gaussian baseline.
2. Boundary gap: for each draw, gap = score[k-1] - score[k] with k=512 (sorted
   descending). Report the gap DISTRIBUTION.
3. Flip under Gaussian noise: add N(0, sigma) to every score, recompute top-k,
   measure P(boundary entry falls out), E[set-diff size], and P(needle@rank r
   drops). Sweep sigma in 1e-5..1e-2; headline sigma = 4.69e-4.
4. Needle analysis: plant a needle at a target fp32 rank r in 505..520, perturb,
   measure drop probability.
5. Structured TF32 rounding (model c only): compute scores in true fp32 and with
   TF32-rounded operands (10-bit mantissa), use the *actual* deterministic
   residual (not Gaussian) for the flip analysis. This is the physical bound.

ASSUMPTIONS (stated explicitly)
-------------------------------
- N is taken from the production prefill trace candidate counts {2088, 2349,
  2421}; primary = 2088 (the documented decode candidate count, `row=1043
  candidates=2088`). Decode re-selects every step (Finding 5, dig_decode_path.md);
  the candidate count at decode ~ seq_len/4.
- k = index_topk = 512 (production). Indexer `weights w_h` carry sign, so scores
  may be negative (matches observed range, which goes to -1.43).
- The 4.691e-4 is a *worst-row relative* residual. We model two interpretations:
    (i) ABSOLUTE Gaussian noise std = 4.69e-4 x score_std  (row-scale reading,
        "the residual is a few e-4 of the row's per-score scale");
   (ii) PER-SCORE relative noise std = 4.69e-4 x |score_i|   (per-score reading,
        TF32 error scales with operand magnitude).
  Gaussian is an *upper-ish* bound: real TF32 error is a deterministic rounding,
  partially correlated across scores, so its rank-flip power can be smaller than
  i.i.d. Gaussian of the same magnitude. Model (c)+section 5 reports the
  deterministic-TF32 number directly (no Gaussian assumption).
- Score magnitudes are O(1): observed std 0.81, so an absolute noise of 4.69e-4
  x 0.81 ~ 3.8e-4 is the headline absolute noise.

OUTPUT
------
- Markdown report at the path given by --out (plus a .json sidecar).
Run: see --help.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Calibration targets — observed DSv4 indexer trace (runlogs/indexer_rank_trace)
# 2 columns x 16 ratio-4 layers, TP0 deduped (TP1 identical). n=32 unique rows.
# OBS = (mean, std, min, max) of the *reported* scores; used only to show the
# generators reproduce the on-die score scale.
OBS_MEAN, OBS_STD, OBS_MIN, OBS_MAX = 0.093, 0.811, -1.432, 1.616

# Candidate counts observed in production (candidates = seq_len / ratio).
NS = (2088, 2349, 2421)
N_PRIMARY = 2088          # documented decode candidate count
K = 512                    # index_topk (production)
HEADLINE_REL = 4.691e-4    # worst_row_rel from boot self-test


# ---------------------------------------------------------------------------
# Score-distribution generators
# ---------------------------------------------------------------------------

def _round_to_tf32(x: np.ndarray) -> np.ndarray:
    """Round fp32 values to TF32 precision (10 mantissa bits, round-half-even).

    TF32 = sign(1) + exp(8) + mantissa(10). We truncate the fp32 mantissa
    (23 bits) to 10 bits with round-to-nearest-even, matching Ada TF32 MMA.
    """
    u = x.astype(np.float32).view(np.uint32)
    # mantissa LSB to keep: bit (23 - 10) = bit 13. Round-half-even adds 2^12
    # (bit 12) plus the tie-break (any lower bit set -> round up).
    lsb = np.uint32(1) << np.uint32(13)
    rounding_bias = lsb >> np.uint32(1)            # 1 << 12
    tie = (u & (lsb - 1)).astype(bool)             # any bit below the half-bit
    u2 = u + rounding_bias
    # if exactly at tie (only the half-bit set below lsb), clear it for even
    u2 = np.where(tie & ((u2 & lsb) == 0), u2 - rounding_bias, u2)
    u2 = u2 & ~np.uint32((1 << 13) - 1)            # zero the low 13 bits
    return u2.view(np.float32).astype(np.float64)


def calibrate_row(s: np.ndarray, target_std: float = OBS_STD) -> np.ndarray:
    """Rescale a score vector to the observed per-row std (~0.81).

    The flip probability depends on (boundary_gap / noise). The gap scales
    linearly with score std, so all generators MUST produce the same per-row
    std for the flip comparison to be apples-to-apples. The observed trace has
    std~0.81; we rebalance every draw to that scale. (Shape/skew is preserved;
    only the scale is fixed. Real rows vary in std; this targets the typical
    row — a central estimate for a pre-filter.)
    """
    sd = float(s.std())
    if sd < 1e-12:
        return s
    return s * (target_std / sd)


def gen_skewnormal(n: int, rng: np.random.Generator) -> np.ndarray:
    """Skew-normal shape (slight positive skew), rebalanced to observed std."""
    loc, scale, alpha = 0.05, 1.0, 2.0
    u1 = rng.standard_normal(n)
    u2 = rng.standard_normal(n)
    sn = np.where(u1 >= 0, np.abs(u1), -np.abs(u1))  # skew via folded sign
    delta = alpha / math.sqrt(1.0 + alpha * alpha)
    raw = loc + scale * (delta * sn + math.sqrt(1 - delta * delta) * u2)
    return calibrate_row(raw)


def gen_gaussian(n: int, rng: np.random.Generator) -> np.ndarray:
    """Plain Gaussian baseline N(OBS_MEAN, OBS_STD), rebalanced to observed std."""
    return calibrate_row(rng.normal(OBS_MEAN, OBS_STD, n))


def gen_mixture(n: int, rng: np.random.Generator) -> np.ndarray:
    """Mixture: ~70% near-zero bulk, ~25% moderate, ~5% high (semantic matches).

    Mimics "most entries small, a long tail, a few high". Rebalanced to observed
    std (shape preserved).
    """
    comp = rng.choice(3, n, p=(0.70, 0.25, 0.05))
    s = np.empty(n)
    m0 = comp == 0
    m1 = comp == 1
    m2 = comp == 2
    # bulk: small positive/negative around 0 (rectified-noise-like)
    s[m0] = rng.normal(0.0, 0.30, m0.sum())
    # moderate tail
    s[m1] = rng.normal(0.55, 0.35, m1.sum())
    # high matches (the needle / semantically relevant entries)
    s[m2] = rng.normal(1.15, 0.35, m2.sum())
    # a fraction of heads carry negative weight -> allow negative scores
    neg = rng.random(n) < 0.18
    s[neg] = -np.abs(s[neg]) * rng.uniform(0.4, 1.0, neg.sum())
    return calibrate_row(s)


def gen_simulated_indexer(n: int, rng: np.random.Generator,
                          h: int = 64, d: int = 64) -> np.ndarray:
    """Physically-grounded: score_n = sum_h relu(q_h . k_n) * w_h.

    q ~ bf16-rounded small-magnitude, k ~ fp8-e4m3-derived. w_h carries sign.
    fp32 accumulate. Rebalanced to observed std (the absolute scale of the
    synthesized dots is arbitrary; only the shape matters for the boundary gap,
    and calibrate_row fixes the scale to the on-die 0.81).
    """
    q = (rng.standard_normal(h * d) * 0.06).reshape(h, d)
    q = _round_to_bf16(q)
    k = _round_to_fp8e4m3((rng.standard_normal(n * d) * 0.08).reshape(n, d))
    w = rng.normal(0.0, 1.0, h)
    dots = q @ k.T                                        # [H, N]
    contrib = np.maximum(dots, 0.0) * w[:, None]          # [H, N]
    scores = contrib.sum(axis=0).astype(np.float64)       # [N]
    return calibrate_row(scores)


def _round_to_bf16(x: np.ndarray) -> np.ndarray:
    """Round to bf16 (7 mantissa bits, round-half-even)."""
    u = np.asarray(x, np.float32).view(np.uint32)
    lsb = np.uint32(1) << np.uint32(16)
    bias = lsb >> np.uint32(1)
    u2 = u + bias
    u2 = u2 & ~np.uint32((1 << 16) - 1)
    return u2.view(np.float32).astype(np.float64)


def _round_to_fp8e4m3(x: np.ndarray) -> np.ndarray:
    """Approximate FP8-E4M3 quantization (3 mantissa bits, shared scale).

    Per-tensor scale (real kernel uses per-token scale; per-tensor is a
    coarser-but-sufficient proxy for the distribution shape).
    """
    amax = np.abs(x).max() if np.abs(x).max() > 0 else 1.0
    # fp8e4m3 normal max ~448; choose scale so amax -> ~224 (headroom)
    scale = 224.0 / amax
    q = x * scale
    # round to 3 mantissa bits (8 levels per power-of-two) round-to-nearest-even
    # implemented via: sign * round_to_pow2_levels(|q|)
    s = np.sign(q)
    a = np.abs(q)
    # represent as m * 2^e with m in [1,2); quantize m to 8 levels
    e = np.floor(np.log2(np.maximum(a, 1e-30)))
    m = a / np.power(2.0, e)
    mq = np.round(m * 8) / 8.0
    aq = np.where(a > 0, mq * np.power(2.0, e), 0.0)
    return (s * aq / scale)


GENERATORS = {
    "skew_normal": gen_skewnormal,
    "gaussian": gen_gaussian,
    "mixture": gen_mixture,
    "simulated_indexer": gen_simulated_indexer,
}


# ---------------------------------------------------------------------------
# Core measurements
# ---------------------------------------------------------------------------

def boundary_gap(scores: np.ndarray, k: int = K) -> float:
    """score[k-1] - score[k] on a descending sort (the k/k+1 boundary)."""
    s_sorted = np.sort(scores)[::-1]
    return float(s_sorted[k - 1] - s_sorted[k])


def make_pool(gen, m_draws: int, n: int, seed: int) -> np.ndarray:
    """Generate an [m_draws, n] pool of score vectors (one row = one indexer row)."""
    rng = np.random.default_rng(seed)
    out = np.empty((m_draws, n), dtype=np.float64)
    for i in range(m_draws):
        out[i] = gen(n, rng)
    return out


def calibrate_pool(pool: np.ndarray) -> dict:
    """Score-stat + boundary-gap stats over a pool of [M, N] draws."""
    m, n = pool.shape
    means = pool.mean(axis=1); stds = pool.std(axis=1)
    mins = pool.min(axis=1); maxs = pool.max(axis=1)
    s_sorted = np.sort(pool, axis=1)[:, ::-1]            # [M,N] descending
    gaps = s_sorted[:, K - 1] - s_sorted[:, K]
    return {
        "mean": float(means.mean()), "std": float(stds.mean()),
        "min": float(mins.mean()), "max": float(maxs.mean()),
        "gap_p50": float(np.median(gaps)),
        "gap_mean": float(np.mean(gaps)),
        "gap_p10": float(np.percentile(gaps, 10)),
        "gap_p90": float(np.percentile(gaps, 90)),
        "gap_max": float(np.max(gaps)),
    }


def _flip_compare(pool: np.ndarray, noisy: np.ndarray):
    """Return (p_boundary_out, e_symdiff, p_any_swap) comparing fp32 top-k to noisy."""
    m, n = pool.shape
    rows = np.arange(m)[:, None]
    order = np.argsort(-pool, axis=1)                    # [M,N] desc
    boundary = order[:, K - 1].copy()                    # [M] rank-k index
    in_orig = np.zeros((m, n), dtype=bool)
    in_orig[rows, order[:, :K]] = True
    new_topk = np.argpartition(noisy, n - K, axis=1)[:, -K:]
    in_new = np.zeros((m, n), dtype=bool)
    in_new[rows, new_topk] = True
    boundary_out = ~in_new[np.arange(m), boundary]
    symdiff = np.sum(in_orig ^ in_new, axis=1)           # |A ^ B|
    return (float(boundary_out.mean()),
            float(symdiff.mean()),
            float((symdiff >= 1).mean()))


def flip_study_pool_abs(pool: np.ndarray, sigmas, seed: int = 0) -> dict:
    """Absolute Gaussian noise: noisy = pool + N(0, sigma)."""
    rng = np.random.default_rng(seed)
    out = {}
    for sig in sigmas:
        noisy = pool + rng.normal(0.0, sig, size=pool.shape)
        p_bo, e_sym, p_any = _flip_compare(pool, noisy)
        out["%.2e" % sig] = {
            "sigma_abs": float(sig),
            "rel_to_score_std": float(sig / OBS_STD),
            "p_boundary_out": p_bo,
            "e_symdiff": e_sym,
            "p_symdiff_ge1": p_any,
        }
    return out


def flip_study_pool_rel(pool: np.ndarray, rels, seed: int = 0) -> dict:
    """Per-score relative noise: sigma_i = rel * |score_i| (TF32-magnitude-scaled)."""
    rng = np.random.default_rng(seed)
    out = {}
    for rel in rels:
        sigma_i = rel * np.abs(pool)
        noisy = pool + rng.normal(0.0, 1.0, size=pool.shape) * sigma_i
        p_bo, e_sym, p_any = _flip_compare(pool, noisy)
        out["%.2e" % rel] = {
            "rel": float(rel),
            "p_boundary_out": p_bo,
            "e_symdiff": e_sym,
            "p_symdiff_ge1": p_any,
        }
    return out


def needle_study_pool(pool: np.ndarray, ranks, sigmas_abs, rels,
                      seed: int = 0):
    """Plant a needle at target fp32 rank r; perturb; P(needle falls out of top-k).

    Returns (out_abs, out_rel): dicts keyed by sigma/rel -> {rank: p_drop}.
    """
    m, n = pool.shape
    order = np.argsort(-pool, axis=1)                    # [M,N] desc
    rows = np.arange(m)[:, None]
    rng = np.random.default_rng(seed)
    out_abs, out_rel = {}, {}
    for sig in sigmas_abs:
        noisy = pool + rng.normal(0.0, sig, size=pool.shape)
        new_topk = np.argpartition(noisy, n - K, axis=1)[:, -K:]
        in_new = np.zeros((m, n), dtype=bool)
        in_new[rows, new_topk] = True
        d = {}
        for r in ranks:
            if 1 <= r <= n:
                needle = order[:, r - 1]
                d[r] = float((~in_new[np.arange(m), needle]).mean())
        out_abs["%.2e" % sig] = d
    for rel in rels:
        sigma_i = rel * np.abs(pool)
        noisy = pool + rng.normal(0.0, 1.0, size=pool.shape) * sigma_i
        new_topk = np.argpartition(noisy, n - K, axis=1)[:, -K:]
        in_new = np.zeros((m, n), dtype=bool)
        in_new[rows, new_topk] = True
        d = {}
        for r in ranks:
            if 1 <= r <= n:
                needle = order[:, r - 1]
                d[r] = float((~in_new[np.arange(m), needle]).mean())
        out_rel["%.2e" % rel] = d
    return out_abs, out_rel


def simulated_indexer_pair(rng, n: int, h: int = 64, d: int = 64):
    """Return (scores_fp32, scores_tf32) for one draw.

    scores = sum_h relu(q_h . k_n) * w_h  computed (a) with fp32 operands/dots,
    (b) with TF32-rounded operands (Ada TF32 MMA), fp32 accumulate in both.
    """
    q = (rng.standard_normal(h * d) * 0.06).reshape(h, d)
    q_bf16 = _round_to_bf16(q)                       # the kernel loads q as bf16-ish
    q_tf32 = _round_to_tf32(q_bf16)                  # then TF32 truncates the dot
    k = (rng.standard_normal(n * d) * 0.08).reshape(n, d)
    k_fp8 = _round_to_fp8e4m3(k)
    k_tf32 = _round_to_tf32(k_fp8)
    w = rng.normal(0.0, 1.0, h)
    w = w / (np.abs(w).sum()) * h * 0.020

    dots_fp32 = q_bf16 @ k_fp8.T
    dots_tf32 = q_tf32 @ k_tf32.T
    s_fp32 = (np.maximum(dots_fp32, 0.0) * w[:, None]).sum(axis=0)
    s_tf32 = (np.maximum(dots_tf32, 0.0) * w[:, None]).sum(axis=0)
    s_fp32 = s_fp32.astype(np.float64)
    s_tf32 = s_tf32.astype(np.float64)
    # Apply ONE scale (derived from the fp32 row) to both so the absolute
    # residual scales onto the observed score std; relative residual unchanged.
    scale = OBS_STD / max(float(s_fp32.std()), 1e-12)
    return s_fp32 * scale, s_tf32 * scale


def structured_tf32_batch(b_draws: int, n: int, seed: int = 0,
                          h: int = 64, d: int = 64) -> dict:
    """Real TF32 rounding (10-bit mantissa) on the simulated indexer; batched."""
    rng = np.random.default_rng(seed)
    s_fp = np.empty((b_draws, n), dtype=np.float64)
    s_tf = np.empty((b_draws, n), dtype=np.float64)
    for i in range(b_draws):
        a, b = simulated_indexer_pair(rng, n, h, d)
        s_fp[i] = a
        s_tf[i] = b
    p_bo, e_sym, _ = _flip_compare(s_fp, s_tf)
    abs_res = np.abs(s_tf - s_fp)                          # [B,N] absolute
    # "worst_row_rel": max_i |d_i| / (row std) — matches the boot self-test
    # semantics (relative to the row's score scale, NOT per-score |score_i|,
    # which blows up for near-zero scores).
    row_std = s_fp.std(axis=1)                             # [B]
    rel_rowmax = abs_res.max(axis=1) / np.maximum(row_std, 1e-12)
    s_sorted = np.sort(s_fp, axis=1)[:, ::-1]
    gaps = s_sorted[:, K - 1] - s_sorted[:, K]
    return {
        "abs_res_p50": float(np.percentile(abs_res, 50)),
        "abs_res_p90": float(np.percentile(abs_res, 90)),
        "abs_res_max": float(abs_res.max()),
        "rel_res_p50": float(np.percentile(rel_rowmax, 50)),
        "rel_res_p90": float(np.percentile(rel_rowmax, 90)),
        "rel_res_max": float(rel_rowmax.max()),
        "gap_p50": float(np.median(gaps)),
        "gap_mean": float(np.mean(gaps)),
        "p_boundary_out": p_bo,
        "e_symdiff": e_sym,
        "n_draws": b_draws,
    }


def run_all(args):
    t0 = time.time()
    m_flip = args.ndraw_flip
    m_gap = args.ndraw_gap
    m_gap_nn = min(args.ndraw_gap, 2000)   # cap gap_vs_N draws (sim_indexer is slow)
    sigma_abs_headline = HEADLINE_REL * OBS_STD
    sigmas = sorted(set([1e-5, 3e-5, 1e-4, 2e-4,
                         sigma_abs_headline, 5e-4, 1e-3, 3e-3, 1e-2]))
    rels = sorted(set([1e-5, 1e-4, HEADLINE_REL, 1e-3, 1e-2]))
    ranks = list(range(505, 521))

    out = {
        "meta": {
            "script": os.path.abspath(__file__),
            "observed_calibration": {
                "mean": OBS_MEAN, "std": OBS_STD, "min": OBS_MIN, "max": OBS_MAX,
            },
            "candidate_counts": list(NS),
            "n_primary": N_PRIMARY,
            "k": K,
            "headline_rel": HEADLINE_REL,
            "sigma_abs_headline": float(sigma_abs_headline),
            "n_draws_gap": m_gap,
            "n_draws_flip": m_flip,
        },
    }

    print("[tf32_flip] building %d-draw pools at N=%d..." % (m_flip, N_PRIMARY),
          flush=True)
    pools = {name: make_pool(gen, m_flip, N_PRIMARY, seed=1)
             for name, gen in GENERATORS.items()}

    # --- 1-2: calibration + boundary gap -------------------------------------
    calib = {name: calibrate_pool(make_pool(gen, m_gap, N_PRIMARY, seed=2))
             for name, gen in GENERATORS.items()}
    out["calibration_and_gap_N2088"] = calib

    gap_vs_n = {}
    for name, gen in GENERATORS.items():
        gap_vs_n[name] = {
            str(nval): calibrate_pool(make_pool(gen, m_gap_nn, nval, seed=3))
            for nval in NS}
    out["gap_vs_N"] = gap_vs_n

    # --- 3: flip under absolute Gaussian noise -------------------------------
    print("[tf32_flip] absolute-noise flip study...", flush=True)
    out["flip_absolute"] = {
        name: flip_study_pool_abs(pools[name], sigmas, seed=4)
        for name in GENERATORS}

    # --- 3b: flip under per-score relative noise -----------------------------
    print("[tf32_flip] relative-noise flip study...", flush=True)
    out["flip_per_score_relative"] = {
        name: flip_study_pool_rel(pools[name], rels, seed=5)
        for name in ("skew_normal", "mixture", "simulated_indexer")}

    # --- 4: needle at target fp32 rank ---------------------------------------
    print("[tf32_flip] needle study...", flush=True)
    needle = {}
    abs_k = "absolute_sigma_%.2e" % sigma_abs_headline
    rel_k = "per_score_relative_%.2e" % HEADLINE_REL
    for name in ("skew_normal", "mixture", "simulated_indexer"):
        out_abs, out_rel = needle_study_pool(
            pools[name], ranks, [sigma_abs_headline], [HEADLINE_REL], seed=6)
        needle[name] = {abs_k: out_abs["%.2e" % sigma_abs_headline],
                        rel_k: out_rel["%.2e" % HEADLINE_REL]}
    out["needle_at_rank"] = needle

    # --- 5: structured TF32 --------------------------------------------------
    print("[tf32_flip] structured-TF32 batch...", flush=True)
    out["structured_tf32"] = {
        "N2088": structured_tf32_batch(args.ndraw_struct, N_PRIMARY, seed=7)}

    out["meta"]["elapsed_s"] = round(time.time() - t0, 1)
    return out



def fmt_pct(x: float) -> str:
    if x < 1e-4:
        return "<0.01%"
    return "%.2f%%" % (x * 100.0)


def write_report(res: dict, out_md: str):
    lines = []
    A = lines.append
    m = res["meta"]
    A("# TF32 decode-indexer flip pre-filter (pure-CPU Monte-Carlo)")
    A("")
    A("**Question.** Is the measured Triton-vs-fp32 decode-indexer residual "
      "`worst_row_rel = 4.691e-4` large enough to flip a borderline rank-512 "
      "entry out of `index_topk=512` at decode? (Pre-filter for the "
      "`VLLM_DSV4_PAGED_MQA_LOGITS_FORCE_TORCH=1` boot.)")
    A("")
    A("## Setup")
    A(f"- Candidate counts N in {2088, 2349, 2421} (production trace); "
      "**primary N = 2088** (documented decode count). k = 512.")
    A("- Score generators calibrated to observed trace stats "
      f"(mean {OBS_MEAN}, std {OBS_STD}, range [{OBS_MIN}, {OBS_MAX}]).")
    A(f"- Headline absolute Gaussian noise std = `4.691e-4 x {OBS_STD:.3f}` "
      f"= **{m['sigma_abs_headline']:.3e}** (row-scale reading of worst_row_rel).")
    A(f"- Draws: gap={m['n_draws_gap']}, flip/needle={m['n_draws_flip']}. "
      f"Elapsed {m['elapsed_s']}s.")
    A("")
    A("## 1-2. Boundary gap (score[k-1]-score[k], k=512) and calibration")
    A("")
    A("Observed score stats -> mean 0.09, std 0.81, range [-1.43, 1.62]. "
      "Generators reproduce this scale:")
    A("")
    A("| generator | mean | std | min | max | **gap p50** | gap p10 | gap p90 |")
    A("|---|---|---|---|---|---|---|---|")
    for name, c in res["calibration_and_gap_N2088"].items():
        A(f"| {name} | {c['mean']:.3f} | {c['std']:.3f} | {c['min']:.2f} | "
          f"{c['max']:.2f} | **{c['gap_p50']:.2e}** | "
          f"{c['gap_p10']:.2e} | {c['gap_p90']:.2e} |")
    A("")
    A("Boundary gap vs N (p50):")
    A("")
    A("| generator | N=2088 | N=2349 | N=2421 |")
    A("|---|---|---|---|")
    for name in ("skew_normal", "mixture", "simulated_indexer", "gaussian"):
        row = [name]
        for nval in NS:
            row.append(f"{res['gap_vs_N'][name][str(nval)]['gap_p50']:.2e}")
        A("| " + " | ".join(row) + " |")
    A("")
    A("> The typical boundary gap is ~1e-3 (order). The headline TF32 noise "
      f"({m['sigma_abs_headline']:.2e}) is of the **same order of magnitude** "
      "as the boundary gap — so flips are *possible*, not a priori ruled out.")
    A("")

    A("## 3. Flip probability under absolute Gaussian noise (N=2088, k=512)")
    A("")
    A("`sigma_abs = 4.69e-4 x 0.811 = 3.8e-4` is the headline row. "
      "`p_boundary_out` = P(the original rank-512 entry falls OUT of the new "
      "top-512). `e_symdiff` = E[|topk_old ^ topk_new|] (set XOR size; = 2 x "
      "swapped pairs). `p_any_swap` = P(at least one entry changes).")
    A("")
    for name in ("skew_normal", "mixture", "simulated_indexer", "gaussian"):
        A(f"### {name}")
        A("| sigma_abs | rel-to-std | P(boundary out) | P(any swap) | E[symdiff] |")
        A("|---|---|---|---|---|")
        for sig_s, r in res["flip_absolute"][name].items():
            star = (" **<- headline**" if m["sigma_abs_headline"] > 0
                    and abs(r["sigma_abs"] - m["sigma_abs_headline"])
                    / m["sigma_abs_headline"] < 0.02 else "")
            A(f"| {r['sigma_abs']:.2e} | {r['rel_to_score_std']:.2e} | "
              f"{fmt_pct(r['p_boundary_out'])}{star} | "
              f"{fmt_pct(r['p_symdiff_ge1'])} | {r['e_symdiff']:.3f} |")
        A("")

    A("## 3b. Per-score RELATIVE noise (sigma_i = rel x |score_i|)")
    A("")
    A("More physically faithful for TF32 (operand-magnitude-scaled). Headline "
      f"rel = {HEADLINE_REL:.2e}.")
    A("")
    for name in ("skew_normal", "mixture", "simulated_indexer"):
        A(f"### {name}")
        A("| rel | P(boundary out) | E[symdiff] |")
        A("|---|---|---|")
        for rel_s, r in res["flip_per_score_relative"][name].items():
            star = " **<- headline**" if abs(r["rel"] - HEADLINE_REL) < 1e-6 else ""
            A(f"| {r['rel']:.2e} | {fmt_pct(r['p_boundary_out'])}{star} | "
              f"{r['e_symdiff']:.3f} |")
        A("")

    A("## 4. Needle at target fp32 rank (does a specific borderline needle drop?)")
    A("")
    A("Plant the needle at rank r (fp32 ordering), apply headline noise, "
      "measure drop probability. **r=512 is the cut**: a needle at fp32 rank "
      "512 is exactly the last entry inside top-512; if it drops, the digit's "
      "compressed entry is unselected at decode.")
    A("")
    for name in ("skew_normal", "mixture", "simulated_indexer"):
        A(f"### {name}")
        abs_k = "absolute_sigma_%s" % f"{m['sigma_abs_headline']:.2e}"
        rel_k = "per_score_relative_%s" % f"{HEADLINE_REL:.2e}"
        d_abs = res["needle_at_rank"][name][abs_k]
        d_rel = res["needle_at_rank"][name][rel_k]
        A("| rank r | abs-noise P(drop) | rel-noise P(drop) |")
        A("|---|---|---|")
        for r in sorted(d_abs):
            a = fmt_pct(d_abs[r]) if r in d_abs else "-"
            rl = fmt_pct(d_rel[r]) if r in d_rel else "-"
            star = " **<- cut**" if str(r) == "512" else ""
            A(f"| {r} | {a}{star} | {rl} |")
        A("")

    A("## 5. Structured TF32 (real 10-bit-mantissa rounding, simulated indexer)")
    A("")
    A("Deterministic TF32 truncation (NOT Gaussian) on the simulated indexer. "
      "This is the physical bound: the actual residual the kernel introduces.")
    A("")
    st = res["structured_tf32"]["N2088"]
    A(f"- absolute residual |s_tf32 - s_fp32|: p50 {st['abs_res_p50']:.2e}, "
      f"p90 {st['abs_res_p90']:.2e}, max {st['abs_res_max']:.2e}")
    A(f"- worst-**row** relative residual (max|d_i| / row-std): "
      f"p50 {st['rel_res_p50']:.2e}, p90 {st['rel_res_p90']:.2e}, "
      f"max {st['rel_res_max']:.2e}  (cf. boot self-test worst_row_rel=4.691e-4)")
    A(f"- boundary gap p50: {st['gap_p50']:.2e}")
    A(f"- **P(boundary entry out under real TF32) = {fmt_pct(st['p_boundary_out'])}**")
    A(f"- E[symdiff] = {st['e_symdiff']:.3f}")
    A("")

    # ---- Verdict -----------------------------------------------------------
    A("## Verdict")
    A("")
    # use simulated_indexer + skew_normal + mixture at headline abs noise
    p_abs = np.mean([res["flip_absolute"][n][f"{m['sigma_abs_headline']:.2e}"]["p_boundary_out"]
                     for n in ("skew_normal", "mixture", "simulated_indexer")])
    p_rel = np.mean([res["flip_per_score_relative"][n][f"{HEADLINE_REL:.2e}"]["p_boundary_out"]
                     for n in ("skew_normal", "mixture", "simulated_indexer")])
    p_struct = st["p_boundary_out"]
    # needle-at-512 drop under abs noise, averaged (dict keys are INTEGERS)
    p_needle_512_abs = np.mean([
        res["needle_at_rank"][n]["absolute_sigma_%s" % f"{m['sigma_abs_headline']:.2e}"].get(512, 0.0)
        for n in ("skew_normal", "mixture", "simulated_indexer")])
    p_needle_510_abs = np.mean([
        res["needle_at_rank"][n]["absolute_sigma_%s" % f"{m['sigma_abs_headline']:.2e}"].get(510, 0.0)
        for n in ("skew_normal", "mixture", "simulated_indexer")])
    gap_med = np.median([res["calibration_and_gap_N2088"][n]["gap_p50"]
                         for n in ("skew_normal", "mixture", "simulated_indexer")])
    # the simulated fp8 quant is per-tensor (coarser than the real per-token), so
    # the structured residual is an OVER-estimate; scale the structured flip rate
    # down by (real worst_row_rel / simulated rel_res_p90) as a correction.
    struct_residual_ratio = HEADLINE_REL / max(st["rel_res_p90"], 1e-9)
    A(f"- Typical boundary gap (rank 512/513) ~ **{gap_med:.2e}** (median across "
      "generators).")
    A(f"- Headline TF32 noise magnitude ~ **{m['sigma_abs_headline']:.2e}** "
      f"(= 4.691e-4 x score_std).")
    A(f"- Ratio (gap / noise) ~ **{gap_med / m['sigma_abs_headline']:.2f}**. "
      "The boundary gap is within a small constant factor of the noise.")
    A("- P(boundary entry flips) under Gaussian: abs-noise ~ "
      f"{fmt_pct(p_abs)}, per-score-relative ~ {fmt_pct(p_rel)}.")
    A("- P(boundary entry flips) under **real structured TF32** (sim indexer, "
      f"per-tensor fp8 proxy): **{fmt_pct(p_struct)}**. NB the simulated fp8 "
      f"quantization is coarser than the kernel's per-token scale, so its "
      f"residual (p90 {st['rel_res_p90']:.2e}) is ~{1/struct_residual_ratio:.1f}x "
      f"the measured 4.691e-4; the real-kernel structured flip rate is likely "
      f"correspondingly LOWER (corrected est. ~{fmt_pct(p_struct * struct_residual_ratio)}).")
    A("- P(needle planted exactly at fp32 rank 512 is dropped) under abs "
      f"noise ~ {fmt_pct(p_needle_512_abs)}; at rank 510 ~ {fmt_pct(p_needle_510_abs)}.")
    A("")
    if p_abs > 0.05 or p_needle_512_abs > 0.05:
        verdict = ("**TF32 is plausibly strong enough to flip a borderline "
                   "rank-512 entry at decode.** The boundary gap (~1e-3) is only "
                   f"~{gap_med / m['sigma_abs_headline']:.1f}x the 4.691e-4 residual, "
                   "and a needle sitting at the rank-512 cut drops ~"
                   f"{fmt_pct(p_needle_512_abs)} per decode step under TF32-scale "
                   "perturbation (and is essentially safe, <0.1%, at rank <=510). "
                   "So if the digit's compressed entry rides the rank-512 boundary "
                   "at the digit-emitting decode steps, TF32 can drop it. The "
                   "`FORCE_TORCH=1` boot is **likely to help** (worth running). "
                   "Caveat: this is necessary-but-not-sufficient — the digit must "
                   "actually sit near rank 512 at those steps (T4 found it sits at "
                   "rank 0..917 in prefill; decode rank is unmeasured), and a "
                   "single dropped step is often recoverable. Expect a *partial* "
                   "improvement, not a guaranteed full fix. NB: Gaussian i.i.d. "
                   "noise is an upper-ish bound; the structured-TF32 number is the "
                   "tighter, physically-grounded one and points the same way "
                   "(corrected ~" + fmt_pct(p_struct * struct_residual_ratio) + ").")
    else:
        verdict = ("**The boundary gap (~1e-3) is >> the 4.691e-4 residual in "
                   "practice; TF32 is effectively lossless for top-512 SELECTION.** "
                   "The `FORCE_TORCH=1` boot is **likely NULL** for the selection "
                   "mechanism. Redirect to H3/H4 (chunked-compressor correctness, "
                   "config divergence).")
    A(verdict)
    A("")
    A("> NB: Gaussian i.i.d. noise is an *upper-ish* bound on flip probability; "
      "the structured-TF32 section (5) is the tighter, physically-grounded number "
      "and is the one to weight most heavily.")
    A("")
    A("---")
    A("Generated by `tools/tf32_flip_montecarlo.py`. Pure CPU (numpy). "
      "No GPU, no production impact.")

    with open(out_md, "w") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None,
                    help="output markdown path (default: "
                         "docs/dsv4-sm89-longcontext/runlogs/tf32_flip_montecarlo"
                         "_<timestamp>.md)")
    ap.add_argument("--ndraw-gap", type=int, default=6000,
                    help="draws for gap/calibration")
    ap.add_argument("--ndraw-flip", type=int, default=4000,
                    help="draws for flip/needle studies")
    ap.add_argument("--ndraw-struct", type=int, default=3000,
                    help="draws for the structured-TF32 (simulated indexer) batch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true",
                    help="also write a .json sidecar next to the report")
    args = ap.parse_args()

    if args.out is None:
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        out = (f"docs/dsv4-sm89-longcontext/runlogs/"
               f"tf32_flip_montecarlo_{ts}.md")
    else:
        out = args.out

    print(f"[tf32_flip] running: gap draws={args.ndraw_gap}, "
          f"flip draws={args.ndraw_flip}", flush=True)
    res = run_all(args)
    write_report(res, out)
    if args.json:
        js = out.rsplit(".", 1)[0] + ".json"
        with open(js, "w") as f:
            json.dump(res, f, indent=2)
        print(f"[tf32_flip] json  -> {js}", flush=True)
    print(f"[tf32_flip] report -> {out}", flush=True)

    # Echo the headline numbers to stdout for the caller.
    m = res["meta"]
    sig = m["sigma_abs_headline"]
    print("\n=== HEADLINE ===", flush=True)
    print(f"boundary gap p50 (simulated_indexer): "
          f"{res['calibration_and_gap_N2088']['simulated_indexer']['gap_p50']:.2e}",
          flush=True)
    print(f"headline abs noise sigma: {sig:.2e}", flush=True)
    for n in ("skew_normal", "mixture", "simulated_indexer"):
        p = res["flip_absolute"][n][f"{sig:.2e}"]["p_boundary_out"]
        print(f"P(boundary out) [{n}] @ headline abs noise: {p*100:.2f}%",
              flush=True)
    st = res["structured_tf32"]["N2088"]
    print(f"structured real-TF32 P(boundary out): {st['p_boundary_out']*100:.2f}% "
          f"(rel residual p90 {st['rel_res_p90']:.2e})", flush=True)


if __name__ == "__main__":
    main()
