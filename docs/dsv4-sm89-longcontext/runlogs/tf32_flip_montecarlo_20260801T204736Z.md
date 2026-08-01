# TF32 decode-indexer flip pre-filter (pure-CPU Monte-Carlo)

**Question.** Is the measured Triton-vs-fp32 decode-indexer residual `worst_row_rel = 4.691e-4` large enough to flip a borderline rank-512 entry out of `index_topk=512` at decode? (Pre-filter for the `VLLM_DSV4_PAGED_MQA_LOGITS_FORCE_TORCH=1` boot.)

## Setup
- Candidate counts N in (2088, 2349, 2421) (production trace); **primary N = 2088** (documented decode count). k = 512.
- Score generators calibrated to observed trace stats (mean 0.093, std 0.811, range [-1.432, 1.616]).
- Headline absolute Gaussian noise std = `4.691e-4 x 0.811` = **3.804e-04** (row-scale reading of worst_row_rel).
- Draws: gap=4000, flip/needle=6000. Elapsed 307.6s.

## 1-2. Boundary gap (score[k-1]-score[k], k=512) and calibration

Observed score stats -> mean 0.09, std 0.81, range [-1.43, 1.62]. Generators reproduce this scale:

| generator | mean | std | min | max | **gap p50** | gap p10 | gap p90 |
|---|---|---|---|---|---|---|---|
| skew_normal | 0.041 | 0.811 | -2.75 | 2.83 | **8.64e-04** | 1.23e-04 | 3.04e-03 |
| gaussian | 0.093 | 0.811 | -2.70 | 2.89 | **8.37e-04** | 1.33e-04 | 2.94e-03 |
| mixture | 0.203 | 0.811 | -2.64 | 3.58 | **1.02e-03** | 1.68e-04 | 3.41e-03 |
| simulated_indexer | 0.006 | 0.811 | -2.90 | 2.91 | **8.68e-04** | 1.30e-04 | 2.86e-03 |

Boundary gap vs N (p50):

| generator | N=2088 | N=2349 | N=2421 |
|---|---|---|---|
| skew_normal | 8.41e-04 | 7.85e-04 | 7.79e-04 |
| mixture | 1.05e-03 | 1.05e-03 | 1.05e-03 |
| simulated_indexer | 8.59e-04 | 8.15e-04 | 8.28e-04 |
| gaussian | 8.87e-04 | 8.24e-04 | 7.71e-04 |

> The typical boundary gap is ~1e-3 (order). The headline TF32 noise (3.80e-04) is of the **same order of magnitude** as the boundary gap — so flips are *possible*, not a priori ruled out.

## 3. Flip probability under absolute Gaussian noise (N=2088, k=512)

`sigma_abs = 4.69e-4 x 0.811 = 3.8e-4` is the headline row. `p_boundary_out` = P(the original rank-512 entry falls OUT of the new top-512). `e_symdiff` = E[|topk_old ^ topk_new|] (set XOR size; = 2 x swapped pairs). `p_any_swap` = P(at least one entry changes).

### skew_normal
| sigma_abs | rel-to-std | P(boundary out) | P(any swap) | E[symdiff] |
|---|---|---|---|---|
| 1.00e-05 | 1.23e-05 | 0.50% | 0.50% | 0.010 |
| 3.00e-05 | 3.70e-05 | 1.32% | 1.33% | 0.027 |
| 1.00e-04 | 1.23e-04 | 4.62% | 4.78% | 0.096 |
| 2.00e-04 | 2.47e-04 | 8.23% | 8.97% | 0.180 |
| 3.80e-04 | 4.69e-04 | 13.45% **<- headline** | 15.72% | 0.316 |
| 5.00e-04 | 6.17e-04 | 17.63% | 21.02% | 0.427 |
| 1.00e-03 | 1.23e-03 | 26.63% | 37.02% | 0.789 |
| 3.00e-03 | 3.70e-03 | 41.83% | 78.53% | 2.194 |
| 1.00e-02 | 1.23e-02 | 47.17% | 99.53% | 6.750 |

### mixture
| sigma_abs | rel-to-std | P(boundary out) | P(any swap) | E[symdiff] |
|---|---|---|---|---|
| 1.00e-05 | 1.23e-05 | 0.23% | 0.23% | 0.005 |
| 3.00e-05 | 3.70e-05 | 0.98% | 1.00% | 0.020 |
| 1.00e-04 | 1.23e-04 | 3.05% | 3.08% | 0.062 |
| 2.00e-04 | 2.47e-04 | 6.33% | 7.02% | 0.141 |
| 3.80e-04 | 4.69e-04 | 11.43% **<- headline** | 12.97% | 0.261 |
| 5.00e-04 | 6.17e-04 | 14.37% | 16.73% | 0.337 |
| 1.00e-03 | 1.23e-03 | 23.62% | 32.42% | 0.678 |
| 3.00e-03 | 3.70e-03 | 39.63% | 71.50% | 1.819 |
| 1.00e-02 | 1.23e-02 | 46.73% | 99.03% | 5.555 |

### simulated_indexer
| sigma_abs | rel-to-std | P(boundary out) | P(any swap) | E[symdiff] |
|---|---|---|---|---|
| 1.00e-05 | 1.23e-05 | 0.45% | 0.45% | 0.009 |
| 3.00e-05 | 3.70e-05 | 1.10% | 1.12% | 0.022 |
| 1.00e-04 | 1.23e-04 | 4.23% | 4.33% | 0.087 |
| 2.00e-04 | 2.47e-04 | 8.35% | 9.12% | 0.183 |
| 3.80e-04 | 4.69e-04 | 14.48% **<- headline** | 17.03% | 0.343 |
| 5.00e-04 | 6.17e-04 | 18.02% | 22.13% | 0.451 |
| 1.00e-03 | 1.23e-03 | 27.33% | 38.62% | 0.822 |
| 3.00e-03 | 3.70e-03 | 41.53% | 79.63% | 2.190 |
| 1.00e-02 | 1.23e-02 | 47.08% | 99.52% | 6.789 |

### gaussian
| sigma_abs | rel-to-std | P(boundary out) | P(any swap) | E[symdiff] |
|---|---|---|---|---|
| 1.00e-05 | 1.23e-05 | 0.52% | 0.52% | 0.010 |
| 3.00e-05 | 3.70e-05 | 1.10% | 1.10% | 0.022 |
| 1.00e-04 | 1.23e-04 | 4.18% | 4.40% | 0.088 |
| 2.00e-04 | 2.47e-04 | 8.25% | 8.80% | 0.177 |
| 3.80e-04 | 4.69e-04 | 13.72% **<- headline** | 16.07% | 0.324 |
| 5.00e-04 | 6.17e-04 | 17.55% | 21.25% | 0.431 |
| 1.00e-03 | 1.23e-03 | 26.32% | 37.92% | 0.803 |
| 3.00e-03 | 3.70e-03 | 40.40% | 78.00% | 2.207 |
| 1.00e-02 | 1.23e-02 | 47.08% | 99.62% | 6.742 |

## 3b. Per-score RELATIVE noise (sigma_i = rel x |score_i|)

More physically faithful for TF32 (operand-magnitude-scaled). Headline rel = 4.69e-04.

### skew_normal
| rel | P(boundary out) | E[symdiff] |
|---|---|---|
| 1.00e-05 | 0.33% | 0.007 |
| 1.00e-04 | 2.50% | 0.051 |
| 4.69e-04 | 11.12% **<- headline** | 0.247 |
| 1.00e-03 | 20.07% | 0.512 |
| 1.00e-02 | 45.33% | 4.145 |

### mixture
| rel | P(boundary out) | E[symdiff] |
|---|---|---|
| 1.00e-05 | 0.18% | 0.004 |
| 1.00e-04 | 2.22% | 0.045 |
| 4.69e-04 | 9.68% **<- headline** | 0.214 |
| 1.00e-03 | 18.43% | 0.456 |
| 1.00e-02 | 44.48% | 3.742 |

### simulated_indexer
| rel | P(boundary out) | E[symdiff] |
|---|---|---|
| 1.00e-05 | 0.30% | 0.006 |
| 1.00e-04 | 2.65% | 0.055 |
| 4.69e-04 | 11.12% **<- headline** | 0.258 |
| 1.00e-03 | 18.80% | 0.531 |
| 1.00e-02 | 42.15% | 4.413 |

## 4. Needle at target fp32 rank (does a specific borderline needle drop?)

Plant the needle at rank r (fp32 ordering), apply headline noise, measure drop probability. **r=512 is the cut**: a needle at fp32 rank 512 is exactly the last entry inside top-512; if it drops, the digit's compressed entry is unselected at decode.

### skew_normal
| rank r | abs-noise P(drop) | rel-noise P(drop) |
|---|---|---|
| 505 | <0.01% | <0.01% |
| 506 | <0.01% | <0.01% |
| 507 | <0.01% | <0.01% |
| 508 | 0.02% | <0.01% |
| 509 | 0.03% | 0.02% |
| 510 | 0.30% | 0.17% |
| 511 | 2.08% | 1.22% |
| 512 | 13.98% **<- cut** | 11.00% |
| 513 | 86.12% | 89.13% |
| 514 | 98.02% | 98.63% |
| 515 | 99.55% | 99.85% |
| 516 | 99.92% | 99.98% |
| 517 | 99.98% | 100.00% |
| 518 | 100.00% | 100.00% |
| 519 | 100.00% | 100.00% |
| 520 | 100.00% | 100.00% |

### mixture
| rank r | abs-noise P(drop) | rel-noise P(drop) |
|---|---|---|
| 505 | <0.01% | <0.01% |
| 506 | <0.01% | <0.01% |
| 507 | <0.01% | <0.01% |
| 508 | <0.01% | <0.01% |
| 509 | <0.01% | <0.01% |
| 510 | 0.13% | 0.08% |
| 511 | 1.35% | 0.92% |
| 512 | 11.28% **<- cut** | 9.65% |
| 513 | 88.72% | 90.23% |
| 514 | 98.65% | 99.18% |
| 515 | 99.87% | 99.93% |
| 516 | 100.00% | 100.00% |
| 517 | 100.00% | 100.00% |
| 518 | 100.00% | 100.00% |
| 519 | 100.00% | 100.00% |
| 520 | 100.00% | 100.00% |

### simulated_indexer
| rank r | abs-noise P(drop) | rel-noise P(drop) |
|---|---|---|
| 505 | <0.01% | <0.01% |
| 506 | <0.01% | <0.01% |
| 507 | <0.01% | <0.01% |
| 508 | <0.01% | <0.01% |
| 509 | 0.08% | 0.08% |
| 510 | 0.43% | 0.32% |
| 511 | 1.83% | 1.77% |
| 512 | 14.90% **<- cut** | 10.93% |
| 513 | 85.23% | 88.93% |
| 514 | 98.02% | 98.30% |
| 515 | 99.57% | 99.70% |
| 516 | 99.95% | 99.97% |
| 517 | 100.00% | 100.00% |
| 518 | 100.00% | 100.00% |
| 519 | 99.98% | 100.00% |
| 520 | 100.00% | 100.00% |

## 5. Structured TF32 (real 10-bit-mantissa rounding, simulated indexer)

Deterministic TF32 truncation (NOT Gaussian) on the simulated indexer. This is the physical bound: the actual residual the kernel introduces.

- absolute residual |s_tf32 - s_fp32|: p50 1.92e-04, p90 5.29e-04, max 2.98e-03
- worst-**row** relative residual (max|d_i| / row-std): p50 1.54e-03, p90 2.08e-03, max 3.68e-03  (cf. boot self-test worst_row_rel=4.691e-4)
- boundary gap p50: 8.47e-04
- **P(boundary entry out under real TF32) = 9.40%**
- E[symdiff] = 0.213

## Verdict

- Typical boundary gap (rank 512/513) ~ **8.68e-04** (median across generators).
- Headline TF32 noise magnitude ~ **3.80e-04** (= 4.691e-4 x score_std).
- Ratio (gap / noise) ~ **2.28**. The boundary gap is within a small constant factor of the noise.
- P(boundary entry flips) under Gaussian: abs-noise ~ 13.12%, per-score-relative ~ 10.64%.
- P(boundary entry flips) under **real structured TF32** (sim indexer, per-tensor fp8 proxy): **9.40%**. NB the simulated fp8 quantization is coarser than the kernel's per-token scale, so its residual (p90 2.08e-03) is ~4.4x the measured 4.691e-4; the real-kernel structured flip rate is likely correspondingly LOWER (corrected est. ~2.12%).
- P(needle planted exactly at fp32 rank 512 is dropped) under abs noise ~ 13.39%; at rank 510 ~ 0.29%.

**TF32 is plausibly strong enough to flip a borderline rank-512 entry at decode.** The boundary gap (~1e-3) is only ~2.3x the 4.691e-4 residual, and a needle sitting at the rank-512 cut drops ~13.39% per decode step under TF32-scale perturbation (and is essentially safe, <0.1%, at rank <=510). So if the digit's compressed entry rides the rank-512 boundary at the digit-emitting decode steps, TF32 can drop it. The `FORCE_TORCH=1` boot is **likely to help** (worth running). Caveat: this is necessary-but-not-sufficient — the digit must actually sit near rank 512 at those steps (T4 found it sits at rank 0..917 in prefill; decode rank is unmeasured), and a single dropped step is often recoverable. Expect a *partial* improvement, not a guaranteed full fix. NB: Gaussian i.i.d. noise is an upper-ish bound; the structured-TF32 number is the tighter, physically-grounded one and points the same way (corrected ~2.12%).

> NB: Gaussian i.i.d. noise is an *upper-ish* bound on flip probability; the structured-TF32 section (5) is the tighter, physically-grounded number and is the one to weight most heavily.

---
Generated by `tools/tf32_flip_montecarlo.py`. Pure CPU (numpy). No GPU, no production impact.