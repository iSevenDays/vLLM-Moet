"""Offline observability: does antirez's indexer QAT (Hadamard128 + FP4 e2m1)
produce materially different activations/scoring than our FP8-no-Hadamard path?

This is the cheap test to resolve the agent disagreement (indexer QAT vs MoE weights)
WITHOUT a boot. If the two paths diverge a lot on the scoring distribution, the QAT
matters (port it). If they agree, the QAT is minor (focus on MoE weights).

Indexer head_dim = 128 (DS4_N_INDEXER_HEAD_DIM). Hadamard is 128-wide.
FP4 e2m1 values: {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6} per antirez ds4.c:3231-3235.
"""
import torch

HD = 128  # indexer head dim
torch.manual_seed(0)

# --- Hadamard128 (the orthogonal transform antirez applies before FP4) ---
def hadamard128():
    H = torch.tensor([[1.0]])
    for _ in range(7):  # 2^7 = 128
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / (128 ** 0.5)
H128 = hadamard128()

# --- FP4 e2m1 quant (antirez ds4.c:3231-3235) ---
E2M1_VALS = torch.tensor([0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])  # +mag; sign separately
def quant_fp4_e2m1(x, block=32):
    """FP4 e2m1 with per-32-tile UE8M0 (power-of-two) scale, like antirez."""
    orig = x
    nb = x.shape[-1] // block
    out = torch.empty_like(x)
    for i in range(nb):
        tile = x[..., i*block:(i+1)*block]
        amax = tile.abs().amax(-1, keepdim=True).clamp_min(1e-12)
        scale = torch.pow(2.0, torch.ceil(torch.log2(amax / 6.0)))  # UE8M0, max e2m1 = 6
        q = tile / scale
        mag = E2M1_VALS[torch.argmin((q.unsqueeze(-1).abs() - E2M1_VALS).abs(), dim=-1)]
        sign = torch.sign(q)
        out[..., i*block:(i+1)*block] = (mag * sign) * scale
    return out

# --- FP8 e4m3 quant (our path: no Hadamard) ---
def quant_fp8_e4m3(x, block=32):
    orig = x
    nb = x.shape[-1] // block
    out = torch.empty_like(x)
    for i in range(nb):
        tile = x[..., i*block:(i+1)*block]
        amax = tile.abs().amax(-1, keepdim=True).clamp_min(1e-12)
        scale = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))
        # fp8 e4m3 has ~3-bit mantissa; approximate 8 levels per binade via round to nearest/64 of range
        q = (tile / scale).clamp(-448, 448)
        # simulate e4m3 granularity: 8 levels per binade
        lvl = torch.pow(2.0, torch.floor(torch.log2(q.abs().clamp_min(1e-12))))
        mant_steps = lvl * 0.125  # 8 levels per binade (3-bit mantissa)
        q_q = torch.round(q / mant_steps.clamp_min(1e-12)) * mant_steps.clamp_min(1e-12)
        out[..., i*block:(i+1)*block] = q_q * scale
    return out

print("=== Indexer QAT observability: FP8-no-Hadamard (ours) vs FP4-Hadamard (antirez) ===\n")
print("Measuring: do the two paths produce different Q·K scoring → different top-512 selection?\n")

# Realistic indexer Q and K (128-dim, post-RoPE). Magnitudes O(0.1-1).
N = 4096  # candidate tokens (compressed rows the indexer scores)
Q = (torch.randn(1, HD) * 0.5)         # one query
K = (torch.randn(N, HD) * 0.5)         # N candidate keys
# Make token N//2 the "needle" (aligned with Q)
K[N//2] = Q[0] * 0.9 + (torch.randn(HD) * 0.05)

# Reference scoring (bf16, no quant)
ref_scores = (Q @ K.T)[0]
ref_top512 = set(ref_scores.topk(512).indices.tolist())

# Our path: FP8 e4m3, NO Hadamard
Q_fp8 = quant_fp8_e4m3(Q); K_fp8 = quant_fp8_e4m3(K)
our_scores = (Q_fp8 @ K_fp8.T)[0]
our_top512 = set(our_scores.topk(512).indices.tolist())

# antirez path: Hadamard128 THEN FP4 e2m1
Qh = (H128 @ Q.T).T; Kh = (K @ H128.T)  # apply Hadamard
Q_fp4 = quant_fp4_e2m1(Qh); K_fp4 = quant_fp4_e2m1(Kh)
ant_scores = (Q_fp4 @ K_fp4.T)[0]
ant_top512 = set(ant_scores.topk(512).indices.tolist())

print(f"  Reference (bf16) top-512 contains needle? {'N//2' if N//2 in ref_top512 else 'NO'}: {N//2 in ref_top512}")
print(f"  Ours (FP8 no-Hadamard): needle in top-512? {N//2 in our_top512}   overlap w/ ref: {len(our_top512 & ref_top512)/512:.0%}")
print(f"  antirez (FP4+Hadamard): needle in top-512? {N//2 in ant_top512}   overlap w/ ref: {len(ant_top512 & ref_top512)/512:.0%}")
print(f"  Ours vs antirez top-512 overlap: {len(our_top512 & ant_top512)/512:.0%}  (low = the paths SELECT DIFFERENT TOKENS)")
print()
print("  Score-correlation (Spearman proxy via rank overlap):")
def rank_overlap(a, b, k=512): return len(set(a.topk(k).indices.tolist()) & set(b.topk(k).indices.tolist()))/k
print(f"    ref vs ours:     {rank_overlap(ref_scores, our_scores):.0%}")
print(f"    ref vs antirez:  {rank_overlap(ref_scores, ant_scores):.0%}")
print(f"    ours vs antirez: {rank_overlap(our_scores, ant_scores):.0%}  <-- if low, the QAT materially changes selection")
print()
# Activation distribution divergence (the distribution-shift proxy)
print("  Activation value divergence (would the learned indexer weights see different inputs?):")
print(f"    mean|our_Q - ant_Q|/|Q|:    {(Q_fp8 - Q_fp4).norm()/Q.norm():.1%}")
print(f"    mean|our_K - ant_K|/|K|:    {(K_fp8 - K_fp4).norm()/K.norm():.1%}")
