"""Offline VRAM predictor for the FP4 delta tier on sm89 (2x48GB).
Calibrated to the MEASURED boot data point:
  batch=1024, MAXLEN=32768, UTIL=0.98, RESERVE=0:
    Available KV = 2.77 GiB, free_after_KV = 1.62 GiB, pool=256 slots(1.5GiB),
    warmup OOM (512 MiB needed, only 0.12 GiB free after pool).

Model: vLLM fills KV greedily, so free_after_KV is ~constant (~1.62 GiB) regardless
of batch. The FIT condition is: warmup_transient(batch) <= free_after_KV - pool_size.
warmup_transient scales linearly with max_num_batched_tokens (0.5 GiB at 1024).
"""
# Calibrated constants (GiB)
TOTAL_VRAM = 47.37
UTIL = 0.98
BUDGET = TOTAL_VRAM * UTIL  # 46.42
BASE_PLANES = 36.3
NON_MOE = 5.0
FREE_AFTER_KV = 1.62        # measured (vLLM leaves this for the pool)
WARMUP_AT_1024 = 0.50       # measured (the OOM was 512 MiB)
POOL_256_SLOTS = 1.50       # 256 * 6 MiB (one full layer union = ~100% coverage)

def warmup(batch):
    return WARMUP_AT_1024 * (batch / 1024)

def fits(batch, pool=POOL_256_SLOTS, free=FREE_AFTER_KV, margin_extra=0.0):
    need = pool + warmup(batch) + margin_extra
    return need <= free, free - need

print("=== FP4 delta VRAM fit predictor (256 slots = ~100% coverage) ===")
print(f"Calibrated: free_after_KV={FREE_AFTER_KV} GiB, pool(256)={POOL_256_SLOTS} GiB")
print(f"Post-pool headroom = {FREE_AFTER_KV - POOL_256_SLOTS:.3f} GiB for the warmup transient\n")
print(f"{'batch':>6} {'warmup':>8} {'pool+warmup':>12} {'free':>6} {'fits?':>6} {'margin':>8}")
print("-"*56)
for batch in [1024, 512, 384, 256, 192, 160, 128, 96]:
    w = warmup(batch)
    ok, margin = fits(batch)
    print(f"{batch:>6} {w:>7.3f}G {POOL_256_SLOTS+w:>11.3f}G {FREE_AFTER_KV:>5.2f}G {('YES' if ok else 'NO'):>6} {margin:>+7.3f}G")

print("\n=== H2D prefill cost (16K context, per layer = 256 experts x 6MiB FP4 promote) ===")
print("(measured: 98s for 16K at batch=1024 => ~6.1s/chunk; scales with chunk count)")
CHUNKS_16K = lambda batch: 16384 / batch
for batch in [1024, 512, 256, 192, 128]:
    chunks = CHUNKS_16K(batch)
    t = 6.1 * chunks  # seconds, rough
    print(f"  batch={batch:>4}: {chunks:>5.0f} chunks x 6.1s = {t:>6.0f}s = {t/60:.1f} min for 16K prefill")

print("\n=== RECOMMENDED decisive-boot config ===")
# pick the largest batch that fits with >=0.05 GiB margin (fastest prefill that's safe)
import math
best = None
for batch in range(1024, 64, -8):
    ok, margin = fits(batch, margin_extra=0.05)
    if ok:
        best = batch; break
print(f"  Largest safe batch (>=0.05G margin): BATCHED_TOKENS={best}")
print(f"  => pool 256 slots (~100% FP4), warmup {warmup(best):.3f}G, margin {FREE_AFTER_KV-POOL_256_SLOTS-warmup(best):+.3f}G")
print(f"  => 16K prefill ~{6.1*CHUNKS_16K(best)/60:.1f} min")
