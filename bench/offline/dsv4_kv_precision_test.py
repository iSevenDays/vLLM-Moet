"""Offline quantification of fp8_ds_mla KV precision loss vs bf16 (CPU, no GPU/server).

Uses the ACTUAL pack function from triton_sparse_mla_dsv4.py and replicates the
kernel's dequant math. Measures round-trip error on realistic MLA latents."""
import torch, sys

# Replicate the constants + pack logic from triton_sparse_mla_dsv4.py
_D = 512; _D_NOPE = 448; _D_ROPE = 64
_QUANT_TILE = 64; _NUM_SCALES = 7; _PBS = 64

def pack_dsv4(kv_rows):
    """fp8_ds_mla pack: UE8M0 power-of-two scales + fp8 e4m3 nope + bf16 rope."""
    nope = kv_rows[:, :_D_NOPE].reshape(-1, _NUM_SCALES, _QUANT_TILE)
    amax = nope.abs().amax(dim=-1).clamp_min(2.0**-126)
    exponent = torch.ceil(torch.log2(amax / 448.0))
    scale_byte = (exponent + 127).clamp(0, 254).to(torch.uint8)
    inv_scale = torch.exp2(-exponent)
    quant = (nope * inv_scale[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    rope = kv_rows[:, _D_NOPE:].to(torch.bfloat16)
    return quant, scale_byte, rope

def dequant_dsv4(quant, scale_byte, rope):
    """Replicate the kernel's gather dequant: fp8 * exp2(scale_byte-127), rope bf16."""
    scale = torch.exp2(scale_byte.to(torch.float32) - 127.0)
    nope_f32 = quant.to(torch.float32) * scale[..., None]
    nope_f32 = nope_f32.reshape(quant.shape[0], _D_NOPE)
    return torch.cat([nope_f32, rope.to(torch.float32)], dim=-1)

def roundtrip_error(latent):
    """Measure fp8_ds_mla round-trip error vs bf16 (lossless) reference."""
    quant, scale_byte, rope = pack_dsv4(latent)
    recon = dequant_dsv4(quant, scale_byte, rope)
    orig = latent.to(torch.float32)
    err = (recon - orig)
    rel = (err.abs() / orig.abs().clamp_min(1e-6)).mean().item()
    mae = err.abs().mean().item()
    # "digit discrimination": if two latents differ by 1 unit (e.g., a code digit),
    # does fp8 round them to the same value? Measure the smallest distinguishable diff.
    return rel, mae, recon, orig

torch.manual_seed(0)
print("=== fp8_ds_mla KV precision loss vs bf16 (MLA latent, 512 dims) ===\n")

# Realistic MLA latent magnitudes (after qnorm, values are typically O(0.01-1.0))
# but the nope part carries the semantic content. Test a range of scales.
for desc, scale in [("typical (O(0.1))", 0.1), ("moderate (O(1.0))", 1.0),
                    ("large (O(10))", 10.0), ("small (O(0.01))", 0.01)]:
    latent = (torch.randn(2048, _D) * scale).to(torch.bfloat16)
    rel, mae, recon, orig = roundtrip_error(latent)
    # How often does fp8 round to EXACTLY equal (total info loss)?
    exact_match = (recon == orig).float().mean().item()
    # Worst-case per-tile dynamic-range waste (UE8M0 rounds up to next pow2)
    print(f"  {desc:20s}: mean_rel_err={rel:6.1%}  mae={mae:.4f}  exact_match={exact_match:5.1%}")

print("\n=== KEY: relative error per 64-dim tile (UE8M0 power-of-two waste) ===")
latent = (torch.randn(2048, _D) * 0.5).to(torch.bfloat16)
quant, scale_byte, rope = pack_dsv4(latent)
scale = torch.exp2(scale_byte.to(torch.float32) - 127.0)
# The 'proper' scale would be amax/448; UE8M0 uses next-pow2(amax/448). Show the waste.
nope = latent[:, :_D_NOPE].reshape(-1, _NUM_SCALES, _QUANT_TILE)
amax = nope.abs().amax(dim=-1).clamp_min(2.0**-126)
proper_scale = amax / 448.0
waste = (scale - proper_scale) / proper_scale  # how much bigger the UE8M0 scale is
print(f"  UE8M0 scale overestimate: mean={waste.mean():.2f}x  median={waste.median():.2f}x  p90={waste.quantile(0.9):.2f}x")
print(f"  (A 2.0x overestimate means HALF the fp8 dynamic range is wasted in that tile)")
print(f"  Tiles wasting >=1.5x range: {(waste>=0.5).float().mean():.1%}")

# Critical: does fp8 preserve enough precision to distinguish nearby code-digit latents?
print("\n=== Digit-discrimination test (can fp8 KV distinguish nearby needles?) ===")
base = (torch.randn(1, _D) * 0.5).to(torch.bfloat16)
perturbed = base.clone()
perturbed[0, 200] += 0.3  # a small perturbation (like a different code digit)
_,_,_,orig0 = roundtrip_error(base)
_,_,_,orig1 = roundtrip_error(perturbed)
true_dist = (orig0 - orig1).norm().item()
print(f"  True latent distance (bf16): {true_dist:.4f}")
qb,_ = pack_dsv4(base); qp,_ = pack_dsv4(perturbed)
rb = dequant_dsv4(qb, _, torch.zeros(1,_D_ROPE,dtype=torch.bfloat16) if False else base[:,_D_NOPE:].to(torch.bfloat16))
rp = dequant_dsv4(qp, _, perturbed[:,_D_NOPE:].to(torch.bfloat16))
fp8_dist = (rb-rp).norm().item()
print(f"  fp8_ds_mla distance:        {fp8_dist:.4f}  ({fp8_dist/true_dist:.0%} of true)")
print(f"  → If fp8_dist << true_dist, fp8 CANNOT distinguish the two needles (precision lost)")
