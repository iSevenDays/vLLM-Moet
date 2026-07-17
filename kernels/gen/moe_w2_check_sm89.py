#!/usr/bin/env python3
"""Op-level validation of the Ada (sm_89) Triton moe_w2_mm port.

The EXACT reference math and verdict of moe_w2_check.py (2-bit planes +
block-32 UE8M0 vs f32 ref; worst_rel < 2.5e-2 AND RUNS byte-identical
outputs), with the SM120 ctypes/cubin launch replaced by the Triton
kernel kernels/triton/moe_w2_sm89.py. Descs are built as torch CUDA
tensors carrying the same 6 x u64 {a, as, b, bs, c, m_rows} per pair.

Env: K (512|1024|2048|4096|6144|7168), N, E (pairs), M (<=16), RUNS.
Run on an sm_89 GPU: for K in 512 1024 2048 4096 6144 7168; do
    K=$K python3 kernels/gen/moe_w2_check_sm89.py || break; done
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "triton"))
from moe_w2_sm89 import make_launcher  # noqa: E402

N = int(os.environ.get("N", "4096"))
K = int(os.environ.get("K", "4096"))
E = int(os.environ.get("E", "5"))
M = int(os.environ.get("M", "4"))
RUNS = int(os.environ.get("RUNS", "4"))
torch.manual_seed(int(os.environ.get("SEED", "7")))

LEVELS = torch.tensor([-4.0, -1.0, 1.0, 4.0])


def pack_fragment_major(codes):
    n, k = codes.shape
    c = codes.view(n // 16, 2, 8, k // 64, 2, 2, 4, 4)
    c = c.permute(0, 3, 2, 6, 1, 4, 5, 7).contiguous().view(-1, 4).to(torch.int32)
    return (c[:, 0] | (c[:, 1] << 2) | (c[:, 2] << 4) | (c[:, 3] << 6)).to(torch.uint8)


def pack_scales(s):
    n, ks = s.shape
    return s.view(n // 16, 16, ks).transpose(1, 2).contiguous().flatten()


def quant_a32(a):
    m, k = a.shape
    ab = a.view(m, k // 32, 32)
    a_s = (ab.abs().amax(-1).clamp_min(1e-10) / 448.0)
    a8 = (ab / a_s[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).view(m, k)
    deq = a8.float() * a_s.repeat_interleave(32, 1)
    return a8, a_s, deq


assert torch.cuda.is_available(), "needs a CUDA device (sm_89)"
cap = torch.cuda.get_device_capability()
assert cap == (8, 9), f"this shim validates the Ada port; device is sm_{cap[0]}{cap[1]}"
dev = "cuda"
launch = make_launcher(K)

descs = torch.zeros(E, 6, dtype=torch.int64)
refs, d_cs, keep = [], [], []
for e in range(E):
    codes = torch.randint(0, 4, (N, K), dtype=torch.uint8)
    sexp = torch.randint(120, 132, (N, K // 32), dtype=torch.uint8)  # e8m0 ~1.0
    a8, a_s, a_deq = quant_a32(torch.randn(M, K) * 0.5)

    w_deq = LEVELS[codes.long()] * torch.exp2(sexp.float() - 127.0).repeat_interleave(32, 1)
    ref = a_deq @ w_deq.T
    refs.append(ref)

    d_a = a8.to(dev)
    d_as = a_s.float().to(dev)
    d_b = pack_fragment_major(codes).to(dev)
    d_bs = pack_scales(sexp).to(dev)
    d_c = torch.zeros(M, N, dtype=torch.bfloat16, device=dev)
    d_cs.append(d_c)
    keep += [d_a, d_as, d_b, d_bs]
    descs[e] = torch.tensor([d_a.data_ptr(), d_as.data_ptr(), d_b.data_ptr(),
                             d_bs.data_ptr(), d_c.data_ptr(), M])

d_desc = descs.to(dev)

outs, worst = [], 0.0
for r in range(RUNS):
    for d_c in d_cs:
        d_c.zero_()
    launch(d_desc, N, E)
    torch.cuda.synchronize()
    blob = b""
    for e, d_c in enumerate(d_cs):
        raw = d_c.cpu()
        blob += raw.view(torch.uint16).numpy().tobytes()
        got = raw.float()
        rel = (got - refs[e]).abs().max().item() / refs[e].abs().max().item()
        worst = max(worst, rel)
    outs.append(blob)

ok = worst < 2.5e-2 and len(set(outs)) == 1
print(f"moe_w2 sm89 N={N} K={K} E={E} M={M}: worst_rel={worst:.3e} "
      f"distinct={len(set(outs))}")
print(f"RESULT: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
