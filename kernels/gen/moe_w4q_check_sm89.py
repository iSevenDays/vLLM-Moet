#!/usr/bin/env python3
"""Op-level validation of the Ada (sm_89) Triton moe_w4q_mm port.

Desc ABI (64B/pair): {a, as, base, ref, bs, c, m_rows, pad}.
Env: K, N, E, M, RUNS, BENCH.
"""
import os
import sys
import importlib.util

import torch

try:
    from vllm.model_executor.layers.quantization.utils.moe_w2_sm89 import (
        make_w4q_launcher)
except ImportError:
    root = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    path = os.path.join(root, "overlay", "vllm", "vllm", "model_executor",
                        "layers", "quantization", "utils", "moe_w2_sm89.py")
    spec = importlib.util.spec_from_file_location("moe_w2_sm89_overlay", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    make_w4q_launcher = mod.make_w4q_launcher

N = int(os.environ.get("N", "4096"))
K = int(os.environ.get("K", "4096"))
E = int(os.environ.get("E", "5"))
M = int(os.environ.get("M", "4"))
RUNS = int(os.environ.get("RUNS", "4"))
BENCH = os.environ.get("BENCH", "0").lower() not in ("0", "false", "no", "off")
WARMUP = int(os.environ.get("WARMUP", "5"))
assert 1 <= M <= 16
torch.manual_seed(int(os.environ.get("SEED", "3")))

E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6] * 2)
E2M1[8:] *= -1
NIBBLE_TO_CODE = torch.tensor([2] * 5 + [3] * 3 + [1] * 5 + [0] * 3,
                              dtype=torch.uint8)


def pack_fragment_major(codes):
    n, k = codes.shape
    c = codes.view(n // 16, 2, 8, k // 64, 2, 2, 4, 4)
    c = c.permute(0, 3, 2, 6, 1, 4, 5, 7).contiguous().view(-1, 4).to(torch.int32)
    return (c[:, 0] | (c[:, 1] << 2) | (c[:, 2] << 4) | (c[:, 3] << 6)).to(torch.uint8)


def pack_quintal_fragment_major(nibs):
    mag = (nibs & 7).to(torch.int64)
    code = NIBBLE_TO_CODE[nibs.long()]
    big = (code == 0) | (code == 3)
    digit = torch.where(big, mag - 5, mag)
    n, k = nibs.shape
    d = digit.view(n // 16, 2, 8, k // 64, 2, 2, 4, 4)
    d = d.permute(0, 3, 2, 6, 1, 4, 5, 7).contiguous().view(-1, 8, 4)
    words = d[..., 0] + 5 * d[..., 1] + 25 * d[..., 2] + 125 * d[..., 3]
    shorts = torch.zeros(words.shape[0], 5, dtype=torch.int64)
    for w in range(8):
        bit = 10 * w
        s, off = bit // 16, bit % 16
        shorts[:, s] |= words[:, w] << off
        if off > 6:
            shorts[:, s + 1] |= words[:, w] >> (16 - off)
    shorts &= 0xFFFF
    by = torch.stack([shorts & 0xFF, shorts >> 8], dim=-1).view(-1, 10)
    by = by.view(-1, 32, 10).to(torch.uint8)
    return torch.cat([by[:, :, :8].reshape(-1, 256),
                      by[:, :, 8:].reshape(-1, 64)], dim=1).flatten()


def pack_scales(s):
    n, ks = s.shape
    return s.view(n // 16, 16, ks).transpose(1, 2).contiguous().flatten()


def quant_a32(a):
    m, k = a.shape
    ab = a.view(m, k // 32, 32)
    a_s = ab.abs().amax(-1).clamp_min(1e-10) / 448.0
    a8 = (ab / a_s[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).view(m, k)
    deq = a8.float() * a_s.repeat_interleave(32, 1)
    return a8, a_s, deq


assert torch.cuda.is_available(), "needs a CUDA device"
cap = torch.cuda.get_device_capability()
assert cap == (8, 9), f"this shim validates the Ada port; device is sm_{cap[0]}{cap[1]}"
dev = "cuda"
launch = make_w4q_launcher(K)

descs = torch.zeros(E, 8, dtype=torch.int64)
refs, d_cs, keep = [], [], []
for e in range(E):
    nibs = torch.randint(0, 16, (N, K), dtype=torch.uint8)
    nibs.view(-1)[:16] = torch.arange(16, dtype=torch.uint8)
    sexp = torch.randint(120, 132, (N, K // 32), dtype=torch.uint8)
    a8, a_s, a_deq = quant_a32(torch.randn(M, K) * 0.5)
    code = NIBBLE_TO_CODE[nibs.long()]
    scale = torch.exp2(sexp.float() - 127.0).repeat_interleave(32, 1)
    refs.append(a_deq @ (E2M1[nibs.long()] * scale).T)

    d_a = a8.to(dev)
    d_as = a_s.float().to(dev)
    d_base = pack_fragment_major(code).to(dev)
    d_ref = pack_quintal_fragment_major(nibs).to(dev)
    d_bs = pack_scales(sexp).to(dev)
    d_c = torch.zeros(M, N, dtype=torch.bfloat16, device=dev)
    d_cs.append(d_c)
    keep += [d_a, d_as, d_base, d_ref, d_bs]
    descs[e] = torch.tensor([d_a.data_ptr(), d_as.data_ptr(),
                             d_base.data_ptr(), d_ref.data_ptr(),
                             d_bs.data_ptr(), d_c.data_ptr(), M, 0])

d_desc = descs.to(dev)
if BENCH:
    for _ in range(WARMUP):
        for d_c in d_cs:
            d_c.zero_()
        launch(d_desc, N, E)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(RUNS):
        for d_c in d_cs:
            d_c.zero_()
        launch(d_desc, N, E)
    end.record()
    torch.cuda.synchronize()
    print(f"bench_ms_per_launch={start.elapsed_time(end) / RUNS:.4f} "
          f"runs={RUNS} warmup={WARMUP}")

outs, worst = [], 0.0
for _ in range(RUNS):
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
print(f"moe_w4q sm89 N={N} K={K} E={E} M={M}: worst_rel={worst:.3e} "
      f"distinct={len(set(outs))} (reference is TRUE e2m1)")
print(f"RESULT: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
