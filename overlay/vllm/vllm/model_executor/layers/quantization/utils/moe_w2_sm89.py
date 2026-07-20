# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ada (sm_89) Triton port of the cubit ``moe_w2_mm`` 2-bit MoE GEMM.

Ada has FP8 (e4m3) tensor cores but no QMMA / QMMA.SF / FP4, so the
shipped sm_120 SASS cubins cannot run there. This kernel EMULATES the
same GEMM: it reads the identical host-packed planes (2-bit fragment-major
codes, block-32 UE8M0 scale bytes, a32 fp8 activations + f32 per-32
scales, delivered through the unchanged 6-field descriptor ABI
``{a, as, b, bs, c, m_rows}``), decodes the 2-bit codes {-4,-1,1,4} to
BF16 in registers, and runs a standard BF16 MMA accumulating in FP32,
folding BOTH scales as explicit multiplies (Ada's MMA has no
scale-factor operand). Output BF16 rows land exactly where the cubin
wrote them; pairs with ``m_rows == 0`` early-EXIT. The HBM win is
preserved: expert weights are still read at 2 bits/elem.

Design decisions (vs the SM120 cubins):

* Decode target BF16, not FP8. Every quantity entering ``tl.dot`` is
  EXACT in bf16: a8 is e4m3 (3-bit mantissa) and the decoded levels are
  {-4,-1,1,4}. Products are exact in the fp32 accumulator, so the only
  rounding is fp32 accumulation order — comfortably inside the op gate
  (worst_rel < 2.5e-2) and byte-deterministic run-to-run (no atomics,
  fixed schedule).
* Scale fold ONCE PER 32-GROUP on the accumulator: the K loop walks one
  32-group per iteration (one UE8M0 byte per N-row, one f32 A scale per
  M-row), does a [16,32]x[32,16] bf16 dot of the UNSCALED operands, and
  folds ``a_s[m] * 2^(sb[n]-127)`` as an [M,N] outer product. This is
  bit-faithful to the reference math (scales are constant across the
  32-group) and keeps ALU low. 2^(sb-127) is built by exponent-field
  bitcast, not exp2 — exact for every sb.
* Grid contract kept: ``(n_rows // 16, pairs)``, one program per
  (16-row N-tile, pair) — the same geometry ``_launch`` always used, so
  the dispatch stays arch-blind. BLOCK_M = 16 with an ``m < m_rows``
  mask serves BOTH the decode tier (mblock 4) and the prefill tier
  (mblock 16); the SAME kernel is registered under the ``w2`` and
  ``w2mc4`` keys (``_require_kernels`` asserts both).
* AFRAG is NOT ported (the fragment-major-A trick is a QMMA A-fragment
  LDG.128 optimization; it does not transfer). The loader simply leaves
  ``w2mc4afrag`` unregistered and prefill falls back to ``w2mc4``.
* w4 / w4q delta tiers are NOT ported yet: serve with
  ``VLLM_MOE_W2_DELTA=0`` on Ada.

Plane addressing (host packing unchanged — moe_w2_planes.pack_fragment_major
/ pack_scales; layout golden: tools/test_moe_w2_planes.py plane_byte /
expect_byte). Per (nb, kb) 16-row x 64-K block of 256 bytes, the byte at
in-block offset

    g*32 + t*8 + tile*4 + k32*2 + half        (g=row&7, tile=row>>3)

packs, little-endian 2 bits/code, the 4 codes at

    k = kb*64 + k32*32 + half*16 + t*4 + {0,1,2,3}.

The kernel loads the 8 bytes per row of one 32-group (k32 fixed) and
expands them to 32 codes in the byte-major order

    f = 0..31 -> k_in_group = ((f//4)&1)*16 + (f//8)*4 + (f&3)

and gathers the A columns THROUGH THE SAME PERMUTATION, so the dot's
K-axis pairs matching elements while every code byte is read exactly
once. Scale plane: byte for (row16 r, 32-group g) at
``(nb*(K/32) + g)*16 + r`` (pack_scales).

Deployment: this file is the published mirror of
``vllm/model_executor/layers/quantization/utils/moe_w2_sm89.py`` on the
vllm fork branch (which is what the loader imports); keep them
byte-identical. Validation: kernels/gen/moe_w2_check_sm89.py (op gate,
needs an sm_89 GPU) and kernels/gen/moe_w2_sm89_cpu_check.py (CPU golden
of the addressing + numerics, runs anywhere).
"""

import os

import torch
import triton
import triton.language as tl

_NUM_WARPS = int(os.getenv("VLLM_MOE_W2_SM89_WARPS", "2"))


@triton.jit
def moe_w2_mm_sm89(
    desc_ptr,                  # *i64 [pairs, 6] {a, as, b, bs, c, m_rows}
    n_rows,                    # C row stride in bf16 elements (= N)
    K: tl.constexpr,           # GEMM contraction (per-launch, like the cubins)
):
    tl.static_assert(K % 64 == 0)
    pid_n = tl.program_id(0)               # 16-row N tile
    pid_p = tl.program_id(1)               # (expert, token-group) pair

    d = desc_ptr + pid_p * 6
    m_rows = tl.load(d + 5)
    if m_rows == 0:                        # dead pair: early EXIT
        return
    a_ptr = tl.load(d + 0).to(tl.pointer_type(tl.float8e4nv))
    as_ptr = tl.load(d + 1).to(tl.pointer_type(tl.float32))
    b_ptr = tl.load(d + 2).to(tl.pointer_type(tl.uint8))
    bs_ptr = tl.load(d + 3).to(tl.pointer_type(tl.uint8))
    c_ptr = tl.load(d + 4).to(tl.pointer_type(tl.bfloat16))

    offs_m = tl.arange(0, 16)
    mask_m = offs_m < m_rows
    r = tl.arange(0, 16)                   # N-rows within the tile
    # fragment-major in-block byte offset, row part: g*32 + tile*4
    row_off = (r % 8) * 32 + (r // 8) * 4
    # byte-column part for one 32-group: t*8 + half (k32 term added in-loop)
    b_idx = tl.arange(0, 8)
    byte_col = (b_idx // 2) * 8 + (b_idx % 2)
    # byte-major k permutation within the 32-group (see module docstring)
    f = tl.arange(0, 32)
    k_perm = ((f // 4) % 2) * 16 + (f // 8) * 4 + (f % 4)
    shift = (f % 4) * 2

    acc = tl.zeros((16, 16), dtype=tl.float32)
    for g in range(0, K // 32):
        kb = g // 2
        k32 = g % 2
        # ---- A: fp8 e4m3 [16, 32], gathered through k_perm (exact in bf16)
        k_glob = kb * 64 + k32 * 32 + k_perm
        a = tl.load(a_ptr + offs_m[:, None] * K + k_glob[None, :],
                    mask=mask_m[:, None], other=0.0).to(tl.bfloat16)
        # ---- A scale: f32, one per (row, 32-group)
        a_s = tl.load(as_ptr + offs_m * (K // 32) + g,
                      mask=mask_m, other=0.0)
        # ---- B: 8 code bytes per N-row -> 32 codes -> levels {-4,-1,1,4}
        blk = (pid_n * (K // 64) + kb) * 256
        code_bytes = tl.load(
            b_ptr + blk + row_off[:, None] + (byte_col + k32 * 2)[None, :]
        ).to(tl.int32)
        expanded = tl.reshape(
            tl.broadcast_to(code_bytes[:, :, None], (16, 8, 4)), (16, 32))
        codes = (expanded >> shift[None, :]) & 3
        mag = tl.where((codes == 0) | (codes == 3), 4.0, 1.0)
        w = tl.where(codes < 2, -mag, mag).to(tl.bfloat16)
        # ---- B scale: UE8M0 byte -> 2^(sb-127), exact via exponent bitcast
        sb = tl.load(bs_ptr + (pid_n * (K // 32) + g) * 16 + r).to(tl.int32)
        w_s = (sb << 23).to(tl.float32, bitcast=True)      # 2^(sb-127), sb>=1
        # sb=0 -> 2^-127 is an f32 SUBNORMAL: correct per UE8M0, but an
        # FTZ multiply would flush it to 0. Real packs never carry sb=0
        # (an all-zero 32-group is degenerate); noted for completeness.
        w_s = tl.where(sb == 0, 5.877471754111438e-39, w_s)  # sb=0 subnormal
        # ---- unscaled bf16 dot (exact products), fp32 accumulate,
        #      then fold both per-32 scales as one [M, N] outer product
        acc += tl.dot(a, tl.trans(w)) * (a_s[:, None] * w_s[None, :])

    offs_n = pid_n * 16 + r
    tl.store(c_ptr + offs_m[:, None] * n_rows + offs_n[None, :],
             acc.to(tl.bfloat16), mask=mask_m[:, None])


def make_launcher(K: int):
    """Launcher matching moe_w2_cubit._launch's arch-blind call shape:
    fn(desc, n_rows, pairs). Grid = (n_rows//16, pairs) — the cubin
    geometry. Runs on the current torch stream (the same stream _launch
    passes to cuLaunchKernel), so ordering vs the desc build and the
    epilogue is unchanged and CUDA-graph capture works as before."""

    def launch(desc: torch.Tensor, n_rows: int, pairs: int) -> None:
        moe_w2_mm_sm89[(n_rows // 16, pairs)](
            desc, n_rows, K=K, num_warps=_NUM_WARPS)

    return launch


def make_launchers(ks) -> dict:
    """{(tier, K): launcher} for every K, for the two mandatory tiers.
    One kernel serves both: decode (mblock 4) and prefill MC4 (mblock 16)
    differ only in m_rows, which the kernel masks per pair."""
    fns = {}
    for k in ks:
        fn = make_launcher(k)
        fns[("w2", k)] = fn
        fns[("w2mc4", k)] = fn
    return fns


def self_test(k: int = 512, n_rows: int = 32, m_rows: int = 4) -> float:
    """Tiny op-level GEMM vs a torch reference on the CURRENT device.

    Startup observability for the emulation path: one JIT compile plus
    <1 ms of GPU time at engine init buys a loud, attributable verdict
    line — a Triton codegen/driver regression fails HERE with a clear
    message instead of surfacing as silent output corruption a deploy
    cycle later. Reference math and the 2.5e-2 gate are verbatim
    kernels/gen/moe_w2_check.py; per-K compiles still happen lazily at
    first forward (a per-K compile failure raises loudly there).
    Returns worst_rel; raises RuntimeError on failure."""
    dev = torch.device("cuda")
    g = torch.Generator().manual_seed(7)
    codes = torch.randint(0, 4, (n_rows, k), dtype=torch.uint8, generator=g)
    sexp = torch.randint(120, 132, (n_rows, k // 32), dtype=torch.uint8,
                         generator=g)
    a = torch.randn(m_rows, k, generator=g) * 0.5
    ab = a.view(m_rows, k // 32, 32)
    a_s = ab.abs().amax(-1).clamp_min(1e-10) / 448.0
    a8 = (ab / a_s[..., None]).clamp(-448, 448).to(
        torch.float8_e4m3fn).view(m_rows, k)
    levels = torch.tensor([-4.0, -1.0, 1.0, 4.0])
    w = levels[codes.long()] * torch.exp2(
        sexp.float() - 127.0).repeat_interleave(32, 1)
    ref = (a8.float() * a_s.repeat_interleave(32, 1)) @ w.T
    # host packers, layout verbatim moe_w2_planes/moe_w2_check.py
    c = codes.view(n_rows // 16, 2, 8, k // 64, 2, 2, 4, 4)
    c = c.permute(0, 3, 2, 6, 1, 4, 5, 7).contiguous().view(-1, 4).to(
        torch.int32)
    plane = (c[:, 0] | (c[:, 1] << 2) | (c[:, 2] << 4)
             | (c[:, 3] << 6)).to(torch.uint8)
    sb = sexp.view(n_rows // 16, 16, k // 32).transpose(
        1, 2).contiguous().flatten()
    d_a, d_as = a8.to(dev), a_s.float().contiguous().to(dev)
    d_b, d_bs = plane.to(dev), sb.to(dev)
    d_c = torch.zeros(m_rows, n_rows, dtype=torch.bfloat16, device=dev)
    desc = torch.tensor([[d_a.data_ptr(), d_as.data_ptr(), d_b.data_ptr(),
                          d_bs.data_ptr(), d_c.data_ptr(), m_rows]],
                        dtype=torch.int64, device=dev)
    make_launcher(k)(desc, n_rows, 1)
    torch.cuda.synchronize()
    got = d_c.float().cpu()
    finite = bool(torch.isfinite(got).all())
    worst = (got - ref).abs().max().item() / ref.abs().max().item()
    if not finite or worst >= 2.5e-2:
        raise RuntimeError(
            f"moe_w2_sm89 self-test FAILED: worst_rel={worst:.3e} "
            f"(gate 2.5e-2), finite={finite} — the Triton emulation is "
            "miscompiling on this device/driver; run "
            "kernels/gen/moe_w2_check_sm89.py for the full op gate and "
            "report triton/torch/driver versions")
    return worst
