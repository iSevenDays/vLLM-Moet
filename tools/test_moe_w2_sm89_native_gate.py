#!/usr/bin/env python3
"""Full op-gate for the sm_89-native decode W2 cubin vs the validated Triton
emulation, across both decode shapes, many seeds, and adversarial activations.

The sm_89-specific suspect from docs/dsv4-sm89-longcontext/runlogs/dig_moe_expert_path.md:
``kernels/cuda/moe_w2_sm89_decode.cu`` is hand-written SASS using Ada's native
e4m3 tensor core (``mma.sync.m16n8k32.f32.e4m3.e4m3``) with in-register PRMT
decode of the 2-bit codes. Its own header carries a TODO: "Run
kernels/gen/moe_w2_sm89_native_check.py ... before integrating" -- the FULL
op-gate was apparently never run; only a single-seed boot self-test with a LOOSE
2.5e-2 gate. A systematic bias on digit-critical activations (outlier channels,
heavy cancellation) could hide behind that single random case. This harness
closes that gap.

Two complementary halves:

  PART A -- invoke the EXISTING op-gate (kernels/gen/moe_w2_sm89_native_check.py)
            as a subprocess: the cuobjdump static gate (FP8 tensor-core SASS
            present, symbols resolve) + the cubin-vs-fp32-numpy-reference
            correctness/perf gate, for BOTH native decode shapes
            (K,N) in {(1024,4096),(4096,2048)}. Its built-in gate is loose
            (max-relative 2e-2 / rmse 1e-2); we surface its raw numbers.

  PART B -- cubin vs Triton-emulation A/B on IDENTICAL torch descriptors via the
            PRODUCTION dispatch (moe_w2_cubit._launch_sm89_native runs the cubin
            in torch's own context -- no context-mixing). For each shape x seed
            x adversarial activation pattern, the SAME 6-field descriptor
            {a, as, b, bs, c, m_rows} is run through both moe_w2_mm_sm89 (the
            self-tested Triton emulation, VLLM_MOE_W2_SM89_NATIVE=0 path) and the
            native cubin (VLLM_MOE_W2_SM89_NATIVE=1 path). The diagnostic metric
            is max|cubin - emulation| / max|ref|: BOTH outputs are bf16-rounded,
            so bf16 rounding cancels and any excess over the bf16 floor is a
            REAL cubin-vs-emulation divergence.

Verdict gate: the bf16 output rounding floor is ~2.4e-3 (the emulation self-test
reports ~2.6e-3 vs the fp32 reference). If the cubin tracks the emulation within
that floor on every shape/pattern/seed, the cubin is exonerated; if any case
exceeds it, the cubin corrupts decode expert output and the production falsifier
``VLLM_MOE_W2_SM89_NATIVE=0`` (fall back to Triton emulation) is the one-boot fix.

Run inside the serving image (mount the repo so the cubin + op-gate are visible,
and point the loader at the repo cubin):

  docker run --rm --gpus '"device=0"' --entrypoint python3 \
    -v $PWD:/workspace:ro -w /workspace \
    -e VLLM_MOE_W2_SM89_NATIVE_CUBIN=/workspace/kernels/cubins-sm89/moe_w2_sm89_decode.cubin \
    vllm-moet-sm89:v0251 /workspace/tools/test_moe_w2_sm89_native_gate.py

Blockers you may need to adjust:
  * Cubin present? The loader defaults to $VLLM_MOE_W2_SM89_NATIVE_CUBIN or
    /cubit-share/moe_w2_sm89_decode.cubin. The repo cubin is at
    kernels/cubins-sm89/moe_w2_sm89_decode.cubin -- the -e above points there.
    Rebuild with kernels/cuda/build_moe_w2_sm89_decode.sh if stale.
  * If a live server holds the GPU in EXCLUSIVE-PROCESS mode, the PART A
    subprocess (own CUDA context) cannot init -- stop the server or use a free
    device. PART B shares torch's context and coexists with a server.
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys

import torch

# --- The two decode shapes the native cubin serves (moe_w2_cubit._SM89_NATIVE_SHAPES)
SHAPES = [(1024, 4096), (4096, 2048)]
BF16_FLOOR = 2.4e-3          # ~bf16 output rounding (emulation self-test ~2.6e-3)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_CUBIN = os.path.join(
    REPO_ROOT, "kernels", "cubins-sm89", "moe_w2_sm89_decode.cubin")
OP_GATE = os.path.join(REPO_ROOT, "kernels", "gen", "moe_w2_sm89_native_check.py")

LEVELS = torch.tensor([-4.0, -1.0, 1.0, 4.0], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Plane packing + activation quantization (verbatim moe_w2_check_sm89.py golden
# math -- copied because that module EXECUTES at import time, so it cannot be
# imported as a library).
# ---------------------------------------------------------------------------
def pack_fragment_major(codes: torch.Tensor) -> torch.Tensor:
    n, k = codes.shape
    c = (codes.view(n // 16, 2, 8, k // 64, 2, 2, 4, 4)
         .permute(0, 3, 2, 6, 1, 4, 5, 7)
         .contiguous().view(-1, 4).to(torch.int32))
    return (c[:, 0] | (c[:, 1] << 2) | (c[:, 2] << 4) | (c[:, 3] << 6)).to(torch.uint8)


def pack_scales(s: torch.Tensor) -> torch.Tensor:
    n, ks = s.shape
    return s.view(n // 16, 16, ks).transpose(1, 2).contiguous().flatten()


def quant_a32(a: torch.Tensor):
    """fp32 [M,K] -> (a8 e4m3 [M,K], a_s f32 [M,K//32], a_deq f32 [M,K])."""
    m, k = a.shape
    ab = a.view(m, k // 32, 32)
    a_s = ab.abs().amax(-1).clamp_min(1e-10) / 448.0
    a8 = (ab / a_s[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).view(m, k)
    deq = a8.float() * a_s.repeat_interleave(32, 1)
    return a8, a_s.float(), deq


# ---------------------------------------------------------------------------
# Adversarial activation generators (return fp32 [M,K] on CPU).
# ---------------------------------------------------------------------------
def act_random(m, k, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(m, k, generator=g) * 0.5


def act_outlier(m, k, seed):
    """Most channels tiny, a few channels huge -> inflates the per-32 group
    scale and quashes the other 31 fp8 values (a systematic-bias magnet)."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(m, k, generator=g) * 0.02
    n_out = max(1, k // 50)                       # ~2% of channels
    chan = torch.randperm(k, generator=g)[:n_out]
    a[:, chan] = torch.randn(m, n_out, generator=g) * 50.0
    return a


def act_all_equal(m, k, seed):
    """Constant activation -> identical products, stresses accumulation order."""
    _ = seed
    return torch.full((m, k), 0.5)


def act_sign_alternating(m, k, seed):
    """Alternating sign per channel -> heavy cancellation against uniform-sign
    weight codes (relative error amplifies near zero)."""
    g = torch.Generator().manual_seed(seed)
    mag = (torch.randn(m, k, generator=g) * 0.5).abs() + 0.25
    sign = torch.where(torch.arange(k) % 2 == 0, 1.0, -1.0)
    return mag * sign


def act_max_magnitude(m, k, seed):
    """Near-fp8-saturating magnitude -> exercises the UE8M0 scale fold at the
    top of the dynamic range."""
    g = torch.Generator().manual_seed(seed)
    sign = torch.sign(torch.randn(m, k, generator=g))
    sign[sign == 0] = 1.0
    return sign * 440.0


PATTERNS = {
    "random": act_random,
    "outlier": act_outlier,
    "all_equal": act_all_equal,
    "sign_alt": act_sign_alternating,
    "max_mag": act_max_magnitude,
}


# ---------------------------------------------------------------------------
# Descriptor builder (6-field {a, as, b, bs, c, m_rows} -- the cubin/emulation ABI)
# ---------------------------------------------------------------------------
def build_case(k: int, n: int, m_rows: int, pattern: str, seed: int, dev):
    """Returns (desc [1,6] int64 cuda, out_buf [m,n] bf16 cuda, ref [m,n] f32 cpu,
    keepalive list of the device tensors desc points into)."""
    a_f32 = PATTERNS[pattern](m_rows, k, seed)
    a8, a_s, a_deq = quant_a32(a_f32)

    g = torch.Generator().manual_seed(10_000 + seed)
    codes = torch.randint(0, 4, (n, k), dtype=torch.uint8, generator=g)
    sexp = torch.randint(120, 132, (n, k // 32), dtype=torch.uint8, generator=g)
    w_deq = (LEVELS[codes.long()]
             * torch.exp2(sexp.float() - 127.0).repeat_interleave(32, 1))
    ref = a_deq @ w_deq.T                            # [m, n] fp32

    d_a = a8.to(dev)
    d_as = a_s.to(dev)
    d_b = pack_fragment_major(codes).to(dev)
    d_bs = pack_scales(sexp).to(dev)
    out = torch.zeros(m_rows, n, dtype=torch.bfloat16, device=dev)
    desc = torch.tensor([[d_a.data_ptr(), d_as.data_ptr(), d_b.data_ptr(),
                          d_bs.data_ptr(), out.data_ptr(), m_rows]],
                        dtype=torch.int64, device=dev)
    keep = [d_a, d_as, d_b, d_bs, out]               # hold refs while kernel runs
    return desc, out, ref, keep


# ---------------------------------------------------------------------------
# PART A: invoke the existing op-gate (isolated subprocess, own CUDA context)
# ---------------------------------------------------------------------------
def run_op_gate(cubin_path: str) -> bool:
    print("=== PART A: existing op-gate (kernels/gen/moe_w2_sm89_native_check.py) ===")
    if not os.path.isfile(OP_GATE):
        print(f"  SKIP: op-gate not found at {OP_GATE}")
        return False
    if not os.path.isfile(cubin_path):
        print(f"  SKIP: cubin not found at {cubin_path}")
        return False

    # Static inspect first (cuobjdump, no CUDA init) -- safe even with server up.
    print("  [1/2] static inspect (--inspect-only):")
    try:
        out = subprocess.run(
            ["python3", OP_GATE, "--cubin", cubin_path, "--inspect-only"],
            check=False, capture_output=True, text=True, cwd=REPO_ROOT)
        print("    " + out.stdout.strip().replace("\n", "\n    "))
        if out.returncode != 0:
            print(f"    STATIC GATE FAILED (exit {out.returncode})")
            print("    " + out.stderr.strip().replace("\n", "\n    "))
            return False
    except FileNotFoundError:
        print("    SKIP: python3/cuobjdump unavailable in this image")
        return False

    # Full driver-API correctness/perf gate for both shapes.
    print("  [2/2] driver-API correctness + perf gate (both shapes, m=4):")
    py = os.environ.get("PYTHON", "python3")
    try:
        out = subprocess.run(
            [py, OP_GATE, "--cubin", cubin_path, "--kernel", "both",
             "--m", "4", "--pairs", "1", "--warmup", "5", "--runs", "20",
             "--seed", "1234"],
            check=False, capture_output=True, text=True, cwd=REPO_ROOT)
    except FileNotFoundError:
        print("    SKIP: interpreter unavailable")
        return False
    print("    " + out.stdout.strip().replace("\n", "\n    "))
    if out.returncode != 0:
        print(f"    OP-GATE FAILED (exit {out.returncode})")
        print("    " + out.stderr.strip().replace("\n", "\n    "))
    return out.returncode == 0


# ---------------------------------------------------------------------------
# PART B: cubin vs Triton emulation A/B via the production dispatch
# ---------------------------------------------------------------------------
def _stream_handle() -> ctypes.c_void_p:
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


def _load_native(dev) -> "dict[tuple[int,int], ctypes.c_void_p] | None":
    """Drive the production loader so the cubin is registered in torch's own
    CUDA context (same context the Triton emulation JITs into -- no context
    mixing). Returns the {(K,N): fn} map, or None if the cubin didn't load."""
    cubin = os.environ.get("VLLM_MOE_W2_SM89_NATIVE_CUBIN")
    if not cubin and os.path.isfile(DEFAULT_CUBIN):
        cubin = DEFAULT_CUBIN
        os.environ["VLLM_MOE_W2_SM89_NATIVE_CUBIN"] = cubin
    # Native path must be ON for the loader to register the cubin.
    os.environ.setdefault("VLLM_MOE_W2_SM89_NATIVE", "1")

    from vllm.model_executor.layers.quantization.utils import moe_w2_cubit
    # _ensure_ready also JITs the Triton emulation + runs the boot parity gates.
    moe_w2_cubit._ensure_ready()
    fns = dict(moe_w2_cubit._native_fns)
    if not fns:
        print("  cubin did NOT register (_native_fns empty). Cubin absent at "
              f"{os.environ.get('VLLM_MOE_W2_SM89_NATIVE_CUBIN')} or its boot "
              "parity gate already FAILED -- that itself is a finding.")
        return None
    return fns


def _rel_diff(a: torch.Tensor, b: torch.Tensor, ref: torch.Tensor) -> float:
    denom = max(float(ref.abs().max().item()), 1.0)
    return float((a - b).abs().max().item()) / denom


def run_ab(dev) -> dict:
    print("\n=== PART B: cubin vs Triton emulation A/B (production dispatch) ===")
    from vllm.model_executor.layers.quantization.utils import moe_w2_sm89
    from vllm.model_executor.layers.quantization.utils import moe_w2_cubit

    nat = _load_native(dev)
    if nat is None:
        return {"ok": False}

    cap = torch.cuda.get_device_capability(dev)
    print(f"  device sm_{cap[0]}{cap[1]}; native cubin ACTIVE for {sorted(nat)}")

    worst_nat_emu = 0.0
    worst_nat_ref = 0.0
    worst_emu_ref = 0.0
    worst_case = None
    seeds = [0, 1, 2, 3]
    m_rows_list = [1, 4]
    cases = 0
    print(f"  {'shape':>11} {'m':>2} {'pattern':>9} {'nat-emu':>9} "
          f"{'nat-ref':>9} {'emu-ref':>9}  verdict")
    print("  " + "-" * 70)
    for (k, n) in SHAPES:
        emu_launch = moe_w2_sm89.make_launcher(k)
        nat_fn = nat.get((k, n))
        if nat_fn is None:
            print(f"  ({k},{n}): cubin NOT registered for this shape -- SKIP")
            continue
        for m_rows in m_rows_list:
            for pattern in PATTERNS:
                for seed in seeds:
                    desc, out_emu, ref, keep = build_case(
                        k, n, m_rows, pattern, seed, dev)
                    # --- Triton emulation (VLLM_MOE_W2_SM89_NATIVE=0 path)
                    emu_launch(desc, n, 1)
                    # --- native cubin (VLLM_MOE_W2_SM89_NATIVE=1 path) into a
                    #     fresh buffer via a second descriptor (identical inputs)
                    out_nat = torch.zeros(m_rows, n, dtype=torch.bfloat16,
                                          device=dev)
                    desc_nat = desc.clone()
                    desc_nat[0, 4] = out_nat.data_ptr()
                    keep += [out_nat, desc_nat]
                    moe_w2_cubit._launch_sm89_native(
                        nat_fn, desc_nat, n, 1, _stream_handle())
                    torch.cuda.synchronize()

                    got_emu = out_emu.float()
                    got_nat = out_nat.float()
                    ne = _rel_diff(got_nat, got_emu, ref)
                    nr = _rel_diff(got_nat, ref, ref)
                    er = _rel_diff(got_emu, ref, ref)
                    cases += 1
                    if ne > worst_nat_emu:
                        worst_nat_emu, worst_nat_ref, worst_emu_ref = ne, nr, er
                        worst_case = f"(K={k},N={n}) m={m_rows} {pattern} seed={seed}"
                    flag = "OVER-FLOOR" if ne > BF16_FLOOR else "ok"
                    print(f"  ({k:>4},{n:>4}) {m_rows:>2} {pattern:>9} "
                          f"{ne:>9.3e} {nr:>9.3e} {er:>9.3e}  {flag}")
    print(f"\n  cases run: {cases}")
    print(f"  worst |cubin-emulation|/|ref| = {worst_nat_emu:.3e} "
          f"(bf16 floor {BF16_FLOOR:.1e})")
    print(f"  worst |cubin-reference|/|ref| = {worst_nat_ref:.3e}")
    print(f"  worst |emulation-reference|/|ref| = {worst_emu_ref:.3e}")
    if worst_case:
        print(f"  worst cubin-vs-emulation case: {worst_case}")
    return {"ok": True, "worst_nat_emu": worst_nat_emu,
            "worst_nat_ref": worst_nat_ref, "worst_emu_ref": worst_emu_ref,
            "cases": cases, "worst_case": worst_case}


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("needs an sm_89 CUDA device")
    dev = torch.device("cuda", torch.cuda.current_device())
    cap = torch.cuda.get_device_capability(dev)
    print(f"device: sm_{cap[0]}{cap[1]} {torch.cuda.get_device_name(dev)} "
          f"(torch {torch.__version__})")
    if cap != (8, 9):
        print(f"WARN: expected sm_89; the native decode cubin is sm_89-only. "
              f"Continuing on sm_{cap[0]}{cap[1]} may fail at cubin load.")

    cubin = os.environ.get("VLLM_MOE_W2_SM89_NATIVE_CUBIN", DEFAULT_CUBIN)
    gate_ok = run_op_gate(cubin)
    ab = run_ab(dev)

    print("\n=== VERDICT ===")
    if not ab.get("ok"):
        print("INCONCLUSIVE: native cubin did not register; PART B skipped. "
              "Fix the cubin path (VLLM_MOE_W2_SM89_NATIVE_CUBIN) or rebuild "
              "via kernels/cuda/build_moe_w2_sm89_decode.sh and rerun. PART A's "
              "verdict above is the existing op-gate's.")
        return 3
    worst = ab["worst_nat_emu"]
    over = worst > BF16_FLOOR
    if over:
        print(f"OVER-FLOOR: cubin diverges from the validated Triton emulation "
              f"by {worst:.3e} > bf16 floor {BF16_FLOOR:.1e} on at least one "
              "shape/pattern/seed.")
        print(f"  worst case: {ab['worst_case']}")
        print("  -> The native decode W2 cubin corrupts decode expert output "
              "under that activation. Production falsifier: set "
              "VLLM_MOE_W2_SM89_NATIVE=0 (decode W2 falls back to the "
              "self-tested Triton emulation). Rebuild the cubin and rerun.")
        return 2
    print(f"PASS: cubin tracks the Triton emulation within {worst:.3e} "
          f"(<= bf16 floor {BF16_FLOOR:.1e}) across all {ab['cases']} "
          "shape/pattern/seed cases.")
    print("  -> The sm_89-native decode W2 cubin is exonerated on these "
          "activations; the MoE expert value path is faithful to the "
          "emulation. If the end-to-end digit failure persists, the value "
          "mechanism is elsewhere (revisit dig_value_readout.md / the "
          "decode-SELECTION TF32 hypothesis).")
    if not gate_ok:
        print("  NOTE: PART A (existing op-gate) did not pass cleanly -- "
              "see its output above. PART B is the tighter gate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
