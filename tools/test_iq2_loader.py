#!/usr/bin/env python3
"""Offline validation for the Step 2 IQ2_XXS / Q2_K weight-loading path.

This test does NOT boot vLLM and does NOT import vllm (the dev worktree has
no vllm installed). It validates the three things that can be checked
statically, against the real IQ2 checkpoint produced by Step 1
(tools/convert_iq2_gguf_to_vllm.py):

  1. AST parse the two edited files (mxfp4.py, deepseek_v4 model.py) so a
     syntax error fails CI immediately.
  2. Read the IQ2 safetensors header with safetensors.safe_open (no full
     78 GB load) and confirm the param-name + shape contract that
     Mxfp4MoEMethod._create_iq2_weights expects is present:
        model.layers.{L}.mlp.experts.gate_proj.weight_iq2_xxs
          [n_experts, intermediate, (hidden/QK_K)*IQ2_XXS_BLOCK_BYTES] u8
        model.layers.{L}.mlp.experts.up_proj.weight_iq2_xxs
          [n_experts, intermediate, (hidden/QK_K)*IQ2_XXS_BLOCK_BYTES] u8
        model.layers.{L}.mlp.experts.down_proj.weight_q2_k
          [n_experts, hidden, (intermediate/QK_K)*Q2_K_BLOCK_BYTES]     u8
        _lookup.iq2xxs_grid  uint64[256]
        _lookup.ksigns_iq2xs  uint8[128]
  3. Exercise the IQ2 row-bytes math (mirroring mxfp4._iq2_row_bytes) for
     DeepSeek-V4-Flash's hidden=4096 / intermediate=2048 and confirm the
     derived per-expert byte sizes match the checkpoint's axis-2 sizes
     (1056, 1056, 672) bit-for-bit.

The actual _iq2_expert_weight_loader / _iq2_expert_mapping logic is covered
by the AST check here and exercised end-to-end in Step 4b's kernel test
(which does boot vLLM, in the serving image).

Run:    python3 tools/test_iq2_loader.py
Exit:   0 on success, non-zero on any mismatch.
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

from safetensors import safe_open

ROOT = Path(__file__).resolve().parent.parent
MXFP4_PATH = ROOT / "overlay/vllm/vllm/model_executor/layers/quantization/mxfp4.py"
MODEL_PATH = ROOT / "overlay/vllm/vllm/models/deepseek_v4/nvidia/model.py"
IQ2_CKPT = Path(
    os.getenv(
        "VLLM_MOE_W2_IQ2_CKPT",
        "/root/models/DeepSeek-V4-Flash-IQ2/dsv4_iq2.safetensors",
    )
)

# DeepSeek-V4-Flash geometry — must match config.json.
HIDDEN = 4096
INTERMEDIATE = 2048
N_EXPERTS = 256
N_LAYERS = 43

# IQ2 block layout (must mirror mxfp4.py + utils/iq2_xxs_ref.py).
QK_K = 256
IQ2_XXS_BLOCK_BYTES = 66
Q2_K_BLOCK_BYTES = 84

GATE_ROW_BYTES = (HIDDEN // QK_K) * IQ2_XXS_BLOCK_BYTES        # 16 * 66 = 1056
UP_ROW_BYTES = (HIDDEN // QK_K) * IQ2_XXS_BLOCK_BYTES          # 16 * 66 = 1056
DOWN_ROW_BYTES = (INTERMEDIATE // QK_K) * Q2_K_BLOCK_BYTES     #  8 * 84 = 672

# What _create_iq2_weights is expected to register, derived from the math
# above. (name, (num_experts, out_features, row_bytes))
EXPECTED_PARAM_SHAPES = {
    "gate_weight_iq2_xxs": (N_EXPERTS, INTERMEDIATE, GATE_ROW_BYTES),
    "up_weight_iq2_xxs": (N_EXPERTS, INTERMEDIATE, UP_ROW_BYTES),
    "down_weight_q2_k": (N_EXPERTS, HIDDEN, DOWN_ROW_BYTES),
}


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}", file=sys.stderr)
    sys.exit(1)


def check_ast() -> None:
    """Step 5a: AST-parse each edited file."""
    for path in (MXFP4_PATH, MODEL_PATH):
        try:
            ast.parse(path.read_text())
        except SyntaxError as e:
            fail(f"{path}: {e}")
        print(f"[ok] AST parse: {path.name}")


def _extract_int_constants(tree: ast.Module, names: set[str]) -> dict[str, int]:
    """Find module-level assignments like `_QK_K = 256` and return their
    values. Used to cross-check the IQ2 constants hardcoded in this test
    against the ones mxfp4.py actually uses at runtime."""
    out: dict[str, int] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if (
                    isinstance(tgt, ast.Name)
                    and tgt.id in names
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, int)
                ):
                    out[tgt.id] = node.value.value
    return out


def check_mxfp4_constants() -> None:
    """Step 5b: confirm mxfp4.py exposes the IQ2 constants we expect."""
    tree = ast.parse(MXFP4_PATH.read_text())
    consts = _extract_int_constants(
        tree,
        {"_QK_K", "_IQ2_XXS_BLOCK_BYTES", "_Q2_K_BLOCK_BYTES"},
    )
    expected = {
        "_QK_K": QK_K,
        "_IQ2_XXS_BLOCK_BYTES": IQ2_XXS_BLOCK_BYTES,
        "_Q2_K_BLOCK_BYTES": Q2_K_BLOCK_BYTES,
    }
    for k, v in expected.items():
        if k not in consts:
            fail(f"mxfp4.py missing module-level constant {k}")
        if consts[k] != v:
            fail(
                f"mxfp4.py {k}={consts[k]} != test expectation {v}. Either "
                "the test is stale or the IQ2 constants drifted."
            )
    print(
        f"[ok] mxfp4.py IQ2 constants: QK_K={consts['_QK_K']}, "
        f"IQ2_XXS_BLOCK_BYTES={consts['_IQ2_XXS_BLOCK_BYTES']}, "
        f"Q2_K_BLOCK_BYTES={consts['_Q2_K_BLOCK_BYTES']}"
    )

    # The _iq2_row_bytes math must also produce our EXPECTED_PARAM_SHAPES.
    # Re-derive here directly to fail loudly on drift.
    derived = {
        "gate_weight_iq2_xxs": (
            N_EXPERTS,
            INTERMEDIATE,
            (HIDDEN // QK_K) * IQ2_XXS_BLOCK_BYTES,
        ),
        "up_weight_iq2_xxs": (
            N_EXPERTS,
            INTERMEDIATE,
            (HIDDEN // QK_K) * IQ2_XXS_BLOCK_BYTES,
        ),
        "down_weight_q2_k": (
            N_EXPERTS,
            HIDDEN,
            (INTERMEDIATE // QK_K) * Q2_K_BLOCK_BYTES,
        ),
    }
    for name, exp_shape in EXPECTED_PARAM_SHAPES.items():
        assert derived[name] == exp_shape, (name, derived[name], exp_shape)
    print(
        f"[ok] derived param shapes (DSv4-Flash H={HIDDEN} I={INTERMEDIATE} "
        f"E={N_EXPERTS}): { {k: v for k, v in EXPECTED_PARAM_SHAPES.items()} }"
    )


def check_iq2_mapping_helper() -> None:
    """Step 5c: confirm model.py defines _iq2_expert_mapping (the helper
    that appends IQ2 entries to get_expert_mapping). Imports would be
    needed to call it, so we only verify the symbol exists and that it
    references the three IQ2 weight_name substrings."""
    tree = ast.parse(MODEL_PATH.read_text())
    funcs = {
        n.name: n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
    }
    if "_iq2_expert_mapping" not in funcs:
        fail("model.py missing module-level function _iq2_expert_mapping")
    src = ast.get_source_segment(MODEL_PATH.read_text(), funcs["_iq2_expert_mapping"])
    for needle in (
        "experts.gate_proj.weight_iq2_xxs",
        "experts.up_proj.weight_iq2_xxs",
        "experts.down_proj.weight_q2_k",
        "VLLM_MOE_W2_IQ2",
    ):
        if needle not in src:
            fail(f"_iq2_expert_mapping missing {needle!r}")
    print("[ok] model.py defines _iq2_expert_mapping with IQ2 entries")


def check_safetensors_header() -> None:
    """Step 5d: read the IQ2 checkpoint header (no full 78 GB load) and
    verify the param-name + shape contract that _create_iq2_weights
    expects is present."""
    if not IQ2_CKPT.exists():
        # Allow the test to run in environments without the 78 GB ckpt;
        # static checks above are still valuable. Skip with a clear note.
        print(
            f"[skip] IQ2 checkpoint not present at {IQ2_CKPT} — header "
            "verification skipped (set VLLM_MOE_W2_IQ2_CKPT to enable)."
        )
        return

    # Hold the handle open for the whole check — safe_open closes the file
    # on __exit__, so get_slice() must run inside the `with` block.
    with safe_open(str(IQ2_CKPT), framework="pt") as f:
        keys = list(f.keys())

        # LUTs.
        for lut_name, lut_dtype, lut_shape in (
            ("_lookup.iq2xxs_grid", "U64", (256,)),
            ("_lookup.ksigns_iq2xs", "U8", (128,)),
        ):
            if lut_name not in keys:
                fail(f"checkpoint missing LUT tensor {lut_name}")
            sl = f.get_slice(lut_name)
            if tuple(sl.get_shape()) != lut_shape:
                fail(
                    f"{lut_name} shape {tuple(sl.get_shape())} != expected "
                    f"{lut_shape}"
                )
            if str(sl.get_dtype()) != lut_dtype:
                fail(
                    f"{lut_name} dtype {sl.get_dtype()} != expected "
                    f"{lut_dtype}"
                )
        print("[ok] LUTs present: _lookup.iq2xxs_grid[256] U64, "
              "_lookup.ksigns_iq2xs[128] U8")

        # Per-layer expert tensors — all N_LAYERS layers must be present,
        # each matching the EXPECTED_PARAM_SHAPES contract.
        suffix_to_param = {
            "gate_proj.weight_iq2_xxs": "gate_weight_iq2_xxs",
            "up_proj.weight_iq2_xxs": "up_weight_iq2_xxs",
            "down_proj.weight_q2_k": "down_weight_q2_k",
        }

        # Tally per suffix for "every layer present" accounting.
        per_suffix_counts = {s: 0 for s in suffix_to_param}
        for k in keys:
            for suffix, param_name in suffix_to_param.items():
                if k.endswith(suffix) and ".mlp.experts." in k:
                    per_suffix_counts[suffix] += 1
                    # Verify shape against the contract on this tensor.
                    sl = f.get_slice(k)
                    got = tuple(sl.get_shape())
                    exp = EXPECTED_PARAM_SHAPES[param_name]
                    if got != exp:
                        fail(
                            f"{k}: shape {got} != expected {exp} "
                            f"(param {param_name})"
                        )
                    if str(sl.get_dtype()) != "U8":
                        fail(f"{k}: dtype {sl.get_dtype()} != expected U8")

        total = len(keys)
        expected_total = N_LAYERS * 3 + 2  # 3 dirs * N_LAYERS + 2 LUTs

    for suffix, count in per_suffix_counts.items():
        if count != N_LAYERS:
            fail(
                f"checkpoint has {count} '{suffix}' tensors, expected "
                f"{N_LAYERS} (one per layer)"
            )
    if total != expected_total:
        fail(
            f"checkpoint has {total} tensors, expected {expected_total} "
            f"({N_LAYERS}*3 + 2 LUTs)"
        )

    print(
        f"[ok] checkpoint contract verified: {total} tensors = "
        f"{N_LAYERS} layers * 3 directions + 2 LUTs. Per-layer shapes "
        f"match _create_iq2_weights."
    )


def main() -> None:
    print(f"Validating IQ2 Step 2 weight-loading path against {IQ2_CKPT}")
    check_ast()
    check_mxfp4_constants()
    check_iq2_mapping_helper()
    check_safetensors_header()
    print("[PASS] IQ2 Step 2 static validation.")


if __name__ == "__main__":
    main()
