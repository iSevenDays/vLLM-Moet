#!/usr/bin/env python3
"""Regression tests for the host-pack loader-skip identity gate."""

import importlib.util
import json
import os
import tempfile


spec = importlib.util.spec_from_file_location(
    "moe_w2_store",
    "overlay/vllm/vllm/model_executor/layers/quantization/utils/moe_w2_store.py",
)
store = importlib.util.module_from_spec(spec)
spec.loader.exec_module(store)

store._rank_suffix = lambda: "rank0of2"
store._ckpt_id = lambda: "checkpoint-a"
store._quantizer_id = lambda: {
    "version": "tsym4-fragmajor-v2",
    "zero_mode": "auto",
    "scale_refit": True,
}

with tempfile.TemporaryDirectory() as tmp:
    os.environ["VLLM_MOE_W2_STORE_DIR"] = tmp
    path = os.path.join(tmp, "base.rank0of2.json")
    stride = 4096
    exact = {
        "version": store._PACK_VERSION,
        "E": 256,
        "n_layers": 43,
        "slot_bytes": 1000,
        "stride": stride,
        "ckpt_id": "checkpoint-a",
        "quantizer_id": store._quantizer_id(),
        "layers": [0, 1],
    }

    with open(path, "w") as handle:
        json.dump(exact, handle)
    assert store.pack_has_layer("base", 1, 43, 256, 1000)

    for field, bad_value in (
        ("ckpt_id", "checkpoint-b"),
        ("n_layers", 44),
        ("quantizer_id", {"scale_refit": False}),
    ):
        stale = dict(exact)
        stale[field] = bad_value
        with open(path, "w") as handle:
            json.dump(stale, handle)
        assert not store.pack_has_layer("base", 1, 43, 256, 1000), field

print("host-pack loader-skip identity gate: PASS")
