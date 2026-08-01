#!/usr/bin/env python3
"""CPU-only regression for persistent base/exact GPU-pool heat metadata."""

import json
import os
import tempfile
import threading

import torch


with tempfile.TemporaryDirectory(prefix="moe-w2-pool-heat-") as heat_dir:
    # Globals are intentionally read at module import time in production.
    os.environ["VLLM_MOE_W2_POOL_HEAT"] = "1"
    os.environ["VLLM_MOE_W2_POOL_HEAT_DIR"] = heat_dir

    from vllm.model_executor.layers.quantization.utils import moe_w2_delta

    assert moe_w2_delta._pool_heat_enabled_for("base")
    assert moe_w2_delta._pool_heat_enabled_for("w8x")
    assert not moe_w2_delta._pool_heat_enabled_for("w8")
    assert not moe_w2_delta._pool_heat_enabled_for("delta")

    def stub(tag: str, slot_bytes: int = 64):
        tier = object.__new__(moe_w2_delta.DeltaTier)
        tier._tag = tag
        tier.n_layers = 2
        tier.E = 3
        tier.slot_bytes = slot_bytes
        tier._lock = threading.Lock()
        tier._owner_li = torch.tensor([0, 1, -1], dtype=torch.int32)
        tier._owner_ei = torch.tensor([1, 2, -1], dtype=torch.int32)
        tier._freq = torch.tensor([[0.0, 2.0, 0.0],
                                   [0.0, 0.0, 7.0]])
        tier._heat_pending = None
        tier._heat_preloaded = True
        return tier

    exact = stub("w8x")
    exact._dump_pool_heat()
    path = exact._heat_path()
    assert os.path.exists(path), path
    with open(path) as f:
        snapshot = json.load(f)
    assert snapshot["meta"] == exact._heat_meta()
    assert snapshot["owners"] == [[1, 2], [0, 1]], snapshot
    assert exact._read_heat_file() == [(1, 2), (0, 1)]

    # Shape/config drift must invalidate an otherwise correctly named file.
    mismatch = stub("w8x", slot_bytes=128)
    assert mismatch._read_heat_file() is None

    # The legacy base tag remains supported and gets a distinct rank file.
    base = stub("base")
    base._dump_pool_heat()
    assert os.path.exists(base._heat_path())
    assert base._heat_path() != path

print("MOE_W2_POOL_HEAT PASS tags=base,w8x mismatch=invalidated")
