#!/usr/bin/env python3
"""HW validation of the QMMA.SF sfa mapping probe (moe_sfa_probe).

The probe runs one QMMA.16832 with A=B=1.0, sfb neutral, and
sfa(thread t) = bytes [64+t, 112+t, 160+t, 208+t] (b0..b3). Every C
element is then 32 * 2^(v-127) with v identifying exactly which
(thread, byte position) the HW consumed for that row:

  v in [64, 96)   -> byte 0 of thread v-64
  v in [112, 144) -> byte 1 of thread v-112
  v in [160, 192) -> byte 2 of thread v-160
  v in [208, 240) -> byte 3 of thread v-208

Prints the row -> (thread, byte) map and checks it against the CUTLASS
hypothesis row(t) = (t&1)*8 + (t>>2) reading the LOW byte.

Env: CUBIN (default /tmp/moe_sfa_probe.cubin), RUNS.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from culaunch import Cuda  # noqa: E402

CUBIN = os.environ.get("CUBIN", "/tmp/moe_sfa_probe.cubin")
RUNS = int(os.environ.get("RUNS", "4"))

cu = Cuda()
fn = cu.load_kernel(CUBIN, "moe_sfa_probe")
d_out = cu.alloc(32 * 16)

outs = []
for _ in range(RUNS):
    cu.memset32(d_out, 0, 32 * 4)
    cu.launch(fn, (1, 1, 1), (32, 1, 1), [d_out])
    cu.synchronize()
    outs.append(cu.from_device(d_out, 32 * 16, dtype=np.float32).copy())
det = all(np.array_equal(outs[0], o) for o in outs)
c = outs[-1].reshape(32, 4)                      # [lane, c0..c3]

BASES = {0: 64, 1: 112, 2: 160, 3: 208}

row_map = {}                                     # row -> set of (thr, byte)
bad = []
for lane in range(32):
    g, sub = lane >> 2, lane & 3
    for f in range(4):
        row = g * 1 + (8 if f >= 2 else 0)
        col = sub * 2 + (f & 1)
        val = float(c[lane, f])
        if val <= 0:
            bad.append((lane, f, val, "nonpos"))
            continue
        v = 127 + math.log2(val / 32.0)
        vr = round(v)
        if abs(v - vr) > 1e-6:
            bad.append((lane, f, val, f"v={v:.3f} not integral"))
            continue
        hit = None
        for b, base in BASES.items():
            if base <= vr < base + 32:
                hit = (vr - base, b)
        if hit is None:
            bad.append((lane, f, val, f"v={vr} outside probe ranges"))
            continue
        row_map.setdefault(row, set()).add(hit)

print(f"deterministic={det}")
for row in range(16):
    print(f"  row {row:2d}: sfa from {sorted(row_map.get(row, set()))}")
if bad:
    for b in bad[:8]:
        print("  BAD:", b)

# hypothesis: LOW byte (byte 0) of thread t with row(t) = (t&1)*8 + (t>>2);
# redundant threads t/t+2 -> accept either (or both reported identically:
# each row must resolve to exactly ONE (thread, byte) observation and it
# must be byte 0 of a thread whose formula-row matches).
ok = det and not bad
for row in range(16):
    obs = row_map.get(row, set())
    if len(obs) != 1:
        ok = False
        continue
    thr, byte = next(iter(obs))
    if byte != 0 or (thr & 1) * 8 + (thr >> 2) != row:
        ok = False

print("hypothesis row(t)=(t&1)*8+(t>>2), LOW byte:",
      "CONFIRMED" if ok else "REJECTED (see map above)")
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
