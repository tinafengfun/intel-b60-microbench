#!/usr/bin/env python3
import sys, struct, subprocess
from pathlib import Path
SCRIPT_DIR = Path("/home/intel/b70-microbench")
sys.path.insert(0, str(SCRIPT_DIR))
from run_b70_xmx_true import gen_kernel_live
from run_b70_xmx_probe import build_and_compile, cleanup

spv = build_and_compile(gen_kernel_live(1, 1), "numprobe")
r = subprocess.run(f"{SCRIPT_DIR}/probe_runner {spv} dpas_ilp 16 1 1", shell=True,
                   capture_output=True, text=True, cwd=SCRIPT_DIR)
print(r.stdout[:200], r.stderr[:200])
lines = [l for l in r.stdout.splitlines() if "e+" in l or "e-" in l]
got = [float(x) for x in lines[:128]]
print("read", len(got), "floats")

def fill_f32(j):  # matches runner: 1.001f + j*0.001f (float32 arithmetic)
    import numpy as np
    return np.float32(1.001) + np.float32(j) * np.float32(0.001)

import numpy as np
def bf16(u16):
    return struct.unpack("f", struct.pack("I", int(u16) << 16))[0]

def buf_ushorts(bufidx, n):
    out = []
    for j in range((n + 1) // 2):
        b = struct.pack("f", fill_f32(j))
        lo, hi = struct.unpack("HH", b)
        out.append(lo); out.append(hi)
    return out[:n]

A = np.array([bf16(x) for x in buf_ushorts(0, 128)], dtype=np.float64).reshape(8, 16)
B = np.array([bf16(x) for x in buf_ushorts(1, 256)], dtype=np.float64).reshape(16, 16)
C0 = np.array([fill_f32(j) for j in range(128)], dtype=np.float64).reshape(8, 16)
exp = (A @ B + C0).ravel()
got = np.array(got)
err = np.abs(got - exp)
rel = err / (np.abs(exp) + 1e-6)
print("max abs err:", err.max(), " max rel err:", rel.max())
print("sample got:", got[:4])
print("sample exp:", exp[:4])
print("MATCH" if rel.max() < 1e-2 else "MISMATCH -> lowering is NOT a plain collective 8x16x16 mad")
cleanup("numprobe")
