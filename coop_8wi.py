#!/usr/bin/env python3
"""Coop-matrix DPAS at high occupancy: does it reach 16-cyc issue like ESIMD?"""
import sys
from pathlib import Path
SCRIPT_DIR = Path("/home/intel/b70-microbench")
sys.path.insert(0, str(SCRIPT_DIR))
from run_b70_xmx_true import gen_kernel_live, run_cfg
from run_b70_xmx_probe import build_and_compile, cleanup

GHZ = 2.4
for ilp in [2, 4, 6]:
    spv = build_and_compile(gen_kernel_live(ilp, 16384), f"hocc_ilp{ilp}")
    if not spv:
        print(f"ILP={ilp}: BUILD FAILED"); continue
    for wg_x, n_wg, tag in [(16, 32, "2WI/EU"), (16, 64, "4WI/EU"), (16, 128, "8WI/EU")]:
        res, err = run_cfg(spv, wg_x, n_wg, ilp, 16384, repeats=7)
        if res is None:
            print(f"ILP={ilp} {tag}: RUN FAILED {err}"); continue
        median, cyc, tf, sgs = res
        # per-EU instruction interval: cyc_per_thread_instr / threads_per_EU
        tpe = {32: 2, 64: 4, 128: 8}[n_wg]
        print(f"ILP={ilp} {tag} n_wg={n_wg}: med={median:10.0f}ns  {tf:6.1f} TF  "
              f"{cyc:5.2f} cyc/dpas/thread  -> {cyc/tpe:5.2f} cyc/instr/EU")
    cleanup(f"hocc_ilp{ilp}")
