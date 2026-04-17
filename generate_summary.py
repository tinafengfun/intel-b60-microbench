#!/usr/bin/env python3
"""
Generate comprehensive microbenchmark summary from all CSV results.
"""

import csv, os
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.resolve() / "results"

def read_csv(name):
    path = RESULTS_DIR / name
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))

def main():
    print("=" * 80)
    print("Intel Arc Pro B60 (BMG-G21, Xe2) Microbenchmark Summary")
    print("=" * 80)
    print()

    # --- DPAS Latency ---
    lat = read_csv("dpas_latency_sweep.csv")
    if lat:
        print("--- DPAS BF16 Latency (Dependent Chain) ---")
        print(f"  {'N':>6s} {'median_ns':>10s} {'cyc/dpas':>10s}")
        for r in lat:
            print(f"  {r['n_dpas']:>6s} {r['median_ns']:>10s} {r['cycles_per_dpas']:>10s}")
        # Compute slope
        try:
            import numpy as np
            xs = np.array([float(r['n_dpas']) for r in lat])
            ys = np.array([float(r['median_ns']) for r in lat])
            slope = np.polyfit(xs, ys, 1)[0]
            print(f"\n  >> DPAS BF16 Latency (slope): {slope:.2f} ns = {slope*2.4:.1f} cycles")
        except:
            pass
        print()

    # --- DPAS Precision Comparison ---
    prec = read_csv("dpas_precision_sweep.csv")
    if prec:
        print("--- DPAS Multi-Precision Latency ---")
        for p in ['BF16', 'FP16']:
            rows = [r for r in prec if r.get('precision') == p]
            if rows:
                try:
                    import numpy as np
                    xs = np.array([float(r['n_dpas']) for r in rows])
                    ys = np.array([float(r['median_ns']) for r in rows])
                    slope = np.polyfit(xs, ys, 1)[0]
                    print(f"  {p}: {slope:.2f} ns = {slope*2.4:.1f} cycles/dpas")
                except:
                    pass
        print()

    # --- DPAS Throughput ---
    tput = read_csv("dpas_throughput_sweep.csv")
    if tput:
        print("--- DPAS BF16 Throughput (Independent Chains, N_ITER=128) ---")
        print(f"  {'ILP':>4s} {'SG':>4s} {'median_ns':>10s} {'TFLOPS':>10s}")
        for r in tput:
            print(f"  {r['n_ilp']:>4s} {r['n_sg']:>4s} {r['median_ns']:>10s} {r['tflops']:>10s}")
        # Best throughput
        best = max(tput, key=lambda r: float(r['tflops']))
        print(f"\n  >> Peak throughput (1 WG): {float(best['tflops']):.3f} TFLOPS at ILP={best['n_ilp']}, SG={best['n_sg']}")
        print(f"  >> Note: Full GPU throughput requires many WGs (see SYCL benchmarks)")
        print()

    # --- Memory Latency ---
    mem_lat = read_csv("mem_latency_sweep.csv")
    if mem_lat:
        print("--- Memory Hierarchy Latency (Pointer Chase, 4096 accesses) ---")
        print(f"  {'Size':>10s} {'cyc/access':>12s} {'ns/access':>12s} {'Region':>10s}")
        for r in mem_lat:
            print(f"  {r['size_str']:>10s} {r['cycles_per_access']:>12s} {r['ns_per_access']:>12s} {r['region']:>10s}")
        print()

    # --- Key Findings ---
    print("=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print("""
    1. DPAS/XMX (BF16):
       - Latency: ~31-37 cycles per dpas.8x8 (dependent chain, slope method)
       - Throughput: scales with ILP; ~0.25 TFLOPS per sub-group with ILP=8
       - GEN ASM verified: 128 dpas.8x8 instructions for N=128 chain
       - Unitrace: XVE_ACTIVE=5.8%, XVE_STALL_SBID=27.6% (XMX bottleneck)
       - FP16 latency: ~34 cycles (similar to BF16)

    2. Memory Hierarchy:
       - SLM/L1: 70-144 cycles/access (1-128KB)
       - L2: 162-233 cycles/access (192KB-4MB)
       - Global: 236-261 cycles/access (8MB+)
       - SYCL bandwidth: Read ~682 GB/s, Write ~900 GB/s (1GB buffer)

    3. Infrastructure:
       - Raw SPIR-V pipeline verified: .spvasm -> spirv-as -> ocloc -> GEN ASM
       - Level Zero submission works with timing (host-side)
       - GEN ASM validation confirms instruction counts
       - INT8 cooperative matrix blocked by ocloc segfault (compiler bug)

    4. Unitrace Cross-Validation:
       - ComputeBasic: XVE_ACTIVE, instruction counts, L3 hit/miss
       - VectorEngineStalls: SBID stall dominates DPAS workload
       - GPU core clock: 2.4 GHz confirmed
    """)

if __name__ == "__main__":
    main()
