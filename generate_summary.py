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
                    suffix = ':bf' if p == 'BF16' else ':hf'
                    print(f"  {p}: {slope:.2f} ns = {slope*2.4:.1f} cycles/dpas (GEN ASM suffix: {suffix})")
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
        best = max(tput, key=lambda r: float(r['tflops']))
        print(f"\n  >> Peak throughput (1 WG): {float(best['tflops']):.3f} TFLOPS at ILP={best['n_ilp']}, SG={best['n_sg']}")
        print(f"  >> Note: Full GPU throughput requires many WGs across 20 Xe Cores")
        print()

    # --- Memory Latency ---
    mem_lat = read_csv("mem_latency_sweep.csv")
    if mem_lat:
        print("--- Memory Hierarchy Latency (Pointer Chase, 4096 accesses, L1 data cache) ---")
        print(f"  {'Size':>10s} {'cyc/access':>12s} {'ns/access':>12s} {'Region':>20s}")
        for r in mem_lat:
            region = r.get('region', '?')
            print(f"  {r['size_str']:>10s} {r['cycles_per_access']:>12s} {r['ns_per_access']:>12s} {region:>20s}")
        print()

    # --- SLM Latency ---
    slm_lat = read_csv("slm_latency_sweep.csv")
    if slm_lat:
        print("--- SLM Latency (SPIR-V Native, OpVariable Workgroup, send.slm) ---")
        print(f"  {'Size':>8s} {'cyc/access':>12s} {'ns/access':>12s}")
        for r in slm_lat:
            print(f"  {r['size_str']:>8s} {r['cycles_per_access']:>12s} {r['ns_per_access']:>12s}")
        print()

    # --- Key Findings ---
    print("=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print("""
    1. DPAS/XMX (BF16):
       - Latency: ~33-37 cycles per dpas.8x8 (dependent chain, slope method)
       - Per-DPAS FLOPs: 8×16×16×2 = 4096 (BF16)
       - Latency ≠ reciprocal throughput:
         * Latency = 33 cycles (time to complete)
         * Reciprocal throughput ≈ 16 cycles (from GEMM: 89.77T → ~97.6T peak)
       - FP16 latency: ~34 cycles, GEN ASM uses :hf suffix (not :bf)

    2. Memory Hierarchy (L1 data cache via CrossWorkgroup pointer chase):
       - L1 data cache: 71-145 cycles/access (1-128 KB per Xe Core, send.ugm)
       - L2 cache: 162-236 cycles/access (192KB-8MB, 18 MB shared)
       - Global memory: 247-261 cycles/access (16MB+)

    3. SLM (SPIR-V native, OpVariable Workgroup, send.slm):
       - Native SLM latency: ~46 cycles at 256B (send.slm path)
       - Faster than L1 data cache (71 cycles, send.ugm) — separate hardware path
       - SYCL local_accessor adds ~34 cycles overhead (46→80 cycles)
       - Not as fast as NVIDIA shared memory (~30 cycles, separate SRAM)

    4. Memory Bandwidth (1 GB buffer):
       - Read: ~139 GB/s (plateaus even at 524K threads, 30% of 456 GB/s peak)
       - Write: up to 307 GB/s (67% of peak with 524K threads)
       - Read-write asymmetry (~2.2×) is a hardware characteristic

    5. Full GPU Peak:
       - Derived from GEMM: 89.77 TFLOPS achieved → ~97.6 TFLOPS theoretical peak
       - 89.77/97.6 = 92% utilization
       - Implied reciprocal throughput: ~16 cycles/DPAS/XMX
    """)

if __name__ == "__main__":
    main()
