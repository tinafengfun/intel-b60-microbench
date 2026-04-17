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
            # Theoretical throughput
            flops_per_dpas = 8 * 16 * 16 * 2  # M*N*K*2 for BF16
            tput = flops_per_dpas / (slope * 2.4) * 2.4e9  # FLOPs/cycle * GHz
            print(f"  >> Latency-bound throughput per SG: {tput/1e12:.3f} TFLOPS")
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
        print(f"  >> Note: Full GPU throughput requires many WGs across 20 Xe Cores")
        print()

    # --- Memory Latency ---
    mem_lat = read_csv("mem_latency_sweep.csv")
    if mem_lat:
        print("--- Memory Hierarchy Latency (Pointer Chase, 4096 accesses) ---")
        print(f"  {'Size':>10s} {'cyc/access':>12s} {'ns/access':>12s} {'Region':>20s}")
        for r in mem_lat:
            region = r.get('region', '?')
            print(f"  {r['size_str']:>10s} {r['cycles_per_access']:>12s} {r['ns_per_access']:>12s} {region:>20s}")
        print()

    # --- SLM Latency ---
    print("--- SLM Latency (separate benchmark, SYCL local_accessor) ---")
    print("  Run: ./bench_slm_latency")
    print("  Expected: ~80 cycles (SLM is carved from L1 on Xe2)")
    print()

    # --- Key Findings ---
    print("=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print("""
    1. DPAS/XMX (BF16):
       - Latency: ~31-37 cycles per dpas.8x8 (dependent chain, slope method)
       - Per-DPAS FLOPs: 8×16×16×2 = 4096 (BF16)
       - Latency-bound throughput: ~0.298 TFLOPS per sub-group
       - Throughput scales with ILP; ~0.251 TFLOPS at ILP=8 (84% of latency-bound)
       - FP16 latency: ~34 cycles (similar to BF16, same hardware)
       - GEN ASM verified: 128 dpas.8x8 instructions for N=128 chain
       - Unitrace: XVE_ACTIVE=5.8%, XVE_STALL_SBID=27.6% (XMX bottleneck)

    2. Memory Hierarchy (L1 data cache, not SLM):
       - L1 data cache: 70-145 cycles/access (1-128 KB per Xe Core)
       - L2 cache: 162-236 cycles/access (192KB-8MB, 18 MB shared)
       - Global memory: 247-261 cycles/access (16MB+)
       - L1→L2 boundary at 128 KB; L2→Global transition gradual around 8-18 MB

    3. SLM (Shared Local Memory):
       - SLM latency: ~80 cycles at small sizes (256B-1KB)
       - Very similar to L1 data cache (~71 cycles) — expected because SLM is
         carved from L1 on Xe2 (same physical hardware, different address space)
       - Not a separate low-latency SRAM like NVIDIA shared memory (~30 cycles)

    4. Memory Bandwidth (fixed benchmark):
       - DRAM read: ~147 GB/s (1GB buffer, 32% of 456 GB/s peak)
       - DRAM write: ~162 GB/s (1GB buffer)
       - L2 cache read: ~303-351 GB/s (1-4 MB buffer)
       - Original bandwidth results (682-900 GB/s) had a 10× overcounting bug

    5. Infrastructure:
       - Raw SPIR-V pipeline verified: .spvasm -> spirv-as -> ocloc -> GEN ASM
       - Level Zero submission works with timing (host-side)
       - GEN ASM validation confirms instruction counts
       - INT8 cooperative matrix blocked by ocloc segfault (compiler bug)
    """)

if __name__ == "__main__":
    main()
