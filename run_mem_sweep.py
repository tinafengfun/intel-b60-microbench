#!/usr/bin/env python3
"""
Memory Microbenchmark Sweep using SPIR-V kernels
Sweeps buffer size for pointer chase (latency) and read (bandwidth)
"""

import subprocess, os, sys, re, csv, shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
MEM_RUNNER = SCRIPT_DIR / "mem_runner"
RESULTS_DIR = SCRIPT_DIR / "results"
REPEATS = 50

def run_cmd(cmd, check=True):
    r = subprocess.run(cmd, shell=True, cwd=SCRIPT_DIR, capture_output=True, text=True)
    if check and r.returncode != 0:
        print(f"ERROR: {cmd}", file=sys.stderr)
        print(r.stderr, file=sys.stderr)
        return r
    return r


def sweep_latency():
    """Sweep buffer size for pointer chase latency."""
    sizes_kb = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512,
                768, 1024, 1536, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

    RESULTS_DIR.mkdir(exist_ok=True)
    results = []

    print("=" * 80)
    print("Memory Latency Sweep (Pointer Chase, CHASE=4096)")
    print(f"{'Size':>10s} {'Median(ns)':>12s} {'cyc/access':>12s} {'ns/access':>12s} {'Region':>10s}")
    print("-" * 80)

    for sz_kb in sizes_kb:
        cmd = f"./mem_runner spirv_ptr_chase.spv ptr_chase chase {sz_kb} 1 1 {REPEATS}"
        r = run_cmd(cmd, check=False)
        if r.returncode != 0:
            print(f"{'%dKB' % sz_kb:>10s} FAILED")
            continue

        # Parse: "Chase=4096  Latency: 112.9 cycles/access (47.1 ns)"
        m = re.search(r'Latency:\s+([\d.]+)\s+cycles/access\s+\(([\d.]+)\s+ns\)', r.stdout)
        if not m:
            print(f"{'%dKB' % sz_kb:>10s} PARSE ERROR")
            continue

        cyc = float(m.group(1))
        ns_per = float(m.group(2))

        # Parse median
        m2 = re.search(r'Median=([\d.]+)\s+ns', r.stdout)
        median = float(m2.group(1)) if m2 else 0

        # Identify region
        if sz_kb <= 128:
            region = "SLM/L1"
        elif sz_kb <= 4096:
            region = "L2"
        else:
            region = "Global"

        if sz_kb < 1024:
            size_str = f"{sz_kb}KB"
        else:
            size_str = f"{sz_kb//1024}MB"

        print(f"{size_str:>10s} {median:>12.0f} {cyc:>12.1f} {ns_per:>12.1f} {region:>10s}")

        results.append({
            'size_kb': sz_kb,
            'size_str': size_str,
            'median_ns': median,
            'cycles_per_access': cyc,
            'ns_per_access': ns_per,
            'region': region,
        })

    if results:
        csv_path = RESULTS_DIR / "mem_latency_sweep.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {csv_path}")

    return results


def sweep_bandwidth():
    """Sweep buffer size for coalesced read bandwidth."""
    sizes_kb = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]  # 1MB to 128MB
    n_wg_values = [1, 4, 16, 64, 256]

    RESULTS_DIR.mkdir(exist_ok=True)
    results = []

    print("=" * 80)
    print("Memory Bandwidth Sweep (Coalesced Read, 10 iters/thread, WG=256)")
    print(f"{'Size':>10s} {'nWG':>6s} {'Median(ns)':>12s} {'BW(GB/s)':>10s}")
    print("-" * 80)

    for sz_kb in sizes_kb:
        for n_wg in n_wg_values:
            cmd = f"./mem_runner spirv_mem_read.spv mem_read read {sz_kb} 256 {n_wg} {REPEATS}"
            r = run_cmd(cmd, check=False)
            if r.returncode != 0:
                continue

            # Parse: "BW: 123.4 GB/s"
            m = re.search(r'BW:\s+([\d.]+)\s+GB/s', r.stdout)
            if not m:
                continue

            bw = float(m.group(1))
            m2 = re.search(r'Median=([\d.]+)\s+ns', r.stdout)
            median = float(m2.group(1)) if m2 else 0

            size_str = f"{sz_kb//1024}MB" if sz_kb >= 1024 else f"{sz_kb}KB"

            print(f"{size_str:>10s} {n_wg:>6d} {median:>12.0f} {bw:>10.1f}")

            results.append({
                'size_kb': sz_kb,
                'size_str': size_str,
                'n_wg': n_wg,
                'median_ns': median,
                'bandwidth_gbps': bw,
            })

    if results:
        csv_path = RESULTS_DIR / "mem_bandwidth_sweep.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {csv_path}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['latency', 'bandwidth', 'both'], default='both')
    args = parser.parse_args()

    if not MEM_RUNNER.exists():
        print("Building mem_runner...")
        run_cmd("g++ -std=c++17 -O2 -o mem_runner mem_runner.cpp -lze_loader -lm")

    if args.mode in ('latency', 'both'):
        sweep_latency()

    if args.mode in ('bandwidth', 'both'):
        sweep_bandwidth()
