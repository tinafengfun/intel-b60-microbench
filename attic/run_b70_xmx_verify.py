# WARNING: DCE-FLAWED intermediate version — see attic/README.md. Do not use for measurements.
#!/usr/bin/env python3
"""
B70 XMX verification — confirm the steady-state DPAS rate found by run_b70_xmx_probe.py
(~2.07 cyc/dpas/EU, ~4.76 TF/EU, ~1.2 PF full machine) and rule out artifacts.

V1: n_iter sweep (ILP=16, n_wg=32, sg=8) -> linear fit time = a + b*n_iter,
    b gives true steady-state cyc/dpas with all fixed overheads absorbed in a.
V2: ILP sweep at long n_iter (ILP=4/8/16/32) -> latency-hiding curve at full rate.
V3: wall-clock cross-check of one long launch (vs L0 event timer).
"""
import subprocess, sys, re, csv, time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
DEVICE = "bmg-g31"
SPIRV_RUNNER = SCRIPT_DIR / "spirv_runner"
RESULTS_DIR = SCRIPT_DIR / "results"
GHZ = 2.4
FLOPS_PER_DPAS = 4096

sys.path.insert(0, str(SCRIPT_DIR))
from run_b70_xmx_probe import gen_ilp_kernel, build_and_compile, cleanup, run_benchmark, run_cmd


def fit(n_iters, times):
    n = len(n_iters)
    sx, sy = sum(n_iters), sum(times)
    sxx = sum(x * x for x in n_iters)
    sxy = sum(x * y for x, y in zip(n_iters, times))
    denom = n * sxx - sx * sx
    b = (n * sxy - sx * sy) / denom
    a = (sy - b * sx) / n
    return a, b


def experiment_v1():
    print("\n=== V1: n_iter sweep (ILP=16, n_wg=32, sg=8, 256 SGs = 1 wave) ===")
    ilp = 16
    rows = []
    for n_iter in [128, 512, 2048, 8192, 32768]:
        name = f"v1_i{n_iter}"
        spv = build_and_compile(gen_ilp_kernel(ilp, n_iter), name)
        if not spv:
            continue
        median, err = run_benchmark(spv, 128, 32, repeats=20)
        cleanup(name)
        if median is None:
            print(f"  n_iter={n_iter} FAILED: {err}")
            continue
        dpas_per_sg = ilp * n_iter
        ns_per_dpas = median / dpas_per_sg
        tf = dpas_per_sg * 256 * FLOPS_PER_DPAS / (median * 1e-9) / 1e12
        print(f"  n_iter={n_iter:>6d}  median={median:>12.0f} ns  {ns_per_dpas:.4f} ns/dpas  apparent {tf:>8.1f} TF")
        rows.append((n_iter, median))
    if len(rows) >= 3:
        a, b = fit([r[0] for r in rows], [r[1] for r in rows])
        ns_per_dpas = b / ilp
        cyc = ns_per_dpas * GHZ
        tf_eu = FLOPS_PER_DPAS / (ns_per_dpas * 1e-9) / 1e12
        print(f"\n  FIT: time = {a:.0f} ns + {b:.4f} ns/iter")
        print(f"  steady-state: {ns_per_dpas:.4f} ns/dpas = {cyc:.3f} cyc/dpas  -> {tf_eu:.2f} TF/EU, {tf_eu*256:.0f} TF full machine")
        return {'fixed_overhead_ns': a, 'ns_per_iter': b, 'cyc_per_dpas': cyc,
                'tf_per_eu': tf_eu, 'tf_full': tf_eu * 256}
    return None


def experiment_v2():
    print("\n=== V2: ILP sweep at long n_iter (n_wg=32, sg=8) ===")
    n_iter = 16384
    rows = []
    for ilp in [4, 8, 16, 32]:
        name = f"v2_ilp{ilp}"
        spv = build_and_compile(gen_ilp_kernel(ilp, n_iter), name)
        if not spv:
            continue
        median, err = run_benchmark(spv, 128, 32, repeats=10)
        cleanup(name)
        if median is None:
            print(f"  ILP={ilp} FAILED: {err}")
            continue
        ns_per_dpas = median / (ilp * n_iter)
        tf = ilp * n_iter * 256 * FLOPS_PER_DPAS / (median * 1e-9) / 1e12
        print(f"  ILP={ilp:>2d}  median={median:>12.0f} ns  {ns_per_dpas:.4f} ns/dpas = {ns_per_dpas*GHZ:>6.2f} cyc  apparent {tf:>8.1f} TF")
        rows.append({'ilp': ilp, 'median_ns': median, 'ns_per_dpas': ns_per_dpas,
                     'cyc_per_dpas': ns_per_dpas * GHZ, 'tflops': tf})
    return rows


def experiment_v3():
    print("\n=== V3: wall-clock cross-check (ILP=16, n_iter=131072, n_wg=32, sg=8) ===")
    name = "v3_long"
    spv = build_and_compile(gen_ilp_kernel(16, 131072), name)
    if not spv:
        return
    t0 = time.time()
    r = run_cmd(f"{SPIRV_RUNNER} {spv} dpas_ilp 128 32 5", check=False)
    wall = time.time() - t0
    cleanup(name)
    print(r.stdout.strip())
    total_dpas = 16 * 131072 * 256
    m = re.search(r'Median=([\d.]+)\s+ns', r.stdout)
    if m:
        median = float(m.group(1))
        tf = total_dpas * FLOPS_PER_DPAS / (median * 1e-9) / 1e12
        print(f"  event median {median:.0f} ns -> {tf:.0f} TFLOPS;  wall for 5 repeats+cleanup: {wall:.2f} s")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    fit_row = experiment_v1()
    v2_rows = experiment_v2()
    experiment_v3()

    out = RESULTS_DIR / "b70_xmx_verify.csv"
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        if fit_row:
            w.writerow(['V1_fit'] + list(fit_row.keys()))
            w.writerow(['V1_fit'] + list(fit_row.values()))
        w.writerow([])
        w.writerow(['V2', 'ilp', 'median_ns', 'ns_per_dpas', 'cyc_per_dpas', 'tflops'])
        for r in v2_rows:
            w.writerow(['V2', r['ilp'], r['median_ns'], r['ns_per_dpas'], r['cyc_per_dpas'], r['tflops']])
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
