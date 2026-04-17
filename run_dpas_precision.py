#!/usr/bin/env python3
"""
Extended Precision DPAS Latency Sweep
Modifies the working SPIR-V template for different N values and precisions.
"""

import subprocess, os, sys, re, csv, shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
SPIRV_RUNNER = SCRIPT_DIR / "spirv_runner"
RESULTS_DIR = SCRIPT_DIR / "results"
DEVICE = "bmg-g21"
REPEATS = 50

def run_cmd(cmd, check=True):
    r = subprocess.run(cmd, shell=True, cwd=SCRIPT_DIR, capture_output=True, text=True)
    if check and r.returncode != 0:
        print(f"ERROR: {cmd}", file=sys.stderr)
        print(r.stderr, file=sys.stderr)
    return r

# Read the working BF16 template
BF16_TEMPLATE = (SCRIPT_DIR / "spirv_dpas_latency.spvasm").read_text()

def gen_bf16(n):
    """Modify BF16 template to use different N."""
    text = BF16_TEMPLATE
    text = text.replace("uint_128", f"uint_{n}")
    text = text.replace("= OpConstant %uint 128", f"= OpConstant %uint {n}")
    # Fix the comment
    text = text.replace("N=128", f"N={n}")
    return text

def gen_fp16(n):
    """Generate FP16 DPAS kernel by modifying BF16 template.
    FP16 uses half type instead of ushort, same 8x16x16 dimensions."""
    text = gen_bf16(n)
    # Replace ushort with half for operand types
    text = text.replace("OpCapability Int16", "OpCapability Int16\n               OpCapability Float16")
    text = text.replace("%ushort = OpTypeInt 16 0", "%half = OpTypeFloat 16")
    text = text.replace("%ushort", "%half")
    text = text.replace("ptr_cross_ushort", "ptr_cross_half")
    return text

def gen_int8(n):
    """Generate INT8 DPAS kernel. INT8 uses 8x16x32 tile (K=32 instead of 16)."""
    text = gen_bf16(n)
    # Replace ushort with uchar for operand types
    text = text.replace("OpCapability Int16", "OpCapability Int8")
    text = text.replace("%ushort = OpTypeInt 16 0", "%uchar = OpTypeInt 8 0")
    text = text.replace("%ushort", "%uchar")
    text = text.replace("ptr_cross_ushort", "ptr_cross_uchar")
    # Change accumulator from float to int for INT8
    text = text.replace("%float = OpTypeFloat 32", "%int = OpTypeInt 32 1")
    text = text.replace("ptr_cross_float", "ptr_cross_int")
    # Change tile dimensions: A(8x32), B(32x16), C(8x16) for INT8
    # K dimension: %uint_16 → %uint_32 for A rows (K=32)
    text = text.replace("OpTypeCooperativeMatrixKHR %int %uint_3 %uint_8 %uint_16 %uint_2",
                        "OpTypeCooperativeMatrixKHR %int %uint_3 %uint_8 %uint_16 %uint_2")
    # A matrix: 8x32 (M=8, K=32)
    text = text.replace("OpTypeCooperativeMatrixKHR %uchar %uint_3 %uint_8 %uint_16 %uint_0",
                        "OpTypeCooperativeMatrixKHR %uchar %uint_3 %uint_8 %uint_32 %uint_0")
    # B matrix: 32x16 (K=32, N=16)
    text = text.replace("OpTypeCooperativeMatrixKHR %uchar %uint_3 %uint_16 %uint_16 %uint_1",
                        "OpTypeCooperativeMatrixKHR %uchar %uint_3 %uint_32 %uint_16 %uint_1")
    # Add uint_32 constant and fix stride for A load (stride=32 for K=32)
    text = text.replace("%ulong_0 = OpConstant %ulong 0",
                        "%uint_32 = OpConstant %uint 32\n  %ulong_0 = OpConstant %ulong 0")
    # Fix A tile load stride (stride=32 for K=32)
    # This is tricky - the A load uses stride 16, need to change to 32
    # %a_tile = OpCooperativeMatrixLoadKHR %cm_a %buf_a %uint_0 %uint_16 None
    # Need to change the stride operand
    lines = text.split('\n')
    new_lines = []
    a_load_fixed = False
    for line in lines:
        if 'a_tile = OpCooperativeMatrixLoadKHR' and not a_load_fixed:
            # Change stride from %uint_16 to %uint_32 for A matrix
            line = line.replace('%cm_a %buf_a %uint_0 %uint_16',
                              '%cm_a %buf_a %uint_0 %uint_32')
            a_load_fixed = True
        new_lines.append(line)
    text = '\n'.join(new_lines)
    return text

def build_and_run(gen_func, precision, n_values):
    """Build SPIR-V kernels and run latency sweep."""
    RESULTS_DIR.mkdir(exist_ok=True)
    results = []
    ghz = 2.4

    print(f"\n{'='*60}")
    print(f"DPAS {precision} Latency Sweep")
    print(f"{'='*60}")
    print(f"  {'N':>6s} {'median_ns':>10s} {'cyc/dpas':>10s} {'dpas_cnt':>8s}")
    print(f"  {'-'*40}")

    for n in n_values:
        name = f"dpas_{precision.lower()}_n{n}"
        spvasm_path = SCRIPT_DIR / f"sweep_{name}.spvasm"
        spv_path = SCRIPT_DIR / f"sweep_{name}.spv"

        try:
            spvasm_text = gen_func(n)
        except Exception as e:
            print(f"  N={n:4d} GEN FAILED: {e}")
            continue

        spvasm_path.write_text(spvasm_text)

        # Assemble
        r = run_cmd(f"spirv-as {spvasm_path} -o {spv_path}")
        if r.returncode != 0:
            print(f"  N={n:4d} ASSEMBLE FAILED")
            spvasm_path.unlink(missing_ok=True)
            continue

        # Compile
        r = run_cmd(f"ocloc compile -spirv_input -file {spv_path} -device {DEVICE} -output {name}", check=False)
        if r.returncode != 0 or "Build failed" in r.stdout + r.stderr:
            print(f"  N={n:4d} COMPILE FAILED")
            spvasm_path.unlink(missing_ok=True)
            spv_path.unlink(missing_ok=True)
            continue

        # Find binary
        bin_path = None
        for c in [SCRIPT_DIR / f"{name}_bmg.bin"]:
            if c.exists():
                bin_path = c
                break
        if not bin_path:
            print(f"  N={n:4d} NO BINARY")
            continue

        # Validate DPAS count
        disasm_dir = SCRIPT_DIR / f"disasm_{name}"
        run_cmd(f"mkdir -p {disasm_dir}", check=False)
        run_cmd(f"ocloc disasm -file {bin_path} -dump {disasm_dir} -device {DEVICE}", check=False)
        dpas_count = 0
        for f in disasm_dir.iterdir():
            if f.name.endswith('.asm'):
                dpas_count += f.read_text().count("dpas")

        # Run
        r = run_cmd(f"./spirv_runner {spv_path} dpas_latency 16 1 {REPEATS}", check=False)
        if r.returncode != 0:
            print(f"  N={n:4d} RUN FAILED")
        else:
            m = re.search(r'Median=([\d.]+)\s+ns', r.stdout)
            if m:
                median = float(m.group(1))
                cyc = median * ghz / n if n > 0 else 0
                print(f"  N={n:4d} {median:>10.0f} {cyc:>10.1f} {dpas_count:>8d}")
                results.append({
                    'precision': precision,
                    'n_dpas': n,
                    'dpas_count_asm': dpas_count,
                    'median_ns': median,
                    'cycles_per_dpas': cyc,
                })

        # Cleanup
        for f in [spvasm_path, spv_path, bin_path]:
            f.unlink(missing_ok=True)
        shutil.rmtree(disasm_dir, ignore_errors=True)
        for f in SCRIPT_DIR.glob(f"{name}*"):
            f.unlink(missing_ok=True)

    # Compute slope
    if len(results) >= 2:
        import numpy as np
        xs = np.array([r['n_dpas'] for r in results])
        ys = np.array([r['median_ns'] for r in results])
        coeffs = np.polyfit(xs, ys, 1)
        slope_ns = coeffs[0]
        slope_cyc = slope_ns * ghz
        print(f"\n  >> {precision} DPAS Latency: {slope_ns:.2f} ns = {slope_cyc:.1f} cycles")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--precision', choices=['bf16', 'fp16', 'int8', 'all'], default='all')
    args = parser.parse_args()

    if not SPIRV_RUNNER.exists():
        run_cmd("g++ -std=c++17 -O2 -o spirv_runner spirv_runner.cpp -lze_loader -lm")

    n_values = [1, 4, 16, 64, 128, 256]
    all_results = []

    if args.precision in ('bf16', 'all'):
        all_results += build_and_run(gen_bf16, "BF16", n_values)

    if args.precision in ('fp16', 'all'):
        all_results += build_and_run(gen_fp16, "FP16", n_values)

    if args.precision in ('int8', 'all'):
        all_results += build_and_run(gen_int8, "INT8", n_values)

    # Save combined CSV
    if all_results:
        csv_path = RESULTS_DIR / "dpas_precision_sweep.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nAll results saved to {csv_path}")
