#!/usr/bin/env python3
"""
SPIR-V DPAS Microbenchmark Automation
Generates kernels, builds, runs, validates, and outputs CSV results.

Usage:
  python3 run_dpas_sweep.py                    # Run full sweep
  python3 run_dpas_sweep.py --n-values 1 2 4 8 16 32 64 128 256
  python3 run_dpas_sweep.py --mode throughput   # Throughput mode
"""

import subprocess, os, sys, re, csv, tempfile, shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
DEVICE = "bmg-g21"
SPIRV_RUNNER = SCRIPT_DIR / "spirv_runner"
RESULTS_DIR = SCRIPT_DIR / "results"
WARMUP = 10
REPEAT = 100

# SPIR-V template for DPAS latency (dependent chain)
DPAS_LATENCY_TEMPLATE = """; SPIR-V 1.4
; DPAS Latency: {n} dependent CooperativeMatrixMulAddKHR operations
; BF16 8x16x16 tile -> dpas.8x8 on Intel XMX
               OpCapability Addresses
               OpCapability Kernel
               OpCapability Int64
               OpCapability Int16
               OpCapability CooperativeMatrixKHR
               OpExtension "SPV_KHR_cooperative_matrix"
          %1 = OpExtInstImport "OpenCL.std"
               OpMemoryModel Physical64 OpenCL
               OpEntryPoint Kernel %main "dpas_latency"
               OpExecutionMode %main SubgroupSize 16
     %void   = OpTypeVoid
     %bool   = OpTypeBool
     %uint   = OpTypeInt 32 0
     %int    = OpTypeInt 32 1
     %ulong  = OpTypeInt 64 0
     %float  = OpTypeFloat 32
     %ushort = OpTypeInt 16 0
   %uint_0   = OpConstant %uint 0
   %uint_1   = OpConstant %uint 1
   %uint_2   = OpConstant %uint 2
   %uint_3   = OpConstant %uint 3
   %uint_8   = OpConstant %uint 8
  %uint_16   = OpConstant %uint 16
  %uint_{n}  = OpConstant %uint {n}
  %ulong_0   = OpConstant %ulong 0
%ptr_cross_ushort = OpTypePointer CrossWorkgroup %ushort
%ptr_cross_float  = OpTypePointer CrossWorkgroup %float
  %cm_acc = OpTypeCooperativeMatrixKHR %float %uint_3 %uint_8 %uint_16 %uint_2
  %cm_a   = OpTypeCooperativeMatrixKHR %ushort %uint_3 %uint_8 %uint_16 %uint_0
  %cm_b   = OpTypeCooperativeMatrixKHR %ushort %uint_3 %uint_16 %uint_16 %uint_1
  %fn_type = OpTypeFunction %void %ptr_cross_ushort %ptr_cross_ushort %ptr_cross_float
  %main = OpFunction %void None %fn_type
  %buf_a = OpFunctionParameter %ptr_cross_ushort
  %buf_b = OpFunctionParameter %ptr_cross_ushort
  %buf_c = OpFunctionParameter %ptr_cross_float
  %entry = OpLabel
   %a_tile = OpCooperativeMatrixLoadKHR %cm_a %buf_a %uint_0 %uint_16 None
   %b_tile = OpCooperativeMatrixLoadKHR %cm_b %buf_b %uint_0 %uint_16 None
  %acc_init = OpCooperativeMatrixLoadKHR %cm_acc %buf_c %uint_0 %uint_16 None
               OpBranch %loop_header
%loop_header = OpLabel
               OpLoopMerge %loop_exit %loop_header None
  %acc_phi = OpPhi %cm_acc %acc_init %entry %acc_next %loop_body
    %i_phi = OpPhi %uint %uint_0 %entry %i_next %loop_body
      %cond = OpULessThan %bool %i_phi %uint_{n}
               OpBranchConditional %cond %loop_body %loop_exit
%loop_body = OpLabel
  %acc_next = OpCooperativeMatrixMulAddKHR %cm_acc %a_tile %b_tile %acc_phi
    %i_next = OpIAdd %uint %i_phi %uint_1
               OpBranch %loop_header
%loop_exit = OpLabel
               OpCooperativeMatrixStoreKHR %buf_c %acc_phi %uint_0 %uint_16 None
               OpReturn
               OpFunctionEnd
"""



def run_cmd(cmd, cwd=SCRIPT_DIR, check=True):
    """Run a command and return output."""
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if check and r.returncode != 0:
        print(f"ERROR: {cmd}", file=sys.stderr)
        print(r.stderr, file=sys.stderr)
        sys.exit(1)
    return r


def build_spirv_pipeline(spvasm_path, kernel_name):
    """Full pipeline: .spvasm → .spv → _bmg.bin → validate"""
    spv_path = spvasm_path.with_suffix('.spv')
    bin_path = SCRIPT_DIR / f"{kernel_name}_bmg.bin"
    disasm_dir = SCRIPT_DIR / f"{kernel_name}_disasm"

    # Assemble
    run_cmd(f"spirv-as {spvasm_path} -o {spv_path}")

    # Compile
    r = run_cmd(f"ocloc compile -spirv_input -file {spv_path} -device {DEVICE} -output {kernel_name}")
    if "error" in r.stderr.lower() or "error" in r.stdout.lower():
        print(f"  COMPILE FAILED: {kernel_name}")
        return False

    # Find output binary (ocloc naming convention)
    candidates = [
        SCRIPT_DIR / f"{kernel_name}_bmg.bin",
        SCRIPT_DIR / f"{kernel_name}.bin_bmg.bin",
    ]
    for c in candidates:
        if c.exists():
            return c
    print(f"  WARNING: Binary not found for {kernel_name}")
    return False


def validate_dpas_count(bin_path, expected_n):
    """Disassemble and count DPAS instructions."""
    disasm_dir = SCRIPT_DIR / f"disasm_{bin_path.stem}"
    run_cmd(f"mkdir -p {disasm_dir}")
    run_cmd(f"ocloc disasm -file {bin_path} -dump {disasm_dir} -device {DEVICE}", check=False)

    count = 0
    for f in disasm_dir.iterdir():
        if f.name.endswith('.asm'):
            text = f.read_text()
            count += text.count("dpas")

    if count == 0:
        print(f"  WARNING: No DPAS instructions found!")
        return 0

    if count != expected_n:
        print(f"  NOTE: Expected {expected_n} DPAS, got {count} (IGC may have optimized)")

    return count


def run_kernel(spv_path, kernel_name, wg_x=16, n_wg=1, repeats=100):
    """Run kernel via spirv_runner and parse timing."""
    env = os.environ.copy()
    env["IGC_DisableIGCOptimizations"] = "1"

    cmd = f"{SPIRV_RUNNER} {spv_path} {kernel_name} {wg_x} {n_wg} {repeats}"
    r = run_cmd(cmd, check=False)

    if r.returncode != 0:
        print(f"  RUN FAILED: {kernel_name}")
        print(r.stderr)
        return None

    # Parse timing: "Runs=100  Median=7613.0 ns  Mean=7678.5 ns  Min=7360.0 ns  Max=10571.0 ns"
    m = re.search(r'Median=([\d.]+)\s+ns\s+Mean=([\d.]+)\s+ns\s+Min=([\d.]+)\s+ns\s+Max=([\d.]+)\s+ns', r.stdout)
    if m:
        return {
            'median_ns': float(m.group(1)),
            'mean_ns': float(m.group(2)),
            'min_ns': float(m.group(3)),
            'max_ns': float(m.group(4)),
        }
    return None


def sweep_latency(n_values=None, wg_sizes=None):
    """Run DPAS latency sweep: vary N (dependent chain length) and work-group sizes."""
    if n_values is None:
        n_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    if wg_sizes is None:
        wg_sizes = [16]  # Single sub-group first

    RESULTS_DIR.mkdir(exist_ok=True)
    results = []

    print("=" * 80)
    print("DPAS BF16 Latency Sweep (Dependent Chain)")
    print("=" * 80)

    # Get GPU clock
    r = run_cmd(f"{SPIRV_RUNNER} spirv_dpas_latency.spv dpas_latency 16 1 1", check=False)
    ghz = 2.4  # Default; parse from output if available
    m = re.search(r'Clock: ([\d.]+)\s*GHz', r.stdout)
    if m:
        ghz = float(m.group(1))
    else:
        # Try from device props
        m = re.search(r'Device:.*', r.stdout)
        print(f"  Using default clock: {ghz} GHz")

    for wg_x in wg_sizes:
        for n in n_values:
            kernel_name = f"dpas_lat_n{n}"
            spvasm_path = SCRIPT_DIR / f"sweep_{kernel_name}.spvasm"
            spv_path = SCRIPT_DIR / f"sweep_{kernel_name}.spv"

            print(f"\n  N={n:4d}  WG={wg_x}  ... ", end="", flush=True)

            # Generate .spvasm
            spvasm_path.write_text(DPAS_LATENCY_TEMPLATE.format(n=n))

            # Build
            bin_path = build_spirv_pipeline(spvasm_path, kernel_name)
            if not bin_path:
                continue

            # Find the actual .spv path
            if not spv_path.exists():
                spv_path = SCRIPT_DIR / f"sweep_{kernel_name}.spv"

            # Validate DPAS count
            dpas_count = validate_dpas_count(bin_path, n)

            # Run (entry point name is always "dpas_latency" in the SPIR-V)
            timing = run_kernel(spv_path, "dpas_latency", wg_x=wg_x, n_wg=1, repeats=REPEAT)
            if timing is None:
                continue

            cycles_per_dpas = timing['median_ns'] * ghz / n if n > 0 else 0

            print(f"median={timing['median_ns']:.0f}ns  cyc/dpas={cycles_per_dpas:.1f}  dpas_count={dpas_count}")

            results.append({
                'mode': 'latency',
                'n_dpas': n,
                'wg_size': wg_x,
                'n_wg': 1,
                'dpas_count_in_asm': dpas_count,
                'median_ns': timing['median_ns'],
                'mean_ns': timing['mean_ns'],
                'min_ns': timing['min_ns'],
                'max_ns': timing['max_ns'],
                'cycles_per_dpas': cycles_per_dpas,
                'ghz': ghz,
            })

    # Compute per-DPAS latency from slope
    if len(results) >= 2:
        import numpy as np
        xs = np.array([r['n_dpas'] for r in results])
        ys = np.array([r['median_ns'] for r in results])
        # Linear fit: time = overhead + n * per_dpas_time
        coeffs = np.polyfit(xs, ys, 1)
        slope_ns = coeffs[0]
        intercept_ns = coeffs[1]
        slope_cycles = slope_ns * ghz

        print(f"\n{'='*60}")
        print(f"DPAS BF16 Latency (from slope): {slope_ns:.2f} ns = {slope_cycles:.1f} cycles")
        print(f"Fixed overhead (intercept): {intercept_ns:.0f} ns = {intercept_ns * ghz:.0f} cycles")
        print(f"{'='*60}")

    # Write CSV
    if not results:
        print("\nNo results to save.")
        return results

    csv_path = RESULTS_DIR / "dpas_latency_sweep.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {csv_path}")

    # Cleanup temp files
    for f in SCRIPT_DIR.glob("sweep_dpas_lat_n*"):
        if f.suffix in ['.spvasm', '.spv'] or '_bmg.bin' in f.name:
            f.unlink()
    for d in SCRIPT_DIR.glob("disasm_sweep_*"):
        shutil.rmtree(d, ignore_errors=True)

    return results


def sweep_throughput(n_ilp_values=None, sg_counts=None, n_iter=128):
    """Run DPAS throughput sweep: independent chains, vary ILP and sub-group count."""
    if n_ilp_values is None:
        n_ilp_values = [1, 2, 4, 8]
    if sg_counts is None:
        sg_counts = [1, 2, 4, 8, 16]

    RESULTS_DIR.mkdir(exist_ok=True)
    results = []

    print("=" * 80)
    print("DPAS BF16 Throughput Sweep (Independent Chains)")
    print("=" * 80)

    for n_ilp in n_ilp_values:
        for n_sg in sg_counts:
            wg_x = n_sg * 16
            kernel_name = f"dpas_tput_ilp{n_ilp}_sg{n_sg}"
            spvasm_path = SCRIPT_DIR / f"sweep_{kernel_name}.spvasm"
            spv_path = SCRIPT_DIR / f"sweep_{kernel_name}.spv"

            print(f"\n  ILP={n_ilp}  SG={n_sg}  WG={wg_x}  ... ", end="", flush=True)

            # Generate throughput kernel
            spvasm = generate_throughput_spvasm(n_ilp, n_iter)
            spvasm_path.write_text(spvasm)

            bin_path = build_spirv_pipeline(spvasm_path, kernel_name)
            if not bin_path:
                continue

            dpas_count = validate_dpas_count(bin_path, n_ilp * n_iter)

            timing = run_kernel(spv_path, "dpas_throughput", wg_x=wg_x, n_wg=1, repeats=REPEAT)
            if timing is None:
                continue

            total_dpas = n_ilp * n_iter
            total_flops = total_dpas * 2 * 8 * 16 * 16  # 2*M*N*K per DPAS
            tflops = total_flops / (timing['median_ns'] * 1e-9) / 1e12

            print(f"median={timing['median_ns']:.0f}ns  {tflops:.3f}TFLOPS  dpas={dpas_count}")

            results.append({
                'mode': 'throughput',
                'n_ilp': n_ilp,
                'n_sg': n_sg,
                'wg_size': wg_x,
                'n_iter': n_iter,
                'dpas_count_in_asm': dpas_count,
                'median_ns': timing['median_ns'],
                'total_dpas': total_dpas,
                'tflops': tflops,
            })

    csv_path = RESULTS_DIR / "dpas_throughput_sweep.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {csv_path}")

    # Cleanup
    for f in SCRIPT_DIR.glob("sweep_dpas_tput_*"):
        if f.suffix in ['.spvasm', '.spv'] or '_bmg.bin' in f.name:
            f.unlink()
    for d in SCRIPT_DIR.glob("disasm_sweep_*"):
        shutil.rmtree(d, ignore_errors=True)

    return results


def generate_throughput_spvasm(n_ilp, n_iter):
    """Generate complete SPIR-V assembly for throughput kernel."""
    lines = [
        "; SPIR-V 1.4",
        f"; DPAS Throughput: {n_ilp} independent chains, {n_iter} iters",
        "               OpCapability Addresses",
        "               OpCapability Kernel",
        "               OpCapability Int64",
        "               OpCapability Int16",
        "               OpCapability CooperativeMatrixKHR",
        '               OpExtension "SPV_KHR_cooperative_matrix"',
        "          %1 = OpExtInstImport \"OpenCL.std\"",
        "               OpMemoryModel Physical64 OpenCL",
        "               OpEntryPoint Kernel %main \"dpas_throughput\"",
        "               OpExecutionMode %main SubgroupSize 16",
        "     %void   = OpTypeVoid",
        "     %bool   = OpTypeBool",
        "     %uint   = OpTypeInt 32 0",
        "     %ulong  = OpTypeInt 64 0",
        "     %float  = OpTypeFloat 32",
        "     %ushort = OpTypeInt 16 0",
        "   %uint_0   = OpConstant %uint 0",
        "   %uint_1   = OpConstant %uint 1",
        "   %uint_2   = OpConstant %uint 2",
        "   %uint_3   = OpConstant %uint 3",
        "   %uint_8   = OpConstant %uint 8",
        "  %uint_16   = OpConstant %uint 16",
        f"  %uint_{n_iter}  = OpConstant %uint {n_iter}",
        "  %ulong_0   = OpConstant %ulong 0",
        "%ptr_cross_ushort = OpTypePointer CrossWorkgroup %ushort",
        "%ptr_cross_float  = OpTypePointer CrossWorkgroup %float",
        "  %cm_acc = OpTypeCooperativeMatrixKHR %float %uint_3 %uint_8 %uint_16 %uint_2",
        "  %cm_a   = OpTypeCooperativeMatrixKHR %ushort %uint_3 %uint_8 %uint_16 %uint_0",
        "  %cm_b   = OpTypeCooperativeMatrixKHR %ushort %uint_3 %uint_16 %uint_16 %uint_1",
        "  %fn_type = OpTypeFunction %void %ptr_cross_ushort %ptr_cross_ushort %ptr_cross_float",
        "  %main = OpFunction %void None %fn_type",
        "  %buf_a = OpFunctionParameter %ptr_cross_ushort",
        "  %buf_b = OpFunctionParameter %ptr_cross_ushort",
        "  %buf_c = OpFunctionParameter %ptr_cross_float",
        "  %entry = OpLabel",
    ]

    # Load tiles for each ILP chain
    for i in range(n_ilp):
        lines.append(f"  %a_tile{i} = OpCooperativeMatrixLoadKHR %cm_a %buf_a %uint_0 %uint_16 None")
        lines.append(f"  %b_tile{i} = OpCooperativeMatrixLoadKHR %cm_b %buf_b %uint_0 %uint_16 None")
        lines.append(f"  %acc_init{i} = OpCooperativeMatrixLoadKHR %cm_acc %buf_c %uint_0 %uint_16 None")

    lines.append("               OpBranch %loop_header")

    # Loop header with phi nodes
    lines.append("%loop_header = OpLabel")
    lines.append("               OpLoopMerge %loop_exit %loop_header None")
    for i in range(n_ilp):
        lines.append(f"  %acc_phi{i} = OpPhi %cm_acc %acc_init{i} %entry %acc_next{i} %loop_body")
    lines.append("    %i_phi = OpPhi %uint %uint_0 %entry %i_next %loop_body")
    lines.append(f"      %cond = OpULessThan %bool %i_phi %uint_{n_iter}")
    lines.append("               OpBranchConditional %cond %loop_body %loop_exit")

    # Loop body: independent MulAdd for each chain
    lines.append("%loop_body = OpLabel")
    for i in range(n_ilp):
        lines.append(f"  %acc_next{i} = OpCooperativeMatrixMulAddKHR %cm_acc %a_tile{i} %b_tile{i} %acc_phi{i}")
    lines.append("    %i_next = OpIAdd %uint %i_phi %uint_1")
    lines.append("               OpBranch %loop_header")

    # Exit: store results
    lines.append("%loop_exit = OpLabel")
    for i in range(n_ilp):
        lines.append(f"               OpCooperativeMatrixStoreKHR %buf_c %acc_phi{i} %uint_0 %uint_16 None")
    lines.append("               OpReturn")
    lines.append("               OpFunctionEnd")

    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['latency', 'throughput', 'both'], default='both')
    parser.add_argument('--n-values', type=int, nargs='+',
                        default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    parser.add_argument('--repeats', type=int, default=100)
    args = parser.parse_args()

    REPEAT = args.repeats

    # Ensure runner is built
    if not SPIRV_RUNNER.exists():
        print("Building spirv_runner...")
        run_cmd(f"g++ -std=c++17 -O2 -o spirv_runner spirv_runner.cpp -lze_loader -lm")

    if args.mode in ('latency', 'both'):
        sweep_latency(n_values=args.n_values)

    if args.mode in ('throughput', 'both'):
        sweep_throughput()
