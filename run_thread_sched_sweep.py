#!/usr/bin/env python3
"""
Phase 5: Thread Scheduling Granularity Experiments
5A: SG/WG sweep with fixed total work (1280 SGs)
5B: EU thread context switch overhead (ILP=1, multi-SG single WG)
5C: Dispatch overhead per WG (trivial 1-DPAS kernel)
"""
import subprocess, sys, re, csv, shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
DEVICE = "bmg-g21"
SPIRV_RUNNER = SCRIPT_DIR / "spirv_runner"
RESULTS_DIR = SCRIPT_DIR / "results"
GHZ = 2.4
FLOPS_PER_DPAS = 4096


def run_cmd(cmd, cwd=SCRIPT_DIR, check=True):
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if check and r.returncode != 0:
        print(f"ERROR: {cmd}\n{r.stderr}", file=sys.stderr)
    return r


def gen_ilp8_kernel(n_iter=128):
    """ILP=8 throughput kernel (from run_dpas_full_gpu.py)."""
    lines = [
        "; SPIR-V 1.4",
        f"; DPAS ILP=8 throughput, {n_iter} iters",
        "               OpCapability Addresses",
        "               OpCapability Kernel",
        "               OpCapability Int64",
        "               OpCapability Int16",
        "               OpCapability CooperativeMatrixKHR",
        '               OpExtension "SPV_KHR_cooperative_matrix"',
        '          %1 = OpExtInstImport "OpenCL.std"',
        "               OpMemoryModel Physical64 OpenCL",
        "               OpEntryPoint Kernel %main \"dpas_ilp8\"",
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
    ILP = 8
    for i in range(ILP):
        lines.append(f"  %a_tile{i} = OpCooperativeMatrixLoadKHR %cm_a %buf_a %uint_0 %uint_16 None")
        lines.append(f"  %b_tile{i} = OpCooperativeMatrixLoadKHR %cm_b %buf_b %uint_0 %uint_16 None")
        lines.append(f"  %acc_init{i} = OpCooperativeMatrixLoadKHR %cm_acc %buf_c %uint_0 %uint_16 None")
    lines.append("               OpBranch %lh")
    lines.append("%lh = OpLabel")
    lines.append("               OpLoopMerge %lx %lh None")
    for i in range(ILP):
        lines.append(f"  %acc_phi{i} = OpPhi %cm_acc %acc_init{i} %entry %acc_next{i} %lb")
    lines.append("    %i_phi = OpPhi %uint %uint_0 %entry %i_next %lb")
    lines.append(f"      %cond = OpULessThan %bool %i_phi %uint_{n_iter}")
    lines.append("               OpBranchConditional %cond %lb %lx")
    lines.append("%lb = OpLabel")
    for i in range(ILP):
        lines.append(f"  %acc_next{i} = OpCooperativeMatrixMulAddKHR %cm_acc %a_tile{i} %b_tile{i} %acc_phi{i}")
    lines.append("    %i_next = OpIAdd %uint %i_phi %uint_1")
    lines.append("               OpBranch %lh")
    lines.append("%lx = OpLabel")
    lines.append("               OpCooperativeMatrixStoreKHR %buf_c %acc_phi0 %uint_0 %uint_16 None")
    lines.append("               OpReturn")
    lines.append("               OpFunctionEnd")
    return "\n".join(lines)


def gen_ilp1_kernel(n_iter=1024):
    """ILP=1 dependent DPAS chain for TLP measurement."""
    return f"""; SPIR-V 1.4
; DPAS ILP=1, {n_iter} iterations (measures TLP from multiple SGs)
               OpCapability Addresses
               OpCapability Kernel
               OpCapability Int64
               OpCapability Int16
               OpCapability CooperativeMatrixKHR
               OpExtension "SPV_KHR_cooperative_matrix"
          %1 = OpExtInstImport "OpenCL.std"
               OpMemoryModel Physical64 OpenCL
               OpEntryPoint Kernel %main "dpas_ilp1"
               OpExecutionMode %main SubgroupSize 16
     %void   = OpTypeVoid
     %bool   = OpTypeBool
     %uint   = OpTypeInt 32 0
     %ulong  = OpTypeInt 64 0
     %float  = OpTypeFloat 32
     %ushort = OpTypeInt 16 0
   %uint_0   = OpConstant %uint 0
   %uint_1   = OpConstant %uint 1
   %uint_2   = OpConstant %uint 2
   %uint_3   = OpConstant %uint 3
   %uint_8   = OpConstant %uint 8
   %uint_16  = OpConstant %uint 16
  %uint_{n_iter}  = OpConstant %uint {n_iter}
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
               OpBranch %lh
%lh = OpLabel
               OpLoopMerge %lx %lh None
  %acc_phi = OpPhi %cm_acc %acc_init %entry %acc_next %lb
    %i_phi = OpPhi %uint %uint_0 %entry %i_next %lb
      %cond = OpULessThan %bool %i_phi %uint_{n_iter}
               OpBranchConditional %cond %lb %lx
%lb = OpLabel
  %acc_next = OpCooperativeMatrixMulAddKHR %cm_acc %a_tile %b_tile %acc_phi
    %i_next = OpIAdd %uint %i_phi %uint_1
               OpBranch %lh
%lx = OpLabel
               OpCooperativeMatrixStoreKHR %buf_c %acc_phi %uint_0 %uint_16 None
               OpReturn
               OpFunctionEnd
"""


def gen_trivial_1dpas():
    """Single DPAS operation — for dispatch overhead measurement."""
    return """; SPIR-V 1.4
; Trivial single DPAS
               OpCapability Addresses
               OpCapability Kernel
               OpCapability Int64
               OpCapability Int16
               OpCapability CooperativeMatrixKHR
               OpExtension "SPV_KHR_cooperative_matrix"
          %1 = OpExtInstImport "OpenCL.std"
               OpMemoryModel Physical64 OpenCL
               OpEntryPoint Kernel %main "dpas_trivial"
               OpExecutionMode %main SubgroupSize 16
     %void   = OpTypeVoid
     %uint   = OpTypeInt 32 0
     %ulong  = OpTypeInt 64 0
     %float  = OpTypeFloat 32
     %ushort = OpTypeInt 16 0
   %uint_0   = OpConstant %uint 0
   %uint_1   = OpConstant %uint 1
   %uint_2   = OpConstant %uint 2
   %uint_3   = OpConstant %uint 3
   %uint_8   = OpConstant %uint 8
  %uint_16  = OpConstant %uint 16
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
   %a = OpCooperativeMatrixLoadKHR %cm_a %buf_a %uint_0 %uint_16 None
   %b = OpCooperativeMatrixLoadKHR %cm_b %buf_b %uint_0 %uint_16 None
   %c = OpCooperativeMatrixLoadKHR %cm_acc %buf_c %uint_0 %uint_16 None
   %r = OpCooperativeMatrixMulAddKHR %cm_acc %a %b %c
               OpCooperativeMatrixStoreKHR %buf_c %r %uint_0 %uint_16 None
               OpReturn
               OpFunctionEnd
"""


def build_and_compile(spvasm_text, name):
    """Assemble and compile SPIR-V. Returns (spv_path, bin_path) or (None, None)."""
    spvasm = SCRIPT_DIR / f"sweep_{name}.spvasm"
    spv = SCRIPT_DIR / f"sweep_{name}.spv"
    spvasm.write_text(spvasm_text)

    r = run_cmd(f"spirv-as --target-env spv1.4 {spvasm} -o {spv}", check=False)
    if r.returncode != 0:
        print(f"  ASSEMBLE FAILED: {r.stderr[:200]}")
        return None, None

    r = run_cmd(f"ocloc compile -spirv_input -file {spv} -device {DEVICE} -output {name}", check=False)
    if r.returncode != 0 or "error" in (r.stderr + r.stdout).lower():
        print(f"  COMPILE FAILED: {(r.stderr + r.stdout)[:200]}")
        return None, None

    bin_path = SCRIPT_DIR / f"{name}_bmg.bin"
    if not bin_path.exists():
        print("  NO BINARY")
        return None, None
    return spv, bin_path


def cleanup(name, extra=None):
    spvasm = SCRIPT_DIR / f"sweep_{name}.spvasm"
    spv = SCRIPT_DIR / f"sweep_{name}.spv"
    bin_path = SCRIPT_DIR / f"{name}_bmg.bin"
    for f in [spvasm, spv, bin_path]:
        f.unlink(missing_ok=True)
    if extra:
        for f in extra:
            f.unlink(missing_ok=True)
    for f in SCRIPT_DIR.glob(f"{name}*"):
        f.unlink(missing_ok=True)


def run_benchmark(spv_path, kernel_name, wg_x, n_wg, repeats=50):
    """Run benchmark, return median ns or None."""
    r = run_cmd(f"{SPIRV_RUNNER} {spv_path} {kernel_name} {wg_x} {n_wg} {repeats}", check=False)
    if r.returncode != 0:
        return None, r.stderr[:200]
    m = re.search(r'Median=([\d.]+)\s+ns', r.stdout)
    if m:
        return float(m.group(1)), None
    return None, "parse failed"


def experiment_5a():
    """5A: SG/WG sweep with fixed total_SGs=1280, ILP=8 kernel."""
    print("\n" + "=" * 70)
    print("Experiment 5A: SG/WG Sweep (Fixed Total Work=1280 SGs)")
    print("=" * 70)

    N_ITER = 128
    ILP = 8
    TOTAL_SGS = 1280  # 160 EU × 8 threads

    spv, bin_path = build_and_compile(gen_ilp8_kernel(N_ITER), "ilp8_sched")
    if not spv:
        return []

    total_dpas_per_sg = ILP * N_ITER  # 1024
    total_flops_per_sg = total_dpas_per_sg * FLOPS_PER_DPAS

    print(f"  ILP={ILP}, N_ITER={N_ITER}, total_SGs={TOTAL_SGS}")
    print(f"  {total_dpas_per_sg} DPAS/SG, {total_flops_per_sg / 1e6:.1f} MFLOPs/SG")
    print(f"  {'SG/WG':>6s} {'WGs':>6s} {'WG_size':>8s} {'Median(ns)':>12s} {'TFLOPS':>10s} {'cyc/dpas':>10s}")
    print(f"  {'-' * 58}")

    results = []
    for sg_per_wg in [1, 2, 4, 8, 16, 32]:
        n_wg = TOTAL_SGS // sg_per_wg
        wg_x = sg_per_wg * 16  # threads per WG

        median, err = run_benchmark(spv, "dpas_ilp8", wg_x, n_wg, 50)
        if median is None:
            print(f"  {sg_per_wg:>6d} {n_wg:>6d} {wg_x:>8d} FAILED: {err}")
            continue

        total_flops = total_flops_per_sg * TOTAL_SGS
        tflops = total_flops / (median * 1e-9) / 1e12
        cyc = median * GHZ / (total_dpas_per_sg * TOTAL_SGS)

        print(f"  {sg_per_wg:>6d} {n_wg:>6d} {wg_x:>8d} {median:>12.0f} {tflops:>10.2f} {cyc:>10.2f}")
        results.append({
            'sg_per_wg': sg_per_wg, 'n_wg': n_wg, 'wg_x': wg_x,
            'median_ns': median, 'tflops': tflops, 'cyc_per_dpas': cyc
        })

    cleanup("ilp8_sched")
    return results


def experiment_5b():
    """5B: EU thread context switch — ILP=1, single WG, vary SGs."""
    print("\n" + "=" * 70)
    print("Experiment 5B: EU Thread Context Switch (ILP=1, Single WG)")
    print("=" * 70)

    N_ITER = 1024

    spv, bin_path = build_and_compile(gen_ilp1_kernel(N_ITER), "ilp1_tlp")
    if not spv:
        return []

    total_dpas_per_sg = N_ITER  # 1024
    print(f"  ILP=1, N_ITER={N_ITER}, n_wg=1 (single WG)")
    print(f"  Each SG does {total_dpas_per_sg} dependent DPAS")
    print(f"  {'SGs/WG':>7s} {'WG_size':>8s} {'Median(ns)':>12s} {'cyc/dpas':>10s} {'vs_33cyc':>10s}")
    print(f"  {'-' * 52}")

    results = []
    for sg_per_wg in [1, 2, 4, 8, 16]:
        wg_x = sg_per_wg * 16
        n_wg = 1

        median, err = run_benchmark(spv, "dpas_ilp1", wg_x, n_wg, 50)
        if median is None:
            print(f"  {sg_per_wg:>7d} {wg_x:>8d} FAILED: {err}")
            continue

        total_dpas = total_dpas_per_sg * sg_per_wg
        cyc = median * GHZ / total_dpas
        ratio = cyc / 33.0

        print(f"  {sg_per_wg:>7d} {wg_x:>8d} {median:>12.0f} {cyc:>10.1f} {ratio:>10.3f}x")
        results.append({
            'sg_per_wg': sg_per_wg, 'wg_x': wg_x,
            'median_ns': median, 'cyc_per_dpas': cyc,
            'ratio_to_latency': ratio
        })

    cleanup("ilp1_tlp")
    return results


def experiment_5c():
    """5C: Dispatch overhead — trivial 1-DPAS kernel, sweep n_wg."""
    print("\n" + "=" * 70)
    print("Experiment 5C: Dispatch Overhead Per WG (1-DPAS kernel)")
    print("=" * 70)

    spv, bin_path = build_and_compile(gen_trivial_1dpas(), "trivial_dpas")
    if not spv:
        return []

    print(f"  Trivial kernel: load + 1 DPAS + store")
    print(f"  {'n_wg':>6s} {'Median(ns)':>12s} {'ns/wg':>10s} {'slope':>10s}")
    print(f"  {'-' * 42}")

    results = []
    prev_ns = 0
    prev_wg = 0
    for n_wg in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
        median, err = run_benchmark(spv, "dpas_trivial", 16, n_wg, 100)
        if median is None:
            print(f"  {n_wg:>6d} FAILED: {err}")
            continue

        ns_per_wg = median / n_wg
        slope = ""
        if prev_wg > 0:
            delta_ns = median - prev_ns
            delta_wg = n_wg - prev_wg
            slope = f"{delta_ns / delta_wg:.1f}"

        print(f"  {n_wg:>6d} {median:>12.0f} {ns_per_wg:>10.1f} {slope:>10s}")
        results.append({
            'n_wg': n_wg, 'median_ns': median,
            'ns_per_wg': ns_per_wg
        })
        prev_ns = median
        prev_wg = n_wg

    # Linear regression for dispatch overhead
    if len(results) >= 3:
        n = len(results)
        sx = sum(r['n_wg'] for r in results)
        sy = sum(r['median_ns'] for r in results)
        sxx = sum(r['n_wg'] ** 2 for r in results)
        sxy = sum(r['n_wg'] * r['median_ns'] for r in results)
        denom = n * sxx - sx * sx
        if denom != 0:
            slope_val = (n * sxy - sx * sy) / denom
            intercept = (sy - slope_val * sx) / n
            print(f"\n  Linear regression: time = {slope_val:.1f} ns/WG × n_wg + {intercept:.0f} ns")
            print(f"  Per-WG dispatch cost: {slope_val:.1f} ns ({slope_val * GHZ:.0f} cycles)")
            print(f"  Fixed overhead: {intercept:.0f} ns ({intercept * GHZ:.0f} cycles)")

    cleanup("trivial_dpas")
    return results


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    if not SPIRV_RUNNER.exists():
        print("Building spirv_runner...")
        run_cmd("g++ -std=c++17 -O2 -o spirv_runner spirv_runner.cpp -lze_loader -lm")

    results_5a = experiment_5a()
    results_5b = experiment_5b()
    results_5c = experiment_5c()

    # Save results
    if results_5a:
        path = RESULTS_DIR / "thread_sched_sg_wg_sweep.csv"
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=results_5a[0].keys())
            w.writeheader()
            w.writerows(results_5a)
        print(f"\n5A saved to {path}")

    if results_5b:
        path = RESULTS_DIR / "thread_sched_eu_context_switch.csv"
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=results_5b[0].keys())
            w.writeheader()
            w.writerows(results_5b)
        print(f"5B saved to {path}")

    if results_5c:
        path = RESULTS_DIR / "thread_sched_dispatch_overhead.csv"
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=results_5c[0].keys())
            w.writeheader()
            w.writerows(results_5c)
        print(f"5C saved to {path}")


if __name__ == "__main__":
    main()
