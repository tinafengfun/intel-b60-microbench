#!/usr/bin/env python3
"""
Statistical rigor pass: Re-run DPAS latency sweep with R², std dev, CI.
Also runs reload vs no-reload comparison with GEN ASM disassembly.
"""
import subprocess, sys, re, csv, shutil
from pathlib import Path
import math

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


def build_kernel(spvasm_text, name):
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
    return spv, bin_path


def run_benchmark_full(spv_path, kernel_name, wg_x, n_wg, repeats=100):
    """Run benchmark and parse full statistics."""
    r = run_cmd(f"{SPIRV_RUNNER} {spv_path} {kernel_name} {wg_x} {n_wg} {repeats}", check=False)
    if r.returncode != 0:
        return None
    result = {}
    m = re.search(r'Median=([\d.]+)\s+ns', r.stdout)
    if m: result['median'] = float(m.group(1))
    m = re.search(r'Mean=([\d.]+)\s+ns', r.stdout)
    if m: result['mean'] = float(m.group(1))
    m = re.search(r'StdDev=([\d.]+)\s+ns', r.stdout)
    if m: result['stddev'] = float(m.group(1))
    m = re.search(r'CV=([\d.]+)%', r.stdout)
    if m: result['cv'] = float(m.group(1))
    m = re.search(r'95%CI=\[([\d.]+),\s*([\d.]+)\]', r.stdout)
    if m:
        result['ci_lo'] = float(m.group(1))
        result['ci_hi'] = float(m.group(2))
    return result if 'median' in result else None


def cleanup(name):
    for f in SCRIPT_DIR.glob(f"sweep_{name}*"):
        f.unlink(missing_ok=True)


def disassemble(bin_path, name):
    """Disassemble and return GEN ASM text."""
    disasm_dir = SCRIPT_DIR / f"disasm_{name}"
    run_cmd(f"mkdir -p {disasm_dir}")
    run_cmd(f"ocloc disasm -file {bin_path} -dump {disasm_dir} -device {DEVICE}", check=False)
    for f in disasm_dir.iterdir():
        if f.name.endswith('.asm'):
            text = f.read_text()
            shutil.rmtree(disasm_dir, ignore_errors=True)
            return text
    shutil.rmtree(disasm_dir, ignore_errors=True)
    return ""


def linear_regression(xs, ys):
    """Linear regression with R², slope std error."""
    n = len(xs)
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x*x for x in xs)
    sxy = sum(x*y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if denom == 0:
        return None
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n

    # R²
    y_mean = sy / n
    ss_tot = sum((y - y_mean)**2 for y in ys)
    ss_res = sum((y - (slope * x + intercept))**2 for x, y in zip(xs, ys))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Slope standard error
    mse = ss_res / (n - 2) if n > 2 else 0
    slope_se = math.sqrt(mse / (sxx - sx*sx/n)) if (sxx - sx*sx/n) > 0 else 0

    return {
        'slope': slope, 'intercept': intercept,
        'r_squared': r_squared, 'slope_se': slope_se,
        'slope_ci_95': 2.0 * slope_se  # approximate t_{0.025,n-2} ≈ 2
    }


DPAS_LATENCY_TEMPLATE = """; SPIR-V 1.4
; DPAS Latency: {n} dependent operations
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


def experiment_latency_with_stats():
    """Re-run latency sweep with full statistics."""
    print("=" * 70)
    print("DPAS Latency Sweep with Statistical Analysis")
    print("=" * 70)

    n_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    repeats = 100
    results = []

    print(f"\n  {'N':>4s} {'Median':>10s} {'Mean':>10s} {'StdDev':>10s} {'CV%':>6s} {'95%CI':>20s} {'DPAS_ASM':>10s}")
    print(f"  {'-' * 76}")

    for n in n_values:
        name = f"latstat_{n}"
        spvasm = DPAS_LATENCY_TEMPLATE.format(n=n)
        spv, bin_path = build_kernel(spvasm, name)
        if not spv:
            continue

        # Count dpas in GEN ASM
        asm = disassemble(bin_path, name) if bin_path else ""
        dpas_count = asm.count("dpas") if asm else 0

        stats = run_benchmark_full(spv, "dpas_latency", 16, 1, repeats)
        if stats:
            ci_str = f"[{stats.get('ci_lo',0):.0f}, {stats.get('ci_hi',0):.0f}]"
            print(f"  {n:>4d} {stats['median']:>10.0f} {stats['mean']:>10.0f} "
                  f"{stats.get('stddev',0):>10.1f} {stats.get('cv',0):>6.1f} "
                  f"{ci_str:>20s} {dpas_count:>10d}")
            results.append({
                'n': n, 'median_ns': stats['median'], 'mean_ns': stats['mean'],
                'stddev_ns': stats.get('stddev', 0), 'cv_pct': stats.get('cv', 0),
                'ci_lo': stats.get('ci_lo', 0), 'ci_hi': stats.get('ci_hi', 0),
                'dpas_in_asm': dpas_count
            })

        cleanup(name)

    # Linear regression
    if len(results) >= 3:
        xs = [r['n'] for r in results]
        ys = [r['median_ns'] for r in results]
        reg = linear_regression(xs, ys)
        if reg:
            slope_cyc = reg['slope'] * GHZ
            slope_cyc_ci = reg['slope_ci_95'] * GHZ
            print(f"\n  Linear Regression (slope method):")
            print(f"    slope  = {reg['slope']:.2f} ns/dpas = {slope_cyc:.1f} cycles/dpas")
            print(f"    ±{reg['slope_ci_95']:.2f} ns = ±{slope_cyc_ci:.1f} cycles (95% CI)")
            print(f"    intercept = {reg['intercept']:.0f} ns = {reg['intercept']*GHZ:.0f} cycles")
            print(f"    R² = {reg['r_squared']:.6f}")
            print(f"    slope SE = {reg['slope_se']:.3f} ns")

    return results


def experiment_reload_genasm():
    """Disassemble and compare reload vs no-reload kernels."""
    print("\n" + "=" * 70)
    print("GEN ASM Comparison: Reload vs No-Reload")
    print("=" * 70)

    N_ITER = 64

    # Kernel template (common header)
    header = """; SPIR-V 1.4
; {desc}
               OpCapability Addresses
               OpCapability Kernel
               OpCapability Int64
               OpCapability Int16
               OpCapability CooperativeMatrixKHR
               OpExtension "SPV_KHR_cooperative_matrix"
          %1 = OpExtInstImport "OpenCL.std"
               OpMemoryModel Physical64 OpenCL
               OpEntryPoint Kernel %main "{entry}"
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
  %uint_16   = OpConstant %uint 16
  %uint_64   = OpConstant %uint 64
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
"""

    # No reload kernel
    noreload_asm_text = header.format(desc="No reload", entry="noreload")
    noreload_asm_text += """   %a_tile = OpCooperativeMatrixLoadKHR %cm_a %buf_a %uint_0 %uint_16 None
   %b_tile = OpCooperativeMatrixLoadKHR %cm_b %buf_b %uint_0 %uint_16 None
  %acc_init = OpCooperativeMatrixLoadKHR %cm_acc %buf_c %uint_0 %uint_16 None
               OpBranch %lh
%lh = OpLabel
               OpLoopMerge %lx %lh None
  %acc_phi = OpPhi %cm_acc %acc_init %entry %acc_next %lb
    %i_phi = OpPhi %uint %uint_0 %entry %i_next %lb
      %cond = OpULessThan %bool %i_phi %uint_64
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

    # Reload A+B kernel
    reload_asm_text = header.format(desc="Reload A+B", entry="reload")
    reload_asm_text += """  %acc_init = OpCooperativeMatrixLoadKHR %cm_acc %buf_c %uint_0 %uint_16 None
               OpBranch %lh
%lh = OpLabel
               OpLoopMerge %lx %lh None
  %acc_phi = OpPhi %cm_acc %acc_init %entry %acc_next %lb
    %i_phi = OpPhi %uint %uint_0 %entry %i_next %lb
      %cond = OpULessThan %bool %i_phi %uint_64
               OpBranchConditional %cond %lb %lx
%lb = OpLabel
   %a_cur = OpCooperativeMatrixLoadKHR %cm_a %buf_a %uint_0 %uint_16 None
   %b_cur = OpCooperativeMatrixLoadKHR %cm_b %buf_b %uint_0 %uint_16 None
  %acc_next = OpCooperativeMatrixMulAddKHR %cm_acc %a_cur %b_cur %acc_phi
    %i_next = OpIAdd %uint %i_phi %uint_1
               OpBranch %lh
%lx = OpLabel
               OpCooperativeMatrixStoreKHR %buf_c %acc_phi %uint_0 %uint_16 None
               OpReturn
               OpFunctionEnd
"""

    for label, spvasm in [("noreload", noreload_asm_text), ("reload_ab", reload_asm_text)]:
        spv, bin_path = build_kernel(spvasm, f"cmp_{label}")
        if not spv or not bin_path:
            continue

        # Disassemble
        asm = disassemble(bin_path, f"cmp_{label}")
        if not asm:
            continue

        dpas_count = asm.count("dpas")
        send_count = asm.count("send")
        mov_count = asm.count("mov")
        total_lines = len(asm.split('\n'))

        # Count instructions in loop body (between loop label and back-branch)
        lines = asm.split('\n')
        loop_start = -1
        loop_end = -1
        for i, line in enumerate(lines):
            if 'dpas' in line.lower() and loop_start == -1:
                loop_start = i
            if loop_start >= 0 and ('jmpi' in line.lower() or 'jmp' in line.lower()):
                if loop_end == -1:
                    loop_end = i

        print(f"\n  === {label.upper()} ===")
        print(f"  Total: {total_lines} lines, {dpas_count} dpas, {send_count} send, {mov_count} mov")
        print(f"  GEN ASM loop body ({max(0,loop_end-loop_start)} lines around dpas):")
        if loop_start >= 0:
            for i in range(max(0, loop_start-2), min(len(lines), loop_end+3)):
                line = lines[i].strip()
                if line:
                    print(f"    {line}")

        # Also run the benchmark with stats
        stats = run_benchmark_full(spv, label, 16, 1, 100)
        if stats:
            cyc = stats['mean'] * GHZ / N_ITER
            print(f"  Timing: {stats['mean']:.0f} ± {stats.get('stddev',0):.0f} ns "
                  f"(CV={stats.get('cv',0):.1f}%) = {cyc:.1f} cyc/iter")

        cleanup(f"cmp_{label}")


def experiment_barrier_genasm():
    """Disassemble barrier kernel to understand throughput improvement."""
    print("\n" + "=" * 70)
    print("GEN ASM Analysis: Barrier Impact on Throughput")
    print("=" * 70)

    # ILP=8 kernel (the one used in barrier freq sweep)
    ILP = 8
    N_ITER = 128
    extra = f"  %uint_{N_ITER}  = OpConstant %uint {N_ITER}\n"

    header = f"""; SPIR-V 1.4
; ILP=8 throughput kernel
               OpCapability Addresses
               OpCapability Kernel
               OpCapability Int64
               OpCapability Int16
               OpCapability CooperativeMatrixKHR
               OpExtension "SPV_KHR_cooperative_matrix"
          %1 = OpExtInstImport "OpenCL.std"
               OpMemoryModel Physical64 OpenCL
               OpEntryPoint Kernel %main "dpas_thru"
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
  %uint_16   = OpConstant %uint 16
{extra}
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
"""

    # Build no-barrier ILP=8 kernel
    pre = ""
    for i in range(ILP):
        pre += f"  %a_tile{i} = OpCooperativeMatrixLoadKHR %cm_a %buf_a %uint_0 %uint_16 None\n"
        pre += f"  %b_tile{i} = OpCooperativeMatrixLoadKHR %cm_b %buf_b %uint_0 %uint_16 None\n"
        pre += f"  %acc_init{i} = OpCooperativeMatrixLoadKHR %cm_acc %buf_c %uint_0 %uint_16 None\n"
    pre += "               OpBranch %lh\n"

    loop_h = "%lh = OpLabel\n               OpLoopMerge %lx %lh None\n"
    for i in range(ILP):
        loop_h += f"  %acc_phi{i} = OpPhi %cm_acc %acc_init{i} %entry %acc_next{i} %lb\n"
    loop_h += "    %i_phi = OpPhi %uint %uint_0 %entry %i_next %lb\n"
    loop_h += f"      %cond = OpULessThan %bool %i_phi %uint_{N_ITER}\n"
    loop_h += "               OpBranchConditional %cond %lb %lx\n"

    # No-barrier variant
    body_nobar = "%lb = OpLabel\n"
    for i in range(ILP):
        body_nobar += f"  %acc_next{i} = OpCooperativeMatrixMulAddKHR %cm_acc %a_tile{i} %b_tile{i} %acc_phi{i}\n"
    body_nobar += "    %i_next = OpIAdd %uint %i_phi %uint_1\n"
    body_nobar += "               OpBranch %lh\n"

    exit_blk = "%lx = OpLabel\n"
    exit_blk += "               OpCooperativeMatrixStoreKHR %buf_c %acc_phi0 %uint_0 %uint_16 None\n"
    exit_blk += "               OpReturn\n               OpFunctionEnd\n"

    # Disassemble no-barrier kernel
    name = "bar_analysis_nobar"
    spv, bin_path = build_kernel(header + pre + loop_h + body_nobar + exit_blk, name)
    if spv and bin_path:
        asm = disassemble(bin_path, name)
        if asm:
            dpas_count = asm.count("dpas")
            send_count = asm.count("send")
            mov_count = asm.count("mov")
            # Check for hidden memory accesses in loop
            lines = asm.split('\n')
            has_send_in_loop = False
            in_loop = False
            for line in lines:
                if 'dpas' in line.lower():
                    in_loop = True
                if in_loop and 'send' in line.lower():
                    has_send_in_loop = True
                if in_loop and ('jmpi' in line.lower() or 'ret' in line.lower()):
                    break

            print(f"\n  ILP=8 NO BARRIER:")
            print(f"    dpas: {dpas_count}, send: {send_count}, mov: {mov_count}")
            print(f"    send instructions IN loop body: {'YES' if has_send_in_loop else 'NO'}")
            print(f"    Total GEN ASM lines: {len(lines)}")

            # Print first 20 lines around the loop
            for i, line in enumerate(lines):
                if 'dpas' in line.lower():
                    for j in range(max(0,i-3), min(len(lines), i+15)):
                        print(f"    {lines[j].rstrip()}")
                    break
    cleanup(name)

    # With-barrier variant
    body_bar = "%lb = OpLabel\n"
    for i in range(ILP):
        body_bar += f"  %acc_next{i} = OpCooperativeMatrixMulAddKHR %cm_acc %a_tile{i} %b_tile{i} %acc_phi{i}\n"
    body_bar += "               OpControlBarrier %uint_2 %uint_2 %uint_0\n"
    body_bar += "    %i_next = OpIAdd %uint %i_phi %uint_1\n"
    body_bar += "               OpBranch %lh\n"

    name = "bar_analysis_bar"
    spv, bin_path = build_kernel(header + pre + loop_h + body_bar + exit_blk, name)
    if spv and bin_path:
        asm = disassemble(bin_path, name)
        if asm:
            dpas_count = asm.count("dpas")
            send_count = asm.count("send")
            barrier_count = asm.count("barrier") + asm.count("sync") + asm.count("wait")
            lines = asm.split('\n')

            print(f"\n  ILP=8 WITH BARRIER every iter:")
            print(f"    dpas: {dpas_count}, send: {send_count}")
            print(f"    barrier/sync/wait: {barrier_count}")

            # Print around barrier instruction
            for i, line in enumerate(lines):
                if 'dpas' in line.lower():
                    for j in range(max(0,i-3), min(len(lines), i+25)):
                        print(f"    {lines[j].rstrip()}")
                    break
    cleanup(name)

    # Run both with stats to confirm timing
    print(f"\n  Timing comparison (8 SGs, 160 WGs, 50 reps):")
    for label, body in [("no_barrier", body_nobar), ("every_iter", body_bar)]:
        name = f"bar_timing_{label}"
        spv, _ = build_kernel(header + pre + loop_h + body + exit_blk, name)
        if spv:
            stats = run_benchmark_full(spv, "dpas_thru", 128, 160, 50)
            if stats:
                total_dpas = ILP * N_ITER * 8 * 160  # 8 SGs per WG
                tflops = total_dpas * FLOPS_PER_DPAS / (stats['mean'] * 1e-9) / 1e12
                print(f"    {label:>15s}: {stats['mean']:.0f} ± {stats.get('stddev',0):.0f} ns "
                      f"({tflops:.1f} TFLOPS, CV={stats.get('cv',0):.1f}%)")
        cleanup(name)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    if not SPIRV_RUNNER.exists():
        print("Building spirv_runner...")
        run_cmd("g++ -std=c++17 -O2 -o spirv_runner spirv_runner.cpp -lze_loader -lm")

    results = experiment_latency_with_stats()

    if results:
        path = RESULTS_DIR / "dpas_latency_with_stats.csv"
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)
        print(f"\nSaved to {path}")

    experiment_reload_genasm()
    experiment_barrier_genasm()


if __name__ == "__main__":
    main()
