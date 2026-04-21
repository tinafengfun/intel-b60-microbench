#!/usr/bin/env python3
"""
Phase 2: DPAS Scheduling Pattern Microbenchmarks
1C: DPAS + Independent ALU (SBID stall investigation)
3A: Barrier overhead (OpControlBarrier)
3B: Memory barrier vs control barrier
4A: DPAS with operand reload from global memory
4C: DPAS with store-to-global after each iteration
6A: DPAS + ALU throughput (independent pipelines)
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
    return subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)


def build_kernel(spvasm_text, name):
    """Assemble and compile. Returns spv_path or None."""
    spvasm = SCRIPT_DIR / f"sweep_{name}.spvasm"
    spv = SCRIPT_DIR / f"sweep_{name}.spv"
    spvasm.write_text(spvasm_text)

    r = run_cmd(f"spirv-as --target-env spv1.4 {spvasm} -o {spv}", check=False)
    if r.returncode != 0:
        print(f"  ASSEMBLE FAILED [{name}]: {r.stderr[:200]}")
        return None

    r = run_cmd(f"ocloc compile -spirv_input -file {spv} -device {DEVICE} -output {name}", check=False)
    if r.returncode != 0 or "error" in (r.stderr + r.stdout).lower():
        print(f"  COMPILE FAILED [{name}]: {(r.stderr + r.stdout)[:200]}")
        return None

    return spv


def run_benchmark(spv_path, kernel_name, wg_x, n_wg, repeats=50):
    """Run benchmark, return median ns or None."""
    r = run_cmd(f"{SPIRV_RUNNER} {spv_path} {kernel_name} {wg_x} {n_wg} {repeats}", check=False)
    if r.returncode != 0:
        return None, r.stderr[:200]
    m = re.search(r'Median=([\d.]+)\s+ns', r.stdout)
    if m:
        return float(m.group(1)), None
    return None, "parse failed"


def cleanup(name):
    for f in SCRIPT_DIR.glob(f"sweep_{name}*"):
        f.unlink(missing_ok=True)


# Common header builder
def make_header(desc, entry, extra_constants=""):
    return f"""; SPIR-V 1.4
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
{extra_constants}
%ptr_cross_uint   = OpTypePointer CrossWorkgroup %uint
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


def make_loop(n_iter, body_code):
    """Generate standard DPAS loop with given body."""
    return f"""   %a_tile = OpCooperativeMatrixLoadKHR %cm_a %buf_a %uint_0 %uint_16 None
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
{body_code}
    %i_next = OpIAdd %uint %i_phi %uint_1
               OpBranch %lh
%lx = OpLabel
               OpCooperativeMatrixStoreKHR %buf_c %acc_phi %uint_0 %uint_16 None
               OpReturn
               OpFunctionEnd
"""


def experiment_1c():
    """1C: DPAS + Independent ALU ops — SBID stall investigation."""
    print("\n" + "=" * 70)
    print("Experiment 1C: DPAS + Independent ALU (SBID Stall)")
    print("  Dependent DPAS chain + N independent scalar FP32 ALU ops per iter")
    print("=" * 70)

    N_ITER = 64
    results = []
    extra_const = f"  %uint_{N_ITER}  = OpConstant %uint {N_ITER}\n"
    extra_const += "   %float_1  = OpConstant %float 1.0\n"
    extra_const += "   %float_2  = OpConstant %float 2.0\n"

    print(f"\n  {'n_alu':>6s} {'N':>4s} {'Median(ns)':>12s} {'cyc/iter':>10s} {'delta_cyc':>10s}")
    print(f"  {'-' * 48}")

    base_cyc = None
    for n_alu in [0, 1, 2, 4, 8, 16]:
        name = f"alu{n_alu}"
        header = make_header(f"DPAS+{n_alu}ALU, {N_ITER}it", "dpas_alu", extra_const)

        # Load scalar from buf_c (element 0), use as ALU seed
        pre = "  %scalar_seed = OpCooperativeMatrixLoadKHR %cm_acc %buf_c %uint_0 %uint_16 None\n"
        # Actually, can't extract scalar from cooperative matrix easily.
        # Instead, just use constant values and rely on the dependency chain through phi.
        # The compiler won't fold a phi-dependent chain.

        # Body: DPAS + ALU
        body = "  %acc_next = OpCooperativeMatrixMulAddKHR %cm_acc %a_tile %b_tile %acc_phi\n"
        if n_alu == 0:
            body += "  %dummy = OpCopyObject %float %float_1\n"
        else:
            # Build a chain of FP ops using float_1 as base (not folded because we feed result back)
            body += "  %alu0 = OpFMul %float %float_2 %float_2\n"
            for j in range(1, n_alu):
                body += f"  %alu{j} = OpFMul %float %alu{j-1} %alu{j-1}\n"

        loop = make_loop(N_ITER, body)
        # Replace the store to also write the ALU result (prevent DCE)
        # Actually we just need to keep the DPAS chain. The ALU ops are independent.

        spv = build_kernel(header + loop, name)
        if not spv:
            continue

        median, err = run_benchmark(spv, "dpas_alu", 16, 1, 50)
        if median is None:
            print(f"  {n_alu:>6d} {N_ITER:>4d} FAILED: {err}")
            cleanup(name)
            continue

        cyc = median * GHZ / N_ITER
        delta = cyc - base_cyc if base_cyc is not None else 0
        if base_cyc is None:
            base_cyc = cyc
        print(f"  {n_alu:>6d} {N_ITER:>4d} {median:>12.0f} {cyc:>10.1f} {delta:>10.1f}")
        results.append({'n_alu': n_alu, 'n_iter': N_ITER, 'median_ns': median,
                        'cyc_per_iter': cyc, 'delta_cyc': delta})
        cleanup(name)

    return results


def experiment_3a():
    """3A: Barrier overhead — OpControlBarrier with varying SG counts."""
    print("\n" + "=" * 70)
    print("Experiment 3A: Barrier Overhead (OpControlBarrier)")
    print("  DPAS + barrier every iter vs DPAS only, varying SGs per WG")
    print("=" * 70)

    N_ITER = 64
    extra_const = f"  %uint_{N_ITER}  = OpConstant %uint {N_ITER}\n"

    results = []

    # First: DPAS-only baselines
    print(f"\n  --- DPAS-only baselines ---")
    print(f"  {'SGs':>4s} {'WG_size':>8s} {'Median(ns)':>12s} {'cyc/dpas':>10s}")
    print(f"  {'-' * 38}")

    baselines = {}
    for n_sgs in [1, 2, 4, 8, 16]:
        name = f"nobar_{n_sgs}"
        header = make_header(f"DPAS only, {n_sgs} SGs", "dpas_nobar", extra_const)
        body = "  %acc_next = OpCooperativeMatrixMulAddKHR %cm_acc %a_tile %b_tile %acc_phi\n"
        spv = build_kernel(header + make_loop(N_ITER, body), name)
        if not spv:
            continue
        wg_x = n_sgs * 16
        median, err = run_benchmark(spv, "dpas_nobar", wg_x, 1, 50)
        if median:
            cyc = median * GHZ / (N_ITER * n_sgs)
            baselines[n_sgs] = (median, cyc)
            print(f"  {n_sgs:>4d} {wg_x:>8d} {median:>12.0f} {cyc:>10.1f}")
        cleanup(name)

    # Now: DPAS + barrier every iteration
    print(f"\n  --- DPAS + barrier every iter ---")
    print(f"  {'SGs':>4s} {'WG_size':>8s} {'Median(ns)':>12s} {'cyc/dpas':>10s} {'bar_cyc':>10s}")
    print(f"  {'-' * 48}")

    for n_sgs in [1, 2, 4, 8, 16]:
        name = f"bar_{n_sgs}"
        header = make_header(f"DPAS+barrier, {n_sgs} SGs", "dpas_bar", extra_const)
        body = "  %acc_next = OpCooperativeMatrixMulAddKHR %cm_acc %a_tile %b_tile %acc_phi\n"
        body += "               OpControlBarrier %uint_2 %uint_2 %uint_0\n"
        spv = build_kernel(header + make_loop(N_ITER, body), name)
        if not spv:
            continue
        wg_x = n_sgs * 16
        median, err = run_benchmark(spv, "dpas_bar", wg_x, 1, 50)
        if median:
            cyc = median * GHZ / (N_ITER * n_sgs)
            base_cyc = baselines.get(n_sgs, (0, 0))[1]
            bar_cyc = cyc - base_cyc
            print(f"  {n_sgs:>4d} {wg_x:>8d} {median:>12.0f} {cyc:>10.1f} {bar_cyc:>10.1f}")
            results.append({
                'n_sgs': n_sgs, 'wg_x': wg_x, 'median_ns': median,
                'cyc_per_dpas': cyc, 'barrier_overhead_cyc': bar_cyc
            })
        cleanup(name)

    return results


def experiment_3b():
    """3B: Memory barrier vs control barrier (single SG)."""
    print("\n" + "=" * 70)
    print("Experiment 3B: MemoryBarrier vs ControlBarrier (single SG)")
    print("=" * 70)

    N_ITER = 64
    extra_const = f"  %uint_{N_ITER}  = OpConstant %uint {N_ITER}\n"
    results = []

    configs = [
        ("none", "dpas_nothing", "  %acc_next = OpCooperativeMatrixMulAddKHR %cm_acc %a_tile %b_tile %acc_phi\n"),
        ("control_barrier", "dpas_cbar", "  %acc_next = OpCooperativeMatrixMulAddKHR %cm_acc %a_tile %b_tile %acc_phi\n               OpControlBarrier %uint_2 %uint_2 %uint_0\n"),
        ("memory_barrier", "dpas_mbar", "  %acc_next = OpCooperativeMatrixMulAddKHR %cm_acc %a_tile %b_tile %acc_phi\n               OpMemoryBarrier %uint_1 %uint_72\n"),
    ]

    print(f"\n  {'type':>20s} {'Median(ns)':>12s} {'cyc/iter':>10s} {'delta_cyc':>10s}")
    print(f"  {'-' * 56}")

    base_cyc = None
    for label, entry, body in configs:
        name = label.replace(" ", "_")
        header = make_header(f"DPAS + {label}", entry, extra_const)
        spv = build_kernel(header + make_loop(N_ITER, body), name)
        if not spv:
            continue
        median, _ = run_benchmark(spv, entry, 16, 1, 50)
        if median:
            cyc = median * GHZ / N_ITER
            delta = cyc - base_cyc if base_cyc is not None else 0
            if base_cyc is None:
                base_cyc = cyc
            print(f"  {label:>20s} {median:>12.0f} {cyc:>10.1f} {delta:>10.1f}")
            results.append({'type': label, 'median_ns': median, 'cyc_per_iter': cyc, 'delta_cyc': delta})
        cleanup(name)

    return results


def experiment_4a():
    """4A: DPAS with operand reload from global memory."""
    print("\n" + "=" * 70)
    print("Experiment 4A: DPAS with Operand Reload from Global Memory")
    print("=" * 70)

    N_ITER = 64
    extra_const = f"  %uint_{N_ITER}  = OpConstant %uint {N_ITER}\n"
    results = []

    configs = [
        ("none", False, False),
        ("A only", True, False),
        ("B only", False, True),
        ("A+B", True, True),
    ]

    print(f"\n  {'reload':>8s} {'Median(ns)':>12s} {'cyc/iter':>10s} {'delta_cyc':>10s}")
    print(f"  {'-' * 44}")

    base_cyc = None
    for label, reload_a, reload_b in configs:
        name = f"rld_{label.replace(' ','_').replace('+','')}"
        header = make_header(f"DPAS reload {label}", "dpas_rld", extra_const)

        # Pre-load tiles that are NOT reloaded
        pre = ""
        if not reload_a:
            pre += "   %a_tile = OpCooperativeMatrixLoadKHR %cm_a %buf_a %uint_0 %uint_16 None\n"
        if not reload_b:
            pre += "   %b_tile = OpCooperativeMatrixLoadKHR %cm_b %buf_b %uint_0 %uint_16 None\n"
        pre += "  %acc_init = OpCooperativeMatrixLoadKHR %cm_acc %buf_c %uint_0 %uint_16 None\n"
        pre += "               OpBranch %lh\n"

        # Loop body
        body = ""
        if reload_a:
            body += "   %a_cur = OpCooperativeMatrixLoadKHR %cm_a %buf_a %uint_0 %uint_16 None\n"
        if reload_b:
            body += "   %b_cur = OpCooperativeMatrixLoadKHR %cm_b %buf_b %uint_0 %uint_16 None\n"
        a = "%a_cur" if reload_a else "%a_tile"
        b = "%b_cur" if reload_b else "%b_tile"
        body += f"  %acc_next = OpCooperativeMatrixMulAddKHR %cm_acc {a} {b} %acc_phi\n"

        # Full loop
        loop = f"""{pre}
%lh = OpLabel
               OpLoopMerge %lx %lh None
  %acc_phi = OpPhi %cm_acc %acc_init %entry %acc_next %lb
    %i_phi = OpPhi %uint %uint_0 %entry %i_next %lb
      %cond = OpULessThan %bool %i_phi %uint_{N_ITER}
               OpBranchConditional %cond %lb %lx
%lb = OpLabel
{body}
    %i_next = OpIAdd %uint %i_phi %uint_1
               OpBranch %lh
%lx = OpLabel
               OpCooperativeMatrixStoreKHR %buf_c %acc_phi %uint_0 %uint_16 None
               OpReturn
               OpFunctionEnd
"""
        spv = build_kernel(header + loop, name)
        if not spv:
            continue
        median, _ = run_benchmark(spv, "dpas_rld", 16, 1, 50)
        if median:
            cyc = median * GHZ / N_ITER
            delta = cyc - base_cyc if base_cyc is not None else 0
            if base_cyc is None:
                base_cyc = cyc
            print(f"  {label:>8s} {median:>12.0f} {cyc:>10.1f} {delta:>10.1f}")
            results.append({'reload': label, 'median_ns': median, 'cyc_per_iter': cyc, 'delta_cyc': delta})
        cleanup(name)

    return results


def experiment_4c():
    """4C: DPAS with store-to-global at varying frequencies."""
    print("\n" + "=" * 70)
    print("Experiment 4C: DPAS with Store-to-Global (frequency sweep)")
    print("=" * 70)

    N_ITER = 64
    results = []

    print(f"\n  {'interval':>10s} {'Median(ns)':>12s} {'cyc/iter':>10s} {'delta_cyc':>10s}")
    print(f"  {'-' * 46}")

    base_cyc = None
    for interval in [0, 1, 2, 4, 8, 16, 64]:
        name = f"sto_{interval}"
        extra_const = f"  %uint_{N_ITER}  = OpConstant %uint {N_ITER}\n"

        if interval == 0:
            # No store during loop
            header = make_header("DPAS no store", "dpas_sto", extra_const)
            body = "  %acc_next = OpCooperativeMatrixMulAddKHR %cm_acc %a_tile %b_tile %acc_phi\n"
            spv = build_kernel(header + make_loop(N_ITER, body), name)
        elif interval == 1:
            # Store every iteration
            header = make_header("DPAS store every", "dpas_sto", extra_const)
            body = "  %acc_next = OpCooperativeMatrixMulAddKHR %cm_acc %a_tile %b_tile %acc_phi\n"
            body += "               OpCooperativeMatrixStoreKHR %buf_c %acc_next %uint_0 %uint_16 None\n"
            spv = build_kernel(header + make_loop(N_ITER, body), name)
        else:
            # Store every Nth iteration using modulo
            extra_const += f"  %uint_{interval}  = OpConstant %uint {interval}\n"
            header = make_header(f"DPAS store every {interval}", "dpas_sto", extra_const)
            # Need custom loop with conditional store
            loop = f"""   %a_tile = OpCooperativeMatrixLoadKHR %cm_a %buf_a %uint_0 %uint_16 None
   %b_tile = OpCooperativeMatrixLoadKHR %cm_b %buf_b %uint_0 %uint_16 None
  %acc_init = OpCooperativeMatrixLoadKHR %cm_acc %buf_c %uint_0 %uint_16 None
               OpBranch %lh
%lh = OpLabel
               OpLoopMerge %lx %lh None
  %acc_phi = OpPhi %cm_acc %acc_init %entry %acc_next %merge
    %i_phi = OpPhi %uint %uint_0 %entry %i_next %merge
      %cond = OpULessThan %bool %i_phi %uint_{N_ITER}
               OpBranchConditional %cond %lb %lx
%lb = OpLabel
  %acc_next = OpCooperativeMatrixMulAddKHR %cm_acc %a_tile %b_tile %acc_phi
  %rem = OpUMod %uint %i_phi %uint_{interval}
  %is_store = OpIEqual %bool %rem %uint_0
               OpBranchConditional %is_store %st %nost
%st = OpLabel
               OpCooperativeMatrixStoreKHR %buf_c %acc_next %uint_0 %uint_16 None
               OpBranch %merge
%nost = OpLabel
               OpBranch %merge
%merge = OpLabel
    %i_next = OpIAdd %uint %i_phi %uint_1
               OpBranch %lh
%lx = OpLabel
               OpCooperativeMatrixStoreKHR %buf_c %acc_phi %uint_0 %uint_16 None
               OpReturn
               OpFunctionEnd
"""
            spv = build_kernel(header + loop, name)

        if not spv:
            continue
        median, _ = run_benchmark(spv, "dpas_sto", 16, 1, 50)
        if median:
            cyc = median * GHZ / N_ITER
            delta = cyc - base_cyc if base_cyc is not None else 0
            if base_cyc is None:
                base_cyc = cyc
            label = "never" if interval == 0 else f"every {interval}"
            print(f"  {label:>10s} {median:>12.0f} {cyc:>10.1f} {delta:>10.1f}")
            results.append({'store_interval': interval, 'median_ns': median,
                            'cyc_per_iter': cyc, 'delta_cyc': delta})
        cleanup(name)

    return results


def experiment_6a():
    """6A: DPAS + ALU throughput with multiple WGs."""
    print("\n" + "=" * 70)
    print("Experiment 6A: DPAS + ALU Throughput (ILP=8, multi-SG)")
    print("  Can Xe2 EU dual-issue XMX + ALU operations?")
    print("=" * 70)

    N_ITER = 64
    ILP = 8
    results = []

    for n_alu in [0, 2, 4, 8]:
        name = f"thru_a{n_alu}"
        extra_const = f"  %uint_{N_ITER}  = OpConstant %uint {N_ITER}\n"
        extra_const += "   %float_1  = OpConstant %float 1.0\n"
        extra_const += "   %float_2  = OpConstant %float 2.0\n"

        header = make_header(f"ILP=8 DPAS+{n_alu}ALU", "dpas_thru", extra_const)

        # ILP=8 loads
        pre = ""
        for i in range(ILP):
            pre += f"  %a_tile{i} = OpCooperativeMatrixLoadKHR %cm_a %buf_a %uint_0 %uint_16 None\n"
            pre += f"  %b_tile{i} = OpCooperativeMatrixLoadKHR %cm_b %buf_b %uint_0 %uint_16 None\n"
            pre += f"  %acc_init{i} = OpCooperativeMatrixLoadKHR %cm_acc %buf_c %uint_0 %uint_16 None\n"
        pre += "               OpBranch %lh\n"

        # Loop header
        loop_h = "%lh = OpLabel\n               OpLoopMerge %lx %lh None\n"
        for i in range(ILP):
            loop_h += f"  %acc_phi{i} = OpPhi %cm_acc %acc_init{i} %entry %acc_next{i} %lb\n"
        loop_h += "    %i_phi = OpPhi %uint %uint_0 %entry %i_next %lb\n"
        loop_h += f"      %cond = OpULessThan %bool %i_phi %uint_{N_ITER}\n"
        loop_h += "               OpBranchConditional %cond %lb %lx\n"

        # Loop body
        body = "%lb = OpLabel\n"
        for i in range(ILP):
            body += f"  %acc_next{i} = OpCooperativeMatrixMulAddKHR %cm_acc %a_tile{i} %b_tile{i} %acc_phi{i}\n"
        # Independent ALU
        if n_alu > 0:
            body += "  %alu0 = OpFMul %float %float_2 %float_2\n"
            for j in range(1, n_alu):
                body += f"  %alu{j} = OpFMul %float %alu{j-1} %alu{j-1}\n"
        body += "    %i_next = OpIAdd %uint %i_phi %uint_1\n"
        body += "               OpBranch %lh\n"

        # Exit
        exit_block = "%lx = OpLabel\n"
        exit_block += "               OpCooperativeMatrixStoreKHR %buf_c %acc_phi0 %uint_0 %uint_16 None\n"
        exit_block += "               OpReturn\n               OpFunctionEnd\n"

        spv = build_kernel(header + pre + loop_h + body + exit_block, name)
        if not spv:
            continue

        for n_wg in [1, 64, 256, 1024]:
            median, err = run_benchmark(spv, "dpas_thru", 16, n_wg, 50)
            if median:
                total_dpas = ILP * N_ITER * n_wg
                tflops = total_dpas * FLOPS_PER_DPAS / (median * 1e-9) / 1e12
                cyc = median * GHZ / total_dpas
                print(f"  alu={n_alu:>2d} wg={n_wg:>5d}  {median:>10.0f} ns  {tflops:>8.2f} TFLOPS  {cyc:>8.2f} cyc")
                results.append({
                    'n_alu': n_alu, 'n_wg': n_wg,
                    'median_ns': median, 'tflops': tflops, 'cyc_per_dpas': cyc
                })
        cleanup(name)

    return results


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    if not SPIRV_RUNNER.exists():
        print("Building spirv_runner...")
        run_cmd("g++ -std=c++17 -O2 -o spirv_runner spirv_runner.cpp -lze_loader -lm")

    results_1c = experiment_1c()
    results_3a = experiment_3a()
    results_3b = experiment_3b()
    results_4a = experiment_4a()
    results_4c = experiment_4c()
    results_6a = experiment_6a()

    saves = [
        ("dpas_alu_interleave.csv", results_1c),
        ("dpas_barrier_overhead.csv", results_3a),
        ("dpas_barrier_vs_membar.csv", results_3b),
        ("dpas_reload_overhead.csv", results_4a),
        ("dpas_store_overhead.csv", results_4c),
        ("dpas_alu_throughput.csv", results_6a),
    ]
    for fname, data in saves:
        if data:
            path = RESULTS_DIR / fname
            with open(path, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=data[0].keys())
                w.writeheader()
                w.writerows(data)
            print(f"Saved to {path}")


if __name__ == "__main__":
    main()
