#!/usr/bin/env python3
"""
Phase 3: Multi-SG Coordination Microbenchmarks
2A: Cross-SG dependency chain via global memory
3C: Barrier frequency sweep (DPAS throughput impact)
4B: DPAS with SLM operand staging (via global load → shared tile)
6B: DPAS + prefetch overlap (software pipelining)
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
  %entry_label = OpLabel
"""


def experiment_2a():
    """2A: Cross-SG dependency via global memory.
    Two sub-groups in one WG: SG0 does DPAS, stores to global, barrier.
    SG1 loads from global, does DPAS, stores to global, barrier.
    Measures: DPAS + 2 stores + barrier + 2 loads per round.
    """
    print("\n" + "=" * 70)
    print("Experiment 2A: Cross-SG Dependency Chain via Global Memory")
    print("  SG0: DPAS → store → barrier | SG1: load → DPAS → store → barrier")
    print("  Compare with independent DPAS (no cross-SG dep)")
    print("=" * 70)

    N_ROUNDS = 32
    results = []

    extra = f"  %uint_{N_ROUNDS}  = OpConstant %uint {N_ROUNDS}\n"

    # Variant 1: Independent DPAS (baseline — 2 SGs, no cross-SG dependency)
    name = "xsg_indep"
    header = make_header("Cross-SG independent baseline", "xsg_indep", extra)
    # Each SG does its own DPAS chain independently
    spv = header + f"""   %a_tile = OpCooperativeMatrixLoadKHR %cm_a %buf_a %uint_0 %uint_16 None
   %b_tile = OpCooperativeMatrixLoadKHR %cm_b %buf_b %uint_0 %uint_16 None
  %acc_init = OpCooperativeMatrixLoadKHR %cm_acc %buf_c %uint_0 %uint_16 None
               OpBranch %lh
%lh = OpLabel
               OpLoopMerge %lx %lh None
  %acc_phi = OpPhi %cm_acc %acc_init %entry_label %acc_next %lb
    %i_phi = OpPhi %uint %uint_0 %entry_label %i_next %lb
      %cond = OpULessThan %bool %i_phi %uint_{N_ROUNDS}
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
    spv_path = build_kernel(spv, name)
    base_median = None
    if spv_path:
        median, err = run_benchmark(spv_path, "xsg_indep", 32, 1, 50)
        if median:
            base_median = median
            cyc = median * GHZ / (N_ROUNDS * 2)
            print(f"  Independent (2 SGs):  {median:>10.0f} ns  {cyc:>8.1f} cyc/round")
            results.append({'type': 'independent', 'median_ns': median, 'cyc_per_round': cyc})
    cleanup(name)

    # Variant 2: Cross-SG dependency via global memory (store/load barrier)
    # Each round: SG0 does DPAS, stores result. Barrier. SG1 loads SG0's result, does DPAS.
    # Since SGs run independently in SPIR-V, we simulate this with:
    # SG0: store to buf_c after DPAS, barrier, load from buf_c, DPAS again
    # This creates a dependency: store → barrier → load → DPAS
    # With 2 SGs, each round = 2 DPAS + 1 store + 1 barrier + 1 load
    name = "xsg_dep"
    header = make_header("Cross-SG dependent chain", "xsg_dep", extra)
    spv = header + f"""   %a_tile = OpCooperativeMatrixLoadKHR %cm_a %buf_a %uint_0 %uint_16 None
   %b_tile = OpCooperativeMatrixLoadKHR %cm_b %buf_b %uint_0 %uint_16 None
  %acc_init = OpCooperativeMatrixLoadKHR %cm_acc %buf_c %uint_0 %uint_16 None
               OpBranch %lh
%lh = OpLabel
               OpLoopMerge %lx %lh None
  %acc_phi = OpPhi %cm_acc %acc_init %entry_label %acc_next %lb
    %i_phi = OpPhi %uint %uint_0 %entry_label %i_next %lb
      %cond = OpULessThan %bool %i_phi %uint_{N_ROUNDS}
               OpBranchConditional %cond %lb %lx
%lb = OpLabel
  %acc_next = OpCooperativeMatrixMulAddKHR %cm_acc %a_tile %b_tile %acc_phi
               OpCooperativeMatrixStoreKHR %buf_c %acc_next %uint_0 %uint_16 None
               OpControlBarrier %uint_2 %uint_2 %uint_0
  %acc_reload = OpCooperativeMatrixLoadKHR %cm_acc %buf_c %uint_0 %uint_16 None
  %acc_dep = OpCooperativeMatrixMulAddKHR %cm_acc %a_tile %b_tile %acc_reload
    %i_next = OpIAdd %uint %i_phi %uint_1
               OpBranch %lh
%lx = OpLabel
               OpCooperativeMatrixStoreKHR %buf_c %acc_phi %uint_0 %uint_16 None
               OpReturn
               OpFunctionEnd
"""
    spv_path = build_kernel(spv, name)
    if spv_path:
        median, err = run_benchmark(spv_path, "xsg_dep", 16, 1, 50)
        if median:
            cyc = median * GHZ / (N_ROUNDS * 2)  # 2 DPAS per round
            dep_cyc = (median - base_median) * GHZ / N_ROUNDS if base_median else 0
            print(f"  Dependent (store+bar+load): {median:>10.0f} ns  {cyc:>8.1f} cyc/round  dep_overhead={dep_cyc:.0f}cyc")
            results.append({'type': 'dependent', 'median_ns': median, 'cyc_per_round': cyc,
                            'dep_overhead_cyc': dep_cyc})
    cleanup(name)

    return results


def experiment_3c():
    """3C: Barrier frequency sweep — DPAS throughput with varying barrier intervals."""
    print("\n" + "=" * 70)
    print("Experiment 3C: Barrier Frequency Sweep (DPAS Throughput Impact)")
    print("  ILP=8 DPAS, barriers every N iterations, multi-SG")
    print("=" * 70)

    N_ITER = 128
    ILP = 8
    results = []

    extra = f"  %uint_{N_ITER}  = OpConstant %uint {N_ITER}\n"

    for n_sgs in [4, 8, 16]:
        print(f"\n  --- {n_sgs} SGs/WG ---")
        print(f"  {'bar_int':>8s} {'Median(ns)':>12s} {'TFLOPS':>10s} {'cyc/dpas':>10s}")
        print(f"  {'-' * 44}")

        for bar_interval in [0, 4, 8, 16, 32, 64, 128]:  # 0 = no barrier
            name = f"bfreq_{n_sgs}_{bar_interval}"
            need_mod = bar_interval > 0 and bar_interval < N_ITER
            local_extra = extra
            if need_mod:
                local_extra += f"  %uint_{bar_interval}  = OpConstant %uint {bar_interval}\n"

            header = make_header(f"Barrier freq={bar_interval}, {n_sgs}SGs", "dpas_bfreq", local_extra)

            # ILP=8 loads
            pre = ""
            for i in range(ILP):
                pre += f"  %a_tile{i} = OpCooperativeMatrixLoadKHR %cm_a %buf_a %uint_0 %uint_16 None\n"
                pre += f"  %b_tile{i} = OpCooperativeMatrixLoadKHR %cm_b %buf_b %uint_0 %uint_16 None\n"
                pre += f"  %acc_init{i} = OpCooperativeMatrixLoadKHR %cm_acc %buf_c %uint_0 %uint_16 None\n"
            pre += "               OpBranch %lh\n"

            # For conditional barriers, back edge comes from %merge, otherwise from %lb
            back_block = "%merge" if (0 < bar_interval < N_ITER and bar_interval != 1) else "%lb"
            loop_h = "%lh = OpLabel\n               OpLoopMerge %lx %lh None\n"
            for i in range(ILP):
                loop_h += f"  %acc_phi{i} = OpPhi %cm_acc %acc_init{i} %entry_label %acc_next{i} {back_block}\n"
            loop_h += f"    %i_phi = OpPhi %uint %uint_0 %entry_label %i_next {back_block}\n"
            loop_h += f"      %cond = OpULessThan %bool %i_phi %uint_{N_ITER}\n"
            loop_h += "               OpBranchConditional %cond %lb %lx\n"

            body = "%lb = OpLabel\n"
            for i in range(ILP):
                body += f"  %acc_next{i} = OpCooperativeMatrixMulAddKHR %cm_acc %a_tile{i} %b_tile{i} %acc_phi{i}\n"

            if bar_interval == 0:
                # No barrier
                body += "    %i_next = OpIAdd %uint %i_phi %uint_1\n"
                body += "               OpBranch %lh\n"
            elif bar_interval >= N_ITER:
                # Barrier only at end (effectively no barrier)
                body += "    %i_next = OpIAdd %uint %i_phi %uint_1\n"
                body += "               OpBranch %lh\n"
            else:
                # Conditional barrier via modulo
                body += f"  %rem = OpUMod %uint %i_phi %uint_{bar_interval}\n"
                body += "  %is_bar = OpIEqual %bool %rem %uint_0\n" if bar_interval != 1 else "  %is_bar = OpIEqual %bool %rem %uint_0\n"
                if bar_interval == 1:
                    body = "%lb = OpLabel\n"
                    for i in range(ILP):
                        body += f"  %acc_next{i} = OpCooperativeMatrixMulAddKHR %cm_acc %a_tile{i} %b_tile{i} %acc_phi{i}\n"
                    body += "               OpControlBarrier %uint_2 %uint_2 %uint_0\n"
                    body += "    %i_next = OpIAdd %uint %i_phi %uint_1\n"
                    body += "               OpBranch %lh\n"
                else:
                    body += "               OpBranchConditional %is_bar %bar_blk %nob_blk\n"
                    body += "%bar_blk = OpLabel\n"
                    body += "               OpControlBarrier %uint_2 %uint_2 %uint_0\n"
                    body += "               OpBranch %merge\n"
                    body += "%nob_blk = OpLabel\n"
                    body += "               OpBranch %merge\n"
                    body += "%merge = OpLabel\n"
                    body += "    %i_next = OpIAdd %uint %i_phi %uint_1\n"
                    body += "               OpBranch %lh\n"

            exit_blk = "%lx = OpLabel\n"
            exit_blk += "               OpCooperativeMatrixStoreKHR %buf_c %acc_phi0 %uint_0 %uint_16 None\n"
            exit_blk += "               OpReturn\n               OpFunctionEnd\n"

            spv = build_kernel(header + pre + loop_h + body + exit_blk, name)
            if not spv:
                continue

            wg_x = n_sgs * 16
            n_wg = 160  # enough to saturate
            median, _ = run_benchmark(spv, "dpas_bfreq", wg_x, n_wg, 50)
            if median:
                total_dpas = ILP * N_ITER * n_sgs * n_wg
                tflops = total_dpas * FLOPS_PER_DPAS / (median * 1e-9) / 1e12
                cyc = median * GHZ / total_dpas
                label = "never" if bar_interval == 0 else f"every {bar_interval}"
                print(f"  {label:>8s} {median:>12.0f} {tflops:>10.2f} {cyc:>10.2f}")
                results.append({
                    'n_sgs': n_sgs, 'bar_interval': bar_interval,
                    'n_wg': n_wg, 'median_ns': median,
                    'tflops': tflops, 'cyc_per_dpas': cyc
                })
            cleanup(name)

    return results


def experiment_4b():
    """4B: DPAS with operand staging — compare direct global load vs global→SG load."""
    print("\n" + "=" * 70)
    print("Experiment 4B: DPAS Operand Staging (reload frequency)")
    print("  Compare: pre-load once vs reload every 1/2/4/8 iterations")
    print("  With ILP=1, measures cost of cooperative matrix loads")
    print("=" * 70)

    N_ITER = 64
    results = []
    extra = f"  %uint_{N_ITER}  = OpConstant %uint {N_ITER}\n"

    print(f"\n  {'reload':>10s} {'Median(ns)':>12s} {'cyc/iter':>10s} {'delta_cyc':>10s}")
    print(f"  {'-' * 46}")

    base_cyc = None
    for reload_freq in [0, 1, 2, 4, 8, 16]:  # 0 = preload once
        name = f"stage_{reload_freq}"
        need_mod = reload_freq > 0 and reload_freq < N_ITER
        local_extra = extra
        if need_mod:
            local_extra += f"  %uint_{reload_freq}  = OpConstant %uint {reload_freq}\n"

        header = make_header(f"Reload freq={reload_freq}", "dpas_stage", local_extra)

        if reload_freq == 0:
            # Pre-load once, no reload
            body = "  %acc_next = OpCooperativeMatrixMulAddKHR %cm_acc %a_tile %b_tile %acc_phi\n"
            spv = build_kernel(header + make_simple_loop(N_ITER, body), name)
        elif reload_freq == 1:
            # Reload A, B every iteration
            body = "   %a_cur = OpCooperativeMatrixLoadKHR %cm_a %buf_a %uint_0 %uint_16 None\n"
            body += "   %b_cur = OpCooperativeMatrixLoadKHR %cm_b %buf_b %uint_0 %uint_16 None\n"
            body += "  %acc_next = OpCooperativeMatrixMulAddKHR %cm_acc %a_cur %b_cur %acc_phi\n"
            spv = build_kernel(header + make_simple_loop(N_ITER, body), name)
        else:
            # Reload every Nth iteration
            loop = f"""   %a_tile = OpCooperativeMatrixLoadKHR %cm_a %buf_a %uint_0 %uint_16 None
   %b_tile = OpCooperativeMatrixLoadKHR %cm_b %buf_b %uint_0 %uint_16 None
  %acc_init = OpCooperativeMatrixLoadKHR %cm_acc %buf_c %uint_0 %uint_16 None
               OpBranch %lh
%lh = OpLabel
               OpLoopMerge %lx %lh None
  %acc_phi = OpPhi %cm_acc %acc_init %entry_label %acc_next %merge
  %a_phi   = OpPhi %cm_a %a_tile %entry_label %a_next %merge
  %b_phi   = OpPhi %cm_b %b_tile %entry_label %b_next %merge
    %i_phi = OpPhi %uint %uint_0 %entry_label %i_next %merge
      %cond = OpULessThan %bool %i_phi %uint_{N_ITER}
               OpBranchConditional %cond %lb %lx
%lb = OpLabel
  %acc_next = OpCooperativeMatrixMulAddKHR %cm_acc %a_phi %b_phi %acc_phi
  %rem = OpUMod %uint %i_phi %uint_{reload_freq}
  %is_reload = OpIEqual %bool %rem %uint_0
               OpBranchConditional %is_reload %rld %norld
%rld = OpLabel
  %a_rld = OpCooperativeMatrixLoadKHR %cm_a %buf_a %uint_0 %uint_16 None
  %b_rld = OpCooperativeMatrixLoadKHR %cm_b %buf_b %uint_0 %uint_16 None
               OpBranch %merge
%norld = OpLabel
               OpBranch %merge
%merge = OpLabel
  %a_next = OpPhi %cm_a %a_rld %rld %a_phi %norld
  %b_next = OpPhi %cm_b %b_rld %rld %b_phi %norld
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
        median, _ = run_benchmark(spv, "dpas_stage", 16, 1, 50)
        if median:
            cyc = median * GHZ / N_ITER
            delta = cyc - base_cyc if base_cyc is not None else 0
            if base_cyc is None:
                base_cyc = cyc
            label = "never" if reload_freq == 0 else f"every {reload_freq}"
            print(f"  {label:>10s} {median:>12.0f} {cyc:>10.1f} {delta:>10.1f}")
            results.append({'reload_freq': reload_freq, 'median_ns': median,
                            'cyc_per_iter': cyc, 'delta_cyc': delta})
        cleanup(name)

    return results


def make_simple_loop(n_iter, body):
    return f"""   %a_tile = OpCooperativeMatrixLoadKHR %cm_a %buf_a %uint_0 %uint_16 None
   %b_tile = OpCooperativeMatrixLoadKHR %cm_b %buf_b %uint_0 %uint_16 None
  %acc_init = OpCooperativeMatrixLoadKHR %cm_acc %buf_c %uint_0 %uint_16 None
               OpBranch %lh
%lh = OpLabel
               OpLoopMerge %lx %lh None
  %acc_phi = OpPhi %cm_acc %acc_init %entry_label %acc_next %lb
    %i_phi = OpPhi %uint %uint_0 %entry_label %i_next %lb
      %cond = OpULessThan %bool %i_phi %uint_{n_iter}
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


def experiment_6b():
    """6B: Software pipelining — overlap DPAS with next tile load."""
    print("\n" + "=" * 70)
    print("Experiment 6B: Software Pipelining (DPAS + Prefetch Overlap)")
    print("  Compare: DPAS only vs DPAS + prefetch load of next tile")
    print("  If loads overlap with DPAS → no time increase")
    print("=" * 70)

    N_ITER = 64
    results = []
    extra = f"  %uint_{N_ITER}  = OpConstant %uint {N_ITER}\n"

    print(f"\n  {'pattern':>20s} {'Median(ns)':>12s} {'cyc/iter':>10s} {'delta_cyc':>10s}")
    print(f"  {'-' * 56}")

    base_cyc = None

    # Variant 1: DPAS only (no load in loop)
    name = "pipe_none"
    header = make_header("DPAS only", "dpas_pipe", extra)
    body = "  %acc_next = OpCooperativeMatrixMulAddKHR %cm_acc %a_tile %b_tile %acc_phi\n"
    spv = build_kernel(header + make_simple_loop(N_ITER, body), name)
    if spv:
        median, _ = run_benchmark(spv, "dpas_pipe", 16, 1, 50)
        if median:
            cyc = median * GHZ / N_ITER
            base_cyc = cyc
            print(f"  {'DPAS only':>20s} {median:>12.0f} {cyc:>10.1f} {'0.0':>10s}")
            results.append({'pattern': 'dpas_only', 'median_ns': median, 'cyc_per_iter': cyc, 'delta_cyc': 0})
    cleanup(name)

    # Variant 2: DPAS + reload A after DPAS (prefetch for next iter)
    name = "pipe_rld_a"
    header = make_header("DPAS + prefetch A", "dpas_pipe", extra)
    body = "  %acc_next = OpCooperativeMatrixMulAddKHR %cm_acc %a_tile %b_tile %acc_phi\n"
    body += "   %a_next = OpCooperativeMatrixLoadKHR %cm_a %buf_a %uint_0 %uint_16 None\n"
    # Need phi for a_next → a_tile
    spv_text = header + f"""   %a_tile = OpCooperativeMatrixLoadKHR %cm_a %buf_a %uint_0 %uint_16 None
   %b_tile = OpCooperativeMatrixLoadKHR %cm_b %buf_b %uint_0 %uint_16 None
  %acc_init = OpCooperativeMatrixLoadKHR %cm_acc %buf_c %uint_0 %uint_16 None
               OpBranch %lh
%lh = OpLabel
               OpLoopMerge %lx %lh None
  %acc_phi = OpPhi %cm_acc %acc_init %entry_label %acc_next %lb
  %a_phi = OpPhi %cm_a %a_tile %entry_label %a_next_r %lb
    %i_phi = OpPhi %uint %uint_0 %entry_label %i_next %lb
      %cond = OpULessThan %bool %i_phi %uint_{N_ITER}
               OpBranchConditional %cond %lb %lx
%lb = OpLabel
  %acc_next = OpCooperativeMatrixMulAddKHR %cm_acc %a_phi %b_tile %acc_phi
   %a_next_r = OpCooperativeMatrixLoadKHR %cm_a %buf_a %uint_0 %uint_16 None
    %i_next = OpIAdd %uint %i_phi %uint_1
               OpBranch %lh
%lx = OpLabel
               OpCooperativeMatrixStoreKHR %buf_c %acc_phi %uint_0 %uint_16 None
               OpReturn
               OpFunctionEnd
"""
    spv = build_kernel(spv_text, name)
    if spv:
        median, _ = run_benchmark(spv, "dpas_pipe", 16, 1, 50)
        if median:
            cyc = median * GHZ / N_ITER
            delta = cyc - base_cyc if base_cyc else 0
            print(f"  {'DPAS + prefetch A':>20s} {median:>12.0f} {cyc:>10.1f} {delta:>10.1f}")
            results.append({'pattern': 'dpas_prefetch_a', 'median_ns': median, 'cyc_per_iter': cyc, 'delta_cyc': delta})
    cleanup(name)

    # Variant 3: DPAS + reload A+B after DPAS
    name = "pipe_rld_ab"
    header = make_header("DPAS + prefetch A+B", "dpas_pipe", extra)
    spv_text = header + f"""   %a_tile = OpCooperativeMatrixLoadKHR %cm_a %buf_a %uint_0 %uint_16 None
   %b_tile = OpCooperativeMatrixLoadKHR %cm_b %buf_b %uint_0 %uint_16 None
  %acc_init = OpCooperativeMatrixLoadKHR %cm_acc %buf_c %uint_0 %uint_16 None
               OpBranch %lh
%lh = OpLabel
               OpLoopMerge %lx %lh None
  %acc_phi = OpPhi %cm_acc %acc_init %entry_label %acc_next %lb
  %a_phi = OpPhi %cm_a %a_tile %entry_label %a_next_r %lb
  %b_phi = OpPhi %cm_b %b_tile %entry_label %b_next_r %lb
    %i_phi = OpPhi %uint %uint_0 %entry_label %i_next %lb
      %cond = OpULessThan %bool %i_phi %uint_{N_ITER}
               OpBranchConditional %cond %lb %lx
%lb = OpLabel
  %acc_next = OpCooperativeMatrixMulAddKHR %cm_acc %a_phi %b_phi %acc_phi
   %a_next_r = OpCooperativeMatrixLoadKHR %cm_a %buf_a %uint_0 %uint_16 None
   %b_next_r = OpCooperativeMatrixLoadKHR %cm_b %buf_b %uint_0 %uint_16 None
    %i_next = OpIAdd %uint %i_phi %uint_1
               OpBranch %lh
%lx = OpLabel
               OpCooperativeMatrixStoreKHR %buf_c %acc_phi %uint_0 %uint_16 None
               OpReturn
               OpFunctionEnd
"""
    spv = build_kernel(spv_text, name)
    if spv:
        median, _ = run_benchmark(spv, "dpas_pipe", 16, 1, 50)
        if median:
            cyc = median * GHZ / N_ITER
            delta = cyc - base_cyc if base_cyc else 0
            print(f"  {'DPAS + prefetch A+B':>20s} {median:>12.0f} {cyc:>10.1f} {delta:>10.1f}")
            results.append({'pattern': 'dpas_prefetch_ab', 'median_ns': median, 'cyc_per_iter': cyc, 'delta_cyc': delta})
    cleanup(name)

    # Variant 4: Reload before DPAS (sequential, no overlap)
    name = "pipe_seq"
    header = make_header("Load + DPAS sequential", "dpas_pipe", extra)
    spv_text = header + f"""   %a_tile = OpCooperativeMatrixLoadKHR %cm_a %buf_a %uint_0 %uint_16 None
   %b_tile = OpCooperativeMatrixLoadKHR %cm_b %buf_b %uint_0 %uint_16 None
  %acc_init = OpCooperativeMatrixLoadKHR %cm_acc %buf_c %uint_0 %uint_16 None
               OpBranch %lh
%lh = OpLabel
               OpLoopMerge %lx %lh None
  %acc_phi = OpPhi %cm_acc %acc_init %entry_label %acc_next %lb
    %i_phi = OpPhi %uint %uint_0 %entry_label %i_next %lb
      %cond = OpULessThan %bool %i_phi %uint_{N_ITER}
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
    spv = build_kernel(spv_text, name)
    if spv:
        median, _ = run_benchmark(spv, "dpas_pipe", 16, 1, 50)
        if median:
            cyc = median * GHZ / N_ITER
            delta = cyc - base_cyc if base_cyc else 0
            print(f"  {'Load→DPAS sequential':>20s} {median:>12.0f} {cyc:>10.1f} {delta:>10.1f}")
            results.append({'pattern': 'load_dpas_sequential', 'median_ns': median, 'cyc_per_iter': cyc, 'delta_cyc': delta})
    cleanup(name)

    return results


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    if not SPIRV_RUNNER.exists():
        print("Building spirv_runner...")
        run_cmd("g++ -std=c++17 -O2 -o spirv_runner spirv_runner.cpp -lze_loader -lm")

    results_2a = experiment_2a()
    results_3c = experiment_3c()
    results_4b = experiment_4b()
    results_6b = experiment_6b()

    saves = [
        ("dpas_cross_sg_dep.csv", results_2a),
        ("dpas_barrier_freq_sweep.csv", results_3c),
        ("dpas_reload_freq.csv", results_4b),
        ("dpas_software_pipeline.csv", results_6b),
    ]
    for fname, data in saves:
        if data:
            path = RESULTS_DIR / fname
            # Collect all unique keys for consistent fieldnames
            all_keys = list(dict.fromkeys(k for d in data for k in d.keys()))
            with open(path, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=all_keys, extrasaction='ignore')
                w.writeheader()
                w.writerows(data)
            print(f"Saved to {path}")


if __name__ == "__main__":
    main()
