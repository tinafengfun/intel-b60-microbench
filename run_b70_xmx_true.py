#!/usr/bin/env python3
"""
B70 TRUE XMX rate — all previous throughput kernels were defeated by IGC dead-code
elimination (only acc_phi0 was stored, so all other DPAS chains were removed;
every measurement reduced to ONE dependent chain per SG).

This kernel stores ALL accumulators -> all chains stay live.
T1: ILP sweep at long n_iter -> latency hiding -> true issue-rate limit (plateau).
T2: n_iter fit at best ILP -> steady-state cyc/dpas + fixed overhead.
T3: core scaling at best ILP (n_wg=1..32, sg=8) -> true per-EU/full-machine rate.
"""
import subprocess, sys, re, csv, time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))
from run_b70_xmx_probe import build_and_compile, cleanup, run_benchmark, run_cmd

SPIRV_RUNNER = SCRIPT_DIR / "spirv_runner"
RESULTS_DIR = SCRIPT_DIR / "results"
GHZ = 2.4
FLOPS_PER_DPAS = 4096


def gen_kernel_live(ilp, n_iter):
    """ILP independent DPAS chains; ALL accumulators stored -> DCE-proof."""
    lines = [
        "; SPIR-V 1.4",
        f"; DPAS ILP={ilp} ALL-LIVE (stores all accs), {n_iter} iters",
        "               OpCapability Addresses",
        "               OpCapability Kernel",
        "               OpCapability Int64",
        "               OpCapability Int16",
        "               OpCapability CooperativeMatrixKHR",
        '               OpExtension "SPV_KHR_cooperative_matrix"',
        '          %1 = OpExtInstImport "OpenCL.std"',
        "               OpMemoryModel Physical64 OpenCL",
        "               OpEntryPoint Kernel %main \"dpas_ilp\"",
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
    for i in range(ilp):
        lines.append(f"  %a_tile{i} = OpCooperativeMatrixLoadKHR %cm_a %buf_a %uint_0 %uint_16 None")
        lines.append(f"  %b_tile{i} = OpCooperativeMatrixLoadKHR %cm_b %buf_b %uint_0 %uint_16 None")
        lines.append(f"  %acc_init{i} = OpCooperativeMatrixLoadKHR %cm_acc %buf_c %uint_0 %uint_16 None")
    lines.append("               OpBranch %lh")
    lines.append("%lh = OpLabel")
    lines.append("               OpLoopMerge %lx %lh None")
    for i in range(ilp):
        lines.append(f"  %acc_phi{i} = OpPhi %cm_acc %acc_init{i} %entry %acc_next{i} %lb")
    lines.append("    %i_phi = OpPhi %uint %uint_0 %entry %i_next %lb")
    lines.append(f"      %cond = OpULessThan %bool %i_phi %uint_{n_iter}")
    lines.append("               OpBranchConditional %cond %lb %lx")
    lines.append("%lb = OpLabel")
    for i in range(ilp):
        lines.append(f"  %acc_next{i} = OpCooperativeMatrixMulAddKHR %cm_acc %a_tile{i} %b_tile{i} %acc_phi{i}")
    lines.append("    %i_next = OpIAdd %uint %i_phi %uint_1")
    lines.append("               OpBranch %lh")
    lines.append("%lx = OpLabel")
    # store every accumulator to offset 0 sequentially — stores are side-effecting
    # so IGC cannot dead-code-eliminate any chain (non-zero offsets and coop-matrix
    # OpFAdd both segfault IGC on bmg-g31; this is the safe form)
    for i in range(ilp):
        lines.append(f"               OpCooperativeMatrixStoreKHR %buf_c %acc_phi{i} %uint_0 %uint_16 None")
    lines.append("               OpReturn")
    lines.append("               OpFunctionEnd")
    return "\n".join(lines)


def run_cfg(spv, wg_x, n_wg, ilp, n_iter, repeats=10):
    median, err = run_benchmark(spv, wg_x, n_wg, repeats=repeats)
    if median is None:
        return None, err
    total_sgs = (wg_x // 16) * n_wg
    ns_per_dpas = median / (ilp * n_iter)
    tf = ilp * n_iter * total_sgs * FLOPS_PER_DPAS / (median * 1e-9) / 1e12
    return (median, ns_per_dpas * GHZ, tf, total_sgs), None


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    rows = []

    print("=== T1: ILP sweep, ALL chains live (n_iter=16384, n_wg=32, sg=8) ===")
    n_iter = 16384
    for ilp in [3, 4, 5, 6, 7]:
        tag = f"live_ilp{ilp}"
        spv = build_and_compile(gen_kernel_live(ilp, n_iter), tag)
        if not spv:
            continue
        res, err = run_cfg(spv, 128, 32, ilp, n_iter)
        cleanup(tag)
        if res is None:
            print(f"  ILP={ilp} FAILED: {err}")
            continue
        median, cyc, tf, _ = res
        print(f"  ILP={ilp:>2d}  median={median:>12.0f} ns  {cyc:>7.2f} cyc/dpas  -> {tf:>8.1f} TF")
        rows.append({'exp': 'T1_ilp_sweep', 'ilp': ilp, 'n_iter': n_iter, 'n_wg': 32,
                     'median_ns': median, 'cyc_per_dpas': cyc, 'tflops': tf})

    print("=== T2: n_iter fit at ILP=16 (n_wg=32, sg=8) ===")
    ilp = 16
    fit_pts = []
    for n_iter in [512, 2048, 8192, 32768]:
        tag = f"live_fit{n_iter}"
        spv = build_and_compile(gen_kernel_live(ilp, n_iter), tag)
        if not spv:
            continue
        res, err = run_cfg(spv, 128, 32, ilp, n_iter)
        cleanup(tag)
        if res is None:
            continue
        median, cyc, tf, _ = res
        print(f"  n_iter={n_iter:>6d}  median={median:>12.0f} ns  apparent {tf:>8.1f} TF")
        fit_pts.append((n_iter, median))
    if len(fit_pts) >= 3:
        n = len(fit_pts)
        sx = sum(p[0] for p in fit_pts); sy = sum(p[1] for p in fit_pts)
        sxx = sum(p[0]**2 for p in fit_pts); sxy = sum(p[0]*p[1] for p in fit_pts)
        b = (n*sxy - sx*sy) / (n*sxx - sx*sx); a = (sy - b*sx) / n
        ns_dpas = b / ilp
        tf_eu = FLOPS_PER_DPAS / (ns_dpas*1e-9) / 1e12
        print(f"  FIT: time = {a:.0f} ns + {b:.3f} ns/iter -> {ns_dpas*GHZ:.2f} cyc/dpas, "
              f"{tf_eu:.3f} TF/EU, {tf_eu*256:.0f} TF full machine")
        rows.append({'exp': 'T2_fit', 'ilp': ilp, 'n_iter': 0, 'n_wg': 32,
                     'median_ns': a, 'cyc_per_dpas': ns_dpas*GHZ, 'tflops': tf_eu*256})

    print("=== T3: core scaling, ALL live (ILP=16, n_iter=8192, sg=8) ===")
    ilp, n_iter = 16, 8192
    tag = "live_scale"
    spv = build_and_compile(gen_kernel_live(ilp, n_iter), tag)
    if spv:
        for n_wg in [1, 2, 4, 8, 16, 32]:
            res, err = run_cfg(spv, 128, n_wg, ilp, n_iter)
            if res is None:
                continue
            median, cyc, tf, sgs = res
            print(f"  n_wg={n_wg:>3d} ({n_wg:>2d} cores)  median={median:>12.0f} ns  {tf:>8.1f} TF  {tf/sgs:.3f} TF/EU")
            rows.append({'exp': 'T3_core_scaling', 'ilp': ilp, 'n_iter': n_iter, 'n_wg': n_wg,
                         'median_ns': median, 'cyc_per_dpas': cyc, 'tflops': tf})
        cleanup(tag)

    if rows:
        out = RESULTS_DIR / "b70_xmx_true_rate.csv"
        with open(out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"Saved to {out}")


if __name__ == "__main__":
    main()
