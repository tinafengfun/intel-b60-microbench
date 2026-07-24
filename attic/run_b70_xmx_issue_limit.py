# WARNING: DCE-FLAWED intermediate version — see attic/README.md. Do not use for measurements.
#!/usr/bin/env python3
"""
V4: find the true XMX issue-rate limit on B70 (BMG-G31).
V2 showed cyc/dpas = 35.9/ILP all the way to ILP=32 (latency-bound, 2.24 PF apparent).
Probe ILP=40/48/64 to find the plateau = issue-rate limit.
Also cross-check with distinct B tiles (anti-CSE) at ILP=16/32.
"""
import subprocess, sys, re, csv, time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))
from run_b70_xmx_probe import build_and_compile, cleanup, run_benchmark, run_cmd

DEVICE = "bmg-g31"
SPIRV_RUNNER = SCRIPT_DIR / "spirv_runner"
RESULTS_DIR = SCRIPT_DIR / "results"
GHZ = 2.4
FLOPS_PER_DPAS = 4096


def gen_ilp_kernel_dt(ilp, n_iter, distinct_tiles=False):
    """ILP kernel; distinct_tiles=True loads each A/B tile from a different offset."""
    lines = [
        "; SPIR-V 1.4",
        f"; DPAS ILP={ilp}, {n_iter} iters, distinct_tiles={distinct_tiles}",
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
        if distinct_tiles:
            # A tile: 8x16 ushort = 256B = 128 elems; B tile: 16x16 ushort = 512B = 256 elems
            lines.append(f"  %a_off{i} = OpConstant %uint {i * 128}")
            lines.append(f"  %b_off{i} = OpConstant %uint {i * 256}")
            lines.append(f"  %a_tile{i} = OpCooperativeMatrixLoadKHR %cm_a %buf_a %a_off{i} %uint_16 None")
            lines.append(f"  %b_tile{i} = OpCooperativeMatrixLoadKHR %cm_b %buf_b %b_off{i} %uint_16 None")
        else:
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
    lines.append("               OpCooperativeMatrixStoreKHR %buf_c %acc_phi0 %uint_0 %uint_16 None")
    lines.append("               OpReturn")
    lines.append("               OpFunctionEnd")
    return "\n".join(lines)


def run_one(ilp, n_iter, distinct=False, repeats=10):
    tag = f"ilp{ilp}{'_dt' if distinct else ''}_n{n_iter}"
    spv = build_and_compile(gen_ilp_kernel_dt(ilp, n_iter, distinct), tag)
    if not spv:
        return None
    median, err = run_benchmark(spv, 128, 32, repeats=repeats)
    cleanup(tag)
    if median is None:
        print(f"  {tag} FAILED: {err}")
        return None
    ns_per_dpas = median / (ilp * n_iter)
    tf = ilp * n_iter * 256 * FLOPS_PER_DPAS / (median * 1e-9) / 1e12
    print(f"  ILP={ilp:>2d} dt={int(distinct)} n_iter={n_iter:>6d}  median={median:>12.0f} ns  "
          f"{ns_per_dpas:.4f} ns/dpas = {ns_per_dpas*GHZ:>5.2f} cyc  -> {tf:>8.1f} TF")
    return {'ilp': ilp, 'distinct': int(distinct), 'n_iter': n_iter, 'median_ns': median,
            'cyc_per_dpas': ns_per_dpas * GHZ, 'tflops': tf}


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    rows = []
    print("=== V4a: same-tile ILP sweep (n_iter=32768, n_wg=32, sg=8) ===")
    for ilp in [32, 40, 48, 64]:
        r = run_one(ilp, 32768, distinct=False)
        if r:
            rows.append(r)
    print("=== V4b: distinct-tile cross-check (n_iter=32768) ===")
    for ilp in [16, 32]:
        r = run_one(ilp, 32768, distinct=True)
        if r:
            rows.append(r)
    print("=== V4c: long-run at ILP=48 + wall clock ===")
    t0 = time.time()
    r = run_one(48, 131072, distinct=False, repeats=5)
    wall = time.time() - t0
    if r:
        rows.append(r)
        print(f"  wall for 5 repeats + launch overhead: {wall:.2f} s")

    out = RESULTS_DIR / "b70_xmx_issue_limit.csv"
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
