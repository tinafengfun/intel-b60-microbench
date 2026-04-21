#!/usr/bin/env python3
"""
DPAS ILP 1-16 Sweep: Measure per-DPAS time as ILP increases.
Key question: At what ILP does GRF pressure cause register spills?
"""
import subprocess, sys, re, csv, shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
DEVICE = "bmg-g21"
SPIRV_RUNNER = SCRIPT_DIR / "spirv_runner"
RESULTS_DIR = SCRIPT_DIR / "results"
GHZ = 2.4
FLOPS_PER_DPAS = 4096
N_ITER = 64


def gen_ilp_kernel(n_ilp):
    """Generate SPIR-V for N independent DPAS chains, 64 iterations each."""
    lines = [
        "; SPIR-V 1.4",
        f"; DPAS ILP={n_ilp}, {N_ITER} iterations per chain",
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
        f"  %uint_{N_ITER}  = OpConstant %uint {N_ITER}",
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
    # Load A, B tiles once; load initial accumulator for each chain
    lines.append("  %a_tile = OpCooperativeMatrixLoadKHR %cm_a %buf_a %uint_0 %uint_16 None")
    lines.append("  %b_tile = OpCooperativeMatrixLoadKHR %cm_b %buf_b %uint_0 %uint_16 None")
    for i in range(n_ilp):
        lines.append(f"  %c_init{i} = OpCooperativeMatrixLoadKHR %cm_acc %buf_c %uint_0 %uint_16 None")
    lines.append("               OpBranch %lh")

    # Loop header with phi nodes
    lines.append("%lh = OpLabel")
    lines.append("               OpLoopMerge %lx %lh None")
    for i in range(n_ilp):
        lines.append(f"  %acc_phi{i} = OpPhi %cm_acc %c_init{i} %entry %acc_next{i} %lb")
    lines.append("    %i_phi = OpPhi %uint %uint_0 %entry %i_next %lb")
    lines.append(f"      %cond = OpULessThan %bool %i_phi %uint_{N_ITER}")
    lines.append("               OpBranchConditional %cond %lb %lx")

    # Loop body: independent DPAS for each chain
    lines.append("%lb = OpLabel")
    for i in range(n_ilp):
        lines.append(f"  %acc_next{i} = OpCooperativeMatrixMulAddKHR %cm_acc %a_tile %b_tile %acc_phi{i}")
    lines.append("    %i_next = OpIAdd %uint %i_phi %uint_1")
    lines.append("               OpBranch %lh")

    # Exit: store last result from chain 0
    lines.append("%lx = OpLabel")
    lines.append("               OpCooperativeMatrixStoreKHR %buf_c %acc_phi0 %uint_0 %uint_16 None")
    lines.append("               OpReturn")
    lines.append("               OpFunctionEnd")
    return "\n".join(lines)


def run_cmd(cmd, cwd=SCRIPT_DIR, check=True):
    return subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    if not SPIRV_RUNNER.exists():
        print("Building spirv_runner...")
        run_cmd("g++ -std=c++17 -O2 -o spirv_runner spirv_runner.cpp -lze_loader -lm")

    ilp_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16]
    results = []

    print(f"\n{'='*70}")
    print(f"DPAS ILP Sweep (N_ITER={N_ITER}, single SG, single WG)")
    print(f"{'='*70}")
    print(f"  {'ILP':>4s} {'Median(ns)':>12s} {'cyc/dpas':>10s} {'TFLOPS':>10s} {'dpas_asm':>10s}")
    print(f"  {'-'*50}")

    for ilp in ilp_values:
        name = f"ilp_sweep_{ilp}"
        spvasm = SCRIPT_DIR / f"sweep_{name}.spvasm"
        spv = SCRIPT_DIR / f"sweep_{name}.spv"
        spvasm.write_text(gen_ilp_kernel(ilp))

        # Assemble
        r = run_cmd(f"spirv-as --target-env spv1.4 {spvasm} -o {spv}", check=False)
        if r.returncode != 0:
            print(f"  {ilp:>4d} ASSEMBLE FAILED: {r.stderr[:200]}")
            continue

        # Compile
        r = run_cmd(f"ocloc compile -spirv_input -file {spv} -device {DEVICE} -output {name}", check=False)
        if r.returncode != 0 or "error" in (r.stderr+r.stdout).lower():
            print(f"  {ilp:>4d} COMPILE FAILED: {(r.stderr+r.stdout)[:200]}")
            # Cleanup
            for f in [spvasm, spv]: f.unlink(missing_ok=True)
            for f in SCRIPT_DIR.glob(f"{name}*"): f.unlink(missing_ok=True)
            continue

        # Find binary
        bin_path = SCRIPT_DIR / f"{name}_bmg.bin"
        if not bin_path.exists():
            print(f"  {ilp:>4d} NO BINARY")
            continue

        # Disassemble for dpas count
        disasm = SCRIPT_DIR / f"disasm_{name}"
        run_cmd(f"mkdir -p {disasm}")
        run_cmd(f"ocloc disasm -file {bin_path} -dump {disasm} -device {DEVICE}", check=False)
        dpas_count = 0
        has_spill = False
        for f in disasm.iterdir():
            if f.name.endswith('.asm'):
                text = f.read_text()
                dpas_count = text.count("dpas")
                has_spill = "spill" in text.lower() or ("mov" in text and "r" in text and text.count("mov") > dpas_count * 2)
                break

        # Run
        r = run_cmd(f"{SPIRV_RUNNER} {spv} dpas_ilp 16 1 50", check=False)
        if r.returncode != 0:
            print(f"  {ilp:>4d} RUN FAILED: {r.stderr[:200]}")
        else:
            m = re.search(r'Median=([\d.]+)\s+ns', r.stdout)
            if m:
                median = float(m.group(1))
                total_dpas = ilp * N_ITER
                cyc = median * GHZ / total_dpas
                tflops = total_dpas * FLOPS_PER_DPAS / (median * 1e-9) / 1e12
                spill_str = " [SPILL?]" if has_spill and ilp > 8 else ""
                print(f"  {ilp:>4d} {median:>12.0f} {cyc:>10.1f} {tflops:>10.3f} {dpas_count:>10d}{spill_str}")
                results.append({'ilp': ilp, 'median_ns': median, 'cyc_per_dpas': cyc, 'tflops': tflops, 'dpas_in_asm': dpas_count})

        # Cleanup
        for f in [spvasm, spv, bin_path]: f.unlink(missing_ok=True)
        shutil.rmtree(disasm, ignore_errors=True)
        for f in SCRIPT_DIR.glob(f"{name}*"): f.unlink(missing_ok=True)

    # Save
    if results:
        csv_path = RESULTS_DIR / "dpas_ilp_sweep.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
