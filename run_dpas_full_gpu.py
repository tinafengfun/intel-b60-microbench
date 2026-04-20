#!/usr/bin/env python3
"""
DPAS Full-GPU Throughput Sweep
Measures reciprocal throughput directly by running independent DPAS chains
across many work-groups (1 to 4096 WGs).

Goal: Derive reciprocal throughput from microbenchmark data, not from GEMM.
"""
import subprocess, sys, re, shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
DEVICE = "bmg-g21"
SPIRV_RUNNER = SCRIPT_DIR / "spirv_runner"
RESULTS_DIR = SCRIPT_DIR / "results"
REPEAT = 50

ILP = 8  # Independent chains per sub-group
N_ITER = 128  # Iterations per chain
GHZ = 2.4
FLOPS_PER_DPAS = 2 * 8 * 16 * 16  # 4096


def generate_throughput_kernel():
    """Generate SPIR-V for ILP=8 independent DPAS chains, N_ITER=128 iterations."""
    return """; SPIR-V 1.4
; DPAS Throughput: 8 independent chains, 128 iters per chain
; Each sub-group does 1024 DPAS operations total
               OpCapability Addresses
               OpCapability Kernel
               OpCapability Int64
               OpCapability Int16
               OpCapability CooperativeMatrixKHR
               OpExtension "SPV_KHR_cooperative_matrix"
          %1 = OpExtInstImport "OpenCL.std"
               OpMemoryModel Physical64 OpenCL
               OpEntryPoint Kernel %main "dpas_throughput"
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
  %uint_128  = OpConstant %uint 128
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
   %a_tile0 = OpCooperativeMatrixLoadKHR %cm_a %buf_a %uint_0 %uint_16 None
   %b_tile0 = OpCooperativeMatrixLoadKHR %cm_b %buf_b %uint_0 %uint_16 None
  %acc_init0 = OpCooperativeMatrixLoadKHR %cm_acc %buf_c %uint_0 %uint_16 None
   %a_tile1 = OpCooperativeMatrixLoadKHR %cm_a %buf_a %uint_0 %uint_16 None
   %b_tile1 = OpCooperativeMatrixLoadKHR %cm_b %buf_b %uint_0 %uint_16 None
  %acc_init1 = OpCooperativeMatrixLoadKHR %cm_acc %buf_c %uint_0 %uint_16 None
   %a_tile2 = OpCooperativeMatrixLoadKHR %cm_a %buf_a %uint_0 %uint_16 None
   %b_tile2 = OpCooperativeMatrixLoadKHR %cm_b %buf_b %uint_0 %uint_16 None
  %acc_init2 = OpCooperativeMatrixLoadKHR %cm_acc %buf_c %uint_0 %uint_16 None
   %a_tile3 = OpCooperativeMatrixLoadKHR %cm_a %buf_a %uint_0 %uint_16 None
   %b_tile3 = OpCooperativeMatrixLoadKHR %cm_b %buf_b %uint_0 %uint_16 None
  %acc_init3 = OpCooperativeMatrixLoadKHR %cm_acc %buf_c %uint_0 %uint_16 None
   %a_tile4 = OpCooperativeMatrixLoadKHR %cm_a %buf_a %uint_0 %uint_16 None
   %b_tile4 = OpCooperativeMatrixLoadKHR %cm_b %buf_b %uint_0 %uint_16 None
  %acc_init4 = OpCooperativeMatrixLoadKHR %cm_acc %buf_c %uint_0 %uint_16 None
   %a_tile5 = OpCooperativeMatrixLoadKHR %cm_a %buf_a %uint_0 %uint_16 None
   %b_tile5 = OpCooperativeMatrixLoadKHR %cm_b %buf_b %uint_0 %uint_16 None
  %acc_init5 = OpCooperativeMatrixLoadKHR %cm_acc %buf_c %uint_0 %uint_16 None
   %a_tile6 = OpCooperativeMatrixLoadKHR %cm_a %buf_a %uint_0 %uint_16 None
   %b_tile6 = OpCooperativeMatrixLoadKHR %cm_b %buf_b %uint_0 %uint_16 None
  %acc_init6 = OpCooperativeMatrixLoadKHR %cm_acc %buf_c %uint_0 %uint_16 None
   %a_tile7 = OpCooperativeMatrixLoadKHR %cm_a %buf_a %uint_0 %uint_16 None
   %b_tile7 = OpCooperativeMatrixLoadKHR %cm_b %buf_b %uint_0 %uint_16 None
  %acc_init7 = OpCooperativeMatrixLoadKHR %cm_acc %buf_c %uint_0 %uint_16 None
               OpBranch %loop_header
%loop_header = OpLabel
               OpLoopMerge %loop_exit %loop_header None
  %acc_phi0 = OpPhi %cm_acc %acc_init0 %entry %acc_next0 %loop_body
  %acc_phi1 = OpPhi %cm_acc %acc_init1 %entry %acc_next1 %loop_body
  %acc_phi2 = OpPhi %cm_acc %acc_init2 %entry %acc_next2 %loop_body
  %acc_phi3 = OpPhi %cm_acc %acc_init3 %entry %acc_next3 %loop_body
  %acc_phi4 = OpPhi %cm_acc %acc_init4 %entry %acc_next4 %loop_body
  %acc_phi5 = OpPhi %cm_acc %acc_init5 %entry %acc_next5 %loop_body
  %acc_phi6 = OpPhi %cm_acc %acc_init6 %entry %acc_next6 %loop_body
  %acc_phi7 = OpPhi %cm_acc %acc_init7 %entry %acc_next7 %loop_body
    %i_phi = OpPhi %uint %uint_0 %entry %i_next %loop_body
      %cond = OpULessThan %bool %i_phi %uint_128
               OpBranchConditional %cond %loop_body %loop_exit
%loop_body = OpLabel
  %acc_next0 = OpCooperativeMatrixMulAddKHR %cm_acc %a_tile0 %b_tile0 %acc_phi0
  %acc_next1 = OpCooperativeMatrixMulAddKHR %cm_acc %a_tile1 %b_tile1 %acc_phi1
  %acc_next2 = OpCooperativeMatrixMulAddKHR %cm_acc %a_tile2 %b_tile2 %acc_phi2
  %acc_next3 = OpCooperativeMatrixMulAddKHR %cm_acc %a_tile3 %b_tile3 %acc_phi3
  %acc_next4 = OpCooperativeMatrixMulAddKHR %cm_acc %a_tile4 %b_tile4 %acc_phi4
  %acc_next5 = OpCooperativeMatrixMulAddKHR %cm_acc %a_tile5 %b_tile5 %acc_phi5
  %acc_next6 = OpCooperativeMatrixMulAddKHR %cm_acc %a_tile6 %b_tile6 %acc_phi6
  %acc_next7 = OpCooperativeMatrixMulAddKHR %cm_acc %a_tile7 %b_tile7 %acc_phi7
    %i_next = OpIAdd %uint %i_phi %uint_1
               OpBranch %loop_header
%loop_exit = OpLabel
               OpCooperativeMatrixStoreKHR %buf_c %acc_phi0 %uint_0 %uint_16 None
               OpCooperativeMatrixStoreKHR %buf_c %acc_phi1 %uint_0 %uint_16 None
               OpCooperativeMatrixStoreKHR %buf_c %acc_phi2 %uint_0 %uint_16 None
               OpCooperativeMatrixStoreKHR %buf_c %acc_phi3 %uint_0 %uint_16 None
               OpCooperativeMatrixStoreKHR %buf_c %acc_phi4 %uint_0 %uint_16 None
               OpCooperativeMatrixStoreKHR %buf_c %acc_phi5 %uint_0 %uint_16 None
               OpCooperativeMatrixStoreKHR %buf_c %acc_phi6 %uint_0 %uint_16 None
               OpCooperativeMatrixStoreKHR %buf_c %acc_phi7 %uint_0 %uint_16 None
               OpReturn
               OpFunctionEnd
"""


def run_cmd(cmd, cwd=SCRIPT_DIR, check=True):
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if check and r.returncode != 0:
        print(f"ERROR: {cmd}", file=sys.stderr)
        print(r.stderr, file=sys.stderr)
    return r


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    if not SPIRV_RUNNER.exists():
        print("Building spirv_runner...")
        run_cmd("g++ -std=c++17 -O2 -o spirv_runner spirv_runner.cpp -lze_loader -lm")

    # Generate and build kernel
    spvasm_path = SCRIPT_DIR / "sweep_dpas_fullgpu.spvasm"
    spv_path = SCRIPT_DIR / "sweep_dpas_fullgpu.spv"
    bin_path = SCRIPT_DIR / "sweep_dpas_fullgpu_bmg.bin"

    spvasm_path.write_text(generate_throughput_kernel())
    run_cmd(f"spirv-as {spvasm_path} -o {spv_path}")
    r = run_cmd(f"ocloc compile -spirv_input -file {spv_path} -device {DEVICE} -output sweep_dpas_fullgpu", check=False)
    if r.returncode != 0 or "error" in (r.stderr + r.stdout).lower():
        print("Compile failed:", r.stderr[:500])
        return

    # Validate DPAS count
    disasm_dir = SCRIPT_DIR / "disasm_fullgpu"
    run_cmd(f"mkdir -p {disasm_dir}")
    run_cmd(f"ocloc disasm -file {bin_path} -dump {disasm_dir} -device {DEVICE}", check=False)
    for f in disasm_dir.iterdir():
        if f.name.endswith('.asm'):
            text = f.read_text()
            dpas_count = text.count("dpas")
            print(f"GEN ASM: {dpas_count} dpas instructions (expect {ILP * N_ITER}={ILP*N_ITER})")
            break

    # Sweep work-group counts
    wg_counts = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    wg_size = 16  # Single sub-group per WG

    total_dpas_per_wg = ILP * N_ITER  # 1024
    total_flops_per_wg = total_dpas_per_wg * FLOPS_PER_DPAS  # 4,194,304

    print(f"\n{'='*80}")
    print(f"DPAS Full-GPU Throughput Sweep (ILP={ILP}, N_ITER={N_ITER})")
    print(f"  {total_dpas_per_wg} DPAS per WG, {total_flops_per_wg/1e6:.1f} MFLOPs per WG")
    print(f"{'='*80}")
    print(f"  {'WGs':>6s} {'Median(ns)':>12s} {'TFLOPS':>10s} {'DPAS/WG':>10s} {'cyc/dpas':>10s}")
    print(f"  {'-'*52}")

    results = []
    for n_wg in wg_counts:
        r = run_cmd(f"{SPIRV_RUNNER} {spv_path} dpas_throughput {wg_size} {n_wg} {REPEAT}", check=False)
        if r.returncode != 0:
            print(f"  {n_wg:>6d} FAILED: {r.stderr[:100]}")
            continue

        m = re.search(r'Median=([\d.]+)\s+ns', r.stdout)
        if not m:
            print(f"  {n_wg:>6d} PARSE FAILED")
            continue

        median_ns = float(m.group(1))
        total_flops = total_flops_per_wg * n_wg
        tflops = total_flops / (median_ns * 1e-9) / 1e12
        cyc_per_dpas = median_ns * GHZ / total_dpas_per_wg

        print(f"  {n_wg:>6d} {median_ns:>12.0f} {tflops:>10.2f} {total_dpas_per_wg:>10d} {cyc_per_dpas:>10.1f}")
        results.append((n_wg, median_ns, tflops, cyc_per_dpas))

    # Compute reciprocal throughput from saturation point
    if results:
        max_tflops = max(r[2] for r in results)
        max_row = [r for r in results if r[2] == max_tflops][0]
        # reciprocal_throughput = 160 XMX × 4096 FLOPs × 2.4 GHz / peak_TFLOPS
        peak_hardware = 160 * FLOPS_PER_DPAS * GHZ  # GFLOPS if 1 dpas/cycle
        implied_cyc = peak_hardware / (max_tflops * 1000)  # cycles per DPAS at peak

        print(f"\n{'='*60}")
        print(f"Peak achieved: {max_tflops:.2f} TFLOPS at {max_row[0]} WGs")
        print(f"Hardware peak (160 XMX × 4096 FLOPs × 2.4 GHz):")
        print(f"  If 1 DPAS/cycle: {peak_hardware/1000:.1f} TFLOPS")
        print(f"  If 1 DPAS/16 cyc: {peak_hardware/16/1000:.1f} TFLOPS")
        print(f"  If 1 DPAS/33 cyc: {peak_hardware/33/1000:.1f} TFLOPS")
        print(f"\nDirect reciprocal throughput:")
        print(f"  {max_tflops:.2f} TFLOPS / 160 XMX = {max_tflops*1000/160:.2f} GFLOPS per XMX")
        print(f"  Implied: 4096 FLOPs × 2.4 GHz / {max_tflops*1000/160:.2f} GFLOPS = {implied_cyc:.1f} cycles/dpas")
        print(f"{'='*60}")

    # Cleanup
    for f in [spvasm_path, spv_path, bin_path]:
        f.unlink(missing_ok=True)
    shutil.rmtree(disasm_dir, ignore_errors=True)
    for f in SCRIPT_DIR.glob("sweep_dpas_fullgpu*"):
        f.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
