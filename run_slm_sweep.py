#!/usr/bin/env python3
"""
SPIR-V Native SLM Pointer Chase
Generates Workgroup-storage pointer chase kernels to measure true SLM latency.

Single-thread kernel: copies permutation from global to SLM, then chases through SLM.
No SYCL overhead, pure SPIR-V + Level Zero.
"""

import subprocess, sys, re, csv, shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
SPIRV_RUNNER = SCRIPT_DIR / "spirv_runner"
RESULTS_DIR = SCRIPT_DIR / "results"
DEVICE = "bmg-g21"
REPEATS = 50
CHASE = 4096


def gen_slm_chase(n_elems):
    """Generate SPIR-V SLM pointer chase kernel for n_elems int32 elements.
    Single thread (LocalSize 1 1 1): copies permutation from global to SLM, chases."""

    return f"""; SPIR-V 1.4
; SLM Pointer Chase: {n_elems} int32 elements, {CHASE} steps, single thread
; Uses Workgroup storage class for SLM allocation

               OpCapability Addresses
               OpCapability Kernel
               OpCapability Int64
          %1 = OpExtInstImport "OpenCL.std"
               OpMemoryModel Physical64 OpenCL
               OpEntryPoint Kernel %main "slm_chase"
               OpExecutionMode %main LocalSize 1 1 1

     %void = OpTypeVoid
     %bool = OpTypeBool
     %uint = OpTypeInt 32 0
    %ulong = OpTypeInt 64 0
   %uint_0 = OpConstant %uint 0
   %uint_1 = OpConstant %uint 1
   %uint_{n_elems} = OpConstant %uint {n_elems}
  %uint_{CHASE} = OpConstant %uint {CHASE}

; Array type for SLM
  %arr_{n_elems} = OpTypeArray %uint %uint_{n_elems}

; Pointer types
 %ptr_cross_uint = OpTypePointer CrossWorkgroup %uint
 %ptr_work_arr = OpTypePointer Workgroup %arr_{n_elems}
 %ptr_work_uint = OpTypePointer Workgroup %uint

; SLM variable (Workgroup storage, module scope)
     %slm = OpVariable %ptr_work_arr Workgroup

  %fn_type = OpTypeFunction %void %ptr_cross_uint %ptr_cross_uint

  %main = OpFunction %void None %fn_type
    %buf = OpFunctionParameter %ptr_cross_uint
    %out = OpFunctionParameter %ptr_cross_uint

  %entry = OpLabel
               OpBranch %copy_header

; --- Copy permutation from global to SLM (single thread) ---
%copy_header = OpLabel
               OpLoopMerge %copy_exit %copy_header None
   %ci_phi = OpPhi %uint %uint_0 %entry %ci_next %copy_body
    %ccond = OpULessThan %bool %ci_phi %uint_{n_elems}
               OpBranchConditional %ccond %copy_body %copy_exit

%copy_body = OpLabel
 %c_gptr = OpPtrAccessChain %ptr_cross_uint %buf %ci_phi
 %c_val = OpLoad %uint %c_gptr Aligned 4
 %c_sptr = OpAccessChain %ptr_work_uint %slm %ci_phi
               OpStore %c_sptr %c_val Aligned 4
  %ci_next = OpIAdd %uint %ci_phi %uint_1
               OpBranch %copy_header

%copy_exit = OpLabel
               OpBranch %chase_init

; --- Chase through SLM ---
%chase_init = OpLabel
               OpBranch %chase_header

%chase_header = OpLabel
               OpLoopMerge %chase_exit %chase_header None
  %idx_phi = OpPhi %uint %uint_0 %chase_init %idx_next %chase_body
    %i_phi = OpPhi %uint %uint_0 %chase_init %i_next %chase_body
   %chcond = OpULessThan %bool %i_phi %uint_{CHASE}
               OpBranchConditional %chcond %chase_body %chase_exit

%chase_body = OpLabel
  %slm_ptr = OpAccessChain %ptr_work_uint %slm %idx_phi
  %idx_next = OpLoad %uint %slm_ptr Aligned 4
    %i_next = OpIAdd %uint %i_phi %uint_1
               OpBranch %chase_header

%chase_exit = OpLabel
               OpStore %out %idx_phi Aligned 4
               OpReturn
               OpFunctionEnd
"""


def run_cmd(cmd, check=True):
    r = subprocess.run(cmd, shell=True, cwd=SCRIPT_DIR, capture_output=True, text=True)
    if check and r.returncode != 0:
        print(f"ERROR: {cmd}", file=sys.stderr)
        print(r.stderr, file=sys.stderr)
    return r


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    if not SPIRV_RUNNER.exists():
        run_cmd("g++ -std=c++17 -O2 -o spirv_runner spirv_runner.cpp -lze_loader -lm")

    ghz = 2.4
    sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

    results = []

    print(f"\n{'='*60}")
    print(f"SPIR-V Native SLM Pointer Chase ({CHASE} steps, single thread)")
    print(f"{'='*60}")
    print(f"  {'Size':>8s} {'median_ns':>10s} {'cyc/acc':>10s} {'ns/acc':>10s}")
    print(f"  {'-'*44}")

    for n in sizes:
        name = f"slm_chase_{n}"
        spvasm_path = SCRIPT_DIR / f"sweep_{name}.spvasm"
        spv_path = SCRIPT_DIR / f"sweep_{name}.spv"

        # Generate
        spvasm_path.write_text(gen_slm_chase(n))

        # Assemble
        r = run_cmd(f"spirv-as --target-env spv1.4 {spvasm_path} -o {spv_path}", check=False)
        if r.returncode != 0:
            print(f"  {n:>8d} ASSEMBLE FAILED: {r.stderr[:200]}")
            spvasm_path.unlink(missing_ok=True)
            continue

        # Compile
        r = run_cmd(f"ocloc compile -spirv_input -file {spv_path} -device {DEVICE} -output {name}", check=False)
        if r.returncode != 0 or "Build failed" in r.stdout + r.stderr:
            print(f"  {n:>8d} COMPILE FAILED: {(r.stderr+r.stdout)[:300]}")
            spvasm_path.unlink(missing_ok=True)
            spv_path.unlink(missing_ok=True)
            continue

        # Find binary
        bin_path = SCRIPT_DIR / f"{name}_bmg.bin"
        if not bin_path.exists():
            print(f"  {n:>8d} NO BINARY")
            continue

        # Disassemble for verification
        disasm_dir = SCRIPT_DIR / f"disasm_{name}"
        run_cmd(f"mkdir -p {disasm_dir}", check=False)
        run_cmd(f"ocloc disasm -file {bin_path} -dump {disasm_dir} -device {DEVICE}", check=False)
        slm_indicators = ""
        for f in disasm_dir.iterdir():
            if f.name.endswith('.asm'):
                asm_text = f.read_text()
                # Check for SLM-related instructions in GEN ASM
                if 'dword.' in asm_text.lower():
                    slm_indicators += "[has dword ops]"
                if 'store (' in asm_text.lower():
                    slm_indicators += "[has store]"
                # Print snippet for first kernel
                if n == sizes[0]:
                    print(f"  [GEN ASM first 20 lines for n={n}]:")
                    lines = [l for l in asm_text.split('\n') if l.strip()][:20]
                    for line in lines:
                        print(f"    {line}")
                break

        # Run: wg=1 thread, 1 wg, 50 repeats
        r = run_cmd(f"./spirv_runner {spv_path} slm_chase 1 1 {REPEATS}", check=False)
        if r.returncode != 0:
            print(f"  {n:>8d} RUN FAILED: {r.stderr[:200]}")
        else:
            m = re.search(r'Median=([\d.]+)\s+ns', r.stdout)
            if m:
                median = float(m.group(1))
                cyc = median * ghz / CHASE
                ns_per = median / CHASE
                sz_bytes = n * 4
                sz_str = f"{sz_bytes}B" if sz_bytes < 1024 else f"{sz_bytes//1024}KB"
                print(f"  {sz_str:>8s} {median:>10.0f} {cyc:>10.1f} {ns_per:>10.1f}  {slm_indicators}")
                results.append({
                    'n_elems': n,
                    'size_str': sz_str,
                    'median_ns': median,
                    'cycles_per_access': f"{cyc:.1f}",
                    'ns_per_access': f"{ns_per:.1f}",
                })

        # Cleanup
        for f in [spvasm_path, spv_path, bin_path]:
            f.unlink(missing_ok=True)
        shutil.rmtree(disasm_dir, ignore_errors=True)
        for f in SCRIPT_DIR.glob(f"{name}*"):
            f.unlink(missing_ok=True)

    # Save results
    if results:
        csv_path = RESULTS_DIR / "slm_latency_sweep.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {csv_path}")

    # Comparison
    print(f"\n--- Comparison ---")
    print(f"  SYCL local_accessor SLM: ~80 cycles (256B)")
    print(f"  L1 data cache (global ptr chase): ~71 cycles (1KB)")
    print(f"  NVIDIA shared memory: ~30 cycles")


if __name__ == "__main__":
    main()
