#!/usr/bin/env python3
"""
Writeback Bandwidth Sweep — verify low-bitwidth scalar store bandwidth loss.

Hypothesis: On Intel Xe2, a single 16-bit store still sends a 32-bit store message,
wasting half the effective bandwidth. Similarly for 8-bit stores.

Kernel variants:
  1. fp32_scalar  — each thread copies 1×float (32-bit) per iteration  [baseline]
  2. bf16_scalar  — each thread copies 1×ushort (16-bit) per iteration
  3. bf16_vec4    — each thread copies 4×ushort (64-bit) per iteration (vectorized)
  4. u8_scalar    — each thread copies 1×uchar (8-bit) per iteration
  5. u8_vec4      — each thread copies 4×uchar packed into uint32 per iteration

Each kernel reads from buf[0], writes to buf[1], and stores a checksum to buf[2].
Working set exceeds L2 (18 MB) for true DRAM measurement.
"""

import subprocess, os, sys, re, csv, argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
SPIRV_RUNNER = SCRIPT_DIR / "spirv_runner"
RESULTS_DIR = SCRIPT_DIR / "results"
DEVICE = "bmg-g21"
REPEATS = 30

# ─────────────────────────────────────────────────────────────────────────────
# SPIR-V Kernel Templates
# SPIR-V layout order: Capabilities → ExtInst → MemoryModel → EntryPoint
#   → ExecutionMode → Decorations → Types/Constants → Globals → Functions
# ─────────────────────────────────────────────────────────────────────────────

def _preamble(wg, n_iters, stride, kernel_name, extra_caps="", extra_types_before="", extra_types_after=""):
    """Common SPIR-V preamble with correct layout ordering."""
    return f"""; SPIR-V 1.4
; Writeback benchmark: stride={stride}, {n_iters} iters/thread
               OpCapability Addresses
               OpCapability Kernel
               OpCapability Int64{extra_caps}
          %1 = OpExtInstImport "OpenCL.std"
               OpMemoryModel Physical64 OpenCL
               OpEntryPoint Kernel %main "{kernel_name}"
               OpExecutionMode %main LocalSize {wg} 1 1

               OpDecorate %__spirv_BuiltInGlobalInvocationId LinkageAttributes "__spirv_BuiltInGlobalInvocationId" Import
               OpDecorate %__spirv_BuiltInGlobalInvocationId BuiltIn GlobalInvocationId

     %void = OpTypeVoid
     %bool = OpTypeBool
     %uint = OpTypeInt 32 0
    %ulong = OpTypeInt 64 0
     %float = OpTypeFloat 32
   %v3ulong = OpTypeVector %ulong 3
{extra_types_before}
   %uint_0 = OpConstant %uint 0
   %uint_1 = OpConstant %uint 1
  %uint_{n_iters} = OpConstant %uint {n_iters}
 %ulong_0 = OpConstant %ulong 0
%ulong_{stride} = OpConstant %ulong {stride}
{extra_types_after}
%ptr_input_v3ulong = OpTypePointer Input %v3ulong
%__spirv_BuiltInGlobalInvocationId = OpVariable %ptr_input_v3ulong Input
"""


# ── Variant 1: fp32 scalar copy ──────────────────────────────────────────────
def kernel_fp32_scalar(wg, stride, n_iters):
    return _preamble(wg, n_iters, stride, "copy_kernel",
        extra_types_before="""
%ptr_cross_float = OpTypePointer CrossWorkgroup %float
""",
        extra_types_after="""
 %float_0 = OpConstant %float 0
"""
    ) + """
  %fn_type = OpTypeFunction %void %ptr_cross_float %ptr_cross_float %ptr_cross_float

  %main = OpFunction %void None %fn_type
    %src = OpFunctionParameter %ptr_cross_float
    %dst = OpFunctionParameter %ptr_cross_float
    %out = OpFunctionParameter %ptr_cross_float

  %entry = OpLabel
  %gid_vec = OpLoad %v3ulong %__spirv_BuiltInGlobalInvocationId
    %gid_x = OpCompositeExtract %ulong %gid_vec 0
               OpBranch %loop_header

%loop_header = OpLabel
               OpLoopMerge %loop_exit %loop_header None
   %sum_phi = OpPhi %float %float_0 %entry %sum_next %loop_body
     %i_phi = OpPhi %uint %uint_0 %entry %i_next %loop_body
      %cond = OpULessThan %bool %i_phi %uint_{n_iters}
               OpBranchConditional %cond %loop_body %loop_exit

%loop_body = OpLabel
  %i_ulong = OpUConvert %ulong %i_phi
  %stride_val = OpIMul %ulong %i_ulong %ulong_{stride}
     %idx = OpIAdd %ulong %gid_x %stride_val
     %rptr = OpPtrAccessChain %ptr_cross_float %src %idx
      %val = OpLoad %float %rptr Aligned 4
     %wptr = OpPtrAccessChain %ptr_cross_float %dst %idx
               OpStore %wptr %val Aligned 4
 %sum_next = OpFAdd %float %sum_phi %val
   %i_next = OpIAdd %uint %i_phi %uint_1
               OpBranch %loop_header

%loop_exit = OpLabel
               OpStore %out %sum_phi Aligned 4
               OpReturn
               OpFunctionEnd
""".format(n_iters=n_iters, stride=stride)


# ── Variant 2: bf16 scalar copy (ushort) ─────────────────────────────────────
def kernel_bf16_scalar(wg, stride, n_iters):
    return _preamble(wg, n_iters, stride, "copy_kernel",
        extra_types_before="""
%ushort = OpTypeInt 16 0
%ptr_cross_ushort = OpTypePointer CrossWorkgroup %ushort
%ptr_cross_float = OpTypePointer CrossWorkgroup %float
""",
        extra_types_after="""
 %float_0 = OpConstant %float 0
"""
    ) + """
  %fn_type = OpTypeFunction %void %ptr_cross_ushort %ptr_cross_ushort %ptr_cross_float

  %main = OpFunction %void None %fn_type
    %src = OpFunctionParameter %ptr_cross_ushort
    %dst = OpFunctionParameter %ptr_cross_ushort
    %out = OpFunctionParameter %ptr_cross_float

  %entry = OpLabel
  %gid_vec = OpLoad %v3ulong %__spirv_BuiltInGlobalInvocationId
    %gid_x = OpCompositeExtract %ulong %gid_vec 0
               OpBranch %loop_header

%loop_header = OpLabel
               OpLoopMerge %loop_exit %loop_header None
     %i_phi = OpPhi %uint %uint_0 %entry %i_next %loop_body
      %cond = OpULessThan %bool %i_phi %uint_{n_iters}
               OpBranchConditional %cond %loop_body %loop_exit

%loop_body = OpLabel
  %i_ulong = OpUConvert %ulong %i_phi
  %stride_val = OpIMul %ulong %i_ulong %ulong_{stride}
     %idx = OpIAdd %ulong %gid_x %stride_val
     %rptr = OpPtrAccessChain %ptr_cross_ushort %src %idx
      %val = OpLoad %ushort %rptr Aligned 2
     %wptr = OpPtrAccessChain %ptr_cross_ushort %dst %idx
               OpStore %wptr %val Aligned 2
   %i_next = OpIAdd %uint %i_phi %uint_1
               OpBranch %loop_header

%loop_exit = OpLabel
               OpStore %out %float_0 Aligned 4
               OpReturn
               OpFunctionEnd
""".format(n_iters=n_iters, stride=stride)


# ── Variant 3: bf16 vec4 copy (4×ushort = 64bit per store) ──────────────────
def kernel_bf16_vec4(wg, stride, n_iters):
    return _preamble(wg, n_iters, stride, "copy_kernel",
        extra_types_before="""
%ushort = OpTypeInt 16 0
%v4ushort = OpTypeVector %ushort 4
%ptr_cross_v4ushort = OpTypePointer CrossWorkgroup %v4ushort
%ptr_cross_float = OpTypePointer CrossWorkgroup %float
""",
        extra_types_after="""
 %float_0 = OpConstant %float 0
"""
    ) + """
  %fn_type = OpTypeFunction %void %ptr_cross_v4ushort %ptr_cross_v4ushort %ptr_cross_float

  %main = OpFunction %void None %fn_type
    %src = OpFunctionParameter %ptr_cross_v4ushort
    %dst = OpFunctionParameter %ptr_cross_v4ushort
    %out = OpFunctionParameter %ptr_cross_float

  %entry = OpLabel
  %gid_vec = OpLoad %v3ulong %__spirv_BuiltInGlobalInvocationId
    %gid_x = OpCompositeExtract %ulong %gid_vec 0
               OpBranch %loop_header

%loop_header = OpLabel
               OpLoopMerge %loop_exit %loop_header None
     %i_phi = OpPhi %uint %uint_0 %entry %i_next %loop_body
      %cond = OpULessThan %bool %i_phi %uint_{n_iters}
               OpBranchConditional %cond %loop_body %loop_exit

%loop_body = OpLabel
  %i_ulong = OpUConvert %ulong %i_phi
  %stride_val = OpIMul %ulong %i_ulong %ulong_{stride}
     %idx = OpIAdd %ulong %gid_x %stride_val
     %rptr = OpPtrAccessChain %ptr_cross_v4ushort %src %idx
      %val = OpLoad %v4ushort %rptr Aligned 8
     %wptr = OpPtrAccessChain %ptr_cross_v4ushort %dst %idx
               OpStore %wptr %val Aligned 8
   %i_next = OpIAdd %uint %i_phi %uint_1
               OpBranch %loop_header

%loop_exit = OpLabel
               OpStore %out %float_0 Aligned 4
               OpReturn
               OpFunctionEnd
""".format(n_iters=n_iters, stride=stride)


# ── Variant 4: u8 scalar copy ───────────────────────────────────────────────
def kernel_u8_scalar(wg, stride, n_iters):
    return _preamble(wg, n_iters, stride, "copy_kernel",
        extra_types_before="""
%uchar = OpTypeInt 8 0
%ptr_cross_uchar = OpTypePointer CrossWorkgroup %uchar
%ptr_cross_float = OpTypePointer CrossWorkgroup %float
""",
        extra_types_after="""
 %float_0 = OpConstant %float 0
"""
    ) + """
  %fn_type = OpTypeFunction %void %ptr_cross_uchar %ptr_cross_uchar %ptr_cross_float

  %main = OpFunction %void None %fn_type
    %src = OpFunctionParameter %ptr_cross_uchar
    %dst = OpFunctionParameter %ptr_cross_uchar
    %out = OpFunctionParameter %ptr_cross_float

  %entry = OpLabel
  %gid_vec = OpLoad %v3ulong %__spirv_BuiltInGlobalInvocationId
    %gid_x = OpCompositeExtract %ulong %gid_vec 0
               OpBranch %loop_header

%loop_header = OpLabel
               OpLoopMerge %loop_exit %loop_header None
     %i_phi = OpPhi %uint %uint_0 %entry %i_next %loop_body
      %cond = OpULessThan %bool %i_phi %uint_{n_iters}
               OpBranchConditional %cond %loop_body %loop_exit

%loop_body = OpLabel
  %i_ulong = OpUConvert %ulong %i_phi
  %stride_val = OpIMul %ulong %i_ulong %ulong_{stride}
     %idx = OpIAdd %ulong %gid_x %stride_val
     %rptr = OpPtrAccessChain %ptr_cross_uchar %src %idx
      %val = OpLoad %uchar %rptr Aligned 1
     %wptr = OpPtrAccessChain %ptr_cross_uchar %dst %idx
               OpStore %wptr %val Aligned 1
   %i_next = OpIAdd %uint %i_phi %uint_1
               OpBranch %loop_header

%loop_exit = OpLabel
               OpStore %out %float_0 Aligned 4
               OpReturn
               OpFunctionEnd
""".format(n_iters=n_iters, stride=stride)


# ── Variant 5: u8 vec4 copy (4×uchar packed as uint32) ──────────────────────
def kernel_u8_vec4(wg, stride, n_iters):
    return _preamble(wg, n_iters, stride, "copy_kernel",
        extra_types_before="""
%ptr_cross_uint = OpTypePointer CrossWorkgroup %uint
%ptr_cross_float = OpTypePointer CrossWorkgroup %float
""",
        extra_types_after="""
 %float_0 = OpConstant %float 0
"""
    ) + """
  %fn_type = OpTypeFunction %void %ptr_cross_uint %ptr_cross_uint %ptr_cross_float

  %main = OpFunction %void None %fn_type
    %src = OpFunctionParameter %ptr_cross_uint
    %dst = OpFunctionParameter %ptr_cross_uint
    %out = OpFunctionParameter %ptr_cross_float

  %entry = OpLabel
  %gid_vec = OpLoad %v3ulong %__spirv_BuiltInGlobalInvocationId
    %gid_x = OpCompositeExtract %ulong %gid_vec 0
               OpBranch %loop_header

%loop_header = OpLabel
               OpLoopMerge %loop_exit %loop_header None
     %i_phi = OpPhi %uint %uint_0 %entry %i_next %loop_body
      %cond = OpULessThan %bool %i_phi %uint_{n_iters}
               OpBranchConditional %cond %loop_body %loop_exit

%loop_body = OpLabel
  %i_ulong = OpUConvert %ulong %i_phi
  %stride_val = OpIMul %ulong %i_ulong %ulong_{stride}
     %idx = OpIAdd %ulong %gid_x %stride_val
     %rptr = OpPtrAccessChain %ptr_cross_uint %src %idx
      %val = OpLoad %uint %rptr Aligned 4
     %wptr = OpPtrAccessChain %ptr_cross_uint %dst %idx
               OpStore %wptr %val Aligned 4
   %i_next = OpIAdd %uint %i_phi %uint_1
               OpBranch %loop_header

%loop_exit = OpLabel
               OpStore %out %float_0 Aligned 4
               OpReturn
               OpFunctionEnd
""".format(n_iters=n_iters, stride=stride)


# ─────────────────────────────────────────────────────────────────────────────
# Build and run pipeline
# ─────────────────────────────────────────────────────────────────────────────

KERNELS = {
    'fp32_scalar': kernel_fp32_scalar,
    'bf16_scalar': kernel_bf16_scalar,
    'bf16_vec4':   kernel_bf16_vec4,
    'u8_scalar':   kernel_u8_scalar,
    'u8_vec4':     kernel_u8_vec4,
}

# Element size in bytes for each variant
ELEM_BYTES = {
    'fp32_scalar': 4,
    'bf16_scalar': 2,
    'bf16_vec4':   8,   # 4 × 2 bytes
    'u8_scalar':   1,
    'u8_vec4':     4,   # 4 × 1 byte packed
}


def run_cmd(cmd, check=True):
    r = subprocess.run(cmd, shell=True, cwd=SCRIPT_DIR, capture_output=True, text=True)
    if check and r.returncode != 0:
        print(f"ERROR: {cmd}", file=sys.stderr)
        print(r.stderr, file=sys.stderr)
    return r


def build_and_run(variant, n_wg, wg=256, n_iters=256, buf_mb=256, repeats=REPEATS):
    """Build SPIR-V kernel and run via spirv_runner.
    Returns (median_ns, bandwidth_gbps) or (None, None) on failure."""

    gen_func = KERNELS[variant]
    elem_bytes = ELEM_BYTES[variant]
    stride = n_wg * wg
    total_threads = n_wg * wg

    # Working set = total_threads * n_iters * elem_bytes
    working_set_mb = total_threads * n_iters * elem_bytes / (1024 * 1024)

    # Buffer must be at least as large as working set
    buf_kb = max(buf_mb * 1024, int(working_set_mb * 1024) + 1024)

    spvasm = gen_func(wg, stride, n_iters)
    tag = f"{variant}_nwg{n_wg}"
    spvasm_path = SCRIPT_DIR / f"wb_{tag}.spvasm"
    spv_path = SCRIPT_DIR / f"wb_{tag}.spv"

    spvasm_path.write_text(spvasm)

    # Assemble
    r = run_cmd(f"spirv-as {spvasm_path} -o {spv_path}", check=False)
    if r.returncode != 0:
        print(f"  spirv-as failed: {r.stderr[:300]}")
        return None, None

    # Run via spirv_runner (JIT compiles SPIR-V)
    cmd = f"./spirv_runner {spv_path} copy_kernel {wg} {n_wg} {repeats} {buf_kb}"
    r = run_cmd(cmd, check=False)
    if r.returncode != 0:
        print(f"  run failed: {r.stderr[:500]}")
        return None, None

    # Parse median time
    m = re.search(r'Median=([\d.]+)\s+ns', r.stdout)
    if not m:
        print(f"  parse error: {r.stdout[:300]}")
        return None, None

    median_ns = float(m.group(1))

    # Bandwidth calculation:
    # Write bandwidth: total_threads * n_iters * elem_bytes (only the store side)
    # Copy bandwidth: 2× that (read + write)
    total_write_bytes = total_threads * n_iters * elem_bytes
    bw_write_gbps = total_write_bytes / (median_ns * 1e-9) / 1e9
    bw_copy_gbps = 2 * bw_write_gbps

    # Cleanup
    for p in [spvasm_path, spv_path]:
        if p.exists():
            p.unlink()

    return median_ns, bw_write_gbps, bw_copy_gbps


def main():
    parser = argparse.ArgumentParser(description='Writeback bandwidth sweep')
    parser.add_argument('--disasm-only', action='store_true',
                        help='Only build + disasm, do not run benchmarks')
    parser.add_argument('--unitrace', action='store_true',
                        help='Run with unitrace hardware counters')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with fewer WGs and iters')
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    if not SPIRV_RUNNER.exists():
        print("Building spirv_runner...")
        run_cmd("g++ -std=c++17 -O2 -o spirv_runner spirv_runner.cpp -lze_loader -lm")

    # Configuration
    wg = 256
    if args.quick:
        n_wg = 256
        n_iters = 64
        buf_mb = 64
    else:
        n_wg = 4096   # 1M threads
        n_iters = 256
        buf_mb = 256

    total_threads = n_wg * wg
    print(f"Config: {n_wg} WGs × {wg} threads = {total_threads} total threads")
    print(f"        {n_iters} iterations/thread")
    print(f"        Buffer: {buf_mb} MB")
    print()

    # ── Step 1: Build kernels for disassembly ─────────────────────────────
    if args.disasm_only:
        print("=" * 80)
        print("Building all kernels for GEN ASM analysis")
        print("=" * 80)
        stride = n_wg * wg
        for variant in KERNELS:
            gen_func = KERNELS[variant]
            spvasm = gen_func(wg, stride, n_iters)
            tag = f"wb_{variant}_nwg{n_wg}"
            spvasm_path = SCRIPT_DIR / f"{tag}.spvasm"
            spv_path = SCRIPT_DIR / f"{tag}.spv"
            spvasm_path.write_text(spvasm)
            r = run_cmd(f"spirv-as {spvasm_path} -o {spv_path}", check=False)
            if r.returncode == 0:
                print(f"  {variant}: {spv_path} OK")
            else:
                print(f"  {variant}: FAILED - {r.stderr[:200]}")
        print(f"\nKernels built. Use ocloc to compile and disasm manually:")
        print(f"  ocloc compile -spirv_input -file wb_<variant>_nwg{n_wg}.spv -device bmg-g21 -output wb_<variant>")
        print(f"  ocloc disasm -file wb_<variant>_bmg.bin -dump wb_<variant>_disasm -device bmg-g21")
        return

    # ── Step 2: Run bandwidth sweep ───────────────────────────────────────
    print("=" * 80)
    print("Step 1: Bandwidth sweep (all variants)")
    print("=" * 80)

    results = []

    for variant in KERNELS:
        eb = ELEM_BYTES[variant]
        working_set_mb = total_threads * n_iters * eb / (1024 * 1024)

        print(f"\n--- {variant} (elem={eb}B, working_set={working_set_mb:.0f}MB) ---")

        result = build_and_run(variant, n_wg, wg, n_iters, buf_mb, REPEATS)
        if result[0] is not None:
            median_ns, bw_write_gbps, bw_copy_gbps = result
            print(f"  Median: {median_ns:.0f} ns")
            print(f"  Write BW: {bw_write_gbps:.1f} GB/s")
            print(f"  Copy BW:  {bw_copy_gbps:.1f} GB/s")

            results.append({
                'variant': variant,
                'elem_bytes': eb,
                'total_threads': total_threads,
                'n_iters': n_iters,
                'working_set_mb': round(working_set_mb, 1),
                'median_ns': round(median_ns, 1),
                'write_bw_gbps': round(bw_write_gbps, 1),
                'copy_bw_gbps': round(bw_copy_gbps, 1),
            })
        else:
            print(f"  FAILED")

    # ── Step 3: Save results ──────────────────────────────────────────────
    if results:
        csv_path = RESULTS_DIR / "writeback_sweep.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {csv_path}")

    # ── Step 4: Analysis ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("Step 2: Analysis — bandwidth comparison")
    print("=" * 80)

    if results:
        fp32 = next((r for r in results if r['variant'] == 'fp32_scalar'), None)
        bf16_s = next((r for r in results if r['variant'] == 'bf16_scalar'), None)
        bf16_v = next((r for r in results if r['variant'] == 'bf16_vec4'), None)
        u8_s = next((r for r in results if r['variant'] == 'u8_scalar'), None)
        u8_v = next((r for r in results if r['variant'] == 'u8_vec4'), None)

        print(f"\n{'Variant':<15s} {'ElemB':>6s} {'Med(ns)':>10s} {'WriteBW':>10s} {'CopyBW':>10s} {'vs fp32':>8s}")
        print("-" * 65)

        for r in results:
            vs_fp32 = ""
            if fp32 and r['variant'] != 'fp32_scalar':
                ratio = r['copy_bw_gbps'] / fp32['copy_bw_gbps']
                vs_fp32 = f"{ratio:.2f}x"
            print(f"{r['variant']:<15s} {r['elem_bytes']:>6d} {r['median_ns']:>10.0f} "
                  f"{r['write_bw_gbps']:>8.1f} GB {r['copy_bw_gbps']:>8.1f} GB {vs_fp32:>8s}")

        print("\n--- Key Comparisons ---")
        if fp32:
            print(f"\nfp32 scalar (baseline): {fp32['copy_bw_gbps']:.1f} GB/s")

        if bf16_s and fp32:
            ratio = bf16_s['copy_bw_gbps'] / fp32['copy_bw_gbps']
            print(f"bf16 scalar / fp32 scalar: {ratio:.2f}x")
            print(f"  Expected if NO waste: 2.0x (half the bytes → double the throughput)")
            print(f"  Expected if 32-bit msg: ~1.0x (same message overhead)")
            if ratio < 1.3:
                print(f"  >>> CONFIRMED: bf16 scalar store wastes bandwidth! <<<")
            elif ratio < 1.7:
                print(f"  >>> PARTIAL: Some waste detected, but not full 2× <<<")
            else:
                print(f"  >>> No significant waste detected <<<")

        if bf16_v and bf16_s:
            ratio = bf16_v['copy_bw_gbps'] / bf16_s['copy_bw_gbps']
            print(f"bf16 vec4 / bf16 scalar: {ratio:.2f}x (vectorization improvement)")

        if u8_s and fp32:
            ratio = u8_s['copy_bw_gbps'] / fp32['copy_bw_gbps']
            print(f"u8 scalar / fp32 scalar: {ratio:.2f}x")
            print(f"  Expected if NO waste: 4.0x (quarter the bytes)")
            print(f"  Expected if 32-bit msg: ~1.0x")

        if u8_v and u8_s:
            ratio = u8_v['copy_bw_gbps'] / u8_s['copy_bw_gbps']
            print(f"u8 vec4 / u8 scalar: {ratio:.2f}x (vectorization improvement)")

    # ── Step 5: Unitrace profiling ────────────────────────────────────────
    if args.unitrace and results:
        print("\n" + "=" * 80)
        print("Step 3: Unitrace hardware counter profiling")
        print("=" * 80)

        UNITRACE = Path("/home/intel/tianfeng/gemm/pti-gpu/tools/unitrace/build/unitrace")
        if not UNITRACE.exists():
            print(f"unitrace not found at {UNITRACE}")
            return

        for variant in KERNELS:
            print(f"\n--- {variant} unitrace ---")
            gen_func = KERNELS[variant]
            eb = ELEM_BYTES[variant]
            stride = n_wg * wg

            spvasm = gen_func(wg, stride, n_iters)
            tag = f"wb_{variant}_nwg{n_wg}"
            spvasm_path = SCRIPT_DIR / f"{tag}.spvasm"
            spv_path = SCRIPT_DIR / f"{tag}.spv"
            spvasm_path.write_text(spvasm)

            r = run_cmd(f"spirv-as {spvasm_path} -o {spv_path}", check=False)
            if r.returncode != 0:
                print(f"  spirv-as failed")
                continue

            buf_kb = max(buf_mb * 1024, int(total_threads * n_iters * eb / 1024) + 1024)
            bench_cmd = f"./spirv_runner {spv_path} copy_kernel {wg} {n_wg} 10 {buf_kb}"

            r = run_cmd(f"{UNITRACE} -q -g ComputeBasic {bench_cmd}", check=False)
            if r.returncode == 0:
                # Parse unitrace output for memory counters
                for line in r.stdout.split('\n'):
                    if any(k in line for k in ['GPU_MEMORY_BYTE', 'LOAD_STORE', 'READ_RATE', 'WRITE_RATE', 'Bytes']):
                        print(f"  {line.strip()}")
            else:
                print(f"  unitrace failed: {r.stderr[:300]}")

            for p in [spvasm_path, spv_path]:
                if p.exists():
                    p.unlink()


if __name__ == "__main__":
    main()
