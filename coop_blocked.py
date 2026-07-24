#!/usr/bin/env python3
"""Register-blocked coop-matrix: 4x4 tiles of 8x16 per SG per K-step (like a real GEMM).
If IGC packs tiles usefully, useful TF jumps ~8x vs naive single-tile chains."""
import sys
from pathlib import Path
SCRIPT_DIR = Path("/home/intel/b70-microbench")
sys.path.insert(0, str(SCRIPT_DIR))
from run_b70_xmx_probe import build_and_compile, cleanup, run_benchmark

GHZ = 2.4
TILE_FLOPS = 2 * 8 * 16 * 16  # 4096 per SG mad

def gen_blocked(rb, n_iter):
    """rb x rb register block of 8x16 KHR tiles; rb A-tiles + rb B-tiles shared."""
    L = [
        "; SPIR-V 1.4", f"; blocked {rb}x{rb}, {n_iter} k-steps",
        "               OpCapability Addresses", "               OpCapability Kernel",
        "               OpCapability Int64", "               OpCapability Int16",
        "               OpCapability CooperativeMatrixKHR",
        '               OpExtension "SPV_KHR_cooperative_matrix"',
        '          %1 = OpExtInstImport "OpenCL.std"',
        "               OpMemoryModel Physical64 OpenCL",
        '               OpEntryPoint Kernel %main "dpas_blocked"',
        "               OpExecutionMode %main SubgroupSize 16",
        "     %void   = OpTypeVoid", "     %bool   = OpTypeBool",
        "     %uint   = OpTypeInt 32 0", "     %ulong  = OpTypeInt 64 0",
        "     %float  = OpTypeFloat 32", "     %ushort = OpTypeInt 16 0",
        "   %uint_0   = OpConstant %uint 0", "   %uint_1   = OpConstant %uint 1",
        "   %uint_2   = OpConstant %uint 2", "   %uint_3   = OpConstant %uint 3",
        "   %uint_8   = OpConstant %uint 8", "  %uint_16   = OpConstant %uint 16",
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
    for i in range(rb):
        L.append(f"  %a{i} = OpCooperativeMatrixLoadKHR %cm_a %buf_a %uint_0 %uint_16 None")
    for j in range(rb):
        L.append(f"  %b{j} = OpCooperativeMatrixLoadKHR %cm_b %buf_b %uint_0 %uint_16 None")
    for i in range(rb):
        for j in range(rb):
            L.append(f"  %ci{i}{j} = OpCooperativeMatrixLoadKHR %cm_acc %buf_c %uint_0 %uint_16 None")
    L.append("               OpBranch %lh")
    L.append("%lh = OpLabel")
    L.append("               OpLoopMerge %lx %lh None")
    for i in range(rb):
        for j in range(rb):
            L.append(f"  %cp{i}{j} = OpPhi %cm_acc %ci{i}{j} %entry %cn{i}{j} %lb")
    L.append("    %i_phi = OpPhi %uint %uint_0 %entry %i_next %lb")
    L.append(f"      %cond = OpULessThan %bool %i_phi %uint_{n_iter}")
    L.append("               OpBranchConditional %cond %lb %lx")
    L.append("%lb = OpLabel")
    for i in range(rb):
        for j in range(rb):
            L.append(f"  %cn{i}{j} = OpCooperativeMatrixMulAddKHR %cm_acc %a{i} %b{j} %cp{i}{j}")
    L.append("    %i_next = OpIAdd %uint %i_phi %uint_1")
    L.append("               OpBranch %lh")
    L.append("%lx = OpLabel")
    for i in range(rb):
        for j in range(rb):
            L.append(f"               OpCooperativeMatrixStoreKHR %buf_c %cp{i}{j} %uint_0 %uint_16 None")
    L.append("               OpReturn")
    L.append("               OpFunctionEnd")
    return "\n".join(L)

for rb, n_iter in [(2, 16384), (4, 8192)]:
    name = f"blk{rb}"
    spv = build_and_compile(gen_blocked(rb, n_iter), name)
    if not spv:
        print(f"rb={rb}: BUILD FAILED"); continue
    for n_wg, tag in [(32, "2WI/EU")]:
        median, err = run_benchmark(spv, 16, n_wg, repeats=7)
        if median is None:
            print(f"rb={rb} {tag}: RUN FAILED {err}"); continue
        mads = rb * rb * n_iter * n_wg
        tf = mads * TILE_FLOPS / (median * 1e-9) / 1e12
        instrs = rb * rb * n_iter
        cyc = median * GHZ / instrs
        print(f"rb={rb} {tag} n_wg={n_wg}: med={median:10.0f}ns  useful={tf:6.1f} TF  "
              f"{cyc:5.2f} cyc/mad-per-thread")
    cleanup(name)
