---
name: gpu-microbench-xe2
description: This skill should be used when the user asks to "write a GPU microbenchmark", "measure GPU latency", "measure GPU bandwidth", "characterize Intel GPU", "write a SPIR-V kernel for testing", "benchmark XMX/DPAS", "measure SLM latency", "profile Intel Arc", "GPU microbenchmark methodology", or discusses writing low-level GPU benchmarks for Intel Xe2 (Battlemage) GPUs.
---

# GPU Microbenchmark Development for Intel Xe2 (Battlemage)

Skill for writing rigorous GPU microbenchmarks on Intel Arc Pro B60 (BMG-G21, Xe2 architecture), covering SPIR-V kernel development, SYCL bandwidth tests, Level Zero submission, and unitrace validation.

## Overview

Writing GPU microbenchmarks on Intel Xe2 requires careful attention to compiler behavior (IGC constant folding), memory allocation semantics, and hardware counter interpretation. This skill documents hard-won lessons from building a comprehensive microbenchmark suite.

## Architecture Quick Reference

| Component | Intel Xe2 (B60) | Notes |
|---|---|---|
| Xe Cores | 20 | Each has L1 + SLM |
| XMX Engines | 160 | One per EU, dpas.8x8 |
| Sub-group Size | 16 threads | vs NVIDIA's 32 (warp) |
| L1 Data Cache | 128 KB/Xe Core | `send.ugm` path, ~71 cycles |
| SLM | Up to 64 KB usable/Xe Core | 128 KB SRAM split: 64K SLM + 64K L1 |
| L2 Cache | 18 MB shared | Per chip, not per core |
| GDDR6 | 24 GB, 256-bit bus, ~576 GB/s peak | 18 Gbps data rate |
| Core Clock | 2.4 GHz | Confirmed via unitrace |
| DPAS Latency | 33 cycles (BF16/FP16) | Dependent chain slope |
| DPAS Reciprocal Throughput | 16.1 cycles | Directly measured, full GPU sweep |
| Threads/EU | 8 | TLP enables XMX pipelining |

## SPIR-V Pipeline

```
kernel.spvasm → spirv-as [--target-env spv1.4] → kernel.spv
  → ocloc compile -spirv_input -file kernel.spv -device bmg-g21 -output name
  → name_bmg.bin
  → ocloc disasm -file name_bmg.bin -dump dir/ -device bmg-g21 → GEN ASM
```

Submit via Level Zero: `zeModuleCreate → zeKernelCreate → zeCommandListAppendLaunchKernel`

## Critical Lessons (Gotchas)

### 1. IGC Constant Folding — Always Load Operands from Global Memory

**Problem**: SYCL/SPIR-V kernels with constant operands get optimized away by IGC. A chain of `a = a + 1` repeated N times will be folded to `a = a + N`.

**Solution**: Load all operands from global memory buffers. Store results to global memory to prevent DCE.

```spirv
; BAD: constants get folded
%val = OpConstant %uint 42
%result = OpIAdd %uint %acc %val

; GOOD: loaded from buffer, compiler can't eliminate
%ptr = OpPtrAccessChain %ptr_cross_uint %buf %idx
%val = OpLoad %uint %ptr Aligned 4
%result = OpIAdd %uint %acc %val
```

### 2. SPIR-V CooperativeMatrixMulAddKHR — No Trailing Operands

**Problem**: `spirv-as` rejects `None` or `0` as trailing operands for `OpCooperativeMatrixMulAddKHR`.

**Solution**: Omit the operands parameter entirely (it's optional in the spec):
```spirv
; BAD
OpCooperativeMatrixMulAddKHR %acc %a %b %c None

; GOOD
OpCooperativeMatrixMulAddKHR %acc %a %b %c
```

### 3. Memory Allocation Semantics — malloc_device vs buffer

**Problem**: `sycl::buffer` uses `zeMemAllocShared` (unified shared memory) which has coherency overhead. Bandwidth measurements can be **2.5× lower** than device-local memory.

**Solution**: Use `sycl::malloc_device<T>()` for bandwidth benchmarks:
```cpp
// BAD: shared memory, 2.5× overhead for reads
sycl::buffer<float, 1> buf(n);

// GOOD: device-local memory, accurate DRAM measurement
float* buf = sycl::malloc_device<float>(n, q);
```

**Impact**: Read bandwidth: 139 GB/s (shared) → 346 GB/s (device). Write: similar improvement.

### 4. SLM Must Use Workgroup Address Space

**Problem**: A global pointer chase (`OpPtrAccessChain %ptr_cross_uint`) measures **L1 data cache** (~71 cycles), not SLM (~46 cycles). They are different hardware paths.

**Solution**: Use `OpVariable Workgroup` for true SLM measurement:
```spirv
; Module scope — SLM variable
%slm = OpVariable %ptr_work_arr Workgroup

; Chase through SLM (generates send.slm in GEN ASM)
%ptr = OpAccessChain %ptr_work_uint %slm %idx
%val = OpLoad %uint %ptr Aligned 4
```

**Validation**: Always disassemble and check for `send.slm` vs `send.ugm` in GEN ASM.

### 5. SYCL Adds ~34 Cycles Overhead per SLM Access

SYCL `local_accessor` generates extra address computation and barrier code, inflating SLM latency from ~46 to ~80 cycles. For precise measurements, use raw SPIR-V.

### 6. Bandwidth Calculation Bug Pattern

**Problem**: When `global_range ≈ buffer_size`, threads overflow the buffer after the first iteration:
```cpp
// BUG: idx overflows for i >= 1 when global_range ≈ n
idx = gid + i * global_range;  // only i=0 is valid
bytes = n * n_iter * sizeof(float);  // overcounts by n_iter×
```

**Solution**: Use fewer threads doing more work:
```cpp
// CORRECT: stride = total_threads, covers whole buffer exactly once
int n_iter = ceil(n / total_threads);
idx = gid + i * total_threads;
bytes = n * sizeof(float);  // actual buffer size
```

### 7. Latency vs Reciprocal Throughput

- **Latency** = time from instruction issue to result available (33 cycles for DPAS)
- **Reciprocal throughput** = minimum interval between independent instruction issues (~16 cycles)
- Use **slope method** (vary N, linear regression) for latency, not naive total/N
- For throughput: use **independent chains** (ILP), not dependent chains

**Peak TFLOPS calculation**: Don't use latency in the peak formula. Use reciprocal throughput derived from full-chip benchmarks:
```
89.77 TFLOPS achieved / 160 XMX / 4096 FLOPs / 2.4 GHz = ~16 cycles reciprocal throughput
```

### 8. Unitrace Counter Interpretation

`GPU_MEMORY_BYTE_READ` measures actual DRAM reads, not cache reads. If data is already in L3, the counter will show very low values even though the kernel reads 1 GB through the cache path. Always check `LOAD_STORE_CACHE_BYTE_READ` alongside.

```bash
UNITRACE=/path/to/unitrace
$UNITRACE -q -g ComputeBasic ./your_benchmark 2>&1
# Key columns: GPU_MEMORY_BYTE_READ[bytes], GPU_MEMORY_BYTE_WRITE[bytes],
#              GPU_MEMORY_BYTE_READ_RATE[GBpS], GPU_MEMORY_BYTE_WRITE_RATE[GBpS]
#              LOAD_STORE_CACHE_BYTE_READ[bytes]
```

## Benchmark Templates

### Template: SPIR-V DPAS Latency (Dependent Chain)

```spirv
; SPIR-V 1.4
; DPAS BF16 Latency: N dependent MulAdd operations
               OpCapability Addresses
               OpCapability Kernel
               OpCapability Int64
          %1 = OpExtInstImport "OpenCL.std"
               OpMemoryModel Physical64 OpenCL
               OpEntryPoint Kernel %main "dpas_latency"
               OpExecutionMode %main LocalSize 16 1 1

     %void = OpTypeVoid
     %uint = OpTypeInt 32 0
    %ushort = OpTypeInt 16 0       ; BF16 stored as ushort
     %float = OpTypeFloat 32
     %ulong = OpTypeInt 64 0

; Cooperative matrix types: A(8×16 BF16), B(16×16 BF16), C(8×16 float)
  %cm_a = OpTypeCooperativeMatrixKHR %ushort %uint_3 %uint_8 %uint_16 %uint_0
  %cm_b = OpTypeCooperativeMatrixKHR %ushort %uint_3 %uint_16 %uint_16 %uint_1
  %cm_c = OpTypeCooperativeMatrixKHR %float %uint_3 %uint_8 %uint_16 %uint_2

; Load A, B, C from global memory (prevents constant folding)
; Loop N times: acc = A × B + acc  (dependent chain, r→r feedback)
; Store result to global memory (prevents DCE)
```

### Template: SYCL Bandwidth (Device Memory, Copy Pattern)

```cpp
sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::enable_profiling{}};
size_t n = 1024UL*1024*1024 / sizeof(float);  // 1 GB

float* src = sycl::malloc_device<float>(n, q);  // NOT sycl::buffer!
float* dst = sycl::malloc_device<float>(n, q);
q.fill(src, 1.0f, n).wait();

int n_wg = 4096, wg_size = 256;  // 1M threads for saturation
int total_threads = n_wg * wg_size;
int n_iter = (n + total_threads - 1) / total_threads;

// Copy kernel: dst[i] = src[i] (forces both DRAM read and write)
auto ev = q.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<1>(n_wg*wg_size, wg_size),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]]
    {
        int tid = it.get_global_id(0);
        for (int i = 0; i < n_iter; i++) {
            size_t idx = (size_t)tid + (size_t)i * total_threads;
            if (idx < n) dst[idx] = src[idx];
        }
    });
});
ev.wait();
auto ns = ev.get_profiling_info<sycl::info::event_profiling::command_end>()
        - ev.get_profiling_info<sycl::info::event_profiling::command_start>();
double bw = 2.0 * n * sizeof(float) / (ns * 1e-9) / 1e9;  // 2× for copy
printf("Copy: %.1f GB/s\n", bw);
```

### Template: SPIR-V SLM Pointer Chase

Key: use `OpVariable Workgroup` at module scope, `OpAccessChain` for element access:
```spirv
%arr_N = OpTypeArray %uint %uint_N
%ptr_work_arr = OpTypePointer Workgroup %arr_N
%ptr_work_uint = OpTypePointer Workgroup %uint
%slm = OpVariable %ptr_work_arr Workgroup

; In function body:
%ptr = OpAccessChain %ptr_work_uint %slm %idx
%val = OpLoad %uint %ptr Aligned 4
```

GEN ASM should show `send.slm` (not `send.ugm`).

## Validation Checklist

For every microbenchmark:

- [ ] **GEN ASM verification**: `ocloc disasm` → count target instructions (dpas, send.slm, send.ugm)
- [ ] **No constant folding**: Operands loaded from global memory buffers
- [ ] **No DCE**: Results stored to global memory
- [ ] **Correct memory allocation**: `malloc_device` for bandwidth, not `buffer`
- [ ] **Timing**: SYCL profiling events (not host chrono) for kernel-level timing
- [ ] **Slope method**: Vary N, linear regression to eliminate dispatch overhead (~6μs)
- [ ] **Warmup**: ≥5 iterations discarded, 20-50 measured
- [ ] **Unitrace cross-check**: `GPU_MEMORY_BYTE_READ/WRITE` for DRAM traffic validation

## Known Limitations

- **INT8 cooperative matrix**: `ocloc` segfaults with `OpTypeCooperativeMatrixKHR %uchar` + `MulAddKHR` (compiler bug). Type declaration alone works, but any MulAddKHR crashes.
- **FP16 vs BF16**: Both work, GEN ASM suffixes differ (`:hf` vs `:bf`)
- **FP64**: Very limited on Xe2 (only 2 FP64 units per Xe Core)
- **SPIR-V template generation**: Generated SPIR-V from scratch causes `ocloc` segfaults. Always modify a known-working template via string replacement.
- **SLM practical limit**: 64 KB max (128 KB SRAM split with L1). Allocations >64 KB fail.
- **Driver memory info**: Level Zero/xpu-smi reports wrong bus width (64-bit) and clock (0 MHz). True spec: 256-bit at 18 Gbps.

## Additional Gotchas

### 9. Scalar vs Vector Memory Access — 3.6× Bandwidth Difference

Scalar `float` loads achieve only ~204 GB/s read bandwidth (even with ILP/extra threads).
Vector `float4` loads achieve **538 GB/s** (93% of 576 GB/s peak). The 2.6× gap is due to
per-thread memory transaction efficiency. Always use vectorized loads for bandwidth benchmarks.

### 10. XMX Pipeline Saturation Requires ILP≥14

ILP=8 independent DPAS chains within a single sub-group achieve only ~31 cycles/dpas
(close to 33 cycle latency). But **ILP≥14 saturates the pipeline at ~16 cycles/dpas**,
matching reciprocal throughput. ILP=16 uses all 256 GRF registers with no spills.
For GEMM: aim for at least 14 independent DPAS chains per sub-group to maximize XMX utilization.

### 11. L1 Pointer Chase Growth Is NOT TLB

Latency growth from 71→145 cycles in L1 range (1-128 KB) is due to reduced cache line
utilization with random access patterns, not TLB effects. PAGE_STRIDE test (1 element per
4KB page) stays flat at ~64 cycles across all sizes, proving TLB is not the bottleneck.

### 12. Host Timing Overhead

Host-side `chrono` timing includes ~6.7 μs submit+wait overhead. For kernels <50 μs,
use GPU event profiling or the slope method. For kernels >1 ms, host and GPU timing agree within 1%.

### 13. No XMX+ALU Dual-Issue on Xe2

Adding 0-16 independent FP32 ALU operations per DPAS iteration has zero impact on cycle count.
The EU thread is completely blocked during the SBID stall (~33 cycles). Xe2 cannot dual-issue
XMX and ALU operations. ALU work is "free" only because it executes during the DPAS wait.

### 14. Barriers Can IMPROVE Throughput

Counter-intuitively, OpControlBarrier every 4-16 iterations **improves** DPAS throughput
by up to 39% vs no barriers. Without synchronization, SGs compete for shared resources.
Moderate barriers keep SGs in lockstep, improving XMX scheduling. Cost: only 2-11 cycles
per barrier (nearly free).

### 15. Cooperative Matrix Reload from Cache Is Free

Reloading A/B tiles from cache-resident global memory each iteration costs zero extra cycles
(and can be slightly faster due to reduced register pressure). The cooperative matrix load
overlaps with the DPAS SBID stall. GEMM can freely reload operands when data is in L1/L2.

### 16. Per-WG Dispatch Cost Is ~40 ns

Each work-group adds ~40 ns (96 cycles) of scheduling overhead, with a fixed ~3.7 μs
dispatch latency. Larger WGs (more SGs) amortize this cost: 16 SGs/WG = 2.5 ns/SG.

## Tools

| Tool | Path/Command | Purpose |
|---|---|---|
| `icpx` | `icpx -fsycl -O2` | SYCL compilation |
| `spirv-as` | `spirv-as [--target-env spv1.4]` | SPIR-V assembly |
| `ocloc` | `ocloc compile -spirv_input -device bmg-g21` | SPIR-V → GEN binary |
| `ocloc disasm` | `ocloc disasm -file X.bin -dump dir/ -device bmg-g21` | GEN binary → ASM |
| `unitrace` | `~/pti-gpu/tools/unitrace/build/unitrace -q -g ComputeBasic` | Hardware counters |
| `xpu-smi` | `xpu-smi` | Power monitoring, device info |
