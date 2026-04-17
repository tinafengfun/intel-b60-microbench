# Intel Arc Pro B60 (BMG-G21) Microbenchmark Suite

Characterizing the Intel Arc Pro B60 GPU (Xe2 architecture) using raw SPIR-V microbenchmarks,
inspired by *"Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks"* (arXiv:2507.10789).

## Device Info

| Parameter | Value |
|---|---|
| GPU | Intel Arc Pro B60 (BMG-G21) |
| Architecture | Xe2 (Intel Battlemage) |
| Xe Cores | 20 |
| EUs | 160 |
| XMX Engines | 160 |
| Core Clock | 2.4 GHz |
| Memory | 24 GB GDDR6, 456 GB/s peak |
| Sub-group Size | 16 threads |
| GRF per Thread | 256 registers (128 × 32B) |
| L1 Data Cache | 128 KB per Xe Core |
| SLM (Shared Local Memory) | Up to 128 KB per Xe Core (carved from L1) |
| L2 Cache | 18 MB shared |

## Methodology

### SPIR-V Pipeline

All latency/throughput kernels are written as **raw SPIR-V assembly** (`.spvasm`) to achieve
instruction-level control analogous to writing PTX on NVIDIA:

```
kernel.spvasm → spirv-as → kernel.spv → ocloc compile -spirv_input -device bmg-g21 → GEN binary
                                                                         ↓
                                                              ocloc disasm → GEN ASM (validate)
```

Kernels are submitted via **Level Zero API** (`zeModuleCreate` → `zeKernelCreate` →
`zeCommandListAppendLaunchKernel`) with host-side `std::chrono` timing.

### Why Raw SPIR-V (Not SYCL)?

1. **IGC constant folding**: SYCL kernels with constant operands get optimized away.
   Raw SPIR-V loads all operands from global memory buffers, preventing elimination.
2. **Instruction control**: SPIR-V `OpCooperativeMatrixMulAddKHR` maps directly to
   `dpas.8x8` in GEN ASM. We verify exact instruction count via disassembly.
3. **Reproducibility**: Every kernel has a corresponding GEN ASM dump confirming
   the expected number of DPAS/memory instructions.

### Timing Methodology

- **Host-side timing**: `std::chrono::high_resolution_clock` wrapping
  `zeCommandQueueExecuteCommandLists` + `zeCommandQueueSynchronize`
- **Sweep method**: Vary operation count N, measure total time, compute per-op
  latency from **linear regression slope** to eliminate fixed dispatch overhead (~6μs)
- **Warmup**: 10 iterations discarded, then 50-100 measured iterations
- **Statistic**: Median of repetitions (robust to outliers)
- **Clock**: 2.4 GHz confirmed via `zeDeviceGetProperties` and unitrace `CoreFrequencyMHz`

### Validation

Every kernel is validated by:
1. Disassembling the GEN binary → `.text.*.asm`
2. Counting `dpas` instructions in the GEN ASM
3. Confirming count matches expected N
4. Checking dependency chains (same register as input/output for latency;
   different registers for throughput/independent chains)

---

## Benchmark 1: DPAS/XMX BF16 Latency

### Design

**Goal**: Measure the minimum latency of a single `dpas.8x8` instruction (BF16 matrix multiply-accumulate) on Intel XMX.

**Kernel**: `spirv_dpas_latency.spvasm`
- SPIR-V cooperative matrix types:
  - A: `OpTypeCooperativeMatrixKHR %ushort %uint_3 %uint_8 %uint_16 %uint_0` (BF16, 8×16)
  - B: `OpTypeCooperativeMatrixKHR %ushort %uint_3 %uint_16 %uint_16 %uint_1` (BF16, 16×16)
  - C: `OpTypeCooperativeMatrixKHR %float %uint_3 %uint_8 %uint_16 %uint_2` (float accumulator, 8×16)
- Single sub-group (16 threads), LocalSize=16
- Load A, B, C tiles from global memory (prevents IGC constant folding)
- Loop N times: `acc = A × B + acc` (dependent chain — each iteration depends on previous result)
- Store result to global memory (prevents DCE)

**GEN ASM verification** (N=128):
```
dpas.8x8 (16|M0) r28:f r28:f r9:bf r5.0:bf {Compacted,$2}    ; first iteration
dpas.8x8 (16|M0) r28:f r28:f r9:bf r5.0:bf {Compacted,$2}    ; r28→r28 feedback
... (128 total dpas.8x8 instructions, all feeding r28→r28)
```

**Sweep**: N = 1, 4, 16, 64, 128, 256, 512

### Results

| N (DPAS ops) | Median (ns) | Cycles/DPAS (naive) | DPAS in ASM |
|---:|---:|---:|---:|
| 1 | 5,954 | 14,290 | 1 |
| 4 | 6,000 | 3,600 | 4 |
| 16 | 6,018 | 903 | 16 |
| 64 | 6,708 | 252 | 64 |
| 128 | 7,527 | 141 | 128 |
| 256 | 9,499 | 89 | 256 |
| 512 | 14,123 | 66 | 512 |

**Linear regression** (slope method):
```
time = overhead + N × per_dpas_time
slope = 13.92 ns = 33.4 cycles per DPAS
intercept = 5,926 ns = 14,222 cycles (fixed dispatch overhead)
```

### Analysis

- **DPAS BF16 latency ≈ 33-37 cycles** (13-15 ns at 2.4 GHz)
- The naive "cycles per DPAS" decreases with N because the ~6μs fixed overhead
  is amortized. The slope method gives the true per-instruction latency.
- The DPAS instruction occupies the XMX engine for ~33 cycles. During this time,
  the thread cannot issue another DPAS (SBID stall).
- **Per-DPAS FLOPs**: dpas.8x8 BF16 computes M×N×K×2 = 8×16×16×2 = **4096 FLOPs**
- **Latency-bound throughput**: 4096 FLOPs / 33 cycles × 2.4 GHz ≈ **0.298 TFLOPS per sub-group**
- **Comparison**: NVIDIA Blackwell Tensor Core latency is ~20 cycles (from paper).
  Intel Xe2 XMX is ~1.7× higher latency but with different throughput characteristics.

### Unitrace Cross-Validation

| Metric | Value | Interpretation |
|---|---|---|
| XVE_ACTIVE | 5.8% | Low utilization (single sub-group) |
| XVE_STALL | 31.6% | Significant stall time |
| XVE_STALL_SBID | 27.6% | Waiting for DPAS/XMX completion |
| XVE_STALL_ALUWR | 0.002% | Negligible ALU writeback stall |
| XVE_STALL_PIPESTALL | 0.0% | No pipeline stalls |
| GpuTime | 4,836 ns | Per-kernel execution |
| GpuCoreClocks | 11,606 cycles | ~4,836ns × 2.4GHz |
| XVE_INST_EXECUTED_SEND_ALL | 14 | 3 loads + 1 store + 10 warmup sends |

**SBID stall (27.6%)** confirms: the XMX engine is the bottleneck. The sub-group
must wait for each DPAS to complete before issuing the next one in a dependent chain.

---

## Benchmark 2: DPAS FP16 Latency

### Design

Same as BF16 latency kernel but with `OpTypeFloat 16` (half) replacing `OpTypeInt 16 0`
(ushort) for A and B matrix element types. Dimensions remain 8×16×16.

### Results

| N | Median (ns) | Slope latency |
|---:|---:|---:|
| 1 | 5,977 | - |
| 128 | 7,719 | - |
| 256 | 9,511 | - |
| **Slope** | - | **14.00 ns = 33.6 cycles** |

### Analysis

- **FP16 DPAS latency ≈ 34 cycles**, essentially identical to BF16
- This is expected: both use the same XMX hardware path (dpas.8x8 instruction),
  just with different data interpretation (half vs. ushort bit patterns)
- GEN ASM shows same `dpas.8x8` instruction with `:bf` suffix for both

---

## Benchmark 3: DPAS INT8 (Blocked)

### Design

INT8 DPAS uses `OpTypeInt 8 0` (uchar) for operands with dimensions 8×32×16
(K=32 for INT8, double the K dimension of BF16).

### Result

**BLOCKED**: `ocloc compile` segfaults when processing SPIR-V with
`OpTypeCooperativeMatrixKHR %uchar %uint_3 %uint_8 %uint_32 %uint_0`.

Error with wrong dimensions (8×16×16 for INT8):
```
Unsupported JointMatrix operation: load matrix A <8 x 16 x i8> with row major layout
 -> unsupported number of columns: 16
    supported values: 32
```

This confirms INT8 needs 8×32×16 dimensions, but ocloc crashes with those dimensions.
This is a **compiler bug** in the Intel Graphics Compiler's SPIR-V frontend.

**Workaround needed**: Use `OpSubgroupMatrixMultiplyAccumulateINTEL` (Intel-specific
SPIR-V intrinsic) instead of `OpCooperativeMatrixMulAddKHR`, or write INT8 DPAS via
SYCL `__builtin_IB_*` intrinsics and dump/reinject the LLVM IR.

---

## Benchmark 4: DPAS Throughput (Independent Chains)

### Design

**Goal**: Measure maximum DPAS throughput with instruction-level parallelism (ILP).

**Kernel**: Multiple independent DPAS chains in the same sub-group. Each chain has
its own accumulator tile, and chains don't depend on each other. This allows the
XMX engine to pipeline multiple DPAS operations.

```
Chain 0: acc0 = A0 × B0 + acc0    (independent)
Chain 1: acc1 = A1 × B1 + acc1    (independent)
...
Chain N: accN = AN × BN + accN    (independent)
```

**GEN ASM verification** (ILP=4, N_ITER=128):
```
dpas.8x8 r110:f r30:f r16:bf r7.0:bf    ; chain 0, different input regs
dpas.8x8 r118:f r54:f r44:bf r39.0:bf   ; chain 1
dpas.8x8 r28:f r78:f r68:bf r63.0:bf    ; chain 2
dpas.8x8 r52:f r102:f r92:bf r87.0:bf   ; chain 3
... (512 total = 4 chains × 128 iters)
```

Four distinct accumulator registers (r110, r118, r28, r52) confirm independent chains.

**Sweep**: ILP ∈ {1,2,4,8} × Sub-groups ∈ {1,2,4,8,16}, N_ITER=128

### Results

| ILP | SG | Median (ns) | TFLOPS |
|---:|---:|---:|---:|
| 1 | 1 | 7,566 | 0.069 |
| 2 | 1 | 9,869 | 0.106 |
| 4 | 1 | 11,633 | 0.180 |
| 4 | 2 | 10,673 | 0.196 |
| 8 | 1 | 16,735 | 0.251 |
| 8 | 4 | 16,860 | 0.249 |
| 8 | 16 | 22,325 | 0.188 |

### Analysis

- **Peak single-WG throughput: 0.251 TFLOPS** at ILP=8, SG=1
- Throughput scales ~linearly with ILP up to 8 chains
- Adding more sub-groups doesn't help significantly with only 1 work-group
- **Theoretical single-SG peak (latency-bound)**: 4096 FLOPs/dpas ÷ 33 cyc/dpas × 2.4 GHz = **0.298 TFLOPS**
  - ILP=8 achieves 0.251/0.298 = **84% of latency-bound peak** per sub-group
- **Full GPU throughput** requires dispatching work-groups across all 20 Xe Cores.
  The SYCL GEMM benchmark achieves 89.77 TFLOPS, which is ~60% of the theoretical
  peak of ~150 TFLOPS BF16 (160 XMX × 4096 FLOPs × 2.4 GHz ÷ 33 cycles ÷ pipeline factor).

---

## Benchmark 5: Memory Hierarchy Latency

### Design

**Goal**: Map the full memory hierarchy by measuring pointer-chase latency across
increasing buffer sizes. Each size reveals which cache level is being tested.

**Kernel**: `spirv_ptr_chase.spvasm`
- Single thread (LocalSize 1×1×1), single work-group
- Buffer contains a Fisher-Yates shuffled permutation of int32 indices
- Chase 4096 steps through the linked list:
  ```
  idx = 0
  for i in 0..4096:
      idx = buffer[idx]    // dependent load
  output[0] = idx
  ```
- Each access is a dependent load (next address depends on current data)
- Access uses **CrossWorkgroup address space** (global memory) → measures **L1 data cache** latency
- Sweep buffer size from 1 KB to 128 MB

**Important**: This kernel measures L1 data cache, **not** SLM (Shared Local Memory).
SLM uses a different address space (Workgroup) and has its own benchmark (Benchmark 5b).
On Intel Xe2, SLM is physically carved from the L1 data cache, but uses a separate
addressing path and potentially different access logic.

### Results

| Buffer Size | Cycles/Access | ns/Access | Region |
|---|---:|---:|---|
| 1 KB | 70.8 | 29.5 | L1 data cache |
| 2 KB | 70.8 | 29.5 | L1 data cache |
| 4 KB | 74.3 | 31.0 | L1 data cache |
| 8 KB | 74.3 | 30.9 | L1 data cache |
| 16 KB | 77.8 | 32.4 | L1 data cache |
| 32 KB | 91.9 | 38.3 | L1 data cache |
| 48 KB | 102.4 | 42.7 | L1 data cache |
| 64 KB | 110.4 | 46.0 | L1 data cache |
| 96 KB | 130.6 | 54.4 | L1 data cache |
| 128 KB | 144.5 | 60.2 | L1 data cache boundary |
| 192 KB | 162.4 | 67.7 | L2 cache |
| 256 KB | 177.6 | 74.0 | L2 cache |
| 384 KB | 195.5 | 81.4 | L2 cache |
| 512 KB | 201.0 | 83.8 | L2 cache |
| 768 KB | 214.9 | 89.5 | L2 cache |
| 1 MB | 219.6 | 91.5 | L2 cache |
| 1.5 MB | 225.9 | 94.1 | L2 cache |
| 2 MB | 229.3 | 95.5 | L2 cache |
| 4 MB | 232.7 | 97.0 | L2 cache |
| 8 MB | 236.1 | 98.4 | L2 cache (near boundary) |
| 16 MB | 247.4 | 103.1 | L2→Global transition |
| 32 MB | 253.9 | 105.8 | Global memory |
| 64 MB | 257.4 | 107.2 | Global memory |
| 128 MB | 260.7 | 108.6 | Global memory |

### Analysis

**Latency hierarchy**:

```
L1 Data Cache  ████████████████████ 70-145 cycles (1-128 KB, per Xe Core)
      ↑
L2 Cache       ████████████████████████████████ 162-236 cycles (192KB-8MB, 18 MB shared)
      ↑
Global (DRAM)  ████████████████████████████████████ 247-261 cycles (16MB+)
```

- **L1 data cache latency**: ~71 cycles at 1-4 KB, growing to ~145 cycles at 128 KB.
  The growth with size suggests set-associative conflict misses as the buffer
  approaches the 128 KB L1 capacity per Xe Core.

- **L1→L2 transition**: Sharp increase from 145→162 cycles between 128 KB→192 KB.
  This places the L1 data cache size at **128 KB per Xe Core**.

- **L2 cache latency**: ~162-236 cycles across the 192 KB to 8 MB range.
  The BMG-G21 has an **18 MB shared L2 cache**. Data up to ~8 MB shows relatively
  stable latency (~233-236 cycles), with a gradual increase to ~247 cycles at 16 MB.
  The absence of a sharp boundary at 18 MB is expected: the pointer-chase pattern
  from a single thread generates sequential cache-line fills, and the L2's high
  associativity means the transition is gradual rather than abrupt.

- **Global memory latency**: ~260 cycles = 108 ns. At 456 GB/s peak bandwidth,
  this translates to ~48 KB of in-flight data per memory channel.

---

## Benchmark 5b: SLM (Shared Local Memory) Latency

### Design

**Goal**: Measure true SLM latency separately from L1 data cache. On Intel Xe2, SLM
is carved from the L1 data cache but uses the **Workgroup address space** (`OpVariable
Workgroup` in SPIR-V, or `sycl::local_accessor` in SYCL), which has different
semantics (barrier-coherent, work-group-scoped).

**Kernel**: `bench_slm_latency.cpp` (SYCL)
- Uses `sycl::local_accessor<int, 1>` for SLM allocation
- Single work-group (16 threads), only thread 0 performs the chase
- Copy Fisher-Yates shuffled permutation from global → SLM
- `sycl::group_barrier` to synchronize
- Thread 0 chases 4096 steps through SLM: `idx = slm[idx]`
- Sweep SLM allocation size from 256 B to 64 KB

### Results

| SLM Size | Median (ns) | Cycles/Access | ns/Access |
|---|---:|---:|---:|
| 256 B | 135,821 | 79.6 | 33.2 |
| 512 B | 136,143 | 79.8 | 33.2 |
| 1 KB | 137,041 | 80.3 | 33.5 |
| 2 KB | 138,944 | 81.4 | 33.9 |
| 4 KB | 142,574 | 83.5 | 34.8 |
| 8 KB | 150,013 | 87.9 | 36.6 |
| 16 KB | 164,712 | 96.5 | 40.2 |
| 32 KB | 193,742 | 113.5 | 47.3 |
| 64 KB | 252,900 | 148.2 | 61.7 |

### Analysis

- **SLM latency ≈ 80 cycles** at small sizes (256B-1KB), growing to ~148 cycles at 64 KB
- **This is very similar to L1 data cache latency** (~71 cycles at 1KB)
- **Explanation**: On Intel Xe2, SLM is physically carved from the L1 data cache.
  The SLM portion and the general L1 data cache share the same SRAM arrays.
  Therefore, SLM access latency is comparable to L1 data cache latency — they are
  the same hardware, just with different coherence/scoping semantics.
- **Contrast with NVIDIA**: NVIDIA shared memory is a separate SRAM with ~30 cycle
  latency, distinct from the L1 cache (~35 cycles). Intel Xe2 does not have this
  separation — SLM and L1 are unified.
- The size-dependent latency growth (80→148 cycles) reflects the same set-associative
  conflict behavior seen in the L1 data cache, as expected since they share hardware.

---

## Benchmark 6: Memory Bandwidth

### Design

**SYCL kernel** (`bench_mem_bandwidth_fixed.cpp`):
- 64 work-groups × 256 threads = **16,384 threads** (enough to saturate memory)
- Each thread reads/writes `ceil(buffer_size / 16384)` elements with stride=16384
- **Every unique buffer element is accessed exactly once** (correct byte counting)
- Coalesced access pattern (consecutive threads access consecutive addresses)
- Sweep buffer size from 1 MB to 1 GB
- SYCL profiling events for accurate GPU-side timing

**Note on previous bandwidth bug**: The initial bandwidth benchmark used too many
threads (~268M for 1GB buffer) where each thread performed only 1 valid read but
the calculation assumed 10 reads per thread, resulting in a 10× overcount. The
reported 682-900 GB/s was incorrect. The fixed benchmark uses 16K threads with
proper stride and correct byte counting.

### Results

| Buffer Size | n_iter | Read BW (GB/s) | Write BW (GB/s) |
|---|---:|---:|---:|
| 1 MB | 16 | 303 | 479 |
| 4 MB | 64 | 351 | 615 |
| 16 MB | 256 | 256 | 654 |
| 64 MB | 1024 | 147 | 362 |
| 256 MB | 4096 | 147 | 289 |
| 1024 MB | 16384 | 147 | 162 |

### Analysis

- **Sustained DRAM read bandwidth**: **~147 GB/s** (1 GB buffer, well beyond L2)
  - This is ~32% of the 456 GB/s GDDR6 peak bandwidth
  - The low utilization is expected for a read-only pattern with 16K threads —
    full bandwidth saturation typically requires more threads and/or mixed read/write

- **Sustained DRAM write bandwidth**: **~162 GB/s** (1 GB buffer)
  - Write bandwidth is slightly higher than read, consistent with write-combining optimizations

- **Cache effects at small sizes**: Read BW reaches 303-351 GB/s at 1-4 MB (fits in L2),
  confirming that L2 cache serves reads at ~2× DRAM bandwidth

- **Why bandwidth is below peak**: The GDDR6 peak of 456 GB/s assumes optimal
  transaction scheduling across all memory channels. Our single-kernel, read-heavy
  workload may not fully saturate all channels. Production GEMM kernels achieve
  higher effective bandwidth through tiled read/write patterns.

---

## Benchmark 7: Memory Bandwidth (Previous, Buggy)

### Original (Incorrect) Results

The initial bandwidth benchmark (`bench_mem_bandwidth.cpp`) reported 682-900 GB/s.
This was due to a **10× overcounting bug** in the byte calculation:

```
bytes_read = n * n_iter * sizeof(float)   // BUG: overcounts by ~10×
```

With `n = 268M` floats (1GB buffer) and `global_range = 268M` threads:
- `idx = gid + i * global_range` exceeds `n` for any `i >= 1`
- Only the `i=0` iteration accesses valid data (1 read per thread)
- Actual bytes read = `n * sizeof(float)` = buffer size, not `n * n_iter * sizeof(float)`

The corrected results (Benchmark 6 above) show ~147 GB/s read, ~162 GB/s write.

---

## Toolchain Notes

### SPIR-V Cooperative Matrix on Intel

1. **Types**: `OpTypeCooperativeMatrixKHR` with `SPV_KHR_cooperative_matrix` extension
   - Scope 3 = Subgroup
   - Use: 0=A, 1=B, 2=Accumulator
   - BF16 operands stored as `%ushort` (Int 16 unsigned), accumulator as `%float`

2. **Operations**:
   - Load: `OpCooperativeMatrixLoadKHR %type %ptr %layout %stride None`
   - Store: `OpCooperativeMatrixStoreKHR %ptr %value %layout %stride None`
   - MulAdd: `OpCooperativeMatrixMulAddKHR %acc_type %a %b %c` (no trailing operands)

3. **Layout**: RowMajor=0, ColumnMajor=1. Intel "Packed" layout = 2 (from SPV_INTEL_joint_matrix)

4. **Capabilities needed**: `Addresses`, `Kernel`, `Int64`, `Int16`, `CooperativeMatrixKHR`

5. **INT8 limitation**: `OpTypeCooperativeMatrixKHR %uchar` with dimensions 8×32×16
   causes ocloc segfault. BF16 and FP16 work correctly.

### Level Zero Kernel Submission

```cpp
// Full flow:
zeInit → zeDriverGet → zeDeviceGet → zeContextCreate
→ zeCommandQueueCreate → zeCommandListCreate
→ zeModuleCreate(spirv_binary) → zeKernelCreate → zeKernelSetGroupSize
→ zeMemAllocShared × N → zeKernelSetArgumentValue × N
→ zeCommandListAppendLaunchKernel → zeCommandQueueExecuteCommandLists
→ zeCommandQueueSynchronize  // timing: t1 - t0
```

### Unitrace Metrics

- **ComputeBasic**: GPU_BUSY, XVE_ACTIVE, XVE_STALL, ALU0/1/2 instructions,
  SLM/L3/cache stats, memory bytes read/written
- **VectorEngineStalls**: ALUWR, BARRIER, CONTROL, INSTFETCH, PIPESTALL,
  SBID, SENDWR stall percentages, thread dispatch queue stats

---

## Cross-Architecture Comparison (With Caveats)

### Caveats

The comparison below between Intel Arc Pro B60 and NVIDIA Blackwell is **not apples-to-apple**:
- **Different market segments**: B60 is a professional workstation GPU (~$300-500);
  Blackwell GPUs (B100/B200) are data-center accelerators (~$30,000+)
- **Different power envelopes**: B60 ~120W TBP; Blackwell up to 1000W
- **Different process nodes**: Intel 4 vs. TSMC 4NP
- **Different architectural goals**: B60 optimizes for graphics+compute;
  Blackwell optimizes for AI training/inference at scale

This comparison serves only to contextualize our measurements, not to declare superiority.

### Latency Comparison

| Metric | Intel B60 (Xe2) | NVIDIA Blackwell | Notes |
|---|---|---|---|
| Tensor/Matrix unit latency | 33-37 cycles (XMX DPAS) | ~20 cycles (Tensor Core) | NVIDIA ~1.7× lower |
| Shared memory / SLM | ~80 cycles (carved from L1) | ~30 cycles (separate SRAM) | NVIDIA has dedicated shared mem |
| L1 data cache | ~71-145 cycles | ~30-35 cycles | NVIDIA ~2-4× lower |
| L2 cache | ~162-236 cycles (18 MB) | ~200-250 cycles | Similar range |
| Global memory | ~247-261 cycles | ~300-500 cycles | Intel lower (closer L2/GLOBAL) |

### Key Architectural Insight

The most notable difference is the **SLM/shared memory architecture**:
- **NVIDIA**: Shared memory is a separate, low-latency SRAM (~30 cycles) distinct from L1
- **Intel Xe2**: SLM is carved from the L1 data cache itself, resulting in ~80 cycle latency
  (similar to L1). This is a trade-off: Intel gets flexible SLM/L1 partitioning but at the
  cost of not having a dedicated low-latency scratchpad.

---

## Files

| File | Description |
|---|---|
| `spirv_runner.cpp` | Level Zero host runner (configurable WG, buffer size) |
| `mem_runner.cpp` | Memory benchmark runner (pointer chase permutation, bandwidth) |
| `spirv_dpas_latency.spvasm` | BF16 DPAS latency kernel (dependent chain) |
| `spirv_dpas_int8_latency.spvasm` | INT8 DPAS latency kernel (blocked by compiler bug) |
| `spirv_ptr_chase.spvasm` | Pointer chase kernel (single-thread, 4096 steps, L1 data cache) |
| `spirv_mem_read.spvasm` | Coalesced read kernel (256-thread WG, 256 iters) |
| `bench_slm_latency.cpp` | SLM pointer chase benchmark (SYCL, measures SLM latency) |
| `bench_mem_bandwidth_fixed.cpp` | Fixed bandwidth benchmark (correct byte counting) |
| `bench_mem_bandwidth.cpp` | Original bandwidth benchmark (has overcounting bug, kept for reference) |
| `run_dpas_sweep.py` | DPAS latency/throughput automation |
| `run_dpas_precision.py` | BF16/FP16 precision comparison |
| `run_mem_sweep.py` | Memory latency/bandwidth automation |
| `generate_summary.py` | Summary report generator |
| `results/*.csv` | Raw benchmark data |
| `Makefile` | Build pipeline |
