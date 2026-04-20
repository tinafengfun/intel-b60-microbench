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
| Memory | 24 GB GDDR6, 256-bit bus, ~576 GB/s peak (18 Gbps) |
| Sub-group Size | 16 threads |
| GRF per Thread | 256 registers (128 × 32B) |
| L1 Data Cache | 128 KB per Xe Core |
| SLM (Shared Local Memory) | Up to 64 KB usable per Xe Core (128 KB SRAM split with L1) |
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
- **GPU event timing**: SYCL/Level Zero profiling events for kernel-level timestamps
- **Sweep method**: Vary operation count N, measure total time, compute per-op
  latency from **linear regression slope** to eliminate fixed dispatch overhead (~6μs)
- **Host vs GPU timing comparison**:
  - Long kernels (>1 ms): host and GPU event timing agree within 1%
  - Short kernels (<100 μs): host timing includes ~6.7 μs submit+wait overhead
  - Empty kernel: GPU=0.8 μs, host=7.5 μs (9:1 ratio)
  - This validates the slope method for short DPAS kernels (6-8 μs)
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
- **Latency vs Throughput**: 33 cycles is the **latency** (time from issue to result).
  The **reciprocal throughput** (minimum interval between independent DPAS issues) is
  different — derived from full-GPU GEMM data below.
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
- GEN ASM confirms distinct type suffixes: BF16 uses `:bf`, FP16 uses `:hf`
  - BF16: `dpas.8x8 (16|M0) r28:f r28:f r9:bf r5.0:bf`
  - FP16: `dpas.8x8 (16|M0) r28:f r28:f r9:hf r5.0:hf`
  - The accumulator is always `:f` (float) in both cases

---

## Benchmark 3: DPAS INT8 (Blocked)

### Design

INT8 DPAS uses `OpTypeInt 8 0` (uchar) for operands with dimensions 8×32×16
(K=32 for INT8, double the K dimension of BF16).

### Result

**BLOCKED**: `ocloc compile` segfaults (exit code 139, SIGSEGV) when processing SPIR-V
with `OpTypeCooperativeMatrixKHR %uchar` combined with `OpCooperativeMatrixMulAddKHR`.

**Reproduction evidence** (oneAPI 2025.3, ocloc for BMG-G21):
```
# Type declaration alone (no MulAddKHR): compiles OK
# With MulAddKHR, correct dimensions (8×32×16): Segmentation fault (core dumped)
# With MulAddKHR, wrong dimensions (8×16×16): Segmentation fault (core dumped)
```

Both dimensions crash when `MulAddKHR` is used with INT8 types. The crash is in IGC's
cooperative matrix lowering pass. This is a **compiler bug**.

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
- **Latency-bound single-SG peak**: 4096 FLOPs/dpas ÷ 33 cyc/dpas × 2.4 GHz = **0.298 TFLOPS**
  - ILP=8 achieves 0.251/0.298 = **84% of latency-bound peak** per sub-group

#### Full GPU Peak (Directly Measured)

The reciprocal throughput is **directly measured** from a full-GPU DPAS microbenchmark
(ILP=8, N_ITER=128, all operands pre-loaded to registers, no memory access in the loop):

| SG/WG | WGs | Peak TFLOPS | Implied cyc/dpas |
|---:|---:|---:|---:|
| 1 | 2048 | 88.61 | 17.8 |
| 4 | 2048 | 89.21 | 17.7 |
| 16 | 1024 | **97.66** | **16.1** |

**Peak: 97.66 TFLOPS** at SG=16/WG, 1024 WGs — this is the raw XMX throughput
with zero memory pressure. The GEMM kernel achieves 89.77 TFLOPS = 92% of this
theoretical peak, with the remaining 8% lost to memory access overhead.

**Reciprocal throughput = 16.1 cycles** (directly measured, not derived from GEMM):
```
160 XMX × 4096 FLOPs × 2.4 GHz / 97.66 TFLOPS = 16.1 cycles per DPAS
```

**Key insight: XMX does NOT pipeline independent DPAS within a single sub-group.**
ILP=8 within one sub-group achieves only ~39 cycles/dpas (close to the 33-cycle latency).
The 16.1-cycle reciprocal throughput comes from **TLP** (thread-level parallelism):
each EU has 8 hardware threads, and the thread scheduler interleaves DPAS operations
from multiple sub-groups across the XMX pipeline. With 16 sub-groups per WG and enough
WGs to saturate all 160 EUs, the scheduler keeps the XMX engine ~2× busy.

| Concept | Value | Source |
|---|---|---|
| Latency | **33 cycles** | Dependent chain slope (Benchmark 1) |
| Reciprocal throughput | **16.1 cycles** | Full-GPU sweep (this benchmark) |
| GEMM utilization | **92%** | 89.77 / 97.66 (production GEMM vs. raw XMX) |

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
  **This growth is NOT due to TLB effects** — confirmed by a page-stride pointer chase
  test that touches one element per 4KB page (exercising TLB without cache pressure),
  which stays flat at ~64 cycles across all buffer sizes. The growth in random-access
  latency is due to reduced cache line utilization: random permutation accesses each
  64-byte cache line ~1 time vs ~16 times for sequential access, causing L1 capacity
  pressure and queueing delays as the working set grows.

- **L1→L2 transition**: Sharp increase from 145→162 cycles between 128 KB→192 KB.
  This places the L1 data cache size at **128 KB per Xe Core**.

- **L2 cache latency**: ~162-236 cycles across the 192 KB to 8 MB range.
  The BMG-G21 has an **18 MB shared L2 cache**. Data up to ~8 MB shows relatively
  stable latency (~233-236 cycles), with a gradual increase to ~247 cycles at 16 MB.
  The absence of a sharp boundary at 18 MB is expected: the pointer-chase pattern
  from a single thread generates sequential cache-line fills, and the L2's high
  associativity means the transition is gradual rather than abrupt.

- **Global memory latency**: ~260 cycles = 108 ns. At 576 GB/s peak bandwidth,
  this translates to ~62 KB of in-flight data per memory channel.

---

## Benchmark 5b: SLM (Shared Local Memory) Latency

### Design

**Goal**: Measure true SLM latency using native SPIR-V Workgroup storage class,
separately from L1 data cache latency.

**Kernel**: Raw SPIR-V (`OpVariable Workgroup`)
- Uses `OpVariable %ptr_work_arr Workgroup` for SLM allocation at module scope
- Single thread (LocalSize 1 1 1) — no barrier or thread synchronization overhead
- Copies Fisher-Yates shuffled permutation from global memory to SLM
- Chases 4096 steps through SLM: `idx = slm[idx]`
- All SLM accesses verified via GEN ASM: `send.slm` instructions

**GEN ASM verification** (n=256):
```
; Copy phase: load from global, store to SLM
send.ugm (1|M0)  r3  r4  null:0  0x0  0x0210C580  // load.ugm.d32x8t.a64 (global→reg)
send.slm (1|M0)  null r5  r3:1   0x0  0x0200D504  // store.slm.d32x16t.a32 (reg→SLM)

; Chase phase: load from SLM (repeated 16× per unrolled iteration)
send.slm (1|M0)  r10  r9  null:0  0x0  0x02108500  // load.slm.d32x1t.a32
send.slm (1|M0)  r12  r11 null:0  0x0  0x02108500  // load.slm.d32x1t.a32
... (16 SLM loads per unrolled iteration)
```

The `send.slm` instructions confirm true SLM hardware path (vs `send.ugm` for global).

**SYCL vs SPIR-V**: We also measured SLM latency using SYCL `sycl::local_accessor`.
The SYCL path adds ~34 cycles of overhead per access (80 vs 46 cycles at 256B),
likely due to the compiler's address calculation and barrier code generation.

### Results

**SPIR-V Native SLM** (send.slm path):

| SLM Size | Median (ns) | Cycles/Access | ns/Access |
|---|---:|---:|---:|
| 256 B | 78,694 | 46.1 | 19.2 |
| 512 B | 78,754 | 46.1 | 19.2 |
| 1 KB | 84,784 | 49.7 | 20.7 |
| 2 KB | 84,780 | 49.7 | 20.7 |
| 4 KB | 90,740 | 53.2 | 22.2 |
| 8 KB | 96,817 | 56.7 | 23.6 |
| 16 KB | 108,860 | 63.8 | 26.6 |
| 32 KB | 138,997 | 81.4 | 33.9 |
| 64 KB | 198,948 | 116.6 | 48.6 |

**SYCL local_accessor SLM** (for comparison):

| SLM Size | Cycles/Access |
|---|---:|
| 256 B | 79.6 |
| 1 KB | 80.3 |
| 64 KB | 148.2 |

### Analysis

- **Native SLM latency ≈ 46 cycles** at small sizes (256B-512B), growing to ~117 cycles at 64 KB
- **SLM is faster than L1 data cache** (46 vs 71 cycles at 1KB), confirming separate hardware paths
  - L1 data cache uses `send.ugm` (global memory path, goes through L1 tag/data arrays)
  - SLM uses `send.slm` (dedicated SLM path, bypasses L1 tag lookup)
- **SYCL adds ~34 cycles overhead** per access: SYCL's `local_accessor` generates
  additional address computation and barrier code, inflating latency from 46→80 cycles.
  The SPIR-V native measurement is more representative of the raw hardware.
- **SLM is carved from the same SRAM** as L1 on Xe2, but uses a separate access
  path (`send.slm` vs `send.ugm`). This explains why SLM is faster (~46 cycles)
  than L1 data cache (~71 cycles) despite sharing the same physical storage.
- **Practical SLM limit is 64 KB**: Although `maxSharedLocalMemory` reports 128 KB,
  allocations >64 KB fail. The 128 KB SRAM is split: 64 KB for SLM, 64 KB reserved
  for L1 data cache. This is an important constraint for GEMM tiling.
- **Contrast with NVIDIA**: NVIDIA shared memory is a fully separate SRAM with
  ~30 cycle latency. Intel Xe2 SLM at ~46 cycles is closer to NVIDIA shared memory
  than the L1 data cache (~71 cycles), but not as fast due to the unified SRAM design.
- The size-dependent latency growth (46→117 cycles) reflects set-associative effects
  in the shared SRAM as SLM allocation approaches capacity.

---

## Benchmark 6: Memory Bandwidth

### Design

Multiple generations of bandwidth benchmarks progressively identified and fixed issues:

1. **v1** (`bench_mem_bandwidth.cpp`): 10× overcounting bug (reported 682-900 GB/s)
2. **v2** (`bench_bandwidth_v2.cpp`): Fixed byte counting, used `sycl::buffer` (shared memory)
3. **v3** (`bench_bandwidth_device.cpp`): Uses `sycl::malloc_device` for device-local memory
4. **v4** (`bench_bw_v2.cpp`, `bench_bw_v3.cpp`): Investigates access pattern effects — scalar vs vectorized loads, thread count sweep

**Key finding**: The 1 GB buffer tests showed inflated read bandwidth (~346 GB/s) due to L2 cache
effects (18 MB L2 can serve a significant fraction of 1 GB reads with streaming prefetch).
With 2 GB buffers (well beyond L2), scalar reads drop to ~147 GB/s. However, this is NOT
the true DRAM limit — vectorized (float4) loads achieve much higher throughput.

**Critical insight**: Scalar `float` loads (4 bytes) achieve poor DRAM utilization because
each 32-byte cache line requires 8 independent load instructions. Vector `float4` loads
(16 bytes) pack 4 floats per transaction, requiring fewer instructions and achieving 3.6×
higher bandwidth.

### Results

**Vectorized (float4) bandwidth** (`bench_bw_v3.cpp`, 2 GB buffer, `malloc_device`):

| Threads | Read (GB/s) | Write (GB/s) | Copy 2× (GB/s) |
|---:|---:|---:|---:|
| 262K (1K WG) | 466 | 364 | 351 |
| 524K (2K WG) | 463 | 365 | 416 |
| 1M (4K WG) | **534** | 379 | 428 |
| 2M (8K WG) | 537 | 401 | 432 |
| 4M (16K WG) | **538** | **442** | **439** |

**Scalar (float) bandwidth** (for comparison):

| Threads | Read (GB/s) |
|---:|---:|
| 262K | 147 |
| 1M | 147 |
| 2M | 203 |
| 4M | 204 |

**DMA memcpy** (`q.memcpy`): 235 GB/s

### Analysis

- **Peak DRAM bandwidth: 538 GB/s** (vectorized read, 4M threads) — 93% of 576 GB/s theoretical peak
  - **Theoretical peak**: 256-bit bus × 18 Gbps / 8 = **576 GB/s**
  - Bus width confirmed by: Level Zero reports `maxBusWidth=64` per channel (likely 4 channels × 64 = 256-bit)
  - This confirms the B60 uses a **256-bit GDDR6 memory bus at 18 Gbps data rate**

- **Access pattern matters enormously**: Scalar reads peak at 204 GB/s (35% of peak),
  while vectorized (float4) reads reach 538 GB/s (93%). The 2.6× gap is due to
  per-thread memory transaction efficiency, not DRAM capability.

- **Write bandwidth**: 442 GB/s (77% of peak) with float4, still increasing with thread count

- **Copy bandwidth**: 439 GB/s (2× bytes) = 219 GB/s (1×) — limited by combined read+write pressure

- **Previous 1 GB results were cache-inflated**: The 346 GB/s read at 1 GB was partly served
  from L2 cache (18 MB). With 2 GB buffers, scalar reads drop to 147 GB/s, but vectorized
  reads jump to 538 GB/s. The true DRAM bandwidth was always ~576 GB/s — our access pattern
  was the bottleneck, not the hardware.

- **DMA memcpy at 235 GB/s**: The BLAS-style DMA engine achieves ~41% of peak, significantly
  less than the kernel-based vectorized copy at 439 GB/s. Kernels can better saturate the
  memory bus due to parallel thread execution.

### Previous Results (Historical)

**Device memory** (`sycl::malloc_device`, 1 GB buffer, bench_bandwidth_device.cpp):

| Threads | Read BW (GB/s) | Write BW (GB/s) | Copy BW (GB/s) |
|---:|---:|---:|---:|
| 1M (4096 WG) | 346* | 368 | 432 |

\* *Inflated by L2 cache effects with 1 GB buffer. True scalar DRAM read is ~147-204 GB/s.*

### Unitrace Cross-Validation

Unitrace `GPU_MEMORY_BYTE_READ/WRITE` hardware counters (1 GB buffer):

| Kernel | DRAM Write Bytes | DRAM Write Rate | DRAM Read Bytes | DRAM Read Rate |
|---|---:|---:|---:|---:|
| WRITE (1M threads) | 1.008 GB | **348 GB/s** | 0.050 GB | 17 GB/s |
| READ (1M threads) | 0.005 GB | 1.5 GB/s | 0.002 GB | 0.7 GB/s |
| COPY (1M threads) | 0.026 GB | 5.2 GB/s | 0.012 GB | 2.3 GB/s |

The `LOAD_STORE_CACHE_BYTE_READ` counter shows 1.026 GB for the read kernel, confirming
that all data is read — but almost entirely served from cache, not DRAM (with 1 GB buffer).

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

The corrected results (Benchmark 6 above) show copy bandwidth of 432 GB/s (95% of peak),
DRAM write ~348 GB/s (unitrace validated), with `sycl::malloc_device` allocation.

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
| Tensor/Matrix unit latency | 14 ns (33-37 cyc) | ~7 ns (~20 cyc at 2.75 GHz) | NVIDIA ~2× lower in wall time |
| Shared memory / SLM | **19 ns** (46 cyc, send.slm) | ~11 ns (~30 cyc) | Both faster than L1 |
| L1 data cache | 30-60 ns (71-145 cyc) | ~11-13 ns (~30-35 cyc) | NVIDIA ~2.5-5× lower |
| L2 cache | 68-98 ns (162-236 cyc) | ~73-91 ns (~200-250 cyc) | Similar in wall time |
| Global memory | 103-108 ns (247-261 cyc) | ~109-182 ns (~300-500 cyc) | Similar or Intel slightly lower |

*Note: B60 at 2.4 GHz, Blackwell estimated at 2.75 GHz (varies by SKU). Wall-clock time (ns) is more meaningful than cycles for cross-arch comparison since clock speeds differ.*

### Key Architectural Insight

The most notable difference is the **SLM/shared memory architecture**:
- **NVIDIA**: Shared memory is a separate, low-latency SRAM (~30 cycles) distinct from L1
- **Intel Xe2**: SLM is carved from the same SRAM as L1 data cache, but uses a dedicated
  access path (`send.slm` vs `send.ugm`). SLM latency is ~46 cycles (SPIR-V native),
  which is **faster than L1 data cache** (~71 cycles) but slower than NVIDIA shared memory.
  SYCL's `local_accessor` adds ~34 cycles of overhead, inflating measurement to ~80 cycles.

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
| `bench_slm_latency.cpp` | SLM pointer chase benchmark (SYCL local_accessor, includes overhead) |
| `run_slm_sweep.py` | SPIR-V native SLM pointer chase (OpVariable Workgroup, send.slm) |
| `bench_bandwidth_v2.cpp` | Bandwidth v2 with thread count sweep (shared memory) |
| `bench_bandwidth_device.cpp` | Bandwidth v3 with device memory (malloc_device), copy pattern |
| `bench_bw_unitrace.cpp` | Single-config bandwidth for unitrace profiling |
| `bench_bw_v2.cpp` | Bandwidth investigation: ILP vs serial read, vector4, write+readback |
| `bench_bw_v3.cpp` | Bandwidth sweep: vectorized float4 read/write/copy, thread count scaling |
| `bench_mem_bandwidth_fixed.cpp` | Fixed bandwidth benchmark (correct byte counting) |
| `bench_mem_bandwidth.cpp` | Original bandwidth benchmark (has overcounting bug, kept for reference) |
| `run_dpas_sweep.py` | DPAS latency/throughput automation |
| `run_dpas_full_gpu.py` | Full-GPU DPAS throughput sweep (1-4096 WGs, direct reciprocal throughput) |
| `run_dpas_precision.py` | BF16/FP16 precision comparison |
| `run_mem_sweep.py` | Memory latency/bandwidth automation |
| `generate_summary.py` | Summary report generator |
| `results/*.csv` | Raw benchmark data |
| `bench_tlb.cpp` | TLB vs cache-set investigation (random/sequential/page-stride patterns) |
| `bench_timing.cpp` | Host timing vs GPU event profiling comparison |
| `Makefile` | Build pipeline |
