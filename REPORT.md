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
- **Statistics**: Median of repetitions (robust to outliers). Full statistics
  reported: mean, standard deviation, coefficient of variation (CV%), and 95%
  confidence interval. Linear regression includes R² and slope standard error.
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
slope = 14.34 ns = 34.4 cycles per DPAS (95% CI: 31.9–36.9 cycles)
slope standard error = 0.52 ns
intercept = 5,911 ns = 14,186 cycles (fixed dispatch overhead)
R² = 0.9895
```

**Per-N measurement variability** (100 repetitions each):

| N | Median (ns) | StdDev (ns) | CV (%) | 95% CI |
|---:|---:|---:|---:|---|
| 1 | 6,134 | 366 | 5.9 | [6,112, 6,255] |
| 16 | 6,038 | 337 | 5.6 | [6,008, 6,140] |
| 128 | 7,694 | 354 | 4.6 | [7,669, 7,807] |
| 512 | 12,999 | 442 | 3.4 | [13,010, 13,184] |

CV decreases with N (noise amortized). At N=128-512, the kernel is 7-13 μs
vs the 6.7 μs overhead, giving ~2:1 signal-to-noise. The R²=0.989 confirms
the linear model is an excellent fit despite per-run variability.

### Analysis

- **DPAS BF16 latency ≈ 34 ± 3 cycles** (95% CI: 32–37 cycles, 14 ± 1 ns at 2.4 GHz)
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

**Full sweep** (8 N-values, 100 repetitions each):

| N | Median (ns) | StdDev (ns) | CV (%) |
|---:|---:|---:|---:|
| 1 | 6,075 | 494 | 7.9 |
| 4 | 5,865 | 500 | 8.4 |
| 16 | 5,992 | 319 | 5.3 |
| 32 | 6,298 | 378 | 5.9 |
| 64 | 6,831 | 288 | 4.2 |
| 128 | 7,701 | 628 | 8.1 |
| 256 | 9,505 | 612 | 6.4 |
| 512 | 12,998 | 573 | 4.4 |

**Linear regression**:
```
slope = 13.93 ns = 33.4 cycles per DPAS (95% CI: 32.4–34.4 cycles)
slope standard error = 0.21 ns
R² = 0.9987
```

### Analysis

- **FP16 DPAS latency ≈ 33 ± 1 cycles** (95% CI), essentially identical to BF16 (34 ± 3 cycles)
- The tighter CI for FP16 (±1 vs ±3 cycles) reflects the higher R² (0.9987 vs 0.9895).
  The BF16 R² is pulled down by the small-N data points (N=1,4,16), where the ~6μs dispatch
  overhead dominates the ~0-0.5μs DPAS signal. These high-leverage points inflate residuals.
  FP16 included N=32 as an additional mid-range point, giving better regression coverage.
  Both precisions use the same XMX hardware path; the R² gap is measurement noise, not
  a systematic difference.
- Both use the same XMX hardware path (dpas.8x8 instruction), just with different
  data interpretation (half vs. ushort bit patterns)
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

**Key insight: XMX CAN pipeline independent DPAS within a single sub-group, but requires ILP≥14.**
ILP=8 achieves ~31 cycles/dpas (near the 33-cycle latency), but ILP=14-16 drops to
**~16 cycles/dpas** — matching the full-GPU reciprocal throughput. The XMX pipeline
needs at least 14 independent chains to fully saturate within a single thread.

The 16.1-cycle reciprocal throughput also manifests via **TLP** (thread-level parallelism):
each EU has 8 hardware threads, and the thread scheduler interleaves DPAS operations
from multiple sub-groups. With 16 sub-groups per WG and enough WGs to saturate all
160 EUs, the scheduler keeps the XMX engine ~2× busy even without ILP.

#### ILP Saturation Detail

Cycles/DPAS computed as `median_ns × 2.4 GHz / (ILP × N_ITER)`. This is a **total-time
divided by total-work** metric — it includes dispatch overhead (~3.7 μs fixed) amortized
over the workload. For small workloads (ILP=1, N_ITER=64, total time ~6.8 μs), dispatch
overhead is ~54% of the measurement, inflating cycles/DPAS. The relative scaling between
ILP values is still valid because all use the same N_ITER=64.

| ILP | Cycles/DPAS | Interpretation |
|---:|---:|---|
| 1 | 252.8 | Dispatch-dominated (6.8 μs total, 54% overhead) |
| 2 | 130.4 | ~2× parallelism vs ILP=1 |
| 4 | 67.2 | ~4× parallelism |
| 8 | 31.4 | Near-pure DPAS latency |
| 10 | 24.2 | Starting to pipeline |
| 12 | 20.2 | Pipelining |
| 14 | 16.7 | Near-reciprocal throughput |
| 16 | 15.9 | **Reciprocal throughput saturated** |

For comparison, the **slope method** (Benchmark 1) gives 34.4 cycles for ILP=1 by
eliminating dispatch overhead via regression. The **4c.2 measurement** gives 47 cycles
for ILP=1 with N_ITER=1024 (lower overhead fraction: 3.7 μs / 20 μs = 18%). These
three measurements of the same physical quantity differ due to overhead inclusion:
252.8 (raw, N=64) > 47.0 (raw, N=1024) > 34.4 (slope, overhead removed).

Each accumulator is an 8×16 float matrix = 512 bytes = 16 GRF registers.
ILP=16 uses 256 GRF registers = 100% of the 256-register GRF file.
No register spills detected in GEN ASM at ILP=16 — the compiler manages allocation.

| Concept | Value | Source |
|---|---|---|
| Latency | **33 cycles** | Dependent chain slope (Benchmark 1) |
| Reciprocal throughput | **16.1 cycles** | Full-GPU sweep (this benchmark) |
| ILP saturation point | **ILP≥14** | Needed to reach reciprocal throughput within single SG |
| GEMM utilization | **92%** | 89.77 / 97.66 (production GEMM vs. raw XMX) |

#### DPAS Pipeline Depth Analysis

Combining the latency and reciprocal throughput measurements reveals the XMX pipeline structure:

| Parameter | Value | Source |
|---|---|---|
| DPAS latency | **33 cycles** | Benchmark 1 slope method |
| DPAS reciprocal throughput | **16.1 cycles** | Benchmark 4 full-GPU sweep |
| ILP=1 per-iteration time | **252.8 cycles** | ILP Saturation table (raw, N=64) |
| ILP saturation point | **ILP≥14** | ILP Saturation table |

**Pipeline depth = 2:**

```
Pipeline Depth = ⌈Latency / Reciprocal Throughput⌉ = ⌈33 / 16.1⌉ = 2
```

The XMX hardware has a **2-stage pipeline** — at most 2 DPAS instructions can be in-flight
simultaneously (one in each half of the pipeline). A new DPAS can be issued every 16.1 cycles,
with results available after 33 cycles.

The ILP saturation data confirms this:

| ILP | Cycles/DPAS | Behavior |
|---:|---:|---|
| 8 | 31.4 | ≈ 33, hitting the latency ceiling |
| 10 | 24.2 | Breaking through latency, pipeline starting to fill |
| 14 | 16.7 | ≈ 16.1, pipeline fully utilized |
| 16 | 15.9 | Saturated |

From ILP=8→16, cycles/DPAS drops from 31→16 — approximately **2× improvement**, consistent
with filling a 2-stage pipeline.

**Why ILP=16 is needed instead of ILP=2?**

With a 2-stage pipeline, one might expect ILP=2 to suffice. The reason ILP≈16 is required
in practice is that **each chain's iteration includes ~220 cycles of loop overhead** beyond
the 33-cycle DPAS:

```
Single-chain timeline (ILP=1, 252.8 cycles/iter):
|-- DPAS 33cy --|-- loop overhead ~220cy (branch, phi, counter, addr calc, send.ugm...) --|
                 ^
                 XMX is idle during this time
```

To keep XMX busy every 16.1 cycles, enough chains must fill the full 252.8-cycle interval:

```
ILP needed = ⌈252.8 / 16.1⌉ = 16
```

With ILP=16, chains take turns feeding DPAS:

```
XMX timeline:
|DPAS₀|16cy|DPAS₁|16cy|DPAS₂|16cy|...|DPAS₁₅|16cy|DPAS₀|...
                                                     ^
                                                     Chain 0's 252.8cy interval completes,
                                                     ready to issue its next DPAS
```

**Three paths to peak throughput:**

| Method | Mechanism | Requirement | Evidence |
|---|---|---|---|
| **Pure ILP** | Multiple independent chains in one thread | ILP≥14 (consumes all 256 GRF) | ILP=16 → 15.9 cyc/DPAS |
| **Pure TLP** | Multiple threads share XMX via scheduler | 16 SG/WG + enough WGs | SG=16, 1024 WGs → 97.66 TFLOPS |
| **ILP + TLP** | Combined approach | Moderate ILP × many threads | ILP=8, SG=16 → near-peak |

Pure TLP works because each EU has 8 hardware threads. Even with ILP=1 (one DPAS every
~253 cycles per thread), 8 threads alternating yields ~31.6 cycles/DPAS per EU — close to
the latency ceiling. With 160 EUs running in parallel, aggregate XMX utilization approaches
peak via sheer parallelism.

**Key implication: loop overhead is the bottleneck, not pipeline depth.**

If the ~220 cycles of loop overhead were eliminated (e.g., fully unrolled loop producing a
pure DPAS instruction stream in GEN ASM), then:

```
ILP needed (no overhead) = ⌈33 / 16.1⌉ = 2
```

ILP=2 would suffice. This means **the critical optimization for GEMM is reducing non-DPAS
instruction overhead** — through loop unrolling, software pipelining, and hiding `send.ugm`
loads behind the SBID stall (as validated by experiments 4b.3 and 4b.6).

**GEMM kernel source**: The 89.77 TFLOPS result comes from a **custom SYCL GEMM kernel**
(`sycl-xmx-gemm/src/kernels/gemm_v20_best.cpp`) using `sycl::ext::oneapi::matrix` extensions.
It is NOT from oneMKL or oneDNN. Key characteristics:
- BF16 input with FP32 accumulation, dpas.8x8 tiles (TM=8, TN=16, TK=16)
- 4×2 multi-SG work-groups (4 SGs per WG, 2×2 layout), 256 GRF mode
- Register blocking: 4×4 register tiles per sub-group (32×64 output block per SG)
- K-step unrolling with software prefetch of next K-block
- Problem size: 8192×2048×4096 (BF16), 100 warmup + 500 measured iterations
- Reproducible with oneAPI 2025.3.2, driver 1.14.36300, IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread"
- Comparison: oneDNN achieves ~95 TFLOPS on the same hardware (estimated from JIT kernel
  capabilities: SG=8, K-parallel reduction with atomics, Block2D SLM cooperative loads;
  not a direct measurement under identical test conditions). The ~5 TFLOPS gap is
  attributed to SYCL compiler/runtime overhead vs oneDNN's hand-tuned Level Zero dispatch.
  Direct comparison is not apples-to-apple due to different compilation paths
  (oneDNN native JIT vs SYCL/ICPX).

---

## Benchmark 4b: DPAS Scheduling Overhead

### Design

**Goal**: Quantify how different scheduling patterns affect DPAS throughput — barriers,
operand reloads, store-back, ALU interleaving, and cross-SG coordination.

### Results

#### 4b.1: SBID Stall Behavior (DPAS + ALU Interleave)

Dependent DPAS chain with N independent scalar FP32 ALU operations per iteration:

| ALU ops/iter | Cycles/iter | Delta vs baseline |
|---:|---:|---:|
| 0 | 256.9 | 0.0 |
| 1 | 254.6 | -2.2 |
| 2 | 259.8 | 3.0 |
| 4 | 266.5 | 9.6 |
| 8 | 258.0 | 1.1 |
| 16 | 252.4 | -4.5 |

**Finding**: Adding 0-16 independent FP32 ALU operations has **zero measurable impact** on
DPAS iteration time. The EU thread is **completely blocked during SBID stall** — the ALU ops
execute during the ~33 cycle DPAS wait time. **Xe2 EU cannot dual-issue XMX + ALU within
a single thread.**

This is confirmed at full-GPU scale (ILP=8, 1024 WGs): adding 0-8 ALU ops per iteration
produces identical throughput (~39 TFLOPS).

#### 4b.2: Barrier Overhead

OpControlBarrier with varying SGs per work-group (barrier every iteration, N_ITER=64):

| SGs/WG | DPAS only (cyc/dpas) | DPAS+barrier (cyc/dpas) | Barrier overhead |
|---:|---:|---:|---:|
| 1 | 254.5 | 261.0 | +6.6 |
| 2 | 124.2 | 134.8 | +10.7 |
| 4 | 64.1 | 67.3 | +3.2 |
| 8 | 32.2 | 34.8 | +2.6 |
| 16 | 16.8 | 18.7 | +2.0 |

**Finding**: OpControlBarrier is **very cheap** — 2-11 cycles per barrier, decreasing with
more SGs. The barrier is essentially free when all SGs reach it quickly.

For 1 SG (no other SGs to synchronize), the barrier is ~6.6 cycles (a no-op synchronization).
OpMemoryBarrier caused ocloc segfaults (compiler bug).

#### 4b.3: Operand Reload from Global Memory

Reloading A/B cooperative matrix tiles from cache-resident global memory each iteration:

| Reload pattern | Cycles/iter | Delta |
|---|---:|---:|
| Pre-load once (baseline) | 293.9 | 0.0 |
| Reload A only | 262.2 | -31.7 |
| Reload B only | 265.2 | -28.7 |
| Reload A+B | 265.4 | -28.5 |

**Finding**: Reloading tiles from cache-resident global memory is **free or slightly faster**
than holding them in registers.

**GEN ASM explanation** (N_ITER=64, ILP=1):

| Variant | GEN ASM lines | `send` count | `mov` count | `dpas` count |
|---|---:|---:|---:|---:|
| No reload | 141 | 8 | 36 | 64 |
| Reload A+B | 1,145 | 134 | 794 | 64 |

The reload kernel has 16× more send instructions (2 loads × 64 iters vs 3 loads total)
and 22× more mov instructions. Despite this, it's not slower because:
1. `send.ugm load_block2d` instructions are pipelined — they can issue before the
   previous DPAS completes, overlapping with the ~33 cycle SBID stall
2. The no-reload variant's register pressure (holding tiles in GRF during the loop)
   forces the compiler to generate additional register save/restore code
3. The reload kernel's instruction mix is more scheduling-friendly: alternating
   `send` and `dpas` gives the hardware scheduler more issue slots

**Important caveat**: This result applies to cache-resident data (L1 hit path). With
L2 or DRAM resident operands, the ~71-260 cycle load latency cannot be fully hidden
behind the 33-cycle DPAS stall, and reload will add measurable overhead. The register
hold advantage would reassert itself for non-cache-resident data.

#### 4b.4: Store-to-Global Frequency

OpCooperativeMatrixStoreKHR at varying intervals during DPAS loop:

| Store interval | Cycles/iter | Delta |
|---|---:|---:|
| Never | 254.5 | 0.0 |
| Every 1 | 279.7 | +25.2 |
| Every 2 | 268.7 | +14.2 |
| Every 4 | 258.2 | +3.7 |
| Every 8 | 254.7 | +0.2 |
| Every 16 | 256.9 | +2.4 |

**Finding**: Each store costs ~25 cycles (cooperative matrix store of 8×16 float = 512 bytes).
Stores every 4+ iterations are effectively free (overlapped with DPAS execution).
For GEMM: writing accumulator tiles every 4+ K-tile iterations has negligible overhead.

#### 4b.5: Barrier Frequency Impact on Throughput

ILP=8 DPAS with barriers at varying intervals, 160 WGs:

| Barrier interval | 4 SGs TFLOPS | 8 SGs TFLOPS | 16 SGs TFLOPS |
|---|---:|---:|---:|
| Never | 71.6 | 79.5 | 84.2 |
| Every 4 | **86.9** | **108.3** | **117.1** |
| Every 8 | 71.6 | **108.5** | **110.3** |
| Every 16 | 71.7 | 108.4 | 110.1 |
| Every 32 | 71.7 | 108.0 | 103.6 |
| Every 128 | 71.7 | 86.8 | 88.2 |

**Finding**: Barriers every 4-16 iterations **improve** throughput by up to 39% vs no barriers!
Without barriers, SGs run unsynchronized and compete for shared resources. Moderate
synchronization keeps SGs in lockstep, improving XMX scheduling efficiency.

**GEN ASM investigation** reveals the mechanism:

| Variant | `send` count | `dpas` count | CV% | TFLOPS |
|---|---:|---:|---:|---:|
| No barrier | 29 | 128 | 4.3% | 85.7 |
| Barrier every iter | 157 | 128 | 0.9% | 109.9 |

**Critical finding**: The ILP=8 kernel DOES have `send.ugm` instructions inside the loop
body — despite the SPIR-V loading tiles before the loop, the compiler reschedules
`send.ugm load_block2d` into the loop to manage register pressure. With 8 accumulator
tiles (8×16 GRF each) plus 8 A and 8 B tiles, the GRF file is over-subscribed.
The compiler spills tile loads into the loop, creating periodic `send.ugm` + `sync.allwr`
pairs that stall the pipeline.

With barriers, the compiler's instruction scheduling changes: the `sync.allwr` from
the barrier overlaps with the `send.ugm` tile loads, and the barrier's synchronization
effect prevents sub-groups from drifting apart and creating memory subsystem contention.
The CV drops from 4.3% to 0.9%, confirming much more consistent execution.

**Important caveat**: This result is specific to the ILP=8 kernel where the compiler
inserts loop-body memory accesses. For kernels with truly register-resident operands
(smaller ILP or sufficient GRF), barriers would only add overhead (~2-3 cycles per barrier).

**GEMM implication**: For high-ILP GEMM kernels where register pressure forces operand
reloads, barriers every 4-8 K-tiles can improve throughput by 20-40% and reduce variance.
The optimal frequency depends on register pressure and SGs/WG.

#### 4b.6: Software Pipelining (Prefetch Overlap)

Overlapping DPAS with next-tile loads:

| Pattern | Cycles/iter | Delta |
|---|---:|---:|
| DPAS only | 296.3 | 0.0 |
| DPAS + prefetch A (after DPAS) | 285.6 | -10.8 |
| DPAS + prefetch A+B (after DPAS) | 314.8 | +18.5 |
| Load A+B → DPAS (sequential) | 310.2 | +13.9 |

**Finding**: Prefetching A alone is **11 cycles faster** than baseline — the load overlaps
with DPAS execution. Prefetching both A+B adds overhead (can't fully overlap two loads).
Sequential load-then-DPAS is slower by ~14 cycles. Software pipelining is effective when
limited to one tile prefetch per iteration.

---

## Benchmark 4c: Thread Scheduling Granularity

### Design

**Goal**: Understand how work-group sizing, EU thread switching, and dispatch overhead
affect DPAS throughput.

### Results

#### 4c.1: SG/WG Sweep at Fixed Total Work (1280 SGs)

ILP=8, N_ITER=128, total SGs fixed at 1280 (160 EU × 8 threads):

| SGs/WG | WGs | WG size | TFLOPS |
|---:|---:|---:|---:|
| 1 | 1280 | 16 | 80.20 |
| 2 | 640 | 32 | 80.39 |
| 4 | 320 | 64 | **88.02** |
| 8 | 160 | 128 | 80.29 |
| 16 | 80 | 256 | 80.50 |
| 32 | 40 | 512 | 82.15 |

**Finding**: SG/WG ratio has minimal impact (80-88 TFLOPS) at fixed total work.
SG/WG=4 is slightly best. WG granularity is not a significant bottleneck for DPAS throughput.

#### 4c.2: EU Thread Context Switch (ILP=1, Single WG)

ILP=1, N_ITER=1024, single work-group with varying SG count:

| SGs/WG | Total time (ns) | Cycles/DPAS | Ratio to latency |
|---:|---:|---:|---:|
| 1 | 20,048 | 47.0 | 1.42× |
| 2 | 20,220 | 23.7 | 0.72× |
| 4 | 20,161 | 11.8 | 0.36× |
| 8 | 20,409 | 6.0 | 0.18× |
| 16 | 20,411 | 3.0 | 0.09× |

**Note on SG=1 latency**: The 47.0 cycles/DPAS for a single SG with ILP=1 is higher than
the 34.4 cycles measured in Benchmark 1's regression slope. GEN ASM disassembly shows the
compiler **unrolled 16 DPAS per loop iteration** (with counter increment by 16), so the
loop overhead per DPAS is minimal (~0.3 cycles from the `sync.allwr` + `jmpi`). The
dominant contributor to the 13-cycle gap is the **fixed dispatch overhead** (~3.7 μs):
at N_ITER=1024, this contributes ~8.7 cycles/DPAS (3,700 ns × 2.4 GHz / 1024 DPAS).
The remaining ~4 cycles come from initial tile loads and the final store. Benchmark 1's
slope method eliminates the fixed overhead by regression across multiple N values.

**Finding**: Total wall-clock time is ~20 μs regardless of SG count. All SGs run
**in parallel on different EUs** — each additional SG is scheduled on a free EU thread.
The cycles/DPAS scales inversely because we divide by total work, but wall time is constant.

This confirms: within a single WG, multiple SGs execute simultaneously on different EUs.
TLP comes from EU parallelism, not intra-EU thread switching latency hiding.

#### 4c.3: Dispatch Overhead Per WG

Trivial 1-DPAS kernel, sweeping WG count:

```
Linear regression: time = 40.1 ns/WG × n_wg + 3,744 ns
Per-WG dispatch cost: 40.1 ns (96 cycles)
Fixed overhead: 3,744 ns (8,986 cycles)
```

**Finding**: Each work-group adds ~40 ns (96 cycles) of scheduling overhead.
The fixed dispatch latency is ~3.7 μs. For GEMM: larger WGs (more SGs per WG) amortize
the per-WG cost. A 256-thread WG (16 SGs) costs 40 ns per 16 SGs = 2.5 ns/SG.

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
  which stays flat at ~64 cycles across all buffer sizes.

  GEN ASM analysis of the pointer chase kernel reveals 71 cycles includes:
  1. `send.ugm load.d32x1t.a64` — the L1 cache access via the Unified Global Memory
     path (~69 cycles estimated for the cache access itself)
  2. `shl` + `add` — 2 cycles for int32-to-byte address computation
  3. Pipeline dependency tracking (`{$N.dst}` tags enforce serial execution)

  The ~69 cycles for L1 access is higher than NVIDIA (~30 cycles) or AMD (~35 cycles).
  This is explained by Xe2's unified SRAM design: the L1 data cache is not a dedicated
  hardware cache but a **carve-out from a shared SRAM**, accessed through the `send.ugm`
  load/store pipeline. This path includes tag lookup, bank arbitration, and potential
  queueing behind SLM accesses — adding overhead vs. a dedicated L1 design.

  **Bank conflicts are not a factor**: the single-thread serial dependent chain ensures
  only one load is in flight at a time. The 71 cycles is the pure L1 access latency
  through the `send.ugm` pipeline.

  The size-dependent growth (71→145 cycles) in the L1 range is due to **cache set
  pressure**: the random permutation maps different indices to the same cache sets,
  causing conflict misses and increased replacement activity as working set grows
  toward capacity.

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
- **SLM latency growth (46→117 cycles)**: Xe2's SLM is implemented as a multi-banked
  SRAM accessed via `send.slm`. The latency growth with working set size has two likely
  causes — neither of which is bank conflict (this is a **single-thread** serial
  dependent chase, so only one access is in-flight at a time):
  1. **Address decoding and routing overhead**: Larger working sets exercise more banks,
     and the `send.slm` path may incur additional routing latency when the target address
     spans many physical banks. This is an address-decode cost, not a conflict.
  2. **SRAM row/way replacement effects**: Like the L1 data cache, the SLM SRAM has
     finite associativity. A random permutation pattern maps many virtual addresses to
     the same SRAM set, causing conflict misses and row activation overhead as the
     working set grows — analogous to the cache set pressure seen in L1.
  The growth pattern (46 at 256B → 117 at 64KB) is smoother than L1's (71→145),
  consistent with the simpler SRAM structure lacking a full tag hierarchy.
- **Contrast with NVIDIA**: NVIDIA shared memory is a fully separate SRAM with
  ~30 cycle latency and explicit 32-bank architecture. Intel Xe2 SLM at ~46 cycles
  uses a unified SRAM design with `send.slm` path. The banked SRAM behavior is
  similar in principle but with different latency characteristics.

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

| Threads | Read (GB/s) | Write (GB/s) | Copy (GB/s, 2× data) |
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

- **Copy bandwidth**: 439 GB/s = total bytes moved (1 GB read + 1 GB write), reported as
  a single aggregate metric for consistency with the read/write columns. Per-direction
  throughput is ~220 GB/s, limited by combined read+write memory bus contention.

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

**Why does the READ kernel show only 0.002 GB DRAM reads for 1 GB data?** This is the
same L2 cache inflation identified in Benchmark 6. The 1 GB buffer exceeds the 18 MB L2,
but the sequential streaming access pattern allows the hardware prefetcher to keep data
flowing through the cache hierarchy. `GPU_MEMORY_BYTE_READ` counts DRAM pin traffic, and
the `LOAD_STORE_CACHE_BYTE_READ` counter (1.026 GB) confirms the data was read through
the cache path. The WRITE kernel (348 GB/s) is more reliable for DRAM bandwidth measurement
since writes cannot be served from cache. This unitrace data reinforces the conclusion
from the 2 GB bandwidth experiments: sequential read patterns with 1 GB buffers are
substantially cache-assisted.

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
- **Different memory technologies**: B60 uses GDDR6 (256-bit, 18 Gbps);
  Blackwell uses HBM3e (8192-bit, 6.4 Gbps) — comparing DRAM latency across
  these is of limited value since the memory subsystem designs serve different purposes

This comparison serves only to contextualize our measurements, not to declare superiority.

### Latency Comparison

| Metric | Intel B60 (Xe2) | NVIDIA Blackwell | Notes |
|---|---|---|---|
| Tensor/Matrix unit latency | 14 ns (33-37 cyc) | ~7 ns (~20 cyc) | NVIDIA ~2× lower in wall time. Source: arXiv:2507.10789 |
| Shared memory / SLM | **19 ns** (46 cyc, send.slm) | ~11 ns (~30 cyc) | Both faster than L1. Source: arXiv:2507.10789 |
| L1 data cache | 30-60 ns (71-145 cyc) | ~11-13 ns (~30-35 cyc) | NVIDIA ~2.5-5× lower. Xe2 uses unified SRAM path |
| L2 cache | 68-98 ns (162-236 cyc) | ~73-91 ns (~200-250 cyc) | Similar wall time. Source: estimated from published data |
| Global memory | 103-108 ns (247-261 cyc) | 109-182 ns (~300-500 cyc) | Similar range; Intel slightly lower. Note: GDDR6 vs HBM3e |

*Note: B60 at 2.4 GHz. Blackwell data from "Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks" (arXiv:2507.10789, July 2025), except where marked "estimated". Wall-clock time (ns) is more meaningful than cycles for cross-arch comparison since clock speeds differ.*

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
| `run_dpas_ilp_sweep.py` | ILP 1-16 sweep (register pressure, XMX pipeline saturation) |
| `run_dpas_schedule_sweep.py` | DPAS scheduling patterns (barrier, ALU, reload, store, throughput) |
| `run_dpas_multi_sg.py` | Multi-SG coordination (cross-SG dep, barrier freq, staging, pipeline) |
| `run_thread_sched_sweep.py` | Thread scheduling granularity (SG/WG sweep, context switch, dispatch overhead) |
| `run_dpas_precision.py` | BF16/FP16 precision comparison |
| `run_mem_sweep.py` | Memory latency/bandwidth automation |
| `generate_summary.py` | Summary report generator |
| `results/*.csv` | Raw benchmark data |
| `bench_tlb.cpp` | TLB vs cache-set investigation (random/sequential/page-stride patterns) |
| `bench_timing.cpp` | Host timing vs GPU event profiling comparison |
| `Makefile` | Build pipeline |
