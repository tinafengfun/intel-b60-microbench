# Intel Xe2 低比特宽写回带宽浪费 — 验证与分析报告

> 验证平台：Intel Arc Pro B60 (BMG-G21, Xe2), 20 Xe Cores, 24 GB GDDR6
> 验证工具：SPIR-V 手写汇编 microbenchmark + Level Zero 计时 + unitrace 硬件计数器 + GEN ISA 反汇编 + SYCL 验证

---

## 1. 问题陈述

当 GPU kernel 中每个 thread 向 global memory 写回 sub-32-bit 数据（bf16/fp16/int8/uint8/fp8）时，
硬件 store 消息的最小粒度为 32-bit，导致有效载荷占比不足，带宽被浪费。

**预期现象**：memory-bound kernel 中，bf16/fp16 版本吞吐应高于 fp32（数据量更小），但实际反而更低。

---

## 2. 验证方法

### 2.1 测试设计

5 种 SPIR-V copy kernel（读 buf[0] → 写 buf[1]），控制指令为手写 SPIR-V 汇编，避免编译器自动向量化干扰：

| 变体 | 每线程写回 | 元素大小 | SPIR-V Store 类型 |
|------|-----------|---------|------------------|
| fp32_scalar | 1 × float | 4 B | `OpStore %ptr_cross_float Aligned 4` |
| bf16_scalar | 1 × ushort | 2 B | `OpStore %ptr_cross_ushort Aligned 2` |
| bf16_vec4 | 4 × ushort (vec4) | 8 B | `OpStore %ptr_cross_v4ushort Aligned 8` |
| u8_scalar | 1 × uchar | 1 B | `OpStore %ptr_cross_uchar Aligned 1` |
| u8_vec4 | 4 × uchar → uint32 | 4 B | `OpStore %ptr_cross_uint Aligned 4` |

**测试配置**：4096 WGs × 256 threads = 1M 线程，64 iterations/thread，256 MB buffer。
所有变体的 working set 均超过 L2 cache (18 MB)，确保测量真实 DRAM 带宽。

### 2.2 验证工具链

- **SPIR-V 汇编**：`spirv-as` 组装 → `spirv_runner` (Level Zero JIT) 执行
- **GEN ISA 反汇编**：`ocloc compile` → `ocloc disasm` 获取硬件指令
- **硬件计数器**：`unitrace -g ComputeBasic` 采集 DRAM/LSC 流量
- **SYCL 验证**：`icpx -fsycl` 编译，验证编译器行为一致性

---

## 3. 测试结果

### 3.1 带宽对比（SPIR-V Microbenchmark，host chrono 计时）

```
Variant          ElemB   WS(MB)    Med(ns)     CopyBW    vs fp32
--------------------------------------------------------------------
fp32_scalar          4      256     859216    624.8 GB/s   (baseline)
bf16_scalar          2      128     722973    371.3 GB/s   0.59x  ← 浪费
bf16_vec4            8      512    1424960    753.5 GB/s   1.21x  ← 修复
u8_scalar            1       64     950863    141.2 GB/s   0.23x  ← 严重浪费
u8_vec4              4      256     822948    652.4 GB/s   1.04x  ← 修复
```

### 3.2 带宽对比（SYCL 验证，GPU event profiling 计时）

```
fp32_scalar:  512 MB, 1713 μs, 626.7 GB/s   (baseline)
scalar_half:  256 MB, 2706 μs, 198.4 GB/s    0.32x  ← 浪费
vec4_half:    256 MB,  777 μs, 690.8 GB/s    1.10x  ← 修复
pack_uint32:  256 MB, 1349 μs, 398.1 GB/s    0.63x  ← 部分修复
```

两种测量方法结论一致。

### 3.3 关键对比汇总

| 对比项 | 实测比值 | 理想值（无浪费） | 若 32-bit 消息 | 结论 |
|--------|---------|----------------|--------------|------|
| bf16_scalar / fp32_scalar | **0.59x** | 2.0x | ~1.0x | **浪费确认** |
| bf16_vec4 / bf16_scalar | **2.03x** | — | — | **修复有效** |
| u8_scalar / fp32_scalar | **0.23x** | 4.0x | ~1.0x | **严重浪费** |
| u8_vec4 / u8_scalar | **4.62x** | — | — | **修复有效** |
| sycl::vec4 / scalar (SYCL) | **3.48x** | — | — | **修复有效** |

---

## 4. 根因分析

### 4.1 GEN ISA 证据：Store 消息编码

通过 `ocloc disasm` 反汇编，对比各变体的 store 指令编码：

| 变体 | Store 指令 | 消息描述符 | 含义 |
|------|-----------|-----------|------|
| fp32_scalar | `store.ugm.d32.a64` | `0x08000584` | 32-bit 数据 → 32-bit 消息 |
| **bf16_scalar** | `store.ugm.d16u32.a64` | `0x08000B84` | 16-bit 数据 **upscale 到 32-bit 消息** |
| bf16_vec4 | `store.ugm.d32x2.a64` | `0x08001584` | 64-bit store (2×32) |
| **u8_scalar** | `store.ugm.d8u32.a64` | `0x08000984` | 8-bit 数据 **upscale 到 32-bit 消息** |
| u8_vec4 | `store.ugm.d32.a64` | `0x08000584` | 32-bit store（与 fp32 相同） |

**关键编码解读**：
- `d32`：32-bit 数据在 32-bit 消息中 → 100% 有效
- `d16u32`：d(ata)=16-bit, u(pscaled to)=32-bit 消息 → 50% 有效
- `d8u32`：d(ata)=8-bit, u(pscaled to)=32-bit 消息 → 25% 有效
- `d32x2`：2×32-bit = 64-bit 消息 → 100% 有效

### 4.2 数据搬运路径对比

从 GEN ASM 中提取完整的 Load → Mov → Store 数据流：

**fp32_scalar（高效）**：
```
load.ugm.d32 → r44:2 ─────────────────────→ store.ugm.d32 (r44:2)
 [256B/load, 256B 有用]   直通              [256B/store, 256B 有用]
                                              有效率：100%
```

**bf16_scalar（50% 浪费）**：
```
load.ugm.d16u32 → r44 ──→ mov uw→ud ──→ r80:2 ──→ store.ugm.d16u32 (r80:2)
 [256B, 128B 有用]      scatter 展开       [256B 消息, 128B 有用]
 r44.0<2;1,0>:uw → r80.0<1>:ud             有效率：50%
```
IGC 编译器插入了 `mov (32|M0) r80.0<1>:ud, r44.0<2;1,0>:uw` 将 16-bit stride-2
scatter 到 32-bit stride-1 寄存器布局，然后用 `d16u32` 编码发出 store 消息。
消息的有效载荷仅占 50%。

**u8_scalar（87.5% 浪费）**：
```
load.ugm.d8u32 → r32 ──→ (直通) ──→ store.ugm.d8u32 (r2:2)
 [256B, 32B 有用]                  [256B 消息, 32B 有用]
                                    有效率：12.5%
```
8-bit 数据直接放入 32-bit slot，75% 的消息位宽浪费，实际有效载荷仅 12.5%。

**bf16_vec4（修复后）**：
```
load.ugm.d32x2 → r44:4 ──────────────────→ store.ugm.d32x2 (r44:4)
 [512B/load, 512B 有用]   直通             [512B/store, 512B 有用]
                                              有效率：100%
```
4×16-bit = 64-bit 数据通过 `d32x2` 编码一次发出，无浪费。

### 4.3 消息数量瓶颈

所有变体经 IGC 编译后产生**相同数量的 store 消息**（~8 条/unrolled-iteration），
但每条消息携带的有效数据不同：

| 变体 | Store 消息数/iter | 每消息有效载荷 | 总有效数据 |
|------|-----------------|-------------|-----------|
| fp32_scalar | 8 | 32-bit × 32 lanes = 128B | 1024B |
| bf16_scalar | 8 | 16-bit × 32 lanes = 64B | 512B |
| bf16_vec4 | 8 | 64-bit × 32 lanes = 256B | 2048B |
| u8_scalar | 8 | 8-bit × 32 lanes = 32B | 256B |
| u8_vec4 | 8 | 32-bit × 32 lanes = 128B | 1024B |

瓶颈是 **store 消息吞吐率**（每秒发出消息数），而非数据量。同样的消息数，
bf16_scalar 只传了一半的有效数据，u8_scalar 只传了 1/4。

### 4.4 Unitrace 硬件计数器验证

| 变体 | GPU Time | DRAM Read | DRAM Write | LSC Read | DRAM Write Rate |
|------|---------|-----------|------------|----------|----------------|
| fp32_scalar | 855 μs | 42.5 MB | 182.8 MB | 269.5 MB | 213.7 GB/s |
| bf16_scalar | 683 μs | 45.0 MB | 97.2 MB | 135.3 MB | 142.4 GB/s |
| u8_scalar | 959 μs | 28.8 MB | 47.1 MB | 68.2 MB | **49.1 GB/s** |
| u8_vec4 | 810 μs | 42.5 MB | 182.9 MB | 269.5 MB | 225.9 GB/s |

关键发现：
- **LSC Read 正确缩放**：269.5 / 135.3 / 68.2 MB = 4:2:1（匹配数据量）
- **DRAM Write 流量**按数据量正确缩放，但 **Write Rate** 严重偏低
- u8_scalar 的 DRAM Write Rate 仅 49.1 GB/s（vs fp32 的 213.7 GB/s），尽管写入数据量只有 fp32 的 1/4
- u8_vec4 恢复到 225.9 GB/s（**4.6× 提升**）

这说明 DRAM 控制器层面的 traffic 确实是按实际数据量走的，但瓶颈在于 **store pipeline
的消息吞吐率**：`d8u32` 编码的消息数与 `d32` 相同，每条消息只携带 1 字节有效数据。

---

## 5. 编译器/语言层面能否自动修复？

### 5.1 编译器（IGC）能否自动向量化？—— 不能

编译器面对 `out[i] = val`（bf16）时，语义是 "thread i 写地址 base + i×2"。
要改为 vec4 store，地址变为 base + i×8，这是**程序语义的变更**，编译器无权做出。

此外，跨 thread 合并 store 需要子群级别（sub-group level）优化：
1. 证明子群内所有线程在 uniform control flow 中
2. 用 shuffle 指令合并相邻线程的数据
3. 只让偶数线程执行 store
4. 处理非对齐尾部元素

这个优化难度极高，且是 Intel 特有的（NVIDIA 的 warp 天然 32 线程 × 16-bit = 64B = 1 cache line，不存在此问题）。

### 5.2 SYCL `vec` 类型是否有效？—— 有效，已验证

```cpp
// 标量 store → store.ugm.d16u32 (198 GB/s, 0.32x)
dst[idx] = src[idx];

// sycl::vec<4> store → store.ugm.d32x2 (690 GB/s, 1.10x)
reinterpret_cast<sycl::vec<uint16_t,4>*>(dst)[idx] =
    reinterpret_cast<sycl::vec<uint16_t,4>*>(src)[idx];
```

**3.48× 提升**，超过 fp32 基线。IGC 能正确将 `sycl::vec<uint16_t, 4>` 编译为 `d32x2` 64-bit store。

### 5.3 `reinterpret_cast<uint32_t*>` 是否有效？—— 部分有效

```cpp
// pack 2×uint16 → uint32 → store.ugm.d32 (398 GB/s, 0.63x)
reinterpret_cast<uint32_t*>(dst)[idx] = reinterpret_cast<uint32_t*>(src)[idx];
```

比标量好 2×，但仍不如 `vec4`。原因是 `uint32_t` 只 pack 2 个 bf16（32-bit），
而 `vec4` pack 4 个 bf16（64-bit），后者使用 `d32x2` 编码效率更高。

### 5.4 总结

| 方案 | Copy BW | vs fp32 | 改动量 | 结论 |
|------|---------|---------|--------|------|
| 标量 bf16 store | 198 GB/s | 0.32x | 无 | 问题本身 |
| `reinterpret_cast<uint32_t*>` | 398 GB/s | 0.63x | 小 | 部分修复 |
| `sycl::vec<T,4>` / `vec4_t` | **690 GB/s** | **1.10x** | 小 | **推荐** |
| IGC 自动优化 | — | — | — | 不可行 |

---

## 6. 解决方案总结

### 6.1 核心规则

**每个 store 消息必须携带 ≥ 32-bit 有效数据。**

| 元素类型 | 单元素位宽 | 推荐 Packing | Store 编码 | GEN ASM |
|---------|-----------|-------------|-----------|---------|
| fp32 | 32-bit | 标量即可 | `d32` | `store.ugm.d32.a64` |
| bf16/fp16 | 16-bit | **vec4 → 64-bit** | `d32x2` | `store.ugm.d32x2.a64` |
| int8/uint8/fp8 | 8-bit | **vec4 → uint32** | `d32` | `store.ugm.d32.a64` |
| int4/uint4 | 4-bit | **8× pack → uint32** | `d32` | `store.ugm.d32.a64` |

### 6.2 SYCL 代码示例

**bf16 elementwise kernel**：
```cpp
// ❌ Bad: 标量 bf16 store → store.ugm.d16u32 (50% 浪费)
void bad_kernel(bf16* out, const bf16* in, int n) {
    int i = item.get_global_id(0);
    if (i < n) out[i] = compute(in[i]);
}

// ✅ Good: vec4 bf16 store → store.ugm.d32x2 (0% 浪费)
using vec4bf16 = sycl::vec<uint16_t, 4>;  // 或自定义 alignas(8) struct

void good_kernel(bf16* out_raw, const bf16* in_raw, int n) {
    auto* in  = reinterpret_cast<const vec4bf16*>(in_raw);
    auto* out = reinterpret_cast<vec4bf16*>(out_raw);
    int idx = item.get_global_id(0);
    if (idx * 4 + 3 < n) {
        out[idx] = compute_vec4(in[idx]);  // 一次 64-bit store
    } else { /* tail: 标量 fallback */ }
}
```

**int8 量化 kernel**：
```cpp
// ❌ Bad: 逐字节写回 → store.ugm.d8u32 (75% 浪费)
out_i8[i] = quantize(x);

// ✅ Good: pack 4×int8 → uint32 → store.ugm.d32
uint32_t packed = (uint32_t)q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
reinterpret_cast<uint32_t*>(out_i8)[idx] = packed;
```

### 6.3 验证方法

1. **GEN ASM 检查**：`ocloc disasm` 搜索 store 指令，确认无 `d16u32` / `d8u32`
2. **Unitrace 验证**：对比 `GPU_MEMORY_BYTE_WRITE_RATE`，bf16 版本不应显著低于 fp32
3. **Benchmark 对比**：bf16 版本 copy BW 应 ≥ fp32 版本

---

## 附录 A：测试脚本

- `run_writeback_sweep.py`：SPIR-V microbenchmark 全流程脚本
- `bench_writeback_sycl.cpp`：SYCL 验证 benchmark
- 结果数据：`results/writeback_sweep.csv`

## 附录 B：复现命令

```bash
# SPIR-V 基准测试
python3 run_writeback_sweep.py

# SYCL 验证
icpx -fsycl -fsycl-targets=intel_gpu_bmg_g21 -O3 -std=c++17 \
     -o bench_writeback_sycl bench_writeback_sycl.cpp
./bench_writeback_sycl

# Unitrace 硬件计数器
/path/to/unitrace -q -g ComputeBasic ./bench_writeback_sycl

# GEN ASM 反汇编
ocloc compile -spirv_input -file kernel.spv -device bmg-g21 -output kernel
ocloc disasm -file kernel_bmg.bin -dump kernel_disasm -device bmg-g21
grep "store.ugm" kernel_disasm/.text.*.asm
```
