# Skill: Low-Bitwidth Vectorized Store Reviewer

## Purpose

Intel GPU (Xe2/Xe3) 的 hardware store 消息最小粒度为 **32-bit**。当每个 thread 做单次 sub-32-bit
store（bf16/fp16/int8/uint8/fp8/int4）时，有效载荷不足，浪费带宽。

本 skill 用于：
1. **Code review** — 识别违反此模式的 kernel 代码
2. **Code generation** — 生成正确的向量化 store 代码
3. **Optimization** — 将已有标量 store 改写为 packed store

**经验证的影响**（BMG-G21 实测，copy kernel，256 MB working set > L2，4096 WG × 256 threads）：
- bf16 标量 store：带宽仅为 fp32 的 0.59x（应 2.0x）→ **3.4x 性能损失**
- int8 标量 store：带宽仅为 fp32 的 0.23x（应 4.0x）→ **17x 性能损失**
- 改用 vec4 packed store 后：bf16 恢复到 1.10x，int8 恢复到 1.04x

---

## Rule

**每个 store 指令必须携带 ≥ 32-bit 有效数据。**

| 元素类型 | 位宽 | 推荐 Packing | 目标 Store |
|---------|------|-------------|-----------|
| fp32 | 32 | 标量即可 | `d32` |
| bf16 / fp16 | 16 | **vec4 → 64-bit** | `d32x2` |
| int8 / uint8 / fp8 | 8 | **vec4 → uint32** | `d32` |
| int4 / uint4 | 4 | **8× → uint32** | `d32` |

---

## Detection Rules

满足以下**任意一条**即触发：

### D1. 标量 sub-32-bit store 在热点循环中
```cpp
// scalar_t ∈ {half, bf16, bfloat16, int8_t, uint8_t, fp8_e4m3, fp8_e5m2, int4, ...}
out_ptr[i]      = val;       // ← 单元素 store
out_ptr[i + 0]  = v0;        // ← 逐元素展开 store
out_ptr[i + 1]  = v1;
```

### D2. Load 向量化但 Store 退化为标量
```cpp
vec4_t<scalar_t> src = v_in[idx];    // vec load ✅
for (int j = 0; j < 4; ++j)
    out_ptr[idx*4 + j] = f(src[j]);  // scalar store ❌
```

### D3. 量化 kernel 逐元素 store int8/uint8
```cpp
out_int8[i] = static_cast<int8_t>(quantize(x));  // 逐字节 store ❌
```

### D4. 缺少 vec/out 分支
低宽 kernel 中**完全没有** `can_vec` / `is_aligned` / `vectorized` 路径判断。

---

## Code Templates

> 以下模板为 kernel 内部逻辑片段，需嵌入 `parallel_for` / `nd_range` 提交。
> `malloc_device` 返回的指针满足 8-byte 对齐，可直接 `reinterpret_cast` 为 `vec4_t*`。
> 若使用其他分配方式，需确保指针对齐到 `alignof(vec4_t<T>)`。

### T1. bf16/fp16 Elementwise — vec4 store

```cpp
// === 类型定义（放在公共头文件）===
template<typename T>
struct alignas(sizeof(T) * 4) vec4_t {
    T v[4];
    T& operator[](int i) { return v[i]; }
    const T& operator[](int i) const { return v[i]; }
};
// bf16 用 uint16_t 而非 sycl::ext::oneapi::bfloat16，原因：
// 1) 确保 IGC 生成 bit-exact d32x2 store，而非可能插入类型转换指令
// 2) sycl::vec<uint16_t,4> 已验证生成 store.ugm.d32x2.a64（690 GB/s）
using vec4_bf16 = vec4_t<uint16_t>;  // bf16 存储为 uint16_t

// === Kernel body（嵌入 parallel_for 的 lambda 中）===
template<typename scalar_t>
void elementwise_kernel(scalar_t* out_raw, const scalar_t* in_raw, int n,
                        /* compute_fn */) {
    using vec4 = vec4_t<scalar_t>;
    const auto* in  = reinterpret_cast<const vec4*>(in_raw);
    auto*       out = reinterpret_cast<vec4*>(out_raw);

    int idx = item.get_global_id(0);
    int n_vec = n / 4;

    if (idx < n_vec) {
        // 1. vec4 load (64-bit)
        vec4 src = in[idx];
        vec4 dst;

        // 2. 逐元素计算
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            dst[j] = compute(src[j]);

        // 3. vec4 store (64-bit) → store.ugm.d32x2
        out[idx] = dst;

    } else if (idx == n_vec) {
        // tail: 仅由一个线程处理剩余元素
        for (int j = n_vec * 4; j < n; ++j)
            out_raw[j] = compute(in_raw[j]);
    }
}
```

### T2. bf16/fp16 Norm Kernel（RMS Norm 示例）

> 注：本模板仅展示 store 端的 vec4 写回模式。`inv_rms` 作为 kernel 参数由外部传入
> （host 端预计算或由 reduction 子群操作得到）。

```cpp
void rms_norm_kernel(bf16* out_raw, const bf16* in_raw, const bf16* w_raw,
                     int n, float inv_rms) {
    using vec4 = vec4_t<uint16_t>;
    const auto* in = reinterpret_cast<const vec4*>(in_raw);
    const auto* w  = reinterpret_cast<const vec4*>(w_raw);
    auto* out      = reinterpret_cast<vec4*>(out_raw);

    int idx = item.get_global_id(0);
    int n_vec = n / 4;
    if (idx >= n_vec) return;

    // Load weight + input
    vec4 src = in[idx];
    vec4 wt  = w[idx];
    vec4 dst;

    // Compute (elementwise in float, inv_rms 由外部传入)
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        float x = bf16_to_float(src[j]);
        dst[j] = float_to_bf16(x * inv_rms * bf16_to_float(wt[j]));
    }

    // Single 64-bit store
    out[idx] = dst;   // ← vec4 store, not 4 scalar stores
}
```

### T3. int8/uint8 Quantization — pack → uint32

```cpp
void quantize_kernel(int8_t* out_raw, const float* in, int n) {
    int idx = item.get_global_id(0);
    int n_pack = n / 4;
    if (idx >= n_pack) return;

    // Load 4 floats
    float x0 = in[idx*4+0], x1 = in[idx*4+1];
    float x2 = in[idx*4+2], x3 = in[idx*4+3];

    // Quantize
    int8_t q0 = quantize(x0), q1 = quantize(x1);
    int8_t q2 = quantize(x2), q3 = quantize(x3);

    // Pack 4×int8 → uint32 → single store
    uint32_t packed = (uint8_t)q0 | ((uint8_t)q1 << 8)
                    | ((uint8_t)q2 << 16) | ((uint8_t)q3 << 24);
    reinterpret_cast<uint32_t*>(out_raw)[idx] = packed;  // ← d32 store
}
```

### T4. fp8 Dequantization — 从 uint32 加载并解包

> 注：本模板展示的是**读端**如何高效加载 packed sub-32-bit 数据（T3 的逆操作），
> 是完整量化/反量化流水线的一部分。

```cpp
void dequant_fp8_kernel(float* out, const fp8_e4m3* in_raw, int n) {
    int idx = item.get_global_id(0);
    int n_pack = n / 4;
    if (idx >= n_pack) return;

    // Load packed 4×fp8 as uint32
    uint32_t packed = reinterpret_cast<const uint32_t*>(in_raw)[idx];

    // Unpack and dequantize（位移提取，避免 device 端 memcpy）
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        uint8_t byte_val = (packed >> (j * 8)) & 0xFF;
        float v = fp8_byte_to_float(byte_val);  // 自定义 fp8 解码
        out[idx*4 + j] = v;
    }
}
```

### T5. int4 Quantization — pack 8× → uint32

```cpp
void quantize_int4_kernel(uint8_t* out_raw, const float* in, int n) {
    int idx = item.get_global_id(0);
    int n_pack = n / 8;  // 8 × 4-bit = 32-bit
    if (idx >= n_pack) return;

    uint32_t packed = 0;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        int4_val q = quantize_4bit(in[idx*8 + j]);
        packed |= (uint32_t)(q & 0xF) << (j * 4);
    }
    reinterpret_cast<uint32_t*>(out_raw)[idx] = packed;
}
```

---

## Verification Checklist

改写完成后，必须验证：

- [ ] Store 指令无 `d16u32` / `d8u32`（`ocloc disasm` 检查）
- [ ] Load 与 Store **同时向量化**（只优化一侧不够）
- [ ] 主路径 vec store，尾部（tail）标量 fallback
- [ ] 地址对齐检查（如 `n % 4 == 0` 或 `is_aligned(ptr, 8)`）
- [ ] `#pragma unroll` 标记 inner loop，确保编译器生成 packed store
- [ ] Benchmark：bf16/fp16/int8 版本 BW ≥ fp32 版本

### ASM 验证命令
```bash
# 编译并反汇编
ocloc compile -spirv_input -file kernel.spv -device bmg-g21 -output kernel
ocloc disasm -file kernel_bmg.bin -dump disasm/ -device bmg-g21
# 检查：不应出现 d16u32 / d8u32
grep -i "store.*d16u32\|store.*d8u32" disasm/.text.*.asm
```

---

## Review Comment Template

> **Low-bitwidth store 反模式**
>
> 此处 `out[i] = val` 对 `<scalar_t = bf16/fp16/int8>` 做了逐元素 global store。
> 在 Intel GPU 上，单次 16-bit/8-bit store 仍按 32-bit 消息发送，有效带宽利用率仅 50%/25%。
>
> 建议改为 packing + 向量化 store：
> - 将 4 个元素打包进 `vec4_t<scalar_t>`（或 `uint32_t` for int8）
> - 主路径用 `v_out[idx] = dst` 一次性写回
> - 增加 tail 标量 fallback 路径
>
> 参考模板：T1（bf16 elementwise）/ T3（int8 quantization）

---

## Known Exceptions

以下情况可**不强制**改写：
- 写回数据量极小（< 1 cache line），改写收益可忽略
- Kernel 是 compute-bound（如 GEMM 主体），瓶颈不在 store
- 数据天然不可对齐且无 padding 方案
- 仅在 host 端 / debug 路径执行的代码

被判为例外时，必须在 PR 描述或代码注释中说明原因。

---

## 验证结果

### 功能正确性测试（BMG-G21 实测）

测试程序：`verify_writeback_skill.cpp`，编译命令：
```
icpx -fsycl -fsycl-targets=intel_gpu_bmg_g21 -O3 -std=c++17 -o verify_writeback_skill verify_writeback_skill.cpp
```

| 模板 | 测试内容 | 结果 |
|------|---------|------|
| T1 | bf16 elementwise vec4 store（4×bf16 → vec4 写回） | **PASS** |
| T2 | bf16 norm vec4 store（权重乘法 + vec4 写回） | **PASS** |
| T3 | int8 量化 pack→uint32（4×int8 打包为 uint32 写回） | **PASS** |
| T4 | fp8 风格 unpack from uint32（uint32 加载 → 解包为 float） | **PASS** |
| T5 | int4 量化 8×→uint32（8×4-bit 打包为 uint32 写回） | **PASS** |
| Control | bf16 标量 store（对照组，应生成 d16u32） | **编译通过**（ASM 验证见下） |

### GEN ASM 验证

通过 SPIR-V microbenchmark 独立验证各模板对应的 store 指令编码：

| 模板 | Store 指令 | 编码 | 有效性 |
|------|-----------|------|--------|
| T1/T2 (bf16 vec4) | `store.ugm.d32x2.a64` | 64-bit | 100% ✅ |
| T3 (int8→uint32) | `store.ugm.d32.a64` | 32-bit | 100% ✅ |
| T4 (fp8 from uint32) | `load.ugm.d32.a64` | 32-bit load | 100% ✅ |
| T5 (int4→uint32) | `store.ugm.d32.a64` | 32-bit | 100% ✅ |
| Control (bf16 scalar) | `store.ugm.d16u32.a64` | 16-bit→32-bit upscale | 50% ❌ |

### 带宽实测对比

| 方案 | Copy BW | vs fp32 | 结论 |
|------|---------|---------|------|
| bf16 标量 store | 371 GB/s (SPIR-V) / 198 GB/s (SYCL) | 0.59x / 0.32x | 问题确认 |
| bf16 vec4 store (T1/T2) | 754 GB/s (SPIR-V) / 691 GB/s (SYCL) | 1.21x / 1.10x | **修复有效** |
| int8 标量 store | 141 GB/s | 0.23x | 问题确认 |
| int8 vec4 store (T3) | 652 GB/s | 1.04x | **修复有效** |
| fp32 标量 store | 625 GB/s | 1.00x (baseline) | — |

> **SPIR-V vs SYCL 带宽差异说明**：SPIR-V 手写汇编直接生成 `d16u32` store 指令，
> SYCL 编译器(IGC)可能额外插入类型转换指令（`mov uw→ud` scatter）和地址计算开销，
> 导致 bf16 标量 store 性能更低（198 vs 371 GB/s）。两种路径的优化方向一致：
> vec4 packed store 均显著恢复带宽（SPIR-V 754 GB/s，SYCL 691 GB/s）。

验证时间：2026-05-09，平台 Intel Arc Pro B60 (BMG-G21)。
