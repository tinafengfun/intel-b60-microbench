# Skill: Low-Bitwidth Vectorized Store Reviewer

## Purpose

Intel GPU (Xe2/Xe3) hardware store messages have a minimum granularity of **32 bits**. When each
thread performs a single sub-32-bit store (bf16/fp16/int8/uint8/fp8/int4), the effective payload
is underutilized, wasting memory bandwidth.

This skill is used to:
1. **Code review** — Identify kernel code that violates this pattern
2. **Code generation** — Generate correct vectorized store code
3. **Optimization** — Rewrite existing scalar stores to packed stores

**Verified impact** (measured on BMG-G21, copy kernel, 256 MB working set > L2, 4096 WGs × 256 threads):
- bf16 scalar store: bandwidth is only 0.59x of fp32 (should be 2.0x) → **3.4x performance loss**
- int8 scalar store: bandwidth is only 0.23x of fp32 (should be 4.0x) → **17x performance loss**
- After switching to vec4 packed store: bf16 recovers to 1.10x, int8 recovers to 1.04x

---

## Rule

**Every store instruction must carry ≥ 32 bits of useful data.**

| Element Type | Bit Width | Recommended Packing | Target Store Encoding |
|-------------|-----------|-------------------|---------------------|
| fp32 | 32 | Scalar is fine | `d32` |
| bf16 / fp16 | 16 | **vec4 → 64-bit** | `d32x2` |
| int8 / uint8 / fp8 | 8 | **vec4 → uint32** | `d32` |
| int4 / uint4 | 4 | **8× → uint32** | `d32` |

> **vec2 is viable but suboptimal**: bf16 vec2 (32-bit) satisfies the ≥32-bit rule at ~380 GB/s
> (0.60x), 2× better than scalar but far behind vec4 at 690 GB/s (1.10x). Consider only when
> n is not a multiple of 4.

## Scope

- **USM mode** (`malloc_device` + raw pointers): Templates apply directly
- **Buffer/Accessor mode**: Same problem exists (accessor bf16 scalar = 0.32x); use
  `buf.reinterpret<sycl::vec<T,4>>(range(n/4))` for vec4 packing (verified 1.14x)
- **Sub-group collective operations** (`sub_group::store`, etc.): Not covered by this skill,
  but sub-32-bit types may also trigger `d16u32`/`d8u32` — requires separate verification

---

## Detection Rules

Trigger on **any** of the following:

### D1. Scalar sub-32-bit store in a hot loop
```cpp
// scalar_t ∈ {half, bf16, bfloat16, int8_t, uint8_t, fp8_e4m3, fp8_e5m2, int4, ...}
out_ptr[i]      = val;       // ← single-element store
out_ptr[i + 0]  = v0;        // ← unrolled element-wise store
out_ptr[i + 1]  = v1;
```

### D2. Vectorized load but scalar store fallback
```cpp
vec4_t<scalar_t> src = v_in[idx];    // vec load ✅
for (int j = 0; j < 4; ++j)
    out_ptr[idx*4 + j] = f(src[j]);  // scalar store ❌
```

### D3. Quantization kernel with per-element int8/uint8 store
```cpp
out_int8[i] = static_cast<int8_t>(quantize(x));  // per-byte store ❌
```

### D4. Missing vec/out branch
Low-bitwidth kernel has **no** `can_vec` / `is_aligned` / `vectorized` path check at all.

---

## Code Templates

> The following templates are kernel-level logic snippets meant to be embedded in a
> `parallel_for` / `nd_range` submission. `malloc_device` typically returns ≥ 8-byte
> aligned pointers, so `reinterpret_cast` to `vec4_t*` is safe. With other allocation
> methods (e.g., `new`/`malloc`), or if you apply a non-aligned offset to the buffer,
> ensure pointer alignment to `alignof(vec4_t<T>)`.

### T1. bf16/fp16 Elementwise — vec4 store

```cpp
// === Type definition (put in common header) ===
template<typename T>
struct alignas(sizeof(T) * 4) vec4_t {
    T v[4];
    T& operator[](int i) { return v[i]; }
    const T& operator[](int i) const { return v[i]; }
};
// bf16 can use uint16_t or sycl::ext::oneapi::bfloat16 — both give identical vec4 performance:
//   sycl::vec<bfloat16,4>:  689 GB/s (1.10x)    <- native type
//   sycl::vec<uint16_t,4>:  666 GB/s (1.06x)    <- underlying storage
//   custom vec4_t<uint16_t>: 663 GB/s (1.06x)   <- this template
// uint16_t benefit: avoids IGC inserting implicit conversion for bf16 arithmetic
using vec4_bf16 = vec4_t<uint16_t>;  // bf16 stored as uint16_t

// === Kernel body (embed in parallel_for lambda) ===
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

        // 2. Per-element computation
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            dst[j] = compute(src[j]);

        // 3. vec4 store (64-bit) → store.ugm.d32x2
        out[idx] = dst;

    } else if (idx == n_vec) {
        // tail: only one thread handles remaining elements
        for (int j = n_vec * 4; j < n; ++j)
            out_raw[j] = compute(in_raw[j]);
    }
}
```

### T2. bf16/fp16 Norm Kernel (RMS Norm example)

> Note: This template only shows the vec4 store pattern on the write side. `inv_rms`
> is passed as a kernel parameter (pre-computed via sub-group reduction or host-side).

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

    // Compute (elementwise in float, inv_rms passed from outside)
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

> Note: This template assumes `n` is a multiple of 4. If not, add a scalar tail fallback (see T1).

```cpp
void quantize_kernel(int8_t* out_raw, const float* in, int n) {
    int idx = item.get_global_id(0);
    int n_pack = n / 4;
    if (idx >= n_pack) return;

    // Load 4 floats as vec4 (vectorized load, symmetric with store side)
    auto in4 = reinterpret_cast<const sycl::vec<float, 4>*>(in);
    sycl::vec<float, 4> src = in4[idx];
    float x0 = src.s0(), x1 = src.s1(), x2 = src.s2(), x3 = src.s3();

    // Quantize
    int8_t q0 = quantize(x0), q1 = quantize(x1);
    int8_t q2 = quantize(x2), q3 = quantize(x3);

    // Pack 4×int8 → uint32 → single store
    uint32_t packed = (uint8_t)q0 | ((uint8_t)q1 << 8)
                    | ((uint8_t)q2 << 16) | ((uint8_t)q3 << 24);
    reinterpret_cast<uint32_t*>(out_raw)[idx] = packed;  // ← d32 store
}
```

### T4. fp8 Dequantization — load from uint32 and unpack

> Note: This template shows the **read-side** counterpart — how to efficiently load packed
> sub-32-bit data (the inverse of T3). It is part of a complete quantize/dequantize pipeline.

```cpp
void dequant_fp8_kernel(float* out, const fp8_e4m3* in_raw, int n) {
    int idx = item.get_global_id(0);
    int n_pack = n / 4;
    if (idx >= n_pack) return;

    // Load packed 4×fp8 as uint32
    uint32_t packed = reinterpret_cast<const uint32_t*>(in_raw)[idx];

    // Unpack and dequantize (bit-shift extraction, no memcpy in device code)
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        uint8_t byte_val = (packed >> (j * 8)) & 0xFF;
        float v = fp8_byte_to_float(byte_val);  // custom fp8 decode
        out[idx*4 + j] = v;
    }
}
```

### T5. int4 Quantization — pack 8× → uint32

> Note: This template assumes `n` is a multiple of 8. If not, add a scalar tail fallback (see T1).

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

After rewriting, verify the following:

- [ ] No `d16u32` / `d8u32` in store instructions (check with `ocloc disasm`)
- [ ] **Both** Load and Store are vectorized (optimizing only one side is insufficient)
- [ ] Main path uses vec store, tail uses scalar fallback
- [ ] Alignment check present (e.g., `n % 4 == 0` or `is_aligned(ptr, 8)`)
- [ ] `#pragma unroll` on inner loops to ensure compiler generates packed stores
- [ ] Benchmark: bf16/fp16/int8 version BW ≥ fp32 version

### ASM Verification Commands
```bash
# Option 1: SYCL AOT compile + disassemble (recommended for most SYCL users)
icpx -fsycl -fsycl-targets=intel_gpu_bmg_g21 -O3 -o my_kernel my_kernel.cpp
ocloc disasm -file my_kernel -dump disasm/ -device bmg-g21
grep -i "store.*d16u32\|store.*d8u32" disasm/.text.*.asm

# Option 2: SPIR-V hand-written assembly path (for precise instruction control)
spirv-as kernel.spvasm -o kernel.spv --target-env spv1.4
ocloc compile -spirv_input -file kernel.spv -device bmg-g21 -output kernel
ocloc disasm -file kernel_bmg.bin -dump disasm/ -device bmg-g21
grep -i "store.*d16u32\|store.*d8u32" disasm/.text.*.asm
```

---

## Review Comment Template

> **Low-bitwidth store anti-pattern**
>
> This `out[i] = val` performs a per-element global store on `<scalar_t = bf16/fp16/int8>`.
> On Intel GPUs, a single 16-bit/8-bit store still sends a 32-bit message, achieving only
> 50%/25% effective bandwidth utilization.
>
> Suggested fix — pack and vectorize the store:
> - Pack 4 elements into `vec4_t<scalar_t>` (or `uint32_t` for int8)
> - Main path: `v_out[idx] = dst` for a single wide store
> - Add a scalar fallback path for tail elements
>
> Reference templates: T1 (bf16 elementwise) / T3 (int8 quantization)

---

## Known Exceptions

The following cases may be **exempt** from rewriting:
- Writeback data volume is tiny (< 1 cache line) — negligible benefit
- Kernel is compute-bound (e.g., GEMM body) — store is not the bottleneck
- Data is inherently unaligned with no feasible padding scheme
- Code runs only on the host / debug path

When claiming an exception, the reason must be documented in the PR description or code comments.

---

## Verification Results

### Functional Correctness Tests (BMG-G21 Measured)

Test program: `verify_writeback_skill.cpp`, build command:
```
icpx -fsycl -fsycl-targets=intel_gpu_bmg_g21 -O3 -std=c++17 -o verify_writeback_skill verify_writeback_skill.cpp
```

| Template | Test Description | Result |
|----------|-----------------|--------|
| T1 | bf16 elementwise vec4 store (4×bf16 → vec4 writeback) | **PASS** |
| T2 | bf16 norm vec4 store (weight multiply + vec4 writeback) | **PASS** |
| T3 | int8 quantize pack→uint32 (4×int8 packed to uint32 writeback) | **PASS** |
| T4 | fp8-style unpack from uint32 (uint32 load → unpack to float) | **PASS** |
| T5 | int4 quantize 8×→uint32 (8×4-bit packed to uint32 writeback) | **PASS** |
| Control | bf16 scalar store (control group, should generate d16u32) | **Compiled OK** (see ASM below) |

### GEN ASM Verification

Store instruction encodings verified independently via SPIR-V microbenchmarks:

| Template | Store Instruction | Encoding | Efficiency |
|----------|------------------|----------|------------|
| T1/T2 (bf16 vec4) | `store.ugm.d32x2.a64` | 64-bit | 100% ✅ |
| T3 (int8→uint32) | `store.ugm.d32.a64` | 32-bit | 100% ✅ |
| T4 (fp8 from uint32) | `load.ugm.d32.a64` | 32-bit load | 100% ✅ |
| T5 (int4→uint32) | `store.ugm.d32.a64` | 32-bit | 100% ✅ |
| Control (bf16 scalar) | `store.ugm.d16u32.a64` | 16-bit→32-bit upscale | 50% ❌ |

### Measured Bandwidth Comparison

| Approach | Copy BW | vs fp32 | Conclusion |
|----------|---------|---------|-------------|
| bf16 scalar store | 371 GB/s (SPIR-V) / 198 GB/s (SYCL) | 0.59x / 0.32x | Problem confirmed |
| bf16 vec4 store (T1/T2) | 754 GB/s (SPIR-V) / 691 GB/s (SYCL) | 1.21x / 1.10x | **Fix effective** |
| bf16 vec2 store | 380 GB/s (SYCL) | 0.60x | Partial fix |
| int8 scalar store | 141 GB/s (SPIR-V only) | 0.23x | Problem confirmed |
| int8 vec4 store (T3) | 652 GB/s (SPIR-V only) | 1.04x | **Fix effective** |
| fp32 scalar store | 628 GB/s | 1.00x (baseline) | — |

### SYCL Type Comparison (BMG-G21, GPU event profiling)

| Type Approach | Copy BW | vs fp32 | Notes |
|--------------|---------|---------|-------|
| `sycl::ext::oneapi::bfloat16` scalar | 198 GB/s | 0.31x | Same problem |
| `sycl::vec<bfloat16,4>` vec4 | **689 GB/s** | **1.10x** | Native type + vec4 = fully fixed |
| `sycl::vec<uint16_t,4>` vec4 | 666 GB/s | 1.06x | Equivalent to custom vec4_t |
| `vec4_t<uint16_t>` custom | 663 GB/s | 1.06x | Skill default template |
| `vec2_t<uint16_t>` custom | 380 GB/s | 0.60x | Satisfies ≥32-bit but suboptimal |
| accessor bf16 scalar | 203 GB/s | 0.32x | Buffer/Accessor has same issue |
| accessor bf16 vec4 | 718 GB/s | 1.14x | `buf.reinterpret<vec<T,4>>` fix |

> **SPIR-V vs SYCL bandwidth gap**: Hand-written SPIR-V directly generates `d16u32` store
> instructions. The SYCL compilation chain (icpx frontend + IGC backend) may insert
> additional type conversions (e.g., bfloat16→float→bfloat16 round-trip) and address
> computation overhead, making bf16 scalar store even slower (198 vs 371 GB/s).
> Both paths benefit equally from vec4 packed stores (SPIR-V 754 GB/s, SYCL 691 GB/s).

Verified on 2026-05-11, Intel Arc Pro B60 (BMG-G21).

## Trigger Keywords

Match condition: **Type keywords ∩ Operation keywords** must both be present (type alone is insufficient to trigger).

```
Type keywords: sycl::half, sycl::ext::oneapi::bfloat16, int8_t, uint8_t, fp8, int4
Operation keywords: store, write, writeback, quantize, dequantize, elementwise
Additional keywords: low-precision, sub-byte, packed store, vectorized store
```
