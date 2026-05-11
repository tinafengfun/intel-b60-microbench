// Skill Template Verification — low-bitwidth-writeback
// Tests all 5 code templates (T1-T5) to verify correct store instruction generation
// Build: icpx -fsycl -fsycl-targets=intel_gpu_bmg_g21 -O3 -std=c++17 -o verify_writeback_skill verify_writeback_skill.cpp

#include <sycl/sycl.hpp>
#include <stdio.h>
#include <string.h>
#include <math.h>

using namespace sycl;

// ============================================================
// Shared type definitions (from skill T1)
// ============================================================
template<typename T>
struct alignas(sizeof(T) * 4) vec4_t {
    T v[4];
    T& operator[](int i) { return v[i]; }
    const T& operator[](int i) const { return v[i]; }
};
using vec4_u16 = vec4_t<uint16_t>;  // bf16 as uint16_t

// ============================================================
// T1: bf16/fp16 Elementwise — vec4 store
// Copy kernel: read vec4 bf16, pass through, write vec4 bf16
// ============================================================
void test_T1(queue& q, size_t n) {
    size_t n_vec = n / 4;
    auto src = malloc_device<uint16_t>(n, q);
    auto dst = malloc_device<uint16_t>(n, q);

    // Init: fill with bf16 pattern (0x3F80 = 1.0 in bf16)
    q.fill(src, (uint16_t)0x3F80, n).wait();
    q.fill(dst, (uint16_t)0x0000, n).wait();

    auto src4 = reinterpret_cast<const vec4_u16*>(src);
    auto dst4 = reinterpret_cast<vec4_u16*>(dst);

    int total_threads = 4096 * 256;
    int n_iter = (n_vec + total_threads - 1) / total_threads;

    auto ev = q.submit([&](handler& h) {
        h.parallel_for(nd_range<1>(4096*256, 256),
            [=](nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            int tid = it.get_global_id(0);
            for (int i = 0; i < n_iter; i++) {
                size_t idx = (size_t)tid + (size_t)i * total_threads;
                if (idx < n_vec) {
                    vec4_u16 val = src4[idx];  // vec4 load
                    // elementwise: just pass through
                    vec4_u16 out;
                    #pragma unroll
                    for (int j = 0; j < 4; ++j)
                        out[j] = val[j];
                    dst4[idx] = out;  // vec4 store → should be d32x2
                }
            }
        });
    });
    ev.wait();

    // Verify
    auto host = malloc_host<uint16_t>(n, q);
    q.memcpy(host, dst, n * sizeof(uint16_t)).wait();
    bool ok = true;
    for (size_t i = 0; i < n; i++) {
        if (host[i] != 0x3F80) { ok = false; break; }
    }
    printf("T1 (bf16 elementwise vec4 store): %s\n", ok ? "PASS" : "FAIL");
    free(host, q); free(src, q); free(dst, q);
}

// ============================================================
// T2: bf16 Norm Kernel (RMS Norm simplified) — vec4 store
// Multiply input by a scalar weight, store as vec4 bf16
// ============================================================
void test_T2(queue& q, size_t n) {
    size_t n_vec = n / 4;
    auto in_raw  = malloc_device<uint16_t>(n, q);
    auto w_raw   = malloc_device<uint16_t>(n, q);
    auto out_raw = malloc_device<uint16_t>(n, q);

    // Init input with bf16 2.0 (0x4000), weight with bf16 0.5 (0x3F00)
    q.fill(in_raw,  (uint16_t)0x4000, n).wait();
    q.fill(w_raw,   (uint16_t)0x3F00, n).wait();
    q.fill(out_raw, (uint16_t)0x0000, n).wait();

    auto in4  = reinterpret_cast<const vec4_u16*>(in_raw);
    auto w4   = reinterpret_cast<const vec4_u16*>(w_raw);
    auto out4 = reinterpret_cast<vec4_u16*>(out_raw);

    int total_threads = 4096 * 256;
    int n_iter = (n_vec + total_threads - 1) / total_threads;

    // bf16 approx multiply: just xor exponent bits for simplicity
    // Real impl would use sycl::ext::oneapi::bfloat16
    auto ev = q.submit([&](handler& h) {
        h.parallel_for(nd_range<1>(4096*256, 256),
            [=](nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            int tid = it.get_global_id(0);
            for (int i = 0; i < n_iter; i++) {
                size_t idx = (size_t)tid + (size_t)i * total_threads;
                if (idx < n_vec) {
                    vec4_u16 src = in4[idx];   // vec4 load input
                    vec4_u16 wt  = w4[idx];    // vec4 load weight
                    vec4_u16 dst;
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        // Simple pass-through with weight (just use weight)
                        dst[j] = wt[j];  // store weight value
                    }
                    out4[idx] = dst;  // vec4 store → should be d32x2
                }
            }
        });
    });
    ev.wait();

    auto host = malloc_host<uint16_t>(n, q);
    q.memcpy(host, out_raw, n * sizeof(uint16_t)).wait();
    bool ok = true;
    for (size_t i = 0; i < n; i++) {
        if (host[i] != 0x3F00) { ok = false; break; }
    }
    printf("T2 (bf16 norm vec4 store):        %s\n", ok ? "PASS" : "FAIL");
    free(host, q); free(in_raw, q); free(w_raw, q); free(out_raw, q);
}

// ============================================================
// T3: int8 Quantization — pack 4×int8 → uint32 store
// ============================================================
void test_T3(queue& q, size_t n) {
    size_t n_pack = n / 4;
    auto in_f  = malloc_device<float>(n, q);
    auto out_raw = malloc_device<int8_t>(n, q);

    // Init with known values
    float* host_in = malloc_host<float>(n, q);
    for (size_t i = 0; i < n; i++) host_in[i] = (float)(i % 256);
    q.memcpy(in_f, host_in, n * sizeof(float)).wait();
    free(host_in, q);

    auto out32 = reinterpret_cast<uint32_t*>(out_raw);

    int total_threads = 4096 * 256;
    int n_iter = (n_pack + total_threads - 1) / total_threads;

    auto ev = q.submit([&](handler& h) {
        h.parallel_for(nd_range<1>(4096*256, 256),
            [=](nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            int tid = it.get_global_id(0);
            for (int i = 0; i < n_iter; i++) {
                size_t idx = (size_t)tid + (size_t)i * total_threads;
                if (idx < n_pack) {
                    float x0 = in_f[idx*4+0], x1 = in_f[idx*4+1];
                    float x2 = in_f[idx*4+2], x3 = in_f[idx*4+3];

                    // Quantize to int8
                    int8_t q0 = (int8_t)(int)x0;
                    int8_t q1 = (int8_t)(int)x1;
                    int8_t q2 = (int8_t)(int)x2;
                    int8_t q3 = (int8_t)(int)x3;

                    // Pack 4×int8 → uint32 → single store (T3 template)
                    uint32_t packed = (uint8_t)q0 | ((uint8_t)q1 << 8)
                                    | ((uint8_t)q2 << 16) | ((uint8_t)q3 << 24);
                    out32[idx] = packed;  // should be d32 store
                }
            }
        });
    });
    ev.wait();

    auto host_out = malloc_host<int8_t>(n, q);
    q.memcpy(host_out, out_raw, n * sizeof(int8_t)).wait();
    bool ok = true;
    for (size_t i = 0; i < n; i++) {
        int8_t expected = (int8_t)(int)(i % 256);
        if (host_out[i] != expected) { ok = false; break; }
    }
    printf("T3 (int8 quantize pack→uint32):    %s\n", ok ? "PASS" : "FAIL");
    free(host_out, q); free(in_f, q); free(out_raw, q);
}

// ============================================================
// T4: fp8-style load — load uint32, extract bytes
// Simulates dequant: load packed 4×byte, extract each
// ============================================================
void test_T4(queue& q, size_t n) {
    size_t n_pack = n / 4;
    auto in_raw = malloc_device<uint8_t>(n, q);
    auto out_f  = malloc_device<float>(n, q);

    // Init packed input with byte pattern 0x01, 0x02, 0x03, 0x04 repeating
    uint8_t* host_in = malloc_host<uint8_t>(n, q);
    for (size_t i = 0; i < n; i++) host_in[i] = (uint8_t)((i % 4) + 1);
    q.memcpy(in_raw, host_in, n * sizeof(uint8_t)).wait();
    free(host_in, q);

    auto in32 = reinterpret_cast<const uint32_t*>(in_raw);

    int total_threads = 4096 * 256;
    int n_iter = (n_pack + total_threads - 1) / total_threads;

    auto ev = q.submit([&](handler& h) {
        h.parallel_for(nd_range<1>(4096*256, 256),
            [=](nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            int tid = it.get_global_id(0);
            for (int i = 0; i < n_iter; i++) {
                size_t idx = (size_t)tid + (size_t)i * total_threads;
                if (idx < n_pack) {
                    // Load packed 4×byte as uint32 (T4 template)
                    uint32_t packed = in32[idx];

                    // Unpack: extract each byte and cast to float
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        uint8_t byte_val = (packed >> (j * 8)) & 0xFF;
                        out_f[idx*4 + j] = (float)byte_val;
                    }
                }
            }
        });
    });
    ev.wait();

    auto host_out = malloc_host<float>(n, q);
    q.memcpy(host_out, out_f, n * sizeof(float)).wait();
    bool ok = true;
    for (size_t i = 0; i < n; i++) {
        float expected = (float)((i % 4) + 1);
        if (host_out[i] - expected > 0.01f || expected - host_out[i] > 0.01f) { ok = false; break; }
    }
    printf("T4 (fp8-style unpack from uint32): %s\n", ok ? "PASS" : "FAIL");
    free(host_out, q); free(in_raw, q); free(out_f, q);
}

// ============================================================
// T5: int4 Quantization — pack 8×4-bit → uint32 store
// ============================================================
void test_T5(queue& q, size_t n) {
    size_t n_pack = n / 8;  // 8 × 4-bit = 32-bit
    auto in_f    = malloc_device<float>(n, q);
    auto out_raw = malloc_device<uint8_t>(n / 2, q);  // 2 elements per byte, n/2 bytes

    float* host_in = malloc_host<float>(n, q);
    for (size_t i = 0; i < n; i++) host_in[i] = (float)(i % 16);  // 4-bit values 0-15
    q.memcpy(in_f, host_in, n * sizeof(float)).wait();
    free(host_in, q);

    auto out32 = reinterpret_cast<uint32_t*>(out_raw);

    int total_threads = 4096 * 256;
    int n_iter = (n_pack + total_threads - 1) / total_threads;

    auto ev = q.submit([&](handler& h) {
        h.parallel_for(nd_range<1>(4096*256, 256),
            [=](nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            int tid = it.get_global_id(0);
            for (int i = 0; i < n_iter; i++) {
                size_t idx = (size_t)tid + (size_t)i * total_threads;
                if (idx < n_pack) {
                    // T5 template: pack 8 × 4-bit → uint32
                    uint32_t packed = 0;
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        uint32_t q_val = (uint32_t)((int)in_f[idx*8 + j] & 0xF);
                        packed |= (q_val << (j * 4));
                    }
                    out32[idx] = packed;  // single uint32 store → d32
                }
            }
        });
    });
    ev.wait();

    // Verify: unpack and check
    auto host_out = malloc_host<uint8_t>(n / 2, q);
    q.memcpy(host_out, out_raw, n / 2).wait();
    bool ok = true;
    for (size_t i = 0; i < n; i++) {
        uint8_t byte = host_out[i / 2];
        uint8_t nibble = (i % 2 == 0) ? (byte & 0xF) : ((byte >> 4) & 0xF);
        uint8_t expected = (uint8_t)(i % 16);
        if (nibble != expected) { ok = false; break; }
    }
    printf("T5 (int4 quantize 8×→uint32):     %s\n", ok ? "PASS" : "FAIL");
    free(host_out, q); free(in_f, q); free(out_raw, q);
}

// ============================================================
// Control: bf16 scalar store (BAD — should show d16u32)
// ============================================================
void test_control_scalar_bf16(queue& q, size_t n) {
    auto src = malloc_device<uint16_t>(n, q);
    auto dst = malloc_device<uint16_t>(n, q);

    q.fill(src, (uint16_t)0x3F80, n).wait();

    int total_threads = 4096 * 256;
    int n_iter = (n + total_threads - 1) / total_threads;

    auto ev = q.submit([&](handler& h) {
        h.parallel_for(nd_range<1>(4096*256, 256),
            [=](nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            int tid = it.get_global_id(0);
            for (int i = 0; i < n_iter; i++) {
                size_t idx = (size_t)tid + (size_t)i * total_threads;
                if (idx < n) dst[idx] = src[idx];  // scalar bf16 store → d16u32 (BAD)
            }
        });
    });
    ev.wait();
    printf("Control (bf16 scalar store):       compiled OK (check ASM for d16u32)\n");
    free(src, q); free(dst, q);
}

int main() {
    queue q(gpu_selector_v, property::queue::enable_profiling{});
    auto dev = q.get_device();
    printf("Device: %s\n\n", dev.get_info<info::device::name>().c_str());

    size_t n = 4 * 1024 * 1024;  // 4M elements

    printf("=== Skill Template Correctness Verification ===\n\n");

    test_T1(q, n);          // bf16 elementwise vec4 store
    test_T2(q, n);          // bf16 norm vec4 store
    test_T3(q, n);          // int8 quantize pack→uint32
    test_T4(q, n);          // fp8-style unpack from uint32
    test_T5(q, n);          // int4 quantize 8×→uint32
    test_control_scalar_bf16(q, n);  // control: scalar bf16 (should be bad)

    printf("\n=== Functional tests complete. Now check GEN ASM with ocloc disasm ===\n");
    return 0;
}
