// Quantization / dequantization: cvt instruction throughput + memory-bound (de)quant kernels
#include "bench_common.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// runtime seed so constant-folding cannot eliminate cvt chains
__constant__ unsigned SEED[1] = {0x9e3779b9u};

// e2m1 LUT: sign,exp(2),man(1) -> value
__constant__ float E2M1_LUT[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
                                   -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

// ---------- cvt throughput kernels (8 independent chains) ----------
#define THR8(NAME, TYPE, INIT, BODY, OUTT) \
__global__ void thrk_##NAME(int iters, OUTT* out) { \
    TYPE a0=(INIT)^SEED[0],a1=(INIT+1)^SEED[0],a2=(INIT+2)^SEED[0],a3=(INIT+3)^SEED[0], \
         a4=(INIT+4)^SEED[0],a5=(INIT+5)^SEED[0],a6=(INIT+6)^SEED[0],a7=(INIT+7)^SEED[0]; \
    for (int i = 0; i < iters; i++) { \
        a0=(BODY(a0)); asm volatile("" : "+r"(a0)); \
        a1=(BODY(a1)); asm volatile("" : "+r"(a1)); \
        a2=(BODY(a2)); asm volatile("" : "+r"(a2)); \
        a3=(BODY(a3)); asm volatile("" : "+r"(a3)); \
        a4=(BODY(a4)); asm volatile("" : "+r"(a4)); \
        a5=(BODY(a5)); asm volatile("" : "+r"(a5)); \
        a6=(BODY(a6)); asm volatile("" : "+r"(a6)); \
        a7=(BODY(a7)); asm volatile("" : "+r"(a7)); } \
    OUTT s = a0^a1^a2^a3^a4^a5^a6^a7; \
    if (s == (OUTT)-12345) out[0] = s; \
}

// f32x2 -> f16x2 (cvt.rn.f16x2.f32) + xor-accumulate bits
__device__ __forceinline__ unsigned op_f32x2_f16x2(unsigned x) {
    __half2_raw r = __floats2half2_rn(__uint_as_float(x), __uint_as_float(x ^ 0x00800000u));
    return *(unsigned*)&r;
}
// f16x2 -> f32x2 -> re-pack (F2F + F2F)
__device__ __forceinline__ unsigned op_f16x2_f32(unsigned x) {
    __half2_raw r = *(__half2_raw*)&x;
    float2 f = __half22float2(*(__half2*)&r);
    __half2_raw r2 = __floats2half2_rn(f.x + 1e-6f, f.y);
    return *(unsigned*)&r2;
}
// f32x2 -> e4m3x2 (cvt.rn.satfinite.e4m3x2.f32)
__device__ __forceinline__ unsigned op_f32x2_e4m3x2(unsigned x) {
    unsigned short r = __nv_cvt_float2_to_fp8x2(make_float2(__uint_as_float(x), __uint_as_float(x ^ 0x00800000u)), __NV_SATFINITE, __NV_E4M3);
    return (unsigned)r;
}
// e4m3x2 -> f16x2 (cvt.rn.f16x2.e4m3x2)
__device__ __forceinline__ unsigned op_e4m3x2_f16x2(unsigned x) {
    __half2_raw r = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(x & 0xFFFF), __NV_E4M3);
    return *(unsigned*)&r;
}
// f32x2 -> bf16x2
__device__ __forceinline__ unsigned op_f32x2_bf16x2(unsigned x) {
    __nv_bfloat162 r = __floats2bfloat162_rn(__uint_as_float(x), __uint_as_float(x ^ 0x00800000u));
    return *(unsigned*)&r;
}
// fp4 soft-unpack: 8x e2m1 in u32 -> sum of 8 halves (LUT path), return packed bits
__device__ __forceinline__ unsigned op_fp4_unpack_soft(unsigned x) {
    float s = 0; unsigned v = x;
    #pragma unroll
    for (int j = 0; j < 8; j++) s += E2M1_LUT[(v >> (4 * j)) & 0xF];
    __half2_raw r = __floats2half2_rn(s, s);
    return *(unsigned*)&r;
}
// int8x2 -> f16x2 unpack
__device__ __forceinline__ unsigned op_int8_unpack(unsigned x) {
    int lo = (short)(char)(x & 0xFF), hi = (short)(char)((x >> 8) & 0xFF);
    __half2_raw r = __floats2half2_rn((float)lo, (float)hi);
    return *(unsigned*)&r;
}

THR8(cvt_f32x2_f16x2, unsigned, 0x3f800000u, op_f32x2_f16x2, unsigned)
THR8(cvt_f16x2_f32, unsigned, 0x3c003c00u, op_f16x2_f32, unsigned)
THR8(cvt_f32x2_e4m3x2, unsigned, 0x3f800000u, op_f32x2_e4m3x2, unsigned)
THR8(cvt_e4m3x2_f16x2, unsigned, 0x3030u, op_e4m3x2_f16x2, unsigned)
THR8(cvt_f32x2_bf16x2, unsigned, 0x3f800000u, op_f32x2_bf16x2, unsigned)
THR8(fp4_unpack_soft, unsigned, 0x76543210u, op_fp4_unpack_soft, unsigned)
THR8(int8_unpack, unsigned, 0x01020304u, op_int8_unpack, unsigned)

// ---------- memory-bound dequant/quant kernels ----------
// a) int8 + per-channel fp16 scale -> fp16
__global__ void deq_int8(const int4* in, const __half* scale, int C, __half2* out, long long nvec) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < nvec; i += stride) {
        int4 v = in[i];
        __half2 o[8];
        const signed char* p = (const signed char*)&v;
        #pragma unroll
        for (int j = 0; j < 16; j += 2) {
            long long elem = i * 16 + j;
            float s0 = __half2float(scale[(elem) % C]);
            float s1 = __half2float(scale[(elem + 1) % C]);
            o[j / 2] = __floats2half2_rn((float)p[j] * s0, (float)p[j + 1] * s1);
        }
        int4* w = (int4*)&out[i * 8];
        w[0] = *(int4*)&o[0]; w[1] = *(int4*)&o[4];
    }
}
// b) fp8 e4m3 + fp16 scale -> bf16
__global__ void deq_fp8(const int4* in, const __half* scale, int C, __nv_bfloat162* out, long long nvec) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < nvec; i += stride) {
        int4 v = in[i];
        const unsigned short* p = (const unsigned short*)&v;
        __nv_bfloat162 o[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            __half2_raw h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)p[j], __NV_E4M3);
            float2 f = __half22float2(*(__half2*)&h);
            long long elem = i * 16 + 2 * j;
            float s0 = __half2float(scale[elem % C]);
            float s1 = __half2float(scale[(elem + 1) % C]);
            o[j] = __floats2bfloat162_rn(f.x * s0, f.y * s1);
        }
        int4* w = (int4*)&out[i * 8];
        w[0] = *(int4*)&o[0]; w[1] = *(int4*)&o[4];
    }
}
// c) nvfp4 (8x e2m1 per u32) + e4m3 scale per 16 elems -> fp16
__global__ void deq_fp4(const int4* in, const unsigned char* scale8, __half2* out, long long nvec) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < nvec; i += stride) {  // one int4 = 16 bytes = 32 e2m1 elements
        int4 v = in[i];
        __half2_raw sc = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(scale8[2 * i] | (scale8[2 * i + 1] << 8)), __NV_E4M3);
        float2 sf = __half22float2(*(__half2*)&sc);
        const unsigned* p = (const unsigned*)&v;
        __half2 o[16];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            unsigned w = p[j];
            float s = (j < 2) ? sf.x : sf.y;
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                float v0 = E2M1_LUT[(w >> (8 * k)) & 0xF] * s;
                float v1 = E2M1_LUT[(w >> (8 * k + 4)) & 0xF] * s;
                o[j * 4 + k] = __floats2half2_rn(v0, v1);
            }
        }
        int4* w = (int4*)&out[i * 16];
        w[0] = *(int4*)&o[0]; w[1] = *(int4*)&o[4]; w[2] = *(int4*)&o[8]; w[3] = *(int4*)&o[12];
    }
}
// d) quant fp16 -> int8 (satfinite)
__global__ void quant_int8(const int4* in, int4* out, long long nvec) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < nvec; i += stride) {
        int4 v = in[i];  // 8 half2 = 16 halves
        const __half2* p = (const __half2*)&v;
        signed char q[16];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float2 f = __half22float2(p[j]);
            q[2 * j] = (signed char)max(-128, min(127, __float2int_rn(f.x * 16.0f)));
            q[2 * j + 1] = (signed char)max(-128, min(127, __float2int_rn(f.y * 16.0f)));
        }
        out[i] = *(int4*)q;
    }
}
// e) quant fp16 -> e4m3
__global__ void quant_fp8(const int4* in, int4* out, long long nvec) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < nvec; i += stride) {
        int4 v = in[i];
        const __half2* p = (const __half2*)&v;
        unsigned short q[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float2 f = __half22float2(p[j]);
            q[j] = __nv_cvt_float2_to_fp8x2(make_float2(f.x, f.y), __NV_SATFINITE, __NV_E4M3);
        }
        out[i] = *(int4*)q;
    }
}
// f) quant fp16 -> e2m1 (manual: clamp to [-6,6], nearest of {0,.5,1,1.5,2,3,4,6})
__device__ __forceinline__ unsigned f2e2m1(float x) {
    unsigned s = (x < 0) ? 8 : 0;
    float a = fabsf(x);
    unsigned m = a < 0.25f ? 0 : a < 0.75f ? 1 : a < 1.25f ? 2 : a < 1.75f ? 3 : a < 2.5f ? 4 : a < 3.5f ? 5 : a < 5.0f ? 6 : 7;
    return s | m;
}
__global__ void quant_fp4(const int4* in, int2* out, long long nvec) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < nvec; i += stride) {
        int4 v = in[i];
        const __half2* p = (const __half2*)&v;
        unsigned w0 = 0, w1 = 0;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float2 f = __half22float2(p[j]);
            unsigned q0 = f2e2m1(f.x), q1 = f2e2m1(f.y);
            if (j < 4) w0 |= (q0 << (8 * j)) | (q1 << (8 * j + 4));
            else       w1 |= (q0 << (8 * (j - 4))) | (q1 << (8 * (j - 4) + 4));
        }
        out[i] = make_int2(w0, w1);
    }
}
// copy roofline: 16B in -> 16B out
__global__ void copy16(const int4* in, int4* out, long long nvec) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < nvec; i += stride) out[i] = in[i];
}

int main() {
    double freq = measure_clock_ghz();
    csv("clock", "sustained_sm_clock", freq, "GHz");
    int sm = num_sms();

    // ---- cvt throughput ----
    int blocks = sm * 8, threads = 256, iters = 20000;
    long long total = (long long)blocks * threads * 8LL * iters;
    auto runcvt = [&](void (*k)(int, unsigned*), const char* name, double elems_per_op) {
        unsigned* d; CK(cudaMalloc(&d, 4096));
        double ms = time_ms([&] { k<<<blocks, threads>>>(iters, d); }, 2, 7);
        csv("cvt_throughput", name, total / (ms * 1e-3) / (freq * 1e9) / sm, "ops/clk/SM");
        csv("cvt_rate", name, total * elems_per_op / (ms * 1e6), "Gelem/s");
        cudaFree(d);
    };
    runcvt(thrk_cvt_f32x2_f16x2, "f32x2->f16x2", 2);
    runcvt(thrk_cvt_f16x2_f32, "f16x2->f32x2", 2);
    runcvt(thrk_cvt_f32x2_e4m3x2, "f32x2->e4m3x2", 2);
    runcvt(thrk_cvt_e4m3x2_f16x2, "e4m3x2->f16x2", 2);
    runcvt(thrk_cvt_f32x2_bf16x2, "f32x2->bf16x2", 2);
    runcvt(thrk_fp4_unpack_soft, "fp4_soft_unpack_LUT", 8);
    runcvt(thrk_int8_unpack, "int8x2->f16x2", 2);

    // ---- memory-bound kernels ----
    long long N = 1LL << 28;  // 256M elements
    long long nvec16 = N / 16;
    int C = 8192;
    void *din, *dout; CK(cudaMalloc(&din, N * 2)); CK(cudaMalloc(&dout, N * 2));
    __half* dsc; CK(cudaMalloc(&dsc, C * 2)); CK(cudaMemset(dsc, 0x3c, C * 2));
    unsigned char* dsc8; CK(cudaMalloc(&dsc8, N / 8));
    CK(cudaMemset(din, 0x31, N * 2));
    int grid = sm * 8, blk = 256;
    auto runmem = [&](const char* name, auto k, double bytes_in, double bytes_out) {
        double ms = time_ms(k, 3, 10);
        double gbs = (bytes_in + bytes_out) / (ms * 1e6);
        csv("quant_kernel", name, gbs, "GB/s");
        csv("quant_kernel_elem", name, N / (ms * 1e6), "Gelem/s");
    };
    runmem("dequant_int8_fp16", [&] { deq_int8<<<(int)grid, blk>>>((const int4*)din, dsc, C, (__half2*)dout, nvec16); }, N * 1.0, N * 2.0);
    runmem("dequant_fp8_bf16",  [&] { deq_fp8<<<(int)grid, blk>>>((const int4*)din, dsc, C, (__nv_bfloat162*)dout, nvec16); }, N * 1.0, N * 2.0);
    runmem("dequant_fp4_fp16",  [&] { deq_fp4<<<(int)grid, blk>>>((const int4*)din, dsc8, (__half2*)dout, N / 32); }, N * 0.5, N * 2.0);
    runmem("quant_fp16_int8",   [&] { quant_int8<<<(int)grid, blk>>>((const int4*)din, (int4*)dout, nvec16); }, N * 2.0, N * 1.0);
    runmem("quant_fp16_fp8",    [&] { quant_fp8<<<(int)grid, blk>>>((const int4*)din, (int4*)dout, nvec16); }, N * 2.0, N * 1.0);
    runmem("quant_fp16_fp4",    [&] { quant_fp4<<<(int)grid, blk>>>((const int4*)din, (int2*)dout, nvec16); }, N * 2.0, N * 0.5);
    runmem("copy_roofline",     [&] { copy16<<<(int)grid, blk>>>((const int4*)din, (int4*)dout, N / 16); }, N * 1.0, N * 1.0);
    return 0;
}
