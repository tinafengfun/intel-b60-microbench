// Instruction latency & throughput microbenchmarks (dependent-chain / independent-ops)
#include "bench_common.h"
#include <cuda_fp16.h>
#include <cmath>

// runtime operands (compiler cannot fold constant-memory loads)
__constant__ float BC_F[2] = {1.000001f, 0.9999995f};
__constant__ int BC_I[2] = {999331, 7717};
__constant__ double BC_D[2] = {1.000001, 0.9999995};

// ---------------- latency kernels: dependent chain, cycles/op ----------------
#define LAT_F(NAME, BODY) \
__global__ void latk_##NAME(int iters, double* out) { \
    float a = 0.5f + threadIdx.x * 1e-6f, b = BC_F[0], c = BC_F[1]; \
    long long s = clock64(); \
    for (int i = 0; i < iters; i++) { BODY; asm volatile("" : "+f"(a)); } \
    long long e = clock64(); \
    if (threadIdx.x == 0) { out[blockIdx.x * 2] = (double)(e - s) / iters; out[blockIdx.x * 2 + 1] = a; } \
}
#define LAT_I(NAME, BODY) \
__global__ void latk_##NAME(int iters, double* out) { \
    int a = 1234567 + threadIdx.x, b = BC_I[0], c = BC_I[1]; \
    long long s = clock64(); \
    for (int i = 0; i < iters; i++) { BODY; asm volatile("" : "+r"(a)); } \
    long long e = clock64(); \
    if (threadIdx.x == 0) { out[blockIdx.x * 2] = (double)(e - s) / iters; out[blockIdx.x * 2 + 1] = a; } \
}
#define LAT_D(NAME, BODY) \
__global__ void latk_##NAME(int iters, double* out) { \
    double a = 0.5 + threadIdx.x * 1e-9, b = BC_D[0], c = BC_D[1]; \
    long long s = clock64(); \
    for (int i = 0; i < iters; i++) { BODY; asm volatile("" : "+d"(a)); } \
    long long e = clock64(); \
    if (threadIdx.x == 0) { out[blockIdx.x * 2] = (double)(e - s) / iters; out[blockIdx.x * 2 + 1] = a; } \
}

LAT_F(fadd, a = a + b)
LAT_F(fmul, a = a * b)
LAT_F(ffma, a = fmaf(a, b, c))
LAT_F(fmul2, a = a * b * c)   // 2 dependent MULs? compiler may fuse to FFMA-ish; reference
LAT_F(fsetp_sel, a = (a > b) ? c : a)
LAT_F(mufu_sin, a = __sinf(a))
LAT_F(mufu_rcp, a = __fdividef(1.0f, a + 0.3f))
LAT_F(mufu_rsqrt, a = rsqrtf(fabsf(a) + 1e-3f))
LAT_F(mufu_lg2, a = __logf(fabsf(a) + 1.0f))
LAT_F(mufu_ex2, a = __expf(-fabsf(a)))
LAT_F(f2i_iadd, { int t = __float2int_rn(a + b); a = (float)(t & 0xFFFF) * 1e-5f; })
LAT_I(iadd3, a = a + b + c)
LAT_I(imad, a = a * b + c)
LAT_I(lop3, a = (a & b) ^ c)
LAT_I(shf, a = (int)__funnelshift_lc(a, b, 5) + c)
LAT_I(popc, a = __popc(a) + (a ^ b))
LAT_I(i2f_fadd, { float t = __int2float_rn(a & 0xFFFF); a = (int)(t * 100.0f) + c; })
LAT_D(dfma, a = fma(a, b, c))

__global__ void latk_hfma2(int iters, double* out) {
    __half2 a = __floats2half2_rn(0.5f, 0.5f), b = __floats2half2_rn(1.0001f, 1.0001f), c = __floats2half2_rn(0.9999f, 0.9999f);
    long long s = clock64();
    for (int i = 0; i < iters; i++) a = __hfma2(a, b, c);
    long long e = clock64();
    if (threadIdx.x == 0) { out[blockIdx.x * 2] = (double)(e - s) / iters; out[blockIdx.x * 2 + 1] = __low2float(a); }
}
__global__ void latk_empty(int iters, double* out) {  // clock64 read overhead + loop
    long long s = clock64();
    for (int i = 0; i < iters; i++) {}
    long long e = clock64();
    if (threadIdx.x == 0) out[blockIdx.x * 2] = (double)(e - s) / iters;
}

// ---------------- throughput kernels: 8 independent chains ----------------
__device__ __forceinline__ float op_fadd(float x, float b, float c) { return x + b; }
__device__ __forceinline__ float op_fmul(float x, float b, float c) { return x * b; }
__device__ __forceinline__ float op_ffma(float x, float b, float c) { return fmaf(x, b, c); }
__device__ __forceinline__ float op_fsetp_sel(float x, float b, float c) { return (x > b) ? c : x; }
__device__ __forceinline__ float op_mufu_sin(float x, float b, float c) { return __sinf(x); }
__device__ __forceinline__ float op_mufu_rcp(float x, float b, float c) { return __fdividef(1.0f, x + 0.3f); }
__device__ __forceinline__ float op_mufu_rsqrt(float x, float b, float c) { return rsqrtf(fabsf(x) + 1e-3f); }
__device__ __forceinline__ float op_mufu_lg2(float x, float b, float c) { return __logf(fabsf(x) + 1.0f); }
__device__ __forceinline__ float op_mufu_ex2(float x, float b, float c) { return __expf(-fabsf(x)); }
__device__ __forceinline__ float op_f2i(float x, float b, float c) { return (float)(__float2int_rn(x) & 0xFF) * 1e-3f; }
__device__ __forceinline__ int op_iadd3(int x, int b, int c) { return x + b + c; }
__device__ __forceinline__ int op_imad(int x, int b, int c) { return x * b + c; }
__device__ __forceinline__ int op_lop3(int x, int b, int c) { return (x & b) ^ c; }
__device__ __forceinline__ int op_shf(int x, int b, int c) { return (int)__funnelshift_lc(x, b, 5) + c; }
__device__ __forceinline__ int op_popc(int x, int b, int c) { return __popc(x) + (x ^ b); }
__device__ __forceinline__ int op_i2f(int x, int b, int c) { return (int)(__int2float_rn(x & 0xFFFF) * 100.0f) + c; }

#define THR_F(NAME, FUNC) \
__global__ void thrk_##NAME(int iters, float* out) { \
    float a0=0.5f,a1=0.51f,a2=0.52f,a3=0.53f,a4=0.54f,a5=0.55f,a6=0.56f,a7=0.57f; \
    float b = BC_F[0], c = BC_F[1]; \
    for (int i = 0; i < iters; i++) { \
        a0 = FUNC(a0,b,c); asm volatile("" : "+f"(a0)); \
        a1 = FUNC(a1,b,c); asm volatile("" : "+f"(a1)); \
        a2 = FUNC(a2,b,c); asm volatile("" : "+f"(a2)); \
        a3 = FUNC(a3,b,c); asm volatile("" : "+f"(a3)); \
        a4 = FUNC(a4,b,c); asm volatile("" : "+f"(a4)); \
        a5 = FUNC(a5,b,c); asm volatile("" : "+f"(a5)); \
        a6 = FUNC(a6,b,c); asm volatile("" : "+f"(a6)); \
        a7 = FUNC(a7,b,c); asm volatile("" : "+f"(a7)); } \
    float s = a0+a1+a2+a3+a4+a5+a6+a7; \
    if (s == -12345.0f) out[0] = s; \
}
#define THR_I(NAME, FUNC) \
__global__ void thrk_##NAME(int iters, int* out) { \
    int a0=123457,a1=223457,a2=323457,a3=423457,a4=523457,a5=623457,a6=723457,a7=823457; \
    int b = BC_I[0], c = BC_I[1]; \
    for (int i = 0; i < iters; i++) { \
        a0 = FUNC(a0,b,c); asm volatile("" : "+r"(a0)); \
        a1 = FUNC(a1,b,c); asm volatile("" : "+r"(a1)); \
        a2 = FUNC(a2,b,c); asm volatile("" : "+r"(a2)); \
        a3 = FUNC(a3,b,c); asm volatile("" : "+r"(a3)); \
        a4 = FUNC(a4,b,c); asm volatile("" : "+r"(a4)); \
        a5 = FUNC(a5,b,c); asm volatile("" : "+r"(a5)); \
        a6 = FUNC(a6,b,c); asm volatile("" : "+r"(a6)); \
        a7 = FUNC(a7,b,c); asm volatile("" : "+r"(a7)); } \
    int s = a0^a1^a2^a3^a4^a5^a6^a7; \
    if (s == -12345) out[0] = s; \
}

THR_F(fadd, op_fadd)
THR_F(fmul, op_fmul)
THR_F(ffma, op_ffma)
THR_F(fsetp_sel, op_fsetp_sel)
THR_F(mufu_sin, op_mufu_sin)
THR_F(mufu_rcp, op_mufu_rcp)
THR_F(mufu_rsqrt, op_mufu_rsqrt)
THR_F(mufu_lg2, op_mufu_lg2)
THR_F(mufu_ex2, op_mufu_ex2)
THR_F(f2i, op_f2i)
THR_I(iadd3, op_iadd3)
THR_I(imad, op_imad)
THR_I(lop3, op_lop3)
THR_I(shf, op_shf)
THR_I(popc, op_popc)
THR_I(i2f, op_i2f)

__global__ void thrk_hfma2(int iters, float* out) {
    __half2 a[8]; __half2 b = __floats2half2_rn(1.0001f, 1.0001f), c = __floats2half2_rn(0.0001f, 0.0001f);
    #pragma unroll
    for (int j = 0; j < 8; j++) a[j] = __floats2half2_rn(0.5f + j * 0.01f, 0.5f);
    for (int i = 0; i < iters; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) a[j] = __hfma2(a[j], b, c);
    }
    float s = 0;
    #pragma unroll
    for (int j = 0; j < 8; j++) s += __low2float(a[j]);
    if (s == -12345.0f) out[0] = s;
}
__global__ void thrk_dfma(int iters, double* out) {
    double a0=0.5,a1=0.51,a2=0.52,a3=0.53,a4=0.54,a5=0.55,a6=0.56,a7=0.57;
    double b = 1.0000001, c = 0.4999999;
    for (int i = 0; i < iters; i++) {
        a0=fma(a0,b,c); a1=fma(a1,b,c); a2=fma(a2,b,c); a3=fma(a3,b,c);
        a4=fma(a4,b,c); a5=fma(a5,b,c); a6=fma(a6,b,c); a7=fma(a7,b,c); }
    double s = a0+a1+a2+a3+a4+a5+a6+a7;
    if (s == -12345.0) out[0] = s;
}

// ---------------- driver ----------------
template <typename K>
static double run_lat(K k, const char* name) {
    double* d; CK(cudaMalloc(&d, 8 * 4096));
    int iters = 100000;
    k<<<1, 32>>>(iters, d);   // warmup
    CK(cudaDeviceSynchronize());
    double* h = (double*)malloc(8 * 4096);
    k<<<1, 32>>>(iters, d);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(h, d, 16, cudaMemcpyDeviceToHost));
    double cyc = h[0];
    csv("alu_latency", name, cyc, "cycles");
    free(h); cudaFree(d);
    return cyc;
}

template <typename K, typename P>
static void run_thr(K k, P* buf, const char* name, double freq_ghz, double flop_per_op = 1.0) {
    P* d;
    CK(cudaMalloc((void**)&d, 1 << 20));
    int sm = num_sms();
    int blocks = sm * 8, threads = 256, iters = 20000;
    long long total_ops = (long long)blocks * threads * 8LL * iters;
    k<<<blocks, threads>>>(iters, d);  // warmup
    CK(cudaDeviceSynchronize());
    double ms = time_ms([&] { k<<<blocks, threads>>>(iters, d); }, 2, 7);
    double ops_clk_sm = total_ops / (ms * 1e-3) / (freq_ghz * 1e9) / sm;
    csv("alu_throughput", name, ops_clk_sm, "ops/clk/SM");
    csv("alu_throughput_rate", name, total_ops / (ms * 1e9) * flop_per_op, flop_per_op > 1 ? "TFLOPS" : "TOPS");
    cudaFree(d);
}

int main() {
    double freq = measure_clock_ghz();
    csv("clock", "sustained_sm_clock_full_load", freq, "GHz");
    int sm = num_sms();

    printf("# ---- latency (dependent chain, cycles/op) ----\n");
    run_lat(latk_empty, "empty_loop_overhead");
    run_lat(latk_fadd, "FADD_f32");
    run_lat(latk_fmul, "FMUL_f32");
    run_lat(latk_ffma, "FFMA_f32");
    run_lat(latk_fmul2, "FMUL2dep_f32");
    run_lat(latk_fsetp_sel, "FSETP_SEL_f32");
    run_lat(latk_mufu_sin, "MUFU_sin");
    run_lat(latk_mufu_rcp, "MUFU_rcp");
    run_lat(latk_mufu_rsqrt, "MUFU_rsqrt");
    run_lat(latk_mufu_lg2, "MUFU_lg2");
    run_lat(latk_mufu_ex2, "MUFU_ex2");
    run_lat(latk_f2i_iadd, "F2I_chain");
    run_lat(latk_iadd3, "IADD3_i32");
    run_lat(latk_imad, "IMAD_i32");
    run_lat(latk_lop3, "LOP3_i32");
    run_lat(latk_shf, "SHF_i32");
    run_lat(latk_popc, "POPC_i32");
    run_lat(latk_i2f_fadd, "I2F_chain");
    run_lat(latk_dfma, "DFMA_f64");
    run_lat(latk_hfma2, "HFMA2_f16x2");

    printf("# ---- throughput (ops/clk/SM) ----\n");
    float* fb = nullptr; int* ib = nullptr; double* db = nullptr;
    run_thr(thrk_fadd, fb, "FADD_f32", freq);
    run_thr(thrk_fmul, fb, "FMUL_f32", freq);
    run_thr(thrk_ffma, fb, "FFMA_f32", freq, 2.0);
    run_thr(thrk_fsetp_sel, fb, "FSETP_SEL_f32", freq);
    run_thr(thrk_mufu_sin, fb, "MUFU_sin", freq);
    run_thr(thrk_mufu_rcp, fb, "MUFU_rcp", freq);
    run_thr(thrk_mufu_rsqrt, fb, "MUFU_rsqrt", freq);
    run_thr(thrk_mufu_lg2, fb, "MUFU_lg2", freq);
    run_thr(thrk_mufu_ex2, fb, "MUFU_ex2", freq);
    run_thr(thrk_f2i, fb, "F2I_f32_s32", freq);
    run_thr(thrk_iadd3, ib, "IADD3_i32", freq);
    run_thr(thrk_imad, ib, "IMAD_i32", freq);
    run_thr(thrk_lop3, ib, "LOP3_i32", freq);
    run_thr(thrk_shf, ib, "SHF_i32", freq);
    run_thr(thrk_popc, ib, "POPC_i32", freq);
    run_thr(thrk_i2f, ib, "I2F_s32_f32", freq);
    run_thr(thrk_hfma2, fb, "HFMA2_f16x2", freq, 4.0);
    run_thr(thrk_dfma, db, "DFMA_f64", freq, 2.0);
    printf("# SMs=%d freq=%.3f GHz\n", sm, freq);
    return 0;
}
