// Deeper tensor-core probe: acc16 vs acc32, ILP sweep, warp sweep, SASS-visible variants
#include "bench_common.h"

struct F4 { float x, y, z, w; };
struct H2 { unsigned x, y; };

__device__ __forceinline__ F4 mma_f16(F4 d, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(d.x), "+f"(d.y), "+f"(d.z), "+f"(d.w)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    return d;
}
// f16 accumulator variant (m16n8k16, D = 2 x f16x2 regs)
__device__ __forceinline__ H2 mma_f16_acc16(H2 d, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1},{%2,%3,%4,%5},{%6,%7},{%0,%1};"
        : "+r"(d.x), "+r"(d.y)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    return d;
}
__device__ __forceinline__ F4 mma_e4m3(F4 d, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(d.x), "+f"(d.y), "+f"(d.z), "+f"(d.w)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    return d;
}
__device__ __forceinline__ F4 mma_e2m1(F4 d, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(d.x), "+f"(d.y), "+f"(d.z), "+f"(d.w)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    return d;
}

template <int K>
__global__ void thr_f16(int iters, float* out) {
    F4 acc[K];
    #pragma unroll
    for (int j = 0; j < K; j++) acc[j] = {0.f, 0.f, 0.f, 0.f};
    unsigned a0 = threadIdx.x, a1 = threadIdx.x + 1, a2 = threadIdx.x + 2, a3 = threadIdx.x + 3;
    unsigned b0 = threadIdx.x + 4, b1 = threadIdx.x + 5;
    for (int i = 0; i < iters; i++) {
        #pragma unroll
        for (int j = 0; j < K; j++) acc[j] = mma_f16(acc[j], a0, a1, a2, a3, b0, b1);
    }
    float s = 0;
    #pragma unroll
    for (int j = 0; j < K; j++) s += acc[j].x + acc[j].y + acc[j].z + acc[j].w;
    if (s == -12345.0f) out[0] = s;
}
template <int K>
__global__ void thr_f16_acc16(int iters, float* out) {
    H2 acc[K];
    #pragma unroll
    for (int j = 0; j < K; j++) acc[j] = {0u, 0u};
    unsigned a0 = threadIdx.x, a1 = threadIdx.x + 1, a2 = threadIdx.x + 2, a3 = threadIdx.x + 3;
    unsigned b0 = threadIdx.x + 4, b1 = threadIdx.x + 5;
    for (int i = 0; i < iters; i++) {
        #pragma unroll
        for (int j = 0; j < K; j++) acc[j] = mma_f16_acc16(acc[j], a0, a1, a2, a3, b0, b1);
    }
    unsigned s = 0;
    #pragma unroll
    for (int j = 0; j < K; j++) s ^= acc[j].x ^ acc[j].y;
    if (s == 0xDEADBEEF) out[0] = 1.f;
}
template <int K>
__global__ void thr_e4m3(int iters, float* out) {
    F4 acc[K];
    #pragma unroll
    for (int j = 0; j < K; j++) acc[j] = {0.f, 0.f, 0.f, 0.f};
    unsigned a0 = threadIdx.x, a1 = threadIdx.x + 1, a2 = threadIdx.x + 2, a3 = threadIdx.x + 3;
    unsigned b0 = threadIdx.x + 4, b1 = threadIdx.x + 5;
    for (int i = 0; i < iters; i++) {
        #pragma unroll
        for (int j = 0; j < K; j++) acc[j] = mma_e4m3(acc[j], a0, a1, a2, a3, b0, b1);
    }
    float s = 0;
    #pragma unroll
    for (int j = 0; j < K; j++) s += acc[j].x + acc[j].y + acc[j].z + acc[j].w;
    if (s == -12345.0f) out[0] = s;
}
template <int K>
__global__ void thr_e2m1(int iters, float* out) {
    F4 acc[K];
    #pragma unroll
    for (int j = 0; j < K; j++) acc[j] = {0.f, 0.f, 0.f, 0.f};
    unsigned a0 = threadIdx.x, a1 = threadIdx.x + 1, a2 = threadIdx.x + 2, a3 = threadIdx.x + 3;
    unsigned b0 = threadIdx.x + 4, b1 = threadIdx.x + 5;
    for (int i = 0; i < iters; i++) {
        #pragma unroll
        for (int j = 0; j < K; j++) acc[j] = mma_e2m1(acc[j], a0, a1, a2, a3, b0, b1);
    }
    float s = 0;
    #pragma unroll
    for (int j = 0; j < K; j++) s += acc[j].x + acc[j].y + acc[j].z + acc[j].w;
    if (s == -12345.0f) out[0] = s;
}

int main() {
    double freq = measure_clock_ghz();
    csv("clock", "sustained_sm_clock", freq, "GHz");
    int sm = num_sms();
    float* d; CK(cudaMalloc(&d, 4096));
    int iters = 20000;

    // warp sweep at ILP=8 for fp16 acc32
    for (int wsm : {8, 16, 32, 48}) {
        int threads = 256;
        int blocks = sm * (wsm / 8);  // 256 thr = 8 warps per block
        double ms = time_ms([&] { thr_f16<8><<<blocks, threads>>>(iters, d); }, 2, 7);
        double total = (double)blocks * 8 * 8.0 * iters;
        char p[64]; snprintf(p, 64, "warps_%d", wsm);
        csv("mma_probe_fp16", p, total * 4096.0 / (ms * 1e-3) / 1e12, "TFLOPS", "ILP=8");
    }
    // ILP sweep at 32 warps/SM
    {
        int blocks = sm * 4, threads = 256;
        double ms4 = time_ms([&] { thr_f16<4><<<blocks, threads>>>(iters, d); }, 2, 7);
        double ms8 = time_ms([&] { thr_f16<8><<<blocks, threads>>>(iters, d); }, 2, 7);
        double ms16 = time_ms([&] { thr_f16<16><<<blocks, threads>>>(iters, d); }, 2, 7);
        double tot = (double)blocks * 8 * iters;
        csv("mma_probe_fp16", "ILP_4", tot * 4 * 4096.0 / (ms4 * 1e-3) / 1e12, "TFLOPS", "32 warps/SM");
        csv("mma_probe_fp16", "ILP_8", tot * 8 * 4096.0 / (ms8 * 1e-3) / 1e12, "TFLOPS", "32 warps/SM");
        csv("mma_probe_fp16", "ILP_16", tot * 16 * 4096.0 / (ms16 * 1e-3) / 1e12, "TFLOPS", "32 warps/SM");
        // fp16 acc16
        double msA8 = time_ms([&] { thr_f16_acc16<8><<<blocks, threads>>>(iters, d); }, 2, 7);
        double msA16 = time_ms([&] { thr_f16_acc16<16><<<blocks, threads>>>(iters, d); }, 2, 7);
        csv("mma_probe_fp16acc16", "ILP_8", tot * 8 * 4096.0 / (msA8 * 1e-3) / 1e12, "TFLOPS", "f16 accumulator");
        csv("mma_probe_fp16acc16", "ILP_16", tot * 16 * 4096.0 / (msA16 * 1e-3) / 1e12, "TFLOPS", "f16 accumulator");
        // fp8 / fp4 at ILP16
        double msE = time_ms([&] { thr_e4m3<16><<<blocks, threads>>>(iters, d); }, 2, 7);
        csv("mma_probe_fp8e4m3", "ILP_16", tot * 16 * 8192.0 / (msE * 1e-3) / 1e12, "TFLOPS");
        double ms4f = time_ms([&] { thr_e2m1<16><<<blocks, threads>>>(iters, d); }, 2, 7);
        csv("mma_probe_fp4e2m1", "ILP_16", tot * 16 * 8192.0 / (ms4f * 1e-3) / 1e12, "TFLOPS");
    }
    return 0;
}
