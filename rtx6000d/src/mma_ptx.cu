// Raw mma.sync PTX throughput & latency for all precisions supported by sm_120a
#include "bench_common.h"

// ---------------- device mma wrappers ----------------
struct F4 { float x, y, z, w; };
struct I4 { int x, y, z, w; };

__device__ __forceinline__ F4 mma_f16(F4 d, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(d.x), "+f"(d.y), "+f"(d.z), "+f"(d.w)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    return d;
}
__device__ __forceinline__ F4 mma_bf16(F4 d, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(d.x), "+f"(d.y), "+f"(d.z), "+f"(d.w)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    return d;
}
__device__ __forceinline__ F4 mma_tf32(F4 d, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(d.x), "+f"(d.y), "+f"(d.z), "+f"(d.w)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    return d;
}
__device__ __forceinline__ I4 mma_s8(I4 d, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+r"(d.x), "+r"(d.y), "+r"(d.z), "+r"(d.w)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    return d;
}
__device__ __forceinline__ F4 mma_e4m3(F4 d, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(d.x), "+f"(d.y), "+f"(d.z), "+f"(d.w)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    return d;
}
__device__ __forceinline__ F4 mma_e5m2(F4 d, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e5m2.e5m2.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(d.x), "+f"(d.y), "+f"(d.z), "+f"(d.w)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    return d;
}
// FP4 e2m1: m16n8k32, ptxas-accepted vector: A = 4 x .b32, B = 2 x .b32 (as f8f6f4 family)
__device__ __forceinline__ F4 mma_e2m1(F4 d, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(d.x), "+f"(d.y), "+f"(d.z), "+f"(d.w)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    return d;
}
__device__ __forceinline__ int dp4a_op(int c, int a, int b) { return __dp4a(a, b, c); }

// ---------------- generic throughput kernel ----------------
template <typename T>
struct MmaOp;
template <> struct MmaOp<F4> {};

typedef F4 (*mmaf_t)(F4, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned);

template <int W, int K>
__global__ void thr_f4(int iters, float* out) {
    F4 acc[K];
    #pragma unroll
    for (int j = 0; j < K; j++) acc[j] = {0.f, 0.f, 0.f, 0.f};
    unsigned a0 = threadIdx.x, a1 = threadIdx.x + 1, a2 = threadIdx.x + 2, a3 = threadIdx.x + 3;
    unsigned b0 = threadIdx.x + 4, b1 = threadIdx.x + 5;
    for (int i = 0; i < iters; i++) {
        #pragma unroll
        for (int j = 0; j < K; j++) {
            if constexpr (W == 0) acc[j] = mma_f16(acc[j], a0, a1, a2, a3, b0, b1);
            else if constexpr (W == 1) acc[j] = mma_bf16(acc[j], a0, a1, a2, a3, b0, b1);
            else if constexpr (W == 2) acc[j] = mma_tf32(acc[j], a0, a1, a2, a3, b0, b1);
            else if constexpr (W == 3) acc[j] = mma_e4m3(acc[j], a0, a1, a2, a3, b0, b1);
            else if constexpr (W == 4) acc[j] = mma_e5m2(acc[j], a0, a1, a2, a3, b0, b1);
            else acc[j] = mma_e2m1(acc[j], a0, a1, a2, a3, b0, b1);
        }
    }
    float s = 0;
    #pragma unroll
    for (int j = 0; j < K; j++) s += acc[j].x + acc[j].y + acc[j].z + acc[j].w;
    if (s == -12345.0f) out[0] = s;
}

__global__ void thr_s8(int iters, int* out) {
    I4 acc[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) acc[j] = {0, 0, 0, 0};
    unsigned a0 = threadIdx.x, a1 = threadIdx.x + 1, a2 = threadIdx.x + 2, a3 = threadIdx.x + 3;
    unsigned b0 = threadIdx.x + 4, b1 = threadIdx.x + 5;
    for (int i = 0; i < iters; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) acc[j] = mma_s8(acc[j], a0, a1, a2, a3, b0, b1);
    }
    int s = 0;
    #pragma unroll
    for (int j = 0; j < 8; j++) s += acc[j].x + acc[j].y + acc[j].z + acc[j].w;
    if (s == -12345) out[0] = s;
}

__global__ void thr_dp4a(int iters, int* out) {
    int acc[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) acc[j] = j;
    int a = threadIdx.x * 3 + 1, b = threadIdx.x * 5 + 2;
    for (int i = 0; i < iters; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) acc[j] = __dp4a(a, b, acc[j]);
    }
    int s = 0;
    #pragma unroll
    for (int j = 0; j < 8; j++) s ^= acc[j];
    if (s == -12345) out[0] = s;
}

// ---------------- latency kernels (dependent accumulator) ----------------
template <int W>
__global__ void lat_f4(int iters, double* out) {
    F4 acc = {0.f, 0.f, 0.f, 0.f};
    unsigned a0 = threadIdx.x, a1 = 1, a2 = 2, a3 = 3, b0 = 4, b1 = 5;
    long long s = clock64();
    for (int i = 0; i < iters; i++) {
        if constexpr (W == 0) acc = mma_f16(acc, a0, a1, a2, a3, b0, b1);
        else if constexpr (W == 1) acc = mma_bf16(acc, a0, a1, a2, a3, b0, b1);
        else if constexpr (W == 2) acc = mma_tf32(acc, a0, a1, a2, a3, b0, b1);
        else if constexpr (W == 3) acc = mma_e4m3(acc, a0, a1, a2, a3, b0, b1);
        else if constexpr (W == 4) acc = mma_e5m2(acc, a0, a1, a2, a3, b0, b1);
        else acc = mma_e2m1(acc, a0, a1, a2, a3, b0, b1);
    }
    long long e = clock64();
    if (threadIdx.x == 0) { out[0] = (double)(e - s) / iters; out[1] = acc.x; }
}
__global__ void lat_s8(int iters, double* out) {
    I4 acc = {0, 0, 0, 0};
    unsigned a0 = threadIdx.x, a1 = 1, a2 = 2, a3 = 3, b0 = 4, b1 = 5;
    long long s = clock64();
    for (int i = 0; i < iters; i++) acc = mma_s8(acc, a0, a1, a2, a3, b0, b1);
    long long e = clock64();
    if (threadIdx.x == 0) { out[0] = (double)(e - s) / iters; out[1] = acc.x; }
}
__global__ void lat_dp4a(int iters, double* out) {
    int acc = 0, a = threadIdx.x + 1, b = threadIdx.x + 2;
    long long s = clock64();
    for (int i = 0; i < iters; i++) acc = __dp4a(a, b, acc);
    long long e = clock64();
    if (threadIdx.x == 0) { out[0] = (double)(e - s) / iters; out[1] = acc; }
}

// ---------------- driver ----------------
static const char* NAMES[6] = {"fp16_m16n8k16", "bf16_m16n8k16", "tf32_m16n8k8",
                               "fp8e4m3_m16n8k32", "fp8e5m2_m16n8k32", "fp4e2m1_m16n8k32"};
static const double FLOPS_MMA[6] = {4096.0, 4096.0, 2048.0, 8192.0, 8192.0, 8192.0};

int main() {
    double freq = measure_clock_ghz();
    csv("clock", "sustained_sm_clock", freq, "GHz");
    int sm = num_sms();
    float* d; CK(cudaMalloc(&d, 4096));
    int* di; CK(cudaMalloc(&di, 4096));
    double* dd; CK(cudaMalloc(&dd, 4096));

    int blocks = sm * 4, threads = 256, iters = 20000;  // 32 warps/SM
    auto run_w = [&](auto kfn_thr, auto kfn_lat, int w) {
        double ms = time_ms([&] { kfn_thr<<<blocks, threads>>>(iters, d); }, 2, 7);
        CK(cudaGetLastError());
        double total_mma = (double)blocks * (threads / 32) * 8.0 * iters;
        double tflops = total_mma * FLOPS_MMA[w] / (ms * 1e-3) / 1e12;
        double mma_clk_sm = total_mma / (ms * 1e-3) / (freq * 1e9) / sm;
        csv("mma_throughput", NAMES[w], tflops, "TFLOPS", "mma.sync ILP=8");
        csv("mma_inst_rate", NAMES[w], mma_clk_sm, "mma/clk/SM");
        kfn_lat<<<1, 32>>>(100000, dd);
        CK(cudaDeviceSynchronize());
        double h[2]; CK(cudaMemcpy(h, dd, 16, cudaMemcpyDeviceToHost));
        csv("mma_latency", NAMES[w], h[0], "cycles", "dependent accumulator");
    };
    run_w(thr_f4<0, 8>, lat_f4<0>, 0);
    run_w(thr_f4<1, 8>, lat_f4<1>, 1);
    run_w(thr_f4<2, 8>, lat_f4<2>, 2);
    run_w(thr_f4<3, 8>, lat_f4<3>, 3);
    run_w(thr_f4<4, 8>, lat_f4<4>, 4);
    run_w(thr_f4<5, 8>, lat_f4<5>, 5);
    // int8 mma
    {
        double ms = time_ms([&] { thr_s8<<<blocks, threads>>>(iters, di); }, 2, 7);
        double total_mma = (double)blocks * (threads / 32) * 8.0 * iters;
        double tops = total_mma * 8192.0 / (ms * 1e-3) / 1e12;
        csv("mma_throughput", "int8_m16n8k32", tops, "TOPS", "mma.sync ILP=8");
        lat_s8<<<1, 32>>>(100000, dd);
        CK(cudaDeviceSynchronize());
        double h[2]; CK(cudaMemcpy(h, dd, 16, cudaMemcpyDeviceToHost));
        csv("mma_latency", "int8_m16n8k32", h[0], "cycles");
    }
    // dp4a
    {
        double ms = time_ms([&] { thr_dp4a<<<blocks, threads>>>(iters, di); }, 2, 7);
        double total = (double)blocks * threads * 8.0 * iters;
        csv("dp4a_throughput", "dp4a_s8", total / (ms * 1e6), "Ginst/s");
        csv("dp4a_rate", "dp4a_s8", total / (ms * 1e-3) / (freq * 1e9) / sm, "inst/clk/SM");
        lat_dp4a<<<1, 32>>>(100000, dd);
        CK(cudaDeviceSynchronize());
        double h[2]; CK(cudaMemcpy(h, dd, 16, cudaMemcpyDeviceToHost));
        csv("dp4a_latency", "dp4a_s8", h[0], "cycles");
    }
    return 0;
}
