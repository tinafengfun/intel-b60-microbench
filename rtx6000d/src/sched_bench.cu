// Warp scheduling / issue-rate / occupancy / divergence microbenchmarks
#include "bench_common.h"

// fixed work per warp: ILP=4 independent FFMA chains
__global__ void issue_kernel(int iters, float* out) {
    float b = 1.0000001f, c = 0.4999999f;
    float a0 = 0.5f, a1 = 0.51f, a2 = 0.52f, a3 = 0.53f;
    for (int i = 0; i < iters; i++) {
        a0 = fmaf(a0, b, c); a1 = fmaf(a1, b, c);
        a2 = fmaf(a2, b, c); a3 = fmaf(a3, b, c);
    }
    if (a0 + a1 + a2 + a3 == -12345.0f) out[0] = 1.0f;
}

// single dependent chain per warp (latency-bound)
__global__ void dep_kernel(int iters, float* out) {
    float b = 1.0000001f, c = 0.4999999f, a = 0.5f;
    for (int i = 0; i < iters; i++) a = fmaf(a, b, c);
    if (a == -12345.0f) out[0] = 1.0f;
}

// divergence: both branches equal length
__global__ void div_kernel(int iters, int mode, float* out) {
    float b = 1.0000001f, c = 0.4999999f, a = 0.5f;
    int lane = threadIdx.x & 31;
    for (int i = 0; i < iters; i++) {
        if (mode == 0 || lane < 16) { a = fmaf(a, b, c); a = fmaf(a, b, c); }
        else                        { a = fmaf(a, c, b); a = fmaf(a, c, b); }
    }
    if (a == -12345.0f) out[0] = 1.0f;
}

// resident-block verification: count simultaneously resident blocks per SM
__global__ void resident_kernel(long long cycles, int* smid_count) {
    if (threadIdx.x == 0) {
        unsigned smid;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
        atomicAdd(&smid_count[smid], 1);
    }
    long long s = clock64();
    while (clock64() - s < cycles) {}
}

int main() {
    double freq = measure_clock_ghz();
    csv("clock", "sustained_sm_clock", freq, "GHz");
    int sm = num_sms();
    float* d; CK(cudaMalloc(&d, 4096));
    int iters = 50000;

    // 1) issue-rate sweep: warps per SM = 1..32 (one block of N warps per SM)
    printf("# issue sweep: warps/SM, FFMA/clk/SM (=IPC since 1 op each)\n");
    for (int w = 1; w <= 32; w *= 2) {
        int threads = w * 32;
        // enough blocks to give exactly `w` warps per SM
        double ms = time_ms([&] { issue_kernel<<<sm, threads>>>(iters, d); }, 2, 7);
        double total = (double)sm * threads * 4.0 * iters;  // 4 FFMA per iter per thread
        double per_clk_sm = total / (ms * 1e-3) / (freq * 1e9) / sm;
        char param[32]; snprintf(param, 32, "warps_per_SM_%d", w);
        csv("sched_issue", param, per_clk_sm, "FFMA/clk/SM", "ILP=4");
    }
    // finer sweep near knee
    for (int w = 3; w <= 12; w++) {
        int threads = w * 32;
        double ms = time_ms([&] { issue_kernel<<<sm, threads>>>(iters, d); }, 2, 7);
        double total = (double)sm * threads * 4.0 * iters;
        double per_clk_sm = total / (ms * 1e-3) / (freq * 1e9) / sm;
        char param[32]; snprintf(param, 32, "warps_per_SM_%d_fine", w);
        csv("sched_issue", param, per_clk_sm, "FFMA/clk/SM", "ILP=4");
    }

    // 2) latency hiding: dependent chain, warps 1..32
    printf("# dependent-chain sweep\n");
    for (int w = 1; w <= 32; w *= 2) {
        int threads = w * 32;
        double ms = time_ms([&] { dep_kernel<<<sm, threads>>>(iters, d); }, 2, 7);
        double total = (double)sm * threads * (double)iters;
        double per_clk_sm = total / (ms * 1e-3) / (freq * 1e9) / sm;
        char param[32]; snprintf(param, 32, "dep_warps_per_SM_%d", w);
        csv("sched_lathide", param, per_clk_sm, "FFMA/clk/SM", "ILP=1 dependent");
    }

    // 3) divergence cost
    double ms_uni = time_ms([&] { div_kernel<<<sm, 256>>>(iters, 0, d); }, 2, 7);
    double ms_div = time_ms([&] { div_kernel<<<sm, 256>>>(iters, 1, d); }, 2, 7);
    csv("sched_divergence", "uniform_ms", ms_uni, "ms");
    csv("sched_divergence", "divergent_ms", ms_div, "ms");
    csv("sched_divergence", "slowdown", ms_div / ms_uni, "ratio", "half-warp if/else");

    // 4) resident blocks/SM at runtime (1024-thread blocks)
    int* cnt; CK(cudaMalloc(&cnt, 4 * 1024)); CK(cudaMemset(cnt, 0, 4 * 1024));
    resident_kernel<<<sm * 16, 1024>>>(200000000LL, cnt);
    CK(cudaDeviceSynchronize());
    int h[256]; CK(cudaMemcpy(h, cnt, sizeof(h), cudaMemcpyDeviceToHost));
    int mx = 0;
    for (int i = 0; i < sm; i++) if (h[i] > mx) mx = h[i];
    csv("sched_occupancy", "resident_blocks_per_SM_1024t", mx, "blocks", "runtime-verified");
    csv("sched_occupancy", "resident_warps_per_SM", mx * 32, "warps");
    cudaFree(cnt); cudaFree(d);
    return 0;
}
