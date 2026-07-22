// Extra scheduling / latency probes inspired by the Intel B60 microbench report:
//  1) shared-memory pointer-chase latency (analog of B60 SLM latency)
//  2) __syncthreads() barrier overhead vs block size (B60 4b.2)
//  3) mma.sync dependent chain + independent FFMA interleave (B60 4b.1)
//  4) kernel launch / dispatch overhead (B60 4c.3)
#include "bench_common.h"

// ---------- 1) shared memory pointer chase ----------
__global__ void smem_chase_kernel(const int* perm, int n, int steps, long long* out) {
    extern __shared__ int s[];
    for (int i = threadIdx.x; i < n; i += blockDim.x) s[i] = perm[i];
    __syncthreads();
    if (threadIdx.x == 0) {
        int idx = 0;
        long long t0 = clock64();
        for (int i = 0; i < steps; i++) idx = s[idx];
        long long t1 = clock64();
        out[0] = t1 - t0;
        out[1] = idx;  // keep alive
    }
}

// Sattolo single-cycle permutation on host
static void make_cycle(int* p, int n) {
    for (int i = 0; i < n; i++) p[i] = i;
    srand(12345);
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % i;  // 0..i-1
        int t = p[i]; p[i] = p[j]; p[j] = t;
    }
}

// ---------- 2) barrier overhead ----------
__global__ void barrier_kernel(int iters, int with_barrier, long long* out) {
    long long t0 = clock64();
    for (int i = 0; i < iters; i++) {
        if (with_barrier) __syncthreads();
        else asm volatile("");
    }
    long long t1 = clock64();
    if (threadIdx.x == 0) out[blockIdx.x] = t1 - t0;
}

// ---------- 3) mma + FFMA interleave ----------
__global__ void mma_alu_kernel(int iters, int n_ffma, float pb, float pc, long long* out, float* sink) {
    unsigned a0 = threadIdx.x, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3;
    unsigned b0 = a0 + 4, b1 = a0 + 5;
    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    float f[32];
    float b = pb, c = pc;
    #pragma unroll
    for (int i = 0; i < 32; i++) f[i] = 0.5f + 0.01f * i;
    long long t0 = clock64();
    for (int i = 0; i < iters; i++) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
            : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        // independent FFMA work (fold-safe: b,c are runtime values)
        #pragma unroll
        for (int k = 0; k < 32; k++)
            if (k < n_ffma) { f[k] = fmaf(f[k], b, c); asm volatile("" : "+f"(f[k])); }
    }
    long long t1 = clock64();
    if (threadIdx.x == 0) out[blockIdx.x] = t1 - t0;
    float acc = c0 + c1 + c2 + c3;
    #pragma unroll
    for (int k = 0; k < 32; k++) acc += f[k];
    if (acc == -12345.0f) sink[0] = acc;
}

// ---------- 4) launch overhead ----------
__global__ void empty_kernel() {}

int main() {
    double freq = measure_clock_ghz();
    csv("clock", "sustained_sm_clock", freq, "GHz");
    int sm = num_sms();
    long long* d_ll; CK(cudaMalloc(&d_ll, sizeof(long long) * 1024));
    float* d_f; CK(cudaMalloc(&d_f, 4096));

    // ---- 1) shared memory chase latency ----
    int optin = 0;
    CK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    csv("smem_latency", "max_dynamic_shared_per_block", optin / 1024.0, "KB", "optin");
    CK(cudaFuncSetAttribute(smem_chase_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, optin));
    int steps = 8192;
    int sizes_kb[] = {1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 160, 192, 224};
    for (int si = 0; si < 13; si++) {
        int kb = sizes_kb[si];
        if (kb * 1024 > optin) break;
        int n = kb * 1024 / 4;
        int* h = (int*)malloc(n * sizeof(int));
        make_cycle(h, n);
        int* d_perm; CK(cudaMalloc(&d_perm, n * sizeof(int)));
        CK(cudaMemcpy(d_perm, h, n * sizeof(int), cudaMemcpyHostToDevice));
        double* ts = (double*)malloc(sizeof(double) * 15);
        for (int r = 0; r < 15; r++) {
            CK(cudaMemset(d_ll, 0, 16));
            smem_chase_kernel<<<1, 128, kb * 1024>>>(d_perm, n, steps, d_ll);
            CK(cudaDeviceSynchronize());
            long long h0; CK(cudaMemcpy(&h0, d_ll, 8, cudaMemcpyDeviceToHost));
            ts[r] = (double)h0 / steps;
        }
        char param[32]; snprintf(param, 32, "shared_%dKB", kb);
        csv("smem_latency", param, median(ts, 15), "cycles/access", "single-thread ptr chase in dynamic shared");
        cudaFree(d_perm);
        free(h); free(ts);
    }

    // ---- 2) barrier overhead ----
    int biters = 20000;
    for (int w = 1; w <= 32; w *= 2) {
        int threads = w * 32;
        barrier_kernel<<<1, threads>>>(biters, 1, d_ll);
        CK(cudaDeviceSynchronize());
        long long h0; CK(cudaMemcpy(&h0, d_ll, 8, cudaMemcpyDeviceToHost));
        double cyc_bar = (double)h0 / biters;
        barrier_kernel<<<1, threads>>>(biters, 0, d_ll);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(&h0, d_ll, 8, cudaMemcpyDeviceToHost));
        double cyc_loop = (double)h0 / biters;
        char param[32]; snprintf(param, 32, "warps_%d", w);
        csv("barrier_overhead", param, cyc_bar - cyc_loop, "cycles/barrier", "syncthreads minus empty loop");
    }

    // ---- 3) mma + FFMA interleave ----
    int miters = 20000;
    int ks[] = {0, 1, 2, 4, 8, 16, 32};
    for (int ki = 0; ki < 7; ki++) {
        int k = ks[ki];
        double* ts = (double*)malloc(sizeof(double) * 11);
        for (int r = 0; r < 11; r++) {
            CK(cudaMemset(d_ll, 0, 8));
            mma_alu_kernel<<<1, 32>>>(miters, k, 1.0000001f, 0.4999999f, d_ll, d_f);
            CK(cudaDeviceSynchronize());
            long long h0; CK(cudaMemcpy(&h0, d_ll, 8, cudaMemcpyDeviceToHost));
            ts[r] = (double)h0 / miters;
        }
        char param[32]; snprintf(param, 32, "ffma_per_mma_%d", k);
        csv("mma_alu_interleave", param, median(ts, 11), "cycles/iter", "dependent fp16 mma chain + K indep FFMA");
        free(ts);
    }

    // ---- 4) launch overhead ----
    {
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        empty_kernel<<<1, 32>>>();
        CK(cudaDeviceSynchronize());
        double* ts = (double*)malloc(sizeof(double) * 51);
        for (int r = 0; r < 51; r++) {
            cudaEventRecord(a);
            empty_kernel<<<1, 32>>>();
            cudaEventRecord(b);
            CK(cudaEventSynchronize(b));
            float ms; cudaEventElapsedTime(&ms, a, b);
            ts[r] = ms * 1000.0;  // us
        }
        csv("launch_overhead", "single_launch_us", median(ts, 51), "us", "event-timed empty kernel");
        free(ts);
        cudaEventRecord(a);
        for (int r = 0; r < 2000; r++) empty_kernel<<<1, 32>>>();
        cudaEventRecord(b);
        CK(cudaEventSynchronize(b));
        float ms; cudaEventElapsedTime(&ms, a, b);
        csv("launch_overhead", "back_to_back_per_launch_us", ms * 1000.0 / 2000.0, "us", "2000 launches amortized");
        cudaEventRecord(a);
        for (int r = 0; r < 200; r++) empty_kernel<<<sm * 32, 256>>>();
        cudaEventRecord(b);
        CK(cudaEventSynchronize(b));
        cudaEventElapsedTime(&ms, a, b);
        csv("launch_overhead", "full_gpu_empty_per_launch_us", ms * 1000.0 / 200.0, "us", "sm*32 blocks x 256 thr");
    }

    cudaFree(d_ll); cudaFree(d_f);
    return 0;
}
