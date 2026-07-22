// Memory hierarchy: DRAM STREAM bw, pointer-chase latency, L1/L2/shared bw, atomics, TLB
#include "bench_common.h"
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

// ---------------- STREAM ----------------
__global__ void k_read(const float4* in, float* out, long long n) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    float acc = 0;
    for (; i < n; i += stride) { float4 v = in[i]; acc += v.x + v.y + v.z + v.w; }
    if (acc == -12345.0f) out[0] = acc;
}
__global__ void k_write(float4* out, long long n) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    float4 v = make_float4(1.f, 2.f, 3.f, 4.f);
    for (; i < n; i += stride) out[i] = v;
}
__global__ void k_copy(const float4* in, float4* out, long long n) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) out[i] = in[i];
}
__global__ void k_triad(const float4* a, const float4* b, float4* c, long long n) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        float4 x = a[i], y = b[i];
        c[i] = make_float4(x.x + 0.5f * y.x, x.y + 0.5f * y.y, x.z + 0.5f * y.z, x.w + 0.5f * y.w);
    }
}

// ---------------- pointer chase ----------------
__global__ void k_chase(const unsigned* next, long long steps, unsigned start, double* out, int use_cg) {
    unsigned idx = start;
    long long s = clock64();
    if (use_cg) {
        for (long long i = 0; i < steps; i++) idx = __ldcg(next + idx);
    } else {
        for (long long i = 0; i < steps; i++) idx = next[idx];
    }
    long long e = clock64();
    if (threadIdx.x == 0) { out[0] = (double)(e - s) / steps; out[1] = idx; }
}

// ---------------- L1/L2 bandwidth (read-only, cached) ----------------
__global__ void k_read_ws(const float4* in, float* out, long long n, int passes) {
    long long i = threadIdx.x;
    float acc = 0;
    for (int p = 0; p < passes; p++) {
        for (long long j = i; j < n; j += blockDim.x) {
            float4 v = __ldca(in + j);
            acc += v.x + v.y + v.z + v.w;
        }
    }
    if (acc == -12345.0f) out[0] = acc;
}
__global__ void k_read_ws_cg(const float4* in, float* out, long long n, int passes) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    float acc = 0;
    for (int p = 0; p < passes; p++) {
        for (long long j = i; j < n; j += stride) {
            float4 v = __ldcg(in + j);
            acc += v.x + v.y + v.z + v.w;
        }
    }
    if (acc == -12345.0f) out[0] = acc;
}

// ---------------- shared memory bandwidth ----------------
__global__ void k_shared_bw(float* out, int iters, int stride_mode) {
    extern __shared__ float4 sh[];
    int t = threadIdx.x;
    #pragma unroll
    for (int j = 0; j < 4; j++) sh[t * 4 / 64 + ((t * 4 + j) % 64)] = make_float4(1.f, 2.f, 3.f, 4.f);
    __syncthreads();
    float acc = 0;
    if (stride_mode == 0) {
        for (int i = 0; i < iters; i++) {
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                float4 v = sh[(t + j * 256 + i * 61) % 4096];
                acc += v.x + v.y + v.z + v.w;
            }
        }
    } else {  // bank-conflict pattern: stride 32 floats
        for (int i = 0; i < iters; i++) {
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                acc += ((float*)sh)[(t * 32 + j + i * 61) % 16384];
            }
        }
    }
    if (acc == -12345.0f) out[0] = acc;
}

// ---------------- atomics ----------------
__global__ void k_atomic(unsigned* addr, int iters, int contended, unsigned* sink) {
    unsigned v = threadIdx.x;
    for (int i = 0; i < iters; i++) {
        if (contended) atomicAdd(addr, v);
        else atomicAdd(addr + (blockIdx.x * blockDim.x + threadIdx.x) % 65536, v);
    }
    if (v == 0xFFFFFFFF) sink[0] = v;
}
__global__ void k_atomic_shared(unsigned* sink, int iters, int contended) {
    __shared__ unsigned sh[32];
    if (threadIdx.x < 32) sh[threadIdx.x] = 0;
    __syncthreads();
    unsigned v = threadIdx.x;
    for (int i = 0; i < iters; i++) {
        if (contended) atomicAdd(&sh[0], v);
        else atomicAdd(&sh[threadIdx.x % 32], v);
    }
    if (v == 0xFFFFFFFF) sink[0] = sh[0];
}

// ---------------- TLB stride sweep ----------------
__global__ void k_tlb(const unsigned* buf, long long stride_elems, int touches, int passes, unsigned seed, double* out) {
    unsigned idx = (seed * 2654435761u) % (unsigned)touches;
    long long s = clock64();
    for (int p = 0; p < passes; p++) {
        for (int i = 0; i < touches; i++) {
            idx = (idx * 1103515245u + 12345u) % (unsigned)touches;
            unsigned v = __ldcg(buf + (long long)idx * stride_elems);
            if (v == 0xDEADBEEF) out[1] = v;
        }
    }
    long long e = clock64();
    if (threadIdx.x == 0) out[0] = (double)(e - s) / ((double)touches * passes);
}

int main() {
    double freq = measure_clock_ghz();
    csv("clock", "sustained_sm_clock", freq, "GHz");
    int sm = num_sms();
    long long GB = 1LL << 30;

    // ---------- STREAM: 2 GB per buffer ----------
    long long n4 = (2 * GB) / 16;  // float4 count
    float4 *a, *b, *c; CK(cudaMalloc(&a, 2 * GB)); CK(cudaMalloc(&b, 2 * GB)); CK(cudaMalloc(&c, 2 * GB));
    float* sink; CK(cudaMalloc(&sink, 1024));
    int grid = sm * 8, blk = 256;
    double ms;
    ms = time_ms([&] { k_read<<<grid, blk>>>(a, sink, n4); }, 3, 10);
    csv("dram_stream", "read", 2 * GB / (ms * 1e6), "GB/s");
    ms = time_ms([&] { k_write<<<grid, blk>>>(a, n4); }, 3, 10);
    csv("dram_stream", "write", 2 * GB / (ms * 1e6), "GB/s");
    ms = time_ms([&] { k_copy<<<grid, blk>>>(a, b, n4); }, 3, 10);
    csv("dram_stream", "copy", 4 * GB / (ms * 1e6), "GB/s", "r+w");
    ms = time_ms([&] { k_triad<<<grid, blk>>>(a, b, c, n4); }, 3, 10);
    csv("dram_stream", "triad", 6 * GB / (ms * 1e6), "GB/s", "2r+1w");
    cudaFree(a); cudaFree(b); cudaFree(c);

    // ---------- pointer chase latency vs working set ----------
    double* dd; CK(cudaMalloc(&dd, 1024));
    std::vector<unsigned> sizes;
    for (long long kb = 8; kb <= 8LL * 1024 * 1024; kb *= 2) sizes.push_back((unsigned)(kb * 1024 / 4));  // elements
    for (unsigned n : sizes) {
        unsigned* dnext; CK(cudaMalloc(&dnext, (size_t)n * 4));
        std::vector<unsigned> h(n);
        std::iota(h.begin(), h.end(), 0u);
        std::mt19937 rng(42);
        std::shuffle(h.begin(), h.end(), rng);
        // make it a single cycle: h[i] holds next index; shuffle gives permutation (may have multiple cycles, fine)
        CK(cudaMemcpy(dnext, h.data(), (size_t)n * 4, cudaMemcpyHostToDevice));
        long long steps = n < (1 << 16) ? (1 << 22) : (n < (1 << 22) ? (1 << 21) : (1 << 20));
        k_chase<<<1, 1>>>(dnext, steps, 0, dd, 0);
        CK(cudaDeviceSynchronize());
        double o[2]; CK(cudaMemcpy(o, dd, 16, cudaMemcpyDeviceToHost));
        char param[64]; snprintf(param, 64, "ws_%zuKB", (size_t)n * 4 / 1024);
        csv("mem_latency", param, o[0], "cycles", "pointer-chase L1-enabled");
        csv("mem_latency_ns", param, o[0] / freq, "ns");
        // L2-only view for larger sizes
        if ((size_t)n * 4 >= (1 << 20)) {
            k_chase<<<1, 1>>>(dnext, steps, 0, dd, 1);
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(o, dd, 16, cudaMemcpyDeviceToHost));
            char p2[64]; snprintf(p2, 64, "ws_%zuKB_ldcg", (size_t)n * 4 / 1024);
            csv("mem_latency", p2, o[0], "cycles", "pointer-chase L2-only");
        }
        cudaFree(dnext);
    }

    // ---------- L1 / L2 bandwidth ----------
    long long l1_n4 = (256LL << 10) / 16;   // 256 KB per-SM working set (L1)
    CK(cudaMalloc(&a, 256 << 10));
    ms = time_ms([&] { k_read_ws<<<sm, 1024>>>(a, sink, l1_n4, 4000); }, 2, 5);
    // total bytes = passes * sm * ws_bytes ; each SM reads its own 256KB copy
    double l1_bytes = 4000.0 * sm * (256.0 * 1024);
    csv("cache_bw", "L1_read", l1_bytes / (ms * 1e6), "GB/s", "256KB per SM, ldca");
    cudaFree(a);
    long long l2_n4 = (64LL << 20) / 16;   // 64 MB in L2
    CK(cudaMalloc(&a, 64 << 20));
    ms = time_ms([&] { k_read_ws_cg<<<grid, blk>>>(a, sink, l2_n4, 100); }, 2, 5);
    double l2_bytes = 100.0 * (64.0 * 1024 * 1024);
    csv("cache_bw", "L2_read", l2_bytes / (ms * 1e6), "GB/s", "64MB ldcg");
    cudaFree(a);

    // ---------- shared memory bandwidth ----------
    cudaFuncSetAttribute(k_shared_bw, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
    ms = time_ms([&] { k_shared_bw<<<sm, 256, 65536>>>(sink, 30000, 0); }, 2, 5);
    double sh_bytes = (double)sm * 256 * 30000 * 16 * 16.0;
    csv("shared_bw", "no_conflict_read", sh_bytes / (ms * 1e6), "GB/s", "128-bit ld.shared");
    csv("shared_bw", "bytes_per_clk_SM", sh_bytes / (ms * 1e-3) / (freq * 1e9) / sm, "B/clk/SM");
    ms = time_ms([&] { k_shared_bw<<<sm, 256, 65536>>>(sink, 30000, 1); }, 2, 5);
    double sh_bytes2 = (double)sm * 256 * 30000 * 16 * 4.0;
    csv("shared_bw", "stride32_conflict_read", sh_bytes2 / (ms * 1e6), "GB/s", "32-bit ld.shared stride32");

    // ---------- atomics ----------
    unsigned* uaddr; CK(cudaMalloc(&uaddr, 65536 * 4)); CK(cudaMemset(uaddr, 0, 65536 * 4));
    ms = time_ms([&] { k_atomic<<<grid, blk>>>(uaddr, 2000, 1, uaddr + 65535); }, 2, 5);
    csv("atomics", "global_same_addr", (double)grid * blk * 2000 / (ms * 1e6), "Gops/s");
    ms = time_ms([&] { k_atomic<<<grid, blk>>>(uaddr, 2000, 0, uaddr + 65535); }, 2, 5);
    csv("atomics", "global_spread", (double)grid * blk * 2000 / (ms * 1e6), "Gops/s");
    ms = time_ms([&] { k_atomic_shared<<<grid, blk>>>(uaddr, 2000, 1); }, 2, 5);
    csv("atomics", "shared_same_addr", (double)grid * blk * 2000 / (ms * 1e6), "Gops/s");
    ms = time_ms([&] { k_atomic_shared<<<grid, blk>>>(uaddr, 2000, 0); }, 2, 5);
    csv("atomics", "shared_spread", (double)grid * blk * 2000 / (ms * 1e6), "Gops/s");

    // ---------- TLB stride sweep over 2 GB ----------
    CK(cudaMalloc(&a, 2 * GB));
    CK(cudaMemset(a, 1, 2 * GB));
    const unsigned* ub = (const unsigned*)a;
    for (long long stride_kb = 4; stride_kb <= 65536; stride_kb *= 4) {
        long long stride_elems = stride_kb * 1024 / 4;
        int touches = (int)((2 * GB) / (stride_elems * 4));
        if (touches > 65536) touches = 65536;
        k_tlb<<<1, 1>>>(ub, stride_elems, touches, 20, 7, dd);
        CK(cudaDeviceSynchronize());
        double o[2]; CK(cudaMemcpy(o, dd, 16, cudaMemcpyDeviceToHost));
        char param[64]; snprintf(param, 64, "stride_%lldKB", stride_kb);
        csv("tlb_latency", param, o[0], "cycles", "random touch ldcg");
        csv("tlb_pages", param, touches, "pages");
    }
    cudaFree(a);
    return 0;
}
