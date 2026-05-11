// SYCL Writeback Bandwidth Test
// Compare: scalar half vs vec<4, half> vs reinterpret_cast<uint32*>
// Build: icpx -fsycl -fsycl-targets=intel_gpu_bmg_g21 -O3 -std=c++17 -o bench_writeback_sycl bench_writeback_sycl.cpp

#include <sycl/sycl.hpp>
#include <stdio.h>

using namespace sycl;

// Approach 1: Scalar half store (bad)
void kernel_scalar_half(queue& q, size_t n, int repeats) {
    auto src = malloc_device<uint16_t>(n, q);
    auto dst = malloc_device<uint16_t>(n, q);

    q.fill(src, (uint16_t)0x3F00, n).wait();

    int total_threads = 4096 * 256;
    int n_iter = (n + total_threads - 1) / total_threads;

    auto ev = q.submit([&](handler& h) {
        h.parallel_for(nd_range<1>(4096*256, 256),
            [=](nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            int tid = it.get_global_id(0);
            for (int i = 0; i < n_iter; i++) {
                size_t idx = (size_t)tid + (size_t)i * total_threads;
                if (idx < n) dst[idx] = src[idx];
            }
        });
    });
    ev.wait();

    auto ns = ev.get_profiling_info<info::event_profiling::command_end>()
            - ev.get_profiling_info<info::event_profiling::command_start>();
    double copy_gb = 2.0 * n * sizeof(uint16_t) / 1e9;
    double bw = (2.0 * n * sizeof(uint16_t)) / (ns * 1e-9) / 1e9;
    printf("scalar_half:  %zu MB, %.0f ns, %.1f GB/s\n",
           n*sizeof(uint16_t)/1024/1024, (double)ns, bw);

    free(src, q); free(dst, q);
}

// Approach 2: vec<4, uint16_t> store (should be good)
void kernel_vec4_half(queue& q, size_t n, int repeats) {
    size_t n_vec = n / 4;
    auto src = malloc_device<uint16_t>(n, q);
    auto dst = malloc_device<uint16_t>(n, q);

    q.fill(src, (uint16_t)0x3F00, n).wait();

    using vec4u16 = vec<uint16_t, 4>;
    auto src4 = reinterpret_cast<vec4u16*>(src);
    auto dst4 = reinterpret_cast<vec4u16*>(dst);

    int total_threads = 4096 * 256;
    int n_iter = (n_vec + total_threads - 1) / total_threads;

    auto ev = q.submit([&](handler& h) {
        h.parallel_for(nd_range<1>(4096*256, 256),
            [=](nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            int tid = it.get_global_id(0);
            for (int i = 0; i < n_iter; i++) {
                size_t idx = (size_t)tid + (size_t)i * total_threads;
                if (idx < n_vec) dst4[idx] = src4[idx];
            }
        });
    });
    ev.wait();

    auto ns = ev.get_profiling_info<info::event_profiling::command_end>()
            - ev.get_profiling_info<info::event_profiling::command_start>();
    double copy_gb = 2.0 * n * sizeof(uint16_t) / 1e9;
    double bw = (2.0 * n * sizeof(uint16_t)) / (ns * 1e-9) / 1e9;
    printf("vec4_half:    %zu MB, %.0f ns, %.1f GB/s\n",
           n*sizeof(uint16_t)/1024/1024, (double)ns, bw);

    free(src, q); free(dst, q);
}

// Approach 3: Pack 2×uint16 into uint32
void kernel_pack_uint2(queue& q, size_t n, int repeats) {
    size_t n_pack = n / 2;
    auto src = malloc_device<uint16_t>(n, q);
    auto dst = malloc_device<uint16_t>(n, q);

    q.fill(src, (uint16_t)0x3F00, n).wait();

    auto src32 = reinterpret_cast<uint32_t*>(src);
    auto dst32 = reinterpret_cast<uint32_t*>(dst);

    int total_threads = 4096 * 256;
    int n_iter = (n_pack + total_threads - 1) / total_threads;

    auto ev = q.submit([&](handler& h) {
        h.parallel_for(nd_range<1>(4096*256, 256),
            [=](nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            int tid = it.get_global_id(0);
            for (int i = 0; i < n_iter; i++) {
                size_t idx = (size_t)tid + (size_t)i * total_threads;
                if (idx < n_pack) dst32[idx] = src32[idx];
            }
        });
    });
    ev.wait();

    auto ns = ev.get_profiling_info<info::event_profiling::command_end>()
            - ev.get_profiling_info<info::event_profiling::command_start>();
    double copy_gb = 2.0 * n * sizeof(uint16_t) / 1e9;
    double bw = (2.0 * n * sizeof(uint16_t)) / (ns * 1e-9) / 1e9;
    printf("pack_uint32:  %zu MB, %.0f ns, %.1f GB/s\n",
           n*sizeof(uint16_t)/1024/1024, (double)ns, bw);

    free(src, q); free(dst, q);
}

// Approach 4: FP32 baseline
void kernel_scalar_fp32(queue& q, size_t n, int repeats) {
    auto src = malloc_device<float>(n, q);
    auto dst = malloc_device<float>(n, q);

    q.fill(src, 1.0f, n).wait();

    int total_threads = 4096 * 256;
    int n_iter = (n + total_threads - 1) / total_threads;

    auto ev = q.submit([&](handler& h) {
        h.parallel_for(nd_range<1>(4096*256, 256),
            [=](nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            int tid = it.get_global_id(0);
            for (int i = 0; i < n_iter; i++) {
                size_t idx = (size_t)tid + (size_t)i * total_threads;
                if (idx < n) dst[idx] = src[idx];
            }
        });
    });
    ev.wait();

    auto ns = ev.get_profiling_info<info::event_profiling::command_end>()
            - ev.get_profiling_info<info::event_profiling::command_start>();
    double copy_gb = 2.0 * n * sizeof(float) / 1e9;
    double bw = (2.0 * n * sizeof(float)) / (ns * 1e-9) / 1e9;
    printf("fp32_scalar:  %zu MB, %.0f ns, %.1f GB/s\n",
           n*sizeof(float)/1024/1024, (double)ns, bw);

    free(src, q); free(dst, q);
}

int main() {
    queue q(gpu_selector_v, property::queue::enable_profiling{});
    auto dev = q.get_device();
    printf("Device: %s\n", dev.get_info<info::device::name>().c_str());

    size_t n = 128 * 1024 * 1024;

    printf("\n--- Writeback bandwidth: fp32 scalar vs bf16 approaches ---\n");
    printf("Elements: %zu, bf16 data: %zu MB, fp32 data: %zu MB\n\n",
           n, n*sizeof(uint16_t)/1024/1024, n*sizeof(float)/1024/1024);

    // Warmup
    { auto tmp = malloc_device<float>(1024, q); free(tmp, q); }

    for (int r = 0; r < 3; r++) {
        printf("=== Run %d ===\n", r+1);
        kernel_scalar_fp32(q, n, 1);
        kernel_scalar_half(q, n, 1);
        kernel_vec4_half(q, n, 1);
        kernel_pack_uint2(q, n, 1);
        printf("\n");
    }

    return 0;
}
