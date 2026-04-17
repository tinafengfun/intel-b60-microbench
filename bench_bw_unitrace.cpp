// Single-run bandwidth for unitrace profiling
// Build: icpx -fsycl -O2 -o bench_bw_unitrace bench_bw_unitrace.cpp

#include <sycl/sycl.hpp>
#include <cstdio>

int main() {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::enable_profiling{}};
    auto dev = q.get_device();

    size_t n = 256UL * 1024 * 1024;  // 1 GB
    float* src = sycl::malloc_device<float>(n, q);
    float* dst = sycl::malloc_device<float>(n, q);
    q.fill(src, 1.0f, n).wait();
    q.fill(dst, 2.0f, n).wait();

    int n_wg = 4096, wg_size = 256;  // 1M threads

    // READ test
    {
        float* dummy = sycl::malloc_device<float>(1, q);
        int total_threads = n_wg * wg_size;
        int n_iter = (n + total_threads - 1) / total_threads;
        double total_ns = 0;
        for (int r = 0; r < 25; r++) {
            auto ev = q.submit([&](sycl::handler& h) {
                h.parallel_for(sycl::nd_range<1>(n_wg * wg_size, wg_size),
                    [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]]
                {
                    int tid = it.get_global_id(0);
                    float sum = 0;
                    for (int i = 0; i < n_iter; i++) {
                        size_t idx = (size_t)tid + (size_t)i * total_threads;
                        if (idx < n) sum += src[idx];
                    }
                    if (tid == 0) *dummy = sum;
                });
            });
            ev.wait();
            if (r >= 5) {
                auto t0 = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
                auto t1 = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
                total_ns += (t1 - t0);
            }
        }
        printf("READ:  %.1f GB/s\n", (double)n*4 / (total_ns/20 * 1e-9) / 1e9);
        sycl::free(dummy, q);
    }

    // WRITE test
    {
        float* dummy = sycl::malloc_device<float>(1, q);
        int total_threads = n_wg * wg_size;
        int n_iter = (n + total_threads - 1) / total_threads;
        double total_ns = 0;
        for (int r = 0; r < 25; r++) {
            auto ev = q.submit([&](sycl::handler& h) {
                h.parallel_for(sycl::nd_range<1>(n_wg * wg_size, wg_size),
                    [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]]
                {
                    int tid = it.get_global_id(0);
                    float val = (float)tid;
                    for (int i = 0; i < n_iter; i++) {
                        size_t idx = (size_t)tid + (size_t)i * total_threads;
                        if (idx < n) dst[idx] = val + i;
                    }
                    if (tid == 0) *dummy = val;
                });
            });
            ev.wait();
            if (r >= 5) {
                auto t0 = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
                auto t1 = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
                total_ns += (t1 - t0);
            }
        }
        printf("WRITE: %.1f GB/s\n", (double)n*4 / (total_ns/20 * 1e-9) / 1e9);
        sycl::free(dummy, q);
    }

    // COPY test (read + write)
    {
        int total_threads = n_wg * wg_size;
        int n_iter = (n + total_threads - 1) / total_threads;
        double total_ns = 0;
        for (int r = 0; r < 25; r++) {
            auto ev = q.submit([&](sycl::handler& h) {
                h.parallel_for(sycl::nd_range<1>(n_wg * wg_size, wg_size),
                    [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]]
                {
                    int tid = it.get_global_id(0);
                    for (int i = 0; i < n_iter; i++) {
                        size_t idx = (size_t)tid + (size_t)i * total_threads;
                        if (idx < n) dst[idx] = src[idx];
                    }
                });
            });
            ev.wait();
            if (r >= 5) {
                auto t0 = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
                auto t1 = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
                total_ns += (t1 - t0);
            }
        }
        printf("COPY:  %.1f GB/s (2x bytes)\n", 2.0*(double)n*4 / (total_ns/20 * 1e-9) / 1e9);
    }

    sycl::free(src, q);
    sycl::free(dst, q);
    return 0;
}
