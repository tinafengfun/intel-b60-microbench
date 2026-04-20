// Compare host timing vs GPU event profiling for DPAS latency
// Build: icpx -fsycl -O2 -o bench_timing bench_timing.cpp

#include <sycl/sycl.hpp>
#include <cstdio>
#include <chrono>

int main() {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::enable_profiling{}};
    auto dev = q.get_device();

    printf("=== Host vs GPU Event Timing Comparison ===\n");
    printf("Device: %s\n", dev.get_info<sycl::info::device::name>().c_str());

    size_t n = 1024;
    float* src = sycl::malloc_device<float>(n, q);
    float* dst = sycl::malloc_device<float>(n, q);
    q.fill(src, 1.0f, n).wait();

    printf("\n%8s %14s %14s %8s\n", "Work", "Host(ms)", "GPU(ms)", "Ratio");
    printf("%8s %14s %14s %8s\n", "", "(chrono)", "(event)", "H/G");

    // Test 1: Very short kernel
    for (int size : {1024, 65536, 1048576, 16777216}) {
        int iters = 1024;
        double host_total = 0, gpu_total = 0;
        int n_rep = 50;

        for (int r = 0; r < n_rep; r++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto ev = q.submit([&](sycl::handler& h) {
                h.parallel_for(sycl::nd_range<1>(size, 256),
                    [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]]
                {
                    int tid = it.get_global_id(0);
                    float sum = 0;
                    for (int i = 0; i < iters; i++) {
                        sum += src[tid % n] * (float)i;
                    }
                    if (tid == 0) dst[0] = sum;
                });
            });
            ev.wait();
            auto t1 = std::chrono::high_resolution_clock::now();

            if (r >= 5) {
                double host_ns = (t1 - t0).count();
                auto g0 = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
                auto g1 = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
                double gpu_ns = (double)(g1 - g0);

                host_total += host_ns;
                gpu_total += gpu_ns;
            }
        }

        double host_ms = host_total / (n_rep - 5) / 1e6;
        double gpu_ms = gpu_total / (n_rep - 5) / 1e6;
        printf("%8d %14.4f %14.4f %8.3f\n", size, host_ms, gpu_ms, host_ms/gpu_ms);
    }

    // Test 2: DPAS-like workload using SYCL joint_matrix if available
    // Simple compute-heavy kernel to measure dispatch overhead
    printf("\n--- Dispatch Overhead (empty-ish kernel) ---\n");
    {
        double host_total = 0, gpu_total = 0;
        int n_rep = 100;

        for (int r = 0; r < n_rep; r++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto ev = q.submit([&](sycl::handler& h) {
                h.single_task([=]() {
                    float sum = 0;
                    for (int i = 0; i < 100; i++) sum += src[i];
                    dst[0] = sum;
                });
            });
            ev.wait();
            auto t1 = std::chrono::high_resolution_clock::now();

            if (r >= 10) {
                host_total += (t1 - t0).count();
                auto g0 = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
                auto g1 = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
                gpu_total += (double)(g1 - g0);
            }
        }

        double host_us = host_total / (n_rep - 10) / 1e3;
        double gpu_us = gpu_total / (n_rep - 10) / 1e3;
        printf("  Host: %.1f us  GPU: %.1f us  Overhead: %.1f us\n",
               host_us, gpu_us, host_us - gpu_us);
        printf("  Ratio: %.2f (host includes submit + wait + queue mgmt)\n", host_us/gpu_us);
    }

    sycl::free(src, q);
    sycl::free(dst, q);
    return 0;
}
