// SLM (Shared Local Memory) Latency Benchmark
// Measures pointer chase within Workgroup-local memory (SLM)
// Expected: ~20-30 cycles (vs L1 cache ~70-145 cycles)
//
// Build: icpx -fsycl -O2 -o bench_slm_latency bench_slm_latency.cpp
// Usage: ./bench_slm_latency

#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

int main() {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::enable_profiling{}};
    auto dev = q.get_device();
    double ghz = dev.get_info<sycl::info::device::max_clock_frequency>() / 1000.0;

    printf("=== SLM (Shared Local Memory) Latency ===\n");
    printf("Device: %s  Clock: %.1f GHz\n",
           dev.get_info<sycl::info::device::name>().c_str(), ghz);
    printf("Local memory size: %zu bytes\n\n",
           dev.get_info<sycl::info::device::local_mem_size>());

    constexpr int CHASE = 4096;
    constexpr int REPEAT = 30, WARMUP = 5;

    // SLM sizes to test (in elements = int32)
    // Max SLM per WG on B60: 128-256 KB
    int slm_sizes[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};

    printf("%-10s %12s %12s %12s\n", "SLM Size", "Median(ns)", "cyc/access", "ns/access");
    printf("%-10s %12s %12s %12s\n", "-------", "--------", "----------", "---------");

    for (int n : slm_sizes) {
        // Build permutation for this SLM size
        std::vector<int> perm(n);
        for (int i = 0; i < n; i++) perm[i] = (i + 1) % n;
        // Shuffle
        for (int i = n - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            std::swap(perm[i], perm[j]);
        }

        sycl::buffer<int, 1> perm_buf(perm.data(), sycl::range<1>(n));
        sycl::buffer<int, 1> res_buf(1);
        double total_ns = 0;

        for (int r = 0; r < WARMUP + REPEAT; r++) {
            auto ev = q.submit([&](sycl::handler& h) {
                auto p = perm_buf.get_access<sycl::access::mode::read>(h);
                auto res = res_buf.get_access<sycl::access::mode::write>(h);
                sycl::local_accessor<int, 1> slm(n, h);

                h.parallel_for(sycl::nd_range<1>(16, 16), [=](sycl::nd_item<1> it) {
                    int lid = it.get_local_id(0);

                    // Copy permutation to SLM
                    for (int i = lid; i < n; i += 16) slm[i] = p[i];
                    sycl::group_barrier(it.get_group());

                    // Thread 0 does the chase
                    if (lid == 0) {
                        int idx = 0;
                        #pragma unroll 1
                        for (int i = 0; i < CHASE; i++) idx = slm[idx];
                        res[0] = idx;
                    }
                });
            });
            ev.wait();
            if (r >= WARMUP) {
                auto t0 = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
                auto t1 = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
                total_ns += (t1 - t0);
            }
        }

        double avg_ns = total_ns / REPEAT;
        double cyc = avg_ns * ghz / CHASE;
        double ns_per = avg_ns / CHASE;

        int sz_bytes = n * sizeof(int);
        char sz_str[32];
        if (sz_bytes < 1024) snprintf(sz_str, sizeof(sz_str), "%dB", sz_bytes);
        else if (sz_bytes < 1024*1024) snprintf(sz_str, sizeof(sz_str), "%dKB", sz_bytes / 1024);
        else snprintf(sz_str, sizeof(sz_str), "%dMB", sz_bytes / (1024*1024));

        printf("%-10s %12.1f %12.1f %12.1f\n", sz_str, avg_ns, cyc, ns_per);
    }

    printf("\n--- Comparison ---\n");
    printf("  L1 cache (from global pointer chase, 1KB): ~71 cycles/access\n");
    printf("  SLM above: if ~20-30 cycles, confirms separate hardware path\n");

    return 0;
}
