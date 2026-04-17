// Microbenchmark 3: Memory Hierarchy Latency via Pointer Chase
// Maps latency across: registers → SLM/L1 → L2 → Global Memory
// Analogous to paper Section VI (Memory Subsystem)
//
// Usage:
//   ./bench_mem_latency
//   unitrace --metric-query --group ComputeBasic ./bench_mem_latency

#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

int main() {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::enable_profiling{}};
    auto dev = q.get_device();
    double ghz = dev.get_info<sycl::info::device::max_clock_frequency>() / 1000.0;

    printf("=== Memory Hierarchy Latency (Pointer Chase) ===\n");
    printf("Device: %s  Clock: %.1f GHz\n",
           dev.get_info<sycl::info::device::name>().c_str(), ghz);

    constexpr int REPEAT = 30, WARMUP = 5, CHASE = 4096;

    // Sizes in KB to sweep across cache boundaries
    size_t kb_sizes[] = {
        1, 2, 4, 8, 16, 32, 48, 64, 96, 128,
        192, 256, 384, 512, 768, 1024,
        1536, 2048, 3072, 4096,
        8192, 16384, 32768, 65536, 131072, 262144
    };

    printf("\n%-10s %12s %10s %12s %10s\n", "buf_size", "elements", "avg(ns)", "cyc/access", "region");
    printf("%-10s %12s %10s %12s %10s\n", "-------", "--------", "-------", "----------", "------");

    for (auto sz_kb : kb_sizes) {
        size_t buf_bytes = sz_kb * 1024UL;
        size_t n = buf_bytes / sizeof(int);
        if (n < (size_t)CHASE) n = CHASE;

        // Build random permutation (pointer chase list)
        std::vector<int> next_h(n);
        for (size_t i = 0; i < n; i++) next_h[i] = (int)((i + 1) % n);
        // Fisher-Yates shuffle
        for (size_t i = n - 1; i > 0; i--) {
            size_t j = rand() % (i + 1);
            std::swap(next_h[i], next_h[j]);
        }

        sycl::buffer<int, 1> next_buf(next_h.data(), sycl::range<1>(n));
        sycl::buffer<int, 1> res_buf(1);
        double total_ns = 0;

        for (int r = 0; r < WARMUP + REPEAT; r++) {
            auto ev = q.submit([&](sycl::handler& h) {
                auto nacc = next_buf.get_access<sycl::access::mode::read>(h);
                auto racc = res_buf.get_access<sycl::access::mode::write>(h);
                h.parallel_for(sycl::nd_range<1>(1, 1), [=](sycl::nd_item<1>) {
                    int idx = 0;
                    #pragma unroll 1
                    for (int i = 0; i < CHASE; i++) idx = nacc[idx];
                    racc[0] = idx;
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

        // Identify region
        const char* region = "???";
        if (sz_kb <= 128) region = "SLM/L1?";
        else if (sz_kb <= 4096) region = "L2?";
        else region = "Global";

        char size_str[32];
        if (sz_kb < 1024) snprintf(size_str, sizeof(size_str), "%zuKB", sz_kb);
        else snprintf(size_str, sizeof(size_str), "%zuMB", sz_kb / 1024);

        printf("%-10s %12zu %10.1f %12.1f %-10s\n", size_str, n, avg_ns, cyc, region);
    }

    return 0;
}
