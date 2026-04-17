// Microbenchmark 2: Sub-group Scheduling Behavior
// Measures throughput vs sub-group count (TLP) and instruction count (ILP)
// Analogous to paper Section IV-D (Warp Scheduler Behavior)
//
// Usage:
//   ./bench_sg_scheduling
//   unitrace --metric-query --group VectorEngineStalls ./bench_sg_scheduling

#include <sycl/sycl.hpp>
#include <cstdio>

int main() {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::enable_profiling{}};
    auto dev = q.get_device();
    double ghz = dev.get_info<sycl::info::device::max_clock_frequency>() / 1000.0;
    int max_wg = dev.get_info<sycl::info::device::max_work_group_size>();

    printf("=== Sub-group Scheduling Behavior ===\n");
    printf("Device: %s  Max WG: %d\n",
           dev.get_info<sycl::info::device::name>().c_str(), max_wg);

    constexpr int WARMUP = 5, REPEAT = 100;

    // Sweep 1: Fixed N=256, vary sub-group count
    printf("\n--- FP32 Dependent Chain, N=256, vary SG count ---\n");
    printf("%-4s %-8s %10s %10s %10s\n", "SG", "threads", "avg(ns)", "ops/cyc", "cyc/SG");
    printf("%-4s %-8s %10s %10s %10s\n", "", "", "", "", "(x256)");

    for (int nsg : {1, 2, 4, 8, 16, 32, 64, 128, 256}) {
        int wg = nsg * 16;
        if (wg > max_wg) break;
        constexpr int N = 256;

        sycl::buffer<float, 1> buf(wg);
        double total_ns = 0;

        for (int r = 0; r < WARMUP + REPEAT; r++) {
            auto ev = q.submit([&](sycl::handler& h) {
                auto acc = buf.get_access<sycl::access::mode::write>(h);
                h.parallel_for(sycl::nd_range<1>(wg, wg),
                    [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]]
                {
                    float x = 1.001f + it.get_local_id(0);
                    #pragma unroll 1
                    for (int i = 0; i < N; i++) { x = x * x + 1.0f; }
                    acc[it.get_local_id(0)] = x;
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
        double cycles = avg_ns * ghz;
        double ops_per_cyc = (double)wg * N / cycles;
        double cyc_per_sg = cycles;  // total cycles for this many SGs
        printf("%-4d %-8d %10.1f %10.1f %10.0f\n", nsg, wg, avg_ns, ops_per_cyc, cyc_per_sg);
    }

    // Sweep 2: Vary instruction count, fixed 16 SGs
    printf("\n--- FP32, 16 SGs, vary instruction count ---\n");
    printf("%-6s %10s %10s %10s\n", "N", "avg(ns)", "cyc", "cyc/op");
    printf("%-6s %10s %10s %10s\n", "", "", "total", "");

    for (int N : {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}) {
        int wg = 16 * 16;  // 16 SGs
        sycl::buffer<float, 1> buf(wg);
        double total_ns = 0;

        for (int r = 0; r < WARMUP + REPEAT; r++) {
            auto ev = q.submit([&](sycl::handler& h) {
                auto acc = buf.get_access<sycl::access::mode::write>(h);
                h.parallel_for(sycl::nd_range<1>(wg, wg),
                    [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]]
                {
                    float x = 1.001f + it.get_local_id(0);
                    #pragma unroll 1
                    for (int i = 0; i < N; i++) { x = x * x + 1.0f; }
                    acc[it.get_local_id(0)] = x;
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
        double cycles = avg_ns * ghz;
        printf("%-6d %10.1f %10.0f %10.2f\n", N, avg_ns, cycles, cycles / N);
    }

    return 0;
}
