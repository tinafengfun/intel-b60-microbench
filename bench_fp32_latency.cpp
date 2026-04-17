// Microbenchmark 1: FP32/BF16/INT32 Compute Pipeline Latency
// Measures true latency (dependent chain) and throughput for different types
// Analogous to paper Section IV (Compute Pipeline)
//
// Usage:
//   ./bench_fp32_latency
//   unitrace --metric-query --group ComputeBasic --device-timing ./bench_fp32_latency
//   unitrace --metric-query --group VectorEngineStalls ./bench_fp32_latency

#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::enable_profiling{}};
    auto dev = q.get_device();
    double ghz = dev.get_info<sycl::info::device::max_clock_frequency>() / 1000.0;

    printf("=== FP32 Compute Pipeline Latency ===\n");
    printf("Device: %s  Clock: %.1f GHz\n",
           dev.get_info<sycl::info::device::name>().c_str(), ghz);

    constexpr int WARMUP = 10, REPEAT = 200;
    int sizes[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};

    printf("\n%-6s %10s %10s %10s\n", "N", "avg(ns)", "cyc/op", "ops/cyc");
    printf("------ ---------- ---------- ----------\n");

    for (int N : sizes) {
        sycl::buffer<float, 1> buf(1);
        double total_ns = 0;

        for (int r = 0; r < WARMUP + REPEAT; r++) {
            auto ev = q.submit([&](sycl::handler& h) {
                auto acc = buf.get_access<sycl::access::mode::write>(h);
                h.parallel_for(sycl::nd_range<1>(16, 16), [=](sycl::nd_item<1> it)
                    [[sycl::reqd_sub_group_size(16)]]
                {
                    float x = 1.001f + it.get_local_id(0);
                    #pragma unroll 1
                    for (int i = 0; i < N; i++) {
                        x = x * x + 1.0f;  // dependent FMA chain
                    }
                    if (it.get_local_id(0) == 0) acc[0] = x;
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
        double cyc_per_op = avg_ns * ghz / N;
        double ops_per_cyc = 1.0 / cyc_per_op;
        printf("%-6d %10.1f %10.2f %10.3f\n", N, avg_ns, cyc_per_op, ops_per_cyc);
    }

    // INT32 dependent chain
    printf("\n=== INT32 MAD Dependent Chain ===\n");
    printf("%-6s %10s %10s\n", "N", "avg(ns)", "cyc/op");
    printf("------ ---------- ----------\n");

    for (int N : sizes) {
        sycl::buffer<int, 1> buf(1);
        double total_ns = 0;

        for (int r = 0; r < WARMUP + REPEAT; r++) {
            auto ev = q.submit([&](sycl::handler& h) {
                auto acc = buf.get_access<sycl::access::mode::write>(h);
                h.parallel_for(sycl::nd_range<1>(16, 16), [=](sycl::nd_item<1> it)
                    [[sycl::reqd_sub_group_size(16)]]
                {
                    int x = it.get_local_id(0) + 1;
                    #pragma unroll 1
                    for (int i = 0; i < N; i++) {
                        x = x * x + 1;
                    }
                    if (it.get_local_id(0) == 0) acc[0] = x;
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
        double cyc_per_op = avg_ns * ghz / N;
        printf("%-6d %10.1f %10.2f\n", N, avg_ns, cyc_per_op);
    }

    return 0;
}
