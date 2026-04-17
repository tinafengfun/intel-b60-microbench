// Microbenchmark 4: Memory Bandwidth
// Measures sustained read/write bandwidth across the memory hierarchy
// Analogous to paper Section VI-D (Global Memory)
//
// Usage:
//   ./bench_mem_bandwidth
//   unitrace --metric-query --group ComputeBasic ./bench_mem_bandwidth

#include <sycl/sycl.hpp>
#include <cstdio>

// Coalesced read kernel
void bench_read(sycl::queue& q, size_t buf_bytes, int n_iter, double ghz) {
    size_t n = buf_bytes / sizeof(float);
    int wg = 256;
    int n_wg = (n + wg - 1) / wg;
    if (n_wg < 1) n_wg = 1;

    sycl::buffer<float, 1> src(n);
    sycl::buffer<float, 1> dst(1);

    constexpr int REPEAT = 50, WARMUP = 5;
    double total_ns = 0;

    for (int r = 0; r < WARMUP + REPEAT; r++) {
        auto ev = q.submit([&](sycl::handler& h) {
            auto s = src.get_access<sycl::access::mode::read>(h);
            auto d = dst.get_access<sycl::access::mode::write>(h);
            h.parallel_for(sycl::nd_range<1>(n_wg * wg, wg),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]]
            {
                float sum = 0;
                #pragma unroll 4
                for (int i = 0; i < n_iter; i++) {
                    size_t idx = it.get_global_id(0) + i * it.get_global_range(0);
                    if (idx < n) sum += s[idx];
                }
                if (it.get_local_id(0) == 0) d[0] = sum;
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
    double bytes_read = (double)n * n_iter * sizeof(float);
    double bw_gbps = bytes_read / (avg_ns * 1e-9) / 1e9;

    printf("  Read  %6zuMB  %7.1f ns  %7.1f GB/s\n",
           buf_bytes / (1024*1024), avg_ns, bw_gbps);
}

// Coalesced write kernel
void bench_write(sycl::queue& q, size_t buf_bytes, int n_iter, double ghz) {
    size_t n = buf_bytes / sizeof(float);
    int wg = 256;
    int n_wg = (n + wg - 1) / wg;
    if (n_wg < 1) n_wg = 1;

    sycl::buffer<float, 1> dst(n);
    sycl::buffer<float, 1> dummy(1);

    constexpr int REPEAT = 50, WARMUP = 5;
    double total_ns = 0;

    for (int r = 0; r < WARMUP + REPEAT; r++) {
        auto ev = q.submit([&](sycl::handler& h) {
            auto d = dst.get_access<sycl::access::mode::write>(h);
            auto dd = dummy.get_access<sycl::access::mode::write>(h);
            h.parallel_for(sycl::nd_range<1>(n_wg * wg, wg),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]]
            {
                float val = it.get_global_id(0);
                #pragma unroll 4
                for (int i = 0; i < n_iter; i++) {
                    size_t idx = it.get_global_id(0) + i * it.get_global_range(0);
                    if (idx < n) d[idx] = val + i;
                }
                if (it.get_local_id(0) == 0) dd[0] = val;
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
    double bytes_written = (double)n * n_iter * sizeof(float);
    double bw_gbps = bytes_written / (avg_ns * 1e-9) / 1e9;

    printf("  Write %6zuMB  %7.1f ns  %7.1f GB/s\n",
           buf_bytes / (1024*1024), avg_ns, bw_gbps);
}

int main() {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::enable_profiling{}};
    auto dev = q.get_device();
    double ghz = dev.get_info<sycl::info::device::max_clock_frequency>() / 1000.0;

    printf("=== Memory Bandwidth ===\n");
    printf("Device: %s  Clock: %.1f GHz\n",
           dev.get_info<sycl::info::device::name>().c_str(), ghz);
    printf("Peak bandwidth: 456 GB/s (GDDR6 spec)\n\n");

    size_t buf_sizes[] = {
        1UL*1024*1024,    // 1 MB
        4UL*1024*1024,    // 4 MB
        16UL*1024*1024,   // 16 MB
        64UL*1024*1024,   // 64 MB
        256UL*1024*1024,  // 256 MB
        1024UL*1024*1024  // 1 GB
    };

    for (auto sz : buf_sizes) {
        bench_read(q, sz, 10, ghz);
        bench_write(q, sz, 10, ghz);
        printf("\n");
    }

    return 0;
}
