// Fixed Memory Bandwidth Benchmark
// Uses few threads (16K), each reading many sequential elements
// to achieve proper memory throughput saturation
//
// Build: icpx -fsycl -O2 -o bench_mem_bandwidth_fixed bench_mem_bandwidth_fixed.cpp

#include <sycl/sycl.hpp>
#include <cstdio>
#include <vector>

void bench_read(sycl::queue& q, size_t buf_bytes, double ghz) {
    size_t n = buf_bytes / sizeof(float);
    int wg = 256;
    int n_wg = 64;                    // 64 * 256 = 16384 threads
    int total_threads = n_wg * wg;
    int n_iter = (n + total_threads - 1) / total_threads; // cover whole buffer

    sycl::buffer<float, 1> src(n);
    sycl::buffer<float, 1> dst(1);

    constexpr int REPEAT = 30, WARMUP = 5;
    double total_ns = 0;

    for (int r = 0; r < WARMUP + REPEAT; r++) {
        auto ev = q.submit([&](sycl::handler& h) {
            auto s = src.get_access<sycl::access::mode::read>(h);
            auto d = dst.get_access<sycl::access::mode::write>(h);
            h.parallel_for(sycl::nd_range<1>(n_wg * wg, wg),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]]
            {
                int tid = it.get_global_id(0);
                float sum = 0;
                for (int i = 0; i < n_iter; i++) {
                    size_t idx = (size_t)tid + (size_t)i * total_threads;
                    if (idx < n) sum += s[idx];
                }
                if (tid == 0) d[0] = sum;
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
    // Count actual bytes: each unique index < n is read exactly once
    double bytes_read = (double)n * sizeof(float);
    double bw_gbps = bytes_read / (avg_ns * 1e-9) / 1e9;

    printf("  Read  %6zuMB  n_iter=%d  %7.1f ns  %7.1f GB/s\n",
           buf_bytes / (1024*1024), n_iter, avg_ns, bw_gbps);
}

void bench_write(sycl::queue& q, size_t buf_bytes, double ghz) {
    size_t n = buf_bytes / sizeof(float);
    int wg = 256;
    int n_wg = 64;
    int total_threads = n_wg * wg;
    int n_iter = (n + total_threads - 1) / total_threads;

    sycl::buffer<float, 1> dst(n);
    sycl::buffer<float, 1> dummy(1);

    constexpr int REPEAT = 30, WARMUP = 5;
    double total_ns = 0;

    for (int r = 0; r < WARMUP + REPEAT; r++) {
        auto ev = q.submit([&](sycl::handler& h) {
            auto d = dst.get_access<sycl::access::mode::write>(h);
            auto dd = dummy.get_access<sycl::access::mode::write>(h);
            h.parallel_for(sycl::nd_range<1>(n_wg * wg, wg),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]]
            {
                int tid = it.get_global_id(0);
                float val = (float)tid;
                for (int i = 0; i < n_iter; i++) {
                    size_t idx = (size_t)tid + (size_t)i * total_threads;
                    if (idx < n) d[idx] = val + i;
                }
                if (tid == 0) dd[0] = val;
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
    double bytes_written = (double)n * sizeof(float);
    double bw_gbps = bytes_written / (avg_ns * 1e-9) / 1e9;

    printf("  Write %6zuMB  n_iter=%d  %7.1f ns  %7.1f GB/s\n",
           buf_bytes / (1024*1024), n_iter, avg_ns, bw_gbps);
}

int main() {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::enable_profiling{}};
    auto dev = q.get_device();
    double ghz = dev.get_info<sycl::info::device::max_clock_frequency>() / 1000.0;

    printf("=== Memory Bandwidth (Fixed) ===\n");
    printf("Device: %s  Clock: %.1f GHz\n",
           dev.get_info<sycl::info::device::name>().c_str(), ghz);
    printf("Peak bandwidth: 456 GB/s (GDDR6 spec)\n");
    printf("Threads: 16384 (64 WG x 256)\n\n");

    size_t buf_sizes[] = {
        1UL*1024*1024,
        4UL*1024*1024,
        16UL*1024*1024,
        64UL*1024*1024,
        256UL*1024*1024,
        1024UL*1024*1024
    };

    for (auto sz : buf_sizes) {
        bench_read(q, sz, ghz);
        bench_write(q, sz, ghz);
        printf("\n");
    }

    return 0;
}
