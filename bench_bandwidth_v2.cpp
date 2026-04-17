// Bandwidth v2: Multiple thread counts + ILP to find true DRAM bandwidth
// Build: icpx -fsycl -O2 -o bench_bandwidth_v2 bench_bandwidth_v2.cpp

#include <sycl/sycl.hpp>
#include <cstdio>
#include <vector>

double bench_read(sycl::queue& q, size_t buf_bytes, int n_wg, int wg_size, int n_ilp) {
    size_t n = buf_bytes / sizeof(float);
    int total_threads = n_wg * wg_size;
    int n_iter = (n + (size_t)total_threads * n_ilp - 1) / ((size_t)total_threads * n_ilp);

    // Each thread reads n_iter * n_ilp elements
    // Total elements = total_threads * n_iter * n_ilp >= n
    sycl::buffer<float, 1> src(n);
    sycl::buffer<float, 1> dst(n_ilp);  // one result per ILP lane

    constexpr int REPEAT = 20, WARMUP = 5;
    double total_ns = 0;

    for (int r = 0; r < WARMUP + REPEAT; r++) {
        auto ev = q.submit([&](sycl::handler& h) {
            auto s = src.get_access<sycl::access::mode::read>(h);
            auto d = dst.get_access<sycl::access::mode::write>(h);
            h.parallel_for(sycl::nd_range<1>(n_wg * wg_size, wg_size),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]]
            {
                int tid = it.get_global_id(0);
                float sums[16]; // max n_ilp
                for (int k = 0; k < n_ilp; k++) sums[k] = 0;

                for (int i = 0; i < n_iter; i++) {
                    for (int k = 0; k < n_ilp; k++) {
                        size_t idx = (size_t)tid + (size_t)(i * n_ilp + k) * total_threads;
                        if (idx < n) sums[k] += s[idx];
                    }
                }
                for (int k = 0; k < n_ilp; k++) {
                    if (tid == 0) d[k] = sums[k];
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
    double bytes_read = (double)n * sizeof(float);
    double bw = bytes_read / (avg_ns * 1e-9) / 1e9;
    return bw;
}

double bench_write(sycl::queue& q, size_t buf_bytes, int n_wg, int wg_size, int n_ilp) {
    size_t n = buf_bytes / sizeof(float);
    int total_threads = n_wg * wg_size;
    int n_iter = (n + (size_t)total_threads * n_ilp - 1) / ((size_t)total_threads * n_ilp);

    sycl::buffer<float, 1> dst_buf(n);
    sycl::buffer<float, 1> dummy(1);

    constexpr int REPEAT = 20, WARMUP = 5;
    double total_ns = 0;

    for (int r = 0; r < WARMUP + REPEAT; r++) {
        auto ev = q.submit([&](sycl::handler& h) {
            auto d = dst_buf.get_access<sycl::access::mode::write>(h);
            auto dd = dummy.get_access<sycl::access::mode::write>(h);
            h.parallel_for(sycl::nd_range<1>(n_wg * wg_size, wg_size),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]]
            {
                int tid = it.get_global_id(0);
                float val = (float)tid;

                for (int i = 0; i < n_iter; i++) {
                    for (int k = 0; k < n_ilp; k++) {
                        size_t idx = (size_t)tid + (size_t)(i * n_ilp + k) * total_threads;
                        if (idx < n) d[idx] = val + (float)(i * n_ilp + k);
                    }
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
    double bw = bytes_written / (avg_ns * 1e-9) / 1e9;
    return bw;
}

int main() {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::enable_profiling{}};
    auto dev = q.get_device();
    double ghz = dev.get_info<sycl::info::device::max_clock_frequency>() / 1000.0;

    printf("=== Memory Bandwidth v2 ===\n");
    printf("Device: %s  Clock: %.1f GHz\n\n",
           dev.get_info<sycl::info::device::name>().c_str(), ghz);

    size_t buf_bytes = 1024UL * 1024 * 1024;  // 1 GB

    // Sweep thread count
    printf("--- Read Bandwidth (1GB, stride access, n_ilp=1) ---\n");
    printf("%-12s %12s %12s\n", "WG_count", "Threads", "GB/s");
    for (int n_wg : {32, 64, 128, 256, 512, 1024, 2048}) {
        int wg_size = 256;
        double bw = bench_read(q, buf_bytes, n_wg, wg_size, 1);
        printf("%-12d %12d %12.1f\n", n_wg, n_wg * wg_size, bw);
    }

    printf("\n--- Write Bandwidth (1GB, stride access, n_ilp=1) ---\n");
    printf("%-12s %12s %12s\n", "WG_count", "Threads", "GB/s");
    for (int n_wg : {32, 64, 128, 256, 512, 1024, 2048}) {
        int wg_size = 256;
        double bw = bench_write(q, buf_bytes, n_wg, wg_size, 1);
        printf("%-12d %12d %12d %12.1f\n", n_wg, n_wg * wg_size, wg_size, bw);
    }

    // Sweep ILP at best thread count
    printf("\n--- Read Bandwidth (1GB, 256 WG, sweep ILP) ---\n");
    printf("%-12s %12s\n", "n_ilp", "GB/s");
    for (int ilp : {1, 2, 4, 8}) {
        double bw = bench_read(q, buf_bytes, 256, 256, ilp);
        printf("%-12d %12.1f\n", ilp, bw);
    }

    return 0;
}
