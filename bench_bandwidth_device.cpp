// Bandwidth v3: Device-only memory allocation + mixed read/write patterns
// Uses sycl::malloc_device to avoid zeMemAllocShared overhead
// Build: icpx -fsycl -O2 -o bench_bandwidth_device bench_bandwidth_device.cpp

#include <sycl/sycl.hpp>
#include <cstdio>

double bench_read(sycl::queue& q, float* src, size_t n, int n_wg, int wg_size) {
    int total_threads = n_wg * wg_size;
    int n_iter = (n + total_threads - 1) / total_threads;

    float* dummy = sycl::malloc_device<float>(1, q);

    constexpr int REPEAT = 20, WARMUP = 5;
    double total_ns = 0;

    for (int r = 0; r < WARMUP + REPEAT; r++) {
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
        if (r >= WARMUP) {
            auto t0 = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
            auto t1 = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
            total_ns += (t1 - t0);
        }
    }

    sycl::free(dummy, q);
    double avg_ns = total_ns / REPEAT;
    return (double)n * sizeof(float) / (avg_ns * 1e-9) / 1e9;
}

double bench_write(sycl::queue& q, float* dst, size_t n, int n_wg, int wg_size) {
    int total_threads = n_wg * wg_size;
    int n_iter = (n + total_threads - 1) / total_threads;

    float* dummy = sycl::malloc_device<float>(1, q);

    constexpr int REPEAT = 20, WARMUP = 5;
    double total_ns = 0;

    for (int r = 0; r < WARMUP + REPEAT; r++) {
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
        if (r >= WARMUP) {
            auto t0 = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
            auto t1 = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
            total_ns += (t1 - t0);
        }
    }

    sycl::free(dummy, q);
    double avg_ns = total_ns / REPEAT;
    return (double)n * sizeof(float) / (avg_ns * 1e-9) / 1e9;
}

double bench_copy(sycl::queue& q, float* src, float* dst, size_t n, int n_wg, int wg_size) {
    int total_threads = n_wg * wg_size;
    int n_iter = (n + total_threads - 1) / total_threads;

    constexpr int REPEAT = 20, WARMUP = 5;
    double total_ns = 0;

    for (int r = 0; r < WARMUP + REPEAT; r++) {
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
        if (r >= WARMUP) {
            auto t0 = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
            auto t1 = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
            total_ns += (t1 - t0);
        }
    }

    double avg_ns = total_ns / REPEAT;
    // Copy reads n floats AND writes n floats
    return 2.0 * (double)n * sizeof(float) / (avg_ns * 1e-9) / 1e9;
}

int main() {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::enable_profiling{}};
    auto dev = q.get_device();
    double ghz = dev.get_info<sycl::info::device::max_clock_frequency>() / 1000.0;

    printf("=== Bandwidth v3 (Device Memory, malloc_device) ===\n");
    printf("Device: %s  Clock: %.1f GHz\n\n",
           dev.get_info<sycl::info::device::name>().c_str(), ghz);

    size_t buf_bytes = 1024UL * 1024 * 1024;  // 1 GB
    size_t n = buf_bytes / sizeof(float);

    // Allocate device-only memory
    float* buf1 = sycl::malloc_device<float>(n, q);
    float* buf2 = sycl::malloc_device<float>(n, q);
    if (!buf1 || !buf2) {
        printf("ERROR: malloc_device failed\n");
        return 1;
    }

    // Initialize
    q.fill(buf1, 1.0f, n).wait();
    q.fill(buf2, 2.0f, n).wait();

    printf("--- Thread Count Sweep (1GB, device memory) ---\n");
    printf("%-12s %12s %12s %12s %12s\n", "Threads", "Read GB/s", "Write GB/s", "Copy GB/s", "Copy(2×)");
    printf("%-12s %12s %12s %12s %12s\n", "-------", "---------", "----------", "---------", "--------");

    for (int n_wg : {64, 128, 256, 512, 1024, 2048, 4096}) {
        int wg_size = 256;
        double r_bw = bench_read(q, buf1, n, n_wg, wg_size);
        double w_bw = bench_write(q, buf2, n, n_wg, wg_size);
        double c_bw = bench_copy(q, buf1, buf2, n, n_wg, wg_size);
        printf("%-12d %12.1f %12.1f %12.1f %12.1f\n",
               n_wg * wg_size, r_bw, w_bw, c_bw, c_bw);
    }

    sycl::free(buf1, q);
    sycl::free(buf2, q);

    printf("\nNote: Copy GB/s uses 2×bytes (read+write) per the STREAM convention\n");
    printf("      Copy(2×) column shows the same number for easy comparison\n");

    return 0;
}
