// Bandwidth diagnostic: verify read kernel actually reads data,
// check unitrace counter reliability, measure true DRAM bandwidth.
// Build: icpx -fsycl -O2 -o bench_bw_diag bench_bw_diag.cpp

#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>

int main() {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::enable_profiling{}};
    auto dev = q.get_device();

    size_t buf_bytes = 2UL * 1024 * 1024 * 1024;  // 2 GB - well beyond any cache
    size_t n = buf_bytes / sizeof(float);

    printf("=== Bandwidth Diagnostic ===\n");
    printf("Device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
    printf("Buffer: %zu MB, %zu floats\n", buf_bytes/(1024*1024), n);

    // Allocate device memory
    float* src = sycl::malloc_device<float>(n, q);
    float* dst = sycl::malloc_device<float>(n, q);
    float* checksum = sycl::malloc_device<float>(1, q);

    if (!src || !dst || !checksum) {
        printf("ERROR: malloc_device failed\n");
        return 1;
    }

    // Initialize with non-trivial pattern
    printf("\n--- Initializing buffer with pattern ---\n");
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(1024*256, 256), [=](sycl::nd_item<1> it) {
            size_t tid = it.get_global_id(0);
            for (size_t i = tid; i < n; i += 1024*256) {
                src[i] = (float)(i & 0xFFFF) + 0.5f;
            }
        });
    }).wait();
    printf("Done.\n");

    // === Test 1: Read with checksum verification ===
    printf("\n--- Test 1: Read + Checksum (1M threads, 2GB) ---\n");
    {
        int n_wg = 4096, wg_size = 256;
        int total_threads = n_wg * wg_size;
        int n_iter = (n + total_threads - 1) / total_threads;
        printf("  total_threads=%d, n_iter=%d, total_reads=%d (expect %zu)\n",
               total_threads, n_iter, (size_t)total_threads * n_iter, n);

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
                    // Reduction: add to checksum via atomic (only first thread of each WG)
                    if (tid == 0) *checksum = sum;  // last iteration wins (approximate)
                });
            });
            ev.wait();
            if (r >= 5) {
                auto t0 = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
                auto t1 = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
                total_ns += (t1 - t0);
            }
        }

        // Read back checksum
        float cs_val = 0;
        q.memcpy(&cs_val, checksum, sizeof(float)).wait();
        printf("  Checksum (thread 0 only): %.1f (should be non-zero)\n", cs_val);

        double avg_ns = total_ns / 20;
        double bw = (double)n * sizeof(float) / (avg_ns * 1e-9) / 1e9;
        printf("  READ bandwidth: %.1f GB/s (%.2f ms for %zu MB)\n",
               bw, avg_ns/1e6, buf_bytes/(1024*1024));
    }

    // === Test 2: Write bandwidth ===
    printf("\n--- Test 2: Write (1M threads, 2GB) ---\n");
    {
        int n_wg = 4096, wg_size = 256;
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
                        if (idx < n) dst[idx] = val + (float)i;
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
        double avg_ns = total_ns / 20;
        double bw = (double)n * sizeof(float) / (avg_ns * 1e-9) / 1e9;
        printf("  WRITE bandwidth: %.1f GB/s (%.2f ms)\n", bw, avg_ns/1e6);
    }

    // === Test 3: Copy bandwidth ===
    printf("\n--- Test 3: Copy dst=src (1M threads, 2GB) ---\n");
    {
        int n_wg = 4096, wg_size = 256;
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
        double avg_ns = total_ns / 20;
        double bw = 2.0 * (double)n * sizeof(float) / (avg_ns * 1e-9) / 1e9;
        printf("  COPY bandwidth: %.1f GB/s (2x bytes, %.2f ms)\n", bw, avg_ns/1e6);
        double pin_bw = (double)n * sizeof(float) / (avg_ns * 1e-9) / 1e9;
        printf("  Pin bandwidth (1x): %.1f GB/s\n", pin_bw);
    }

    // === Test 4: Verify data correctness ===
    printf("\n--- Test 4: Data correctness check ---\n");
    {
        // Spot check: read back 10 values from dst after copy
        float spot[10];
        size_t offsets[] = {0, n/10, n/4, n/2, n-1, n/3, n/7, n/5, n/9, n/11};
        for (int i = 0; i < 10; i++) {
            if (offsets[i] < n) {
                q.memcpy(&spot[i], &src[offsets[i]], sizeof(float)).wait();
            }
        }
        printf("  src spot check: ");
        for (int i = 0; i < 10; i++) {
            if (offsets[i] < n)
                printf("%.1f@%zu ", spot[i], offsets[i]);
        }
        printf("\n");

        // Read back same from dst
        float spot2[10];
        for (int i = 0; i < 10; i++) {
            if (offsets[i] < n)
                q.memcpy(&spot2[i], &dst[offsets[i]], sizeof(float)).wait();
        }
        printf("  dst spot check: ");
        for (int i = 0; i < 10; i++) {
            if (offsets[i] < n)
                printf("%.1f@%zu ", spot2[i], offsets[i]);
        }
        printf("\n");

        // Verify src==dst
        bool match = true;
        for (int i = 0; i < 10 && offsets[i] < n; i++) {
            if (spot[i] != spot2[i]) { match = false; break; }
        }
        printf("  src==dst: %s\n", match ? "YES" : "NO");
    }

    // === Test 5: Device memory bandwidth spec ===
    printf("\n--- Test 5: Device info ---\n");
    printf("  global_mem_size: %zu MB\n", dev.get_info<sycl::info::device::global_mem_size>()/(1024*1024));
    printf("  max_clock: %u MHz\n", dev.get_info<sycl::info::device::max_clock_frequency>());

    sycl::free(src, q);
    sycl::free(dst, q);
    sycl::free(checksum, q);

    return 0;
}
