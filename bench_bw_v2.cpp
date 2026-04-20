// Bandwidth investigation v2: fix read serial dependency, add DMA test
// The original read kernel had sum += src[idx] (serial dependency)
// which limits load throughput. This version uses ILP to hide latency.
// Build: icpx -fsycl -O2 -o bench_bw_v2 bench_bw_v2.cpp

#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>

int main() {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::enable_profiling{}};
    auto dev = q.get_device();

    size_t buf_bytes = 2UL * 1024 * 1024 * 1024;  // 2 GB
    size_t n = buf_bytes / sizeof(float);

    printf("=== Bandwidth Investigation v2 ===\n");
    printf("Device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
    printf("Buffer: %zu MB, %zu floats\n", buf_bytes/(1024*1024), n);

    float* src = sycl::malloc_device<float>(n, q);
    float* dst = sycl::malloc_device<float>(n, q);
    float* result = sycl::malloc_device<float>(64, q);  // for storing read results

    if (!src || !dst || !result) {
        printf("ERROR: malloc_device failed\n");
        return 1;
    }

    // Initialize
    printf("\nInitializing...\n");
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(1024*256, 256), [=](sycl::nd_item<1> it) {
            size_t tid = it.get_global_id(0);
            for (size_t i = tid; i < n; i += 1024*256)
                src[i] = (float)(i & 0xFFFF) + 0.5f;
        });
    }).wait();
    printf("Done.\n");

    int n_wg = 4096, wg_size = 256;
    int total_threads = n_wg * wg_size;  // 1M threads
    int n_iter = (n + total_threads - 1) / total_threads;

    auto run_test = [&](const char* name, int n_warmup, int n_meas, auto kernel_fn) {
        double total_ns = 0;
        for (int r = 0; r < n_warmup + n_meas; r++) {
            sycl::event ev = q.submit(kernel_fn);
            ev.wait();
            if (r >= n_warmup) {
                auto t0 = ev.template get_profiling_info<sycl::info::event_profiling::command_start>();
                auto t1 = ev.template get_profiling_info<sycl::info::event_profiling::command_end>();
                total_ns += (t1 - t0);
            }
        }
        return total_ns / n_meas;
    };

    // === Test 1: Read with serial dependency (original) ===
    printf("\n--- Test 1: Read serial (sum += src[idx], 1 accumulator) ---\n");
    {
        double avg_ns = run_test("read_serial", 5, 20, [&](sycl::handler& h) {
            h.parallel_for(sycl::nd_range<1>(n_wg * wg_size, wg_size),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]]
            {
                int tid = it.get_global_id(0);
                float sum = 0;
                for (int i = 0; i < n_iter; i++) {
                    size_t idx = (size_t)tid + (size_t)i * total_threads;
                    if (idx < n) sum += src[idx];
                }
                if (it.get_local_id(0) == 0)
                    result[0] = sum;
            });
        });
        double bw = (double)n * sizeof(float) / (avg_ns * 1e-9) / 1e9;
        printf("  READ (serial): %.1f GB/s (%.2f ms)\n", bw, avg_ns/1e6);
    }

    // === Test 2: Read with ILP (4 independent accumulators) ===
    printf("\n--- Test 2: Read ILP4 (4 independent accumulators) ---\n");
    {
        double avg_ns = run_test("read_ilp4", 5, 20, [&](sycl::handler& h) {
            h.parallel_for(sycl::nd_range<1>(n_wg * wg_size, wg_size),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]]
            {
                int tid = it.get_global_id(0);
                float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                for (int i = 0; i < n_iter; i += 4) {
                    size_t idx0 = (size_t)tid + (size_t)(i+0) * total_threads;
                    size_t idx1 = (size_t)tid + (size_t)(i+1) * total_threads;
                    size_t idx2 = (size_t)tid + (size_t)(i+2) * total_threads;
                    size_t idx3 = (size_t)tid + (size_t)(i+3) * total_threads;
                    if (idx0 < n) s0 += src[idx0];
                    if (idx1 < n) s1 += src[idx1];
                    if (idx2 < n) s2 += src[idx2];
                    if (idx3 < n) s3 += src[idx3];
                }
                if (it.get_local_id(0) == 0)
                    result[0] = s0 + s1 + s2 + s3;
            });
        });
        double bw = (double)n * sizeof(float) / (avg_ns * 1e-9) / 1e9;
        printf("  READ (ILP4): %.1f GB/s (%.2f ms)\n", bw, avg_ns/1e6);
    }

    // === Test 3: Read with ILP8 ===
    printf("\n--- Test 3: Read ILP8 (8 independent accumulators) ---\n");
    {
        double avg_ns = run_test("read_ilp8", 5, 20, [&](sycl::handler& h) {
            h.parallel_for(sycl::nd_range<1>(n_wg * wg_size, wg_size),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]]
            {
                int tid = it.get_global_id(0);
                float s0=0, s1=0, s2=0, s3=0, s4=0, s5=0, s6=0, s7=0;
                for (int i = 0; i < n_iter; i += 8) {
                    for (int j = 0; j < 8 && i+j < n_iter; j++) {
                        size_t idx = (size_t)tid + (size_t)(i+j) * total_threads;
                        if (idx < n) {
                            float v = src[idx];
                            if (j==0) s0 += v;
                            else if (j==1) s1 += v;
                            else if (j==2) s2 += v;
                            else if (j==3) s3 += v;
                            else if (j==4) s4 += v;
                            else if (j==5) s5 += v;
                            else if (j==6) s6 += v;
                            else s7 += v;
                        }
                    }
                }
                if (it.get_local_id(0) == 0)
                    result[0] = s0+s1+s2+s3+s4+s5+s6+s7;
            });
        });
        double bw = (double)n * sizeof(float) / (avg_ns * 1e-9) / 1e9;
        printf("  READ (ILP8): %.1f GB/s (%.2f ms)\n", bw, avg_ns/1e6);
    }

    // === Test 4: Read with vector load (float4) ===
    printf("\n--- Test 4: Read vector4 (sycl::float4 load, 4x bandwidth per load) ---\n");
    {
        // Use float4 for wider loads
        size_t n4 = n / 4;
        int n_iter4 = (n4 + total_threads - 1) / total_threads;

        // Allocate float4 source
        sycl::float4* src4 = (sycl::float4*)src;  // reinterpret
        double avg_ns = run_test("read_vec4", 5, 20, [&](sycl::handler& h) {
            h.parallel_for(sycl::nd_range<1>(n_wg * wg_size, wg_size),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]]
            {
                int tid = it.get_global_id(0);
                float sum = 0;
                for (int i = 0; i < n_iter4; i++) {
                    size_t idx = (size_t)tid + (size_t)i * total_threads;
                    if (idx < n4) {
                        sycl::float4 v = src4[idx];
                        sum += v.x() + v.y() + v.z() + v.w();
                    }
                }
                if (it.get_local_id(0) == 0)
                    result[0] = sum;
            });
        });
        double bw = (double)n * sizeof(float) / (avg_ns * 1e-9) / 1e9;
        printf("  READ (float4): %.1f GB/s (%.2f ms)\n", bw, avg_ns/1e6);
    }

    // === Test 5: Write ===
    printf("\n--- Test 5: Write (baseline) ---\n");
    {
        double avg_ns = run_test("write", 5, 20, [&](sycl::handler& h) {
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
        double bw = (double)n * sizeof(float) / (avg_ns * 1e-9) / 1e9;
        printf("  WRITE: %.1f GB/s (%.2f ms)\n", bw, avg_ns/1e6);
    }

    // === Test 6: Write + readback (force write to DRAM) ===
    printf("\n--- Test 6: Write then Read-back (2 kernels, sequential) ---\n");
    {
        double total_ns = 0;
        for (int r = 0; r < 25; r++) {
            // Write kernel
            auto ev1 = q.submit([&](sycl::handler& h) {
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
            ev1.wait();

            // Read-back from dst (forces write to complete to DRAM)
            auto ev2 = q.submit([&](sycl::handler& h) {
                h.parallel_for(sycl::nd_range<1>(n_wg * wg_size, wg_size),
                    [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]]
                {
                    int tid = it.get_global_id(0);
                    float sum = 0;
                    for (int i = 0; i < n_iter; i++) {
                        size_t idx = (size_t)tid + (size_t)i * total_threads;
                        if (idx < n) sum += dst[idx];
                    }
                    if (it.get_local_id(0) == 0)
                        result[0] = sum;
                });
            });
            ev2.wait();

            if (r >= 5) {
                auto t0 = ev1.get_profiling_info<sycl::info::event_profiling::command_start>();
                auto t1 = ev2.get_profiling_info<sycl::info::event_profiling::command_end>();
                total_ns += (t1 - t0);
            }
        }
        double avg_ns = total_ns / 20;
        double bw_write = (double)n * sizeof(float) / (avg_ns * 1e-9) / 1e9;  // 1x for write
        double bw_total = 2.0 * (double)n * sizeof(float) / (avg_ns * 1e-9) / 1e9;  // 2x total
        printf("  WRITE+READ: %.1f GB/s (1x, %.2f ms) = %.1f GB/s (2x)\n",
               bw_write, avg_ns/1e6, bw_total);
    }

    // === Test 7: Copy (dst=src) ===
    printf("\n--- Test 7: Copy (dst=src) ---\n");
    {
        double avg_ns = run_test("copy", 5, 20, [&](sycl::handler& h) {
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
        double bw_2x = 2.0 * (double)n * sizeof(float) / (avg_ns * 1e-9) / 1e9;
        double bw_1x = (double)n * sizeof(float) / (avg_ns * 1e-9) / 1e9;
        printf("  COPY: %.1f GB/s (2x bytes, %.2f ms) | %.1f GB/s (1x)\n",
               bw_2x, avg_ns/1e6, bw_1x);
    }

    // === Test 8: DMA memcpy (sycl::memcpy) ===
    printf("\n--- Test 8: DMA memcpy (q.memcpy) ---\n");
    {
        double total_ns = 0;
        for (int r = 0; r < 25; r++) {
            auto ev = q.memcpy(dst, src, buf_bytes);
            ev.wait();
            if (r >= 5) {
                auto t0 = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
                auto t1 = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
                total_ns += (t1 - t0);
            }
        }
        double avg_ns = total_ns / 20;
        double bw = (double)buf_bytes / (avg_ns * 1e-9) / 1e9;
        printf("  DMA COPY: %.1f GB/s (%.2f ms for %zu MB)\n",
               bw, avg_ns/1e6, buf_bytes/(1024*1024));
    }

    // === Test 9: More threads (8K WGs = 2M threads) ===
    printf("\n--- Test 9: Read ILP4 with 2M threads (8K WGs) ---\n");
    {
        int n_wg2 = 8192;
        int tt2 = n_wg2 * wg_size;  // 2M threads
        int n_iter2 = (n + tt2 - 1) / tt2;
        double avg_ns = run_test("read_ilp4_2m", 5, 20, [&](sycl::handler& h) {
            h.parallel_for(sycl::nd_range<1>(n_wg2 * wg_size, wg_size),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]]
            {
                int tid = it.get_global_id(0);
                float s0=0, s1=0, s2=0, s3=0;
                for (int i = 0; i < n_iter2; i += 4) {
                    size_t idx0 = (size_t)tid + (size_t)(i+0) * tt2;
                    size_t idx1 = (size_t)tid + (size_t)(i+1) * tt2;
                    size_t idx2 = (size_t)tid + (size_t)(i+2) * tt2;
                    size_t idx3 = (size_t)tid + (size_t)(i+3) * tt2;
                    if (idx0 < n) s0 += src[idx0];
                    if (idx1 < n) s1 += src[idx1];
                    if (idx2 < n) s2 += src[idx2];
                    if (idx3 < n) s3 += src[idx3];
                }
                if (it.get_local_id(0) == 0)
                    result[0] = s0+s1+s2+s3;
            });
        });
        double bw = (double)n * sizeof(float) / (avg_ns * 1e-9) / 1e9;
        printf("  READ (ILP4, 2M threads): %.1f GB/s (%.2f ms)\n", bw, avg_ns/1e6);
    }

    printf("\n=== Summary ===\n");
    printf("Expected peak: ~456 GB/s (if 192-bit GDDR6 at 19 Gbps)\n");
    printf("Key question: Is read limited by serial dependency or true DRAM bottleneck?\n");

    sycl::free(src, q);
    sycl::free(dst, q);
    sycl::free(result, q);
    return 0;
}
