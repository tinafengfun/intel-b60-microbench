// Bandwidth v3: vectorized (float4) read/write/copy sweep
// Build: icpx -fsycl -O2 -o bench_bw_v3 bench_bw_v3.cpp

#include <sycl/sycl.hpp>
#include <cstdio>

int main() {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::enable_profiling{}};
    auto dev = q.get_device();

    size_t buf_bytes = 2UL * 1024 * 1024 * 1024;  // 2 GB
    size_t n = buf_bytes / sizeof(float);
    size_t n4 = n / 4;

    printf("=== Bandwidth v3: Vectorized Sweep ===\n");
    printf("Device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
    printf("Buffer: %zu MB\n", buf_bytes/(1024*1024));

    float* src = sycl::malloc_device<float>(n, q);
    float* dst = sycl::malloc_device<float>(n, q);
    float* result = sycl::malloc_device<float>(1, q);
    auto src4 = (sycl::float4*)src;
    auto dst4 = (sycl::float4*)dst;

    // Initialize
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(1024*256, 256), [=](sycl::nd_item<1> it) {
            size_t tid = it.get_global_id(0);
            for (size_t i = tid; i < n; i += 1024*256)
                src[i] = (float)(i & 0xFFFF) + 0.5f;
        });
    }).wait();

    // Sweep thread counts: 1K, 2K, 4K, 8K, 16K WGs (256 threads each)
    int wg_sizes[] = {256};
    int wg_counts[] = {1024, 2048, 4096, 8192, 16384};

    auto bench = [&](const char* label, int nwg, int wgs, auto kernel) -> double {
        double total = 0;
        for (int r = 0; r < 25; r++) {
            sycl::event ev = q.submit(kernel);
            ev.wait();
            if (r >= 5) {
                auto t0 = ev.template get_profiling_info<sycl::info::event_profiling::command_start>();
                auto t1 = ev.template get_profiling_info<sycl::info::event_profiling::command_end>();
                total += (t1 - t0);
            }
        }
        return total / 20;
    };

    printf("\n=== Scalar (float) Read ===\n");
    printf("%8s %10s %10s %10s\n", "Threads", "Time(ms)", "GB/s", "iter/thread");
    for (int nwg : wg_counts) {
        int tt = nwg * 256;
        int ni = (n + tt - 1) / tt;
        double ns = bench("read_scalar", nwg, 256, [&](sycl::handler& h) {
            h.parallel_for(sycl::nd_range<1>(nwg*256, 256),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]]
            {
                int tid = it.get_global_id(0);
                float sum = 0;
                for (int i = 0; i < ni; i++) {
                    size_t idx = (size_t)tid + (size_t)i * tt;
                    if (idx < n) sum += src[idx];
                }
                if (it.get_local_id(0) == 0) *result = sum;
            });
        });
        double bw = (double)n * 4 / (ns * 1e-9) / 1e9;
        printf("%8d %10.2f %10.1f %10d\n", tt, ns/1e6, bw, ni);
    }

    printf("\n=== Vector (float4) Read ===\n");
    printf("%8s %10s %10s %10s\n", "Threads", "Time(ms)", "GB/s", "iter/thread");
    for (int nwg : wg_counts) {
        int tt = nwg * 256;
        int ni = (n4 + tt - 1) / tt;
        double ns = bench("read_vec4", nwg, 256, [&](sycl::handler& h) {
            h.parallel_for(sycl::nd_range<1>(nwg*256, 256),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]]
            {
                int tid = it.get_global_id(0);
                float sum = 0;
                for (int i = 0; i < ni; i++) {
                    size_t idx = (size_t)tid + (size_t)i * tt;
                    if (idx < n4) {
                        sycl::float4 v = src4[idx];
                        sum += v.x() + v.y() + v.z() + v.w();
                    }
                }
                if (it.get_local_id(0) == 0) *result = sum;
            });
        });
        double bw = (double)n * 4 / (ns * 1e-9) / 1e9;
        printf("%8d %10.2f %10.1f %10d\n", tt, ns/1e6, bw, ni);
    }

    printf("\n=== Vector (float4) Write ===\n");
    printf("%8s %10s %10s %10s\n", "Threads", "Time(ms)", "GB/s", "iter/thread");
    for (int nwg : wg_counts) {
        int tt = nwg * 256;
        int ni = (n4 + tt - 1) / tt;
        double ns = bench("write_vec4", nwg, 256, [&](sycl::handler& h) {
            h.parallel_for(sycl::nd_range<1>(nwg*256, 256),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]]
            {
                int tid = it.get_global_id(0);
                sycl::float4 val = {(float)tid, (float)(tid+1), (float)(tid+2), (float)(tid+3)};
                for (int i = 0; i < ni; i++) {
                    size_t idx = (size_t)tid + (size_t)i * tt;
                    if (idx < n4) dst4[idx] = val;
                }
            });
        });
        double bw = (double)n * 4 / (ns * 1e-9) / 1e9;
        printf("%8d %10.2f %10.1f %10d\n", tt, ns/1e6, bw, ni);
    }

    printf("\n=== Vector (float4) Copy ===\n");
    printf("%8s %10s %10s %10s %10s\n", "Threads", "Time(ms)", "GB/s(2x)", "GB/s(1x)", "iter/thr");
    for (int nwg : wg_counts) {
        int tt = nwg * 256;
        int ni = (n4 + tt - 1) / tt;
        double ns = bench("copy_vec4", nwg, 256, [&](sycl::handler& h) {
            h.parallel_for(sycl::nd_range<1>(nwg*256, 256),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]]
            {
                int tid = it.get_global_id(0);
                for (int i = 0; i < ni; i++) {
                    size_t idx = (size_t)tid + (size_t)i * tt;
                    if (idx < n4) dst4[idx] = src4[idx];
                }
            });
        });
        double bw2x = 2.0 * (double)n * 4 / (ns * 1e-9) / 1e9;
        double bw1x = (double)n * 4 / (ns * 1e-9) / 1e9;
        printf("%8d %10.2f %10.1f %10.1f %10d\n", tt, ns/1e6, bw2x, bw1x, ni);
    }

    printf("\n=== DMA memcpy ===\n");
    {
        double total = 0;
        for (int r = 0; r < 25; r++) {
            sycl::event ev = q.memcpy(dst, src, buf_bytes);
            ev.wait();
            if (r >= 5) {
                auto t0 = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
                auto t1 = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
                total += (t1 - t0);
            }
        }
        double ns = total / 20;
        printf("DMA: %.1f GB/s (%.2f ms)\n", (double)buf_bytes / (ns*1e-9) / 1e9, ns/1e6);
    }

    sycl::free(src, q);
    sycl::free(dst, q);
    sycl::free(result, q);
    return 0;
}
