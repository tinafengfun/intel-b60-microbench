// Verify SYCL type alternatives for low-bitwidth writeback skill
// Tests: (1) bfloat16 scalar vs vec4, (2) sycl::vec<uint16_t,4>, (3) vec2 (32-bit),
//        (4) accessor pattern, (5) custom vec4_t<uint16_t>
// Build: icpx -fsycl -fsycl-targets=intel_gpu_bmg_g21 -O3 -std=c++17 -o verify_sycl_types verify_sycl_types.cpp

#include <sycl/sycl.hpp>
#include <stdio.h>
#include <string.h>

using namespace sycl;

// Custom vec4_t (from skill T1)
template<typename T>
struct alignas(sizeof(T) * 4) vec4_t {
    T v[4];
    T& operator[](int i) { return v[i]; }
    const T& operator[](int i) const { return v[i]; }
};

// Custom vec2_t for 32-bit packing (2 × 16-bit)
template<typename T>
struct alignas(sizeof(T) * 2) vec2_t {
    T v[2];
    T& operator[](int i) { return v[i]; }
    const T& operator[](int i) const { return v[i]; }
};

static double run_copy_bench(queue& q, size_t n_bytes, int n_wg, int wg_size, int repeats,
                             std::function<void(queue&, size_t)> kernel_fn) {
    // Warmup
    for (int i = 0; i < 2; i++) kernel_fn(q, n_bytes);
    // Measure
    double total_ns = 0;
    for (int i = 0; i < repeats; i++) {
        // Use chrono since some kernels may not return profiling events easily
        auto start = std::chrono::high_resolution_clock::now();
        kernel_fn(q, n_bytes);
        auto end = std::chrono::high_resolution_clock::now();
        total_ns += std::chrono::duration<double, std::nano>(end - start).count();
    }
    return total_ns / repeats;
}

// ============================================================
// Test 1: bfloat16 scalar store (using sycl::ext::oneapi::bfloat16)
// ============================================================
void test_bfloat16_scalar(queue& q, size_t n) {
    using bf16 = sycl::ext::oneapi::bfloat16;
    auto src = malloc_device<bf16>(n, q);
    auto dst = malloc_device<bf16>(n, q);

    // Init with bf16 1.0
    float* host_f = malloc_host<float>(n, q);
    for (size_t i = 0; i < n; i++) host_f[i] = 1.0f;
    auto tmp = malloc_device<float>(n, q);
    q.memcpy(tmp, host_f, n * sizeof(float)).wait();
    q.submit([&](handler& h) {
        h.parallel_for(range<1>(n), [=](item<1> it) {
            src[it] = bf16(tmp[it]);  // init src
        });
    }).wait();
    free(tmp, q); free(host_f, q);

    int total_threads = 4096 * 256;
    int n_iter = (n + total_threads - 1) / total_threads;

    auto ev = q.submit([&](handler& h) {
        h.parallel_for(nd_range<1>(4096*256, 256),
            [=](nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            int tid = it.get_global_id(0);
            for (int i = 0; i < n_iter; i++) {
                size_t idx = (size_t)tid + (size_t)i * total_threads;
                if (idx < n) dst[idx] = src[idx];
            }
        });
    });
    ev.wait();

    // Verify correctness
    auto host_out = malloc_host<bf16>(n, q);
    q.memcpy(host_out, dst, n * sizeof(bf16)).wait();
    bool ok = true;
    for (size_t i = 0; i < n && ok; i++) {
        if (fabs((float)host_out[i] - 1.0f) > 0.01f) ok = false;
    }
    auto ns = ev.get_profiling_info<info::event_profiling::command_end>()
            - ev.get_profiling_info<info::event_profiling::command_start>();
    double bw = 2.0 * n * sizeof(bf16) / (ns * 1e-9) / 1e9;
    printf("bfloat16 scalar:     %7.1f GB/s  %s\n", bw, ok ? "PASS" : "FAIL");
    free(host_out, q); free(src, q); free(dst, q);
}

// ============================================================
// Test 2: bfloat16 vec4 store (using sycl::ext::oneapi::bfloat16)
// ============================================================
void test_bfloat16_vec4(queue& q, size_t n) {
    using bf16 = sycl::ext::oneapi::bfloat16;
    size_t n_vec = n / 4;
    auto src = malloc_device<bf16>(n, q);
    auto dst = malloc_device<bf16>(n, q);

    float* host_f = malloc_host<float>(n, q);
    for (size_t i = 0; i < n; i++) host_f[i] = 1.0f;
    auto tmp = malloc_device<float>(n, q);
    q.memcpy(tmp, host_f, n * sizeof(float)).wait();
    q.submit([&](handler& h) {
        h.parallel_for(range<1>(n), [=](item<1> it) {
            src[it] = bf16(tmp[it]);
        });
    }).wait();
    free(tmp, q); free(host_f, q);

    auto src4 = reinterpret_cast<sycl::vec<bf16, 4>*>(src);
    auto dst4 = reinterpret_cast<sycl::vec<bf16, 4>*>(dst);

    int total_threads = 4096 * 256;
    int n_iter = (n_vec + total_threads - 1) / total_threads;

    auto ev = q.submit([&](handler& h) {
        h.parallel_for(nd_range<1>(4096*256, 256),
            [=](nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            int tid = it.get_global_id(0);
            for (int i = 0; i < n_iter; i++) {
                size_t idx = (size_t)tid + (size_t)i * total_threads;
                if (idx < n_vec) dst4[idx] = src4[idx];
            }
        });
    });
    ev.wait();

    auto host_out = malloc_host<bf16>(n, q);
    q.memcpy(host_out, dst, n * sizeof(bf16)).wait();
    bool ok = true;
    for (size_t i = 0; i < n && ok; i++) {
        if (fabs((float)host_out[i] - 1.0f) > 0.01f) ok = false;
    }
    auto ns = ev.get_profiling_info<info::event_profiling::command_end>()
            - ev.get_profiling_info<info::event_profiling::command_start>();
    double bw = 2.0 * n * sizeof(bf16) / (ns * 1e-9) / 1e9;
    printf("bfloat16 vec4:       %7.1f GB/s  %s\n", bw, ok ? "PASS" : "FAIL");
    free(host_out, q); free(src, q); free(dst, q);
}

// ============================================================
// Test 3: sycl::vec<uint16_t, 4> (uint16_t as bf16)
// ============================================================
void test_sycl_vec_u16(queue& q, size_t n) {
    size_t n_vec = n / 4;
    auto src = malloc_device<uint16_t>(n, q);
    auto dst = malloc_device<uint16_t>(n, q);

    q.fill(src, (uint16_t)0x3F80, n).wait();  // bf16 1.0

    auto src4 = reinterpret_cast<sycl::vec<uint16_t, 4>*>(src);
    auto dst4 = reinterpret_cast<sycl::vec<uint16_t, 4>*>(dst);

    int total_threads = 4096 * 256;
    int n_iter = (n_vec + total_threads - 1) / total_threads;

    auto ev = q.submit([&](handler& h) {
        h.parallel_for(nd_range<1>(4096*256, 256),
            [=](nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            int tid = it.get_global_id(0);
            for (int i = 0; i < n_iter; i++) {
                size_t idx = (size_t)tid + (size_t)i * total_threads;
                if (idx < n_vec) dst4[idx] = src4[idx];
            }
        });
    });
    ev.wait();

    auto host_out = malloc_host<uint16_t>(n, q);
    q.memcpy(host_out, dst, n * sizeof(uint16_t)).wait();
    bool ok = true;
    for (size_t i = 0; i < n && ok; i++) {
        if (host_out[i] != 0x3F80) ok = false;
    }
    auto ns = ev.get_profiling_info<info::event_profiling::command_end>()
            - ev.get_profiling_info<info::event_profiling::command_start>();
    double bw = 2.0 * n * sizeof(uint16_t) / (ns * 1e-9) / 1e9;
    printf("sycl::vec<u16,4>:    %7.1f GB/s  %s\n", bw, ok ? "PASS" : "FAIL");
    free(host_out, q); free(src, q); free(dst, q);
}

// ============================================================
// Test 4: custom vec4_t<uint16_t> (from skill T1)
// ============================================================
void test_custom_vec4(queue& q, size_t n) {
    using vec4 = vec4_t<uint16_t>;
    size_t n_vec = n / 4;
    auto src = malloc_device<uint16_t>(n, q);
    auto dst = malloc_device<uint16_t>(n, q);

    q.fill(src, (uint16_t)0x3F80, n).wait();

    auto src4 = reinterpret_cast<const vec4*>(src);
    auto dst4 = reinterpret_cast<vec4*>(dst);

    int total_threads = 4096 * 256;
    int n_iter = (n_vec + total_threads - 1) / total_threads;

    auto ev = q.submit([&](handler& h) {
        h.parallel_for(nd_range<1>(4096*256, 256),
            [=](nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            int tid = it.get_global_id(0);
            for (int i = 0; i < n_iter; i++) {
                size_t idx = (size_t)tid + (size_t)i * total_threads;
                if (idx < n_vec) dst4[idx] = src4[idx];
            }
        });
    });
    ev.wait();

    auto host_out = malloc_host<uint16_t>(n, q);
    q.memcpy(host_out, dst, n * sizeof(uint16_t)).wait();
    bool ok = true;
    for (size_t i = 0; i < n && ok; i++) {
        if (host_out[i] != 0x3F80) ok = false;
    }
    auto ns = ev.get_profiling_info<info::event_profiling::command_end>()
            - ev.get_profiling_info<info::event_profiling::command_start>();
    double bw = 2.0 * n * sizeof(uint16_t) / (ns * 1e-9) / 1e9;
    printf("custom vec4_t<u16>:  %7.1f GB/s  %s\n", bw, ok ? "PASS" : "FAIL");
    free(host_out, q); free(src, q); free(dst, q);
}

// ============================================================
// Test 5: vec2_t<uint16_t> (32-bit, 2×bf16 → d32 store)
// ============================================================
void test_vec2(queue& q, size_t n) {
    using vec2 = vec2_t<uint16_t>;
    size_t n_vec = n / 2;
    auto src = malloc_device<uint16_t>(n, q);
    auto dst = malloc_device<uint16_t>(n, q);

    q.fill(src, (uint16_t)0x3F80, n).wait();

    auto src2 = reinterpret_cast<const vec2*>(src);
    auto dst2 = reinterpret_cast<vec2*>(dst);

    int total_threads = 4096 * 256;
    int n_iter = (n_vec + total_threads - 1) / total_threads;

    auto ev = q.submit([&](handler& h) {
        h.parallel_for(nd_range<1>(4096*256, 256),
            [=](nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            int tid = it.get_global_id(0);
            for (int i = 0; i < n_iter; i++) {
                size_t idx = (size_t)tid + (size_t)i * total_threads;
                if (idx < n_vec) dst2[idx] = src2[idx];
            }
        });
    });
    ev.wait();

    auto host_out = malloc_host<uint16_t>(n, q);
    q.memcpy(host_out, dst, n * sizeof(uint16_t)).wait();
    bool ok = true;
    for (size_t i = 0; i < n && ok; i++) {
        if (host_out[i] != 0x3F80) ok = false;
    }
    auto ns = ev.get_profiling_info<info::event_profiling::command_end>()
            - ev.get_profiling_info<info::event_profiling::command_start>();
    double bw = 2.0 * n * sizeof(uint16_t) / (ns * 1e-9) / 1e9;
    printf("custom vec2_t<u16>:  %7.1f GB/s  %s\n", bw, ok ? "PASS" : "FAIL");
    free(host_out, q); free(src, q); free(dst, q);
}

// ============================================================
// Test 6: accessor pattern (buffer + accessor) scalar bf16
// ============================================================
void test_accessor_scalar(queue& q, size_t n) {
    using bf16 = sycl::ext::oneapi::bfloat16;

    // Use buffer + accessor pattern
    std::vector<bf16> host_src(n, bf16(1.0f));
    std::vector<bf16> host_dst(n, bf16(0.0f));
    buffer<bf16, 1> src_buf(host_src.data(), range<1>(n));
    buffer<bf16, 1> dst_buf(host_dst.data(), range<1>(n));

    int total_threads = 4096 * 256;
    int n_iter = (n + total_threads - 1) / total_threads;

    auto ev = q.submit([&](handler& h) {
        accessor src_acc(src_buf, h, read_only);
        accessor dst_acc(dst_buf, h, write_only);
        h.parallel_for(nd_range<1>(4096*256, 256),
            [=](nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            int tid = it.get_global_id(0);
            for (int i = 0; i < n_iter; i++) {
                size_t idx = (size_t)tid + (size_t)i * total_threads;
                if (idx < n) dst_acc[idx] = src_acc[idx];
            }
        });
    });
    ev.wait();

    // Check result
    { accessor dst_acc(dst_buf, read_only); }
    bool ok = true;
    for (size_t i = 0; i < n && ok; i++) {
        if (fabs((float)host_dst[i] - 1.0f) > 0.01f) ok = false;
    }
    auto ns = ev.get_profiling_info<info::event_profiling::command_end>()
            - ev.get_profiling_info<info::event_profiling::command_start>();
    double bw = 2.0 * n * sizeof(bf16) / (ns * 1e-9) / 1e9;
    printf("accessor bf16 scalar:%7.1f GB/s  %s\n", bw, ok ? "PASS" : "FAIL");
}

// ============================================================
// Test 7: accessor pattern with vec4 (reinterpret on accessor)
// ============================================================
void test_accessor_vec4(queue& q, size_t n) {
    using bf16 = sycl::ext::oneapi::bfloat16;
    size_t n_vec = n / 4;

    std::vector<bf16> host_src(n, bf16(1.0f));
    std::vector<bf16> host_dst(n, bf16(0.0f));
    buffer<bf16, 1> src_buf(host_src.data(), range<1>(n));
    buffer<bf16, 1> dst_buf(host_dst.data(), range<1>(n));

    int total_threads = 4096 * 256;
    int n_iter = (n_vec + total_threads - 1) / total_threads;

    auto ev = q.submit([&](handler& h) {
        // Reinterpret buffer to vec4 type (requires range argument)
        auto src4_buf = src_buf.reinterpret<sycl::vec<bf16, 4>>(range<1>(n_vec));
        auto dst4_buf = dst_buf.reinterpret<sycl::vec<bf16, 4>>(range<1>(n_vec));
        auto src4 = src4_buf.get_access<access::mode::read>(h);
        auto dst4 = dst4_buf.get_access<access::mode::write>(h);

        h.parallel_for(nd_range<1>(4096*256, 256),
            [=](nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            int tid = it.get_global_id(0);
            for (int i = 0; i < n_iter; i++) {
                size_t idx = (size_t)tid + (size_t)i * total_threads;
                if (idx < n_vec) dst4[idx] = src4[idx];
            }
        });
    });
    ev.wait();

    { accessor dst_acc(dst_buf, read_only); }
    bool ok = true;
    for (size_t i = 0; i < n && ok; i++) {
        if (fabs((float)host_dst[i] - 1.0f) > 0.01f) ok = false;
    }
    auto ns = ev.get_profiling_info<info::event_profiling::command_end>()
            - ev.get_profiling_info<info::event_profiling::command_start>();
    double bw = 2.0 * n * sizeof(bf16) / (ns * 1e-9) / 1e9;
    printf("accessor bf16 vec4:  %7.1f GB/s  %s\n", bw, ok ? "PASS" : "FAIL");
}

// ============================================================
// Test 8: fp32 scalar baseline
// ============================================================
void test_fp32_baseline(queue& q, size_t n) {
    auto src = malloc_device<float>(n, q);
    auto dst = malloc_device<float>(n, q);

    q.fill(src, 1.0f, n).wait();

    int total_threads = 4096 * 256;
    int n_iter = (n + total_threads - 1) / total_threads;

    auto ev = q.submit([&](handler& h) {
        h.parallel_for(nd_range<1>(4096*256, 256),
            [=](nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            int tid = it.get_global_id(0);
            for (int i = 0; i < n_iter; i++) {
                size_t idx = (size_t)tid + (size_t)i * total_threads;
                if (idx < n) dst[idx] = src[idx];
            }
        });
    });
    ev.wait();

    auto ns = ev.get_profiling_info<info::event_profiling::command_end>()
            - ev.get_profiling_info<info::event_profiling::command_start>();
    double bw = 2.0 * n * sizeof(float) / (ns * 1e-9) / 1e9;
    printf("fp32 scalar:         %7.1f GB/s  (baseline)\n", bw);
    free(src, q); free(dst, q);
}

int main() {
    queue q(gpu_selector_v, property::queue::enable_profiling{});
    auto dev = q.get_device();
    printf("Device: %s\n\n", dev.get_info<info::device::name>().c_str());

    size_t n = 128 * 1024 * 1024;  // 128M elements = 256MB bf16, 512MB fp32
    int repeats = 3;

    printf("=== SYCL Type Alternative Verification ===\n");
    printf("Elements: %zu, bf16 data: %zu MB\n\n", n, n*2/1024/1024);

    // Warmup
    { auto tmp = malloc_device<float>(1024, q); free(tmp, q); }

    for (int r = 0; r < repeats; r++) {
        printf("--- Run %d ---\n", r+1);
        test_fp32_baseline(q, n);
        test_bfloat16_scalar(q, n);
        test_bfloat16_vec4(q, n);
        test_sycl_vec_u16(q, n);
        test_custom_vec4(q, n);
        test_vec2(q, n);
        test_accessor_scalar(q, n);
        test_accessor_vec4(q, n);
        printf("\n");
    }

    return 0;
}
