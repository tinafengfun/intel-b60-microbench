// Microbenchmark feasibility test: verify intrinsics produce correct GEN ASM
// Compile:
//   source /opt/intel/oneapi/setvars.sh
//   icpx -fsycl -fsycl-targets=intel_gpu_bmg_g21 -O3 -std=c++17 -o test_intrinsics test_intrinsics.cpp

#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <cstdio>

// Use ext_vector_type directly (avoids SYCL 2020 alias collisions)
// These match the intel_sub_group_* intrinsic signatures
using sshort8 = short   __attribute__((ext_vector_type(8)));
using sint8   = int     __attribute__((ext_vector_type(8)));
using sfloat8 = float   __attribute__((ext_vector_type(8)));

// intel_sub_group DPAS intrinsic (BF16 k16) — device only
#ifdef __SYCL_DEVICE_ONLY__
SYCL_EXTERNAL sfloat8 intel_sub_group_bf16_bf16_matrix_mad_k16(sshort8 a, sint8 b, sfloat8 acc);
#else
// Host stub — never called, just for linking
inline sfloat8 intel_sub_group_bf16_bf16_matrix_mad_k16(sshort8 a, sint8 b, sfloat8 acc) { return acc; }
#endif

int main() {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::enable_profiling{}};

    auto dev = q.get_device();
    std::cout << "=== Intel Arc Pro B60 Microbenchmark Feasibility Test ===" << std::endl;
    std::cout << "Device: " << dev.get_info<sycl::info::device::name>() << std::endl;

    double ghz = dev.get_info<sycl::info::device::max_clock_frequency>() / 1000.0;
    std::cout << "Max clock: " << ghz << " GHz" << std::endl;
    std::cout << "Local mem: " << dev.get_info<sycl::info::device::local_mem_size>() << " bytes" << std::endl;
    std::cout << std::endl;

    // ========== Test 1: FP32 Latency vs N ==========
    {
        std::cout << "=== Test 1: FP32 Dependent Chain ===" << std::endl;
        constexpr int WARMUP = 5, REPEAT = 50;
        int sizes[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};

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
                            x = x * x + 1.0f;
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
            double cycles_per_op = avg_ns * ghz / N;
            printf("  N=%4d  avg=%7.1f ns  cyc/op=%6.2f\n", N, avg_ns, cycles_per_op);
        }
        std::cout << std::endl;
    }

    // ========== Test 2: Sub-group Count Sweep ==========
    {
        std::cout << "=== Test 2: Sub-group Count Sweep (FP32, N=256) ===" << std::endl;
        constexpr int WARMUP = 5, REPEAT = 50, N = 256;
        int sg_counts[] = {1, 2, 4, 8, 16, 32, 64};

        for (int nsg : sg_counts) {
            int wg = nsg * 16;
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
            double throughput = (double)wg * N / cycles;
            printf("  SG=%2d  threads=%3d  avg=%7.1f ns  ops/cyc=%7.1f\n",
                   nsg, wg, avg_ns, throughput);
        }
        std::cout << std::endl;
    }

    // ========== Test 3: DPAS intrinsic ==========
    {
        std::cout << "=== Test 3: DPAS via intel_sub_group_bf16_bf16_matrix_mad_k16 ===" << std::endl;
        constexpr int REPEAT = 100, WARMUP = 10, N_DPAS = 1024;

        sycl::buffer<float, 1> buf(8);
        double total_ns = 0;

        for (int r = 0; r < WARMUP + REPEAT; r++) {
            auto ev = q.submit([&](sycl::handler& h) {
                auto acc = buf.get_access<sycl::access::mode::write>(h);
                h.parallel_for(sycl::nd_range<1>(16, 16),
                    [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]]
                {
                    sshort8 a = (sshort8)(0x3f00);
                    sint8   b = (sint8)(1);
                    sfloat8 acc_v = (sfloat8)(0.0f);

                    #pragma unroll 1
                    for (int i = 0; i < N_DPAS; i++) {
                        acc_v = intel_sub_group_bf16_bf16_matrix_mad_k16(a, b, acc_v);
                    }

                    if (it.get_local_id(0) == 0) {
                        for (int j = 0; j < 8; j++) acc[j] = acc_v[j];
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
        double cyc_per_dpas = avg_ns * ghz / N_DPAS;
        constexpr int FLOPS_PER_DPAS = 8 * 16 * 2;  // 256
        double tflops = (double)N_DPAS * FLOPS_PER_DPAS / (avg_ns * 1e-9) / 1e12;

        printf("  DPAS count=%d  FLOPs/DPAS=%d\n", N_DPAS, FLOPS_PER_DPAS);
        printf("  Avg time=%.1f ns  cycles/DPAS=%.2f\n", avg_ns, cyc_per_dpas);
        printf("  Throughput=%.3f TFLOPS (1 sub-group)\n", tflops);
        std::cout << std::endl;
    }

    // ========== Test 4: DPAS ILP Sweep ==========
    {
        std::cout << "=== Test 4: DPAS ILP x Sub-group Sweep (BF16) ===" << std::endl;
        constexpr int REPEAT = 50, WARMUP = 5, N_ITER = 512;

        for (int ilp : {1, 2, 4}) {
            printf("\n  --- ILP = %d ---\n", ilp);
            for (int nsg : {1, 2, 4, 8, 16, 32}) {
                int wg = nsg * 16;
                sycl::buffer<float, 1> buf(wg);
                double total_ns = 0;

                for (int r = 0; r < WARMUP + REPEAT; r++) {
                    auto ev = q.submit([&](sycl::handler& h) {
                        auto acc = buf.get_access<sycl::access::mode::write>(h);
                        h.parallel_for(sycl::nd_range<1>(wg, wg),
                            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]]
                        {
                            sshort8 a = (sshort8)(0x3f00);
                            sint8   b = (sint8)(1);
                            sfloat8 c0 = (sfloat8)(0.0f), c1 = (sfloat8)(0.0f);
                            sfloat8 c2 = (sfloat8)(0.0f), c3 = (sfloat8)(0.0f);

                            #pragma unroll 1
                            for (int i = 0; i < N_ITER; i++) {
                                if (ilp >= 1) c0 = intel_sub_group_bf16_bf16_matrix_mad_k16(a, b, c0);
                                if (ilp >= 2) c1 = intel_sub_group_bf16_bf16_matrix_mad_k16(a, b, c1);
                                if (ilp >= 4) {
                                    c2 = intel_sub_group_bf16_bf16_matrix_mad_k16(a, b, c2);
                                    c3 = intel_sub_group_bf16_bf16_matrix_mad_k16(a, b, c3);
                                }
                            }
                            acc[it.get_local_id(0)] = c0[0] + c1[0] + c2[0] + c3[0];
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
                double tflops = (double)N_ITER * ilp * 256.0 * nsg / (avg_ns * 1e-9) / 1e12;
                printf("    SG=%2d  time=%7.1f ns  TFLOPS=%.3f\n", nsg, avg_ns, tflops);
            }
        }
        std::cout << std::endl;
    }

    // ========== Test 5: Pointer Chase Memory Latency ==========
    {
        std::cout << "=== Test 5: Pointer Chase - Memory Hierarchy ===" << std::endl;
        constexpr int REPEAT = 30, WARMUP = 3, CHASE = 4096;

        size_t sizes[] = {4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144};
        for (auto sz_kb : sizes) {
            size_t buf_bytes = sz_kb * 1024UL;
            size_t n = buf_bytes / sizeof(int);
            if (n < (size_t)CHASE) n = CHASE;

            std::vector<int> next_h(n);
            for (size_t i = 0; i < n; i++) next_h[i] = (int)((i+1) % n);
            for (size_t i = n-1; i > 0; i--) {
                size_t j = rand() % (i+1);
                std::swap(next_h[i], next_h[j]);
            }

            sycl::buffer<int, 1> next_buf(next_h.data(), sycl::range<1>(n));
            sycl::buffer<int, 1> res_buf(1);
            double total_ns = 0;

            for (int r = 0; r < WARMUP + REPEAT; r++) {
                auto ev = q.submit([&](sycl::handler& h) {
                    auto nacc = next_buf.get_access<sycl::access::mode::read>(h);
                    auto racc = res_buf.get_access<sycl::access::mode::write>(h);
                    h.parallel_for(sycl::nd_range<1>(1, 1), [=](sycl::nd_item<1>) {
                        int idx = 0;
                        #pragma unroll 1
                        for (int i = 0; i < CHASE; i++) idx = nacc[idx];
                        racc[0] = idx;
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
            double cyc = avg_ns * ghz / CHASE;
            printf("  buf=%6zuKB  avg=%8.1f ns  cyc/access=%6.1f\n", sz_kb, avg_ns, cyc);
        }
    }

    std::cout << "\n=== Done. Run with IGC_ShaderDumpEnable=1 to verify GEN ASM. ===" << std::endl;
    return 0;
}
