// Microbenchmark 5: DPAS/XMX Throughput Characterization
// Uses joint_matrix API (verified to produce dpas.8x8 GEN ASM)
// Analogous to paper Section V (Tensor Core)
//
// Usage:
//   ./bench_dpas_throughput
//   unitrace --metric-query --group ComputeBasic ./bench_dpas_throughput
//   unitrace --metric-query --group VectorEngineStalls ./bench_dpas_throughput

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <cstdio>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

int main() {
    queue q{gpu_selector_v, property::queue::enable_profiling{}};
    auto dev = q.get_device();
    double ghz = dev.get_info<info::device::max_clock_frequency>() / 1000.0;

    printf("=== DPAS/XMX Throughput (joint_matrix BF16) ===\n");
    printf("Device: %s  Clock: %.1f GHz\n",
           dev.get_info<info::device::name>().c_str(), ghz);

    // BF16 joint_matrix parameters for XMX on BMG
    // Tile: 8×16×16 (M×N×K), sub-group size 16
    constexpr int TM = 8, TN = 16, TK = 16;
    // Each dpas = 2*M*N*K = 2*8*16*16 = 4096 FLOPs
    constexpr int FLOPS_PER_DPAS = 2 * TM * TN * TK;

    // ========== Sweep 1: ILP vs Sub-group count ==========
    printf("\n--- DPAS ILP x Sub-group Sweep (N_ITER=512) ---\n");
    constexpr int N_ITER = 512;
    constexpr int WARMUP = 5, REPEAT = 100;

    for (int n_mad : {1, 2, 4, 8}) {  // ILP = number of independent joint_matrix_mad
        printf("\n  ILP=%d (independent MAD chains):\n", n_mad);
        printf("  %-4s %10s %10s %10s\n", "SG", "avg(ns)", "TFLOPS", "cyc/dpas");

        for (int nsg : {1, 2, 4, 8, 16, 32, 64, 128}) {
            int wg_size = nsg * 16;
            size_t total_floats = (size_t)wg_size * n_mad * TM * TN * N_ITER;
            size_t alloc_size = total_floats * sizeof(float) + 1024 * 1024;  // extra

            buffer<float, 1> buf(alloc_size / sizeof(float));
            double total_ns = 0;

            for (int r = 0; r < WARMUP + REPEAT; r++) {
                auto ev = q.submit([&](handler& h) {
                    auto acc = buf.get_access<access::mode::write>(h);
                    h.parallel_for(nd_range<1>(wg_size, wg_size),
                        [=](nd_item<1> it) [[sycl::reqd_sub_group_size(16)]]
                    {
                        auto sg = it.get_sub_group();
                        int sg_id = sg.get_group_id()[0];

                        // Accumulator tiles (independent chains)
                        joint_matrix<sub_group, float, use::accumulator, TM, TN> jm_acc[8];
                        for (int m = 0; m < n_mad; m++)
                            jm_acc[m] = 0.0f;

                        // A and B operands (constant)
                        joint_matrix<sub_group, float, use::accumulator, TM, TK> jm_a_val;
                        joint_matrix<sub_group, float, use::accumulator, TK, TN> jm_b_val;
                        jm_a_val = 0.5f;
                        jm_b_val = 0.5f;

                        // Main loop
                        #pragma unroll 1
                        for (int i = 0; i < N_ITER; i++) {
                            for (int m = 0; m < n_mad; m++) {
                                // Use acc as both input and output (dependent chain)
                                // This is equivalent to MMA with accumulator feedback
                                auto a = reinterpret_cast<joint_matrix<sub_group, sycl::half, use::a, TM, TK, layout::row_major>&>(jm_a_val);
                                auto b = reinterpret_cast<joint_matrix<sub_group, sycl::half, use::b, TK, TN, layout::ext_intel_packed>&>(jm_b_val);
                                joint_matrix_mad(sg, jm_acc[m], a, b, jm_acc[m]);
                            }
                        }

                        // Write back to prevent DCE
                        float* out = acc.get_pointer();
                        for (int m = 0; m < n_mad; m++) {
                            joint_matrix_store(sg, jm_acc[m],
                                out + sg_id * n_mad * TM * TN + m * TM * TN, TN);
                        }
                    });
                });
                ev.wait();
                if (r >= WARMUP) {
                    auto t0 = ev.get_profiling_info<info::event_profiling::command_start>();
                    auto t1 = ev.get_profiling_info<info::event_profiling::command_end>();
                    total_ns += (t1 - t0);
                }
            }

            double avg_ns = total_ns / REPEAT;
            double total_flops = (double)N_ITER * n_mad * FLOPS_PER_DPAS * nsg;
            double tflops = total_flops / (avg_ns * 1e-9) / 1e12;
            double cyc_per_dpas = avg_ns * ghz / (N_ITER * n_mad);
            printf("  %-4d %10.1f %10.3f %10.1f\n", nsg, avg_ns, tflops, cyc_per_dpas);
        }
    }

    return 0;
}
