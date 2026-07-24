// Clean cooperative-matrix (joint_matrix) DPAS throughput benchmark for BMG-G31 (B70).
// Mirrors bench_esimd_dpas.cpp mode 0 exactly: same n_wi, n_iter, ILP sweep, timing.
// Metric: cyc/dpas per EU = med_ns * GHz / ((n_wi/EUs) * ILP * n_iter), peak = 16.0.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <cstdio>
#include <algorithm>
#include <vector>

using namespace sycl;
using sycl::ext::oneapi::bfloat16;
using namespace sycl::ext::oneapi::experimental::matrix;

static constexpr int TM = 8, TN = 16, TK = 16;   // one mad = 4096 FLOP (2*8*16*16)
static constexpr int FLOPS_PER_DPAS = 4096;      // per-thread dpas.8x8 instruction

template <int ILP>
void run_cfg(queue &q, int n_wi, int n_iter, int repeats) {
  float *c = malloc_shared<float>(128, q);   // DCE guard landing zone
  for (int i = 0; i < 128; i++) c[i] = 0.0f;

  std::vector<double> times;
  for (int r = 0; r < repeats; r++) {
    auto e = q.submit([&](handler &h) {
      h.parallel_for(nd_range<1>(range<1>(n_wi), range<1>(16)),
                     [=](nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        auto sg = it.get_sub_group();
        joint_matrix<sub_group, bfloat16, use::a, TM, TK, layout::row_major> jmA;
        joint_matrix<sub_group, bfloat16, use::b, TK, TN, layout::ext_intel_packed> jmB;
        joint_matrix<sub_group, float, use::accumulator, TM, TN, layout::dynamic> acc[ILP];
        joint_matrix_fill(sg, jmA, bfloat16(0.01f));
        joint_matrix_fill(sg, jmB, bfloat16(0.01f));
        for (int j = 0; j < ILP; j++) joint_matrix_fill(sg, acc[j], 1.0f);
        for (int i = 0; i < n_iter; i++) {
#pragma unroll
          for (int j = 0; j < ILP; j++)
            joint_matrix_mad(sg, acc[j], jmA, jmB, acc[j]);
        }
        float sink = 0.0f;
        for (int j = 0; j < ILP; j++)
          joint_matrix_apply(sg, acc[j], [&](float &x) { sink += x; });
        if (sink == 12345.678f) c[it.get_global_id(0) % 128] = sink;  // DCE guard
      });
    });
    e.wait();
    uint64_t t0 = e.template get_profiling_info<sycl::info::event_profiling::command_start>();
    uint64_t t1 = e.template get_profiling_info<sycl::info::event_profiling::command_end>();
    times.push_back(double(t1 - t0));
  }
  std::sort(times.begin(), times.end());
  double med = times[times.size() / 2];
  double total_dpas = double(ILP) * n_iter * n_wi;   // per-thread dpas instructions
  double tf = total_dpas * FLOPS_PER_DPAS / (med * 1e-9) / 1e12;
  double cyc = med * 2.4 / ((n_wi / 256.0) * ILP * n_iter);
  printf("coop ILP=%d n_wi=%4d n_iter=%6d  med=%12.0f ns  -> %6.1f TF  %6.2f cyc/dpas  [c0=%.1f]\n",
         ILP, n_wi, n_iter, med, tf, cyc, c[0]);
  free(c, q);
}

int main(int argc, char **argv) {
  queue q(property::queue::enable_profiling{});
  auto dev = q.get_device();
  printf("Device: %s  EUs: %d\n", dev.get_info<sycl::info::device::name>().c_str(),
         dev.get_info<sycl::info::device::max_compute_units>());
  int n_iter = argc > 1 ? atoi(argv[1]) : 16384;
  int n_wi = argc > 2 ? atoi(argv[2]) : 2048;
  run_cfg<1>(q, n_wi, n_iter, 7);
  run_cfg<2>(q, n_wi, n_iter, 7);
  run_cfg<4>(q, n_wi, n_iter, 7);
  run_cfg<8>(q, n_wi, n_iter, 7);
  return 0;
}
