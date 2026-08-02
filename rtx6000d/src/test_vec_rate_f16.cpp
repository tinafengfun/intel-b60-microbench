// B70 (Xe2) vector-engine FMA rate by datatype: f32 vs f16 vs bf16.
// Same ILP-chain methodology as test_xe3_features.cpp mode 1.
// Run: ./test_vec_rate_f16 [n_wi] [n_iter]
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <cstdio>
#include <algorithm>
#include <vector>

using namespace sycl;
using bf16 = sycl::ext::oneapi::bfloat16;

template <typename T, int ILP>
void run_rate(queue &q, int n_wi, int n_iter, const char *name) {
  using namespace sycl::ext::intel::esimd;
  T *buf = malloc_shared<T>(256, q);
  for (int i = 0; i < 256; i++) buf[i] = T(0.5f + 0.001f * (i % 50));
  std::vector<double> times;
  for (int r = 0; r < 7; r++) {
    auto e = q.submit([&](handler &h) {
      h.parallel_for(nd_range<1>(range<1>(n_wi), range<1>(8)),
                     [=](nd_item<1> it) SYCL_ESIMD_KERNEL {
        simd<T, 32> a, b;
        a.copy_from(buf); b.copy_from(buf + 32);
        simd<T, 32> acc[ILP];
        for (int j = 0; j < ILP; j++) acc[j].copy_from(buf + (j * 32) % 128);
        for (int i = 0; i < n_iter; i++) {
#pragma unroll
          for (int j = 0; j < ILP; j++) acc[j] = a * acc[j] + b;
          a = a * T(0.9999999f) + b * T(1e-8f);  // defeat loop folding
        }
        simd<T, 32> s = 0;
        for (int j = 0; j < ILP; j++) s += acc[j];
        T sv = s[0];
        float fv = float(sv);
        if (fv != 0.0f) s.copy_to(buf);  // guard must not be provably false/dead
      });
    });
    e.wait();
    uint64_t t0 = e.template get_profiling_info<sycl::info::event_profiling::command_start>();
    uint64_t t1 = e.template get_profiling_info<sycl::info::event_profiling::command_end>();
    times.push_back(double(t1 - t0));
  }
  std::sort(times.begin(), times.end());
  double med = times[times.size() / 2];
  double ops = double(ILP) * n_iter * n_wi * 32;
  double gops = ops / (med * 1e-9) / 1e9;
  printf("  %-6s n_wi=%5d: %9.2f GOP/s FMA  (%6.2f lane-op/cyc/EU @2.4GHz,256EU)  = %6.2f TFLOPS\n",
         name, n_wi, gops, gops * 1e9 / (256 * 2.4e9), gops * 2 / 1000.0);
  free(buf, q);
}

int main(int argc, char **argv) {
  int n_wi = argc > 1 ? atoi(argv[1]) : 2048;
  int n_iter = argc > 2 ? atoi(argv[2]) : 20000;
  queue q(gpu_selector_v, property::queue::enable_profiling{});
  printf("B70 vector FMA rate (ESIMD, ILP=8, 8 WI/WG):\n");
  run_rate<float, 8>(q, n_wi, n_iter, "f32");
  run_rate<sycl::half, 8>(q, n_wi, n_iter, "f16");
  run_rate<bf16, 8>(q, n_wi, n_iter, "bf16");
  return 0;
}
