// WARNING: superseded by bench_vrt_probe2.cpp — see attic/README.md.
// VRT (Variable Register allocation per Thread) probe for BMG-G31.
// Idea: each work-item holds a live register payload of NGRF GRFs (1 GRF = 8 floats)
// across a latency-bound dependent-FMA loop. EU throughput of dependent chains
// scales linearly with resident threads/EU (T), so measured aggregate throughput
// directly reveals occupancy T(NGRF).
//   - Fixed allocation (Xe2 style): T steps 8 -> 4 -> ... at coarse GRF thresholds.
//   - VRT (Xe3 style): T degrades smoothly ~ floor(512/(NGRF+ovh)).
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <vector>

using namespace sycl;
using namespace sycl::ext::intel::esimd;

template <int NGRF>
double run_config(queue &q, int n_wi, int n_iter, float *buf, float *out) {
  std::vector<double> times;
  for (int r = 0; r < 5; r++) {
    auto e = q.submit([&](handler &h) {
      h.parallel_for(nd_range<1>(range<1>(n_wi), range<1>(8)),
                     [=](nd_item<1> it) SYCL_ESIMD_KERNEL {
        simd<float, NGRF * 8> p;
        p.copy_from(buf);            // live payload, ~NGRF GRFs
        float acc = p[0] + it.get_global_id(0) * 1e-9f;
        for (int i = 0; i < n_iter; i++)
          acc = acc * 1.0000001f + 0.0000001f;   // dependent chain, latency-bound
        float s = acc;
#pragma unroll 4
        for (int i = 0; i < NGRF * 8; i++) s += p[i]; // keep payload live to the end
        out[it.get_global_id(0)] = s;
      });
    });
    e.wait();
    uint64_t t0 = e.template get_profiling_info<sycl::info::event_profiling::command_start>();
    uint64_t t1 = e.template get_profiling_info<sycl::info::event_profiling::command_end>();
    times.push_back(double(t1 - t0));
  }
  std::sort(times.begin(), times.end());
  return times[times.size() / 2];
}

int main(int argc, char **argv) {
  queue q(property::queue::enable_profiling{});
  printf("Device: %s  EUs: %d\n",
         q.get_device().get_info<sycl::info::device::name>().c_str(),
         q.get_device().get_info<sycl::info::device::max_compute_units>());

  int n_wi = argc > 1 ? atoi(argv[1]) : 2048;    // 256 EU x 8 threads
  int n_iter = argc > 2 ? atoi(argv[2]) : 100000;
  float *buf = malloc_shared<float>(8 * 256, q);
  float *out = malloc_shared<float>(n_wi, q);
  for (int i = 0; i < 8 * 256; i++) buf[i] = 0.5f + i * 1e-6f;

  double total_chains = double(n_wi) * n_iter;   // one FMA per chain-step
  printf("GRFs/thread   median_ns    ns/chain   T_est(threads/EU)\n");

  double ref_ns_per_chain = 0;
#define RUN(NG) { \
    double med = run_config<NG>(q, n_wi, n_iter, buf, out); \
    double ns_chain = med / total_chains; \
    if (NG == 8) ref_ns_per_chain = ns_chain; \
    double t_est = ref_ns_per_chain / ns_chain * 8.0; \
    printf("%5d        %12.0f   %8.4f      %6.2f\n", NG, med, ns_chain, t_est); }
  RUN(8) RUN(16) RUN(24) RUN(32) RUN(40) RUN(48) RUN(56) RUN(64)
  RUN(72) RUN(80) RUN(88) RUN(96) RUN(104) RUN(112) RUN(120) RUN(128)
#undef RUN

  printf("out[0]=%.3f (sanity)\n", out[0]);
  free(buf, q); free(out, q);
  return 0;
}
