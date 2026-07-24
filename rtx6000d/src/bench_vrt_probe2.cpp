// VRT probe v2 for BMG-G31.
// Per register-payload size:
//   1) query the kernel's actual register count (num_regs) and spill size
//   2) occupancy knee sweep: time(n_wi) is flat while n_wi <= resident capacity,
//      then grows linearly -> knee reveals true resident threads/EU.
// Fixed allocation (Xe2): occupancy steps at coarse GRF configs.
// VRT (Xe3): occupancy tracks floor(512/num_regs) smoothly.
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <cstdio>
#include <algorithm>
#include <vector>

using namespace sycl;
using namespace sycl::ext::intel::esimd;

template <int NGRF>
struct VRTKernel {
  const float *buf;
  float *out;
  int n_iter;
  void operator()(nd_item<1> it) const SYCL_ESIMD_KERNEL {
    simd<float, NGRF * 8> p;
    p.copy_from(buf);
    float acc = p[0] + it.get_global_id(0) * 1e-9f;
    for (int i = 0; i < n_iter; i++) {
      acc = acc * 1.0000001f + 0.0000001f;
      // dynamic index forces the whole array to live in GRFs (un-rematerializable)
      p[(unsigned)i & (NGRF * 8 - 1)] += acc;
    }
    float s = acc;
#pragma unroll 4
    for (int i = 0; i < NGRF * 8; i++) s += p[i];
    out[it.get_global_id(0)] = s;
  }
};

template <int NGRF>
void analyze(queue &q, float *buf, float *out, int n_iter) {
  auto dev = q.get_device();
  try {
    auto kb = get_kernel_bundle<bundle_state::executable>(q.get_context(), {dev},
                                                          {get_kernel_id<VRTKernel<NGRF>>()});
    kernel k = kb.get_kernel(get_kernel_id<VRTKernel<NGRF>>());
    uint32_t nr = k.get_info<sycl::info::kernel_device_specific::ext_codeplay_num_regs>(dev);
    size_t spill = 0;
    if (dev.has(sycl::aspect::ext_intel_spill_memory_size))
      spill = k.get_info<sycl::ext::intel::info::kernel_device_specific::spill_memory_size>(dev);
    printf("payload=%3d GRF | num_regs=%3u  spill=%6zu B | knee sweep ns/chain: ", NGRF, nr, spill);
  } catch (std::exception &e) {
    printf("payload=%3d GRF | query failed (%s) | ", NGRF, e.what());
  }

  // knee sweep: 1..8 threads/EU
  for (int t = 1; t <= 8; t++) {
    int n_wi = 256 * t;
    std::vector<double> times;
    for (int r = 0; r < 3; r++) {
      auto e = q.submit([&](handler &h) {
        h.parallel_for(nd_range<1>(range<1>(n_wi), range<1>(8)),
                       VRTKernel<NGRF>{buf, out, n_iter});
      });
      e.wait();
      uint64_t t0 = e.template get_profiling_info<sycl::info::event_profiling::command_start>();
      uint64_t t1 = e.template get_profiling_info<sycl::info::event_profiling::command_end>();
      times.push_back(double(t1 - t0));
    }
    std::sort(times.begin(), times.end());
    printf("%.3f ", times[1] / (double(n_wi) * n_iter) * 1000.0); // ps/chain-step
  }
  printf("\n");
}

int main(int argc, char **argv) {
  queue q(property::queue::enable_profiling{});
  int n_iter = argc > 1 ? atoi(argv[1]) : 200000;
  float *buf = malloc_shared<float>(8 * 128, q);
  float *out = malloc_shared<float>(2048, q);
  for (int i = 0; i < 8 * 128; i++) buf[i] = 0.5f + i * 1e-6f;
  printf("(knee sweep columns = 1..8 threads/EU; ps/chain-step, flat-then-rising => knee)\n");
  analyze<16>(q, buf, out, n_iter);
  analyze<32>(q, buf, out, n_iter);
  analyze<48>(q, buf, out, n_iter);
  analyze<64>(q, buf, out, n_iter);
  analyze<80>(q, buf, out, n_iter);
  analyze<96>(q, buf, out, n_iter);
  analyze<112>(q, buf, out, n_iter);
  analyze<128>(q, buf, out, n_iter);
  printf("out[0]=%.2f\n", out[0]);
  free(buf, q); free(out, q);
  return 0;
}
