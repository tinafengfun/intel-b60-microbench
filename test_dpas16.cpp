// Systolic-depth-16 probe: does BMG-G31 XMX accept dpas with SystolicDepth=16?
// On Xe2 (depth 8), K = 8*2 = 16 for fp16. Depth 16 would double K to 32 per
// instruction (8192 FLOP/instr fp16). Compile-time probe first, then rate check.
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <cstdio>
#include <algorithm>
#include <vector>
using namespace sycl;
using namespace sycl::ext::intel::esimd;

int main(int argc, char **argv) {
  queue q(property::queue::enable_profiling{});
  printf("Device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
  int n_iter = argc > 1 ? atoi(argv[1]) : 16384;
  int n_wi = argc > 2 ? atoi(argv[2]) : 2048;

  half *a = malloc_shared<half>(256, q);
  half *b = malloc_shared<half>(512, q);
  float *c = malloc_shared<float>(4 * 128, q);
  for (int i = 0; i < 256; i++) a[i] = half(0.01f);
  for (int i = 0; i < 512; i++) b[i] = half(0.01f);
  for (int i = 0; i < 4 * 128; i++) c[i] = 1.0f;

  std::vector<double> times;
  for (int r = 0; r < 7; r++) {
    auto e = q.submit([&](handler &h) {
      h.parallel_for(nd_range<1>(range<1>(n_wi), range<1>(8)),
                     [=](nd_item<1> it) SYCL_ESIMD_KERNEL {
        simd<half, 256> A; A.copy_from(a);   // M8 x K32 (if depth16 works)
        simd<half, 512> B; B.copy_from(b);   // K32 x N16
        simd<float, 128> acc0, acc1, acc2, acc3;
        acc0.copy_from(c); acc1.copy_from(c + 128);
        acc2.copy_from(c + 256); acc3.copy_from(c + 384);
        for (int i = 0; i < n_iter; i++) {
          acc0 = xmx::dpas<16, 8, float>(acc0, B, A);
          acc1 = xmx::dpas<16, 8, float>(acc1, B, A);
          acc2 = xmx::dpas<16, 8, float>(acc2, B, A);
          acc3 = xmx::dpas<16, 8, float>(acc3, B, A);
        }
        acc0.copy_to(c); acc1.copy_to(c + 128);
        acc2.copy_to(c + 256); acc3.copy_to(c + 384);
      });
    });
    e.wait();
    uint64_t t0 = e.template get_profiling_info<sycl::info::event_profiling::command_start>();
    uint64_t t1 = e.template get_profiling_info<sycl::info::event_profiling::command_end>();
    times.push_back(double(t1 - t0));
  }
  std::sort(times.begin(), times.end());
  double med = times[times.size() / 2];
  // depth16 fp16: 2*8*16*32 = 8192 FLOP per call
  double tf = 4.0 * n_iter * n_wi * 8192.0 / (med * 1e-9) / 1e12;
  printf("dpas<16,8> fp16 n_wi=%4d: med=%12.0f ns  %7.3f ns/instr -> %8.1f TF (if 8192 FLOP/instr)  [c0=%.1f]\n",
         n_wi, med, med / (4.0 * n_iter), tf, c[0]);
  return 0;
}
