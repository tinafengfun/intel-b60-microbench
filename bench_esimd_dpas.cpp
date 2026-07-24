// Raw ESIMD DPAS throughput benchmark for BMG-G31 (B70).
// Bypasses the cooperative-matrix layer: each dpas<8,8> intrinsic maps 1:1 to a
// hardware DPAS instruction (M=8, K=16, N=16, fp16 -> 4096 FLOP).
// Question: is the ~21.1 cyc/dpas measured via cooperative matrix an IGC codegen
// artifact, or the true XMX issue rate (B60 does 16.1 => ~160 TF would be possible)?
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <vector>

using namespace sycl;
using namespace sycl::ext::intel::esimd;

static constexpr int FLOPS_PER_DPAS = 4096;

template <int ILP, int REPEAT>
void run_config(queue &q, int n_wi, int n_iter, int repeats, std::vector<double> &med_out) {
  half *a = malloc_shared<half>(128, q);
  half *b = malloc_shared<half>(256, q);
  float *c = malloc_shared<float>(ILP * 128, q);
  for (int i = 0; i < 128; i++) a[i] = half(0.01f);
  for (int i = 0; i < 256; i++) b[i] = half(0.01f);
  for (int i = 0; i < ILP * 128; i++) c[i] = 1.0f;

  std::vector<double> times;
  for (int r = 0; r < repeats; r++) {
    auto e = q.submit([&](handler &h) {
      h.parallel_for(nd_range<1>(range<1>(n_wi), range<1>(8)),
                     [=](nd_item<1> it) SYCL_ESIMD_KERNEL {
        simd<half, 16 * REPEAT> A;
        simd<half, 256> B;
        A.copy_from(a);
        B.copy_from(b);
        simd<float, 16 * REPEAT> acc[ILP];
        for (int j = 0; j < ILP; j++) acc[j].copy_from(c + (j % ILP) * 128);
        for (int i = 0; i < n_iter; i++) {
#pragma unroll
          for (int j = 0; j < ILP; j++)
            acc[j] = xmx::dpas<8, REPEAT, float>(acc[j], B, A);
        }
        for (int j = 0; j < ILP; j++) acc[j].copy_to(c + j * 128);
      });
    });
    e.wait();
    uint64_t t0 = e.template get_profiling_info<sycl::info::event_profiling::command_start>();
    uint64_t t1 = e.template get_profiling_info<sycl::info::event_profiling::command_end>();
    times.push_back(double(t1 - t0));
  }
  std::sort(times.begin(), times.end());
  double med = times[times.size() / 2];
  // sanity: result must be finite and large (accumulated)
  float v = c[0];
  med_out.push_back(med);
  double total_dpas = double(ILP) * n_iter * n_wi;
  double flops = total_dpas * REPEAT * 512.0;  // per-instr work scales with REPEAT
  double tf = flops / (med * 1e-9) / 1e12;
  double ns_per_dpas = med / (double(ILP) * n_iter);
  printf("ILP=%d R=%d n_wi=%4d n_iter=%6d  med=%12.0f ns  %7.3f ns/instr  -> %8.1f TF  [c0=%.2f]\n",
         ILP, REPEAT, n_wi, n_iter, med, ns_per_dpas, tf, v);
  free(a, q); free(b, q); free(c, q);
}

int main(int argc, char **argv) {
  queue q(property::queue::enable_profiling{});
  auto dev = q.get_device();
  printf("Device: %s  EUs: %d\n", dev.get_info<sycl::info::device::name>().c_str(),
         dev.get_info<sycl::info::device::max_compute_units>());

  int n_iter = argc > 1 ? atoi(argv[1]) : 16384;
  int n_wi = argc > 2 ? atoi(argv[2]) : 2048;   // 2048 = 8 threads/EU, 256 = 1/EU
  int mode = argc > 3 ? atoi(argv[3]) : 0;      // 0=ILP sweep R=8, 1=repeat sweep ILP=4, 2=distinct-B ILP=4
  std::vector<double> _;

  if (mode == 3) {
    // bf16 vs fp16 rate check (coop-matrix asm showed :bf operands)
    uint16_t *a3 = malloc_shared<uint16_t>(128, q);
    uint16_t *b3 = malloc_shared<uint16_t>(256, q);
    float *c3 = malloc_shared<float>(4 * 128, q);
    for (int i = 0; i < 128; i++) a3[i] = 0x3c23;      // ~0.01 bf16
    for (int i = 0; i < 256; i++) b3[i] = 0x3c23;
    for (int i = 0; i < 4 * 128; i++) c3[i] = 1.0f;
    std::vector<double> times;
    for (int r = 0; r < 7; r++) {
      auto e = q.submit([&](handler &h) {
        h.parallel_for(nd_range<1>(range<1>(n_wi), range<1>(8)),
                       [=](nd_item<1> it) SYCL_ESIMD_KERNEL {
          simd<uint16_t, 128> A; A.copy_from(a3);
          simd<uint16_t, 256> B; B.copy_from(b3);
          simd<float, 128> acc0, acc1, acc2, acc3;
          acc0.copy_from(c3); acc1.copy_from(c3 + 128);
          acc2.copy_from(c3 + 256); acc3.copy_from(c3 + 384);
          for (int i = 0; i < n_iter; i++) {
            acc0 = xmx::dpas<8, 8, float, float, uint16_t, uint16_t,
                             xmx::dpas_argument_type::bf16,
                             xmx::dpas_argument_type::bf16>(acc0, B, A);
            acc1 = xmx::dpas<8, 8, float, float, uint16_t, uint16_t,
                             xmx::dpas_argument_type::bf16,
                             xmx::dpas_argument_type::bf16>(acc1, B, A);
            acc2 = xmx::dpas<8, 8, float, float, uint16_t, uint16_t,
                             xmx::dpas_argument_type::bf16,
                             xmx::dpas_argument_type::bf16>(acc2, B, A);
            acc3 = xmx::dpas<8, 8, float, float, uint16_t, uint16_t,
                             xmx::dpas_argument_type::bf16,
                             xmx::dpas_argument_type::bf16>(acc3, B, A);
          }
          acc0.copy_to(c3); acc1.copy_to(c3 + 128);
          acc2.copy_to(c3 + 256); acc3.copy_to(c3 + 384);
        });
      });
      e.wait();
      uint64_t t0 = e.template get_profiling_info<sycl::info::event_profiling::command_start>();
      uint64_t t1 = e.template get_profiling_info<sycl::info::event_profiling::command_end>();
      times.push_back(double(t1 - t0));
    }
    std::sort(times.begin(), times.end());
    double med = times[times.size() / 2];
    double tf = 4.0 * n_iter * n_wi * FLOPS_PER_DPAS / (med * 1e-9) / 1e12;
    printf("bf16 ILP=4 R=8 n_wi=%4d: med=%12.0f ns  %7.3f ns/instr  -> %8.1f TF\n",
           n_wi, med, med / (4.0 * n_iter), tf);
    free(a3, q); free(b3, q); free(c3, q);
    return 0;
  }

  if (mode == 2) {
    // distinct B tile per chain (mimics coop-matrix lowering: IGC duplicates B)
    half *a2 = malloc_shared<half>(128, q);
    half *b2 = malloc_shared<half>(256 * 4, q);
    float *c2 = malloc_shared<float>(4 * 128, q);
    for (int i = 0; i < 128; i++) a2[i] = half(0.01f);
    for (int i = 0; i < 256 * 4; i++) b2[i] = half(0.01f);
    for (int i = 0; i < 4 * 128; i++) c2[i] = 1.0f;
    std::vector<double> times;
    for (int r = 0; r < 7; r++) {
      auto e = q.submit([&](handler &h) {
        h.parallel_for(nd_range<1>(range<1>(n_wi), range<1>(8)),
                       [=](nd_item<1> it) SYCL_ESIMD_KERNEL {
          simd<half, 128> A; A.copy_from(a2);
          simd<half, 256> B0, B1, B2, B3;
          B0.copy_from(b2 + 0); B1.copy_from(b2 + 256);
          B2.copy_from(b2 + 512); B3.copy_from(b2 + 768);
          simd<float, 128> acc0, acc1, acc2, acc3;
          acc0.copy_from(c2); acc1.copy_from(c2 + 128);
          acc2.copy_from(c2 + 256); acc3.copy_from(c2 + 384);
          for (int i = 0; i < n_iter; i++) {
            acc0 = xmx::dpas<8, 8, float>(acc0, B0, A);
            acc1 = xmx::dpas<8, 8, float>(acc1, B1, A);
            acc2 = xmx::dpas<8, 8, float>(acc2, B2, A);
            acc3 = xmx::dpas<8, 8, float>(acc3, B3, A);
          }
          acc0.copy_to(c2); acc1.copy_to(c2 + 128);
          acc2.copy_to(c2 + 256); acc3.copy_to(c2 + 384);
        });
      });
      e.wait();
      uint64_t t0 = e.template get_profiling_info<sycl::info::event_profiling::command_start>();
      uint64_t t1 = e.template get_profiling_info<sycl::info::event_profiling::command_end>();
      times.push_back(double(t1 - t0));
    }
    std::sort(times.begin(), times.end());
    double med = times[times.size() / 2];
    double tf = 4.0 * n_iter * n_wi * FLOPS_PER_DPAS / (med * 1e-9) / 1e12;
    printf("distinct-B ILP=4 R=8 n_wi=%4d: med=%12.0f ns  %7.3f ns/instr  -> %8.1f TF\n",
           n_wi, med, med / (4.0 * n_iter), tf);
    free(a2, q); free(b2, q); free(c2, q);
    return 0;
  }

  if (mode == 0) {
    run_config<1, 8>(q, n_wi, n_iter, 7, _);
    run_config<2, 8>(q, n_wi, n_iter, 7, _);
    run_config<3, 8>(q, n_wi, n_iter, 7, _);
    run_config<4, 8>(q, n_wi, n_iter, 7, _);
    run_config<6, 8>(q, n_wi, n_iter, 7, _);
    run_config<8, 8>(q, n_wi, n_iter, 7, _);
  } else {
    run_config<4, 1>(q, n_wi, n_iter, 7, _);
    run_config<4, 2>(q, n_wi, n_iter, 7, _);
    run_config<4, 4>(q, n_wi, n_iter, 7, _);
    run_config<4, 8>(q, n_wi, n_iter, 7, _);
  }
  return 0;
}
