#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#include <cstdio>
#include <cstdint>
int main(int argc, char** argv) {
  int n = argc > 1 ? atoi(argv[1]) : 8192;
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::enable_profiling{});
  printf("Device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
  auto* A = sycl::malloc_shared<int8_t>((size_t)n*n, q);
  auto* B = sycl::malloc_shared<int8_t>((size_t)n*n, q);
  auto* C = sycl::malloc_shared<int32_t>((size_t)n*n, q);
  for (size_t i = 0; i < (size_t)n*n; i++) { A[i] = 1; B[i] = 2; C[i] = 0; }
  auto gemm = [&]() {
    return oneapi::mkl::blas::column_major::gemm(
        q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
        n, n, n, 1.0f, A, n, B, n, 0.0f, C, n);
  };
  gemm().wait();
  double best = 1e30;
  for (int r = 0; r < 10; r++) {
    auto e = gemm(); e.wait();
    double t = (e.template get_profiling_info<sycl::info::event_profiling::command_end>() -
                e.template get_profiling_info<sycl::info::event_profiling::command_start>()) * 1e-9;
    if (t < best) best = t;
  }
  double tops = 2.0*n*n*(double)n / best / 1e12;
  printf("s8 GEMM %d^3: %.3f ms -> %.1f TOPS  [C0=%d expect %d]\n", n, best*1e3, tops, C[0], 2*n);
  return 0;
}
