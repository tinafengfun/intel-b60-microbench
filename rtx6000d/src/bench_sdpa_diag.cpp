// Bisect the oneMKL per-call slowdown seen in bench_sdpa (H>=2 slow, H=1 fast).
// mode A: distinct buffers per call | B: same pointers every call | C: valid bf16 P
#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <vector>
using namespace sycl;
using bf16 = sycl::ext::oneapi::bfloat16;
namespace mkl = oneapi::mkl::blas::row_major;
namespace mklc = oneapi::mkl::blas::column_major;
static double now_s() {
  return std::chrono::duration<double>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}
int main(int argc, char **argv) {
  int S = argc > 1 ? atoi(argv[1]) : 4096;
  int H = argc > 2 ? atoi(argv[2]) : 8;
  char mode = argc > 3 ? argv[3][0] : 'A';
  int d = 128;
  queue q(gpu_selector_v);
  size_t qkv = (size_t)H * S * d, sc = (size_t)H * S * S;
  bf16 *Q = malloc_device<bf16>(qkv, q);
  bf16 *K = malloc_device<bf16>(qkv, q);
  float *Sc = malloc_device<float>(sc, q);
  q.fill(Q, bf16(0.01f), qkv); q.fill(K, bf16(0.01f), qkv); q.wait();

  // mode 'E': profiling queue, read GPU-side timestamps (bypass host wait latency)
  if (mode == 'E') {
    queue pq(gpu_selector_v, property::queue::enable_profiling{});
    bf16 *Q2 = malloc_device<bf16>(qkv, pq);
    bf16 *K2 = malloc_device<bf16>(qkv, pq);
    float *S2 = malloc_device<float>(sc, pq);
    pq.fill(Q2, bf16(0.01f), qkv); pq.fill(K2, bf16(0.01f), qkv); pq.wait();
    for (int r = 0; r < 5; r++) {
      std::vector<event> evs;
      double t0 = now_s();
      for (int h = 0; h < H; h++)
        evs.push_back(mkl::gemm(pq, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::trans,
                                S, S, d, 1.0f, Q2 + (size_t)h * S * d, d,
                                K2 + (size_t)h * S * d, d, 0.0f, S2 + (size_t)h * S * S, S));
      evs.back().wait();
      double wall = now_s() - t0;
      uint64_t g0 = evs.front().get_profiling_info<info::event_profiling::command_start>();
      uint64_t g1 = evs.back().get_profiling_info<info::event_profiling::command_end>();
      printf("mode E rep %d: wall %.3f ms, gpu-span %.3f ms (%.1f TF gpu) | per-call gpu:",
             r, wall * 1e3, (g1 - g0) / 1e6,
             2.0 * H * S * S * d / ((g1 - g0) / 1e9) / 1e12);
      for (int h = 0; h < H; h++) {
        uint64_t s = evs[h].get_profiling_info<info::event_profiling::command_start>();
        uint64_t e2 = evs[h].get_profiling_info<info::event_profiling::command_end>();
        printf(" %.2f", (e2 - s) / 1e6);
      }
      printf(" ms\n");
    }
    free(Q2, pq); free(K2, pq); free(S2, pq);
    return 0;
  }
  for (int r = 0; r < 5; r++) {
    double t0 = now_s();
    std::vector<event> evs;
    for (int h = 0; h < H; h++) {
      int hh = (mode == 'B') ? 0 : h;
      double tc = now_s();
      event e;
      if (mode == 'C') {
        // column_major restatement: Q/K stored [S,d] row-major == [d,S] col-major.
        // C_cm = C_row^T = K @ Q^T = K_cm^T @ Q_cm
        e = mklc::gemm(q, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans,
                            S, S, d, 1.0f, K + (size_t)hh * S * d, d,
                            Q + (size_t)hh * S * d, d, 0.0f, Sc + (size_t)hh * S * S, S);
      } else {
        e = mkl::gemm(q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::trans,
                           S, S, d, 1.0f, Q + (size_t)hh * S * d, d,
                           K + (size_t)hh * S * d, d, 0.0f, Sc + (size_t)hh * S * S, S);
      }
      if (mode == 'D') {
        evs.push_back(e);  // defer all waits: submit everything first
      } else {
        double t_submit = now_s();  // host time spent inside mkl::gemm call itself
        e.wait();
        double t_end = now_s();
        if (r >= 3)
          printf("  call h=%d: submit %.3f ms, wait %.3f ms (%.1f TF)\n", h,
                 (t_submit - tc) * 1e3, (t_end - t_submit) * 1e3,
                 2.0 * S * S * d / (t_end - tc) / 1e12);
      }
    }
    if (mode == 'D') evs.back().wait();
    printf("mode %c gemm1 rep %d: %.3f ms (%.1f TF)\n", mode, r, (now_s() - t0) * 1e3,
           2.0 * H * S * S * d / (now_s() - t0) / 1e12);
  }
  return 0;
}
