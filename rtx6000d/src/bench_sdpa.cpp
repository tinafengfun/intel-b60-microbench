// SDPA (attention core) on B70: naive serial vs pipelined multi-queue.
// Qwen3-30B-A3B attention shapes: H=32 heads, d=128, S = seq len (sweep).
//   gemm1: scores[S,S] = Q[S,d] @ K[S,d]^T   (oneMKL bf16->fp32, XMX)
//   softmax: fp32 rows -> bf16 P (aliased into scores buffer)
//   gemm2: out[S,d] = P[S,S] @ V[S,d]        (oneMKL bf16->fp32, XMX)
// Modes: serial (1 queue) | pipe (3 queues: gemm1/softmax/gemm2 streams) | gemmonly
// Run: ./bench_sdpa [S=4096] [H=32] [mode=serial|pipe|gemmonly]
#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include <chrono>

using namespace sycl;
using bf16 = sycl::ext::oneapi::bfloat16;
namespace mkl = oneapi::mkl::blas::row_major;

static double now_s() {
  return std::chrono::duration<double>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char **argv) {
  int S = argc > 1 ? atoi(argv[1]) : 4096;
  int H = argc > 2 ? atoi(argv[2]) : 32;
  int d = 128;
  const char *mode = argc > 3 ? argv[3] : "serial";

  queue q0(gpu_selector_v, property::queue::enable_profiling{});
  printf("device: %s  S=%d H=%d d=%d mode=%s\n",
         q0.get_device().get_info<info::device::name>().c_str(), S, H, d, mode);

  size_t qkv_sz = (size_t)H * S * d;
  size_t sc_sz = (size_t)H * S * S;
  bf16 *Q = malloc_device<bf16>(qkv_sz, q0);
  bf16 *K = malloc_device<bf16>(qkv_sz, q0);
  bf16 *V = malloc_device<bf16>(qkv_sz, q0);
  float *O = malloc_device<float>(qkv_sz, q0);
  float *Sc = malloc_device<float>(sc_sz, q0);      // fp32 scores
  bf16 *Pb = malloc_device<bf16>(sc_sz, q0);        // softmax output P (separate: aliasing races)
  if (!Q || !K || !V || !O || !Sc || !Pb) { printf("alloc failed\n"); return 1; }
  q0.fill(Q, bf16(0.01f), qkv_sz); q0.fill(K, bf16(0.01f), qkv_sz);
  q0.fill(V, bf16(0.01f), qkv_sz); q0.wait();
  printf("scores mem: %.1f GB\n", sc_sz * 4.0 / 1e9);

  auto gemm1 = [&](queue &q, int h) {  // Sc_h[S,S] = Q_h @ K_h^T
    return mkl::gemm(q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::trans,
                     S, S, d, 1.0f, Q + (size_t)h * S * d, d,
                     K + (size_t)h * S * d, d, 0.0f, Sc + (size_t)h * S * S, S);
  };
  auto gemm2 = [&](queue &q, int h) {  // O_h[S,d] = P_h[S,S] @ V_h[S,d]
    bf16 *P = Pb + (size_t)h * S * S;
    return mkl::gemm(q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
                     S, d, S, 1.0f, P, S, V + (size_t)h * S * d, d,
                     0.0f, O + (size_t)h * S * d, d);
  };
  auto softmax = [&](queue &q, int h, event dep) {
    float *s = Sc + (size_t)h * S * S;
    bf16 *p = Pb + (size_t)h * S * S;
    int LS = S >= 1024 ? 1024 : S;
    return q.submit([&](handler &c) {
      c.depends_on(dep);
      c.parallel_for(nd_range<1>(range<1>((size_t)S * LS), range<1>(LS)),
                     [=](nd_item<1> it) {
        size_t row = it.get_group(0);
        float m = -1e30f;
        for (int j = it.get_local_id(0); j < S; j += LS)
          m = sycl::fmax(m, s[row * S + j]);
        m = reduce_over_group(it.get_group(), m, maximum<float>());
        float sm = 0;
        for (int j = it.get_local_id(0); j < S; j += LS)
          sm += sycl::exp(s[row * S + j] - m);
        sm = reduce_over_group(it.get_group(), sm, plus<float>());
        float inv = 1.0f / sm;
        for (int j = it.get_local_id(0); j < S; j += LS)
          p[row * S + j] = bf16(sycl::exp(s[row * S + j] - m) * inv);
      });
    });
  };

  // Timing note: host-side e.wait() per call on this stack has 10-100ms event-
  // signaling latency (see bench_sdpa_diag). All timing below uses GPU-side
  // profiling timestamps; host wall is reported for reference only.
  queue qv(gpu_selector_v, property::queue::enable_profiling{});
  queue qg2(gpu_selector_v, property::queue::enable_profiling{});

  double best_gpu = 1e30, best_wall = 1e30;
  double best_g1 = 0, best_sm = 0, best_g2 = 0;
  for (int r = 0; r < 5; r++) {
    std::vector<event> e1(H), e2(H), e3(H);
    double t0 = now_s();
    if (!strcmp(mode, "serial")) {
      for (int h = 0; h < H; h++) {   // in-order queue: implicit serialization
        e1[h] = gemm1(q0, h);
        e2[h] = softmax(q0, h, e1[h]);
        bf16 *P = Pb + (size_t)h * S * S;
        e3[h] = mkl::gemm(q0, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
                          S, d, S, 1.0f, P, S, V + (size_t)h * S * d, d,
                          0.0f, O + (size_t)h * S * d, d, {e2[h]});
      }
    } else if (!strcmp(mode, "pipe")) {
      for (int h = 0; h < H; h++) e1[h] = gemm1(q0, h);
      for (int h = 0; h < H; h++) e2[h] = softmax(qv, h, e1[h]);
      for (int h = 0; h < H; h++) {
        bf16 *P = Pb + (size_t)h * S * S;
        e3[h] = mkl::gemm(qg2, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
                          S, d, S, 1.0f, P, S, V + (size_t)h * S * d, d,
                          0.0f, O + (size_t)h * S * d, d, {e2[h]});
      }
    } else {  // gemmonly: all gemm1 then all gemm2, no softmax
      for (int h = 0; h < H; h++) e1[h] = gemm1(q0, h);
      for (int h = 0; h < H; h++) {
        bf16 *P = Pb + (size_t)h * S * S;
        e3[h] = mkl::gemm(q0, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
                          S, d, S, 1.0f, P, S, V + (size_t)h * S * d, d,
                          0.0f, O + (size_t)h * S * d, d);
      }
    }
    e3[H - 1].wait();
    q0.wait(); qv.wait(); qg2.wait();
    double wall = now_s() - t0;

    auto dur = [](event &e) {
      return (double)(e.get_profiling_info<info::event_profiling::command_end>() -
                      e.get_profiling_info<info::event_profiling::command_start>()) / 1e9;
    };
    auto tstart = [](event &e) {
      return e.get_profiling_info<info::event_profiling::command_start>();
    };
    auto tend = [](event &e) {
      return e.get_profiling_info<info::event_profiling::command_end>();
    };
    double g1 = 0, sm = 0, g2 = 0;
    uint64_t tmin = ~0ull, tmax = 0;
    for (int h = 0; h < H; h++) {
      g1 += dur(e1[h]);
      if (strcmp(mode, "gemmonly")) sm += dur(e2[h]);
      g2 += dur(e3[h]);
      for (event *ep : {&e1[h], &e2[h], &e3[h]}) {
        if (!strcmp(mode, "gemmonly") && ep == &e2[h]) continue;
        tmin = std::min(tmin, tstart(*ep));
        tmax = std::max(tmax, tend(*ep));
      }
    }
    double gpu_span = (tmax - tmin) / 1e9;
    if (gpu_span < best_gpu) {
      best_gpu = gpu_span; best_wall = wall;
      best_g1 = g1; best_sm = sm; best_g2 = g2;
    }
  }

  double gemm_flops = 2.0 * H * S * S * d * 2;       // gemm1 + gemm2
  double sm_ops = 5.0 * H * S * S;                   // softmax vector ops
  printf("TOTAL %s: gpu-span %.2f ms (host wall %.2f ms)\n", mode, best_gpu * 1e3, best_wall * 1e3);
  printf("  stages: gemm1 %.2f ms, softmax %.2f ms, gemm2 %.2f ms (sums over %d heads)\n",
         best_g1 * 1e3, best_sm * 1e3, best_g2 * 1e3, H);
  printf("  gemm throughput %.1f TF (gpu busy), softmax %.1f Gops/s\n",
         gemm_flops / (best_g1 + best_g2) / 1e12, sm_ops / (best_sm > 0 ? best_sm : 1e-9) / 1e9);
  if (strcmp(mode, "gemmonly"))
    printf("  vector(softmax) share of gpu-span: %.1f%%\n", 100.0 * best_sm / best_gpu);
  return 0;
}
