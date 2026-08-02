// LLM decode-representative vector kernels on B70 (SYCL): measure achieved
// bandwidth and GOP/s, classify memory-bound vs ALU-bound.
// Kernels: triad (BW baseline), rmsnorm, swiglu(silu*mul), softmax,
//          fp8(e4m3)->bf16 dequant (with and without scale).
// Sizes follow Qwen3-MoE: hidden 2048, vocab 151936.
// Run: icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31 -O3 bench_llm_vector.cpp
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <chrono>

using namespace sycl;
using bf16 = sycl::ext::oneapi::bfloat16;

static double time_min(queue &q, std::function<void()> fn, int reps = 10) {
  fn(); q.wait();  // warmup
  double best = 1e30;
  for (int r = 0; r < reps; r++) {
    auto t0 = std::chrono::steady_clock::now();
    fn(); q.wait();
    auto t1 = std::chrono::steady_clock::now();
    best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
  }
  return best;
}

static void report(const char *name, double sec, double bytes, double flops) {
  printf("  %-28s %8.3f ms  %8.1f GB/s  %8.2f GOP/s\n",
         name, sec * 1e3, bytes / sec / 1e9, flops / sec / 1e9);
}

int main() {
  queue q(gpu_selector_v);
  printf("device: %s\n", q.get_device().get_info<info::device::name>().c_str());

  // ---- 0. triad bf16: achievable BW baseline ------------------------------
  {
    size_t E = 64ull << 20;
    bf16 *a = malloc_device<bf16>(E, q), *b = malloc_device<bf16>(E, q),
         *o = malloc_device<bf16>(E, q);
    q.fill(a, bf16(1.0f), E); q.fill(b, bf16(2.0f), E); q.wait();
    double t = time_min(q, [&] {
      q.parallel_for(range<1>(E), [=](id<1> i) {
        o[i] = bf16(float(a[i]) + 1.5f * float(b[i]));
      });
    });
    report("triad bf16 (BW baseline)", t, 3.0 * E * 2, 2.0 * E);
    free(a, q); free(b, q); free(o, q);
  }

  // ---- 1. RMSNorm: R rows x N=2048, bf16 io, fp32 accum -------------------
  {
    size_t R = 16384, N = 2048, LS = 256;
    bf16 *x = malloc_device<bf16>(R * N, q), *w = malloc_device<bf16>(N, q),
         *y = malloc_device<bf16>(R * N, q);
    q.fill(x, bf16(0.01f), R * N); q.fill(w, bf16(1.0f), N); q.wait();
    double t = time_min(q, [&] {
      q.parallel_for(nd_range<1>(range<1>(R * LS), range<1>(LS)),
                     [=](nd_item<1> it) {
        size_t row = it.get_group(0), base = row * N;
        float ss = 0;
        for (size_t j = it.get_local_id(0); j < N; j += LS) {
          float v = float(x[base + j]); ss += v * v;
        }
        ss = reduce_over_group(it.get_group(), ss, plus<float>());
        float inv = 1.0f / sycl::sqrt(ss / N + 1e-6f);
        for (size_t j = it.get_local_id(0); j < N; j += LS)
          y[base + j] = bf16(float(x[base + j]) * inv * float(w[j]));
      });
    });
    double bytes = double(R) * N * 2 * 2 + double(R) * N / 8;  // rd x + wr y (+w)
    report("rmsnorm bf16 N=2048", t, bytes, 5.0 * R * N);
    free(x, q); free(w, q); free(y, q);
  }

  // ---- 2. SwiGLU: out = silu(g) * u, bf16 ---------------------------------
  {
    size_t E = 64ull << 20;
    bf16 *g = malloc_device<bf16>(E, q), *u = malloc_device<bf16>(E, q),
         *o = malloc_device<bf16>(E, q);
    q.fill(g, bf16(0.5f), E); q.fill(u, bf16(0.25f), E); q.wait();
    double t = time_min(q, [&] {
      q.parallel_for(range<1>(E), [=](id<1> i) {
        float v = float(g[i]);
        float silu = v / (1.0f + sycl::exp(-v));
        o[i] = bf16(silu * float(u[i]));
      });
    });
    report("swiglu silu*mul bf16", t, 3.0 * E * 2, 6.0 * E);
    free(g, q); free(u, q); free(o, q);
  }

  // ---- 3. softmax: 256 rows x 151936 (vocab), fp32 -------------------------
  {
    size_t R = 256, N = 151936, LS = 1024;
    float *x = malloc_device<float>(R * N, q), *y = malloc_device<float>(R * N, q);
    q.fill(x, 0.001f, R * N); q.wait();
    double t = time_min(q, [&] {
      q.parallel_for(nd_range<1>(range<1>(R * LS), range<1>(LS)),
                     [=](nd_item<1> it) {
        size_t row = it.get_group(0), base = row * N;
        float m = -1e30f;
        for (size_t j = it.get_local_id(0); j < N; j += LS)
          m = sycl::fmax(m, x[base + j]);
        m = reduce_over_group(it.get_group(), m, maximum<float>());
        float s = 0;
        for (size_t j = it.get_local_id(0); j < N; j += LS)
          s += sycl::exp(x[base + j] - m);
        s = reduce_over_group(it.get_group(), s, plus<float>());
        float inv = 1.0f / s;
        for (size_t j = it.get_local_id(0); j < N; j += LS)
          y[base + j] = sycl::exp(x[base + j] - m) * inv;
      });
    });
    double bytes = double(R) * N * 4 * 4;  // ~3 reads + 1 write (exp recomputed)
    report("softmax fp32 N=151936", t, bytes, 4.0 * R * N);
    free(x, q); free(y, q);
  }

  // ---- 4. fp8 e4m3 -> bf16 dequant (per-128 scale), packed u32 loads ------
  {
    size_t E = 128ull << 20;          // elements
    size_t W = E / 4;                 // u32 words (4 fp8 each)
    uint32_t *v = malloc_device<uint32_t>(W, q);
    float *sc = malloc_device<float>(E / 128, q);
    bf16 *o = malloc_device<bf16>(E, q);
    q.fill(v, uint32_t(0x38383838), W); q.fill(sc, 0.5f, E / 128); q.wait();
    auto cvt = [](uint8_t b) {  // e4m3 -> fp16 (shift + rebias via *2^8)
      ushort bits = ushort(b) << 8;
      sycl::half h = *reinterpret_cast<sycl::half *>(&bits);
      return float(h) * 256.0f;
    };
    double t1 = time_min(q, [&] {
      q.parallel_for(range<1>(W), [=](id<1> i) {
        uint32_t w = v[i];
        float s = sc[i / 32];        // 128 elems = 32 words per scale group
#pragma unroll
        for (int k = 0; k < 4; k++)
          o[i * 4 + k] = bf16(cvt(uint8_t(w >> (8 * k))) * s);
      });
    });
    report("dequant fp8->bf16 +scale", t1, double(E) * 3 + E / 128 * 4, 3.0 * E);
    double t2 = time_min(q, [&] {
      q.parallel_for(range<1>(W), [=](id<1> i) {
        uint32_t w = v[i];
#pragma unroll
        for (int k = 0; k < 4; k++)
          o[i * 4 + k] = bf16(cvt(uint8_t(w >> (8 * k))));
      });
    });
    report("dequant fp8->bf16 pure cvt", t2, double(E) * 3, 2.0 * E);
    free(v, q); free(sc, q); free(o, q);
  }
  return 0;
}
