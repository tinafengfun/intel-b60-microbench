// Xe3 feature probes on B70 (Xe2 / BMG-G31):
//   mode 0: device queries (fp64 aspect, SLM, cache, subgroups)
//   mode 1: FP64 vs FP32 vector FMA rate
//   mode 2: transcendental/EM throughput (fma, exp, log, tanh, sin, rsqrt, sqrt)
//   mode 3: MultiQ concurrency (N SYCL queues overlap)
// All rate tests: scalar ILP chains, n_wi work-items, 2.4 GHz lock expected.
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <vector>
#include <chrono>

using namespace sycl;
namespace esimd_ns = sycl::ext::intel::esimd;

// ---- ESIMD vector rate kernel: true vector-engine throughput --------------
template <typename T, int OP, int ILP>
void run_esimd_rate(queue &q, int n_wi, int n_iter, const char *name) {
  using namespace sycl::ext::intel::esimd;
  T *buf = malloc_shared<T>(256, q);
  for (int i = 0; i < 256; i++) buf[i] = T(0.5 + 0.001 * (i % 50));
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
          for (int j = 0; j < ILP; j++) {
            if constexpr (OP == 0) acc[j] = a * acc[j] + b;
            else if constexpr (OP == 1) acc[j] = esimd_ns::exp(acc[j] * T(1e-9));
            else if constexpr (OP == 2) acc[j] = esimd_ns::rsqrt(esimd_ns::abs(acc[j]) + T(1));
            else acc[j] = esimd_ns::sqrt(esimd_ns::abs(acc[j]) + T(1));
          }
        }
        simd<T, 32> s = 0;
        for (int j = 0; j < ILP; j++) s += acc[j];
        if (s[0] == T(12345.678)) s.copy_to(buf);   // DCE guard
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
  printf("  %-10s T=%s n_wi=%5d: %9.2f GOP/s  (%6.2f lane-op/cyc/EU @2.4GHz)\n",
         name, sizeof(T) == 8 ? "f64" : "f32", n_wi, gops, gops * 1e9 / (256 * 2.4e9));
  free(buf, q);
}

// ---- generic ILP-chain rate kernel ----------------------------------------
template <typename T, int OP>
double run_rate(queue &q, int n_wi, int n_iter, const char *name, double *ops_out) {
  // OP: 0=fma 1=exp 2=log 3=tanh 4=sin 5=rsqrt 6=sqrt
  constexpr int ILP = 8;
  T *buf = malloc_shared<T>(ILP * 16, q);
  for (int i = 0; i < ILP * 16; i++) buf[i] = T(0.5 + 0.001 * i);
  std::vector<double> times;
  for (int r = 0; r < 7; r++) {
    auto t0 = std::chrono::steady_clock::now();
    auto e = q.submit([&](handler &h) {
      h.parallel_for(range<1>(n_wi), [=](id<1> wid) {
        T acc[ILP];
        T a = buf[wid % 16] + T(wid % 7), b = buf[(wid + 3) % 16];
        for (int j = 0; j < ILP; j++) acc[j] = buf[(wid + j) % 16];
        for (int i = 0; i < n_iter; i++) {
#pragma unroll
          for (int j = 0; j < ILP; j++) {
            if constexpr (OP == 0) acc[j] = sycl::fma(a, acc[j], b);
            else if constexpr (OP == 1) acc[j] = sycl::exp(acc[j] * T(1e-9));
            else if constexpr (OP == 2) acc[j] = sycl::log(sycl::fabs(acc[j]) + T(1));
            else if constexpr (OP == 3) acc[j] = sycl::tanh(acc[j]);
            else if constexpr (OP == 4) acc[j] = sycl::sin(acc[j]);
            else if constexpr (OP == 5) acc[j] = sycl::rsqrt(sycl::fabs(acc[j]) + T(1));
            else acc[j] = sycl::sqrt(sycl::fabs(acc[j]) + T(1));
          }
        }
        T s = 0;
        for (int j = 0; j < ILP; j++) s += acc[j];
        if (s == T(12345.678)) buf[wid % 16] = s;   // DCE guard
      });
    });
    e.wait();
    auto t1 = std::chrono::steady_clock::now();
    times.push_back(std::chrono::duration<double>(t1 - t0).count());
  }
  std::sort(times.begin(), times.end());
  double med = times[times.size() / 2];
  double ops = double(ILP) * n_iter * n_wi;
  *ops_out = ops;
  double gops = ops / med / 1e9;
  printf("  %-10s T=%s n_wi=%5d: %9.2f GOP/s  (%6.2f op/cyc/EU @2.4GHz,256EU)\n",
         name, sizeof(T) == 8 ? "f64" : "f32", n_wi, gops,
         gops * 1e9 / (256 * 2.4e9));
  free(buf, q);
  return gops;
}

int main(int argc, char **argv) {
  int mode = argc > 1 ? atoi(argv[1]) : 0;
  queue q(property::queue::enable_profiling{});
  auto dev = q.get_device();
  int n_iter = argc > 2 ? atoi(argv[2]) : 16384;
  int n_wi = argc > 3 ? atoi(argv[3]) : 2048;

  if (mode == 0) {
    printf("Device: %s  EUs: %d  max_clk: %d MHz\n",
           dev.get_info<info::device::name>().c_str(),
           dev.get_info<info::device::max_compute_units>(),
           dev.get_info<info::device::max_clock_frequency>());
    printf("aspect fp64: %d  fp16: %d  bf16: %d\n",
           dev.has(aspect::fp64), dev.has(aspect::fp16), dev.has(aspect::accelerator));
    printf("local_mem_size (SLM/WG cap): %zu KB\n",
           dev.get_info<info::device::local_mem_size>() / 1024);
    printf("local_mem_type: %s\n",
           dev.get_info<info::device::local_mem_type>() == info::local_mem_type::local ? "dedicated" : "none/global");
    printf("global_mem_cache_size: %zu KB\n",
           dev.get_info<info::device::global_mem_cache_size>() / 1024);
    printf("global_mem_size: %.1f GB\n",
           dev.get_info<info::device::global_mem_size>() / 1e9);
    printf("max_work_group_size: %zu\n", dev.get_info<info::device::max_work_group_size>());
    auto sg = dev.get_info<info::device::sub_group_sizes>();
    printf("sub_group_sizes:"); for (auto s : sg) printf(" %zu", s); printf("\n");
    printf("max_num_sub_groups: %u\n", dev.get_info<info::device::max_num_sub_groups>());
    printf("usm_shared: %d  usm_atomic_shared: %d\n",
           dev.has(aspect::usm_shared_allocations), dev.has(aspect::usm_atomic_shared_allocations));
    return 0;
  }

  if (mode == 1) {
    double _;
    printf("-- FP32 FMA rate --\n");
    run_rate<float, 0>(q, n_wi, n_iter, "fma", &_);
    printf("-- FP64 FMA rate --\n");
    run_rate<double, 0>(q, n_wi, n_iter, "fma", &_);
    return 0;
  }

  if (mode == 2) {
    double _;
    printf("-- EM/transcendental rate (fp32) --\n");
    run_rate<float, 0>(q, n_wi, n_iter, "fma(base)", &_);
    run_rate<float, 1>(q, n_wi, n_iter, "exp", &_);
    run_rate<float, 2>(q, n_wi, n_iter, "log", &_);
    run_rate<float, 3>(q, n_wi, n_iter, "tanh", &_);
    run_rate<float, 4>(q, n_wi, n_iter, "sin", &_);
    run_rate<float, 5>(q, n_wi, n_iter, "rsqrt", &_);
    run_rate<float, 6>(q, n_wi, n_iter, "sqrt", &_);
    return 0;
  }

  if (mode == 5) {
    printf("-- ESIMD vector rates --\n");
    run_esimd_rate<float, 0, 8>(q, n_wi, n_iter, "fma");
    run_esimd_rate<double, 0, 8>(q, n_wi, n_iter, "fma");
    run_esimd_rate<float, 1, 8>(q, n_wi, n_iter, "exp");
    run_esimd_rate<float, 2, 8>(q, n_wi, n_iter, "rsqrt");
    run_esimd_rate<float, 3, 8>(q, n_wi, n_iter, "sqrt");
    return 0;
  }

  if (mode == 3) {
    // MultiQ: K concurrent queues, long kernel each; overlap ratio
    int K = argc > 4 ? atoi(argv[4]) : 4;
    int heavy_iter = 200000;
    std::vector<queue> qs;
    for (int i = 0; i < K; i++) qs.emplace_back();
    float *sink = malloc_shared<float>(K, q);
    auto job = [&](queue &qq, int idx) {
      return qq.submit([&](handler &h) {
        h.parallel_for(range<1>(8192), [=](id<1> wid) {
          float a = wid % 1000 + idx;
          for (int i = 0; i < heavy_iter; i++) a = sycl::fma(a, 0.9999f, 0.0001f);
          if (a == 12345.f) sink[idx] = a;
        });
      });
    };
    job(qs[0], 99).wait();   // JIT warmup
    // serial baseline
    auto t0 = std::chrono::steady_clock::now();
    job(qs[0], 0).wait();
    auto t1 = std::chrono::steady_clock::now();
    double t_serial = std::chrono::duration<double>(t1 - t0).count();
    // concurrent
    t0 = std::chrono::steady_clock::now();
    std::vector<event> evs;
    for (int i = 0; i < K; i++) evs.push_back(job(qs[i], i));
    for (auto &e : evs) e.wait();
    t1 = std::chrono::steady_clock::now();
    double t_conc = std::chrono::duration<double>(t1 - t0).count();
    printf("MultiQ: K=%d queues  serial1=%.1f ms  concurrent%d=%.1f ms  speedup=%.2fx\n",
           K, t_serial * 1e3, K, t_conc * 1e3, K * t_serial / t_conc);
    return 0;
  }
  return 0;
}
