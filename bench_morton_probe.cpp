// Morton-order ThreadGroup dispatch probe for BMG-G31.
// Each work-group (r,c) reads row slice A[r,:] and col slice B[c,:] (both contiguous,
// K floats each), writes one sum. Cross-WG data sharing: same-r WGs share A row,
// same-c WGs share B col. Concurrently-resident WGs thus have an L2 working set that
// depends on dispatch ORDER:
//   - linear order: resident WGs span a full tile-row  -> B window = all cols (huge)
//   - Morton order: resident WGs form a ~sqrt(N) x sqrt(N) block -> tiny window
// Three variants:
//   L: 1D grid, tile = (id/G, id%G)      -> row-major visitation
//   M: 1D grid, tile = morton_decode(id) -> software Z-order (benefit ceiling)
//   D: native 2D grid                    -> whatever the hardware dispatcher does
// If D ~= M  => hardware Morton dispatch present.  If D ~= L => absent.
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdint>
#include <algorithm>
#include <vector>

using namespace sycl;

static inline uint32_t part1by1(uint32_t x) {
  x &= 0x0000ffff;
  x = (x | (x << 8)) & 0x00FF00FF;
  x = (x | (x << 4)) & 0x0F0F0F0F;
  x = (x | (x << 2)) & 0x33333333;
  x = (x | (x << 1)) & 0x55555555;
  return x;
}

double run_variant(queue &q, const float *A, const float *B, float *out,
                   int G, int K, int variant, int reps) {
  // variant: 0=linear 1D, 1=morton 1D, 2=native 2D
  std::vector<double> times;
  for (int r = 0; r < reps; r++) {
    event e;
    if (variant < 2) {
      e = q.submit([&](handler &h) {
        h.parallel_for(nd_range<1>(range<1>(size_t(G) * G * 16), range<1>(16)),
                       [=](nd_item<1> it) {
          uint32_t id = it.get_group(0);
          uint32_t tr, tc;
          if (variant == 0) { tr = id / G; tc = id % G; }
          else {
            // software Morton (works for pow2 G; clamp for safety)
            tr = part1by1(id); tc = part1by1(id >> 1);
            if (tr >= (uint32_t)G || tc >= (uint32_t)G) { tr = id / G; tc = id % G; }
          }
          const float *a = A + size_t(tr) * K;
          const float *b = B + size_t(tc) * K;
          float s = 0.f;
          int lid = it.get_local_id(0);
          const float4 *a4 = reinterpret_cast<const float4 *>(a);
          const float4 *b4 = reinterpret_cast<const float4 *>(b);
          float4 s4(0.f);
#pragma unroll 4
          for (int k = lid; k < K / 4; k += 16) s4 += a4[k] + b4[k];
          s = s4.x() + s4.y() + s4.z() + s4.w();
          // reduce across 16 lanes
          auto sg = it.get_sub_group();
          s = reduce_over_group(sg, s, plus<>());
          if (lid == 0) out[size_t(tr) * G + tc] = s;
        });
      });
    } else {
      e = q.submit([&](handler &h) {
        h.parallel_for(nd_range<2>(range<2>(G, G * 16), range<2>(1, 16)),
                       [=](nd_item<2> it) {
          uint32_t tr = it.get_group(0), tc = it.get_group(1);
          const float *a = A + size_t(tr) * K;
          const float *b = B + size_t(tc) * K;
          float s = 0.f;
          int lid = it.get_local_id(1);
          const float4 *a4 = reinterpret_cast<const float4 *>(a);
          const float4 *b4 = reinterpret_cast<const float4 *>(b);
          float4 s4(0.f);
#pragma unroll 4
          for (int k = lid; k < K / 4; k += 16) s4 += a4[k] + b4[k];
          s = s4.x() + s4.y() + s4.z() + s4.w();
          auto sg = it.get_sub_group();
          s = reduce_over_group(sg, s, plus<>());
          if (lid == 0) out[size_t(tr) * G + tc] = s;
        });
      });
    }
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
  printf("Device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

  int K = argc > 1 ? atoi(argv[1]) : 4096;   // slice length (floats)
  int g0 = argc > 2 ? atoi(argv[2]) : 256;   // G range: [g0, g1]
  int g1 = argc > 3 ? atoi(argv[3]) : g0;
  int reps = argc > 4 ? atoi(argv[4]) : 3;
  int only = argc > 5 ? atoi(argv[5]) : -1;   // -1=all, else single variant 0/1/2
  printf("slice = %d floats (%.0f KB), traffic/WG = %.0f KB\n",
         K, K * 4.0 / 1024, K * 8.0 / 1024);
  printf("G      WGs      linear_ms   morton_ms   native2D_ms   D vs L/M\n");

  for (int G = g0; G <= g1; G *= 2) {
    size_t nA = size_t(G) * K;
    float *A = malloc_shared<float>(nA, q);
    float *B = malloc_shared<float>(nA, q);
    float *out = malloc_shared<float>(size_t(G) * G, q);
    for (size_t i = 0; i < nA; i++) { A[i] = 1e-4f; B[i] = 1e-4f; }
    q.memset(out, 0, sizeof(float) * G * G).wait();

    if (only >= 0) {
      double t = run_variant(q, A, B, out, G, K, only, reps);
      printf("%4d %8d  variant=%d  %10.3f ms\n", G, G * G, only, t / 1e6);
      free(A, q); free(B, q); free(out, q);
      continue;
    }
    double tl = run_variant(q, A, B, out, G, K, 0, reps);
    double tm = run_variant(q, A, B, out, G, K, 1, reps);
    double td = run_variant(q, A, B, out, G, K, 2, reps);
    printf("%4d %8d  %10.3f  %10.3f  %10.3f     D/L=%.2f D/M=%.2f\n",
           G, G * G, tl / 1e6, tm / 1e6, td / 1e6, td / tl, td / tm);

    free(A, q); free(B, q); free(out, q);
  }
  return 0;
}
