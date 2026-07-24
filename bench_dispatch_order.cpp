// Direct observation of ThreadGroup dispatch order on BMG-G31.
// Each WG atomically claims a slot and records its (group_x, group_y).
// The slot sequence IS the dispatch order. Analysis: mean row-span of
// consecutive WGs — row-major dispatch: span ~1; Morton/block: span ~sqrt(window).
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdint>

using namespace sycl;

int main(int argc, char **argv) {
  int GX = argc > 1 ? atoi(argv[1]) : 64;
  int GY = argc > 2 ? atoi(argv[2]) : 64;
  queue q;
  printf("Device: %s, grid %d x %d WGs\n",
         q.get_device().get_info<sycl::info::device::name>().c_str(), GX, GY);

  int N = GX * GY;
  uint32_t *counter = malloc_shared<uint32_t>(1, q);
  uint32_t *log = malloc_shared<uint32_t>(2 * N, q);
  *counter = 0;

  q.submit([&](handler &h) {
    h.parallel_for(nd_range<2>(range<2>(GX, GY * 16), range<2>(1, 16)),
                   [=](nd_item<2> it) {
      if (it.get_local_id(1) == 0) {
        atomic_ref<uint32_t, memory_order::relaxed, memory_scope::device,
                   access::address_space::global_space> cnt(*counter);
        uint32_t slot = cnt.fetch_add(1);
        log[slot * 2 + 0] = it.get_group(0);
        log[slot * 2 + 1] = it.get_group(1);
      }
    });
  });
  q.wait();

  // host-side analysis
  int span_sum = 0, dx_sum = 0, dy_sum = 0;
  for (int w = 0; w + 64 <= N; w += 64) {
    int ymin = 1 << 30, ymax = -1;
    for (int i = w; i < w + 64; i++) {
      int y = log[i * 2 + 1];
      if (y < ymin) ymin = y;
      if (y > ymax) ymax = y;
    }
    span_sum += (ymax - ymin);
  }
  for (int i = 1; i < N; i++) {
    dx_sum += abs(int(log[i * 2 + 0]) - int(log[(i - 1) * 2 + 0]));
    dy_sum += abs(int(log[i * 2 + 1]) - int(log[(i - 1) * 2 + 1]));
  }
  printf("dispatched WGs: %u\n", *counter);
  printf("mean row-span of 64 consecutive WGs: %.1f  (row-major ~0-1, block/Morton ~8+)\n",
         span_sum / double(N / 64));
  printf("mean |dx|+|dy| between consecutive dispatches: %.1f\n",
         (dx_sum + dy_sum) / double(N - 1));
  printf("first 40 dispatches (x,y): ");
  for (int i = 0; i < 40 && i < N; i++)
    printf("(%u,%u) ", log[i * 2], log[i * 2 + 1]);
  printf("\n");
  free(counter, q); free(log, q);
  return 0;
}
