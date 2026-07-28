// B70 cross-card P2P via Copy Engine: capability matrix, bandwidth, and
// sustained P2P DMA load for fdinfo engine-occupancy sampling.
// Usage: l0_p2p [peer_idx=1] [load_secs=14] [mb_per_copy=512] [ops=8]
// Phases: 1) 8x8 CanAccessPeer matrix  2) P2P bandwidth dev0<->peer
//         3) sustained P2P copy load (for /proc/pid/fdinfo sampling)
#include <level_zero/ze_api.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <unistd.h>

#define CHECK(r) do { ze_result_t _r = (r); if (_r != ZE_RESULT_SUCCESS) { \
  fprintf(stderr, "L0 error 0x%x at line %d\n", _r, __LINE__); exit(1); } } while (0)

static double now_s() {
  return std::chrono::duration<double>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char **argv) {
  int peer_idx = argc > 1 ? atoi(argv[1]) : 1;
  int load_secs = argc > 2 ? atoi(argv[2]) : 14;
  int mb = argc > 3 ? atoi(argv[3]) : 512;
  int ops = argc > 4 ? atoi(argv[4]) : 8;

  CHECK(zeInit(0));
  uint32_t ndrv = 0; zeDriverGet(&ndrv, nullptr);
  std::vector<ze_driver_handle_t> drvs(ndrv); zeDriverGet(&ndrv, drvs.data());
  ze_context_handle_t ctx = nullptr;
  std::vector<ze_device_handle_t> b70s;
  for (auto d : drvs) {
    uint32_t nd = 0; zeDeviceGet(d, &nd, nullptr);
    std::vector<ze_device_handle_t> ds(nd); zeDeviceGet(d, &nd, ds.data());
    for (auto x : ds) {
      ze_device_properties_t p = {ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
      zeDeviceGetProperties(x, &p);
      if (p.deviceId == 0xe223) b70s.push_back(x);
    }
    if (!ctx && !b70s.empty()) {
      ze_context_desc_t cd = {ZE_STRUCTURE_TYPE_CONTEXT_DESC};
      CHECK(zeContextCreate(d, &cd, &ctx));
    }
  }
  int n = b70s.size();
  printf("found %d B70 devices\n", n);
  if (n < 2) return 1;

  // phase 1: P2P capability matrix
  printf("CanAccessPeer matrix (row=from, col=to):\n    ");
  for (int j = 0; j < n; j++) printf(" d%d", j);
  printf("\n");
  for (int i = 0; i < n; i++) {
    printf(" d%d ", i);
    for (int j = 0; j < n; j++) {
      ze_bool_t acc = 0;
      zeDeviceCanAccessPeer(b70s[i], b70s[j], &acc);
      printf(" %2c", acc ? 'Y' : '.');
    }
    printf("\n");
  }
  fflush(stdout);

  if (peer_idx >= n) peer_idx = 1;
  ze_device_handle_t d0 = b70s[0], d1 = b70s[peer_idx];

  auto copy_group = [&](ze_device_handle_t dev) {
    uint32_t nqg = 0;
    zeDeviceGetCommandQueueGroupProperties(dev, &nqg, nullptr);
    std::vector<ze_command_queue_group_properties_t> qg(nqg);
    for (auto &x : qg) x.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
    zeDeviceGetCommandQueueGroupProperties(dev, &nqg, qg.data());
    for (uint32_t i = 0; i < nqg; i++)
      if ((qg[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY) &&
          !(qg[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE)) return i;
    return 0u;
  };
  auto make_queue = [&](ze_device_handle_t dev, uint32_t group) {
    ze_command_queue_desc_t qd = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC};
    qd.ordinal = group; qd.index = 0; qd.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    ze_command_queue_handle_t qq;
    CHECK(zeCommandQueueCreate(ctx, dev, &qd, &qq));
    return qq;
  };

  // phase 2: bandwidth dev0 -> dev1 and back, via copy engine
  size_t csz = (size_t)mb * 1024 * 1024;
  ze_device_mem_alloc_desc_t dd = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
  void *src0, *dst0, *src1, *dst1;
  CHECK(zeMemAllocDevice(ctx, &dd, csz, 4096, d0, &src0));
  CHECK(zeMemAllocDevice(ctx, &dd, csz, 4096, d0, &dst0));
  CHECK(zeMemAllocDevice(ctx, &dd, csz, 4096, d1, &src1));
  CHECK(zeMemAllocDevice(ctx, &dd, csz, 4096, d1, &dst1));

  auto bw_dir = [&](ze_device_handle_t exec_dev, void *dst, void *src, const char *tag) {
    uint32_t g = copy_group(exec_dev);
    ze_command_queue_handle_t q = make_queue(exec_dev, g);
    ze_command_list_desc_t ld = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC};
    ld.commandQueueGroupOrdinal = g;
    ze_command_list_handle_t cl;
    CHECK(zeCommandListCreate(ctx, exec_dev, &ld, &cl));
    for (int i = 0; i < ops; i++)
      CHECK(zeCommandListAppendMemoryCopy(cl, dst, src, csz, nullptr, 0, nullptr));
    CHECK(zeCommandListClose(cl));
    // warmup
    CHECK(zeCommandQueueExecuteCommandLists(q, 1, &cl, nullptr));
    CHECK(zeCommandQueueSynchronize(q, UINT64_MAX));
    double best = 1e9;
    for (int r = 0; r < 3; r++) {
      double t0 = now_s();
      CHECK(zeCommandQueueExecuteCommandLists(q, 1, &cl, nullptr));
      CHECK(zeCommandQueueSynchronize(q, UINT64_MAX));
      double dt = now_s() - t0;
      if (dt < best) best = dt;
    }
    double gb = (double)ops * mb / 1024.0;
    printf("P2P %-14s : %8.1f ms for %.1f GB -> %6.1f GB/s (copy engine)\n",
           tag, best * 1e3, gb, gb / best);
    fflush(stdout);
    zeCommandListDestroy(cl);
    zeCommandQueueDestroy(q);
  };

  ze_result_t rc;
  rc = ZE_RESULT_SUCCESS;
  // intra-card control first: dev0 local d2d
  bw_dir(d0, dst0, src0, "d0->d0 (local)");
  bw_dir(d0, dst1, src0, "d0->d1");
  bw_dir(d1, dst0, src1, "d1->d0");
  (void)rc;

  // phase 3: sustained P2P load for fdinfo sampling
  printf("LOAD_START PID=%d peer=d%d secs=%d -- sample fdinfo now\n",
         getpid(), peer_idx, load_secs);
  fflush(stdout);
  uint32_t g = copy_group(d0);
  ze_command_queue_handle_t q = make_queue(d0, g);
  ze_command_list_desc_t ld = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC};
  ld.commandQueueGroupOrdinal = g;
  ze_command_list_handle_t cl;
  CHECK(zeCommandListCreate(ctx, d0, &ld, &cl));
  for (int i = 0; i < ops; i++)
    CHECK(zeCommandListAppendMemoryCopy(cl, dst1, src0, csz, nullptr, 0, nullptr));
  CHECK(zeCommandListClose(cl));
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(load_secs);
  while (std::chrono::steady_clock::now() < deadline) {
    CHECK(zeCommandQueueExecuteCommandLists(q, 1, &cl, nullptr));
    CHECK(zeCommandQueueSynchronize(q, UINT64_MAX));
  }
  printf("DONE\n");
  return 0;
}
