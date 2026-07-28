// L0 overlap test: compute-queue kernel vs copy-queue DMA, DeepEP-style
// communication/computation concurrency on B70 (Xe2).
// Measures: T_compute alone, T_copy alone, T_both concurrent ->
// overlap efficiency and mutual interference.
#include <level_zero/ze_api.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>

#define CHECK(r) do { ze_result_t _r = (r); if (_r != ZE_RESULT_SUCCESS) { \
  fprintf(stderr, "L0 error %d at line %d\n", _r, __LINE__); exit(1); } } while (0)

static std::vector<uint8_t> read_file(const char *p) {
  FILE *f = fopen(p, "rb"); if (!f) { fprintf(stderr, "no file %s\n", p); exit(1); }
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> b(n); fread(b.data(), 1, n, f); fclose(f); return b;
}

static double now_s() {
  return std::chrono::duration<double>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char **argv) {
  int spin_iters = argc > 1 ? atoi(argv[1]) : 400000;   // ~hundreds of ms
  int copy_mb    = argc > 2 ? atoi(argv[2]) : 512;      // per copy op
  int copy_ops   = argc > 3 ? atoi(argv[3]) : 16;       // total ~8GB traffic

  CHECK(zeInit(0));
  uint32_t ndrv = 0; zeDriverGet(&ndrv, nullptr);
  std::vector<ze_driver_handle_t> drvs(ndrv); zeDriverGet(&ndrv, drvs.data());
  ze_device_handle_t dev = nullptr; ze_context_handle_t ctx = nullptr;
  for (auto d : drvs) {
    uint32_t nd = 0; zeDeviceGet(d, &nd, nullptr);
    std::vector<ze_device_handle_t> ds(nd); zeDeviceGet(d, &nd, ds.data());
    for (auto x : ds) {
      ze_device_properties_t p = {ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
      zeDeviceGetProperties(x, &p);
      if (p.deviceId == 0xe223) { dev = x; }
    }
    if (dev) {
      ze_context_desc_t cd = {ZE_STRUCTURE_TYPE_CONTEXT_DESC};
      CHECK(zeContextCreate(d, &cd, &ctx));
      break;
    }
  }
  if (!dev) { fprintf(stderr, "B70 not found\n"); return 1; }

  // queue groups
  uint32_t nqg = 0;
  zeDeviceGetCommandQueueGroupProperties(dev, &nqg, nullptr);
  std::vector<ze_command_queue_group_properties_t> qg(nqg);
  for (auto &x : qg) x.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
  zeDeviceGetCommandQueueGroupProperties(dev, &nqg, qg.data());
  uint32_t compute_g = UINT32_MAX, copy_g = UINT32_MAX;
  for (uint32_t i = 0; i < nqg; i++) {
    printf("group %u: flags=0x%x numQueues=%u\n", i, qg[i].flags, qg[i].numQueues);
    if ((qg[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) && compute_g == UINT32_MAX) compute_g = i;
    else if ((qg[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY) && copy_g == UINT32_MAX) copy_g = i;
  }
  printf("compute_group=%u copy_group=%u\n", compute_g, copy_g);

  auto make_queue = [&](uint32_t group) {
    ze_command_queue_desc_t qd = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC};
    qd.ordinal = group; qd.index = 0; qd.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    ze_command_queue_handle_t qq;
    CHECK(zeCommandQueueCreate(ctx, dev, &qd, &qq));
    return qq;
  };
  auto make_cl = [&](uint32_t group) {
    ze_command_list_desc_t ld = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC};
    ld.commandQueueGroupOrdinal = group;
    ze_command_list_handle_t l;
    CHECK(zeCommandListCreate(ctx, dev, &ld, &l));
    return l;
  };
  ze_command_queue_handle_t cq = make_queue(compute_g), xq = make_queue(copy_g);
  ze_command_list_handle_t cl_compute = make_cl(compute_g), cl_copy = make_cl(copy_g);

  // spin kernel (ocloc-compiled native bin)
  auto bin = read_file("spin_bmg.bin");
  ze_module_desc_t md = {ZE_STRUCTURE_TYPE_MODULE_DESC};
  md.format = ZE_MODULE_FORMAT_NATIVE; md.inputSize = bin.size(); md.pInputModule = bin.data();
  ze_module_handle_t mod;
  CHECK(zeModuleCreate(ctx, dev, &md, &mod, nullptr));
  ze_kernel_desc_t kd = {ZE_STRUCTURE_TYPE_KERNEL_DESC};
  kd.pKernelName = "spin";
  ze_kernel_handle_t kern;
  CHECK(zeKernelCreate(mod, &kd, &kern));
  CHECK(zeKernelSetGroupSize(kern, 64, 1, 1));

  // args + buffers
  ze_device_mem_alloc_desc_t dd = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
  float *kout; void *src, *dst;
  CHECK(zeMemAllocDevice(ctx, &dd, 256, 64, dev, (void**)&kout));
  size_t csz = (size_t)copy_mb * 1024 * 1024;
  CHECK(zeMemAllocDevice(ctx, &dd, csz, 4096, dev, &src));
  CHECK(zeMemAllocDevice(ctx, &dd, csz, 4096, dev, &dst));
  CHECK(zeKernelSetArgumentValue(kern, 0, sizeof(kout), &kout));
  CHECK(zeKernelSetArgumentValue(kern, 1, sizeof(spin_iters), &spin_iters));
  ze_group_count_t gc = {256, 1, 1};   // 256 WGs x 64 = 16384 threads

  // command lists
  CHECK(zeCommandListAppendLaunchKernel(cl_compute, kern, &gc, nullptr, 0, nullptr));
  for (int i = 0; i < copy_ops; i++)
    CHECK(zeCommandListAppendMemoryCopy(cl_copy, dst, src, csz, nullptr, 0, nullptr));
  CHECK(zeCommandListClose(cl_compute));
  CHECK(zeCommandListClose(cl_copy));

  double gb = (double)copy_ops * copy_mb / 1024.0;

  // 1) compute alone
  double t0 = now_s();
  CHECK(zeCommandQueueExecuteCommandLists(cq, 1, &cl_compute, nullptr));
  CHECK(zeCommandQueueSynchronize(cq, UINT64_MAX));
  double t_comp = now_s() - t0;

  // 2) copy alone
  t0 = now_s();
  CHECK(zeCommandQueueExecuteCommandLists(xq, 1, &cl_copy, nullptr));
  CHECK(zeCommandQueueSynchronize(xq, UINT64_MAX));
  double t_copy = now_s() - t0;

  // 3) concurrent: compute on compute queue + copies on copy queue
  t0 = now_s();
  CHECK(zeCommandQueueExecuteCommandLists(cq, 1, &cl_compute, nullptr));
  CHECK(zeCommandQueueExecuteCommandLists(xq, 1, &cl_copy, nullptr));
  double t_both_sub = now_s() - t0;
  CHECK(zeCommandQueueSynchronize(cq, UINT64_MAX));
  double t_comp_done = now_s() - t0;
  CHECK(zeCommandQueueSynchronize(xq, UINT64_MAX));
  double t_all = now_s() - t0;

  // 4) compute+compute: same queue, two lists in one execute
  ze_command_list_handle_t cl2 = make_cl(compute_g);
  CHECK(zeCommandListAppendLaunchKernel(cl2, kern, &gc, nullptr, 0, nullptr));
  CHECK(zeCommandListClose(cl2));
  ze_command_list_handle_t two[2] = {cl_compute, cl2};
  t0 = now_s();
  CHECK(zeCommandQueueExecuteCommandLists(cq, 2, two, nullptr));
  CHECK(zeCommandQueueSynchronize(cq, UINT64_MAX));
  double t_2comp = now_s() - t0;

  printf("\n=== B70 L0 overlap (spin_iters=%d, copy %dMB x %d = %.1f GB) ===\n",
         spin_iters, copy_mb, copy_ops, gb);
  printf("compute alone        : %8.1f ms\n", t_comp * 1e3);
  printf("copy alone           : %8.1f ms  (%.1f GB/s device-to-device)\n",
         t_copy * 1e3, gb / t_copy);
  printf("concurrent total     : %8.1f ms  (submit took %.2f ms)\n", t_all * 1e3, t_both_sub * 1e3);
  printf("  compute finished at: %8.1f ms  (slowdown vs alone: %.2fx)\n",
         t_comp_done * 1e3, t_comp_done / t_comp);
  printf("overlap efficiency   : (T_c+T_x)/T_both = %.2f\n", (t_comp + t_copy) / t_all);
  printf("compute x2 one queue : %8.1f ms  (vs 2x serial %.1f ms, speedup %.2fx)\n",
         t_2comp * 1e3, 2 * t_comp * 1e3, 2 * t_comp / t_2comp);
  return 0;
}
