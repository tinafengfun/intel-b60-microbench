// B70 engine-occupancy probe: does a blitter (Copy Engine) DMA consume EU?
// Runs a sustained load (copy / spin / both) for N seconds while an external
// monitor samples /proc/<pid>/fdinfo/* drm-engine-* counters.
// Also prints the L0 P2P capability matrix (B70 <-> other devices).
#include <level_zero/ze_api.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <unistd.h>

#define CHECK(r) do { ze_result_t _r = (r); if (_r != ZE_RESULT_SUCCESS) { \
  fprintf(stderr, "L0 error %d at line %d\n", _r, __LINE__); exit(1); } } while (0)

static std::vector<uint8_t> read_file(const char *p) {
  FILE *f = fopen(p, "rb"); if (!f) { fprintf(stderr, "no file %s\n", p); exit(1); }
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> b(n); fread(b.data(), 1, n, f); fclose(f); return b;
}

int main(int argc, char **argv) {
  const char *mode = argc > 1 ? argv[1] : "copy";   // copy | spin | both
  int seconds = argc > 2 ? atoi(argv[2]) : 20;

  CHECK(zeInit(0));
  uint32_t ndrv = 0; zeDriverGet(&ndrv, nullptr);
  std::vector<ze_driver_handle_t> drvs(ndrv); zeDriverGet(&ndrv, drvs.data());
  ze_device_handle_t b70 = nullptr; ze_context_handle_t ctx = nullptr;
  std::vector<ze_device_handle_t> all;
  for (auto d : drvs) {
    uint32_t nd = 0; zeDeviceGet(d, &nd, nullptr);
    std::vector<ze_device_handle_t> ds(nd); zeDeviceGet(d, &nd, ds.data());
    for (auto x : ds) {
      all.push_back(x);
      ze_device_properties_t p = {ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
      zeDeviceGetProperties(x, &p);
      if (p.deviceId == 0xe223 && !b70) b70 = x;
    }
    if (!ctx) {
      ze_context_desc_t cd = {ZE_STRUCTURE_TYPE_CONTEXT_DESC};
      CHECK(zeContextCreate(d, &cd, &ctx));
    }
  }
  if (!b70) { fprintf(stderr, "B70 not found\n"); return 1; }

  // P2P capability matrix is printed at the END: querying the iGPU pair
  // aborts inside NEO memory manager, so do it after the load test.

  // queues
  auto make_queue = [&](uint32_t group) {
    ze_command_queue_desc_t qd = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC};
    qd.ordinal = group; qd.index = 0; qd.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    ze_command_queue_handle_t qq;
    CHECK(zeCommandQueueCreate(ctx, b70, &qd, &qq));
    return qq;
  };
  auto make_cl = [&](uint32_t group) {
    ze_command_list_desc_t ld = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC};
    ld.commandQueueGroupOrdinal = group;
    ze_command_list_handle_t l;
    CHECK(zeCommandListCreate(ctx, b70, &ld, &l));
    return l;
  };
  ze_command_queue_handle_t cq = make_queue(0), xq = make_queue(1);

  ze_device_mem_alloc_desc_t dd = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
  void *src, *dst; float *kout;
  size_t csz = 512ull * 1024 * 1024;
  CHECK(zeMemAllocDevice(ctx, &dd, csz, 4096, b70, &src));
  CHECK(zeMemAllocDevice(ctx, &dd, csz, 4096, b70, &dst));
  CHECK(zeMemAllocDevice(ctx, &dd, 256, 64, b70, (void **)&kout));

  // copy list: 8 GB per execute
  ze_command_list_handle_t cl_copy = make_cl(1);
  for (int i = 0; i < 16; i++)
    CHECK(zeCommandListAppendMemoryCopy(cl_copy, dst, src, csz, nullptr, 0, nullptr));
  CHECK(zeCommandListClose(cl_copy));

  // spin kernel list (~25 ms per execute at 800k iters)
  auto bin = read_file("spin_bmg.bin");
  ze_module_desc_t md = {ZE_STRUCTURE_TYPE_MODULE_DESC};
  md.format = ZE_MODULE_FORMAT_NATIVE; md.inputSize = bin.size(); md.pInputModule = bin.data();
  ze_module_handle_t mod;
  CHECK(zeModuleCreate(ctx, b70, &md, &mod, nullptr));
  ze_kernel_desc_t kd = {ZE_STRUCTURE_TYPE_KERNEL_DESC};
  kd.pKernelName = "spin";
  ze_kernel_handle_t kern;
  CHECK(zeKernelCreate(mod, &kd, &kern));
  CHECK(zeKernelSetGroupSize(kern, 64, 1, 1));
  int iters = 800000;
  CHECK(zeKernelSetArgumentValue(kern, 0, sizeof(kout), &kout));
  CHECK(zeKernelSetArgumentValue(kern, 1, sizeof(iters), &iters));
  ze_group_count_t gc = {256, 1, 1};
  ze_command_list_handle_t cl_spin = make_cl(0);
  CHECK(zeCommandListAppendLaunchKernel(cl_spin, kern, &gc, nullptr, 0, nullptr));
  CHECK(zeCommandListClose(cl_spin));

  printf("PID=%d mode=%s seconds=%d -- sample /proc/%d/fdinfo/* drm-engine-* now\n",
         getpid(), mode, seconds, getpid());
  fflush(stdout);

  bool do_copy = strstr(mode, "copy") != nullptr;
  bool do_spin = strstr(mode, "spin") != nullptr || strstr(mode, "both") != nullptr;
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
  while (std::chrono::steady_clock::now() < deadline) {
    if (do_copy) CHECK(zeCommandQueueExecuteCommandLists(xq, 1, &cl_copy, nullptr));
    if (do_spin) CHECK(zeCommandQueueExecuteCommandLists(cq, 1, &cl_spin, nullptr));
    CHECK(zeCommandQueueSynchronize(do_spin ? cq : xq, UINT64_MAX));
  }
  printf("DONE\n");
  // P2P capability matrix (last: may abort in NEO on the iGPU pair)
  for (auto x : all) {
    ze_device_properties_t p = {ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
    zeDeviceGetProperties(x, &p);
    ze_bool_t acc = 0;
    zeDeviceCanAccessPeer(b70, x, &acc);
    printf("P2P B70 -> 0x%04x : %s\n", p.deviceId, acc ? "YES" : "no");
  }
  fflush(stdout);
  return 0;
}
