// L0 queue-group + cache properties probe (MultiQ / shared-address hardware view)
#include <level_zero/ze_api.h>
#include <cstdio>
#include <vector>

int main() {
  zeInit(0);
  uint32_t ndrv = 0; zeDriverGet(&ndrv, nullptr);
  std::vector<ze_driver_handle_t> drv(ndrv); zeDriverGet(&ndrv, drv.data());
  for (auto d : drv) {
    uint32_t ndev = 0; zeDeviceGet(d, &ndev, nullptr);
    std::vector<ze_device_handle_t> devs(ndev); zeDeviceGet(d, &ndev, devs.data());
    for (auto dev : devs) {
      ze_device_properties_t p = {ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
      zeDeviceGetProperties(dev, &p);
      printf("Device: %s (0x%x) EUs=%u\n", p.name, p.deviceId, p.numEUsPerSubslice * p.numSubslicesPerSlice * p.numSlices);
      uint32_t nqg = 0;
      zeDeviceGetCommandQueueGroupProperties(dev, &nqg, nullptr);
      std::vector<ze_command_queue_group_properties_t> qg(nqg);
      for (auto &x : qg) x.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
      zeDeviceGetCommandQueueGroupProperties(dev, &nqg, qg.data());
      printf("Command queue groups: %u\n", nqg);
      for (uint32_t i = 0; i < nqg; i++)
        printf("  group %u: flags=0x%x numQueues=%u async=%d\n", i,
               qg[i].flags, qg[i].numQueues,
               !!(qg[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE));
      uint32_t ncache = 0;
      zeDeviceGetCacheProperties(dev, &ncache, nullptr);
      std::vector<ze_device_cache_properties_t> cp(ncache);
      for (auto &x : cp) x.stype = ZE_STRUCTURE_TYPE_DEVICE_CACHE_PROPERTIES;
      zeDeviceGetCacheProperties(dev, &ncache, cp.data());
      for (uint32_t i = 0; i < ncache; i++)
        printf("cache %u: flags=0x%x size=%zu KB\n", i, cp[i].flags, cp[i].cacheSize / 1024);
    }
  }
  return 0;
}
