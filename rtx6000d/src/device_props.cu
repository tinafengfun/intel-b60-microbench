#include "bench_common.h"

__global__ void trivial() {}

#define ATTR(name, scale, unit) do { int v = -1; cudaDeviceGetAttribute(&v, cudaDevAttr##name, 0); csv("device_props", #name, (double)v / (scale), unit); } while (0)

int main() {
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    printf("# name=%s cc=%d.%d SMs=%d\n", p.name, p.major, p.minor, p.multiProcessorCount);
    csv("device_props", "name_cc", p.major * 10 + p.minor, "cc", p.name);
    csv("device_props", "SM_count", p.multiProcessorCount, "count");
    ATTR(ClockRate, 1e6, "GHz");
    ATTR(MemoryClockRate, 1e6, "MHz");
    ATTR(GlobalMemoryBusWidth, 1, "bit");
    ATTR(L2CacheSize, 1048576.0, "MiB");
    size_t freeb = 0, totb = 0;
    CK(cudaMemGetInfo(&freeb, &totb));
    csv("device_props", "totalGlobalMem", totb / 1073741824.0, "GiB");
    int busw = -1, memclk = -1, clk = -1;
    cudaDeviceGetAttribute(&busw, cudaDevAttrGlobalMemoryBusWidth, 0);
    cudaDeviceGetAttribute(&memclk, cudaDevAttrMemoryClockRate, 0);
    cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0);
    double theo_bw = (double)busw / 8.0 * (double)memclk * 2.0 / 1e6;  // GB/s (DDR)
    csv("device_props", "theoretical_dram_bw", theo_bw, "GB/s");
    csv("device_props", "sharedMemPerMultiprocessor", p.sharedMemPerMultiprocessor / 1024.0, "KiB");
    csv("device_props", "sharedMemPerBlockOptin", p.sharedMemPerBlockOptin / 1024.0, "KiB");
    csv("device_props", "regsPerMultiprocessor", p.regsPerMultiprocessor, "count");
    csv("device_props", "maxThreadsPerMultiProcessor", p.maxThreadsPerMultiProcessor, "count");
    csv("device_props", "maxThreadsPerBlock", p.maxThreadsPerBlock, "count");
    csv("device_props", "warpSize", p.warpSize, "count");
    ATTR(MaxBlocksPerMultiprocessor, 1, "count");
    ATTR(MaxSharedMemoryPerBlockOptin, 1024.0, "KiB");
    csv("device_props", "persistingL2CacheMaxSize", p.persistingL2CacheMaxSize / 1048576.0, "MiB");
    ATTR(MemoryPoolsSupported, 1, "bool");
    csv("device_props", "concurrentKernels", p.concurrentKernels, "bool");
    csv("device_props", "asyncEngineCount", p.asyncEngineCount, "count");
    csv("device_props", "ECCEnabled", p.ECCEnabled, "bool");
    csv("device_props", "unifiedAddressing", p.unifiedAddressing, "bool");
    csv("device_props", "cooperativeLaunch", p.cooperativeLaunch, "bool");
    csv("device_props", "clusterLaunch", p.clusterLaunch, "bool");
    csv("device_props", "pciBusID", p.pciBusID, "id");
    double fp32 = (double)p.multiProcessorCount * 128 * 2 * (double)clk / 1e9;
    csv("device_props", "theoretical_fp32_at_rated_clk", fp32 / 1000.0, "TFLOPS", "128 FMA/SM/clk x rated clk");
    int nb = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, trivial, 1024, 0);
    csv("device_props", "max_warps_per_SM_1024t", nb * 32, "warps");
    return 0;
}
