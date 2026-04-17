// Memory Microbenchmark Host Runner for SPIR-V Kernels
// Supports pointer chase (int32 permutation) and bandwidth (float) kernels
//
// Usage:
//   ./mem_runner <kernel.spv> <kernel_name> <mode> <buf_kb> [wg_x [n_wg [repeats]]]
//   mode: chase | read | write
//
// Build:
//   g++ -std=c++17 -O2 -o mem_runner mem_runner.cpp -lze_loader -lm

#include <level_zero/ze_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <vector>
#include <fstream>
#include <chrono>
#include <random>

#define CHECK_ZE(call) do { \
    ze_result_t r = (call); \
    if (r != ZE_RESULT_SUCCESS) { \
        fprintf(stderr, "L0 error %d at %s:%d: %s\n", r, __FILE__, __LINE__, #call); \
        exit(1); \
    } \
} while(0)

static void read_file(const char* path, std::vector<uint8_t>& buf) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    size_t sz = f.tellg();
    f.seekg(0);
    buf.resize(sz);
    f.read((char*)buf.data(), sz);
}

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <kernel.spv> <kernel_name> <mode:chase|read|write> <buf_kb> [wg_x [n_wg [repeats]]]\n", argv[0]);
        return 1;
    }
    const char* spv_path = argv[1];
    const char* kern_name = argv[2];
    const char* mode = argv[3];
    size_t buf_kb = atol(argv[4]);
    uint32_t wg_x = argc > 5 ? atoi(argv[5]) : 256;
    uint32_t nwg_x = argc > 6 ? atoi(argv[6]) : 1;
    int repeats = argc > 7 ? atoi(argv[7]) : 50;

    bool is_chase = (strcmp(mode, "chase") == 0);

    CHECK_ZE(zeInit(0));
    uint32_t driverCount = 0;
    CHECK_ZE(zeDriverGet(&driverCount, nullptr));
    std::vector<ze_driver_handle_t> drivers(driverCount);
    CHECK_ZE(zeDriverGet(&driverCount, drivers.data()));
    ze_driver_handle_t driver = drivers[0];

    uint32_t deviceCount = 0;
    CHECK_ZE(zeDeviceGet(driver, &deviceCount, nullptr));
    std::vector<ze_device_handle_t> devices(deviceCount);
    CHECK_ZE(zeDeviceGet(driver, &deviceCount, devices.data()));
    ze_device_handle_t device = devices[0];

    ze_context_desc_t ctxDesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
    ze_context_handle_t context;
    CHECK_ZE(zeContextCreate(driver, &ctxDesc, &context));

    ze_command_queue_desc_t cqDesc = {};
    cqDesc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    cqDesc.ordinal = 0;
    cqDesc.mode = ZE_COMMAND_QUEUE_MODE_DEFAULT;
    cqDesc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    ze_command_queue_handle_t cmdQueue;
    CHECK_ZE(zeCommandQueueCreate(context, device, &cqDesc, &cmdQueue));

    ze_command_list_desc_t clDesc = {};
    clDesc.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
    clDesc.commandQueueGroupOrdinal = 0;
    ze_command_list_handle_t cmdList;
    CHECK_ZE(zeCommandListCreate(context, device, &clDesc, &cmdList));

    std::vector<uint8_t> spv;
    read_file(spv_path, spv);

    ze_module_desc_t modDesc = {};
    modDesc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
    modDesc.format = ZE_MODULE_FORMAT_IL_SPIRV;
    modDesc.inputSize = spv.size();
    modDesc.pInputModule = spv.data();
    ze_module_handle_t module;
    CHECK_ZE(zeModuleCreate(context, device, &modDesc, &module, nullptr));

    ze_kernel_desc_t kernDesc = {};
    kernDesc.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
    kernDesc.pKernelName = kern_name;
    ze_kernel_handle_t kernel;
    CHECK_ZE(zeKernelCreate(module, &kernDesc, &kernel));

    if (is_chase) wg_x = 1;  // Force single thread for pointer chase
    CHECK_ZE(zeKernelSetGroupSize(kernel, wg_x, 1, 1));

    size_t buf_size = buf_kb * 1024;
    const int N_BUF = 2;  // All kernels use 2 args (buf + out)
    ze_device_mem_alloc_desc_t memDesc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
    ze_host_mem_alloc_desc_t hostDesc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};

    void* bufs[N_BUF] = {};

    if (is_chase) {
        // Fill first buffer with random permutation (int32)
        size_t n = buf_size / sizeof(int);
        std::vector<int> perm(n);
        for (size_t i = 0; i < n; i++) perm[i] = (int)((i + 1) % n);  // sequential first
        // Fisher-Yates shuffle for random access pattern
        std::mt19937 rng(42);
        for (size_t i = n - 1; i > 0; i--) {
            size_t j = rng() % (i + 1);
            std::swap(perm[i], perm[j]);
        }
        // Make sure we start at index 0
        // Find which element points to 0 and ensure 0 is in the chain
        CHECK_ZE(zeMemAllocShared(context, &memDesc, &hostDesc, buf_size, 64, device, &bufs[0]));
        CHECK_ZE(zeMemAllocShared(context, &memDesc, &hostDesc, 64, 64, device, &bufs[1]));
        memcpy(bufs[0], perm.data(), buf_size);
        // Zero the output buffer
        memset(bufs[1], 0, 64);
    } else {
        // Bandwidth: fill with float data
        CHECK_ZE(zeMemAllocShared(context, &memDesc, &hostDesc, buf_size, 64, device, &bufs[0]));
        CHECK_ZE(zeMemAllocShared(context, &memDesc, &hostDesc, 64, 64, device, &bufs[1]));
        float* f = (float*)bufs[0];
        for (size_t i = 0; i < buf_size/sizeof(float); i++) f[i] = 1.0f + i * 0.001f;
        memset(bufs[1], 0, 64);
    }

    for (int i = 0; i < N_BUF; i++) {
        ze_result_t r = zeKernelSetArgumentValue(kernel, i, sizeof(void*), &bufs[i]);
        if (r != ZE_RESULT_SUCCESS) break;
    }

    ze_group_count_t dispatch = {nwg_x, 1, 1};
    printf("Kernel: %s  Mode: %s  Buf: %zuKB  WG: %u  Grid: %u  Repeats: %d\n",
           kern_name, mode, buf_kb, wg_x, nwg_x, repeats);

    // Warmup
    for (int i = 0; i < 10; i++) {
        CHECK_ZE(zeCommandListAppendLaunchKernel(cmdList, kernel, &dispatch, nullptr, 0, nullptr));
        CHECK_ZE(zeCommandListAppendBarrier(cmdList, nullptr, 0, nullptr));
    }
    CHECK_ZE(zeCommandListClose(cmdList));
    CHECK_ZE(zeCommandQueueExecuteCommandLists(cmdQueue, 1, &cmdList, nullptr));
    CHECK_ZE(zeCommandQueueSynchronize(cmdQueue, UINT64_MAX));
    CHECK_ZE(zeCommandListReset(cmdList));

    // Timed runs
    std::vector<double> times_ns;
    for (int r = 0; r < repeats; r++) {
        CHECK_ZE(zeCommandListAppendLaunchKernel(cmdList, kernel, &dispatch, nullptr, 0, nullptr));
        CHECK_ZE(zeCommandListAppendBarrier(cmdList, nullptr, 0, nullptr));
        CHECK_ZE(zeCommandListClose(cmdList));

        auto t0 = std::chrono::high_resolution_clock::now();
        CHECK_ZE(zeCommandQueueExecuteCommandLists(cmdQueue, 1, &cmdList, nullptr));
        CHECK_ZE(zeCommandQueueSynchronize(cmdQueue, UINT64_MAX));
        auto t1 = std::chrono::high_resolution_clock::now();
        times_ns.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count());
        CHECK_ZE(zeCommandListReset(cmdList));
    }

    std::sort(times_ns.begin(), times_ns.end());
    double sum = 0;
    for (auto t : times_ns) sum += t;
    double mean = sum / times_ns.size();
    printf("Runs=%d  Median=%.1f ns  Mean=%.1f ns  Min=%.1f ns  Max=%.1f ns\n",
           repeats, times_ns[times_ns.size()/2], mean, times_ns.front(), times_ns.back());

    // Compute derived metrics
    double median_ns = times_ns[times_ns.size()/2];
    if (is_chase) {
        int chase_len = 4096;
        double ghz = 2.4;
        double cyc_per_access = median_ns * ghz / chase_len;
        printf("Chase=%d  Latency: %.1f cycles/access (%.1f ns)\n",
               chase_len, cyc_per_access, median_ns / chase_len);
    } else {
        size_t total_threads = (size_t)nwg_x * wg_x;
        double total_bytes = (double)total_threads * 256 * sizeof(float);  // 256 reads/thread * 4 bytes
        double bw_gbps = total_bytes / (median_ns * 1e-9) / 1e9;
        printf("BW: %.1f GB/s (%.1f MB read in %.0f ns)\n",
               bw_gbps, total_bytes / 1e6, median_ns);
    }

    // Verify output
    if (is_chase) {
        int* out = (int*)bufs[1];
        printf("Chase final idx: %d\n", out[0]);
    } else {
        float* out = (float*)bufs[1];
        printf("Sum[0]: %.4f\n", out[0]);
    }

    for (int i = 0; i < N_BUF; i++) zeMemFree(context, bufs[i]);
    zeCommandListDestroy(cmdList);
    zeCommandQueueDestroy(cmdQueue);
    zeKernelDestroy(kernel);
    zeModuleDestroy(module);
    zeContextDestroy(context);
    return 0;
}
