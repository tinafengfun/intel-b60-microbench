// Level Zero Host Runner for SPIR-V Microbenchmarks
// Loads .spv kernel, allocates device memory, runs with timing
//
// Usage:
//   ./spirv_runner kernel.spv kernel_name [wg_x [n_wg_x [repeats]]]
//
// Build:
//   g++ -std=c++17 -I/usr/include -lze_loader -lm -o spirv_runner spirv_runner.cpp

#define ZET_ENABLE_METRICS 1
#include <level_zero/ze_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <cmath>
#include <vector>
#include <fstream>
#include <chrono>

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
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <kernel.spv> <kernel_name> [wg_x [n_wg_x [repeats [buf_kb]]]]\n", argv[0]);
        return 1;
    }
    const char* spv_path = argv[1];
    const char* kern_name = argv[2];
    uint32_t wg_x = argc > 3 ? atoi(argv[3]) : 16;
    uint32_t nwg_x = argc > 4 ? atoi(argv[4]) : 1;
    int repeats = argc > 5 ? atoi(argv[5]) : 100;
    size_t buf_kb = argc > 6 ? atol(argv[6]) : 64;  // buffer size in KB

    // Init Level Zero
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

    ze_device_properties_t devProps = {ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
    CHECK_ZE(zeDeviceGetProperties(device, &devProps));
    printf("Device: %s  EUs: %u\n", devProps.name,
           devProps.numEUsPerSubslice * devProps.numSubslicesPerSlice * devProps.numSlices);

    ze_context_desc_t ctxDesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
    ze_context_handle_t context;
    CHECK_ZE(zeContextCreate(driver, &ctxDesc, &context));

    ze_command_queue_desc_t cqDesc = {};
    cqDesc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    cqDesc.ordinal = 0;
    cqDesc.index = 0;
    cqDesc.mode = ZE_COMMAND_QUEUE_MODE_DEFAULT;
    cqDesc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    ze_command_queue_handle_t cmdQueue;
    CHECK_ZE(zeCommandQueueCreate(context, device, &cqDesc, &cmdQueue));

    ze_command_list_desc_t clDesc = {};
    clDesc.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
    clDesc.commandQueueGroupOrdinal = 0;
    ze_command_list_handle_t cmdList;
    CHECK_ZE(zeCommandListCreate(context, device, &clDesc, &cmdList));

    // Load SPIR-V
    std::vector<uint8_t> spv;
    read_file(spv_path, spv);
    printf("SPIR-V: %s (%zu bytes)\n", spv_path, spv.size());

    ze_module_desc_t modDesc = {};
    modDesc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
    modDesc.format = ZE_MODULE_FORMAT_IL_SPIRV;
    modDesc.inputSize = spv.size();
    modDesc.pInputModule = spv.data();
    modDesc.pBuildFlags = "";
    ze_module_handle_t module;
    ze_result_t modResult = zeModuleCreate(context, device, &modDesc, &module, nullptr);
    if (modResult != ZE_RESULT_SUCCESS) {
        fprintf(stderr, "Module create failed: %d\nTry: ocloc compile -spirv_input -file %s -device bmg-g21\n", modResult, spv_path);
        return 1;
    }

    ze_kernel_desc_t kernDesc = {};
    kernDesc.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
    kernDesc.pKernelName = kern_name;
    ze_kernel_handle_t kernel;
    CHECK_ZE(zeKernelCreate(module, &kernDesc, &kernel));

    CHECK_ZE(zeKernelSetGroupSize(kernel, wg_x, 1, 1));
    printf("Kernel: %s  WG: %u  Grid: %u  Repeats: %d\n", kern_name, wg_x, nwg_x, repeats);

    // Allocate 4 shared buffers
    const size_t buf_size = buf_kb * 1024;
    const int N_BUF = 4;
    ze_device_mem_alloc_desc_t memDesc = {};
    memDesc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    ze_host_mem_alloc_desc_t hostDesc = {};
    hostDesc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;

    void* bufs[N_BUF] = {};
    for (int i = 0; i < N_BUF; i++) {
        CHECK_ZE(zeMemAllocShared(context, &memDesc, &hostDesc, buf_size, 64, device, &bufs[i]));
        float* f = (float*)bufs[i];
        for (size_t j = 0; j < buf_size/sizeof(float); j++) f[j] = 1.001f + j * 0.001f;
    }

    // Set kernel args (only set args the kernel accepts)
    for (int i = 0; i < N_BUF; i++) {
        ze_result_t r = zeKernelSetArgumentValue(kernel, i, sizeof(void*), &bufs[i]);
        if (r != ZE_RESULT_SUCCESS) break;  // stop at first invalid arg index
    }

    // Dispatch
    ze_group_count_t dispatch = {nwg_x, 1, 1};

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
    printf("\n--- Timing ---\n");
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
    double median = times_ns[times_ns.size()/2];

    // Standard deviation and 95% CI
    double sq_sum = 0;
    for (auto t : times_ns) sq_sum += (t - mean) * (t - mean);
    double stddev = sqrt(sq_sum / times_ns.size());
    double cv = (mean > 0) ? (stddev / mean * 100.0) : 0;
    // 95% CI: t_{0.025, n-1} ≈ 1.96 for large n
    double ci_half = 1.96 * stddev / sqrt((double)times_ns.size());

    printf("Runs=%d  Median=%.1f ns  Mean=%.1f ns  Min=%.1f ns  Max=%.1f ns\n",
           repeats, median, mean, times_ns.front(), times_ns.back());
    printf("StdDev=%.1f ns  CV=%.1f%%  95%%CI=[%.1f, %.1f] ns\n",
           stddev, cv, mean - ci_half, mean + ci_half);

    printf("\nOutput[0..7]:");
    float* out = (float*)bufs[0];
    for (int i = 0; i < 8; i++) printf(" %.4f", out[i]);
    printf("\n");

    for (int i = 0; i < N_BUF; i++) zeMemFree(context, bufs[i]);
    zeCommandListDestroy(cmdList);
    zeCommandQueueDestroy(cmdQueue);
    zeKernelDestroy(kernel);
    zeModuleDestroy(module);
    zeContextDestroy(context);
    return 0;
}
