// Query device properties for memory bandwidth
#include <sycl/sycl.hpp>
#include <cstdio>

int main() {
    sycl::queue q{sycl::gpu_selector_v};
    auto dev = q.get_device();

    printf("Device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
    printf("Global memory size: %zu MB\n", dev.get_info<sycl::info::device::global_mem_size>() / (1024*1024));
    printf("Local memory size: %zu KB\n", dev.get_info<sycl::info::device::local_mem_size>() / 1024);
    printf("Global memory bandwidth: %zu GB/s\n",
           dev.get_info<sycl::info::device::memory_bandwidth>() / (1024*1024*1024));
    printf("Max clock: %u MHz\n", dev.get_info<sycl::info::device::max_clock_frequency>());
    printf("Max work-groups: %zu\n", dev.get_info<sycl::info::device::max_work_groups>());
    printf("Max work-group size: %zu\n", dev.get_info<sycl::info::device::max_work_group_size>());
    printf("EUs: %u\n", dev.get_info<sycl::info::device::max_compute_units>());

    // Check memory bus width via extensions or properties
    auto ext_mem_bw = dev.get_info<sycl::info::device::memory_bandwidth>();
    printf("Memory bandwidth (exact): %zu bytes/sec = %.1f GB/s\n",
           ext_mem_bw, ext_mem_bw / 1e9);

    return 0;
}
