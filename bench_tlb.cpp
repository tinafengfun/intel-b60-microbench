// TLB investigation: compare random vs sequential vs page-stride pointer chase
// Tests whether latency growth in L1 range is TLB or set-associative conflicts.
// Build: icpx -fsycl -O2 -o bench_tlb bench_tlb.cpp

#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>

// Generate different permutation patterns
enum Pattern { RANDOM, SEQUENTIAL, PAGE_STRIDE, STRIDE_2 };

void gen_perm(int* perm, size_t n, Pattern pattern) {
    if (pattern == SEQUENTIAL) {
        for (size_t i = 0; i < n; i++) perm[i] = (i + 1) % n;
    } else if (pattern == RANDOM) {
        for (size_t i = 0; i < n; i++) perm[i] = (i + 1) % n;
        std::mt19937 rng(42);
        for (size_t i = n - 1; i > 0; i--) {
            size_t j = rng() % (i + 1);
            std::swap(perm[i], perm[j]);
        }
    } else if (pattern == PAGE_STRIDE) {
        // Stride by 1024 elements = 4KB page. Forces TLB pressure.
        // Access pattern: 0, 1024, 2048, ... wrapping within n elements
        size_t page_elems = 1024; // 4KB / 4B
        size_t n_pages = (n + page_elems - 1) / page_elems;
        for (size_t i = 0; i < n; i++) {
            size_t cur_page = i / page_elems;
            size_t cur_offset = i % page_elems;
            size_t next_page = (cur_page + 1) % n_pages;
            size_t next_offset = cur_offset; // same offset in next page
            size_t next_idx = next_page * page_elems + next_offset;
            if (next_idx >= n) next_idx = (next_idx % n);
            perm[i] = (int)next_idx;
        }
    } else if (pattern == STRIDE_2) {
        // Stride of 2 elements - tests low-level associativity
        for (size_t i = 0; i < n; i++) {
            perm[i] = ((i + 2) < n) ? (int)(i + 2) : (int)((i + 2) % n);
        }
    }
}

int main() {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::enable_profiling{}};
    auto dev = q.get_device();

    printf("=== TLB vs Set-Associative Conflict Investigation ===\n");
    printf("Device: %s\n", dev.get_info<sycl::info::device::name>().c_str());

    // Buffer sizes to test: 1K, 2K, 4K, 8K, 16K, 32K, 48K, 64K, 96K, 128K, 192K, 256K
    size_t sizes[] = {1024, 2048, 4096, 8192, 16384, 32768, 49152, 65536, 98304, 131072, 196608, 262144};
    int n_sizes = 12;
    int chase_steps = 4096;
    int n_reps = 50;

    const char* pattern_names[] = {"SEQUENTIAL", "RANDOM", "PAGE_STRIDE(4K)", "STRIDE_2"};
    Pattern patterns[] = {SEQUENTIAL, RANDOM, PAGE_STRIDE, STRIDE_2};

    // Header
    printf("\n%12s", "Size");
    for (int p = 0; p < 4; p++) printf(" %16s", pattern_names[p]);
    printf("\n");

    for (int si = 0; si < n_sizes; si++) {
        size_t buf_size = sizes[si];
        size_t n = buf_size / sizeof(int);
        printf("%8zu KB", buf_size/1024);

        for (int p = 0; p < 4; p++) {
            // Generate permutation on host
            std::vector<int> perm(n);
            gen_perm(perm.data(), n, patterns[p]);

            // Allocate device buffers
            int* d_buf = sycl::malloc_device<int>(n, q);
            int* d_out = sycl::malloc_device<int>(1, q);
            q.memcpy(d_buf, perm.data(), buf_size).wait();

            // Run pointer chase: sequential kernel
            double total_ns = 0;
            for (int r = 0; r < n_reps; r++) {
                auto ev = q.submit([&](sycl::handler& h) {
                    h.single_task([=]() {
                        int idx = 0;
                        for (int i = 0; i < chase_steps; i++) {
                            idx = d_buf[idx];
                        }
                        *d_out = idx;
                    });
                });
                ev.wait();
                if (r >= 5) {
                    auto t0 = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
                    auto t1 = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
                    total_ns += (t1 - t0);
                }
            }
            double avg_ns = total_ns / (n_reps - 5);
            double cyc_per_access = avg_ns / chase_steps * 2.4; // 2.4 GHz

            printf(" %10.1f(%4.0f)cyc", avg_ns / chase_steps, cyc_per_access);

            sycl::free(d_buf, q);
            sycl::free(d_out, q);
        }
        printf("\n");
    }

    printf("\nNote: 'cyc' = cycles per access at 2.4 GHz\n");
    printf("If TLB is the bottleneck, PAGE_STRIDE should show sharp jumps at TLB capacity.\n");
    printf("If set-associative conflicts, RANDOM should grow steadily within L1 range.\n");
    printf("SEQUENTIAL should be flat within L1 if no structural hazards.\n");

    return 0;
}
