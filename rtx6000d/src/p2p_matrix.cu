// P2P bandwidth matrix across all visible devices (6000D x6: physical 0,3,4,5,6,7)
#include "bench_common.h"

int main() {
    int n = 0;
    cudaGetDeviceCount(&n);
    printf("# devices=%d\n", n);
    if (n < 2) { printf("need >=2 devices\n"); return 0; }
    size_t sz = 256LL << 20;
    void* buf[16];
    for (int i = 0; i < n; i++) { CK(cudaSetDevice(i)); CK(cudaMalloc(&buf[i], sz)); }
    // enable all peers
    for (int i = 0; i < n; i++) {
        CK(cudaSetDevice(i));
        for (int j = 0; j < n; j++) {
            if (i == j) continue;
            int can = 0;
            cudaDeviceCanAccessPeer(&can, i, j);
            if (can) { cudaDeviceEnablePeerAccess(j, 0); cudaGetLastError(); }
        }
    }
    printf("# matrix: rows=src, cols=dst, GB/s\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) continue;
            CK(cudaSetDevice(i));
            cudaError_t e0 = cudaGetLastError();
            double ms = -1;
            cudaEvent_t a, b;
            cudaEventCreate(&a); cudaEventCreate(&b);
            // warmup
            cudaError_t e = cudaMemcpyPeer(buf[j], j, buf[i], i, sz);
            if (e != cudaSuccess) {
                char p[32]; snprintf(p, 32, "gpu%d_to_gpu%d", i, j);
                csv("p2p_matrix", p, 0, "GB/s", cudaGetErrorString(e));
                cudaGetLastError();
                continue;
            }
            cudaDeviceSynchronize();
            cudaEventRecord(a);
            for (int r = 0; r < 5; r++) cudaMemcpyPeer(buf[j], j, buf[i], i, sz);
            cudaEventRecord(b);
            cudaEventSynchronize(b);
            float tms; cudaEventElapsedTime(&tms, a, b);
            ms = tms / 5;
            char p[32]; snprintf(p, 32, "gpu%d_to_gpu%d", i, j);
            csv("p2p_matrix", p, sz / (ms * 1e6), "GB/s");
        }
    }
    return 0;
}
