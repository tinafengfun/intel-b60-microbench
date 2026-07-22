// PCIe H2D/D2H bandwidth + P2P between two GPUs (needs 2 visible devices)
#include "bench_common.h"

int main() {
    int ndev = 0;
    cudaGetDeviceCount(&ndev);
    printf("# visible devices: %d\n", ndev);
    size_t sz = 512LL << 20;

    // ---------- pinned H2D / D2H ----------
    void *hp, *dd;
    CK(cudaHostAlloc(&hp, sz, cudaHostAllocDefault));
    CK(cudaMalloc(&dd, sz));
    memset(hp, 1, sz);
    double ms = time_ms([&] { CK(cudaMemcpy(dd, hp, sz, cudaMemcpyHostToDevice)); }, 3, 10);
    csv("pcie", "H2D_pinned", sz / (ms * 1e6), "GB/s");
    ms = time_ms([&] { CK(cudaMemcpy(hp, dd, sz, cudaMemcpyDeviceToHost)); }, 3, 10);
    csv("pcie", "D2H_pinned", sz / (ms * 1e6), "GB/s");
    cudaFreeHost(hp);
    // pageable
    void* hp2 = malloc(64LL << 20);
    memset(hp2, 1, 64LL << 20);
    ms = time_ms([&] { CK(cudaMemcpy(dd, hp2, 64LL << 20, cudaMemcpyHostToDevice)); }, 2, 5);
    csv("pcie", "H2D_pageable", (64LL << 20) / (ms * 1e6), "GB/s");
    free(hp2);

    // PCIe link info
    int gen = -1, width = -1;
    cudaDeviceGetAttribute(&gen, cudaDevAttrPciDeviceId, 0);  // placeholder attr check
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("# pciBusID=%d pciDeviceID=%d pciDomainID=%d\n", p.pciBusID, p.pciDeviceID, p.pciDomainID);

    if (ndev < 2) {
        csv("p2p", "skipped", 0, "-", "only 1 device visible");
        cudaFree(dd);
        return 0;
    }

    // ---------- P2P ----------
    int can01 = 0, can10 = 0;
    cudaDeviceCanAccessPeer(&can01, 0, 1);
    cudaDeviceCanAccessPeer(&can10, 1, 0);
    csv("p2p", "can_access_0_to_1", can01, "bool");
    csv("p2p", "can_access_1_to_0", can10, "bool");
    if (can01 && can10) {
        void *d0, *d1;
        CK(cudaSetDevice(0)); CK(cudaMalloc(&d0, sz));
        CK(cudaSetDevice(1)); CK(cudaMalloc(&d1, sz));
        CK(cudaSetDevice(0));
        cudaError_t e1 = cudaDeviceEnablePeerAccess(1, 0);
        CK(cudaSetDevice(1));
        cudaError_t e2 = cudaDeviceEnablePeerAccess(0, 0);
        csv("p2p", "enable_peer", (e1 == cudaSuccess && e2 == cudaSuccess) ? 1 : 0, "bool",
            e1 != cudaSuccess ? cudaGetErrorString(e1) : (e2 != cudaSuccess ? cudaGetErrorString(e2) : ""));
        cudaGetLastError();  // clear sticky
        if (e1 == cudaSuccess && e2 == cudaSuccess) {
            ms = time_ms([&] { CK(cudaMemcpyPeer(d1, 1, d0, 0, sz)); }, 3, 10);
            csv("p2p", "bw_0_to_1", sz / (ms * 1e6), "GB/s");
            ms = time_ms([&] { CK(cudaMemcpyPeer(d0, 0, d1, 1, sz)); }, 3, 10);
            csv("p2p", "bw_1_to_0", sz / (ms * 1e6), "GB/s");
            // small transfer latency
            ms = time_ms([&] { CK(cudaMemcpyPeer(d1, 1, d0, 0, 4096)); }, 5, 100);
            csv("p2p", "latency_4KB", ms * 1000.0, "us");
        }
    }
    return 0;
}
