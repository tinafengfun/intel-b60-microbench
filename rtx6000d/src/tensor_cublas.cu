// cuBLASLt GEMM throughput across precisions (the achievable tensor-core path)
#include "bench_common.h"
#include <cublasLt.h>
#include <vector>

struct GemmCfg {
    const char* name;
    cudaDataType at, bt, ct;
    cublasComputeType_t comp;
    cudaDataType scale;
    std::vector<cudaDataType> ct_fallback;  // alternative C types if heuristic fails
};

static bool try_layout(cublasLtHandle_t h, const GemmCfg& c, int M, int N, int K, cudaDataType ct,
                       const char** fail_note) {
    cublasLtMatmulDesc_t op;
    cublasLtMatmulDescCreate(&op, c.comp, c.scale);
    cublasOperation_t T = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &T, sizeof(T));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &T, sizeof(T));

    cublasLtMatrixLayout_t Ad, Bd, Cd;
    if (cublasLtMatrixLayoutCreate(&Ad, c.at, M, K, K) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutCreate(&Bd, c.bt, K, N, N) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutCreate(&Cd, ct, M, N, N) != CUBLAS_STATUS_SUCCESS) {
        *fail_note = "layout create failed";
        return false;
    }
    size_t asz = (size_t)M * K, bsz = (size_t)K * N, csz = (size_t)M * N;
    int esz = (c.at == CUDA_R_32F || c.at == CUDA_R_32I) ? 4 : 1;
    int esz_c = (ct == CUDA_R_32F || ct == CUDA_R_32I) ? 4 : (ct == CUDA_R_16F || ct == CUDA_R_16BF) ? 2 : 1;
    void *A, *B, *C;
    if (cudaMalloc(&A, asz * esz) || cudaMalloc(&B, bsz * esz) || cudaMalloc(&C, csz * esz_c)) {
        *fail_note = "alloc failed";
        return false;
    }
    cudaMemset(A, 0x11, asz * esz); cudaMemset(B, 0x11, bsz * esz); cudaMemset(C, 0, csz * esz_c);

    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    size_t wssz = 64 << 20;
    void* ws; cudaMalloc(&ws, wssz);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wssz, sizeof(wssz));
    cublasLtMatmulHeuristicResult_t res;
    int nres = 0;
    cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(h, op, Ad, Bd, Cd, Cd, pref, 1, &res, &nres);
    if (st != CUBLAS_STATUS_SUCCESS || nres < 1) {
        *fail_note = "no algo / not supported on sm_120";
        cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(ws);
        return false;
    }
    float alpha = 1.0f, beta = 0.0f;
    int alpha_i = 1, beta_i = 0;
    void *alphap = &alpha, *betap = &beta;
    if (c.scale == CUDA_R_32I) { alphap = &alpha_i; betap = &beta_i; }
    auto run = [&] {
        cublasLtMatmul(h, op, alphap, A, Ad, B, Bd, betap, C, Cd, C, Cd, &res.algo, ws, wssz, 0);
    };
    run();
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        *fail_note = cudaGetErrorString(err);
        cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(ws);
        return false;
    }
    double ms = time_ms(run, 5, 20);
    double tflops = 2.0 * M * N * (double)K / (ms * 1e-3) / 1e12;
    static const char* tn[] = {"R_32F?", "R_16F", "R_32F", "R_16BF", "R_8I", "R_8F_E4M3", "R_8F_E5M2", "R_4F_E2M1", "R_32I?"};
    char note[128];
    snprintf(note, 128, "M=N=K=8192 cublasLt Ctype=%d", (int)ct);
    csv("cublas_gemm", c.name, tflops, "TFLOPS", note);
    cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(ws);
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatmulDescDestroy(op);
    cublasLtMatrixLayoutDestroy(Ad); cublasLtMatrixLayoutDestroy(Bd); cublasLtMatrixLayoutDestroy(Cd);
    return true;
}

static void try_gemm(cublasLtHandle_t h, const GemmCfg& c, int M, int N, int K) {
    const char* note = "";
    if (try_layout(h, c, M, N, K, c.ct, &note)) return;
    for (cudaDataType ct : c.ct_fallback)
        if (try_layout(h, c, M, N, K, ct, &note)) return;
    csv("cublas_gemm", c.name, 0, "TFLOPS", note);
}

int main() {
    cublasLtHandle_t h;
    cublasLtCreate(&h);
    int M = 8192, N = 8192, K = 8192;
    GemmCfg cfgs[] = {
        {"fp32",       CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F, CUDA_R_32F, {}},
        {"tf32",       CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F, {}},
        {"fp16_acc32", CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_32F, CUDA_R_32F, {}},
        {"fp16_acc16", CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_16F, CUDA_R_16F, {}},
        {"bf16_acc32", CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUBLAS_COMPUTE_32F, CUDA_R_32F, {}},
        {"int8_imma",  CUDA_R_8I,  CUDA_R_8I,  CUDA_R_32I, CUBLAS_COMPUTE_32I, CUDA_R_32I, {}},
        {"fp8_e4m3",   CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, CUDA_R_16F, CUBLAS_COMPUTE_32F, CUDA_R_32F,
            {CUDA_R_16BF, CUDA_R_32F, CUDA_R_8F_E4M3}},
        {"fp8_e5m2",   CUDA_R_8F_E5M2, CUDA_R_8F_E5M2, CUDA_R_16F, CUBLAS_COMPUTE_32F, CUDA_R_32F,
            {CUDA_R_16BF, CUDA_R_32F, CUDA_R_8F_E5M2}},
        {"fp4_e2m1",   CUDA_R_4F_E2M1, CUDA_R_4F_E2M1, CUDA_R_16F, CUBLAS_COMPUTE_32F, CUDA_R_32F,
            {CUDA_R_16BF, CUDA_R_32F, CUDA_R_8F_E4M3}},
        // mixed fp8 e4m3 x e5m2 (common inference combo)
        {"fp8_e4m3xe5m2", CUDA_R_8F_E4M3, CUDA_R_8F_E5M2, CUDA_R_16F, CUBLAS_COMPUTE_32F, CUDA_R_32F,
            {CUDA_R_16BF, CUDA_R_32F}},
    };
    for (auto& c : cfgs) try_gemm(h, c, M, N, K);
    cublasLtDestroy(h);
    return 0;
}
