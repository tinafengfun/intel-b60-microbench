// SDPA (attention core) on RTX PRO 5000: naive serial vs pipelined multi-stream.
// Same shapes/logic as src/bench_sdpa.cpp (B70/SYCL): H=32 heads, d=128, S sweep.
//   gemm1: scores[S,S] = Q[S,d] @ K[S,d]^T   (cuBLASLt bf16 -> fp32, tensor core)
//   softmax: fp32 rows -> bf16 P (separate buffer)
//   gemm2: out[S,d] = P[S,S] @ V[S,d]        (cuBLASLt bf16 -> fp32)
// Modes: serial (1 stream) | pipe (3 streams) | gemmonly
// Timing: cudaEvent GPU timestamps (host wall only for reference).
// Run: ./bench_sdpa [S=4096] [H=32] [mode=serial|pipe|gemmonly]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cuda_bf16.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); exit(1);} } while (0)
#define CKL(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
  fprintf(stderr, "cuBLAS error %d at line %d\n", (int)s_, __LINE__); exit(1);} } while (0)

using bf16 = __nv_bfloat16;

// one block per row: 3-pass softmax fp32 -> bf16
__global__ void softmax_rows(const float *s, bf16 *p, long long S, long long head_stride) {
  long long row = blockIdx.x;
  long long base = (row / S) * head_stride + (row % S) * S;
  const float *sr = s + base;
  bf16 *pr = p + base;  // same [H,S,S] flat layout, element-wise
  float m = -1e30f;
  for (long long j = threadIdx.x; j < S; j += blockDim.x) m = fmaxf(m, sr[j]);
  __shared__ float red[1024];
  red[threadIdx.x] = m;
  __syncthreads();
  for (int o = blockDim.x / 2; o > 0; o >>= 1) {
    if (threadIdx.x < o) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + o]);
    __syncthreads();
  }
  m = red[0];
  float sm = 0;
  for (long long j = threadIdx.x; j < S; j += blockDim.x) sm += __expf(sr[j] - m);
  __syncthreads();
  red[threadIdx.x] = sm;
  __syncthreads();
  for (int o = blockDim.x / 2; o > 0; o >>= 1) {
    if (threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o];
    __syncthreads();
  }
  float inv = 1.0f / red[0];
  for (long long j = threadIdx.x; j < S; j += blockDim.x)
    pr[j] = __float2bfloat16(__expf(sr[j] - m) * inv);
}

int main(int argc, char **argv) {
  int S = argc > 1 ? atoi(argv[1]) : 4096;
  int H = argc > 2 ? atoi(argv[2]) : 32;
  int d = 128;
  const char *mode = argc > 3 ? argv[3] : "serial";

  CK(cudaSetDevice(0));
  cublasLtHandle_t lt;
  CKL(cublasLtCreate(&lt));

  size_t qkv = (size_t)H * S * d, sc = (size_t)H * S * S;
  bf16 *Q, *K, *V, *P;
  float *O, *Sc;
  CK(cudaMalloc(&Q, qkv * 2)); CK(cudaMalloc(&K, qkv * 2)); CK(cudaMalloc(&V, qkv * 2));
  CK(cudaMalloc(&O, qkv * 4)); CK(cudaMalloc(&Sc, sc * 4)); CK(cudaMalloc(&P, sc * 2));
  CK(cudaMemset(Q, 0x3C, qkv * 2)); CK(cudaMemset(K, 0x3C, qkv * 2)); CK(cudaMemset(V, 0x3C, qkv * 2));
  printf("S=%d H=%d d=%d mode=%s  scores %.1f GB\n", S, H, d, mode, sc * 4.0 / 1e9);

  // cublasLt descriptors: gemm1 (TN) and gemm2 (NN), row-major via col-major trick
  cublasLtMatmulDesc_t desc1, desc2;
  cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
  CKL(cublasLtMatmulDescCreate(&desc1, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  CKL(cublasLtMatmulDescSetAttribute(desc1, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)));
  CKL(cublasLtMatmulDescCreate(&desc2, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  // gemm1: col-major m=S,n=S,k=d; A<-K [d,S] ld=d op=T ; B<-Q [d,S] ld=d op=N ; D<-Sc [S,S] ld=S
  cublasLtMatrixLayout_t A1, B1, D1;
  CKL(cublasLtMatrixLayoutCreate(&A1, CUDA_R_16BF, d, S, d));
  CKL(cublasLtMatrixLayoutCreate(&B1, CUDA_R_16BF, d, S, d));
  CKL(cublasLtMatrixLayoutCreate(&D1, CUDA_R_32F, S, S, S));
  // gemm2: row-major O(S,d)=P(S,S)@V(S,d) -> col-major m=d,n=S,k=S; A<-V [d,S] ld=d ; B<-P [S,S] ld=S ; D<-O [d,S] ld=d
  cublasLtMatrixLayout_t A2, B2, D2;
  CKL(cublasLtMatrixLayoutCreate(&A2, CUDA_R_16BF, d, S, d));
  CKL(cublasLtMatrixLayoutCreate(&B2, CUDA_R_16BF, S, S, S));
  CKL(cublasLtMatrixLayoutCreate(&D2, CUDA_R_32F, d, S, d));

  cublasLtMatmulPreference_t pref;
  CKL(cublasLtMatmulPreferenceCreate(&pref));
  size_t wsz = 64 << 20;
  void *ws;
  CK(cudaMalloc(&ws, wsz));
  CKL(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsz, sizeof(wsz)));

  cublasLtMatmulHeuristicResult_t heur1, heur2;
  int nres = 0;
  CKL(cublasLtMatmulAlgoGetHeuristic(lt, desc1, A1, B1, D1, D1, pref, 1, &heur1, &nres));
  if (nres < 1) { fprintf(stderr, "no gemm1 algo\n"); return 1; }
  CKL(cublasLtMatmulAlgoGetHeuristic(lt, desc2, A2, B2, D2, D2, pref, 1, &heur2, &nres));
  if (nres < 1) { fprintf(stderr, "no gemm2 algo\n"); return 1; }

  float alpha = 1.0f, beta = 0.0f;
  cudaStream_t s0, sv, s2;
  CK(cudaStreamCreate(&s0)); CK(cudaStreamCreate(&sv)); CK(cudaStreamCreate(&s2));

  auto gemm1 = [&](cudaStream_t st, int h) {
    CKL(cublasLtMatmul(lt, desc1, &alpha, K + (size_t)h * S * d, A1, Q + (size_t)h * S * d, B1,
                       &beta, Sc + (size_t)h * S * S, D1, Sc + (size_t)h * S * S, D1,
                       &heur1.algo, ws, wsz, st));
  };
  auto gemm2 = [&](cudaStream_t st, int h) {
    CKL(cublasLtMatmul(lt, desc2, &alpha, V + (size_t)h * S * d, A2, P + (size_t)h * S * S, B2,
                       &beta, O + (size_t)h * S * d, D2, O + (size_t)h * S * d, D2,
                       &heur2.algo, ws, wsz, st));
  };
  auto softmax = [&](cudaStream_t st, int h) {
    softmax_rows<<<S, 1024, 0, st>>>(Sc + (size_t)h * S * S, P + (size_t)h * S * S, S, (long long)S * S);
  };

  std::vector<cudaEvent_t> e1(H), e2(H), e3(H), evH(H);
  for (auto e : {&e1, &e2, &e3, &evH})
    for (int h = 0; h < H; h++) CK(cudaEventCreate(&(*e)[h]));
  cudaEvent_t evBeg, evEnd;
  CK(cudaEventCreate(&evBeg)); CK(cudaEventCreate(&evEnd));

  bool is_gemmonly = !strcmp(mode, "gemmonly");
  double best_span = 1e30, best_g1 = 0, best_sm = 0, best_g2 = 0, best_wall = 0;
  for (int r = 0; r < 5; r++) {
    cudaDeviceSynchronize();
    double t0 = (double)clock() / CLOCKS_PER_SEC;
    CK(cudaEventRecord(evBeg, s0));
    if (!strcmp(mode, "serial")) {
      for (int h = 0; h < H; h++) {
        gemm1(s0, h); CK(cudaEventRecord(e1[h], s0));
        softmax(s0, h); CK(cudaEventRecord(e2[h], s0));
        gemm2(s0, h); CK(cudaEventRecord(e3[h], s0));
      }
    } else if (!strcmp(mode, "pipe")) {
      for (int h = 0; h < H; h++) { gemm1(s0, h); CK(cudaEventRecord(e1[h], s0)); }
      for (int h = 0; h < H; h++) {
        CK(cudaStreamWaitEvent(sv, e1[h], 0));
        softmax(sv, h); CK(cudaEventRecord(e2[h], sv));
      }
      for (int h = 0; h < H; h++) {
        CK(cudaStreamWaitEvent(s2, e2[h], 0));
        gemm2(s2, h); CK(cudaEventRecord(e3[h], s2));
      }
      CK(cudaStreamWaitEvent(s0, e3[H - 1], 0));
    } else {  // gemmonly
      for (int h = 0; h < H; h++) { gemm1(s0, h); CK(cudaEventRecord(e1[h], s0)); }
      for (int h = 0; h < H; h++) { gemm2(s0, h); CK(cudaEventRecord(e3[h], s0)); }
    }
    CK(cudaEventRecord(evEnd, s0));
    CK(cudaEventSynchronize(evEnd));
    double wall = (double)clock() / CLOCKS_PER_SEC - t0;

    float ms;
    CK(cudaEventElapsedTime(&ms, evBeg, evEnd));
    double span = ms / 1e3;
    double sm = 0, g2 = 0;
    for (int h = 0; h < H; h++) {
      float t;
      if (!is_gemmonly) {
        CK(cudaEventElapsedTime(&t, e1[h], e2[h])); sm += t / 1e3;
        CK(cudaEventElapsedTime(&t, e2[h], e3[h])); g2 += t / 1e3;
      }
    }
    // gemm busy = span - softmax (exact for serial/gemmonly; approx for pipe)
    double g1 = span - sm - g2;
    if (span < best_span) {
      best_span = span; best_g1 = g1; best_sm = sm; best_g2 = g2; best_wall = wall;
    }
  }

  double gemm_flops = 2.0 * H * S * S * d * 2;
  double sm_ops = 5.0 * H * S * S;
  printf("TOTAL %s: gpu-span %.2f ms (host wall %.2f ms)\n", mode, best_span * 1e3, best_wall * 1e3);
  printf("  stages: gemm %.2f ms, softmax %.2f ms\n", (best_g1 + best_g2) * 1e3, best_sm * 1e3);
  printf("  gemm throughput %.1f TF, softmax %.1f Gops/s\n",
         gemm_flops / (best_g1 + best_g2) / 1e12, sm_ops / (best_sm > 0 ? best_sm : 1e-9) / 1e9);
  if (!is_gemmonly)
    printf("  vector(softmax) share of gpu-span: %.1f%%\n", 100.0 * best_sm / best_span);
  return 0;
}
