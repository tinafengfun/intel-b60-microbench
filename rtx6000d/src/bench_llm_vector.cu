// LLM decode-representative vector kernels on NVIDIA (CUDA): same shapes as
// the B70 SYCL version for cross-vendor comparison, plus fp8->half dequant
// via HARDWARE cvt instruction vs software bit-twiddle.
// nvcc -O3 -arch=sm_120 bench_llm_vector.cu -o bench_llm_vector
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdio>
#include <cstdint>
#include <algorithm>

#define CK(x) do { cudaError_t e = (x); if (e) { printf("CUDA err %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1);} } while (0)

// ---------- triad bf16 ----------
__global__ void triad(const __nv_bfloat16 *a, const __nv_bfloat16 *b,
                      __nv_bfloat16 *o, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) o[i] = __float2bfloat16(__bfloat162float(a[i]) + 1.5f * __bfloat162float(b[i]));
}

// ---------- rmsnorm: one block per row, N=2048 ----------
__global__ void rmsnorm(const __nv_bfloat16 *x, const __nv_bfloat16 *w,
                        __nv_bfloat16 *y, int N) {
  int row = blockIdx.x;
  const __nv_bfloat16 *xr = x + (size_t)row * N;
  __nv_bfloat16 *yr = y + (size_t)row * N;
  float ss = 0;
  for (int j = threadIdx.x; j < N; j += blockDim.x) {
    float v = __bfloat162float(xr[j]); ss += v * v;
  }
  __shared__ float sm[32];
  for (int off = 16; off; off >>= 1) ss += __shfl_down_sync(~0u, ss, off);
  if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = ss;
  __syncthreads();
  if (threadIdx.x < 32) {
    ss = threadIdx.x < (blockDim.x + 31) / 32 ? sm[threadIdx.x] : 0;
    for (int off = 16; off; off >>= 1) ss += __shfl_down_sync(~0u, ss, off);
    if (threadIdx.x == 0) sm[0] = rsqrtf(ss / N + 1e-6f);
  }
  __syncthreads();
  float inv = sm[0];
  for (int j = threadIdx.x; j < N; j += blockDim.x)
    yr[j] = __float2bfloat16(__bfloat162float(xr[j]) * inv * __bfloat162float(w[j]));
}

// ---------- swiglu: silu(g)*u ----------
__global__ void swiglu(const __nv_bfloat16 *g, const __nv_bfloat16 *u,
                       __nv_bfloat16 *o, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) {
    float v = __bfloat162float(g[i]);
    float silu = v / (1.0f + expf(-v));
    o[i] = __float2bfloat16(silu * __bfloat162float(u[i]));
  }
}

// ---------- softmax: one block per row, fp32 ----------
__global__ void softmax_k(const float *x, float *y, int N) {
  int row = blockIdx.x;
  const float *xr = x + (size_t)row * N;
  float *yr = y + (size_t)row * N;
  float m = -1e30f;
  for (int j = threadIdx.x; j < N; j += blockDim.x) m = fmaxf(m, xr[j]);
  __shared__ float sm[32];
  for (int off = 16; off; off >>= 1) m = fmaxf(m, __shfl_down_sync(~0u, m, off));
  if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = m;
  __syncthreads();
  if (threadIdx.x < 32) {
    m = threadIdx.x < (blockDim.x + 31) / 32 ? sm[threadIdx.x] : -1e30f;
    for (int off = 16; off; off >>= 1) m = fmaxf(m, __shfl_down_sync(~0u, m, off));
    if (threadIdx.x == 0) sm[0] = m;
  }
  __syncthreads();
  m = sm[0];
  float s = 0;
  for (int j = threadIdx.x; j < N; j += blockDim.x) s += __expf(xr[j] - m);
  __shared__ float ss2[32];
  for (int off = 16; off; off >>= 1) s += __shfl_down_sync(~0u, s, off);
  if ((threadIdx.x & 31) == 0) ss2[threadIdx.x >> 5] = s;
  __syncthreads();
  if (threadIdx.x < 32) {
    s = threadIdx.x < (blockDim.x + 31) / 32 ? ss2[threadIdx.x] : 0;
    for (int off = 16; off; off >>= 1) s += __shfl_down_sync(~0u, s, off);
    if (threadIdx.x == 0) ss2[0] = 1.0f / s;
  }
  __syncthreads();
  float inv = ss2[0];
  for (int j = threadIdx.x; j < N; j += blockDim.x) yr[j] = __expf(xr[j] - m) * inv;
}

// ---------- fp8 e4m3 -> bf16: hardware cvt (2 elems/instr) ----------
__global__ void dequant_hw(const uint16_t *v, __nv_bfloat16 *o, size_t nw) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < nw) {
    __half2_raw h2 = __nv_cvt_fp8x2_to_halfraw2(v[i], __NV_E4M3);
    __half2 h = *reinterpret_cast<__half2 *>(&h2);
    o[i * 2]     = __float2bfloat16(__low2float(h));
    o[i * 2 + 1] = __float2bfloat16(__high2float(h));
  }
}

// ---------- fp8 e4m3 -> bf16: software shift path (Intel-style) ----------
__global__ void dequant_sw(const uint32_t *v, __nv_bfloat16 *o, size_t nw) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < nw) {
    uint32_t w = v[i];
#pragma unroll
    for (int k = 0; k < 4; k++) {
      uint16_t bits = uint16_t((w >> (8 * k)) & 0xff) << 8;
      __half h = *reinterpret_cast<__half *>(&bits);
      o[i * 4 + k] = __float2bfloat16(__half2float(h) * 256.0f);
    }
  }
}

template <typename F>
float bench(F fn, int reps = 20) {
  fn(); CK(cudaDeviceSynchronize());
  cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
  float best = 1e30f;
  for (int r = 0; r < reps; r++) {
    cudaEventRecord(t0); fn(); cudaEventRecord(t1);
    CK(cudaEventSynchronize(t1));
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    best = std::min(best, ms);
  }
  return best;
}

static void report(const char *name, float ms, double bytes, double flops) {
  printf("  %-30s %8.3f ms  %8.1f GB/s  %9.2f GOP/s\n",
         name, ms, bytes / (ms * 1e-3) / 1e9, flops / (ms * 1e-3) / 1e9);
}

int main() {
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  printf("device: %s  SMs=%d\n", p.name, p.multiProcessorCount);

  // triad
  {
    size_t E = 64ull << 20;
    __nv_bfloat16 *a, *b, *o;
    CK(cudaMalloc(&a, E * 2)); CK(cudaMalloc(&b, E * 2)); CK(cudaMalloc(&o, E * 2));
    CK(cudaMemset(a, 0x3c, E * 2)); CK(cudaMemset(b, 0x3c, E * 2));
    int th = 256; size_t bl = (E + th - 1) / th;
    float ms = bench([&] { triad<<<bl, th>>>(a, b, o, E); });
    report("triad bf16 (BW baseline)", ms, 3.0 * E * 2, 2.0 * E);
    cudaFree(a); cudaFree(b); cudaFree(o);
  }
  // rmsnorm
  {
    size_t R = 16384; int N = 2048;
    __nv_bfloat16 *x, *w, *y;
    CK(cudaMalloc(&x, R * N * 2)); CK(cudaMalloc(&w, N * 2)); CK(cudaMalloc(&y, R * N * 2));
    CK(cudaMemset(x, 0x2c, R * N * 2)); CK(cudaMemset(w, 0x3c, N * 2));
    float ms = bench([&] { rmsnorm<<<R, 256>>>(x, w, y, N); });
    report("rmsnorm bf16 N=2048", ms, double(R) * N * 2 * 2 + double(R) * N / 8, 5.0 * R * N);
    cudaFree(x); cudaFree(w); cudaFree(y);
  }
  // swiglu
  {
    size_t E = 64ull << 20;
    __nv_bfloat16 *g, *u, *o;
    CK(cudaMalloc(&g, E * 2)); CK(cudaMalloc(&u, E * 2)); CK(cudaMalloc(&o, E * 2));
    CK(cudaMemset(g, 0x3c, E * 2)); CK(cudaMemset(u, 0x3c, E * 2));
    int th = 256; size_t bl = (E + th - 1) / th;
    float ms = bench([&] { swiglu<<<bl, th>>>(g, u, o, E); });
    report("swiglu silu*mul bf16", ms, 3.0 * E * 2, 6.0 * E);
    cudaFree(g); cudaFree(u); cudaFree(o);
  }
  // softmax
  {
    size_t R = 256; int N = 151936;
    float *x, *y;
    CK(cudaMalloc(&x, R * N * 4)); CK(cudaMalloc(&y, R * N * 4));
    CK(cudaMemset(x, 0x01, R * N * 4));
    float ms = bench([&] { softmax_k<<<R, 1024>>>(x, y, N); });
    report("softmax fp32 N=151936", ms, double(R) * N * 4 * 4, 4.0 * R * N);
    cudaFree(x); cudaFree(y);
  }
  // dequant hw vs sw
  {
    size_t E = 128ull << 20;
    void *v; __nv_bfloat16 *o;
    CK(cudaMalloc(&v, E)); CK(cudaMalloc(&o, E * 2));
    CK(cudaMemset(v, 0x38, E));
    int th = 256;
    size_t bl2 = (E / 2 + th - 1) / th, bl4 = (E / 4 + th - 1) / th;
    float ms1 = bench([&] { dequant_hw<<<bl2, th>>>((uint16_t *)v, o, E / 2); });
    report("dequant fp8->bf16 HW cvt", ms1, double(E) * 3, 2.0 * E);
    float ms2 = bench([&] { dequant_sw<<<bl4, th>>>((uint32_t *)v, o, E / 4); });
    report("dequant fp8->bf16 SW shift", ms2, double(E) * 3, 2.0 * E);
    cudaFree(v); cudaFree(o);
  }
  return 0;
}
