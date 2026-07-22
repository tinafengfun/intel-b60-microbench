#!/bin/bash
# Build & run all microbenchmarks inside CUDA 13.0 devel container.
# Usage: run_all.sh [build|run|all]   (default all)
set -u
cd /work/src
mkdir -p /work/bin /work/results
NVCC="nvcc -O3"
ARCH_LITE="-gencode arch=compute_120a,code=sm_120a"

build() {
  echo "===== BUILD ====="
  nvcc -O3 $ARCH_LITE device_props.cu -o /work/bin/device_props 2>&1 | tail -2
  nvcc -O3 $ARCH_LITE alu_bench.cu -o /work/bin/alu_bench 2>&1 | tail -2
  nvcc -O3 $ARCH_LITE sched_bench.cu -o /work/bin/sched_bench 2>&1 | tail -2
  nvcc -O3 $ARCH_LITE mma_ptx.cu -o /work/bin/mma_ptx 2>&1 | tail -5
  nvcc -O3 $ARCH_LITE tensor_cublas.cu -o /work/bin/tensor_cublas -lcublasLt -lcublas 2>&1 | tail -2
  nvcc -O3 $ARCH_LITE quant.cu -o /work/bin/quant 2>&1 | tail -2
  nvcc -O3 $ARCH_LITE memory_bench.cu -o /work/bin/memory_bench 2>&1 | tail -2
  nvcc -O3 $ARCH_LITE pcie_p2p.cu -o /work/bin/pcie_p2p 2>&1 | tail -2
  nvcc -O3 $ARCH_LITE mma_probe.cu -o /work/bin/mma_probe 2>&1 | tail -2
  nvcc -O3 $ARCH_LITE sched_extra_bench.cu -o /work/bin/sched_extra_bench 2>&1 | tail -2
  nvcc -O3 $ARCH_LITE p2p_matrix.cu -o /work/bin/p2p_matrix 2>&1 | tail -2
  # tcgen05 support probe
  cat > /work/bin/tcgen05_check.cu <<'EOF'
#include <cstdio>
__global__ void k() {
    __shared__ unsigned s[4];
    unsigned sp = (unsigned)__cvta_generic_to_shared(s);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;" :: "r"(sp));
}
int main() {
    k<<<1, 32>>>();
    cudaError_t e = cudaDeviceSynchronize();
    printf("tcgen05 runtime: %s\n", cudaGetErrorString(e));
    return 0;
}
EOF
  if nvcc -O3 $ARCH_LITE /work/bin/tcgen05_check.cu -o /work/bin/tcgen05_check 2>/work/results/tcgen05_compile_err.log; then
    echo "CSV,tcgen05,compile,1,bool,compiles-on-sm_120a" > /work/results/tcgen05.csv
  else
    echo "CSV,tcgen05,compile,0,bool,unsupported-on-sm_120a" > /work/results/tcgen05.csv
    head -3 /work/results/tcgen05_compile_err.log
  fi
  echo "===== BUILD DONE ====="
  ls -la /work/bin/
}

run() {
  echo "===== RUN ====="
  local logdir=/work/results
  runone() {
    local name=$1; shift
    echo "--- $name ---"
    "$@" > $logdir/$name.log 2>&1
    grep '^CSV' $logdir/$name.log > $logdir/$name.csv
    tail -2 $logdir/$name.log
  }
  export CUDA_VISIBLE_DEVICES=0
  nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv,noheader > $logdir/smi_snapshot.txt 2>&1 || true
  runone device_props /work/bin/device_props
  runone alu_bench /work/bin/alu_bench
  runone sched_bench /work/bin/sched_bench
  runone mma_ptx /work/bin/mma_ptx
  runone tensor_cublas /work/bin/tensor_cublas
  runone quant /work/bin/quant
  runone memory_bench /work/bin/memory_bench
  runone mma_probe /work/bin/mma_probe
  runone sched_extra /work/bin/sched_extra_bench
  cuobjdump -sass /work/bin/mma_ptx > $logdir/mma_sass.txt 2>&1 || true
  grep -cE "HMMA|QMMA|OMMA" $logdir/mma_sass.txt || true
  if [ -x /work/bin/tcgen05_check ]; then
    /work/bin/tcgen05_check > $logdir/tcgen05_run.log 2>&1
    cat $logdir/tcgen05_run.log
  fi
  export CUDA_VISIBLE_DEVICES=0,1
  runone pcie_p2p /work/bin/pcie_p2p
  runone p2p_matrix /work/bin/p2p_matrix
  # framework-level GEMM tests (only if torch is available in this environment)
  if python3 -c "import torch" 2>/dev/null; then
    export CUDA_VISIBLE_DEVICES=0
    python3 /work/src/fp8_gemm_test.py > $logdir/fp8_gemm.log 2>&1 || true
    python3 /work/src/nvfp4_gemm_test.py > $logdir/nvfp4_gemm.log 2>&1 || true
    python3 /work/src/nvfp4_torch_test.py > $logdir/nvfp4_torch.log 2>&1 || true
    python3 /work/src/nvfp4_check.py > $logdir/nvfp4_check.log 2>&1 || true
    grep -h '^CSV' $logdir/fp8_gemm.log $logdir/nvfp4_gemm.log $logdir/nvfp4_torch.log $logdir/nvfp4_check.log > $logdir/lowprecision_gemm.csv 2>/dev/null || true
    tail -2 $logdir/nvfp4_torch.log
  fi
  nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv,noheader >> $logdir/smi_snapshot.txt 2>&1 || true
  echo "===== RUN DONE ====="
}

case "${1:-all}" in
  build) build ;;
  run) run ;;
  all) build; run ;;
esac
