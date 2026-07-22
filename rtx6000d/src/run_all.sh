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
  if [ -x /work/bin/tcgen05_check ]; then
    /work/bin/tcgen05_check > $logdir/tcgen05_run.log 2>&1
    cat $logdir/tcgen05_run.log
  fi
  export CUDA_VISIBLE_DEVICES=0,1
  runone pcie_p2p /work/bin/pcie_p2p
  nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv,noheader >> $logdir/smi_snapshot.txt 2>&1 || true
  echo "===== RUN DONE ====="
}

case "${1:-all}" in
  build) build ;;
  run) run ;;
  all) build; run ;;
esac
