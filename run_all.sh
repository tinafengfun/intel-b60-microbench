#!/bin/bash
# Run all microbenchmarks with unitrace profiling
# Usage: ./run_all.sh [test_name]
#
# Available tests:
#   fp32_latency    — FP32 dependent chain latency vs instruction count
#   sg_scheduling   — Sub-group count sweep (TLP analysis)
#   dpas_throughput — DPAS (joint_matrix BF16) throughput sweep
#   mem_latency     — Pointer chase memory hierarchy
#   mem_bandwidth   — Sustained memory bandwidth
#   all             — Run all tests (default)

set -e
source /opt/intel/oneapi/setvars.sh 2>/dev/null

BUILD_DIR=/home/intel/tianfeng/gemm/microbench
UNITRACE=/home/intel/tianfeng/gemm/pti-gpu/tools/unitrace/build/unitrace
RESULTS_DIR=$BUILD_DIR/results
mkdir -p $RESULTS_DIR

# Compile a single test
compile() {
    local src=$1
    local name=$(basename $src .cpp)
    echo "Compiling $name..."
    icpx -fsycl -fsycl-targets=intel_gpu_bmg_g21 -O3 -std=c++17 \
         -o $BUILD_DIR/$name $src 2>&1 | grep -v "^:: " | grep -v "oneAPI" | grep -v "^$" || true
}

# Run with unitrace (ComputeBasic metrics)
run_with_metrics() {
    local name=$1
    shift
    local extra_args="$@"

    echo "Running $name with unitrace ComputeBasic..."
    $UNITRACE --metric-query --group ComputeBasic --device-timing \
        --output $RESULTS_DIR/${name}_compute.txt \
        $BUILD_DIR/$name $extra_args 2>&1 | tail -3 || true

    echo "Running $name with unitrace VectorEngineStalls..."
    $UNITRACE --metric-query --group VectorEngineStalls --device-timing \
        --output $RESULTS_DIR/${name}_stalls.txt \
        $BUILD_DIR/$name $extra_args 2>&1 | tail -3 || true
}

# Run without unitrace (plain execution for timing)
run_plain() {
    local name=$1
    shift
    echo "Running $name..."
    $BUILD_DIR/$name "$@" 2>&1
}

case "${1:-all}" in
    fp32_latency)
        compile $BUILD_DIR/bench_fp32_latency.cpp
        run_plain fp32_latency
        run_with_metrics fp32_latency
        ;;
    sg_scheduling)
        compile $BUILD_DIR/bench_sg_scheduling.cpp
        run_plain sg_scheduling
        run_with_metrics sg_scheduling
        ;;
    dpas_throughput)
        compile $BUILD_DIR/bench_dpas_throughput.cpp
        run_plain dpas_throughput
        run_with_metrics dpas_throughput
        ;;
    mem_latency)
        compile $BUILD_DIR/bench_mem_latency.cpp
        run_plain mem_latency
        run_with_metrics mem_latency
        ;;
    mem_bandwidth)
        compile $BUILD_DIR/bench_mem_bandwidth.cpp
        run_plain mem_bandwidth
        run_with_metrics mem_bandwidth
        ;;
    all)
        echo "=== Building all benchmarks ==="
        for f in bench_*.cpp; do
            compile $BUILD_DIR/$f
        done
        echo ""
        echo "=== Running all benchmarks ==="
        for name in fp32_latency sg_scheduling dpas_throughput mem_latency mem_bandwidth; do
            echo ""
            echo "=========================================="
            echo "Benchmark: $name"
            echo "=========================================="
            run_plain $name
        done
        echo ""
        echo "=== Running unitrace profiling ==="
        for name in fp32_latency dpas_throughput mem_latency mem_bandwidth; do
            run_with_metrics $name
        done
        ;;
    *)
        echo "Unknown test: $1"
        echo "Available: fp32_latency, sg_scheduling, dpas_throughput, mem_latency, mem_bandwidth, all"
        exit 1
        ;;
esac

echo ""
echo "Results saved to $RESULTS_DIR/"
