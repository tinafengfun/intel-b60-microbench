---
name: nv-gpu-microbench
description: 在新的 NVIDIA GPU 节点上复现 Blackwell(sm_120)微基准套件并生成跨卡对比报告。用于用户给出 GPU 节点(IP/凭证/卡号)要求"跑 microbenchmark / 复现测试 / 对比报告"的场景。覆盖 ALU/调度/内存层级/tensor core mma.sync/量化反量化/互连/框架级低精度 GEMM。
---

# NVIDIA GPU 微基准复现与对比流程

套件位置:`/home/tina/DEV/microbench/src/`(12 个 .cu + bench_common.h + run_all.sh + 4 个 python)。
已有基准数据:RTX 6000D(`results/*.csv`)、RTX PRO 5000(`results/pro5000/*.csv`)、
报告在 `report/`。Intel B60 对标见 https://github.com/tinafengfun/intel-b60-microbench 。

## 1. 节点侦察(必做)

```bash
sshpass -p '<pwd>' ssh -o StrictHostKeyChecking=no <user>@<ip> \
  'nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv,noheader; which docker nvcc; docker images | head'
```

- 确认 GPU 型号、SM 数、compute capability(sm_120 = Blackwell 客户端,sm_100 = B200)。
- 确认空闲 GPU 卡号,**只用用户指定的卡**,先看 `nvidia-smi` 是否有他人任务。
- 确认磁盘余量(镜像可能数 GB)。

## 2. 编译环境

- **必须 CUDA 13+ 容器**;宿主 nvcc 经常是旧版(CUDA 12 不认识 sm_120)。
- 首选 `nvidia/cuda:13.0.1-devel-ubuntu22.04`;没有就利用节点上已有镜像(NGC PyTorch/vLLM 镜像通常自带 `/usr/local/cuda/bin/nvcc`,用 `--entrypoint bash` 进入)。
- 编译 flag 必须显式 `-gencode arch=compute_120a,code=sm_120a`;`-arch=sm_120a` 在 CUDA 13 会退化成 sm_120,导致 `kind::f8f6f4` 被 ptxas 拒绝。
- 无外网节点:docker 加 `--network=none`;镜像无法拉取时用节点已有镜像。
- 某些节点 docker 网桥损坏,也必须 `--network=none`。

## 3. 同步与运行

```bash
sshpass -p '<pwd>' scp -r /home/tina/DEV/microbench/src <user>@<ip>:~/microbench/
sshpass -p '<pwd>' ssh <user>@<ip> 'docker run -d --name mb --gpus "\"device=0,1\"" --entrypoint bash \
  -v ~/microbench:/work <image> -c "bash /work/src/run_all.sh all > /work/results_run.log 2>&1"'
```

`run_all.sh all` 覆盖:device_props / alu_bench / sched_bench / sched_extra_bench / mma_ptx /
mma_probe(+SASS dump)/ tensor_cublas / quant / memory_bench / pcie_p2p / p2p_matrix /
tcgen05 编译探测 / torch 框架级(fp8 `_scaled_mm`、nvfp4 torch 原生、vLLM CUTLASS——按镜像可用性自动跳过)。
后台运行,每 3–5 分钟查一次 `results/`;全集约 10–15 分钟。

## 4. 结果回收与校验

- 结果 scp 回 `results/<gpu_name>/`;逐项与已有卡对比:
  - **应相同**:所有延迟(cycles)、每 SM 吞吐(ops/clk/SM)、barrier、shared 34cyc、mma 依赖 29.7cyc。
  - **按 SM×时钟缩放**:FP32/FP64/HFMA2 绝对 TFLOPS、L1/DRAM 带宽。
  - **重点看 tensor**:fp16 mma.sync 发射速率(mma/clk/SM)——满血 ≈0.249,6000D 合规版 ≈0.097(39%)。
- 可疑结果先怀疑方法学(见第 6 节),再下硬件结论。

## 5. 报告生成

对比表一律**同时给绝对值和每 SM·clk 归一值**;框架级数据注明软件栈版本(不同 CUDA/cuBLAS
版本差异可能很大,如 13.2 cuBLASLt fp16 崩溃、tf32 异常翻倍)。报告放 `report/<GPU_A>_vs_<GPU_B>_report.md`,
数据放 `results/<gpu>/`,源码改动同步回 `src/`。默认同步 /winshare 和 GitHub(流程同前,token 用后清除)。

## 6. 方法学陷阱(复测必读,详见 6000D 报告 §9)

1. 时钟测量必须单波次(≤1536 threads/SM),否则得 1.16GHz 假时钟。
2. 操作数必须运行时来源(`__constant__`/参数)+ `asm volatile` 屏障,防编译器闭式折叠。
3. mma 内核不能带运行时 switch,用模板特化。
4. shared 带宽测试下标必须依赖循环变量。
5. `__syncthreads()` 不能放在 `threadIdx.x==0` 等分歧分支里(死锁挂死 GPU)。
6. torch 侧:`torch.rand/randn` 不支持 uint8/float8/float4,先在 float 生成再 `.to()`。
7. cuBLASLt 探测 FP8/FP4 要遍历多种输出类型才能下"不支持"结论;版本升级可能引入新 bug(崩溃要隔离重跑,避免污染后续 case)。
8. python 入口脚本要 `if __name__ == "__main__":`(vLLM spawn 会重新 import)。
