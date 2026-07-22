# RTX 6000D (Blackwell GB202) vs Intel Arc Pro B60 (Xe2 BMG-G21) 横向对比

数据来源:NVIDIA 侧为本仓库 `rtx6000d/` 微基准实测(`report/RTX6000D_vs_B200_report.md`);
Intel 侧为本仓库根目录 `REPORT.md` 微基准实测。两套套件方法学同出一源(均参考 arXiv:2507.10789),
且 barrier / 启动开销 / tensor 停顿窗口交织 / shared 延迟四个维度已做交叉对标。
跨架构对比优先看 **wall-clock(ns)**,cycles 仅作参考(6000D @2.30GHz,B60 @2.40GHz)。

> 定位声明(与 B60 报告一致):两者同为工作站形态但价位/功耗不同档(B60 ~120W 级、24GB;
> 6000D 600W、84GB),本对比用于刻画架构特征,不用于宣判优劣。

## 1. 规格与实测峰值

| 维度 | RTX 6000D (GB202, sm_120) | Arc Pro B60 (BMG-G21, Xe2) | 6000D / B60 |
|---|---|---|---|
| 计算单元 | 156 SM(4 SMSP/SM) | 20 Xe core(8 EU/Xe core) | — |
| 线程模型 | 48 warps/SM(32 线程/warp) | 8 threads/EU(subgroup=16) | — |
| 持续时钟 | 2.30 GHz(满载无降频) | 2.40 GHz | 0.96× |
| FP32 | **83.0 TFLOPS** | ~12.3 TFLOPS(标称,报告未实测) | ~6.7× |
| FP16/BF16 矩阵 | **143 TFLOPS**(mma.sync 硬上限) | **97.66 TFLOPS**(XMX 原生峰值) | 1.5× |
| 低精度矩阵 | **651.6 TFLOPS**(nvfp4 CUTLASS) | INT8 DPAS 被 IGC 编译器 bug 阻塞;FP4 无路 | ≥6.7× |
| 矩阵单元利用率 | mma.sync 仅达 datasheet 推导值 ~36%(nvfp4 路径证明有 4.5× 隐藏余量) | 自定义 SYCL GEMM 达 XMX 峰值 **92%** | B60 更"实" |
| 显存 | 84GB GDDR7 448-bit | 24GB GDDR6 256-bit | 3.5× 容量 |
| 显存带宽(实测读) | **1344 GB/s**(标称 96.1%) | **538 GB/s**(标称 93%) | 2.5× |
| L2 | **112 MB**,5.41 TB/s | 18 MB | 6.2× 容量 |

## 2. 延迟对比(ns;cycles 按各自持续时钟换算)

| 层级 | 6000D | B60 | 说明 |
|---|---|---|---|
| 矩阵指令依赖延迟 | 29.7 cyc = **12.9 ns** | 33–34 cyc = **13.8–14.2 ns** | 几乎相同;6000D 全精度同延迟,B60 FP16=BF16 |
| shared / SLM | 34 cyc = **14.8 ns**(1–96KB 平坦) | 46 cyc = **19.2 ns**(原生 send.slm) | 6000D 独立 SRAM,平坦;B60 与 L1 统一 SRAM 但独立路径 |
| L1 | 44 cyc = **19.1 ns** | 71 cyc = **29.6 ns**(随容量升至 145 cyc) | 6000D 专用 L1 更快且平坦;B60 send.ugm 统一路径有 tag/队列开销 |
| L2 | 366 cyc = **159 ns** | 162–236 cyc = **68–98 ns** | **B60 显著更快**;但 6000D L2 有 112MB(6.2×),用延迟换容量 |
| DRAM | 759–877 cyc = **330–381 ns** | 247–261 cyc = **103–109 ns** | **B60 快 3.5×**:GDDR6 短链路 + 小容量 vs GDDR7 高速率长链路,设计取向相反 |

延迟形态总结:6000D = "低 L1/shared 延迟 + 大容量高延迟 L2/DRAM"(为吞大模型优化);
B60 = "高 L1 延迟 + 低 L2/DRAM 延迟"(为小工作集、图形类局部性优化)。
两者 shared 都快于各自 L1(6000D 34<44;B60 46<71),但 B60 的 SYCL `local_accessor`
会加 ~34 cyc 开销抹平优势,CUDA 无此问题。

## 3. 调度与运行时开销

| 维度 | 6000D | B60 | 说明 |
|---|---|---|---|
| barrier | 5–15 cyc(1→32 warps) | 2–11 cyc(OpControlBarrier) | 都"近似免费",B60 略低 |
| kernel 启动 | 2.05 µs(摊销)/ 2.82 µs(单次) | 3.7 µs 固定 + 40 ns/WG | 6000D 更低 |
| tensor 停顿窗口吸收 ALU | 32 个独立 FFMA 仅 +8 cyc(mma 依赖链) | 16 个 FFMA 完全免费(DPAS SBID 窗口) | **结论同构**:逐元素 dequant/scale 可与矩阵指令免费交织 |
| 矩阵单元流水线 | 全精度同延迟 29.7 cyc;16 warps/SM 或 ILP≥4 饱和 | 延迟 33 / 互逆吞吐 16.1 cyc ⇒ **2 级流水线**,ILP≥14 或纯 TLP 饱和 | B60 把流水线深度测得更完整;6000D 靠 warp 数饱和 |

## 4. 软件栈成熟度(双方各自的"掉链子"环节)

| | 6000D (CUDA 13) | B60 (oneAPI 2025.3) |
|---|---|---|
| 阻塞级问题 | cuBLASLt 对 sm_120 的 FP8/FP4 GEMM **返回 no algo** | IGC 对 INT8 cooperative matrix **直接 segfault** |
| 可用绕行 | torch `_scaled_mm` / vLLM CUTLASS(nvfp4 651.6 TFLOPS,正确性已验证) | 需改用 `OpSubgroupMatrixMultiplyAccumulateINTEL` 或 `__builtin_IB_*`(报告中未走通) |
| 隐形开销 | 无(shared 直达) | SYCL `local_accessor` +34 cyc/access;scalar load 只有向量化带宽的 35% |
| 框架路径 | vLLM 端到端已验证(Qwen3.6-27B-NVFP4,decode 47.3 tok/s = 带宽 roofline 88–93%) | oneDNN ~95 TFLOPS(估算);端到端 LLM 未测 |

共同教训:**裸指令峰值 ≠ 框架可用性能**,两家的官方库在新架构上都滞后于硬件;
区别在于 CUDA 生态的绕行路径(CUTLASS/torch/vLLM)当天可用且已到 roofline,
Intel 侧 INT8/低精度路径仍被编译器 bug 卡死。

## 5. 单卡 LLM 推理含义

| 场景 | 6000D | B60 | 比值 |
|---|---|---|---|
| 可装模型(nvfp4) | 27B 级(26GB 权重)轻松,70B nvfp4(~40GB)可装 | ≤13B 级(24GB 总量) | 容量 3.5× |
| decode bs1(带宽受限) | 1344 GB/s → 27B nvfp4 实测 **47.3 tok/s** | 538 GB/s → 同比例模型约 **2/5** | **2.5×** |
| prefill / 大 batch | 651.6 TFLOPS(nvfp4) | 97.66 TFLOPS(BF16;无 nvfp4 路径) | **≥6.7×** |

B60 的 DRAM 延迟低 3.5× 对 decode 无直接帮助(decode 是带宽受限而非延迟受限,
带宽差距 2.5× 才是决定项);其 24GB 容量是更硬的约束——27B nvfp4 已经装不下。

## 6. 结论

1. **算力/带宽/容量全面碾压但非同档产品**:6000D 对 B60 是 1.5×(BF16)~6.7×(nvfp4)算力、
   2.5× 带宽、3.5× 显存,同时功耗约 5×;单位功耗 BF16 算力 B60 反而更高(0.81 vs 0.24 TF/W,
   按各自报告口径),但 nvfp4 路径下 6000D 反超(1.09 TF/W)。
2. **延迟哲学相反**:B60 用容量换延迟(小 L2、低 DRAM 延迟),6000D 用延迟换容量
   (112MB L2、84GB GDDR7)——分别是图形/小模型与大模型推理的取向。
3. **架构洞察互相印证**:tensor 停顿窗口免费吸收 ALU、barrier 近似免费、shared 快于 L1,
   三条结论在两个完全不同的架构上独立成立,可视为当代 GPU 的普适规律。
4. **软件生态是真实差距**:硬件上 B60 的 XMX 利用率(92%)比 6000D 的 mma.sync(36%)健康得多;
   但 CUDA 生态让 6000D 的隐藏算力(nvfp4 651.6 TF)在 vLLM 里立即可用,
   B60 的 INT8 路径还被编译器 bug 堵着——**买得到算力和用得到算力是两回事**。
