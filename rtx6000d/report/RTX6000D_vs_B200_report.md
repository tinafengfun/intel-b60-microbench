# NVIDIA RTX 6000D (Blackwell GB202, sm_120) 微基准评测与 B200 (GB100, sm_100) 对比报告

**测试日期**:2026-07-21
**测试平台**:远程节点 172.16.120.54(6530PPT),GPU 0 = NVIDIA RTX 6000D(同机另有 5 张 6000D、2 张 RTX 5090 未使用)
**软件栈**:驱动 580.105.08,docker `nvidia/cuda:13.0.1-devel-ubuntu22.04`(nvcc 13.0.88,`-gencode arch=compute_120a,code=sm_120a`)
**参考文献**:
- 论文 1:[Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks (arXiv 2507.10789)](https://arxiv.org/abs/2507.10789) — 实测 GB203 (RTX 5080, 同为 sm_120) 对比 GH100
- 论文 2:[Microbenchmarking NVIDIA's Blackwell Architecture (arXiv 2512.02189)](https://arxiv.org/abs/2512.02189) — 实测 B200 (GB100, sm_100)
- 规格:[NVIDIA RTX PRO 6000 Blackwell Workstation Edition Datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/data-center/rtx-pro-6000-blackwell-workstation-edition/workstation-blackwell-rtx-pro-6000-workstation-edition-nvidia-us-3519208-web.pdf)

**方法学**:自研 CUDA microbenchmark 套件(9 个测试程序,源码在 `src/`,原始数据在 `results/*.csv`)。cudaEvent 计时,≥3 次预热、≥10 次重复取中位数;延迟类用 `clock64()` 依赖链测 cycles;持续时钟用 FFMA 负载自旋内核实测(2.30 GHz);吞吐换算用实测时钟。

---

## 0. 关键结论(TL;DR)

1. **RTX 6000D 是满血 GB202 的"合规阉割版"**:156 SM(标准版 188 SM 的 83%)、448-bit/84GB GDDR7(标准版 512-bit/96GB 的 87.5%)。SM 微架构与论文 1 的 GB203 完全一致(同 sm_120),所有微架构特征可复现。
2. **sm_120 没有 tcgen05/TMEM**:ptxas 直接拒绝 `tcgen05.alloc`("not supported on .target 'sm_120a'")。Tensor Core 只能走 `mma.sync`(Ampere 风格 warp 级指令),这是与 B200 (sm_100) 最本质的架构差异。B200 的 tcgen05 单指令延迟 ~11 cycles 且支持 TMEM 累加器驻留;6000D 的 mma.sync 累加器依赖链延迟为 **29.7 cycles**。
3. **低精度 GEMM 的可用路径(实测三条)**:`mma.sync.kind::f8f6f4` PTX 支持 FP8(287 TFLOPS)/FP4(376 TFLOPS);CUDA 13.0 cuBLASLt 对 sm_120 的 FP8/FP4 GEMM **返回 no algo**;但 **PyTorch `torch._scaled_mm`(CUTLASS 内核)FP8 = 264 TFLOPS**、vLLM cutlass FP8 = 262、**vLLM CUTLASS nvfp4 GEMM = 651.6 TFLOPS**(正确性已校验,与反量化参考相对差 0.6%)。结论:硬件完整可用,cuBLASLt 是唯一掉链子的环节,推理框架(vLLM)不受影响。
4. **FP4 真实梯度与 B200 相当**:nvfp4 block-scaled GEMM(651.6)= FP16 GEMM(152.8)的 **4.26×**,与 B200 的 4× 梯度一致。注意分层:裸 `mma.sync kind::f8f6f4` 的 FP4 只有 2.63×(376,QMMA 回落),**必须走 block-scaled(nvfp4, e4m3 scale)路径才触发完整 5 代 tensor 数据通路**——与论文 1"仅 ue8m0 scaling 时出现 OMMA"的观察互为印证。
5. **CUDA core 与内存子系统是强项**:FP32 83 TFLOPS(理论 97 的 86~90%)、DRAM 读带宽 1344 GB/s(理论 1398 的 **96.1%**,ECC 开启下);L2 高达 **112 MB**(大于 B200 单 die 可见容量);L2 延迟 ~366 cycles、DRAM ~870 cycles,与 GB203 论文值(358/876.7)高度一致。
6. **FP64 仅为兼容性存在**:2 DFMA/SM/clk,实测 1.21 TFLOPS,依赖链延迟 64 cycles(论文 GB203:63.57)——与 B200 的 44.8 TFLOPS 差 37 倍。
7. **量化/反量化是访存受限**:dequant 内核(int8/fp8/fp4→fp16)均达到拷贝 roofline 的 90~95%,瓶颈在 GDDR7 带宽而非转换指令;转换指令吞吐 32~64 ops/clk/SM 足够富余。
8. **调度**:4 个 SMSP、1536 threads/SM(48 warps,比 B200 的 64 warps 少 25%);ILP=4 时 8 warps/SM 即饱和发射;分支分歧在编译器 if-conversion 下无可见惩罚;`__syncthreads()` 仅 5–15 cycles,kernel 启动 ~2 µs,调度基元都极廉价。
9. **互连**:无 NVLink;PCIe Gen5 x16 实测 H2D 56.5/D2H 57.3 GB/s(理论 63 的 90%);**P2P 可用**,GPU0↔GPU3 43.9 GB/s、延迟 7.3 µs。
10. **端到端对拍闭环**:Qwen3.6-27B-NVFP4 单卡 vLLM 推理,decode batch=1 实测 47.3 tok/s = DRAM 带宽 roofline 的 **88~93%**,prefill 达 nvfp4 GEMM 峰值的 **54~59%**——微基准数据可直接用于推理性能容量规划。

---

## 1. 设备规格:实测 vs 标称

| 项目 | RTX 6000D 实测 | RTX PRO 6000 标称(非D) | 6000D 标称(泄露/渠道) | B200 (sm_100) |
|---|---|---|---|---|
| GPU | GB202, sm_120 | GB202, sm_120 | GB202 | 双 die GB100, sm_100 |
| SM 数 | **156** | 188 | 156 | 148(论文2) |
| CUDA cores | 19968 | 24064 | 19968 | ~18944 |
| 持续 SM 时钟 | **2.30 GHz**(额定 2.43) | 2.617 boost | 2.43 | — |
| 显存 | 83 GiB GDDR7 ECC | 96 GB | 84 GB | 192 GB HBM3e |
| 位宽/速率 | **448-bit** × 24.96 Gbps | 512-bit | 448-bit | 8 堆栈 |
| 理论显存带宽 | **1397.9 GB/s** | 1792 | 1398 | 8000 |
| L2 | **112 MiB** | 128 MiB(推测) | — | 4 分区(容量未公开) |
| L1/shared | 128 KB 统一/SM(shared 上限 100 KB,opt-in 99 KB) | 同 | 同 | 256 KB/SM(+TMEM 256KB) |
| 寄存器堆 | 64K × 32-bit / SM(256 KB) | 同 | 同 | 256 KB/SM |
| 最大线程/SM | **1536(48 warps)** | 同 | 同 | 2048(64 warps) |
| 最大 blocks/SM | 24 | — | — | — |
| 互连 | PCIe Gen5 x16,无 NVLink | 同 | 同 | NVLink 5 × 18(1.8 TB/s) |
| TDP | 600W(实测满载仅 ~200W) | 600W | 450-600W | 1000W |
| Cluster launch | 支持 | — | — | 支持(+CTA pair) |

6000D 与标准版 RTX PRO 6000 的差异完全符合出口合规裁剪:SM -17%、显存 -12.5%、位宽 -12.5%,微架构不变。

---

## 2. SM 执行单元:指令延迟与吞吐

### 2.1 延迟(依赖链,cycles;单 warp)

| 指令 | 6000D 实测 | GB203(论文1) | 说明 |
|---|---|---|---|
| FADD/FMUL/FFMA (f32) | 4.44 | 4(fma) | 与论文一致 |
| HFMA2 (f16x2) | 4.44 | — | 同 FP32 |
| IMAD (i32) | 4.44 | 4(mad.lo) | |
| IADD3 (i32) | 2.44 | — | 简单 ALU 更短 |
| LOP3 | ~1.76 | — | |
| SHF(漏斗移位) | 4.44 | — | |
| FSETP+SEL | 10.5 | — | |
| POPC | 22.4 | — | 四分之一速率单元 |
| F2I / I2F | 44.8 / 39.8 | — | 转换走慢速路径 |
| MUFU: sin / rcp / rsqrt / lg2 / ex2 | 22.1 / 44.3 / 24.0 / 30.3 / 46.3 | — | 特殊函数单元 |
| **DFMA (f64)** | **64.07** | **63.57** | 与论文几乎相同——FP64 单元仅 2 个/SM,深流水 |

### 2.2 吞吐(ops/clk/SM,8 独立链 × 满载)

| 指令 | 实测 | 推断流水线宽度 | 对比 |
|---|---|---|---|
| FFMA f32 | 115.7(90% of 128) | **128/SM**(4×32/SMSP) | B200 同为 128 |
| FADD/FMUL f32 | 84.2 / 84.0 | 128(实测偏低,见注) | |
| HFMA2 f16x2 | 63.1 | 64/SM ⇒ FP16 FMA 速率 = FP32(128 FLOP/clk/SM) | |
| IADD3 i32 | 119.5 | **统一 INT32/FP32 128/SM**(论文1结论证实) | GH100 仅 64 INT |
| IMAD i32 | 61.4 | **64/SM INT 乘法/移位管** | |
| SHF | 61.4 | 64/SM | |
| LOP3 | 211.9 | **可同时上 FP32+INT 两条管**(128+64≈192,实测略超) | |
| MUFU 全系 | 15.2~15.9 | **16/SM**(4/SMSP) | |
| F2I / I2F | 39.3 / 23.3 | ~32-40 / ~24 | |
| POPC | 28.6 | ~32 | |
| DFMA f64 | 1.69 | **2/SM** | B200:64/SM |

绝对性能(2.30 GHz 持续):**FP32 FFMA 83.0 TFLOPS**(2.43 GHz 额定理伦 97 TF,达成 86%;按时钟归一 90%);**FP64 1.21 TFLOPS**;FP16(HFMA2)90.5 TFLOPS。

> 注:FADD/FMUL 测得 84 而 FFMA 116,怀疑编译器对 FADD/FMUL 链的调度与 FFMA 不同(或发射端口竞争),非硬件宽度减半;两值均已加编译屏障防常量折叠。

---

## 3. Warp 调度器

| 测试 | 结果 | 解读 |
|---|---|---|
| 占用上限 | 1536 threads/SM(48 warps),24 blocks/SM | 与 Ada (sm_89) 相同;B200 为 64 warps/SM |
| 发射率扫描(ILP=4 FFMA) | 1 warp→24.7;2→49.1;4→98.2;**8→115.2(饱和)**;16/32→116 | 4 SMSP 每周期各发 1 条;**8 warps/SM(2/SMSP)即打满**;细扫在 5/9 warps 出现凹陷(warp 在 SMSP 间分布不均) |
| 延迟隐藏(ILP=1 依赖链) | 1 warp→6.9;线性爬升;**16 warps→94.8;32 warps→104** | FFMA 延迟 4.4 cycles,需要 ~16 warps/SM 完全隐藏 |
| 分支分歧(半 warp if/else) |  slowdown = 1.0005 | 编译器 if-conversion 消除分歧,无可见惩罚 |
| **`__syncthreads()` 开销**(单 block,扣除空循环) | 1 warp→5.0;2→7.0;4→11.0;8→14.2;**16/32 warps→15.25 cyc/barrier** | 与 B60 的 OpControlBarrier(2–11 cyc)同为"近似免费";32 warps 时趋近 ~15 cyc 的固定收敛成本 |
| **kernel 启动开销**(empty kernel) | 单次 2.82 µs;背靠背摊销 **2.05 µs**;满 GPU(sm×32 blocks)空 kernel 4.10 µs | 量级与 B60(固定 ~3.7 µs + 40 ns/WG)相当;grid 规模本身只增加 ~2 µs |
| 时钟行为 | 满载 FFMA 自旋:2302~2325 MHz,120~163W,`clocks_event_reasons=0x0` | 600W 功耗墙富余极大,无降频;空闲 28-30W |

---

## 4. Tensor Core

### 4.1 mma.sync PTX 微基准(sm_120a,ILP=8,32 warps/SM)

| 指令 | 吞吐 | 指令速率 | 累加器依赖延迟 | 相对 FP16 |
|---|---|---|---|---|
| m16n8k16 **FP16**→f32 | **143.2 TFLOPS** | 0.0974 mma/clk/SM | 29.69 cyc | 1× |
| m16n8k16 **BF16**→f32 | 142.8 TFLOPS | 0.0972 | 29.69 | 1.00× |
| m16n8k8 **TF32** | 72.0 TFLOPS | 0.0980 | 29.69 | 0.50× |
| m16n8k32 **FP8 e4m3**(kind::f8f6f4) | **287.5 TFLOPS** | 0.0978 | 29.69 | 2.01× |
| m16n8k32 **FP8 e5m2** | 286.3 TFLOPS | 0.0974 | 29.69 | 2.00× |
| m16n8k32 **FP4 e2m1**(kind::f8f6f4) | **375.9 TFLOPS** | **0.1279** | 29.69 | **2.63×** |
| m16n8k32 **INT8** | **288.1 TOPS** | — | 27.69 | 2.01× |
| dp4a (s8) | 22.7 Tinst/s | 63.3 inst/clk/SM | 4.57 | — |

要点:
- 所有精度**延迟完全相同(29.7 cycles)**——与论文 2 在 B200 上的观察一致(11.2-12.6 cyc 几乎不随精度变),说明低精度靠加宽数据通路而非改流水线。但 sm_120 的 mma.sync 依赖链延迟是 B200 tcgen05 的 **2.6×**。
- FP8 = 2× FP16(与 B200 相同比例);**FP4 仅 2.63× FP16(B200 为 4×)**——sm_120 的 FP4 靠提升指令速率而非数据通路翻倍,e2m1 仍以 4 寄存器形式喂入(f8f6f4 家族)。

### 4.1.1 饱和性探针(mma_probe.cu)与 SASS 确认

| 变体 | 吞吐 | 结论 |
|---|---|---|
| fp16, warps/SM = 8 / 16 / 32 / 48 | 114.7 / 142.5 / 143.3 / **143.6** | 16 warps/SM 即饱和,48 warps 无增益 |
| fp16, ILP = 4 / 8 / 16(32 warps/SM) | 142.1 / 143.3 / 144.1 | ILP≥4 即饱和 |
| **fp16 f16 累加器**(ILP=8/16) | 143.8 / 143.8 | **acc16 无加速**(与老 GeForce 的 2× acc16 传统不同) |
| fp8 e4m3, ILP=16 | 287.9 | 与 ILP=8 相同 |
| fp4 e2m1, ILP=16 | 377.1 | 与 ILP=8 相同 |

SASS 反汇编(cuobjdump):
```
812 × HMMA.16816.F32        (fp16/bf16 → f32)
696 × HMMA.16816.F16        (fp16 → f16, acc16 变体)
464 × QMMA.16832.F32.E4M3   (fp8)
464 × QMMA.16832.F32.E2M1   (fp4 → 同样落 QMMA)
```
**FP4 在 sm_120 上回落到 QMMA 而非 OMMA**——与论文 1 在 GB203 上的发现完全一致(仅当使用 ue8m0/block scaling 时才出现 OMMA)。143-144 TFLOPS 是裸 mma.sync 路径的硬上限(发射速率 ~0.097 HMMA/clk/SM ≈ 400 FLOP/clk/SM),与 ILP、warp 数、累加器类型均无关。**但这不是芯片上限**:block-scaled nvfp4 CUTLASS GEMM 实测 651.6 TFLOPS(见 4.2.1),证明 scale 参与的指令形态能解锁更高的 tensor 数据通路吞吐——裸 mma.sync 的 QMMA 回落路径反而成为瓶颈。
- 与 datasheet 推导值对比:标准版 6000 标称 ~4000 AI TOPS(FP4 稀疏)⇒ 稠密 FP16 ≈ 500 TF(188 SM)⇒ 6000D 等效 ~415 TF。实测 mma.sync/cuBLAS 仅 143-153 TF(**~36%**)。差距来源:标称值基于稀疏+boost 时钟+完整 tensor 阵列利用率,而 mma.sync 是 sm_120 唯一可用路径(无 tcgen05),其发射模型无法吃满 5 代 Tensor Core 的标称阵列。

### 4.1.2 Tensor 停顿窗口与 CUDA core 重叠(mma + FFMA interleave)

单 warp、fp16 mma.sync 依赖链,每迭代插入 K 个独立 FFMA(`sched_extra_bench.cu`;基线含 ~170 cyc 的运行时分支循环开销,看**增量**):

| 每迭代插入 FFMA 数 K | 0 | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|---|
| cycles/iter | 203.0 | 203.0 | 204.0 | 204.0 | 206.0 | 207.0 | 211.0 |

**结论**:32 个独立 FFMA 总共只增加 ~8 cycles(0.25 cyc/FFMA)——**CUDA core 的 FP32 通路在 mma.sync 依赖停顿窗口内几乎免费并行执行**,与 B60 的 SBID 停顿窗口吸收 ALU 工作(XMX 33 cyc 停顿内 16 个 FFMA 免费)完全同构。实践含义:dequant/scale 等逐元素操作可以直接与 tensor 指令交织,不需要专门的隐藏手段;这也解释了第 6 节 dequant 内核为何能贴近访存 roofline。

### 4.2 cuBLASLt GEMM(M=N=K=8192)

| 精度 | 实测 | 说明 |
|---|---|---|
| FP32 (compute f32) | 66.1 TFLOPS | 接近 CUDA core 峰值 83 的 80% |
| TF32 | 70.3 TFLOPS | = mma.sync tf32 微基准(72)的 98% |
| FP16 (acc f32) | 136.1 TFLOPS | |
| FP16 (acc f16) | 152.8 TFLOPS | 全套件最高 GEMM 值 |
| BF16 (acc f32) | 134.7 TFLOPS | |
| INT8 (IMMA) | 136.2 TOPS | |
| **FP8 e4m3 / e5m2 / e4m3×e5m2** | **不支持(no algo)** | 尝试 f16/bf16/f32/fp8 四种输出类型均被 cublasLtMatmulAlgoGetHeuristic 拒绝 |
| **FP4 e2m1 (nvfp4)** | **不支持(no algo)** | 同上 |

### 4.2.1 cuBLASLt 之外的实测可用路径(M=N=K=8192,均验证可用)

| 路径 | 精度 | 实测 | 备注 |
|---|---|---|---|
| torch `_scaled_mm`(CUTLASS 内核) | FP8 e4m3 per-tensor | **264.1 TFLOPS** | torch 2.11+cu130;达 mma 指令峰值 287 的 92% |
| torch `_scaled_mm` | FP8 e4m3 rowwise | 261.1 TFLOPS | |
| vLLM `cutlass_scaled_mm` | FP8 e4m3 | 261.9 / 254.6 TFLOPS | 两个 vLLM 镜像交叉验证 |
| **vLLM `cutlass_scaled_fp4_mm`** | **nvfp4(block16, e4m3 scale)** | **651.6 TFLOPS** | torch 2.12+cu132;**= 4.26× FP16 GEMM**;正确性校验通过(与反量化参考 rel-diff 0.6%,量化噪声本身 11.3%) |
| torch matmul | FP16 | 134.8 TFLOPS | 与 cuBLASLt 一致 |

**结论修正**:cuBLASLt 的 no algo 只是 cuBLASLt 自己的问题;PyTorch/vLLM 的 CUTLASS 内核在 sm_120 上 FP8/nvfp4 全部打通。特别是 **nvfp4 的 651.6 TFLOPS 远超裸 mma.sync FP4 微基准(376)**——block-scaled 路径(scale 参与的 mma,疑为 OMMA)比裸 `kind::f8f6f4` 的 QMMA 回落快 73%,说明 6000D 的 5 代 tensor core 完整能力只有经 block-scaled 指令形态才能释放。

### 4.3 tcgen05 / TMEM 探测

```
ptxas error: Instruction 'tcgen05.alloc' not supported on .target 'sm_120a'
ptxas error: Feature '.cta_group::1' not supported on .target 'sm_120a'
```

sm_120 **没有 tcgen05、没有 TMEM、没有 CTA-pair**。B200 的三大 AI 架构创新(TMEM 256KB/SM、单线程发射 warp 级 MMA、CTA 对共享操作数)在 6000D 上全部缺席——这是 sm_120 与 sm_100 的分水岭,不是简单的"规格缩小"。

---

## 5. 内存层级

### 5.1 DRAM(STREAM 风格,2 GB/缓冲区,ECC 开)

| 测试 | 实测 GB/s | 达成率(理论 1397.9) | B200(论文2,标称 8000) |
|---|---|---|---|
| read | **1344.1** | **96.1%** | — |
| write | 1283.3 | 91.8% | — |
| copy (r+w) | 1267.0 | 90.6% | — |
| triad (2r+1w) | 1285.2 | 91.9% | 4140(51.8%,4-16GB 工作集) |

GDDR7 + 448-bit 的效率极高(96% 读达成率,ECC 开启)。B200 在论文 2 的小工作集 STREAM 只有 52%,但那是 HBM3e 小数组效应;按标称 8 TB/s,B200 带宽是 6000D 的 **5.7×**。

### 5.2 延迟(指针追逐,cycles @2.30GHz)

| 工作集 | 延迟 | 层级判定 |
|---|---|---|
| **shared memory(1–96 KB)** | **34.0 cyc(14.8 ns),全程平坦** | 独立 SRAM、无 tag/组相联结构(B60 SLM:46 cyc 且随容量上升;GB203 论文:~29 cyc) |
| ≤64 KB | **44.1 cyc(19.2 ns)** | **L1 命中**(GB203 论文:30-40 cyc) |
| 128 KB | 246 cyc | L1(128KB 容量)溢出,部分 L2 |
| 256 KB | 322 cyc | 过渡区 |
| 1 MB – 64 MB | **~366 cyc(159 ns)** | **L2 命中**(GB203:358;GH100:273) |
| 128 MB(>112MB L2) | 640 cyc | L2 溢出过渡 |
| 256 MB – 4 GB | **759–877 cyc(330–381 ns)** | **DRAM**(GB203:876.7;GH100:658.7) |
| 8 GB | 1603 cyc | DRAM + 页表/行冲突恶化 |

- **L2 容量实测 112 MiB**(cudaDeviceProp),延迟曲线在 64 MB 稳定、128 MB 开始爬升,吻合。这个 L2 比 GB203 的 65 MB 大 72%,对 LLM KV cache/权重驻留有利。
- L2-only(`__ldcg`)与 L1-enabled 在大工作集下数值一致,符合"global load 默认过 L1,容量行为由 L2 决定"。
- **shared memory 延迟(34 cyc)比 L1(44 cyc)低 23%**,且 1–96 KB 全程平坦——shared 是独立物理 SRAM、无 tag 查找与组冲突,与 Intel Xe2 的 SLM/L1 统一 SRAM(SLM 46 vs L1 71 cyc)设计取向不同,与 NVIDIA 传统一致。单 block 动态 shared 上限 opt-in 为 **99 KB**(101376 B)。

### 5.3 各级带宽

| 层级 | 实测 | 归一化 |
|---|---|---|
| L1 (256KB/SM 工作集, ldca) | 29.1 TB/s | ~186 B/clk/SM |
| **L2 (64MB, ldcg)** | **5.41 TB/s** | — |
| **shared(128-bit, 无冲突)** | **44.8 TB/s** | **124.9 B/clk/SM ≈ 理论 128** |
| shared(32-bit, stride-32) | 1.44 TB/s | **31× 下降 = 教科书式 32 路 bank 冲突** |
| global atomicAdd(同地址) | 53.4 Gops/s | 串行化 ~19 ns/op |
| global atomicAdd(分散地址) | 492.6 Gops/s | L2 原子单元吞吐 |

### 5.4 TLB

2 GB 缓冲、随机触页:小 stride(4–64 KB,触 32K–64K 页)与 大 stride(≥1 MB,触 ≤2K 页)两轮测量分别在 [944–964] 与 [438–454] cycles;第二轮(大页命中较好时)各档全部 ~438–454。**TLB 容量效应存在但依赖物理页映射**,未测得干净拐点;定性结论:小页(4KB)随机访存在 ~2× 的页表惩罚,大页(2MB)下 2GB 内无可见 TLB 瓶颈。

---

## 6. 量化 / 反量化

### 6.1 转换指令吞吐(8 独立链)

| 转换 | ops/clk/SM | 速率 |
|---|---|---|
| f32x2 → f16x2 (cvt.rn.f16x2.f32) | 31.9 | 22.9 Telem/s |
| f16x2 → f32x2 | 26.4 | 18.9 Telem/s |
| **f32x2 → e4m3x2**(satfinite) | 33.2 | 23.8 Telem/s |
| **e4m3x2 → f16x2** | **63.4** | 45.5 Telem/s |
| f32x2 → bf16x2 | 31.8 | 22.8 Telem/s |
| e2m1 软解包(LUT,8 elem/u32) | 1.97(=15.8 elem/clk) | 5.7 Telem/s |
| int8x2 → f16x2 | 8.0 | 5.7 Telem/s |

FP8 有硬件 cvt(进/出都是全速);**FP4 没有独立 cvt 指令**,软解包(LUT+移位)速率只有 FP8 硬件路径的 ~1/4——但 FP4 的反量化融合在 mma 内(见 4.1),独立软解包只影响 weight-only 量化的预处理。

### 6.2 量化/反量化内核(256M 元素,访存受限)

| 内核 | 字节吞吐 | 元素吞吐 | vs 拷贝 roofline(1260 GB/s) |
|---|---|---|---|
| dequant int8+scale → fp16 | 1184 GB/s | 395 Gelem/s | **94%** |
| dequant fp8+scale → bf16 | 1191 GB/s | 397 Gelem/s | **95%** |
| dequant nvfp4+fp8scale → fp16 | 1131 GB/s | 453 Gelem/s | **90%** |
| quant fp16 → int8 | 1893 GB/s | 631 Gelem/s | 100%(按 3B/elem 足迹) |
| quant fp16 → e4m3 | 1889 GB/s | 630 Gelem/s | 100% |
| quant fp16 → e2m1(pack) | 2099 GB/s | 840 Gelem/s | 100%(按 2.5B/elem) |

所有 (反)量化内核都顶到 GDDR7 带宽墙(~1.26–1.28 TB/s 有效),**计算完全不是瓶颈**——这印证论文 2 在 B200 上的结论:量化后瓶颈从带宽移向计算;在 6000D 上意味着 dequant 可以免费做,模型大小×带宽才是推理上限。

---

## 7. 互连

| 测试 | 实测 | 说明 |
|---|---|---|
| PCIe H2D (pinned) | 56.5 GB/s | Gen5 x16 理论 63,达成 90% |
| PCIe D2H (pinned) | 57.3 GB/s | 90% |
| PCIe H2D (pageable) | 13.8 GB/s | |
| P2P GPU0↔GPU3 | 支持,**43.9 GB/s** 双向 | 同 NUMA(NODE 路径,经 PCIe);延迟 7.3 µs |
| **P2P 全矩阵(6×6000D,30 对)** | **41.8–44.4 GB/s,全部均匀** | 跨 NUMA(SYS 路径)仅低 ~2%(42.3 vs 43.9),PCIe 拓扑无瓶颈点对 |
| NVLink | **无**(nvidia-smi nvlink 空) | B200: NVLink5 1.8 TB/s |

6000D 保留了 P2P(许多消费卡被禁用),8 卡单机做张量并行可行,但 43.9 GB/s 的 P2P 带宽只有 NVLink5 的 **2.4%**——大规模 TP 的 all-reduce 会成为主要瓶颈。

---

## 7.5 端到端对拍:vLLM nvfp4 推理(Qwen3.6-27B-NVFP4,单卡 GPU0)

用 `verdictai/glm52-nvfp4-dcpmtp:v3.3` 镜像(torch 2.12+cu132,含 `cutlass_scaled_fp4_mm`)离线加载 **Qwen3.6-27B-NVFP4**(`nvfp4-pack-quantized`,group_size=16,权重文件 26.38 GB,混合 Mamba 架构),vLLM V1 引擎 + CUDA graph,测 prefill/decode 吞吐,与本报告微基准 roofline 对拍(`src/e2e_vllm.py` → `results/e2e_vllm.json`):

| 场景 | 实测 | 微基准预测(roofline) | 达成率 |
|---|---|---|---|
| decode batch=1 | **47.3 tok/s** | 权重逐 token 读取 ≈ 26.38 GB → 1344 GB/s ÷ 26.38 GB = **53.8 tok/s** | **88%**(实际权重流量 47.3×26.38 ≈ 1248 GB/s = DRAM 读带宽的 **93%**) |
| decode batch=32 | **1029 tok/s**(聚合,32.2 步/s) | 带宽口径:32.2 步/s × 26.38 GB = 848 GB/s | 63%(开始向 compute bound 过渡,Mamba 层与 KV 开销显现) |
| prefill 2048 tok batch=1 | **6481 tok/s**(0.316 s) | ≈2×27B×6481 = 350 TFLOPS vs nvfp4 GEMM 峰值 651.6 | **54%** |
| prefill 2048 tok batch=8 | **6798 tok/s**(2.42 s,引擎自报 7076 tok/s) | 367~382 TFLOPS vs 651.6 | **56~59%** |

结论:

1. **微基准能准确预测端到端**:decode batch=1 是纯带宽受限场景,实测直接顶到 DRAM 读 roofline 的 88~93%,与第 5.1 节 STREAM 读带宽(1344 GB/s)和第 6 节 dequant roofline(~1.19 TB/s)互相印证;prefill 达到 nvfp4 CUTLASS GEMM 峰值(651.6 TFLOPS)的 54~59%,考虑 Mamba/attention/归一化等非 GEMM 开销,属于正常区间。
2. **nvfp4 的双重收益在真实模型上兑现**:27B 模型权重压到 26 GB(nvfp4 ~0.56 B/param),84GB 单卡装下且 batch=1 decode 接近带宽极限;若用 fp16 权重(~54 GB),同带宽下 decode 上限仅 ~25 tok/s——nvfp4 直接翻倍。
3. 工程注意:该镜像需 `max_num_seqs ≤ 974`(Mamba cache 每序列占一个 block,否则 CUDA graph 捕获失败);模型架构 `Qwen3_5ForConditionalGeneration` 已被该镜像与 voipmonitor b12x 系列镜像原生支持。

---

## 8. RTX 6000D vs B200:总表

| 维度 | RTX 6000D (GB202/sm_120) 实测 | B200 (GB100/sm_100,论文2实测) | 比值 |
|---|---|---|---|
| FP32 CUDA core | 83.0 TFLOPS | ~80 TF(标称) | **≈1×** |
| FP64 | 1.21 TFLOPS | 44.8 TFLOPS(99.6% peak) | 1/37 |
| FP16/BF16 tensor | 143–153 TFLOPS | 1926–1930 TFLOPS(96.5% peak) | **1/13** |
| TF32 tensor | 72.0 | 964.5 | 1/13 |
| FP8 tensor | 264(PT/CUTLASS);287 PTX 峰值 | 3850.6 | **1/15** |
| FP6 tensor | 不支持(e2m3/e3m2 未测得独立路径) | 5134.4 | — |
| FP4 tensor | **651.6(nvfp4 CUTLASS)**;376 裸 mma | 7700.2 | **1/12** |
| INT8 | 136–288 TOPS | 3928.5 TOPS | 1/14–29 |
| 显存 | 83 GiB GDDR7,实测读 1.34 TB/s | 192 GB HBM3e,实测 triad 4.14 TB/s | 容量 1/2.3,带宽 1/3(实测)/1/5.7(标称) |
| L2 | 112 MiB,~366 cyc | 4 分区(容量未公开) | — |
| L1 延迟 | 44 cyc | 未获取 | — |
| DRAM 延迟 | ~870 cyc | 未获取(GB203 同值 877) | — |
| MMA 依赖延迟 | 29.7 cyc(mma.sync) | 11.0–12.6 cyc(tcgen05) | 2.6× |
| TMEM | 无 | 256 KB/SM,16 TB/s 读 | — |
| warps/SM | 48 | 64 | 3/4 |
| 互连 | PCIe5 P2P 43.9 GB/s | NVLink5 1.8 TB/s | 1/41 |
| 硬件解压引擎 | 无 | 有(~100-500 GB/s 输出) | — |
| TDP / 实测功耗 | 600W / 测试峰值 ~200W | 1000W | — |

### 架构层面关键差异总结

1. **指令集断层**:sm_120 = "Ampere 式 mma.sync + 5 代 tensor 数据通路",sm_100 = "tcgen05 + TMEM + CTA pair"。6000D 与 B200 虽然都叫 Blackwell,编程模型相差一代。
2. **低精度梯度**:以实际可用 GEMM 计,6000D FP16:FP8:FP4 = 1 : 1.7 : 4.3(152.8 / 264 / 651.6 TFLOPS);B200 = 1 : 2 : 4(1929 / 3850 / 7700)。**nvfp4 推理在 6000D 上同时吃到算力(4.3×)与显存/带宽(4× 权重压缩)双重收益**,是部署首选精度;但注意必须走 block-scaled 内核(如 vLLM CUTLASS),裸 mma.sync 和 cuBLASLt 都拿不到这个性能。
3. **目标负载分野**:6000D 的强项是 FP32 图形/渲染类吞吐(≈B200)、超大 L2(112MB)、高显存达成率(96%)、单卡 84GB 容量与 600W 静音工作站形态;B200 的强项是 tensor 计算密度(13–20×)、HBM 带宽(3–6×)、NVLink 扩展(41×)与 FP64(37×)。
4. **同族验证**:6000D 的全部微架构参数(L1/L2/DRAM 延迟、统一 INT/FP 管、FP64 双单元、48 warps/SM)与论文 1 的 GB203 完全吻合,sm_120 家族行为可互相外推;论文 1 的 cuBLASLt FP8 异常低值(<1 TF)未在本平台复现——本平台是干脆"无算法可用",更接近真相。

### 8.1 单卡 nvfp4 LLM 推理等效比(排除互连因素)

由本报告的微基准与 §7.5 端到端数据推导(同模型、同 nvfp4 权重,推理吞吐在两个 regime 分别由 tensor 算力和显存带宽决定):

| 推理场景 | 决定因素 | 6000D 实测 | B200(论文2/标称) | **6000D ≈ B200 的** |
|---|---|---|---|---|
| prefill / 大 batch decode(compute bound) | nvfp4 GEMM 峰值 | 651.6 TFLOPS | 7700.2 TFLOPS(论文2实测) | **~8.5%**(按标称 9 PFLOPS 为 7.2%) |
| decode 小 batch(memory bound) | 显存带宽 ÷ 权重字节数 | 读 1.34 TB/s | 8 TB/s 标称 / 4.14 TB/s 论文 triad | **~17%(标称)~ 31%(论文实测)**,现实取 **~17–20%** |
| 典型混合 serving 负载 | 两者加权 | — | — | **~10–20%**,越偏 decode 越有利 |

锚点验证:6000D 上 Qwen3.6-27B-NVFP4(26.4 GB 权重)decode bs1 = 47.3 tok/s;同模型在 B200 上按 ~7 TB/s 有效带宽外推约 250 tok/s,比值 ~19%,与带宽比一致。物理原因:nvfp4 权重压缩对两卡等效,而 tensor 算力差 ~12×、显存带宽只差 3–6×,因此 **decode 是 6000D 相对差距最小、性价比最高的推理区间**(单并发、低延迟、84GB 装大模型);prefill/大批量吞吐场景则与 B200 差一个数量级。

---

## 9. 方法学陷阱记录(复现者必读)

1. **时钟测量波次 bug**:测持续时钟时若发射的线程数超过 SM 容量(156×1536),内核会分 2 个波次执行,wall time 翻倍,得到 1.16 GHz 的假时钟(真实 2.30 GHz)。单波次(≤1536 threads/SM)后正常,与 `%globaltimer` 交叉验证一致(clock64 在 GB202 上按 SM 时钟走,无半速问题)。
2. **编译器常量折叠**:操作数为编译期常量时,nvcc 会把 IADD3/LOP3/FADD/FMUL 的迭代链化简为闭式(FADD 表现为吞吐虚低 84/128,IADD3 表现为 59000 ops/clk 的荒谬值)。修复:`__constant__` 运行时操作数 + 每迭代 `asm volatile("" : "+f"(a))` 编译屏障。
3. **mma 内核内的 switch**:按运行时分支选择 mma 类型会给每次迭代加分支开销,且扭曲不同精度的对比(fp16 虚高、bf16 虚低);改为模板特化后 fp16=bf16(142.8≈143.2),e4m3=e5m2,完全符合硬件对称性。
4. **CUDA 13 变化**:`cudaDeviceProp` 移除了 clockRate/memoryClock/L2CacheSize 字段(改用 `cudaDeviceGetAttribute`);`-arch=sm_120a` 不再生成 compute_120a(需显式 `-gencode arch=compute_120a,code=sm_120a`,否则 `kind::f8f6f4` 被 ptxas 拒绝)。
5. **shared memory 带宽测试**:循环不变的下标会被提升为寄存器复用(测出 904 B/clk 的非物理值),下标必须依赖循环变量。
6. **cuBLASLt FP8/FP4 探测**:要遍历多种输出类型(f16/bf16/f32/fp8)才能下"不支持"结论。
7. **`__syncthreads()` 必须在非分歧路径**:第一版 barrier 测试把 `__syncthreads()` 写进 `if (threadIdx.x == 0)` 分支,只有 thread 0 到达 barrier,内核死锁(GPU 占用 100% 挂死,必须 kill 容器恢复)。计时读 `clock64()` 可以分歧,barrier 不行。

---

## 10. 局限与后续工作

- 未测:RT core、MIG 实例化行为、稀疏(sparsity)路径、fp6(e2m3/e3m2)mma(PTX 支持但本次未单列)、持久化 L2(persisting 70MB)效果、NVENC/NVDEC。
- 已补充验证(深挖轮):tensor 饱和性探针(ILP/warp/acc16 扫描 + SASS 反汇编确认 HMMA/QMMA、FP4 回落 QMMA 无 OMMA);6 卡 P2P 全矩阵(30 对全部 41.8–44.4 GB/s);**低精度真实框架路径**:torch `_scaled_mm` FP8(264.1 TFLOPS)、vLLM CUTLASS FP8(261.9)、vLLM CUTLASS nvfp4(651.6 TFLOPS,正确性已校验);**端到端对拍**:Qwen3.6-27B-NVFP4 单卡 vLLM 推理(decode 47.3 tok/s @bs1 = 带宽 roofline 88~93%,prefill ~6.5k tok/s = nvfp4 峰值 54~59%,见 §7.5)。
- 功耗仅在 microbenchmark 负载下采样(峰值 ~200W),未做 600W 压力墙测试。
- B200 侧数据全部引自论文 2(及规格表),未实机复测;论文 2 未覆盖的 B200 项(时钟、L1/L2/global 延迟、NVLink 实测、CUDA core FP32)在本报告中标注"未获取"。
- 后续可做:fp6(e2m3/e3m2)mma 单列测试;8 卡 all-reduce(NCCL)实测;更大 batch(64~512)下 decode 的 compute-bound 拐点定位。

---

## 附录:原始数据位置

- 姊妹篇对比:[RTX6000D_vs_B60_comparison.md](RTX6000D_vs_B60_comparison.md)(与本仓库 Intel Arc Pro B60/Xe2 套件的横向对比)

- 源码:`/home/tina/DEV/microbench/src/`(12 个 .cu + bench_common.h + run_all.sh + 4 个 python 框架级/端到端测试:fp8_gemm_test.py / nvfp4_gemm_test.py / nvfp4_check.py / e2e_vllm.py;sched_extra_bench.cu 为 B60 报告对标补充:smem 延迟/barrier 开销/mma+FFMA 交织/启动开销)
- 原始 CSV:`/home/tina/DEV/microbench/results/*.csv`(含 lowprecision_gemm.csv 框架级结果、e2e_vllm.json 端到端结果、sched_extra.csv 对标补充;远程 `sdf@172.16.120.54:~/microbench/results/` 同步)
- 复现:`docker run --rm --network=none --gpus '"device=0,3"' -v ~/microbench:/work nvidia/cuda:13.0.1-devel-ubuntu22.04 bash /work/src/run_all.sh all`
