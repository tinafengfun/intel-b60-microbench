# NVIDIA RTX PRO 5000 72GB Blackwell vs RTX 6000D 微基准对比报告

同一套微基准套件(`src/`,12 个 .cu + 4 个 python)在两个 sm_120 平台上复现:

| 平台 | GPU | 节点 | 软件栈 |
|---|---|---|---|
| A | RTX 6000D(GB202 合规版,156 SM) | 172.16.120.54,GPU0 | CUDA 13.0.1 容器,vLLM 镜像(torch 2.12+cu132) |
| B | **RTX PRO 5000 72GB Blackwell(110 SM)** | 10.239.11.69,GPU0/1 | NGC PyTorch 26.04 容器(CUDA 13.2,torch 2.12) |

> 注意:B 侧框架级测试(cuBLASLt/torch)用的是 **CUDA 13.2**,A 侧是 13.0.1/13.2 混合;
> 微基准(PTX 级)两侧都用各自 CUDA 13 编译 `-gencode arch=compute_120a,code=sm_120a`,指令一致,可直接对比。

## 0. 关键结论(TL;DR)

1. **RTX PRO 5000 的每 SM tensor 吞吐是 6000D 的 2.56 倍**:同一条 `mma.sync.m16n8k16` PTX 指令、
   同一份 SASS(HMMA.16816),fp16 发射速率 0.2493 vs 0.0974 mma/clk/SM。
   PRO 5000 = **满血 Blackwell 客户端速率(1021 FLOP/clk/SM,与 RTX 5090/GB203 datasheet 推导值 1022 一致)**;
   6000D 被限制在 399(约满血的 39%)——**这是 6000D 出口合规阉割落在硬件指令发射级的直接证据**。
2. **6000D 报告 §4.1.1 的"mma.sync 无法吃满 5 代 tensor"结论需要修正**:在 PRO 5000 上,裸 mma.sync
   就能达到满血速率(289 TF @ 110 SM ≈ 等效 188 SM 的 ~494 TF,与 datasheet 稠密 FP16 ~500 TF 吻合)。
   6000D 上 mma.sync 与 datasheet 的 36% 差距不是编程模型问题,而是合规上限。
3. **FP4 梯度差异**:6000D 的 fp4 mma.sync 有 +31% 指令速率加成(375.9,部分绕开限制);
   PRO 5000 的 fp4 = fp8(577.4,无加成)——满血路径下裸 mma.sync 的 fp4 增益消失,
   真 fp4 收益只走 block-scaled nvfp4(PRO 5000 torch 原生 `_scaled_mm` **876.6 TFLOPS**,= bf16 的 4.2×)。
4. **SM 微架构逐项相同**:全部 ALU 延迟(FFMA 4.44/DFMA 63.4)、mma 依赖延迟(29.7 cyc)、
   L1 44 / shared 34 / L2 ~366 / DRAM ~910 cyc、barrier 5–14.5 cyc、tensor 停顿窗口吸收 FFMA——
   两侧完全吻合,sm_120 家族行为稳定可外推。
5. **平台差异**:PRO 5000 时钟更高(2.576 vs 2.30 GHz,300W vs 600W 形态),P2P 更快(53.5–54.1 vs 43.9 GB/s),
   但 kernel 启动更慢(3.9–7.0 vs 2.1–2.8 µs,平台/驱动差异);DRAM 读 1230 GB/s(标称 91.5%,ECC 关)
   vs 6000D 1344(96.1%,ECC 开)。**8 卡 P2P 全矩阵显示 PRO 5000 节点为 2×4 分组拓扑(跨 NUMA 组间 -26%),
   6000D 节点 6 卡则完全均匀**——8 卡 TP 部署应约束在 4 卡组内。
6. **CUDA 13.2 cuBLASLt 新 bug**:fp16 GEMM 在 PRO 5000 上确定性触发 illegal memory access
   (fp32/tf32 正常,torch matmul 正常);tf32 成绩 138 TFLOPS 异常高(=满血 fp16 速率的一半,
   疑似 13.2 把 tf32 映射到了新内核)——两侧软件栈差异已在文中标注。

## 1. 设备规格

| 维度 | RTX 6000D | RTX PRO 5000 72GB | B/A |
|---|---|---|---|
| SM 数 | 156 | **110** | 0.705 |
| 持续时钟(满载实测) | 2.30 GHz | **2.576 GHz** | 1.12 |
| 显存 | 84GB GDDR7 448-bit | 72GB GDDR7 **384-bit** | 0.857 |
| 标称带宽 | 1398 GB/s | 1344 GB/s | 0.96 |
| L2 | 112 MiB | 96 MiB | 0.857 |
| ECC | 开 | **关** | — |
| TDP | 600W | 300W | 0.5 |
| shared/block opt-in | 99 KB | 99 KB | 1× |
| warps/SM, blocks/SM | 48, 24 | 48, 24 | 1× |
| compute capability | sm_120 | sm_120 | 同 |

## 2. SM 执行单元:逐项相同

延迟(cycles)两侧完全一致:FFMA/FADD/FMUL 4.44、DFMA 63.4、IADD3 2.44、IMAD 4.44、LOP3 1.88、
SHF 4.44、MUFU_sin 22.1、F2I 45.3、HFMA2 4.44。
吞吐(ops/clk/SM)一致:FFMA ~116、IADD3 ~123、IMAD ~62、LOP3 ~212–223(双管)、DFMA 1.69、MUFU 16。

绝对性能按 SM×时钟缩放(0.705×1.12 = 0.79):

| 指标 | 6000D | PRO 5000 | B/A | 预期(SM×clk) |
|---|---|---|---|---|
| FP32 FFMA | 83.0 TFLOPS | 65.7 | 0.79 | 0.79 ✓ |
| FP64 | 1.21 TFLOPS | 0.955 | 0.79 | 0.79 ✓ |
| FP16(HFMA2) | 90.5 TFLOPS | 71.7 | 0.79 | 0.79 ✓ |

**CUDA core 部分每 SM 完全同构**——阉割不在 CUDA core。

## 3. Tensor Core:核心差异

### 3.1 裸 mma.sync PTX(ILP=8,32 warps/SM)

| 指令 | 6000D | PRO 5000 | B/A 绝对 | **每 SM·clk** 6000D | **每 SM·clk** PRO 5000 | **比值** |
|---|---|---|---|---|---|---|
| fp16 m16n8k16 | 143.2 TF | **289.4 TF** | 2.02× | 399 FLOP | **1021 FLOP** | **2.56×** |
| bf16 | 142.8 | 289.6 | 2.03× | 398 | 1022 | 2.57× |
| tf32 m16n8k8 | 72.0 | 144.9 | 2.01× | 201 | 511 | 2.55× |
| fp8 e4m3 m16n8k32 | 287.5 | **577.5** | 2.01× | 801 | 2039 | 2.54× |
| fp8 e5m2 | 286.3 | 577.6 | 2.02× | 798 | 2039 | 2.56× |
| fp4 e2m1 m16n8k32 | 375.9 | 577.4 | 1.54× | 1047 | 2039 | 1.95× |
| int8 m16n8k32 | 288.1 TOPS | 577.4 TOPS | 2.00× | 802 | 2039 | 2.54× |
| 依赖延迟 | 29.69 cyc | 29.69 cyc | 1× | — | — | 同 |
| dp4a | 63.3/clk/SM | 63.4/clk/SM | 1× | 同 | 同 | 同 |

- **指令发射速率**:fp16 0.0974 vs 0.2493 mma/clk/SM(2.56×);fp8/fp4/int8 0.0978 vs 0.2488(2.54×)。
- **fp4 特例**:6000D fp4 发射速率 0.1279(比 fp8 高 31%,部分绕开限制);PRO 5000 fp4 = fp8 = 0.2488(无加成)。
- SASS 操作码两侧完全相同(HMMA.16816.F32 / QMMA.16832 / IMMA.16832),差异在硬件发射级而非指令形态。
- 饱和性(mma_probe):两侧都是 16 warps/SM、ILP≥4 即饱和;acc16 都无加速。

### 3.2 框架级 GEMM(M=N=K=8192)

| 路径 | 6000D | PRO 5000 | B/A | 备注 |
|---|---|---|---|---|
| cuBLASLt fp32 | 66.1 | 51.8 | 0.78 | 按 SM×clk 缩放 ✓ |
| cuBLASLt tf32 | 70.3 | **138.0** | 1.96 | 13.2 疑似新内核路径 |
| cuBLASLt fp16/bf16/int8 | 136–153 | **illegal memory access** | — | **CUDA 13.2 确定性 bug**(fp16_acc32 触发,污染后续所有 case;已在 GPU0 与 GPU2 上两次复现,系统性非单卡问题;torch matmul 正常) |
| cuBLASLt fp8/fp4 | no algo | (未执行到) | — | 13.0 对 sm_120 无算法,与 6000D 一致预期 |
| torch matmul bf16 | 未测 | 208.5 | — | 满血速率:208.5e12/(110×2.576e9)= 735 FLOP/clk/SM |
| torch `_scaled_mm` fp8 | 264.1 | **488.4**(rowwise 480.8) | 1.85 | |
| **nvfp4 block-scaled** | 651.6(vLLM CUTLASS) | **876.6**(torch 原生 `_scaled_mm`) | 1.35 | 各自 bf16 的 4.26× / **4.20×** —— 梯度一致 |

nvfp4 每 SM·clk:6000D 1817 vs PRO 5000 3093 FLOP(1.70×)——block-scaled 路径在 6000D 上
部分绕开了 mma.sync 的 39% 限制(1817 > 399×4=1596),但仍低于满血。

## 4. 内存层级:几乎相同

| 指标 | 6000D | PRO 5000 | 说明 |
|---|---|---|---|
| DRAM 读 | 1344 GB/s(96.1%) | 1230.6 GB/s(91.5%) | 6000D ECC 开仍更高(448-bit 宽 17%) |
| DRAM copy / triad | 1267 / 1285 | 1083 / 1121 | |
| L1 延迟 | 44 cyc | 44 cyc | 同 |
| shared 延迟 | 34 cyc(平坦) | 34 cyc(平坦) | 同 |
| L2 延迟 | ~366 cyc | ~366–373 cyc | 同 |
| DRAM 延迟 | 759–877 cyc | 859–919 cyc | 333–357 ns,同量级 |
| L2 容量边界 | 128MB 起爬升 | 128MB 起爬升(96MB L2) | 一致 |
| L1/L2 带宽 | 29.1 / 5.41 TB/s | 12.2 / 5.31 TB/s | L1 按 SM 数缩放 ✓ |
| shared 带宽 | 124.9 B/clk/SM | 126.6 B/clk/SM | 同 |
| TLB(2GB 随机触页) | 438–964 cyc(不稳定) | **445–464 cyc(全平坦)** | PRO 5000 无可见 TLB 容量惩罚 |

## 5. 调度与运行时

| 指标 | 6000D | PRO 5000 | 说明 |
|---|---|---|---|
| 发射率饱和(ILP=4) | 8 warps/SM | 8 warps/SM(113.9/116) | 同 |
| 依赖链隐藏 | 16 warps | 同 | 同 |
| 分歧惩罚 | 1.0005 | 1.0015 | 同(无) |
| barrier | 5–15.25 cyc | 5–14.5 cyc | 同 |
| mma+FFMA 交织 | 32 FFMA +8 cyc | 完全一致(203→211) | 同 |
| kernel 启动 | 2.05–2.82 µs | **3.92–7.01 µs** | 平台差异(CPU/驱动),非 GPU 架构 |

## 6. 互连

| 指标 | 6000D | PRO 5000 |
|---|---|---|
| PCIe H2D / D2H | 56.5 / 57.3 GB/s | 55.3 / 57.4 GB/s |
| P2P(2 卡) | 43.9 GB/s | **53.5–54.1 GB/s**(+23%) |
| P2P 延迟 | 7.3 µs | 12.4 µs |
| P2P 全矩阵 | 6 卡 30 对全部 41.8–44.4(均匀,跨 NUMA 仅 -2%) | **8 卡 56 对呈两组各 4 卡**:组内(0-3 / 4-7,PIX/NODE)53.4–54.4;**跨组(SYS,跨 NUMA)38.4–40.9(-26%)** |
| NVLink | 无 | 无 |

PRO 5000 节点拓扑(`nvidia-smi topo -m`):GPU0–3 在 NUMA 0、GPU4–7 在 NUMA 1,组内 PIX/NODE、
跨组必走 SYS——**8 卡 TP 的 all-reduce 会受跨组 39 GB/s 限制**,部署时张量并行应控制在 4 卡一组内。
6000D 节点 6 卡则无此分层(全矩阵均匀)。

## 7. 量化/反量化

转换指令吞吐一致(e4m3→f16 63.4/clk/SM、fp4 LUT 软解包 1.98/clk/SM 等),dequant/quant 内核均为访存受限,结论与 6000D 报告 §6 相同。

## 8. 总表:6000D 的合规阉割在哪里

| 维度 | 每 SM·clk(PRO 5000 = 满血) | 6000D 相对满血 | 结论 |
|---|---|---|---|
| CUDA core(FP32/FP64/INT) | 同 | **100%** | 未阉割 |
| 内存带宽/容量 | — | 带宽 96–104%、容量 87.5%(对标准版) | 基本未阉割 |
| **mma.sync tensor 发射率** | 0.2493 mma/clk/SM | **39%** | **主要阉割点** |
| nvfp4 block-scaled | 3093 FLOP/clk/SM | 59% | 部分绕开 |
| 互连 | — | P2P 82%、无 NVLink(两者皆无) | 形态差异 |

**推定**:6000D 为满足出口管制 TPP 上限,在硬件/固件级把 tensor core 的 mma 指令发射速率限制到满血的 ~39%(fp16/fp8/int8)——这与 H20 对 Hopper 的做法同构。CUDA core、缓存、显存子系统未受影响。FP4 有 +31% 的残余弹性,nvfp4 block-scaled 路径能恢复到满血的 ~59%。

## 9. 复现与数据位置

- 远程:`root@10.239.11.69:/root/microbench/`(容器 `focusluo-nv_vllm_ubuntu24.04_vllm-26.04-py3-dev:latest`,CUDA 13.2)
- 本地:`results/pro5000/*.csv`(本目录)
- 复现:`docker run --rm --gpus '"device=0,1"' --entrypoint bash -v /root/microbench:/work <cuda13-image> bash /work/src/run_all.sh all`
- 已知平台坑:宿主 nvcc 为 CUDA 12.0(不认识 sm_120),必须用容器内 CUDA 13;CUDA 13.2 cuBLASLt fp16 GEMM 确定性崩溃,框架级 fp16 用 torch matmul 替代。
