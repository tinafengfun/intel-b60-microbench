# Intel Arc Pro B70 (BMG-G31) vs Arc Pro B60 (BMG-G21) 微基准对比报告

B60 侧数据来自本仓库 `REPORT.md`;B70 侧为同套微基准在同频(2.40 GHz 锁频)下的复现,
外加两轮针对 XMX/DPAS 的重新设计实验(cooperative-matrix 版 `results/b70/b70_xmx_true_rate.csv`、
裸 ESIMD dpas 版 `src/bench_esimd_dpas.cpp` 实测)。

> **v3 更正说明**:本报告经历两次自我纠错——
> v1("峰值 87.6 TF、每 XMX 只有 56%")错在 IGC 死代码消除(DCE);
> v2("发射极限 21.1 cyc/dpas、峰值 119 TF")的 kernel 已修正 DCE,但仍带
> cooperative-matrix 抽象层的代码生成开销;
> **v3 用 ESIMD 裸 dpas 指令(1:1 映射硬件)测得真实硬件极限:每 EU 16.0 cyc/dpas、
> 全机 157.2 TF @2.4GHz / 183.4 TF @2.8GHz。**

> 测试环境:B70 机器 172.16.124.12(i7-12700 主机),独显 `Intel Graphics [0xe223]` = BMG-G31,
> 256 EU(32 Xe core),驱动 Level-Zero V2 1.13.35563,oneAPI 2025.3。

## 0. 方法学:三个陷阱(本次最大的教训)

1. **xe 驱动 DVFS**:空闲 517MHz,必须 `xpu-smi config --frequencyrange 2400,2400` 锁频
   (测完恢复 `400,2800`)。
2. **IGC 静默死代码消除**:吞吐 kernel 只 store 部分累加器时,其余 DPAS 链被删除且
   不报错。kernel 必须消费全部中间结果,并用"工作量 ×N ⇒ 时间 ×N"自校验。
3. **抽象层代码生成开销**:即便 DCE 修复,SPIR-V cooperative-matrix 路径生成的 dpas
   序列仍比裸指令慢 24%(21.1 vs 16.0 cyc/dpas)。**测硬件极限必须用 ESIMD 裸 dpas
   (或等价底层路径)交叉验证**。另外 IGC 在 bmg-g31 上还有三个 segfault:INT8 DPAS、
   coop-matrix 多 offset store、coop-matrix OpFAdd。

## 1. 规格与实测峰值(v3 终版)

| 维度 | B70 (BMG-G31) | B60 (BMG-G21) | B70 / B60 |
|---|---|---|---|
| 计算单元 | 32 Xe core(256 EU) | 20 Xe core(160 EU) | 1.6× |
| **XMX 发射速率(每 EU,裸 dpas)** | **16.0 cyc/dpas = 256 FLOP/cyc** | **~16.1 cyc/dpas = 254 FLOP/cyc** | **相同** |
| **全机 FP16/BF16 峰值(实测)** | **157.2 TF @2.4GHz**;**183.4 TF @2.8GHz** | 97.66 TF @2.4GHz | **1.61× / 1.88×** |
| cooperative-matrix 路径可达 | 119.1 TF(IGC 开销 -24%) | 97.66 TF(IGC 无损耗) | B70 软件栈不成熟 |
| 显存带宽(float4 读) | 665 GB/s | 538 GB/s | 1.24× |
| kernel dispatch | 6.9 µs + 38–40 ns/WG | 3.7 µs + 40 ns/WG | 更差 |

**B70 的 XMX 硬件与 B60 完全相同(每 EU 同速率),性能优势纯粹来自 1.6× 的 EU 数量。**
时钟线性已验证:1200/1800/2400 MHz 下 cyc/dpas 恒定,2.8GHz 实测 183.4 TF = 157.2×2.8/2.4。

## 2. B70 XMX/DPAS 流水线细节(ESIMD 裸 dpas 实测)

**指令模型**:`dpas<8,R>` = 单条硬件指令(M=R 行, K=16, N=16, fp16, 4096×R/8 FLOP)。

**依赖延迟:~37 cyc**(单链 ILP=1:69.2ns/instr ÷ 8 线程/EU... 实测单线程 22.6ns/instr)。

**每指令发射成本(8 线程/EU 摊到单 EU)**:

| Repeat | cyc/instr | 分解 |
|---|---|---|
| 1 | 3.1 | 固定开销 ~1.1 + 2×1 |
| 2 | 4.5 | ~0.5 + 2×2 |
| 4 | 8.1 | ~0.1 + 2×4 |
| 8 | **16.0** | **2×8,开销完全摊销** |

**XMX 流水线为 2 cyc/行**:一条完整 dpas.8×16×16 需 16 cyc;repeat 数越大
每指令固定开销(~1 cyc)摊得越薄,R=8 时达到满速 256 FLOP/cyc/EU。

**饱和条件**:
- 8 线程/EU(2048 WIs):**ILP=2 即饱和 157.2 TF**——TLP 是最便宜的延迟隐藏手段;
- 1 线程/EU(256 WIs):需 ILP≥8,达 136.9 TF(18.4 cyc/dpas,接近但未满);
- ESIMD 路径在 8 线程/EU 下 ILP=2–8 全部满速,无 coop-matrix 版的寄存器悬崖
  (线程数多时驱动自动降 occupancy,单线程 GRF 预算反而充裕)。

## 3. 延迟与微架构(不受上述问题影响,与 B60 同构)

| 层级 | B70 | B60 | 差异 |
|---|---|---|---|
| L1 | 71.3 cyc | 70.8 | 同 |
| L2(192KB) | 170.9 | 162.4 | +5% |
| DRAM(128MB) | 270.6 | 260.7 | +4% |
| SLM | 47.3(64KB 处 119.4) | 46.1(116.6) | 同 |
| barrier | 0.66–18 cyc | 2–11 | 同 |

缓存组织、barrier、ALU 交织特性全部复现 B60——G31/G21 同为 Xe2 微架构。

## 4. 内存子系统

float4 读带宽:B70 **665 GB/s** vs B60 538 GB/s(+24%)。
算力/带宽比:B70 183.4 TF / 665 GB/s ≈ 276(满频)vs B60 97.66/538 ≈ 182
——B70 更偏计算型,B60 更均衡;decode(带宽受限)两者差距只有 24%,
prefill(矩阵受限)B70 领先 61–88%。

## 5. 软件栈成熟度(bmg-g31 实测)

| 问题 | 现象 | 绕行 |
|---|---|---|
| INT8 DPAS | IGC segfault(B60 同,设备无关) | 无 |
| coop-matrix 多 offset store / OpFAdd | IGC segfault | 顺序同址 store |
| **coop-matrix 代码生成低效** | 比裸 dpas 慢 24%(119 vs 157 TF);B60 上无此损耗 | **用 ESIMD 裸 dpas** |
| 静默 DCE | 删除未消费的 DPAS 链,不报错 | 消费全部 acc + 自校验 |

## 6. 结论(v3 终版)

1. **B70 = B60 的等比放大**:XMX 每 EU 速率完全相同(16 cyc/dpas、256 FLOP/cyc),
   峰值差异 = EU 数量比:157.2 vs 97.66 TF @2.4GHz(1.61×);B70 满频 2.8GHz 达
   **183.4 TF**(1.88×)。用户预估的 ~160 TF 已被实测证实。
2. **XMX 流水线**:2 cyc/行,单条 dpas.8×16×16 为 16 cyc;依赖延迟 ~37 cyc;
   8 线程/EU + ILP=2 即饱和,repeat=8 摊销指令开销。
3. **达到算力顶峰的正确姿势**:绕过 cooperative-matrix,用 ESIMD 裸 dpas
   (`xmx::dpas<8,8>`)+ 8 线程/EU。IGC 的 coop-matrix 路径在 bmg-g31 上损失 24%。
4. **微架构同构**:缓存/SLM/DRAM 延迟、barrier 与 B60 差异 ≤5%。
5. **带宽 +24%**(665 vs 538 GB/s),B70 更偏计算型定位。
6. **方法学**:锁频、防 DCE、裸指令交叉验证,缺一不可。

## 7. Xe3 特性探测:VRT 与 Morton TG Dispatch(间接观测)

针对两个 Xe3 代的新特性,设计间接观测实验验证 B70(Xe2)是否具备。
**结论:两者均无证据支持,B70 表现为经典 Xe2 行为。**

### 7.1 Morton-order ThreadGroup Dispatch:不存在(实测分发顺序为哈希条带)

**直接观测(dispatch trace)**:每个 WG 用原子计数器记录自己的 (group_x, group_y),
slot 序列即真实分发顺序(`src/bench_dispatch_order.cpp`,64×64 网格):

- 连续 64 个 WG 的行跨度 = **60.6/64**(行优先应 ~0–1,Morton 块应 ~8);
- 首 40 个分发:x ∈ {13,14,17} 聚集,y 在 {8,24,48,56,60…} 大范围跳动;
- 1D 网格同样非顺序(平均相邻 |Δid| = 1270)。

**B70 的分发器既不是行优先也不是 Morton,而是哈希/条带式散布**
(目的推测为跨核/跨内存通道的负载均衡)。

**缓存敏感性 A/B(旁证)**:1D 线性映射 vs 1D 软件 Morton 映射 vs 原生 2D 网格,
每个 WG 读 A 行切片 + B 列切片(行共享 A、列共享 B 的局部性敏感模式),
规模一直推到常驻窗口 128MB(>任何可能的 L2):
三者时间在所有规模下差异 ≤6%(512² WGs × 512KB:38.0/38.1/38.0ms)。
分发顺序对该访存模式无可观测影响——即便软件 Morton 也无收益,
说明 B70 的缓存层级本身就能吸收跨行复用。

### 7.2 VRT(可变寄存器分配):无证据,表现为固定 128 GRF/线程硬顶

- **寄存器天花板**:DPAS 活链实验(store 所有累加器)中,每线程需求
  6×16+16=112 GRF 时正常,ILP=7(需求 ≥128)触发 3× 性能悬崖;
  但 L0 `zeKernelGetProperties` 查询 **spillMemSize=0**——编译器选择
  串行化/重算而非溢出到 scratch,与固定架构寄存器寻址上限(128 GRF/线程)一致。
- **无平滑 occupancy 降级**:合成寄存器压力 kernel 被 IGC 整体移到内存
  (num_regs 恒为 3),无法构造中间压力;真实 DPAS kernel 在多 WG/核时
  未观察到细粒度分配带来的平滑过渡。
- **附带异常(值得记录)**:live-dpas kernel 在 **2 WG/核时单核 DPAS 吞吐骤降
  至 35%**(835→290 dpas/µs),之后随 WG 数线性恢复(8 WG/核时 1071)。
  多 WG 共核时 XMX 存在某种仲裁惩罚,kernel 调度应避开 2 WG/核的点。

### 7.3 保留意见

以上为间接证据,非证伪:缺 VTune/zet 硬件计数器(本机 xpu-smi 的内存读计数器
返回 0,EU active 为 N/A),无法直接读出 L2 命中率或寄存器分配配置。
最终确认需 Intel BMG-G31 PRM 或 VTune GPU 计数器。

## 附录:复现

```bash
sudo xpu-smi config -d 0 -t 0 --frequencyrange 2400,2400
icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31 -O3 -std=c++17 -o bench_esimd_dpas bench_esimd_dpas.cpp
./bench_esimd_dpas 16384 2048 0   # ILP 扫描, 8 线程/EU
./bench_esimd_dpas 16384 2048 1   # repeat 扫描
sudo xpu-smi config -d 0 -t 0 --frequencyrange 400,2800
```

数据:`results/b70/b70_xmx_true_rate.csv`(coop-matrix 路径)、
`results/b70/` 其余 CSV(延迟/带宽类);ESIMD 原始输出见附录 A(下方)。

### 附录 A:ESIMD 裸 dpas 原始数据

```
# 2.4GHz, 2048 WIs (8 线程/EU), ILP 扫描:
ILP=1 121.2 TF   ILP=2 155.1 TF   ILP=3 156.6 TF
ILP=4 156.9 TF   ILP=6 157.1 TF   ILP=8 157.2 TF
# 2.4GHz, 256 WIs (1 线程/EU):
ILP=1 46.3 TF    ILP=2 92.6 TF    ILP=4 121.1 TF   ILP=8 136.9 TF
# repeat 扫描 (2048 WIs, ILP=4):
R=1 100.2 TF     R=2 139.9 TF     R=4 156.3 TF     R=8 156.9 TF
# 2.8GHz, 2048 WIs:
ILP=2 181.0 TF   ILP=4 183.0 TF   ILP=8 183.4 TF
```
