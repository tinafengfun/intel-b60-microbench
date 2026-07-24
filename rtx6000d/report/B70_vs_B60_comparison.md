# Intel Arc Pro B70 (BMG-G31) vs Arc Pro B60 (BMG-G21) 微基准对比报告

B60 侧数据来自本仓库 `REPORT.md`;B70 侧为同套微基准在同频(2.40 GHz 锁频)下的复现,
外加两轮针对 XMX/DPAS 的重新设计实验(cooperative-matrix 版 `results/b70/b70_xmx_true_rate.csv`、
裸 ESIMD dpas 版 `src/bench_esimd_dpas.cpp` 实测)。

> **v4 更正说明(2026-07-24)**:v3 的"coop-matrix 路径 119 TF(IGC 开销 -24%)"结论
> 再次被推翻——这次错在**记账口径**:旧 coop 微基准把每线程 dpas 的**阵列工作**
> (2048 FLOP/指令)当作有用 FLOP 记账。数值探针证实:KHR SG16 的 8×16×16 mad
> 集体只做 4096 有用 FLOP(每线程 256),但 IGC 把它 lowering 成每线程一整条
> dpas.8x8(N=8,2048 FLOP 阵列工作)——**阵列工作的 7/8 是冗余的**。
> naive 单 tile coop 路径真实有用吞吐 ≈ **15.5 TF**(2 WI/EU),而非 119 TF;
> 与 ESIMD 裸 dpas(157.2 TF)的真实差距是 **~8×(有用 FLOP/指令比)**,不是 24%。
> oneMKL bf16 GEMM 实测 153.7 TF(8192³),与 ESIMD 互证硬件真实水平。详见第 8 章。
>
> **v3 更正说明**:本报告经历两次自我纠错——
> v1("峰值 87.6 TF、每 XMX 只有 56%")错在 IGC 死代码消除(DCE);
> v2("发射极限 21.1 cyc/dpas、峰值 119 TF")的 kernel 已修正 DCE,但仍带
> cooperative-matrix 抽象层的代码生成开销;
> **v3 用 ESIMD 裸 dpas 指令(1:1 映射硬件)测得真实硬件极限:每 EU 16.0 cyc/dpas、
> 全机 157.2 TF @2.4GHz / 183.4 TF @2.8GHz。**(v3 的硬件极限数字仍然成立;
> 被更正的是它对 coop 路径差距的解读。)

> 测试环境:B70 机器 172.16.124.12(i7-12700 主机),独显 `Intel Graphics [0xe223]` = BMG-G31,
> 256 EU(32 Xe core),驱动 Level-Zero V2 1.13.35563,oneAPI 2025.3。

## 0. 方法学:三个陷阱(本次最大的教训)

1. **xe 驱动 DVFS**:空闲 517MHz,必须 `xpu-smi config --frequencyrange 2400,2400` 锁频
   (测完恢复 `400,2800`)。
2. **IGC 静默死代码消除**:吞吐 kernel 只 store 部分累加器时,其余 DPAS 链被删除且
   不报错。kernel 必须消费全部中间结果,并用"工作量 ×N ⇒ 时间 ×N"自校验。
3. **抽象层 lowering 决定了"有用 FLOP/指令"**:SPIR-V cooperative-matrix(KHR)单 tile
   路径的每条 dpas 只有 1/8 阵列工作是有用的(详见第 8 章),旧记账把阵列工作当有用,
   得出"慢 24%"的错误结论;真实差距是 ~8×。**测硬件极限必须用 ESIMD 裸 dpas
   (或等价底层路径,如 oneMKL JIT kernel)交叉验证**;评价 coop 路径必须按
   tile 数学(2·M·N·K)记有用 FLOP,并用数值探针验证正确性。另外 IGC 在 bmg-g31
   上还有三个 segfault:INT8 DPAS、coop-matrix 多 offset store、coop-matrix OpFAdd。

## 1. 规格与实测峰值(v3 终版)

| 维度 | B70 (BMG-G31) | B60 (BMG-G21) | B70 / B60 |
|---|---|---|---|
| 计算单元 | 32 Xe core(256 EU) | 20 Xe core(160 EU) | 1.6× |
| **XMX 发射速率(每 EU,裸 dpas)** | **16.0 cyc/dpas = 256 FLOP/cyc** | **~16.1 cyc/dpas = 254 FLOP/cyc** | **相同** |
| **全机 FP16/BF16 峰值(实测)** | **157.2 TF @2.4GHz**;**183.4 TF @2.8GHz** | 97.66 TF @2.4GHz | **1.61× / 1.88×** |
| **全机 INT8 峰值(ESIMD 裸 dpas)** | **314.4 TOPS @2.4GHz**(512 OP/cyc/EU,恰好 2× FP16);~367 TOPS @2.8GHz | 197 TOPS(spec) | 1.6× |
| oneMKL s8 GEMM(真实库) | **298.2 TOPS**(8192³,95% 峰值,C0 数值校验通过) | — | — |
| cooperative-matrix(naive 单 tile)有用吞吐 | **≈15.5 TF**(阵列工作口径曾记 119 TF,v4 更正) | 微基准同口径 ~12 TF;**register-blocked GEMM 89.77 TF 实测(2MNK,92% 峰值)** | B70 缺 256-GRF 模式,register-blocked 补救路径走不通(见第 8 章) |
| oneMKL bf16 GEMM(真实库) | **153.7 TF**(8192³,98% 峰值) | ~95 TF(oneDNN 估计) | 1.6×,与 EU 数一致 |
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

## 8. coop-matrix vs ESIMD:TFLOPS 差距的完整归因(v4 新增)

问题:同一块 B70,joint_matrix(cooperative-matrix)GEMM 路径与 ESIMD 裸 dpas 路径的
TFLOPS 为什么差这么多?是编译器、IR 分层、调度,还是硬件?

**结论:根源在编译器后端(IGC)把 CooperativeMatrixKHR lowering 到 XMX 指令的方式,
具体是"有用 FLOP/指令"比例;不是 ISA、不是线程调度、不是数据类型。**

### 8.1 三条独立证据链

**证据一:反汇编(ocloc disasm,bmg-g31)。** 两条路径的循环体都是纯 `dpas.8x8` 流,
无 mov 风暴、无多余 sync,SWSB 结构相同——**ISA 指令流不是差距来源**。差异在操作数:

| | coop KHR(`gen_kernel_live`, ILP=4) | ESIMD(`bench_esimd_dpas`, ILP=4) |
|---|---|---|
| 每链操作数 | **每链独立 A、B 副本**(~24 GRF/链:acc 8 + A 8 + B 8) | 4 链**共享**同一 A、B |
| acc 寄存器 | 8 GRF/链/线程(64 float 计算量,**仅 8 float 有用**) | 16 GRF/链/线程(128 float,**全有用**) |
| 每线程每指令 | N=8,2048 FLOP 阵列工作,**256 FLOP 有用** | N=16,4096 FLOP 阵列工作,**4096 FLOP 全有用** |

**证据二:数值探针(`num_probe.py`)。** A、B 全填 bf16 1.0,C 填 ramp:
kernel 每执行一次,输出 tile 每个元素恰好 +16.0(= 一次真实的 8×16×16 mad)。
证明 KHR mad 的集体语义 = 4096 有用 FLOP/ SG——lowering 没有"藏"额外有用计算,
每线程 64 float 的指令输出里只有 8 float 进入最终 tile,**冗余度 8×**。

**证据三:oneMKL 锚点(`mkl_gemm.cpp`)。** oneMKL bf16 GEMM 8192³ 实测
**153.7 TF**(2MNK/t,真实有用)= 256 FLOP/cyc/EU 的 98%。与 ESIMD 微基准
(157.2 TF)互证:**硬件没问题,接近峰值是可达的**——前提是每线程拥有私有
8×16 输出块,使每条 dpas 的阵列工作全部有用。

### 8.2 量化对照(2.4 GHz 锁频,有用 FLOP 口径)

| 路径 | 每指令有用 FLOP(每线程) | 实测有用 TF | 占硬件峰值 |
|---|---|---|---|
| ESIMD 裸 dpas(8 WI/EU, ILP≥2) | 4096(全有用) | **157.2** | ~100% |
| oneMKL bf16 GEMM 8192³ | (JIT kernel,全有用) | **153.7** | 98% |
| coop KHR naive 单 tile 链(2 WI/EU, ILP=6 最优) | 256(1/8 有用) | **15.5** | ~10% |
| coop KHR 理论天花板(阵列打满) | 256 | ~19.6 | 12.5% |
| (旧报告记账口径:阵列工作当有用) | 2048 | "119 TF" | 口径错误,作废 |

### 8.3 次要因素(真实存在,但不是主因)

1. **操作数逐链复制 → 寄存器压力**:IGC 给 coop 每条累加链独立复制 A/B tile,
   而 ESIMD 手写版共享。ESIMD 模拟"独立 B 副本"(bench mode 2)实测:
   1 WI/EU 时 20.8→25.3 cyc/dpas(**-18%**),8 WI/EU 时仅 -2.4%(多线程掩盖)。
2. **ILP 上限**:coop 每链 24 GRF → ILP=7 即 spill 崩塌(21→68 cyc/dpas);
   ESIMD 共享操作数,ILP=8 仍正常。coop 最优只到 ILP=6。
3. **bf16 vs fp16:已排除**。bench mode 3 实测 bf16 与 fp16 完全同速
   (121.1 TF @ 1 WI/EU ILP=4,两者一致);coop asm 的 `:bf` 操作数不是性能因素。

### 8.4 为什么 B60 的 joint_matrix GEMM 能到 92% 峰值,而 B70 不行?

B60 报告中的 89.77 TF GEMM(`gemm_v20_best.cpp`)是真实有用吞吐(2MNK 实测),
关键在两点:

- **Register blocking**:每个 SG 拥有 4×4 = 16 个 8×16 tile(32×64 输出块),
  IGC 可以把 tile 重打包,使每线程的每条 dpas 落在一个完整有用的 8×8 块上
  (16 线程 × 2 块 × 64 float = 2048 = 32×64,零冗余)。
- **256-GRF 模式**(`-cl-intel-256-GRF-per-thread`):寄存器预算翻倍,16 个
  acc tile + 共享 A/B 才放得下。

在 B70 上复现该结构(`coop_blocked.py`):**rb=4(4×4)直接 spill**
(spillMemSize=3072,有用吞吐反而掉到 3.5 TF);rb=2 为 11.8 TF,不比 naive 好。
原因与第 7.2 节的 VRT 探测一致:**B70 的 128 GRF/线程是硬顶,256-GRF 模式在
bmg-g31 上不可用**,register-blocked coop 补救路径目前走不通。
(B60 微基准的"97.66 TF 原始 XMX 吞吐"与 B70 的"119 TF"是同一记账口径,
同样应视为阵列工作口径;B60 的 89.77 TF GEMM 是 2MNK 实测,真实有效。)

### 8.5 实践建议

1. B70 上要打满 XMX:用 **ESIMD 裸 dpas**(或 oneMKL),每线程私有 8×16 acc tile、
   多链共享 A/B 操作数,8 线程/EU + ILP≥2 即饱和(157 TF)。
2. 不要对 bmg-g31 的 naive KHR joint_matrix 路径抱性能预期(有用吞吐 ~1/8);
   需要 cooperative matrix 语义时,关注 IGC 后端是否修复 per-thread 打包。
3. 评价任何 XMX 微基准:先分清"阵列工作口径"与"有用 FLOP 口径",
   再用数值探针验证 kernel 真的在做你以为的计算。

### 8.6 INT8 峰值与 Level-Zero SPIR-V 路径(2026-07-24 补测)

**问题**:能否从 Level-Zero 的 SPIR-V 接口(`zeModuleCreate` + `ZE_MODULE_FORMAT_IL`)
角度榨干算力,做 INT8 GEMM 峰值测试?

**答案:API 只是投递方式,JIT 仍是同一个 IGC——榨干算力的关键仍是 lowering 路径选择,
不是从哪个 API 入口加载。** 本项目的 SPIR-V 流程(spirv-as → ocloc 离线编译 →
L0 `zeModuleCreate` 加载 native binary)与直接给 L0 喂 SPIR-V IL(驱动内 JIT)在
代码生成上完全等价。INT8 的具体障碍:bmg-g31 上 **coop-matrix INT8 DPAS 会 segfault
IGC**(第 0 章三个 IGC 崩溃之一),SPIR-V KHR 路径走不通;**ESIMD 裸 dpas 支持
u8/s8,是唯一干净路径**。

**实测结果(2.4 GHz 锁频,`bench_esimd_dpas` mode 4,`xmx::dpas<8,8,int>` + u8)**:

| 配置 | TOPS | 说明 |
|---|---|---|
| 8 WI/EU, ILP=8 | **314.4** | = 512 OP/cyc/EU,**恰好 2× FP16 峰值**,硬件 INT8 满速 |
| 8 WI/EU, ILP=4 | 313.8 | 同 fp16,ILP=2 即接近饱和(310.2) |
| 1 WI/EU, ILP=8 | 273.9 | 与 fp16 的 136.9 TF 严格 2× 对应 |

每指令耗时与 fp16 **逐点相同**(53.36 ns/instr @8WI/EU)——XMX 在 INT8 下每周期做
2× 操作(K=32 vs 16),发射速率不变。数值校验:c0 累积值 = 7 次 repeat × 16384 迭代
× K=32 全 1 乘加,分毫不差(顺带解释了此前 fp16 c0 的"7×"——跨 repeat 累积,
不是计算错误)。

**oneMKL s8 GEMM 锚点**(`mkl_gemm_s8.cpp`,gemm_s8s8s32):8192³ 实测
**298.2 TOPS**(94.8% 峰值),C0 = 16384 数值正确。

**结论**:B70 INT8 算力 = **314 TOPS @2.4GHz / ~367 TOPS @2.8GHz**(线性外推),
是 FP16/BF16 的严格 2×;INT8 GEMM 用 oneMKL 即可到 95% 峰值,无需手写 kernel。

### 8.7 复现

```bash
# ESIMD 对照(bench_esimd_dpas.cpp,新 mode 2/3)
./bench_esimd_dpas 16384 256 2    # 独立 B 副本 ILP=4 -> 99.5 TF (vs 共享 121.1)
./bench_esimd_dpas 16384 2048 2   # 8WI/EU            -> 153.2 TF (vs 共享 156.9)
./bench_esimd_dpas 16384 256 3    # bf16 对照          -> 121.1 TF (= fp16)
./bench_esimd_dpas 16384 2048 4   # INT8 u8 ILP 扫描   -> 314.4 TOPS @ILP=8
# coop 干净对拍(bench_coop_dpas_v2.cpp 需 oneAPI ≤2025.2 运行时;SPIR-V 路径用 coop_8wi.py)
python3 coop_8wi.py               # gen_kernel_live ILP×occupancy 扫描
python3 coop_blocked.py           # register-blocked rb=2/4(验证 8.4)
python3 num_probe.py              # 数值探针:每次运行输出 tile +16.0
./mkl_gemm 8192                   # oneMKL bf16 锚点 -> 153.7 TF
./mkl_gemm_s8 8192                # oneMKL s8 锚点  -> 298.2 TOPS
```

数据:`results/b70/coop_vs_esimd_v4.csv`。

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
