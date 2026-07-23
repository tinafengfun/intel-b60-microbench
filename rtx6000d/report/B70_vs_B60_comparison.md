# Intel Arc Pro B70 (BMG-G31) vs Arc Pro B60 (BMG-G21) 微基准对比报告

B60 侧数据来自本仓库 `REPORT.md`(完整微基准套件实测);
B70 侧为同一套微基准(本目录 `results/b70/`)在同频(2.40 GHz 锁频)下复现的实测结果。
两侧 cycles 直接可比;方法学、脚本、测试参数完全一致。

> 测试环境:B70 机器 172.16.124.12(Alder Lake i7-12700 主机),独显
> `Intel Graphics [0xe223]` = BMG-G31,驱动 Level-Zero V2 1.13.35563,oneAPI 2025.3。
> 所有 `bmg-g21` 编译目标已改为 `bmg-g31`(ocloc 原生支持)。

## 0. 关键方法学教训:xe 驱动 DVFS 必须锁频

**这是本次复现踩到的最大的坑,也是任何在 Intel Xe 平台跑微基准的前置条件。**

BMG 上的 xe 驱动 DVFS 非常激进:空闲时核心频率降到 **517 MHz**,微内核(几十 µs 级)
跑完都来不及升频,首轮全部数据作废(DPAS 延迟 22 µs、全机峰值 18 TFLOPS,均为假象)。

```bash
# 测试前锁频(实测 xpu-smi 读回 2400MHz 稳定)
sudo xpu-smi config -d 0 -t 0 --frequencyrange 2400,2400
# 测试后恢复默认
sudo xpu-smi config -d 0 -t 0 --frequencyrange 400,2800
```

注意:`sysfs` 的 `gpu_frequency_request` 读回不准(显示 400 不影响 xpu-smi clamp 生效),
以 `xpu-smi dump` 的实际频率为准。本报告 B70 全部数据均为锁频 @2.40 GHz 下测得。

## 1. 规格与实测峰值

| 维度 | B70 (BMG-G31) | B60 (BMG-G21) | B70 / B60 |
|---|---|---|---|
| 计算单元 | 32 Xe core(256 EU) | 20 Xe core(160 EU) | 1.6× |
| XMX 矩阵单元 | 256 | 160 | 1.6× |
| 测试时钟(锁频) | 2.40 GHz | 2.40 GHz | 1.0× |
| 最大时钟 | 2.80 GHz | 2.40 GHz | 1.17× |
| **全机 DPAS BF16/FP16 峰值** | **87.6 TFLOPS** | **97.66 TFLOPS** | **0.90×** |
| **每 XMX 吞吐** | **0.342 TF** | **0.610 TF** | **0.56×** |
| 显存带宽(float4 读, 2GB) | **665 GB/s** | **538 GB/s** | **1.24×** |
| 显存带宽(scalar 读) | 255 GB/s | 204 GB/s | 1.25× |
| kernel dispatch | 6.9 µs + 38–40 ns/WG | 3.7 µs + 40 ns/WG | 更差 |

## 2. 头号发现:B70 每个 XMX 只有 B60 的 56% 吞吐

B70 核心规模是 B60 的 1.6×(256 vs 160 EU),但全机 DPAS 峰值反而低 10%:

- 单 EU 微基准:依赖链延迟 ~30–33 cyc/dpas(B60 34.4),**ILP=16 互逆吞吐 18.36 cyc/dpas**(B60 15.9)——单 EU 层面 B70 的 XMX 互逆吞吐已经慢了 ~15%。
- 全机扩展:2048 SGs(256 EU × 8 threads)各种 SG/WG 配比全部收敛在 **87.4–87.6 TFLOPS**,折合每 EU 每时钟 142.6 FLOP/cyc(B60 254.5),**恰好是 B60 的 56%**。
- 单 EU 互逆(18.36 cyc)与全机等效(28.7 cyc)之间的差距说明存在 **die/chip 级共享瓶颈**(XMX 阵列的共享前端、或 Xe core 间矩阵数据通路仲裁),而非单纯频率或 EU 内问题。
- 已排除降频:满载时 xpu-smi 显示 2400 MHz 稳定,功耗仅 ~80W(burst 上限 230W),离功耗墙很远。
- 推定:BMG-G31 的 XMX 单元按每 EU 半速设计(或两组 EU 共享一个 XMX 前端)。
  G31 面向更大显存/更高带宽定位,矩阵单元面积被让渡给了内存子系统(见第 4 节带宽高 24%)。

**对 LLM 推理的含义:在矩阵受限(prefill/大 batch)场景,B70 的矩阵算力不升反降,
规模红利被每单元降速吃掉;在带宽受限(decode)场景,B70 反而更优。**

## 3. 延迟与微架构:与 B60 几乎完全同构

指针追逐(cycles,@2.4GHz 两侧同频直接可比):

| 层级 | B70 | B60 | 差异 |
|---|---|---|---|
| L1 | 71.3 | 70.8 | 同 |
| 128KB 处 | 147.1 | 144.5 | 同 |
| L2(192KB) | 170.9 | 162.4 | +5% |
| DRAM(128MB) | 270.6 | 260.7 | +4% |
| SLM(send.slm) | 47.3 起,64KB 处 119.4 | 46.1 / 116.6 | 同 |

L2 容量延迟曲线形态与 B60 一致(8MB 后渐变爬升),G31 的缓存层级组织未变。
其余微架构行为逐项复现:

- **barrier**:0.66–18 cyc(B60 2–11),同样"近似免费";
- **DPAS 停顿窗口吸收 ALU**:与 B60 同结论,dequant/scale 可与 DPAS 免费交织;
- **store/writeback/prefetch 开销**:数值与 B60 同量级;
- **INT8 DPAS**:同样触发 IGC segfault——这是编译器 bug,与设备无关,G31 上未修复。

## 4. 内存子系统:B70 带宽高 24%

SYCL `bench_bw_v3`(2GB 工作集,float4 读):B70 **665 GB/s** vs B60 538 GB/s。
G31 的显存配置(更宽总线或更高速率 GDDR6)真实兑现到了可读带宽,scalar 读也同比提升。
这与此前的定位判断一致:G31 把面积/功耗预算从 XMX 移向了内存子系统。

## 5. 调度异常:大量小 WG 串行化

`run_dpas_full_gpu.py`(1 SG/WG, ILP=8, 扫 WG 数):B60 随 WG 数线性扩展到 88.6 TF;
**B70 在 >32 WGs 后 kernel 时间线性增长,峰值仅 17.59 TF**——B70 的 WG 调度器
对大量单 SG 小 WG 存在串行化(推测与 G31 双 tile/更大 die 的 WG 分发结构有关)。

实际意义:在 B70 上应避免"每 WG 只含 1 个 subgroup"的 kernel 组织方式,
改用 8 SGs/WG(2048 SGs = 256 WGs × 8)即可拿到满配 87.6 TF,绕开该瓶颈。

## 6. 结论

1. **微架构同构**:G31 与 G21 同为 Xe2,L1/L2/SLM/DRAM 延迟、barrier、DPAS 流水线
   行为、交织特性全部复现,差异 ≤5%。B60 报告中的所有微架构结论可平移到 B70。
2. **矩阵单元降格**:B70 每 XMX 吞吐只有 B60 的 56%,全机 DPAS 峰值 87.6 vs 97.66 TFLOPS,
   存在单 EU 之外的芯片级共享瓶颈。矩阵受限负载(prefill)B70 不占优。
3. **带宽升级**:665 vs 538 GB/s(+24%),带宽受限负载(decode)B70 更优。
4. **调度器退化**:大量小 WG 场景严重串行化(17.6 vs 88.6 TF),kernel 组织需避开。
5. **方法学**:Xe 平台微基准第一步永远是 `xpu-smi --frequencyrange` 锁频,
   否则所有延迟/吞吐数据都是 DVFS 假象。

## 附录:复现步骤

```bash
# 1. 部署套件(B60 仓库),替换编译目标
sed -i 's/bmg-g21/bmg-g31/g' Makefile run_*.py
# 2. 锁频(见第 0 节)
sudo xpu-smi config -d 0 -t 0 --frequencyrange 2400,2400
# 3. 编译 + 全量跑
make && ./run_all.sh   # INT8 DPAS 会因 IGC segfault 失败,符合预期
# 4. 满配 DPAS 补测(2048 SGs 版本,需放在套件目录内跑)
python3 rts2048.py
# 5. 测完恢复频率
sudo xpu-smi config -d 0 -t 0 --frequencyrange 400,2800
```

数据位置:本目录 `results/b70/`(21 个 CSV);原始日志在 B70 机器 `~/b70-microbench/battery.log`。
