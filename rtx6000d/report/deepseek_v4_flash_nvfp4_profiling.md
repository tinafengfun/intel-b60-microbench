# DeepSeek-V4-Flash NVFP4 vLLM 算子级 Profiling 报告(Prefill / Decode 拆分)

测试日期:2026-08-06
测试平台:NVIDIA RTX PRO 5000 Blackwell 48GB(sm_120,节点 10.239.11.161,单卡 GPU0)
模型:`DeepSeek-V4-Flash-0731-NVFP4`(43 层 → 裁剪为前 2 层,TP=1,关闭 MTP 投机)
推理栈:vLLM 0.11.2.dev279(b12x fork,cu132)+ `B12X_MLA_SPARSE` attention + b12x MoE/Linear backend + KV cache FP8
Profiler:vLLM `--profiler-config`(torch profiler,with_stack + record_shapes),`/start_profile` ~ `/stop_profile` 窗口

---

## 0. 测试方法

### 0.1 为什么减层 + 如何减层

生产服务(172.16.120.54)8 卡全满,无法直接 profile。为在单张 48GB 卡上复现,采用**权重裁剪**方式构建 2 层模型:

- 保留 `embed.weight`、`layers.0.*`、`layers.1.*`、末尾 `norm.weight`、`head.weight`、`hc_head_*`,丢弃 `layers.2..42` 与全部 `mtp.*` 张量
- `config.json`:`num_hidden_layers 43→2`,`compress_ratios` 同步截断;`hf_quant_config.json` 的 `quantized_layers` 只保留 layers.0/1
- 裁剪脚本:`src/trim_dsv4.py`;产物 9.0 GiB(6208 张量,原 138365 张量 / 163.5 GiB)
- 层 0/1 是 `compress_ratios=0` 的全注意力层(无 KV compressor),结构与原模型完全一致,每层的算子构成与深层一致(MoE + MLA + MHC 超连接),因此**每层/每步的相对 breakdown 对全模型有代表性**;但 lm_head、embedding、采样器等"每步固定开销"在 2 层模型里占比被人为放大,报告中凡涉及全模型结论均做 43 层外推并明确标注

### 0.2 两个 profiling 场景

| 场景 | 输入 | 输出 | 目的 |
|---|---|---|---|
| Prefill | 16 token | 1 token | 观察 prefill 步算子构成 |
| Decode | 1 token | 16 token | 观察 16 个 decode 步的稳态算子构成 |

两个场景分别在独立的 profiler 窗口内采集,trace 文件存于 `results/dsv4_prof/`。

### 0.3 复现要点(踩坑记录)

- 该 fork 不支持 `VLLM_TORCH_PROFILER_DIR` 环境变量,必须用 `--profiler-config '{"profiler":"torch","torch_profiler_dir":...}'`
- 必须复刻生产环境的 b12x 开关,否则 attention o_proj 会走 DeepGEMM `fp8_einsum` 路径并在 sm120 上断言(`layout.hpp: t.dim()==N`)失败:`VLLM_USE_B12X_WO_PROJECTION=1`、`VLLM_USE_B12X_FP8_GEMM=1`、`VLLM_USE_B12X_MOE=1`、`VLLM_USE_B12X_MHC=1`、`VLLM_USE_B12X_SPARSE_INDEXER=1`、`B12X_MLA_SM120_UNIFIED=1` 等(完整列表见 `src/inner_prof.sh` 与 §0.4)
- 启动脚本:`src/inner_prof.sh`(TP=1);采集客户端:`src/prof_client.py`;解析:`src/parse_trace2.py`

---

## 1. 模型与量化结构(静态分析)

### 1.1 DeepSeek-V4-Flash 关键配置

- 43 层,hidden 4096;MLA 变体:`q_lora_rank=1024`、`o_lora_rank=1024`、`head_dim=512`、`qk_rope_head_dim=64`、`num_attention_heads=64`、`o_groups=8`
- MoE:256 routed experts + 1 shared expert,每 token 选 6 个,`moe_intermediate_size=2048`,gate 为 `sqrtsoftplus` 打分 + `noaux_tc`
- 稀疏注意力:层 2~42 带 KV compressor(`compress_ratios` 4/128 交替)+ indexer(`index_topk=512`);层 0/1 为全注意力
- MHC(multi-hyper-connection):每层 `hc_attn_*`/`hc_ffn_*` + 共享 `hc_head_*`,TileLang 融合 kernel
- MTP:`num_nextn_predict_layers=1`(dspark,3 个 MTP 子层),本次 profiling 关闭

### 1.2 精度分布(全模型字节级统计,`src/byte_breakdown.py`)

| 模块 | 精度 | 字节 | 占比 |
|---|---|---|---|
| **MoE routed experts 权重** | **NVFP4 (e2m1, group=16, ue8m0 scale)** | **129.00 GiB** | **78.9%** |
| **MoE experts ue8m0 scales** | UE8M0 | **16.13 GiB** | **9.9%** |
| MTP(dspark) | 混合(fp8 为主) | 10.12 GiB | 6.2% |
| attention 投影 (wq_a/wq_b/wkv/wo_a/wo_b) | FP8 e4m3 + ue8m0 | 5.02 GiB | 3.1% |
| shared expert (w1/w2/w3) | FP8 e4m3 | 1.01 GiB | 0.6% |
| embed + lm_head | **BF16** | 1.97 GiB | 1.2% |
| gate / hyperconn / norm 等 | BF16/FP32 | 0.23 GiB | 0.1% |
| **合计** | | **163.48 GiB** | 100% |

**NVFP4 只用于 MoE routed experts 的 w1/w2/w3**(`hf_quant_config`:仅 `layers.N.ffn.experts` 为 NVFP4;`*.attn.*`、`shared_experts`、`head`、`mtp.*` 在 ignore 列表)。注意力投影与共享专家是 FP8,embedding/lm_head/gate 是 BF16。NVFP4 部分合计 145.1 GiB = **88.8% 的模型字节**。

---

## 2. Prefill 拆解(16 token 输入 / 1 token 输出)

GPU busy **3.51 ms**(kernel 时间戳跨度 46 ms,wall 55 ms —— 16 token 小 prefill 是 CPU/launch overhead 主导,GPU 占用率仅 ~6%,详见 §5)。

### 2.1 nn 层 → 子算子 breakdown(GPU busy 口径)

| nn 模块 | 子算子 / kernel | 时间 | 占比 | 每层 |
|---|---|---|---|---|
| **MoE(NVFP4)** | `b12xmoefusedsiluMoEDynamicKernelSilu`(NVFP4 GEMM×3 + SiLU 融合,2 次调用) | **1.973 ms** | **56.2%** | 986 µs |
| **lm_head** | cuBLAS `gemvx` bf16(129280×4096,1 次) | 0.887 ms | 25.3% | — |
| attention 投影(FP8) | `b12xgemmdenseDenseGemmKernel`(wq_a/wq_b/wkv/wo_a/wo_b,fp8e4m3) | 0.311 ms(含共享专家) | 8.9% | 156 µs |
| MHC 超连接 | `b12xintegrationresidual` Pre/Post/Finalize + `hc_head_fuse_tilelang` | 0.073 ms | 2.1% | 36 µs |
| 量化/逆 RoPE | `_quantize_dense_tk_to_tk`、`_quantize_attention_inv_rope_to_tdg`(激活 FP8 量化) | 0.072 ms | 2.1% | 36 µs |
| MLA 注意力核心 | `b12xattentionmlaprefill_mgUnifiedPrefillMGKernel` | 0.044 ms | 1.2% | 22 µs |
| router | `topkGatingSoftplusSqrt` + gate GEMV(bf16) | 0.017 ms | 0.5% | 8 µs |
| embedding / sampler / 拷贝填充等 | | 0.125 ms | 3.6% | |

### 2.2 观察

1. **16-token prefill 的 MoE 仍然是显存带宽 bound**:NVFP4 MoE kernel 每层 986 µs。16 token × 6 expert = 96 个 expert 指派,去重后需读取 ~96 个 expert 的 FP4 权重 ≈ 0.9 GB,折合有效带宽 ~0.9 TB/s —— 即使 prefill 也在"读权重"而不是"算矩阵"。
2. Prefill 与 decode 使用**不同的 NVFP4 MoE kernel**:prefill 是 `MoEDynamicKernelSilu`(动态分组 GEMM),decode 是 `MoEMicroKernelSilu`(微批优化)。
3. 注意力核心(MLA prefill kernel)占比极小(1.2%)—— 16 token 上下文太短,注意力计算可忽略;投影 GEMM 才是 attention 块的开销大头。
4. 激活 FP8 量化/逆 RoPE 是独立可见的开销(2.1%),这是 FP8 attention 投影的"过路费"。

---

## 3. Decode 拆解(1 token 输入 / 16 token 输出)

16 步合计 GPU busy **27.30 ms**(每步 **1.71 ms** busy / 3.6 ms span / wall 4.6 ms,GPU 占用率 ~37%,详见 §5)。

### 3.1 nn 层 → 子算子 breakdown(每步)

| nn 模块 | 子算子 / kernel | 每步时间 | 占比 | 每层每步 |
|---|---|---|---|---|
| **lm_head** | cuBLAS `gemvx` bf16(129280×4096,1.06 GB 权重读) | **885.8 µs** | **51.9%** | — |
| attention 投影 + shared expert(FP8) | `b12xgemmdenseDenseGemmKernel` ×5 变体(2048=shared,1024/8192/4096×2=attn 投影) | 363.4 µs | 21.3% | 181.7 µs(shared 62 + attn 120) |
| **MoE(NVFP4)** | `b12xmoefusedsiluMoEMicroKernelSilu`(2 次/步) | **229.1 µs** | **13.4%** | **114.6 µs** |
| mem/拷贝/填充 | memcpy32_post、FillFunctor 等(部分为 graph replay 内部) | 52.1 µs | 3.1% | 26.1 µs |
| MHC 超连接 | `b12xintegrationresidual` + `hc_head_fuse_tilelang` | 37.8 µs | 2.2% | 18.9 µs |
| MLA 注意力核心 | `pertok_b12xattentionmlakernelUnifiedDecodeKernel` + `mergeSparseMLASplitDecodeSinkMerge` | 33.0 µs | 1.9% | 16.5 µs |
| elementwise / 量化 / router / sampler / embed | | ~140 µs | 8.2% | |

### 3.2 观察

1. **decode 的第一大头是 lm_head 的 BF16 GEMV**(51.9%):每步读 1.06 GB bf16 权重,实测有效带宽 ≈1.2 TB/s,已接近 GDDR7 峰值 —— kernel 本身已优化到位,**问题是 lm_head 没量化**(见 §6)。注意:这是 2 层模型的放大效应,43 层外推后 lm_head 占 ~5%(见 §4.2),但它依然是"单步固定税"。
2. **MoE NVFP4 kernel 每层每步 114.6 µs**:每 token 只激活 6 个 expert,读 ~57 MB FP4 权重+scale,折合 ~0.5 TB/s。与 prefill 的 986 µs/层 对比:token 数 16×、读字节 ~16×、耗时 8.6× —— MoE 全程带宽 bound。
3. 注意力核心(MLA decode + merge)每层仅 16.5 µs(1.9%):短上下文 + KV cache FP8,注意力不是瓶颈;瓶颈在投影 GEMM(每层 120 µs)。
4. **通信为零**:TP=1 无 NCCL/allreduce。生产 TP=2 会在每层 attention 输出与 MoE 输出各加一次 allreduce(payload 仅 16 token×4096×2B=128KB 级,PCIe P2P 下预计每层几 µs~十几 µs),不改变本报告的占比排序。

---

## 4. NVFP4 涉及运算与占比(Prefill / Decode 拆分)

### 4.1 NVFP4 涉及的运算清单

NVFP4(e2m1 + ue8m0 group-16 scale)**只出现在 MoE routed experts 的 w1/w2/w3 GEMM**,且与 SiLU·mul 融合为单个 kernel:

| 场景 | kernel | 说明 |
|---|---|---|
| Prefill | `b12xmoefusedsiluMoEDynamicKernelSilu` | 动态分组 NVFP4 GEMM + SiLU 融合 |
| Decode | `b12xmoefusedsiluMoEMicroKernelSilu` | 微批 NVFP4 GEMM + SiLU 融合 |

其余所有计算都不是 NVFP4:attention 五组投影与 shared expert 是 FP8 e4m3(带 ue8m0 scale),KV cache 是 FP8,embedding/lm_head/gate/MHC/采样是 BF16/FP32,另有独立的激活 FP8 量化 kernel(`_quantize_*`)与逆 RoPE 量化 kernel。

### 4.2 NVFP4 占总计算的比重

| 口径 | Prefill (16 tok) | Decode (每步) |
|---|---|---|
| **GPU busy 时间占比(实测,2 层模型)** | **56.2%** | **13.4%**(lm_head 被放大) |
| GPU busy 时间占比(43 层外推*) | ~50% | **~27%** |
| FLOPs 占比(43 层解析估算**) | ~51% | ~51% |
| 模型字节占比(静态) | 88.8% | 88.8% |
| 每 token 激活权重读字节占比(43 层解析) | — | ~55%(57 MB NVFP4 / ~104 MB 总权重读/层) |

\* 43 层外推:每层每步 decode 成本 ~400 µs(MoE 114.6 + FP8 投影 181.7 + MHC/注意力/量化/杂项 ~104),43 层 ≈ 17.2 ms + lm_head 0.886 ms ≈ 18.1 ms/步 → 单流 decode 约 55 tok/s 量级;NVFP4 MoE = 43×114.6/18.1ms ≈ 27%,lm_head 降至 ~5%。
\*\* 每 token FLOPs:routed experts 6×3×2×4096×2048 = 302 MFLOP;attention 投影合计 ~214 MFLOP;shared expert 50 MFLOP;lm_head 1060 MFLOP(43 层摊薄后仅占 4%)。NVFP4 FLOPs 占比 = 43×302 / 25.4 GFLOP ≈ 51%。

**结论:无论 prefill 还是 decode,MoE 的 NVFP4 GEMM 都是模型里最大的单一计算/显存税源(FLOPs ~51%、权重字节 89%);但在 2 层实测 decode 中它被 BF16 lm_head 的带宽税盖过 —— 量化覆盖率(89%)与实际时间占比(27%)之间的差,主要来自 NVFP4 kernel 已接近带宽极限、而 BF16 模块没有量化。**

---

## 5. 计算 vs 通信 vs  overhead:谁占大头

| 成分 | Prefill 16/1 | Decode 1/16 |
|---|---|---|
| GPU 计算(busy) | 3.51 ms | 1.71 ms/步 |
| GPU 空闲(span 内 gap,CPU launch/调度) | ~42.5 ms(92%) | ~1.9 ms/步(53%) |
| 通信(NCCL) | 0(TP=1) | 0(TP=1) |
| wall | 55 ms | 4.6 ms/步 |

- **小 batch 下 vLLM 的 CPU 侧开销(scheduler、prepare_inputs、sample、python overhead)是 wall time 的第一大头**,尤其是 16-token prefill(GPU 占用 6%)。这是 2 层小模型 + batch=1 的固有属性;43 层时 GPU busy/步会到 ~18 ms,CPU 开销被摊薄到 <20%。
- decode trace 里 `vllm:v2/target/decode/full_graph_replay`(CUDA graph 回放)本身 506 µs/步,而 `logits`(lm_head)在 graph 外单独计 886 µs/步 —— lm_head 未被纳入 decode CUDA graph,每步单独 launch + 读 1.06 GB,是最值得优化的单点。
- 通信:本测试按设计要求单卡排除互连;生产 TP=2 的 allreduce 量级见 §3.2.4。

---

## 6. 用 vs 不用 NVFP4:计算量与显存差异及收益

### 6.1 显存(全模型静态)

| 方案 | MoE experts 字节 | 全模型字节 | vs NVFP4 |
|---|---|---|---|
| **NVFP4(现状)** | **145.1 GiB**(129.0 权重 + 16.1 scale) | **163.5 GiB** | 1.0× |
| FP8 等效 | ~258 GiB | ~277 GiB | 1.7× |
| BF16 等效 | ~516 GiB | ~545 GiB | **3.3×** |

NVFP4 让 163 GiB 的模型可以 2×85GB(RTX 6000D)装下;BF16 需要 ~7 张同样的卡。

### 6.2 Decode 每 token 权重读带宽(43 层,带宽 bound 部分)

| 模块 | NVFP4 | FP8 | BF16 |
|---|---|---|---|
| routed experts(6 激活)/层 | 57 MB | 100 MB | 201 MB |
| shared expert + attn 投影/层 | 25 MB(FP8 不变) | 25 MB | 50 MB |
| lm_head/步 | 1060 MB(BF16 不变) | 1060→530 MB | 1060 MB |

- MoE 部分 NVFP4 vs BF16 每层每 token 省 **3.5×** 带宽;实测 MoE kernel 114.6 µs/层 对应 ~0.5 TB/s,若回退 BF16 将升至 ~400 µs/层,43 层 decode 每步从 ~4.9 ms 涨到 ~17 ms —— **仅 MoE 一项,NVFP4 就决定了 decode 是可交互(55 tok/s 级)还是不可用(15 tok/s 级)**。
- NVFP4 vs FP8 再省一半,但在本模型上的边际收益小于"FP8 vs BF16"那一步。

### 6.3 计算量(FLOPs)

FLOPs 本身不随量化变化(同样的 GEMM 形状),变化的是**单位 FLOPs 的成本**:sm_120 tensor core 上 NVFP4 峰值 ≈ 2× FP8 ≈ 4× BF16。本测试的两个场景(16 token prefill、1 token decode)MoE 都是带宽 bound,FLOPs 速率优势体现不出来;**NVFP4 的算力收益只在大 batch prefill(expert 分组 GEMM 打满 tensor core)时兑现**,小 batch 下收益 100% 来自显存/带宽。

### 6.4 代价

- 激活需要动态 FP8/FP4 量化:`_quantize_*` kernel 占 GPU busy 1.2~2.1%,属于固定过路费。
- ue8m0 scale 增加 12.5% 的 MoE 字节开销(16.1/129)。
- 精度:group=16 细粒度 scale 是 NVFP4 可用性的关键(对比 group 更大或整 tensor scale 的方案),vLLM 侧用 `sqrtsoftplus` gate + ue8m0 已工程化。

---

## 7. 结论与 Insights

1. **NVFP4 的覆盖面极窄但极准**:只覆盖 MoE routed experts(89% 字节、51% FLOPs),就决定了整卡显存可行性与 decode 吞吐。注意力投影走 FP8、lm_head 走 BF16 是当前的"精度阶梯"。
2. **下一步最大的优化目标不是更多 NVFP4,而是 BF16 lm_head**:2 层实测它吃掉了 decode 52% 的 GPU 时间(43 层摊薄后 ~5%),1.06 GB/步已跑满 1.2 TB/s —— 唯一的出路是量化(lm_head FP8/NVFP4)或 vocab 分片/稀疏化。它甚至没有被包进 decode CUDA graph。
3. **Prefill 与 decode 的 NVFP4 时间占比差异(56% vs 13~27%)本质上是"lm_head/固定开销摊薄程度"的差异**,而不是 MoE 行为差异:MoE 在两个场景下都是带宽 bound,耗时与"激活 expert 数 × 单 expert 字节"成正比(实测 96 experts→986 µs vs 6 experts→114.6 µs)。
4. 小 batch 场景下 vLLM CPU 侧开销(scheduler/prepare_inputs/sample)超过 GPU busy,prefill 尤其严重(92% gap);生产部署应靠 CUDA graph(prefill 目前 eager)+ 连续批处理掩盖。
5. 对硬件的启示:该 workload 里 GDDR 带宽是唯一贯穿始终的瓶颈(MoE、lm_head 都打满),NVFP4 本质是"带宽放大器";tensor core 的 FP4 算力在小 batch 推理中是过剩的。

---

## 附录

- trace 文件:`results/dsv4_prof/*.pt.trace.json.gz`(prefill/decode 各一份 rank0 + async_llm 前端)
- 字节 breakdown:`src/byte_breakdown.py`(输出见 §1.2)
- 裁剪:`src/trim_dsv4.py`;服务脚本:`src/inner_prof.sh`;采集:`src/prof_client.py`;解析:`src/parse_trace2.py`
- 原始模型:172.16.120.54 `/home/sdf/disk/DeepSeek-V4-Flash-0731-NVFP4`;裁剪模型:10.239.11.161 `/root/models/DeepSeek-V4-Flash-2L-NVFP4`
- 环境约束:120.54 八卡被 GLM-5.2 + DSV4 生产服务占满(各卡剩 ~1 GB),故全部分析在 10.239.11.161 GPU0 完成,未影响任何现有服务
