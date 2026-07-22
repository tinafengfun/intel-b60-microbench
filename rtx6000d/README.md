# NVIDIA RTX 6000D (Blackwell GB202, sm_120) Microbenchmark Suite

Workstation Blackwell 微基准评测,与 B200 (GB100, sm_100) 对比。方法学参考
*Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks* (arXiv:2507.10789, 2512.02189),
并与本仓库的 Intel Arc Pro B60 (Xe2) 套件互为对标(barrier / 启动开销 / tensor 停顿窗口交织 / shared 延迟等维度已交叉验证)。

- 报告:[report/RTX6000D_vs_B200_report.md](report/RTX6000D_vs_B200_report.md)(vs B200)
- 横向对比:[report/RTX6000D_vs_B60_comparison.md](report/RTX6000D_vs_B60_comparison.md)(vs 本仓库 Arc Pro B60/Xe2)
- 源码:`src/`(12 个 .cu + 4 个 python 框架级/端到端测试)
- 原始数据:`results/*.csv`、`results/e2e_vllm.json`
- 复现:`docker run --rm --network=none --gpus '"device=0"' -v $PWD:/work nvidia/cuda:13.0.1-devel-ubuntu22.04 bash /work/src/run_all.sh all`(编译需 `-gencode arch=compute_120a,code=sm_120a`)
