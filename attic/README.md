# Attic — 仅供考古参考,请勿直接使用

本目录文件是 B70 XMX 调查过程中的**中间版本**,含有已知缺陷:

- `run_b70_xmx_probe.py` / `run_b70_xmx_verify.py` / `run_b70_xmx_issue_limit.py`:
  kernel 只 store 第 0 个累加器,IGC 死代码消除(DCE)会把其余 DPAS 链删除,
  测出的"吞吐"全是单链延迟假象。正确版本见仓库根目录 `run_b70_xmx_true.py`
  (store 全部累加器)。分析过程见 `rtx6000d/report/B70_vs_B60_comparison.md` 第 0 节。
- `bench_vrt_probe.cpp`: VRT 探针 v1,IGC 将寄存器 payload 重物化为内存读取,
  测不到真实寄存器压力。由 `bench_vrt_probe2.cpp`(根目录)取代——但 v2 同样
  被编译器击败(num_regs 恒为 3),仅存档说明该路径不可行。
