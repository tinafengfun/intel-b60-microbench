#!/usr/bin/env python3
"""Vector-op time share vs context length S (prefill, per layer per token).

Models (from vector_vs_matrix_analysis.md §7 derivations):
  Qwen3-30B-A3B : matrix = 114e6 + 8192*S FLOP ; vector = 0.12e6 + 64*S ord + 16*S exp
  Qwen3-235B    : matrix = 446e6 + 16384*S FLOP; vector = 0.23e6 + 128*S ord + 32*S exp

Platform rates (measured in this repo's microbenchmarks):
  B70     : fp32 ord 9.83 Tops/s, exp 1.84 Tops/s, XMX bf16 157 TF peak / 100 TF practical
  RTX5000 : fp32 ord ~32.9 Tops/s (FMA 65.7 TF / 2), MUFU exp 4.34 Tops/s,
            tensor bf16 289 TF peak / ~200 TF practical (69%, typical large-GEMM eff.)
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = {
    "Qwen3-30B-A3B": dict(m0=114e6, mS=8192.0, v0=0.12e6, vS=64.0, eS=16.0),
    "Qwen3-235B-A22B": dict(m0=446e6, mS=16384.0, v0=0.23e6, vS=128.0, eS=32.0),
}
PLATFORMS = {
    "B70": dict(ord=9.83e12, exp=1.84e12, mx_peak=157e12, mx_prac=100e12),
    "RTX5000": dict(ord=32.9e12, exp=4.34e12, mx_peak=289e12, mx_prac=200e12),
}

S = np.logspace(3, np.log10(256e3), 64)

def share(m, p, mx):
    mat = m["m0"] + m["mS"] * S
    vec_ord = m["v0"] + m["vS"] * S
    vec_exp = m["eS"] * S
    t_mat = mat / mx
    t_vec = vec_ord / p["ord"] + vec_exp / p["exp"]
    return 100.0 * t_vec / (t_mat + t_vec)

rows = []
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
styles = {"Qwen3-30B-A3B": "-", "Qwen3-235B-A22B": "--"}
colors = {"B70": "tab:red", "RTX5000": "tab:blue"}

for ax, mode in zip(axes, ["peak", "prac"]):
    for mn, m in MODELS.items():
        for pn, p in PLATFORMS.items():
            mx = p["mx_peak"] if mode == "peak" else p["mx_prac"]
            sh = share(m, p, mx)
            for s, v in zip(S, sh):
                rows.append([mn, pn, mode, int(s), f"{v:.4f}"])
            ax.plot(S / 1e3, sh, styles[mn], color=colors[pn],
                    label=f"{mn} @ {pn}")
    ax.set_xscale("log")
    ax.set_xlabel("context length S (K tokens)")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title("matrix @ peak tensor rate" if mode == "peak"
                 else "matrix @ practical rate (B70 100 TF, RTX5000 200 TF)")
    for sref in [8, 32, 128]:
        ax.axvline(sref, color="gray", lw=0.5, alpha=0.5)
axes[0].set_ylabel("vector time share in prefill (%)")
axes[1].legend(fontsize=8, loc="lower right")
fig.suptitle("Vector-op time share vs context length — prefill, per layer per token\n"
             "(vector = QK-norm/RoPE/RMSNorm/SwiGLU/softmax; measured ALU/SFU rates)")
fig.tight_layout()
fig.savefig("/home/tina/DEV/microbench/report/vector_share_vs_S.png", dpi=150)

with open("/home/tina/DEV/microbench/results/vector_share_vs_S.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "platform", "matrix_rate", "S_tokens", "vector_share_pct"])
    w.writerows(rows)

# console summary at key context lengths
print(f"{'model':<18}{'platform':<10}{'rate':<6}" +
      "".join(f"S={s//1000}K".rjust(9) for s in [1024, 8192, 32768, 131072, 262144]))
for mn, m in MODELS.items():
    for pn, p in PLATFORMS.items():
        for mode in ["peak", "prac"]:
            mx = p["mx_peak"] if mode == "peak" else p["mx_prac"]
            sh = share(m, p, mx)
            vals = []
            for sref in [1024, 8192, 32768, 131072, 262144]:
                i = int(np.argmin(np.abs(S - sref)))
                vals.append(f"{sh[i]:8.2f}%")
            print(f"{mn:<18}{pn:<10}{mode:<6}" + "".join(v.rjust(9) for v in vals))
print("\nwrote report/vector_share_vs_S.png and results/vector_share_vs_S.csv")
