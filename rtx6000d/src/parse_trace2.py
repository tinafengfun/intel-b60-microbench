#!/usr/bin/env python3
"""Parse vLLM torch-profiler chrome trace: module-level breakdown, per-step normalization."""
import json, gzip, sys, re
from collections import defaultdict

def load_trace(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        data = json.load(f)
    return data["traceEvents"] if isinstance(data, dict) else data

def classify(name, avg_dur):
    n = name
    if "gemvx" in n:  # cublas bf16 GEMV
        return "lm_head_gemv" if avg_dur > 100 else "router_gate_gemv"
    if "moefusedsilu" in n.lower() or "MoEMicroKernel" in n or "MoEDynamicKernel" in n:
        return "moe_nvfp4_fused"
    if "topkGating" in n: return "moe_router_topk"
    if "attentionmlaprefill" in n or "attentionmlakernel" in n or "attentionmlamerge" in n:
        return "attention_mla_core"
    if "gemmdense" in n:  # b12x fp8 dense gemms: attn proj + shared experts
        m = re.search(r"o(\d{3,5})\d*gmem|tensor0*o(\d+)", n)
        return "fp8_dense_gemm"
    if "integrationresidual" in n or "MHC" in n or "mhc" in n or "hc_head" in n:
        return "hyperconnection"
    if "quantize" in n or "Quant" in n: return "quant_invrope"
    if "QNormRopeKVRope" in n: return "quant_invrope"
    if "act_and_mul" in n or "silu" in n.lower(): return "activation"
    if "norm" in n.lower() and "aten" in n: return "norm"
    if "sample" in n or "gumbel" in n: return "sampler"
    if "indexSelect" in n or "index_elementwise" in n: return "embed_gather"
    if "Memcpy" in n or "memcpy" in n or "Fill" in n or "apply_write" in n or "gather_block" in n or "post_update" in n:
        return "mem_misc"
    if "elementwise" in n or "direct_copy" in n: return "elementwise"
    if "cutlass_80_wmma" in n: return "router_gate_gemv"
    return "other"

def main(path, nsteps):
    events = load_trace(path)
    kernels = [e for e in events if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    if not kernels:
        print("no kernels"); return
    total = sum(e["dur"] for e in kernels)
    t0 = min(e["ts"] for e in kernels); t1 = max(e["ts"] + e["dur"] for e in kernels)
    by_name = defaultdict(lambda: [0.0, 0])
    for e in kernels:
        a = by_name[e["name"]]
        a[0] += e["dur"]; a[1] += 1
    # classify with per-call duration hint
    by_mod = defaultdict(lambda: [0.0, 0])
    for name, (dur, cnt) in by_name.items():
        m = classify(name, dur / cnt)
        by_mod[m][0] += dur; by_mod[m][1] += cnt

    print(f"== {path.split('/')[-1]}")
    print(f"GPU busy {total/1e3:.2f} ms | span {(t1-t0)/1e3:.2f} ms | {len(kernels)} launches | steps={nsteps}\n")
    print(f"{'module':<22} {'tot_ms':>8} {'%busy':>6} {'per_step_us':>11} {'per_layerstep_us':>16} {'launch':>6}")
    for m, (dur, cnt) in sorted(by_mod.items(), key=lambda x: -x[1][0]):
        print(f"{m:<22} {dur/1e3:>8.2f} {100*dur/total:>5.1f}% {dur/nsteps:>11.1f} {dur/nsteps/2:>16.1f} {cnt:>6}")
    print(f"\nTOTAL {total/1e3:.2f} ms | per step {total/nsteps:,.0f} us | per layer-step {total/nsteps/2:,.0f} us")

if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
