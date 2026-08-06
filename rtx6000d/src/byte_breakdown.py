import json, re
from collections import defaultdict
from safetensors import safe_open

SRC = "/disk/DeepSeek-V4-Flash-0731-NVFP4"
idx = json.load(open(f"{SRC}/model.safetensors.index.json"))
wm = idx["weight_map"]

def cat(k):
    if k.startswith("mtp."): return "mtp"
    if "ffn.experts." in k:
        if k.endswith(".weight"): return "expert_weight_fp4"
        if "weight_scale_2" in k: return "expert_scale2"
        if "weight_scale" in k: return "expert_scale_ue8m0"
        if "input_scale" in k: return "expert_input_scale"
        return "expert_other"
    if ".attn." in k or "attn." in k:
        if k.endswith(".weight"): return "attn_weight_fp8"
        if k.endswith(".scale"): return "attn_scale_ue8m0"
        return "attn_other"
    if "shared_experts" in k:
        if k.endswith(".weight"): return "shared_weight_fp8"
        if k.endswith(".scale"): return "shared_scale"
        return "shared_other"
    if k in ("embed.weight",): return "embed"
    if k == "head.weight": return "lm_head"
    if "gate" in k: return "moe_gate"
    if k.startswith("hc_") or ".hc_" in k: return "hyperconn"
    return "other_norm"

DT_SIZE = {"F8_E4M3":1,"F8_E8M0":1,"BF16":2,"F16":2,"F32":4,"F64":8,"U8":1,"I8":1,"I32":4,"I64":8,"U32":4}
agg = defaultdict(lambda: [0,0])  # cat -> [bytes, count]
by_shard = defaultdict(list)
for k, shard in wm.items():
    by_shard[shard].append(k)

for shard, keys in sorted(by_shard.items()):
    with safe_open(f"{SRC}/{shard}", framework="pt") as f:
        for k in keys:
            sl = f.get_slice(k)
            shape = sl.get_shape(); dt = sl.get_dtype()
            n = 1
            for d in shape: n *= d
            if dt == "F4_E2M1": nbytes = (n+1)//2
            else: nbytes = n * DT_SIZE.get(dt, 2)
            c = cat(k)
            agg[c][0] += nbytes; agg[c][1] += 1

total = sum(v[0] for v in agg.values())
print(f"total {total/2**30:.2f} GiB")
for c, (b, n) in sorted(agg.items(), key=lambda x:-x[1][0]):
    print(f"{c:<22} {b/2**30:>8.3f} GiB  {100*b/total:>5.1f}%  tensors={n}")
