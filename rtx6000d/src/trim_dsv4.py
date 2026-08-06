import json, os, shutil, sys
from safetensors import safe_open
from safetensors.torch import save_file

SRC = "/disk/DeepSeek-V4-Flash-0731-NVFP4"
DST = "/disk/DeepSeek-V4-Flash-2L-NVFP4"
KEEP_LAYERS = [0, 1]

os.makedirs(DST, exist_ok=True)
idx = json.load(open(f"{SRC}/model.safetensors.index.json"))
wm = idx["weight_map"]

def keep(k):
    if k.startswith("layers."):
        n = int(k.split(".")[1])
        return n in KEEP_LAYERS
    if k.startswith("mtp."):
        return False
    return True  # embed.weight, head.weight, norm.weight, hc_head_*

kept = {k: v for k, v in wm.items() if keep(k)}
print(f"keeping {len(kept)}/{len(wm)} tensors")

tensors = {}
total = 0
for k, shard in kept.items():
    with safe_open(f"{SRC}/{shard}", framework="pt") as f:
        t = f.get_tensor(k)
    tensors[k] = t
    total += t.numel() * t.element_size()
print(f"total bytes: {total/2**30:.2f} GiB")

save_file(tensors, f"{DST}/model-00001-of-00001.safetensors", metadata={"format": "pt"})
new_idx = {"metadata": {"total_size": total},
           "weight_map": {k: "model-00001-of-00001.safetensors" for k in kept}}
json.dump(new_idx, open(f"{DST}/model.safetensors.index.json", "w"), indent=2)

# config.json edits
cfg = json.load(open(f"{SRC}/config.json"))
cfg["num_hidden_layers"] = len(KEEP_LAYERS)
cr = cfg.get("compress_ratios", [])
cfg["compress_ratios"] = cr[:len(KEEP_LAYERS)] + cr[43:]  # 2 main + mtp entries
if "quantization_config" in cfg:
    qc = cfg["quantization_config"]
    if "quantized_layers" in qc:
        qc["quantized_layers"] = {k: v for k, v in qc["quantized_layers"].items()
                                  if not k.startswith("layers.") or int(k.split(".")[1]) in KEEP_LAYERS}
json.dump(cfg, open(f"{DST}/config.json", "w"), indent=2)

# hf_quant_config.json edits
hq = json.load(open(f"{SRC}/hf_quant_config.json"))
if "quantization" in hq and "quantized_layers" in hq["quantization"]:
    hq["quantization"]["quantized_layers"] = {k: v for k, v in hq["quantization"]["quantized_layers"].items()
        if not k.startswith("layers.") or int(k.split(".")[1]) in KEEP_LAYERS}
json.dump(hq, open(f"{DST}/hf_quant_config.json", "w"), indent=2)

# tokenizer + misc
for f in ["generation_config.json", "tokenizer_config.json", "tokenizer.json"]:
    shutil.copy(f"{SRC}/{f}", f"{DST}/{f}")
shutil.copytree(f"{SRC}/encoding", f"{DST}/encoding", dirs_exist_ok=True)
print("DONE")
