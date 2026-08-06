#!/usr/bin/env python3
"""Profile prefill (64x16 tok in / 1 out) and decode (64x1 tok in / 16 out), batch=64."""
import json, time, urllib.request, sys
from concurrent.futures import ThreadPoolExecutor

BASE = "http://127.0.0.1:8100"
BATCH = 64

def post(path, payload, timeout=900):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
        return json.loads(body) if body.strip() else {}

def wait_ready():
    for i in range(720):
        try:
            with urllib.request.urlopen(BASE + "/health", timeout=5) as r:
                if r.status == 200:
                    print("server ready"); return
        except Exception:
            pass
        time.sleep(5)
    sys.exit("server never became ready")

def completion(prompt_ids, max_tokens):
    return post("/v1/completions", {
        "model": "DeepSeek-V4-Flash",
        "prompt": prompt_ids,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
    })

def run_batch(prompt_len, max_tokens):
    with ThreadPoolExecutor(max_workers=BATCH) as ex:
        futs = [ex.submit(completion, [1000 + i * 100 + j for j in range(prompt_len)], max_tokens)
                for i in range(BATCH)]
        return [f.result() for f in futs]

wait_ready()
run_batch(16, 4)  # warmup outside profile window
print("warmup done")

# ---- prefill: 64 x 16 tokens in, 1 token out ----
post("/start_profile", {})
t0 = time.time()
rs = run_batch(16, 1)
t1 = time.time()
post("/stop_profile", {})
pt = sum(r["usage"]["prompt_tokens"] for r in rs)
print(f"PREFILL_B64 wall={t1-t0:.3f}s total_prompt_tokens={pt}")
time.sleep(15)

# ---- decode: 64 x 1 token in, 16 tokens out ----
post("/start_profile", {})
t0 = time.time()
rs = run_batch(1, 16)
t1 = time.time()
post("/stop_profile", {})
ct = sum(r["usage"]["completion_tokens"] for r in rs)
print(f"DECODE_B64  wall={t1-t0:.3f}s total_completion_tokens={ct}")
time.sleep(15)
print("PROFILING_DONE")
