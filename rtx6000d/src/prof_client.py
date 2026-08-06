#!/usr/bin/env python3
"""Profile prefill (16 tok in / 1 out) and decode (1 tok in / 16 out) phases."""
import json, time, urllib.request, sys

BASE = "http://127.0.0.1:8100"

def post(path, payload):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
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

wait_ready()
# warmup (outside profile window)
r = completion(list(range(1000, 1016)), 4)
print("warmup:", r["usage"])

# ---- prefill: 16 tokens in, 1 token out ----
post("/start_profile", {})
t0 = time.time()
r = completion(list(range(2000, 2016)), 1)
t1 = time.time()
post("/stop_profile", {})
print(f"PREFILL wall={t1-t0:.3f}s usage={r['usage']}")
time.sleep(10)  # let profiler flush

# ---- decode: 1 token in, 16 tokens out ----
post("/start_profile", {})
t0 = time.time()
r = completion([3000], 16)
t1 = time.time()
post("/stop_profile", {})
print(f"DECODE  wall={t1-t0:.3f}s usage={r['usage']}")
time.sleep(10)
print("PROFILING_DONE")
