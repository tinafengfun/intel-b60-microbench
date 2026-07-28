#!/usr/bin/env python3
# Sample xe-driver per-engine busy% for all DRM fds of a process.
# Usage: engmon2.py <pid> <sample_seconds>
# Prints one line per PCI device: bcs/ccs/rcs/vcs/vecs busy percent.
import glob, re, sys, time

pid, secs = sys.argv[1], int(sys.argv[2])

def snap():
    out = {}
    for f in glob.glob(f"/proc/{pid}/fdinfo/*"):
        try:
            txt = open(f).read()
        except Exception:
            continue
        if "drm-driver:\txe" not in txt:
            continue
        pdev = re.search(r"drm-pdev:\t(\S+)", txt).group(1)
        d = out.setdefault(pdev, {})
        for m in re.finditer(r"drm-(cycles|total-cycles)-(\w+):\t(\d+)", txt):
            d[f"{m.group(2)}:{m.group(1)}"] = int(m.group(3))
    return out

a = snap()
time.sleep(secs)
b = snap()
for pdev in sorted(b):
    if pdev not in a:
        continue
    tot = {k.split(":")[0]: b[pdev][k] - a[pdev].get(k, 0)
           for k in b[pdev] if ":total-cycles" in k}
    line = []
    for eng in ("bcs", "ccs", "rcs", "vcs", "vecs"):
        k = f"{eng}:cycles"
        if k in b[pdev]:
            t = tot.get(eng) or max(tot.values() or [1])
            line.append(f"{eng}={100 * (b[pdev][k] - a[pdev].get(k, 0)) / max(t, 1):6.2f}%")
    print(pdev, " ".join(line))
