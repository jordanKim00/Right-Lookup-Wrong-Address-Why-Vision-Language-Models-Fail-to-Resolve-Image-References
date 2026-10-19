"""S736 analysis, frozen with the pre-registration (knowledge/DECISIONS.md 2026-10-10 16:35) before any S736 data.
Accuracy = answer on the target j.  Cells: (N, k), k != middle; written address t = N + 1 - k.
    FRAME_NEEDED_NEAR[m] = over cells with t <= 3: acc(R3) - acc(ADDR) >= .20
    FRAME_SEMANTIC[m]    = over the same cells: acc(R3) - acc(RANDF) >= .20 and acc(RANDF) - acc(ADDR) <= .10
    FRAME_IDLE_FAR[m]    = (the four smaller models) over cells with t >= 5: |acc(R3) - acc(ADDR)| <= .10
    FRAME_NEEDED_ALL[27B]= over all cells: acc(R3) - acc(ADDR) >= .20
Reported: accuracy and matcher-query cosine to START per arm and per (N, k).
Usage: python s736_analyze.py TAG [TAG ...]   (tags ending in _27b get FRAME_NEEDED_ALL; others FRAME_IDLE_FAR)
"""
import json, sys
from collections import defaultdict
import numpy as np

ARMS = ["START", "END", "ADDR", "R3", "RANDF", "FRAME"]


def analyze(tag):
    d = json.load(open(f"/vlm/results/r3/s736_{tag}_frame.json")); R = d["rows"]
    acc = lambda rows, c: float(np.mean([r[f"land_{c}"] == r["j"] for r in rows])) if rows else float("nan")
    cos = lambda rows, c: float(np.mean([r[f"cos_{c}"] for r in rows])) if rows else float("nan")
    near = [r for r in R if r["t"] <= 3]; far = [r for r in R if r["t"] >= 5]
    S = {"near": {c: acc(near, c) for c in ARMS}, "far": {c: acc(far, c) for c in ARMS}, "all": {c: acc(R, c) for c in ARMS},
         "cos_all": {c: cos(R, c) for c in ARMS}, "cos_near": {c: cos(near, c) for c in ARMS}}
    cells = defaultdict(list)
    for r in R:
        cells[(r["N"], r["k"])].append(r)
    T = {f"N{N}k{k}": {c: acc(rs, c) for c in ARMS} for (N, k), rs in sorted(cells.items())}
    chk = {"FRAME_NEEDED_NEAR": S["near"]["R3"] - S["near"]["ADDR"] >= .20,
           "FRAME_SEMANTIC": (S["near"]["R3"] - S["near"]["RANDF"] >= .20) and (S["near"]["RANDF"] - S["near"]["ADDR"] <= .10)}
    if tag.endswith("_27b"):
        chk["FRAME_NEEDED_ALL"] = S["all"]["R3"] - S["all"]["ADDR"] >= .20
    else:
        chk["FRAME_IDLE_FAR"] = abs(S["far"]["R3"] - S["far"]["ADDR"]) <= .10
    print(f"[{tag}] near (t<=3, n={len(near)}): " + " · ".join(f"{c} {S['near'][c]:.2f}" for c in ARMS))
    print(f"[{tag}] far  (t>=5, n={len(far)}): " + " · ".join(f"{c} {S['far'][c]:.2f}" for c in ARMS))
    print(f"[{tag}] cos(START) all: " + " · ".join(f"{c} {S['cos_all'][c]:.3f}" for c in ARMS))
    print(f"[{tag}] checks {chk}")
    return dict(summary=S, cells=T, checks=chk, n=len(R))


if __name__ == "__main__":
    import os
    P = "/vlm/results/r3/s736_analysis.json"
    out = json.load(open(P)) if os.path.exists(P) else {}
    out.update({t: analyze(t) for t in sys.argv[1:]})
    json.dump(out, open(P, "w"), indent=1)
