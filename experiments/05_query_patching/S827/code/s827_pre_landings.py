"""Descriptive (not registered), 2026-10-16 19:4x: natural END(k) landing table across N = 6-9 from saved clean runs (R3_READ=max, two-form
menu readout), to shape the S827 hypotheses before any S827 data.  N = 7 / 8 from S824 (matched count a{N}_s{N}, seed 82400), N = 6 / 9 from
S825 (seed 82500); 40 scenes per cell.  Per cell: P(k - 1), P(k) (the forward copy), P(k + 1), P(target j = N + 1 - k) and the full landing.
Usage: python exp/s827_pre_landings.py  ->  research/ces/outputs/s827_pre_landings.json"""
from __future__ import annotations

import collections, json

V = "/home/storage2/ai26269345/vlm"


def main():
    tab = {}
    d = json.load(open(f"{V}/results/r3/s824_s824_27b_count.json")); R = d["rows"]
    for N in (7, 8):
        for k in d["ks"]:
            tab[(N, k)] = (collections.Counter(r["land"][f"a{N}_s{N}_k{k}_E"] for r in R), len(R), "S824")
    d = json.load(open(f"{V}/results/r3/s825_s825_27b_newN.json")); R = d["rows"]
    for N in (6, 9):
        for k in range(1, N + 1):
            nm = f"N{N}_k{k}_E"
            def pos(r):
                w = r["land"][nm]
                return r["labs"][str(N)].index(w) + 1 if isinstance(w, str) and w in r["labs"][str(N)] else w
            tab[(N, k)] = (collections.Counter(pos(r) for r in R), len(R), "S825")
    out = {}
    for (N, k), (c, n, src) in sorted(tab.items()):
        j = N + 1 - k; f = lambda x: c.get(x, 0) / n
        out[f"N{N}_k{k}"] = dict(src=src, n=n, target=j, p_km1=f(k - 1), p_k=f(k), p_kp1=f(k + 1), p_target=f(j),
                                 landing={str(x): f(x) for x in sorted(c, key=lambda z: (isinstance(z, str), z))})
    json.dump(out, open(f"{V}/research/ces/outputs/s827_pre_landings.json", "w"), indent=1)
    for key, v in out.items():
        print(key, "j", v["target"], "k-1 %.3f k %.3f k+1 %.3f j %.3f" % (v["p_km1"], v["p_k"], v["p_kp1"], v["p_target"]))


if __name__ == "__main__":
    main()
