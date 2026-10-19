"""S834 analysis (rules fixed in knowledge/DECISIONS.md 10-17 02:32, before any S834 data).  Descriptive; no hypothesis.
I0: t1 >= .95; START(k) -> k >= .90 in every new cell; the 40 scenes' labels and menu equal S824's saved ones; the k = 3 END answer equals
S824's in >= 90% of scenes (batching may move near-ties).
Only if I0 holds, write the fig_grid data: N = 6, 9 from S825 (all k), N = 7, 8 from S824 (k = 3-7) + S834 (k = 1, 2 and k = 8 at N = 8).
Usage: python exp/s834_analyze.py [TAG]  ->  results/r3/s834_analysis.json (+ research/ces/outputs/v2_grid_landings.json)"""
from __future__ import annotations

import collections, json, sys

V = "/home/storage2/ai26269345/vlm"


def main(tag="s834_27b"):
    d = json.load(open(f"{V}/results/r3/s834_{tag}_grid.json")); R = d["rows"]
    s824 = json.load(open(f"{V}/results/r3/s824_s824_27b_count.json")); R4 = s824["rows"]
    out = dict(tag=tag, n=len(R), t1=d["t1_menu"], I0={})
    same = sum(r["labs"] == q["labs"] and r["menu"] == q["menu"] for r, q in zip(R, R4)) if len(R) <= len(R4) else 0
    out["I0"]["scenes_same_as_s824"] = same / len(R)
    st = {}
    for N, ks in d["cells"].items():
        for k in ks:
            if k == 3:
                continue
            st[f"N{N}_k{k}"] = sum(r["land"][f"a{N}_s{N}_k{k}_S"] == k for r in R) / len(R)
    out["I0"]["start_correct"] = st
    rep = {N: sum(r["land"][f"a{N}_s{N}_k3_E"] == q["land"][f"a{N}_s{N}_k3_E"] for r, q in zip(R, R4)) / len(R) for N in d["cells"]}
    out["I0"]["k3_end_reproduces_s824"] = rep
    ok = (d["t1_menu"] >= .95 and min(st.values()) >= .90 and out["I0"]["scenes_same_as_s824"] == 1.0 and min(rep.values()) >= .90)
    out["I0"]["holds"] = bool(ok)
    land = {}
    for N, ks in d["cells"].items():
        for k in ks:
            c = collections.Counter(r["land"][f"a{N}_s{N}_k{k}_E"] for r in R)
            land[f"N{N}_k{k}"] = {str(x): c[x] / len(R) for x in sorted(c)}
    out["new_cells_end"] = land
    json.dump(out, open(f"{V}/results/r3/s834_analysis.json" if tag == "s834_27b" else f"{V}/results/r3/s834_analysis_{tag}.json", "w"), indent=1)
    print(json.dumps(out["I0"], indent=1))
    for key, v in land.items():
        print(key, v)
    if not ok or tag != "s834_27b":
        print("fig_grid data not written (I0 fails or non-default tag)")
        return
    grid = {}
    for N in (7, 8):
        for k in range(1, N + 1):
            src, rows = ("S824", R4) if 3 <= k <= 7 else ("S834", R)
            c = collections.Counter(r["land"][f"a{N}_s{N}_k{k}_E"] for r in rows)
            grid[f"N{N}_k{k}"] = dict(src=src, n=len(rows), target=N + 1 - k, landing={str(x): c[x] / len(rows) for x in sorted(c)})
    s825 = json.load(open(f"{V}/results/r3/s825_s825_27b_newN.json")); R5 = s825["rows"]
    for N in (6, 9):
        for k in range(1, N + 1):
            nm = f"N{N}_k{k}_E"
            def pos(r):
                w = r["land"][nm]
                return r["labs"][str(N)].index(w) + 1 if isinstance(w, str) and w in r["labs"][str(N)] else w
            c = collections.Counter(pos(r) for r in R5)
            grid[f"N{N}_k{k}"] = dict(src="S825", n=len(R5), target=N + 1 - k,
                                      landing={str(x): c[x] / len(R5) for x in sorted(c, key=lambda z: (isinstance(z, str), z))})
    json.dump(dict(note="natural END(k) answer shares; 0 = a menu word that is not a shown picture", cells=grid),
              open(f"{V}/research/ces/outputs/v2_grid_landings.json", "w"), indent=1)
    print("wrote research/ces/outputs/v2_grid_landings.json")


if __name__ == "__main__":
    main(*sys.argv[1:])
