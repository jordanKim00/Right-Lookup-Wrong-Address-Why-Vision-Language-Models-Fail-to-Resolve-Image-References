"""S828 predictions, fixed from STAGE 1 (END(k), START(j) natural runs) before STAGE 2; procedure registered in knowledge/DECISIONS.md before any
S828 data.  The two-route readout law is the one found after S827 and frozen here (research/ces/outputs/s827_post_tworoute.json, entry
B51_A51_r2: P(i) proportional to exp(bB zB_i + bA zA_i), bB 0.7, bA 1.4; zB = mean of the boundary newline rows B1, B4, zA = the answer row,
index-matching-head block scores at L51).  Every arm's internal scores are predicted by replacement: the edited rows take the donor's natural
scores (half: the mean of donor and recipient), unedited rows keep the recipient's natural scores, and an edit of the non-address part (BA_PERP)
leaves them unchanged.  Writes research/ces/outputs/s828_predictions_<tag>.json and prints its sha256.
Usage: python s828_predict.py TAG"""
from __future__ import annotations

import hashlib, json, sys
import numpy as np

# arm -> (recipient, zB source, zA source); sources: "E" recipient END natural, "S" START(j) natural, "H" the mean
ARMS = dict(B_FULL=("END", "S", "E"), A1_FULL=("END", "E", "S"), A4_FULL=("END", "E", "S"), BA_FULL=("END", "S", "S"), BA_HALF=("END", "H", "H"),
            BA_PERP=("END", "E", "E"), R_B=("START", "E", "S"), R_A4=("START", "S", "E"), R_BA=("START", "E", "E"))


def main(tag):
    law = json.load(open("/vlm/research/ces/outputs/s827_post_tworoute.json"))["B51_A51_r2"]
    bB, bA = law["bB"], law["bA"]
    d = json.load(open(f"/vlm/results/r3/s828_{tag}_natural.json")); R = d["rows"]
    L = str(min(d["look"], key=lambda l: abs(l - 51)))
    zB = lambda r, run: np.mean([np.array(r["qk"][run][L][2]), np.array(r["qk"][run][L][5])], 0)
    zA = lambda r, run: np.array(r["qk"][run][L][9])
    soft = lambda s: np.exp(s - s.max()) / np.exp(s - s.max()).sum()
    out = dict(tag=tag, law=dict(bB=bB, bA=bA, layer=L, source="s827_post_tworoute.json B51_A51_r2"), cases={})
    for N, k, j in ((7, 4, 4), (9, 5, 5), (8, 5, 4)):
        rows = [r for r in R if r["N"] == N]
        if not rows:
            continue
        cand = sorted({j - 1, j, j + 1, k})
        res = dict(natural={}, arms={})
        for run in ("END", "START"):
            ps = [dict(si=r["si"], p=soft(bB * zB(r, run) + bA * zA(r, run)).tolist()) for r in rows]
            res["natural"][run] = dict(per_scene=ps, shares={x: float(np.mean([p["p"][x - 1] for p in ps])) for x in cand})
        for nm, (rcp, sb, sa) in ARMS.items():
            src = dict(E=lambda r, f: f(r, "END"), S=lambda r, f: f(r, "START"), H=lambda r, f: (f(r, "END") + f(r, "START")) / 2)
            ps = []
            for r in rows:
                zb = src[sb](r, zB) if not (rcp == "START" and sb == "S") else zB(r, "START")
                za = src[sa](r, zA) if not (rcp == "START" and sa == "S") else zA(r, "START")
                ps.append(dict(si=r["si"], zB=zb.tolist(), zA=za.tolist(), p=soft(bB * zb + bA * za).tolist()))
            res["arms"][nm] = dict(recipient=rcp, per_scene=ps, shares={x: float(np.mean([p["p"][x - 1] for p in ps])) for x in cand})
        out["cases"][f"N{N}k{k}j{j}"] = res
        print(f"N{N} k{k} j{j} natural END pred", {x: round(v_, 3) for x, v_ in res["natural"]["END"]["shares"].items()},
              "obs", {x: float(np.mean([r["land"]["END"] == x for r in rows])) for x in cand})
        for nm, v_ in res["arms"].items():
            print(f"   {nm:8s}", {x: round(y, 3) for x, y in v_["shares"].items()})
    fn = f"/vlm/research/ces/outputs/s828_predictions_{tag}.json"
    js = json.dumps(out, indent=1, default=float); open(fn, "w").write(js)
    print("sha256", hashlib.sha256(js.encode()).hexdigest(), fn)


if __name__ == "__main__":
    main(sys.argv[1])
