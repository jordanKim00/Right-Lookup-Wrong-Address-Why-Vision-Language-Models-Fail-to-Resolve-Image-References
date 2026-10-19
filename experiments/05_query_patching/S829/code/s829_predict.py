"""S829 predictions, fixed from STAGE 1 (END(k), START(j)) before STAGE 2; procedure registered in knowledge/DECISIONS.md before any S829 data.
Mixture law: P(i) = w L(z_arm)(i) + (1 - w) L(z_START(j))(i); L = the two-route law frozen after S827 (softmax(0.7 zB + 1.4 zA), zB = B1, B4
mean, zA = the answer row, index-matching-head block scores at L51; research/ces/outputs/s827_post_tworoute.json B51_A51_r2); w = the run's
late picture-reading mass (index-matching-head picture attention mass, mean of B1, B4 and the answer row, mean of L51 and L55) rescaled per
scene between its START(j) (0) and END(k) (1) natural values, clipped to [0, 1].
Replacement rules for an arm's z and mass (the edited quantity takes the donor's natural value where it is edited; the rest keeps the
recipient's): U_FULL, U_LATE -> z donor, mass recipient; U_EARLY -> both recipient (no edit at L51 / L55); P_FULL, P_LATE, MASS -> z recipient,
mass donor; R_U -> z END, mass START; R_P -> z START, mass END; R_ALL, R_UM -> z END, mass END.
Writes research/ces/outputs/s829_predictions_<tag>.json and prints its sha256.  Usage: python s829_predict.py TAG"""
from __future__ import annotations

import hashlib, json, sys
import numpy as np

# arm -> (recipient, z source, mass source); "E" = END(k) natural, "S" = START(j) natural
ARMS = dict(U_FULL=("END", "S", "E"), U_EARLY=("END", "E", "E"), U_LATE=("END", "S", "E"), P_FULL=("END", "E", "S"), P_LATE=("END", "E", "S"),
            MASS=("END", "E", "S"), R_U=("START", "E", "S"), R_P=("START", "S", "E"), R_ALL=("START", "E", "E"), R_UM=("START", "E", "E"))


def mass_of(r, run, L1, L2):
    return float(np.mean([np.mean([r["mass"][run][L][i] for i in (2, 5, 9)]) for L in (L1, L2)]))


def main(tag):
    law = json.load(open("/vlm/research/ces/outputs/s827_post_tworoute.json"))["B51_A51_r2"]
    bB, bA = law["bB"], law["bA"]
    d = json.load(open(f"/vlm/results/r3/s829_{tag}_natural.json")); R = d["rows"]
    look = d["look"]; L = str(min(look, key=lambda l: abs(l - 51))); L2 = str(min(look, key=lambda l: abs(l - 55)))
    zB = lambda r, run: np.mean([np.array(r["qk"][run][L][2]), np.array(r["qk"][run][L][5])], 0)
    zA = lambda r, run: np.array(r["qk"][run][L][9])
    soft = lambda s: np.exp(s - s.max()) / np.exp(s - s.max()).sum()
    Lz = lambda zb, za: soft(bB * zb + bA * za)
    out = dict(tag=tag, law=dict(bB=bB, bA=bA, layer=L, mass_layers=[L, L2]), cases={})
    for N, k, j in ((7, 4, 4), (9, 5, 5), (8, 5, 4)):
        rows = [r for r in R if r["N"] == N]
        if not rows:
            continue
        cand = sorted({j - 1, j, j + 1, k})
        res = dict(natural={}, arms={}, mass={})
        src = dict(E="END", S="START")
        for run in ("END", "START"):
            ps = [dict(si=r["si"], p=Lz(zB(r, run), zA(r, run)).tolist()) for r in rows]
            res["natural"][run] = dict(per_scene=ps, shares={x: float(np.mean([p["p"][x - 1] for p in ps])) for x in cand})
        res["mass"] = dict(END=float(np.mean([mass_of(r, "END", L, L2) for r in rows])), START=float(np.mean([mass_of(r, "START", L, L2) for r in rows])))
        for nm, (rcp, zs, ms) in ARMS.items():
            ps = []
            for r in rows:
                mE, mS = mass_of(r, "END", L, L2), mass_of(r, "START", L, L2)
                m = mass_of(r, src[ms], L, L2)
                w = float(np.clip((m - mS) / (mE - mS), 0, 1)) if abs(mE - mS) > 1e-6 else (1.0 if ms == "E" else 0.0)
                pz = Lz(zB(r, src[zs]), zA(r, src[zs])); p0 = Lz(zB(r, "START"), zA(r, "START"))
                ps.append(dict(si=r["si"], w=w, p=(w * pz + (1 - w) * p0).tolist()))
            res["arms"][nm] = dict(recipient=rcp, per_scene=ps, shares={x: float(np.mean([p["p"][x - 1] for p in ps])) for x in cand})
        out["cases"][f"N{N}k{k}j{j}"] = res
        print(f"N{N} k{k} j{j} mass END {res['mass']['END']:.3f} START {res['mass']['START']:.3f} | natural END pred",
              {x: round(v_, 3) for x, v_ in res["natural"]["END"]["shares"].items()}, "obs", {x: float(np.mean([r["land"]["END"] == x for r in rows])) for x in cand})
        for nm, v_ in res["arms"].items():
            print(f"   {nm:8s}", {x: round(y, 3) for x, y in v_["shares"].items()})
    fn = f"/vlm/research/ces/outputs/s829_predictions_{tag}.json"
    js = json.dumps(out, indent=1, default=float); open(fn, "w").write(js)
    print("sha256", hashlib.sha256(js.encode()).hexdigest(), fn)


if __name__ == "__main__":
    main(sys.argv[1])
