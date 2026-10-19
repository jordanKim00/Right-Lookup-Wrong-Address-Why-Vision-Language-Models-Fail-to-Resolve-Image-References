"""Post hoc (not registered; 10-16 21:1x, after the S827 verdicts): does a two-route readout law - boundary newline rows (B1, B4) plus the answer
row - predict the answers under S827's interventions, where the registered boundary-only law failed (PRED c: calibration .56 / .51)?
Law: P(land = i) proportional to exp(bB zB_i + bA zA_i) (index-matching-head block scores at the given layers), fitted by maximum likelihood on
stage-1 non-identity natural runs only; evaluated, from each run's own measured scores, on END(k) (stage 1) and on every stage-2 arm:
calibration (share of arm x {k - 1, k, k + 1} within .15) and mean log-probability.  Exploratory: it chooses among a few layer pairs on the
training cells, and any law it suggests must be registered before it is used to predict a new run.
Usage: python exp/s827_post_tworoute.py  ->  research/ces/outputs/s827_post_tworoute.json"""
from __future__ import annotations

import itertools, json
import numpy as np

V = "/home/storage2/ai26269345/vlm"
d1 = json.load(open(f"{V}/results/r3/s827_s827_27b_natural.json")); d2 = json.load(open(f"{V}/results/r3/s827_s827_27b_identity.json"))
NONID = ["START", "START_m1", "START_p1", "END_m1", "END_p1"]
ARMS = ["A_FULL", "A_HALF", "A_DIP", "A_DIP50", "A_PREV", "A_PREV50", "A_NEXT", "A_FLANK", "A_UK", "R_FULL", "R_DIP", "R_PREV", "R_NEXT"]
zB = lambda r, run, L: np.mean([np.array(r["qk"][run][L][2]), np.array(r["qk"][run][L][5])], 0)
zA = lambda r, run, L: np.array(r["qk"][run][L][6])


def fit(LB, LA, route):
    obs = [(r, run) for r in d1["rows"] for run in NONID if r["land"][run] >= 1]
    def ll(bB, bA):
        t = 0.0
        for r, run in obs:
            s = bB * zB(r, run, LB) + (bA * zA(r, run, LA) if route == 2 else 0); s = s - s.max()
            t += s[r["land"][run] - 1] - np.log(np.exp(s).sum())
        return t
    grid = np.linspace(0, 4, 41)
    best = max(((ll(b1, b2), b1, b2) for b1 in grid for b2 in (grid if route == 2 else [0.0])))
    return best


def evaluate(LB, LA, bB, bA):
    out = {}; hits = []; lps = []
    for N, k in ((7, 4), (9, 5)):
        r2 = [r for r in d2["rows"] if r["N"] == N]; r1 = [r for r in d1["rows"] if r["N"] == N]
        pred = lambda r, run: (lambda s: np.exp(s - s.max()) / np.exp(s - s.max()).sum())(bB * zB(r, run, LB) + bA * zA(r, run, LA))
        res = {}
        for run, rows in [("END", r1)] + [(a, r2) for a in ARMS]:
            ok = [r for r in rows if r["land"][run] >= 1]
            p = np.mean([pred(r, run) for r in ok], 0)
            o = {x: float(np.mean([r["land"][run] == x for r in ok])) for x in (k - 1, k, k + 1)}
            h = [abs(p[x - 1] - o[x]) <= .15 for x in o]
            if run != "END":
                hits.extend(h); lps.extend([float(np.log(max(pred(r, run)[r["land"][run] - 1], 1e-9))) for r in ok])
            res[run] = dict(pred={x: float(p[x - 1]) for x in o}, obs=o)
        out[f"N{N}k{k}"] = res
    return out, float(np.mean(hits)), float(np.mean(lps))


rep = {}
for LB, LA in [("51", "51"), ("51", "55"), ("55", "55"), ("47", "55")]:
    for route in (1, 2):
        llv, bB, bA = fit(LB, LA, route)
        res, cal, lp = evaluate(LB, LA, bB, bA)
        rep[f"B{LB}_A{LA}_r{route}"] = dict(train_ll=llv, bB=bB, bA=bA, arm_calibration=cal, arm_logp=lp, per_case=res)
        print(f"B L{LB} + A L{LA} routes {route}: train ll {llv:.1f} bB {bB:.1f} bA {bA:.1f} | arms calibrated {cal:.2f} mean logp {lp:.3f}")
best = max(rep, key=lambda x: rep[x]["train_ll"])
print("chosen on training cells:", best)
for c, res in rep[best]["per_case"].items():
    for run in ("END", "A_FULL", "A_DIP", "A_PREV", "A_FLANK", "R_FULL"):
        print("  ", c, f"{run:7s}", "pred", {x: round(y, 2) for x, y in res[run]["pred"].items()}, "obs", res[run]["obs"])
json.dump(rep, open(f"{V}/research/ces/outputs/s827_post_tworoute.json", "w"), indent=1)
