"""S881 analysis (rules fixed before the data; knowledge/DECISIONS.md S881 entry).  Input results/r3/s881_<tag>_anchorrm.json.
Address score difference at late layer L: T1_L = z_(p-1) - z_p, matcher-head (late_mean) mean over rows B1 / B4 / answer; for E, R, E_BANDR from the
stored caps, for the edited runs (AR, PR, FR, ID) from the block scores recomputed with their edited queries.  Writer address effect
W_X(L) = T1_L(X) - T1_L(E_BANDR) for X in {E, AR, PR, FR, ID}.  Scene bootstrap 4,000 (seed 881).
I0: t1 >= .95; START(p) answers p >= .95; L47 anchor = p for E and R >= .90; the identity recompute reproduces E (answers equal in >= .95 of
  runs and median |W_ID(L51) - W_E(L51)| <= .02).  Nothing is read if I0 fails.
E1 (part of the step): ratio_AR = mean W_AR(L51) / mean W_E(L51) <= .80 with the upper bound < 1, and the far control's ratio_FR >= .90 ->
  'removing the anchor-suppression component removes part of the writer's step'; else 'not'.
E2 (the per-scene variation): within-pin sd of W_AR(L51) over sd of W_E(L51) <= .70 with the upper bound < .90, and below the same ratio for PR
  -> 'the anchor-suppression component carries the per-scene variation of the writer'; else 'not'.
E3 (exact-target protection): at p = 4 (the identity pin: END(5) at N 8 targets 4), P(answer = 4 | AR) - P(answer = 4 | E) >= .10 with the
  interval above 0 -> 'removing the anchor suppression keeps the exact target'; else 'not'.
Descriptive: ratios and sd ratios for PR / FR, at L55 / L59 / L63; answers per pin and arm.
Usage: python3 s881_analyze.py TAG"""
from __future__ import annotations

import collections, json, sys
import numpy as np

TAG = sys.argv[1]
J = json.load(open(f"/vlm/results/r3/s881_{TAG}_anchorrm.json"))
OUT = f"/vlm/results/r3/s881_analysis_{TAG}.json"
rng = np.random.default_rng(881); NB = 4000
rows = J["rows"]; S = len(rows); PINS = J["pins"]; lm = J["late_mean"]; LAYS = [str(l) for l in J["lateq"]]
ARMS = ("E", "AR", "PR", "FR", "ID")


def T1cap(r, key, p, L):
    z = np.asarray(r["zl"][key][L], dtype=float)[:, lm[L]].mean(1).mean(0); return z[p - 2] - z[p - 1]


def T1ed(r, key, p, L):
    z = np.asarray(r["zedit"][key][L], dtype=float)[:, lm[L]].mean(1).mean(0); return z[p - 2] - z[p - 1]


def anchor(q):
    q = np.asarray(q); am = [int(np.argmax(q[i])) + 1 for i in (2, 5, 9)]
    c = collections.Counter(am).most_common(); return c[0][0] if c[0][1] >= 2 else am[2]


REC = {}
for si, r in enumerate(rows):
    for p in PINS:
        d = dict(si=si, p=p)
        for L in LAYS:
            tb = T1cap(r, f"Ep{p}_E_BANDR", p, L)
            d[f"W_E_{L}"] = T1cap(r, f"Ep{p}", p, L) - tb
            for nm in ("AR", "PR", "FR", "ID"):
                d[f"W_{nm}_{L}"] = T1ed(r, f"Ep{p}_{nm}", p, L) - tb
        for nm in ARMS:
            d[f"x_{nm}"] = r["land"][f"Ep{p}" if nm == "E" else f"Ep{p}_{nm}"]
        d["anch"] = (anchor(r["qk47"][f"Ep{p}"]), anchor(r["qk47"][f"Rp{p}"]))
        REC[(si, p)] = d
SC = sorted({k[0] for k in REC})


def bt(fn):
    n = len(SC); est = fn(np.arange(n)); bs = [fn(rng.integers(0, n, n)) for _ in range(NB)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return dict(est=float(est), lo=float(lo), hi=float(hi))


def mW(ix, nm, L="51", pins=PINS):
    return float(np.mean([REC[(SC[i], p)][f"W_{nm}_{L}"] for i in ix for p in pins]))


def sdr(ix, nm, L="51"):
    a_, b_ = [], []
    for p in PINS:
        x = np.array([REC[(SC[i], p)][f"W_{nm}_{L}"] for i in ix]); y = np.array([REC[(SC[i], p)][f"W_E_{L}"] for i in ix])
        a_ += list(x - x.mean()); b_ += list(y - y.mean())
    return float(np.std(a_) / np.std(b_))


res = dict(tag=TAG, n_scenes=S)
st = float(np.mean([r["land"][f"START{p}"] == p for r in rows for p in PINS]))
reach = float(np.mean([a_ == d["p"] for d in REC.values() for a_ in d["anch"]]))
same = float(np.mean([d["x_ID"] == d["x_E"] for d in REC.values()])); idd = float(np.median([abs(d["W_ID_51"] - d["W_E_51"]) for d in REC.values()]))
I0 = dict(t1=bool((J["t1_menu"] or 0) >= .95), START=bool(st >= .95), pin_reach=bool(reach >= .90), identity=bool(same >= .95 and idd <= .02))
res["I0"] = dict(**I0, all=bool(all(I0.values())), t1_value=J["t1_menu"], START_acc=st, reach=reach, ID_same_answer=same, ID_median_dW=idd)
ok = res["I0"]["all"]
e1 = bt(lambda ix: mW(ix, "AR") / mW(ix, "E")); fr = bt(lambda ix: mW(ix, "FR") / mW(ix, "E"))
res["E1"] = dict(ratio_AR=e1, ratio_FR=fr, verdict=(("removing the anchor-suppression component removes part of the writer's step"
                                                     if e1["est"] <= .80 and e1["hi"] < 1 and fr["est"] >= .90 else "not") if ok else "not read"))
e2 = bt(lambda ix: sdr(ix, "AR")); e2p = bt(lambda ix: sdr(ix, "PR"))
res["E2"] = dict(sd_ratio_AR=e2, sd_ratio_PR=e2p, verdict=(("the anchor-suppression component carries the per-scene variation of the writer"
                                                            if e2["est"] <= .70 and e2["hi"] < .90 and e2["est"] < e2p["est"] else "not") if ok else "not read"))
def acc4(ix, nm):
    return float(np.mean([REC[(SC[i], 4)][f"x_{nm}"] == 4 for i in ix]))
e3 = bt(lambda ix: acc4(ix, "AR") - acc4(ix, "E"))
res["E3"] = dict(diff=e3, verdict=(("removing the anchor suppression keeps the exact target" if e3["est"] >= .10 and e3["lo"] > 0 else "not") if ok else "not read"))
desc = {}
for L in LAYS:
    desc[f"L{L}"] = {nm: dict(ratio=mW(range(len(SC)), nm, L) / mW(range(len(SC)), "E", L), sd_ratio=sdr(range(len(SC)), nm, L)) for nm in ("AR", "PR", "FR", "ID")}
desc["answers"] = {f"p{p}": {nm: {str(v_): float(np.mean([REC[(s, p)][f"x_{nm}"] == v_ for s in SC])) for v_ in range(1, 9)} for nm in ARMS} for p in PINS}
desc["mean_W_E_51_by_pin"] = {f"p{p}": mW(range(len(SC)), "E", "51", [p]) for p in PINS}
res["descriptive"] = desc
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"]["all"], E1=[res["E1"]["verdict"], e1["est"], fr["est"]], E2=[res["E2"]["verdict"], e2["est"], e2p["est"]],
                      E3=[res["E3"]["verdict"], e3["est"]]), indent=1))
print("->", OUT)
