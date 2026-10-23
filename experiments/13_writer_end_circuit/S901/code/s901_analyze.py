"""S901 analysis (rules fixed before the data; knowledge/DECISIONS.md S901 entry).  Input results/r3/s901_<tag>_ctxroute.json.
W_X = T1(X) - T1(E_BANDR) at L51 (T1 = z_(p-1) - z_p, matcher-head mean block scores, rows B1 / B4 / answer).  Context effect of a value
transplant: dHP = W(E_HPV) - W(E) (S889 K1, negative); with route X blocked in the source hybrid: dHP_X = W(E_HPV_X) - W(E).  Remaining share
rho_X = mean dHP_X / mean dHP; blocked share s_X = 1 - rho_X (ratios of means over (scene, pin)); scene bootstrap 4,000 (seed 901).
I0: t1 >= .95; START(p) answers p >= .95; L47 anchor = p for E and R >= .90; the identity transplant E_HPV_SELF reproduces E (max late
  block-score difference <= 1e-3 in >= .99); closure: H_ALL's anchor value difference rel <= .01 in >= .99 and |mean dHP_ALL| <= .10 |mean dHP|;
  the context effect is present (mean dHP < 0 with the interval below 0).  Nothing is read if I0 fails.
C1 (which mixer): s_DN >= .70 with the lower bound > .50 and s_FA <= .30 -> 'the predecessor's context enters the anchor through DeltaNet
  recurrence'; s_FA >= .70 with the lower bound > .50 and s_DN <= .30 -> 'it enters through full attention among image tokens'; else 'both'.
C2 (depth): among D0 (layers 0-15), D1 (16-31), D2 (32-46) the largest s; s >= .50 with the lower bound > .30 -> 'it enters mainly at layers
  <range>'; else 'distributed'.
Descriptive: s_X at L55-63; the value-level share blocked (1 - along_X) and rel_X; dHP / W; answers per arm.
Usage: python3 s901_analyze.py TAG"""
from __future__ import annotations

import collections, json, sys
import numpy as np

TAG = sys.argv[1]
J = json.load(open(f"/vlm/results/r3/s901_{TAG}_ctxroute.json"))
OUT = f"/vlm/results/r3/s901_analysis_{TAG}.json"
rng = np.random.default_rng(901); NB = 4000
rows = J["rows"]; S = len(rows); PINS = J["pins"]; lm = J["late_mean"]; LAYS = ("51", "55", "59", "63")
ROUTES = ("FA", "DN", "ALL", "D0", "D1", "D2")
DEPTH = {"D0": "0-15", "D1": "16-31", "D2": "32-46"}


def zv(r, key, L):
    return np.asarray(r["zl"][key][L], dtype=float)[:, lm[L]].mean(1).mean(0)


def T1(r, key, p, L="51"):
    z = zv(r, key, L); return z[p - 2] - z[p - 1]


def anchor(q):
    q = np.asarray(q); am = [int(np.argmax(q[i])) + 1 for i in (2, 5, 9)]
    c = collections.Counter(am).most_common(); return c[0][0] if c[0][1] >= 2 else am[2]


REC = {}
for si, r in enumerate(rows):
    for p in PINS:
        d = dict(si=si, p=p)
        for L in LAYS:
            tb = T1(r, f"Ep{p}_E_BANDR", p, L); wE = T1(r, f"Ep{p}", p, L) - tb
            d[f"W_{L}"] = wE; d[f"dHP_{L}"] = T1(r, f"Ep{p}_HPV", p, L) - tb - wE
            for X in ROUTES:
                d[f"dHP_{X}_{L}"] = T1(r, f"Ep{p}_HPV_{X}", p, L) - tb - wE
        d["vid"] = max(float(np.abs(zv(r, f"Ep{p}_HPV_SELF", L) - zv(r, f"Ep{p}", L)).max()) for L in LAYS)
        d["vd"] = r["vdiff"][str(p)]
        d["anch"] = (anchor(r["qk47"][f"Ep{p}"]), anchor(r["qk47"][f"Rp{p}"]))
        d["land"] = {k: v for k, v in r["land"].items() if f"p{p}" in k}
        REC[(si, p)] = d
SC = sorted({k[0] for k in REC})


def bt(fn):
    n = len(SC); est = fn(np.arange(n)); bs = [fn(rng.integers(0, n, n)) for _ in range(NB)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return dict(est=float(est), lo=float(lo), hi=float(hi))


def M(ix, fn):
    return float(np.mean([fn(REC[(SC[i], p)]) for i in ix for p in PINS]))


sX = lambda ix, X, L="51": 1 - M(ix, lambda d: d[f"dHP_{X}_{L}"]) / M(ix, lambda d: d[f"dHP_{L}"])
res = dict(tag=TAG, n_scenes=S)
st = float(np.mean([r["land"][f"START{p}"] == p for r in rows for p in PINS]))
reach = float(np.mean([a_ == d["p"] for d in REC.values() for a_ in d["anch"]]))
vid_ok = float(np.mean([d["vid"] <= 1e-3 for d in REC.values()]))
clo_v = float(np.mean([d["vd"]["ALL"]["rel"] <= .01 for d in REC.values()]))
allx = range(len(SC))
clo_w = abs(M(allx, lambda d: d["dHP_ALL_51"])) / abs(M(allx, lambda d: d["dHP_51"]))
bD = bt(lambda ix: M(ix, lambda d: d["dHP_51"]))
I0 = dict(t1=bool((J["t1_menu"] or 0) >= .95), START=bool(st >= .95), pin_reach=bool(reach >= .90), identity=bool(vid_ok >= .99),
          closure=bool(clo_v >= .99 and clo_w <= .10), context=bool(bD["hi"] < 0))
res["I0"] = dict(**I0, all=bool(all(I0.values())), t1_value=J["t1_menu"], START_acc=st, reach=reach, vid_ok=vid_ok, closure_value=clo_v,
                 closure_W=clo_w, dHP=bD, dHP_over_W=M(allx, lambda d: d["dHP_51"]) / M(allx, lambda d: d["W_51"]))
ok = res["I0"]["all"]
bs_ = {X: bt(lambda ix, X=X: sX(ix, X)) for X in ROUTES}
sF, sDN = bs_["FA"], bs_["DN"]
res["C1"] = dict(s_FA=sF, s_DN=sDN, verdict=(("the predecessor's context enters the anchor through DeltaNet recurrence" if sDN["est"] >= .70 and sDN["lo"] > .50 and sF["est"] <= .30 else
                                              "it enters through full attention among image tokens" if sF["est"] >= .70 and sF["lo"] > .50 and sDN["est"] <= .30 else "both")
                                             if ok else "not read"))
top = max(DEPTH, key=lambda X: bs_[X]["est"])
res["C2"] = dict(s_depth={X: bs_[X] for X in DEPTH}, top=top, verdict=((f"it enters mainly at layers {DEPTH[top]}" if bs_[top]["est"] >= .50 and bs_[top]["lo"] > .30 else "distributed")
                                                                     if ok else "not read"))
res["descriptive"] = dict(
    s={f"L{L}": {X: sX(allx, X, L) for X in ROUTES} for L in LAYS},
    value_blocked={X: M(allx, lambda d, X=X: 1 - d["vd"][X]["along"]) for X in ROUTES},
    value_rel={X: M(allx, lambda d, X=X: d["vd"][X]["rel"]) for X in ROUTES + ("H",)},
    dHP_over_W={L: M(allx, lambda d, L=L: d[f"dHP_{L}"]) / M(allx, lambda d, L=L: d[f"W_{L}"]) for L in LAYS},
    answers={X: dict(P_pm1=float(np.mean([d["land"][f"Ep{d['p']}_{X}"] == d["p"] - 1 for d in REC.values()])),
                     P_p=float(np.mean([d["land"][f"Ep{d['p']}_{X}"] == d["p"] for d in REC.values()])))
             for X in ("HPV",) + tuple(f"HPV_{X}" for X in ROUTES)})
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"]["all"], C1=[res["C1"]["verdict"], sF["est"], sDN["est"]], C2=[res["C2"]["verdict"], top, bs_[top]["est"]],
                      dHP_over_W=res["I0"]["dHP_over_W"]), indent=1))
print("->", OUT)
