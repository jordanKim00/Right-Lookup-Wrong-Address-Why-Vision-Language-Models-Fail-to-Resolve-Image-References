"""S882 analysis (rules fixed before the data; knowledge/DECISIONS.md S882 entry).  Input results/r3/s882_<tag>_core.json.
W(arm, L) = T1_L(arm) - T1_L(E_BANDR), T1 = z_(p-1) - z_p (matcher-head mean block scores, rows B1 / B4 / answer); W_E = the full writer.
Within-pin Pearson over (scene, pin) and ratio of means; scene bootstrap 4,000 (seed 882).  Primary layer L51.
I0: t1 >= .95; START(p) answers p >= .95; L47 anchor = p for E and R >= .90; the full writer is present (mean W_E(L51) > 0 with the interval above 0).
F1 (sufficiency): r(W_KEEP_CORE, W_E) >= .80 with the lower bound > .65 and mean W_KEEP_CORE / mean W_E >= .70 -> 'the core circuit (h16 +
  self-address + anti-self heads + MLP46) is sufficient for the writer's per-scene address effect'; else 'not'.
F2 (division of labour): r(W_KEEP_H16, W_E) <= .30 with mean W_KEEP_H16 / mean W_E >= .15, and r(W_KEEP_ANCH, W_E) >= .60 -> 'h16 carries a
  scene-constant shift and the anchor heads + MLP46 carry the per-scene variation'; else 'not'.
Descriptive: KEEP_FA47; L55 / L59 / L63; answers per arm (P(answer = p - 1), P(= p)).
Usage: python3 s882_analyze.py TAG"""
from __future__ import annotations

import collections, json, sys
import numpy as np

TAG = sys.argv[1]
J = json.load(open(f"/vlm/results/r3/s882_{TAG}_core.json"))
OUT = f"/vlm/results/r3/s882_analysis_{TAG}.json"
rng = np.random.default_rng(882); NB = 4000
rows = J["rows"]; S = len(rows); PINS = J["pins"]; lm = J["late_mean"]; LAYS = [str(l) for l in J["lateq"]]
ARMS = ("E", "KEEP_CORE", "KEEP_H16", "KEEP_ANCH", "KEEP_FA47")


def T1(r, key, p, L):
    z = np.asarray(r["zl"][key][L], dtype=float)[:, lm[L]].mean(1).mean(0); return z[p - 2] - z[p - 1]


def anchor(q):
    q = np.asarray(q); am = [int(np.argmax(q[i])) + 1 for i in (2, 5, 9)]
    c = collections.Counter(am).most_common(); return c[0][0] if c[0][1] >= 2 else am[2]


REC = {}
for si, r in enumerate(rows):
    for p in PINS:
        d = dict(si=si, p=p)
        for L in LAYS:
            tb = T1(r, f"Ep{p}_E_BANDR", p, L)
            for a_ in ARMS:
                d[f"W_{a_}_{L}"] = T1(r, f"Ep{p}" if a_ == "E" else f"Ep{p}_{a_}", p, L) - tb
        for a_ in ARMS:
            d[f"x_{a_}"] = r["land"][f"Ep{p}" if a_ == "E" else f"Ep{p}_{a_}"]
        d["anch"] = (anchor(r["qk47"][f"Ep{p}"]), anchor(r["qk47"][f"Rp{p}"]))
        REC[(si, p)] = d
SC = sorted({k[0] for k in REC})


def bt(fn):
    n = len(SC); est = fn(np.arange(n)); bs = [fn(rng.integers(0, n, n)) for _ in range(NB)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return dict(est=float(est), lo=float(lo), hi=float(hi))


def corr(ix, a_, L="51"):
    x, y = [], []
    for p in PINS:
        u = np.array([REC[(SC[i], p)][f"W_{a_}_{L}"] for i in ix]); w = np.array([REC[(SC[i], p)][f"W_E_{L}"] for i in ix])
        x += list(u - u.mean()); y += list(w - w.mean())
    return float(np.corrcoef(x, y)[0, 1])


def ratio(ix, a_, L="51"):
    return float(np.mean([REC[(SC[i], p)][f"W_{a_}_{L}"] for i in ix for p in PINS]) / np.mean([REC[(SC[i], p)][f"W_E_{L}"] for i in ix for p in PINS]))


res = dict(tag=TAG, n_scenes=S)
st = float(np.mean([r["land"][f"START{p}"] == p for r in rows for p in PINS]))
reach = float(np.mean([a_ == d["p"] for d in REC.values() for a_ in d["anch"]]))
wE = bt(lambda ix: float(np.mean([REC[(SC[i], p)]["W_E_51"] for i in ix for p in PINS])))
I0 = dict(t1=bool((J["t1_menu"] or 0) >= .95), START=bool(st >= .95), pin_reach=bool(reach >= .90), writer_present=bool(wE["est"] > 0 and wE["lo"] > 0))
res["I0"] = dict(**I0, all=bool(all(I0.values())), t1_value=J["t1_menu"], START_acc=st, reach=reach, mean_W_E=wE)
ok = res["I0"]["all"]
f1r = bt(lambda ix: corr(ix, "KEEP_CORE")); f1m = bt(lambda ix: ratio(ix, "KEEP_CORE"))
res["F1"] = dict(r=f1r, ratio=f1m, verdict=(("the core circuit (h16 + self-address + anti-self heads + MLP46) is sufficient for the writer's per-scene address effect"
                                             if f1r["est"] >= .80 and f1r["lo"] > .65 and f1m["est"] >= .70 else "not") if ok else "not read"))
h16r, h16m, anr = bt(lambda ix: corr(ix, "KEEP_H16")), bt(lambda ix: ratio(ix, "KEEP_H16")), bt(lambda ix: corr(ix, "KEEP_ANCH"))
res["F2"] = dict(r_h16=h16r, ratio_h16=h16m, r_anch=anr, ratio_anch=bt(lambda ix: ratio(ix, "KEEP_ANCH")),
                 verdict=(("h16 carries a scene-constant shift and the anchor heads + MLP46 carry the per-scene variation"
                           if h16r["est"] <= .30 and h16m["est"] >= .15 and anr["est"] >= .60 else "not") if ok else "not read"))
desc = {f"L{L}": {a_: dict(r=corr(range(len(SC)), a_, L), ratio=ratio(range(len(SC)), a_, L)) for a_ in ARMS[1:]} for L in LAYS}
desc["answers"] = {a_: dict(P_pm1=float(np.mean([d[f"x_{a_}"] == d["p"] - 1 for d in REC.values()])), P_p=float(np.mean([d[f"x_{a_}"] == d["p"] for d in REC.values()])))
                   for a_ in ARMS}
res["descriptive"] = desc
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"]["all"], F1=[res["F1"]["verdict"], f1r["est"], f1m["est"]], F2=[res["F2"]["verdict"], h16r["est"], h16m["est"], anr["est"]]), indent=1))
print("->", OUT)
