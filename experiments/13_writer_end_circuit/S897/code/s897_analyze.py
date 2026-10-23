"""S897 analysis (rules fixed before the data; knowledge/DECISIONS.md S897 entry).  Input results/r3/s897_<tag>_qpathheads.json.
W_X = T1(X) - T1(E_BANDR) at L51; FA47's END part F = W_E - W_ALL47R; path share f_X = (W_E - W_X) / F (ratios of means over (scene, pin));
scene bootstrap 4,000 (seed 890).
I0: t1 >= .95; START(p) answers p >= .95; L47 anchor = p for E and R >= .90; QID reproduces E (max late block-score difference <= 1e-3 in >= .99);
  F > 0 with the interval above 0; the full path E_QM is present (f_QM > 0 with the interval above 0).  Nothing is read if I0 fails.
H1 (which single head): among h3, h14, h23, h10, h18, h19 the head with the largest f; its share of f_QM_REST >= .50 with the bootstrap lower
  bound > .30 -> 'head h<i> carries most of MLP46's REST path'; else 'distributed'.
Descriptive: every f_X at L51-63 (the REST group replicates S890's .41), the sum of single heads + REST7 against REST, answers per arm.
Usage: python3 s890_analyze.py TAG"""
from __future__ import annotations

import collections, json, sys
import numpy as np

TAG = sys.argv[1]
J = json.load(open(f"/vlm/results/r3/s897_{TAG}_qpathheads.json"))
OUT = f"/vlm/results/r3/s897_analysis_{TAG}.json"
rng = np.random.default_rng(897); NB = 4000
rows = J["rows"]; S = len(rows); PINS = J["pins"]; lm = J["late_mean"]; LAYS = [str(l) for l in J["lateq"]]
SA = J["selfaddr"]
ARMS = ("ALL47R", "QM_REST", "QM_h3", "QM_h14", "QM_h23", "QM_h10", "QM_h18", "QM_h19", "QM_REST7", "QID")
GROUPS = ("QM_h3", "QM_h14", "QM_h23", "QM_h10", "QM_h18", "QM_h19")


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
            tb = T1(r, f"Ep{p}_E_BANDR", p, L); d[f"W_E_{L}"] = T1(r, f"Ep{p}", p, L) - tb
            for X in ARMS:
                d[f"W_{X}_{L}"] = T1(r, f"Ep{p}_{X}", p, L) - tb
        d["vid"] = max(float(np.abs(zv(r, f"Ep{p}_QID", L) - zv(r, f"Ep{p}", L)).max()) for L in LAYS)
        d["att"] = {X: float(np.asarray(r["mass47"][f"Ep{p}" + ("" if X == "E" else f"_{X}")], float)[:, SA, p - 1].mean()) for X in ("E",) + ARMS}
        d["attR"] = float(np.asarray(r["mass47"][f"Rp{p}"], float)[:, SA, p - 1].mean())
        d["anch"] = (anchor(r["qk47"][f"Ep{p}"]), anchor(r["qk47"][f"Rp{p}"]))
        d["land"] = {k: v for k, v in r["land"].items() if f"p{p}" in k}
        REC[(si, p)] = d
SC = sorted({k[0] for k in REC})


def bt(fn):
    n = len(SC); est = fn(np.arange(n)); bs = [fn(rng.integers(0, n, n)) for _ in range(NB)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return dict(est=float(est), lo=float(lo), hi=float(hi))


def M(ix, fn):
    return float(np.mean([fn(REC[(SC[i], p)]) for i in ix for p in PINS]))


F_ = lambda d, L="51": d[f"W_E_{L}"] - d[f"W_ALL47R_{L}"]
fX = lambda ix, X, L="51": M(ix, lambda d: d[f"W_E_{L}"] - d[f"W_{X}_{L}"]) / M(ix, lambda d: F_(d, L))
res = dict(tag=TAG, n_scenes=S)
st = float(np.mean([r["land"][f"START{p}"] == p for r in rows for p in PINS]))
reach = float(np.mean([a_ == d["p"] for d in REC.values() for a_ in d["anch"]]))
vid_ok = float(np.mean([d["vid"] <= 1e-3 for d in REC.values()]))
bF = bt(lambda ix: M(ix, F_)); bQ = bt(lambda ix: fX(ix, "QM_REST"))
I0 = dict(t1=bool((J["t1_menu"] or 0) >= .95), START=bool(st >= .95), pin_reach=bool(reach >= .90), identity_path=bool(vid_ok >= .99),
          fa47_part=bool(bF["lo"] > 0), path_present=bool(bQ["lo"] > 0))
res["I0"] = dict(**I0, all=bool(all(I0.values())), t1_value=J["t1_menu"], START_acc=st, reach=reach, vid_ok=vid_ok, F=bF, f_QM=bQ)
ok = res["I0"]["all"]
fgroups = {X: fX(range(len(SC)), X) for X in GROUPS}
top = max(GROUPS, key=lambda X: fgroups[X])
sh = bt(lambda ix: fX(ix, top) / fX(ix, "QM_REST"))
res["H1"] = dict(f_singles=fgroups, top=top, share_of_REST=sh, verdict=((f"head {top.split('_')[1]} carries most of MLP46's REST path" if sh["est"] >= .50 and sh["lo"] > .30 else "distributed")
                                                                     if ok else "not read"))
allx = range(len(SC))
res["descriptive"] = dict(f={f"L{L}": {X: fX(allx, X, L) for X in ARMS if X not in ("ALL47R", "QID")} for L in LAYS},
                          selfaddr_anchor_attention={**{X: M(allx, lambda d, X=X: d["att"][X]) for X in ("E",) + ARMS}, "R": M(allx, lambda d: d["attR"])},
                          mean_W={X: M(allx, lambda d, X=X: d[f"W_{X}_51"]) for X in ARMS} | {"E": M(allx, lambda d: d["W_E_51"])},
                          answers={X: dict(P_pm1=float(np.mean([d["land"][f"Ep{d['p']}_{X}"] == d["p"] - 1 for d in REC.values()])),
                                           P_p=float(np.mean([d["land"][f"Ep{d['p']}_{X}"] == d["p"] for d in REC.values()]))) for X in ARMS})
json.dump(res, open(OUT, "w"), indent=1)
res["descriptive"]["sum_singles_plus_rest7"] = sum(fX(range(len(SC)), X) for X in GROUPS + ("QM_REST7",))
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"]["all"], H1=[res["H1"]["verdict"], top, sh["est"]], f_REST=bQ["est"]), indent=1))
print("->", OUT)
