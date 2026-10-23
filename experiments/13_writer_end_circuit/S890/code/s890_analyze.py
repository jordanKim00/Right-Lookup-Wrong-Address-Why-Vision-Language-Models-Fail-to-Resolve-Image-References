"""S890 analysis (rules fixed before the data; knowledge/DECISIONS.md S890 entry).  Input results/r3/s890_<tag>_qpathsplit.json.
W_X = T1(X) - T1(E_BANDR) at L51; FA47's END part F = W_E - W_ALL47R; path share f_X = (W_E - W_X) / F (ratios of means over (scene, pin));
scene bootstrap 4,000 (seed 890).
I0: t1 >= .95; START(p) answers p >= .95; L47 anchor = p for E and R >= .90; QID reproduces E (max late block-score difference <= 1e-3 in >= .99);
  F > 0 with the interval above 0; the full path E_QM is present (f_QM > 0 with the interval above 0).  Nothing is read if I0 fails.
R1 (queries or gates): f_QMg >= .50 with the lower bound > .25 and f_QMq <= .20 -> 'the output gates carry MLP46's path into FA47';
  f_QMq >= .50 with the lower bound > .25 and f_QMg <= .20 -> 'the queries carry it'; else 'both'.
R2 (which heads): among SA, ANTI, H16, REST the group with the largest f_QM_group; its share of f_QM >= .50 with the bootstrap lower bound > .30
  -> 'the <group> heads carry MLP46's path'; else 'distributed'.
Descriptive: every f_X at L51-63, f_QMg_SA, the self-address anchor attention per arm, answers per arm.
Usage: python3 s890_analyze.py TAG"""
from __future__ import annotations

import collections, json, sys
import numpy as np

TAG = sys.argv[1]
J = json.load(open(f"/vlm/results/r3/s890_{TAG}_qpathsplit.json"))
OUT = f"/vlm/results/r3/s890_analysis_{TAG}.json"
rng = np.random.default_rng(890); NB = 4000
rows = J["rows"]; S = len(rows); PINS = J["pins"]; lm = J["late_mean"]; LAYS = [str(l) for l in J["lateq"]]
SA = J["selfaddr"]
ARMS = ("ALL47R", "QM", "QMq", "QMg", "QM_SA", "QM_ANTI", "QM_H16", "QM_REST", "QMg_SA", "QID")
GROUPS = ("QM_SA", "QM_ANTI", "QM_H16", "QM_REST")


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
bF = bt(lambda ix: M(ix, F_)); bQ = bt(lambda ix: fX(ix, "QM"))
I0 = dict(t1=bool((J["t1_menu"] or 0) >= .95), START=bool(st >= .95), pin_reach=bool(reach >= .90), identity_path=bool(vid_ok >= .99),
          fa47_part=bool(bF["lo"] > 0), path_present=bool(bQ["lo"] > 0))
res["I0"] = dict(**I0, all=bool(all(I0.values())), t1_value=J["t1_menu"], START_acc=st, reach=reach, vid_ok=vid_ok, F=bF, f_QM=bQ)
ok = res["I0"]["all"]
fq, fg = bt(lambda ix: fX(ix, "QMq")), bt(lambda ix: fX(ix, "QMg"))
res["R1"] = dict(f_QMq=fq, f_QMg=fg, verdict=(("the output gates carry MLP46's path into FA47" if fg["est"] >= .50 and fg["lo"] > .25 and fq["est"] <= .20 else
                                               "the queries carry it" if fq["est"] >= .50 and fq["lo"] > .25 and fg["est"] <= .20 else "both") if ok else "not read"))
fgroups = {X: fX(range(len(SC)), X) for X in GROUPS}
top = max(GROUPS, key=lambda X: fgroups[X])
sh = bt(lambda ix: fX(ix, top) / fX(ix, "QM"))
res["R2"] = dict(f_groups=fgroups, top=top, share_of_QM=sh, verdict=((f"the {top.split('_')[1]} heads carry MLP46's path" if sh["est"] >= .50 and sh["lo"] > .30 else "distributed")
                                                                   if ok else "not read"))
allx = range(len(SC))
res["descriptive"] = dict(f={f"L{L}": {X: fX(allx, X, L) for X in ARMS if X not in ("ALL47R", "QID")} for L in LAYS},
                          selfaddr_anchor_attention={**{X: M(allx, lambda d, X=X: d["att"][X]) for X in ("E",) + ARMS}, "R": M(allx, lambda d: d["attR"])},
                          mean_W={X: M(allx, lambda d, X=X: d[f"W_{X}_51"]) for X in ARMS} | {"E": M(allx, lambda d: d["W_E_51"])},
                          answers={X: dict(P_pm1=float(np.mean([d["land"][f"Ep{d['p']}_{X}"] == d["p"] - 1 for d in REC.values()])),
                                           P_p=float(np.mean([d["land"][f"Ep{d['p']}_{X}"] == d["p"] for d in REC.values()]))) for X in ARMS})
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"]["all"], R1=[res["R1"]["verdict"], fq["est"], fg["est"]], R2=[res["R2"]["verdict"], top, sh["est"]]), indent=1))
print("->", OUT)
