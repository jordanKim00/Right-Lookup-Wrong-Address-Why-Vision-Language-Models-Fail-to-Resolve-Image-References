"""S889 analysis (rules fixed before the data; knowledge/DECISIONS.md S889 entry).  Input results/r3/s889_<tag>_context.json.
z_b = matcher-head mean LSE block score (rows B1 / B4 / answer); W(b) = z_b(E) - z_b(E_BANDR); under a context transplant X (the anchor's FA47
values at the read rows <- the same positions' values of a hybrid scene that differs only in slot p + offset): W_X(b) = z_b(E_X) - z_b(E_X_E_BANDR).
Step W(T1) = W(p-1) - W(p); change dX = W_X(T1) - W(T1), and the predecessor's own change dX(p-1) = W_X(p-1) - W(p-1).  Ratios to the mean
natural step; scene bootstrap 4,000 (seed 889) over (scene, pin).
I0: t1 >= .95; START(p) answers p >= .95; L47 anchor = p for E and R >= .90; the HS control (the changed picture follows the anchor, so the
  anchor's values cannot change) reproduces E, R and E_BANDR (max late block-score difference <= 1e-3 in >= .99 of (scene, pin)); the writer is
  present (mean W(T1) > 0, interval above 0).  Nothing is read if I0 fails.
K1 (context): mean dHP / mean W(T1) <= -.10 with the upper bound < 0 -> 'the writer's step depends on the predecessor context carried in the
  anchor's FA47 values'; else 'not observed'.
K2 (specificity): mean(dHP - dHPP) / mean W(T1) <= -.05 with the upper bound < 0 -> 'specifically the immediate predecessor's context'; else
  'not specific'.
Descriptive: dX(p-1), dX(p) per source, L55-63, answers (P(p - 1), P(p)), the hybrids' own natural answers.
Usage: python3 s889_analyze.py TAG"""
from __future__ import annotations

import collections, json, sys
import numpy as np

TAG = sys.argv[1]
J = json.load(open(f"/vlm/results/r3/s889_{TAG}_context.json"))
OUT = f"/vlm/results/r3/s889_analysis_{TAG}.json"
rng = np.random.default_rng(889); NB = 4000
rows = J["rows"]; S = len(rows); PINS = J["pins"]; lm = J["late_mean"]; LAYS = [str(l) for l in J["lateq"]]
SRCS = ("HP", "HPP", "HS")


def zv(r, key, L):
    return np.asarray(r["zl"][key][L], dtype=float)[:, lm[L]].mean(1).mean(0)


def anchor(q):
    q = np.asarray(q); am = [int(np.argmax(q[i])) + 1 for i in (2, 5, 9)]
    c = collections.Counter(am).most_common(); return c[0][0] if c[0][1] >= 2 else am[2]


REC = {}
for si, r in enumerate(rows):
    for p in PINS:
        d = dict(si=si, p=p)
        for L in LAYS:
            W = zv(r, f"Ep{p}", L) - zv(r, f"Ep{p}_E_BANDR", L); d[f"W_{L}"] = W
            for X in SRCS:
                d[f"W{X}_{L}"] = zv(r, f"Ep{p}_{X}", L) - zv(r, f"Ep{p}_{X}_E_BANDR", L)
        d["hs"] = max(float(np.abs(zv(r, f"{a_}p{p}_HS{s_}", L) - zv(r, f"{a_}p{p}{s_}", L)).max()) for L in LAYS for a_, s_ in (("E", ""), ("R", ""), ("E", "_E_BANDR")))
        d["anch"] = (anchor(r["qk47"][f"Ep{p}"]), anchor(r["qk47"][f"Rp{p}"]))
        d["land"] = {k: v for k, v in r["land"].items() if f"p{p}" in k}
        REC[(si, p)] = d
SC = sorted({k[0] for k in REC})


def bt(fn):
    n = len(SC); est = fn(np.arange(n)); bs = [fn(rng.integers(0, n, n)) for _ in range(NB)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return dict(est=float(est), lo=float(lo), hi=float(hi))


def M(ix, fn):
    return float(np.mean([fn(REC[(SC[i], p)]) for i in ix for p in PINS]))


t1 = lambda W, p: W[p - 2] - W[p - 1]
dX = lambda d, X, L="51": t1(d[f"W{X}_{L}"], d["p"]) - t1(d[f"W_{L}"], d["p"])
res = dict(tag=TAG, n_scenes=S)
st = float(np.mean([r["land"][f"START{p}"] == p for r in rows for p in PINS]))
reach = float(np.mean([a_ == d["p"] for d in REC.values() for a_ in d["anch"]]))
hs_ok = float(np.mean([d["hs"] <= 1e-3 for d in REC.values()]))
wT = bt(lambda ix: M(ix, lambda d: t1(d["W_51"], d["p"])))
I0 = dict(t1=bool((J["t1_menu"] or 0) >= .95), START=bool(st >= .95), pin_reach=bool(reach >= .90), causal_control=bool(hs_ok >= .99), writer=bool(wT["lo"] > 0))
res["I0"] = dict(**I0, all=bool(all(I0.values())), t1_value=J["t1_menu"], START_acc=st, reach=reach, hs_ok=hs_ok,
                 hs_max=float(max(d["hs"] for d in REC.values())), mean_W_T1=wT)
ok = res["I0"]["all"]
k1 = bt(lambda ix: M(ix, lambda d: dX(d, "HP")) / M(ix, lambda d: t1(d["W_51"], d["p"])))
res["K1"] = dict(ratio=k1, verdict=(("the writer's step depends on the predecessor context carried in the anchor's FA47 values" if k1["est"] <= -.10 and k1["hi"] < 0 else
                                     "not observed") if ok else "not read"))
k2 = bt(lambda ix: M(ix, lambda d: dX(d, "HP") - dX(d, "HPP")) / M(ix, lambda d: t1(d["W_51"], d["p"])))
res["K2"] = dict(ratio=k2, verdict=(("specifically the immediate predecessor's context" if k2["est"] <= -.05 and k2["hi"] < 0 else "not specific") if ok else "not read"))
allx = range(len(SC))
desc = {}
for L in LAYS:
    desc[f"L{L}"] = {X: dict(dT1=M(allx, lambda d, X=X, L=L: dX(d, X, L)) / M(allx, lambda d, L=L: t1(d[f"W_{L}"], d["p"])),
                             d_pred=M(allx, lambda d, X=X, L=L: d[f"W{X}_{L}"][d["p"] - 2] - d[f"W_{L}"][d["p"] - 2]),
                             d_anchor=M(allx, lambda d, X=X, L=L: d[f"W{X}_{L}"][d["p"] - 1] - d[f"W_{L}"][d["p"] - 1])) for X in SRCS}
arms = sorted({k.replace(f"p{d['p']}", "p") for d in REC.values() for k in d["land"]})
desc["answers"] = {a_: dict(P_pm1=float(np.mean([d["land"].get(a_.replace("p", f"p{d['p']}", 1)) == d["p"] - 1 for d in REC.values()])),
                            P_p=float(np.mean([d["land"].get(a_.replace("p", f"p{d['p']}", 1)) == d["p"] for d in REC.values()]))) for a_ in arms}
res["descriptive"] = desc
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"]["all"], K1=[res["K1"]["verdict"], k1["est"]], K2=[res["K2"]["verdict"], k2["est"]]), indent=1))
print("->", OUT)
