"""S906 analysis (rules fixed before the data; knowledge/DECISIONS.md S906 entry).  Input results/r3/s906_<tag>_modek.json.
Natural runs.  For each test scene and k in {2, 3, 5}: answers x of E_k (END(k)), S_k (START(k)), S_k + d_k, S_k + r_k (random), E_k - d_k, and
for k in {2, 3}: S_k + d_5, E_k - d_5 (d = calibration mean END(k) - START(k) at the layer-35 input, read rows).  Agreement with END's own
answer: g(X) = P(x(X) = x(E_k)); with START's: h(X) = P(x(X) = x(S_k)).  Scene bootstrap 4,000 (seed 906).
I0: t1 >= .95; START(k) answers k >= .95; END(k) differs from START(k) (P(x(E_k) != x(S_k)) >= .50 pooled); S_k + r_k stays START-like
  (h(S_k + r_k) >= .80 pooled).  Nothing is read if I0 fails.
M1 (one vector or several): mean of the three cosines between d_2, d_3, d_5 (calibration) >= .80 -> 'one shared END mode vector across k';
  <= .50 -> 'k-specific mode vectors'; else 'partly shared'.
M2 (own vector, sufficiency): A_own = mean over k of [g(S_k + d_k) - g(S_k)] >= .30 with the lower bound > .15 -> 'adding the k's own mode vector
  turns START(k) into END(k)'; else 'not'.
M3 (transfer): ratio = mean over k in {2, 3} of [g(S_k + d_5) - g(S_k)] / mean over k in {2, 3} of [g(S_k + d_k) - g(S_k)]: >= .70 with the lower
  bound > .40 -> 'the k = 5 vector transfers: END(k) = START(k) + a k-independent mode vector'; <= .30 with the upper bound < .50 -> 'not
  transferable (k-specific)'; else 'partial'.
M4 (own vector, necessity): B_own = mean over k of [h(E_k - d_k) - h(E_k)] >= .30 with the lower bound > .15 -> 'removing the own vector turns
  END(k) into START(k)'; else 'not'.
Descriptive: per k every arm's answer distribution, g and h; cross removal E_k - d_5; the late block-score profile correlation (L51 / L63,
matcher-head mean, rows B1 / B4 / answer) of each arm with E_k; the calibration cosines and energy shares.
Usage: python3 s906_analyze.py TAG"""
from __future__ import annotations

import collections, json, sys
import numpy as np

TAG = sys.argv[1]
J = json.load(open(f"/vlm/results/r3/s906_{TAG}_modek.json"))
OUT = f"/vlm/results/r3/s906_analysis_{TAG}.json"
rng = np.random.default_rng(906); NB = 4000
rows = J["rows"]; S = len(rows); lm = J["late_mean"]; KS = J["ks"]; CAL = J["calibration"]
ARMS = ("E", "S", "S_ADD", "S_RAND", "E_SUB", "S_ADD5", "E_SUB5")


def key(X, k):
    return f"{X[0]}{k}" + ("" if len(X) == 1 else X[1:])


def prof(r, k_, L):
    return np.asarray(r["zl"][k_][L], dtype=float)[:, lm[L]].mean(1).mean(0)


REC = []
for si, r in enumerate(rows):
    for k in KS:
        d = dict(si=si, k=k)
        for X in ARMS:
            kk = key(X, k)
            if kk in r["land"]:
                d[X] = r["land"][kk]
                d[f"r51_{X}"] = float(np.corrcoef(prof(r, kk, "51"), prof(r, f"E{k}", "51"))[0, 1])
                d[f"r63_{X}"] = float(np.corrcoef(prof(r, kk, "63"), prof(r, f"E{k}", "63"))[0, 1])
        REC.append(d)


def bt(fn):
    est = fn(np.arange(S)); bs = [fn(rng.integers(0, S, S)) for _ in range(NB)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return dict(est=float(est), lo=float(lo), hi=float(hi))


def P(ix, k, cond):
    cnt = collections.Counter(int(i) for i in ix)
    v = [cond(d) for d in REC if d["k"] == k for _ in range(cnt.get(d["si"], 0))]
    return float(np.mean(v))


g = lambda ix, k, X: P(ix, k, lambda d: d[X] == d["E"])
h = lambda ix, k, X: P(ix, k, lambda d: d[X] == d["S"])
res = dict(tag=TAG, n_scenes=S, calibration={k_: v_ for k_, v_ in CAL.items() if k_ != "row_norm"})
allx = np.arange(S)
st = float(np.mean([d["S"] == d["k"] for d in REC])); diff = float(np.mean([d["E"] != d["S"] for d in REC]))
hr = float(np.mean([h(allx, k, "S_RAND") for k in KS]))
I0 = dict(t1=bool((J["t1_menu"] or 0) >= .95), START=bool(st >= .95), END_differs=bool(diff >= .50), random_control=bool(hr >= .80))
res["I0"] = dict(**I0, all=bool(all(I0.values())), t1_value=J["t1_menu"], START_acc=st, END_differs_value=diff, h_random=hr)
ok = res["I0"]["all"]
cb = [CAL["cos_between"][f"{a}_{b}"] for i, a in enumerate(KS) for b in KS[i + 1:]]
m1 = float(np.mean(cb))
res["M1"] = dict(mean_cos=m1, cos=cb, verdict=(("one shared END mode vector across k" if m1 >= .80 else "k-specific mode vectors" if m1 <= .50 else "partly shared")
                                              if ok else "not read"))
own = lambda ix, ks=KS: float(np.mean([g(ix, k, "S_ADD") - g(ix, k, "S") for k in ks]))
b2 = bt(own)
res["M2"] = dict(A_own=b2, verdict=(("adding the k's own mode vector turns START(k) into END(k)" if b2["est"] >= .30 and b2["lo"] > .15 else "not") if ok else "not read"))
KX = [k for k in KS if k != 5]
b3 = bt(lambda ix: float(np.mean([g(ix, k, "S_ADD5") - g(ix, k, "S") for k in KX])) / own(ix, KX))
res["M3"] = dict(ratio=b3, verdict=(("the k = 5 vector transfers: END(k) = START(k) + a k-independent mode vector" if b3["est"] >= .70 and b3["lo"] > .40 else
                                     "not transferable (k-specific)" if b3["est"] <= .30 and b3["hi"] < .50 else "partial") if ok else "not read"))
b4 = bt(lambda ix: float(np.mean([h(ix, k, "E_SUB") - h(ix, k, "E") for k in KS])))
res["M4"] = dict(B_own=b4, verdict=(("removing the own vector turns END(k) into START(k)" if b4["est"] >= .30 and b4["lo"] > .15 else "not") if ok else "not read"))
desc = {}
for k in KS:
    ds = [d for d in REC if d["k"] == k]
    desc[f"k{k}"] = {X: dict(dist=dict(collections.Counter(d[X] for d in ds)), g=float(np.mean([d[X] == d["E"] for d in ds])),
                             h=float(np.mean([d[X] == d["S"] for d in ds])), r51=float(np.mean([d[f"r51_{X}"] for d in ds])),
                             r63=float(np.mean([d[f"r63_{X}"] for d in ds])))
                     for X in ARMS if X in ds[0]}
res["descriptive"] = desc
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"]["all"], M1=[res["M1"]["verdict"], m1], M2=[res["M2"]["verdict"], b2["est"]], M3=[res["M3"]["verdict"], b3["est"]],
                      M4=[res["M4"]["verdict"], b4["est"]]), indent=1))
print("->", OUT)
