"""S909 analysis (rules fixed before the data; knowledge/DECISIONS.md S909 entry).  Input results/r3/s909_<tag>_modeN.json (one N per tag).
Natural runs at N (7 or 9).  For k in {2, 3, 5}: answers x of E_k (END(k)), S_k (START(k)), S_k + d_k(N8) (S906's vectors), S_k + d_k(N)
(own-N calibration), S_k + r_k (random).  g(X) = P(x(X) = x(E_k)); h(X) = P(x(X) = x(S_k)).  Scene bootstrap 4,000 (seed 909).
I0: t1 >= .95; START(k) answers k >= .95; END(k) differs from START(k) in >= .50 (pooled); the random arm stays START-like (h >= .80).  Nothing
  is read if I0 fails.
X1 (same vector?): mean over k of cos(d_k(N8), d_k(N)) >= .90 -> 'the END(k) transform vector is the same at N8 and N'; <= .60 -> 'N-specific';
  else 'partly shared'.
X2 (transfer): ratio = mean_k[g(S_k + d_k(N8)) - g(S_k)] / mean_k[g(S_k + d_k(N)) - g(S_k)] >= .70 with the lower bound > .40 -> 'the N8 vector
  produces END(k) at this N'; <= .30 with the upper bound < .50 -> 'not transferable across N'; else 'partial'.
X3 (relative or absolute; k in {2, 3}): d_rel = P(x(S_k + d_k(N8)) = N + 1 - k) - P(x(S_k + d_k(N8)) = 9 - k): > 0 with the interval above 0 ->
  'relative: the offset is resolved against the visible end'; < 0 with the interval below 0 -> 'absolute position'; else 'not separable'.
Descriptive: per k every arm's answer distribution, g and h; the per-row cosines; the same d_rel for E_k as a reference.
Usage: python3 s909_analyze.py TAG"""
from __future__ import annotations

import collections, json, sys
import numpy as np

TAG = sys.argv[1]
J = json.load(open(f"/vlm/results/r3/s909_{TAG}_modeN.json"))
OUT = f"/vlm/results/r3/s909_analysis_{TAG}.json"
rng = np.random.default_rng(909); NB = 4000
rows = J["rows"]; S = len(rows); KS = J["ks"]; CAL = J["calibration"]; N = J["N"]
ARMS = ("E", "S", "S_ADD8", "S_ADDN", "S_RAND")


def key(X, k):
    return f"{X[0]}{k}" + ("" if len(X) == 1 else X[1:])


REC = []
for si, r in enumerate(rows):
    for k in KS:
        REC.append(dict(si=si, k=k, **{X: r["land"][key(X, k)] for X in ARMS}))


def bt(fn):
    est = fn(np.arange(S)); bs = [fn(rng.integers(0, S, S)) for _ in range(NB)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return dict(est=float(est), lo=float(lo), hi=float(hi))


def P(ix, ks, cond):
    cnt = collections.Counter(int(i) for i in ix)
    return float(np.mean([cond(d) for d in REC if d["k"] in ks for _ in range(cnt.get(d["si"], 0))]))


g = lambda ix, k, X: P(ix, [k], lambda d: d[X] == d["E"])
h = lambda ix, k, X: P(ix, [k], lambda d: d[X] == d["S"])
res = dict(tag=TAG, N=N, n_scenes=S, calibration={k_: v_ for k_, v_ in CAL.items() if k_ != "row_norm"})
allx = np.arange(S)
st = float(np.mean([d["S"] == d["k"] for d in REC])); diff = float(np.mean([d["E"] != d["S"] for d in REC]))
hr = float(np.mean([h(allx, k, "S_RAND") for k in KS]))
I0 = dict(t1=bool((J["t1_menu"] or 0) >= .95), START=bool(st >= .95), END_differs=bool(diff >= .50), random_control=bool(hr >= .80))
res["I0"] = dict(**I0, all=bool(all(I0.values())), t1_value=J["t1_menu"], START_acc=st, END_differs_value=diff, h_random=hr)
ok = res["I0"]["all"]
cs = [CAL["cos_N8_vs_N"][str(k)] for k in KS]; x1 = float(np.mean(cs))
res["X1"] = dict(mean_cos=x1, cos=cs, verdict=(("the END(k) transform vector is the same at N8 and N" if x1 >= .90 else "N-specific" if x1 <= .60 else "partly shared")
                                              if ok else "not read"))
eff = lambda ix, X: float(np.mean([g(ix, k, X) - g(ix, k, "S") for k in KS]))
b2 = bt(lambda ix: eff(ix, "S_ADD8") / eff(ix, "S_ADDN"))
res["X2"] = dict(ratio=b2, own_effect=eff(allx, "S_ADDN"), n8_effect=eff(allx, "S_ADD8"),
                 verdict=(("the N8 vector produces END(k) at this N" if b2["est"] >= .70 and b2["lo"] > .40 else
                           "not transferable across N" if b2["est"] <= .30 and b2["hi"] < .50 else "partial") if ok else "not read"))
K23 = [k for k in KS if k in (2, 3)]
drel = lambda ix, X: P(ix, K23, lambda d: d[X] == N + 1 - d["k"]) - P(ix, K23, lambda d: d[X] == 9 - d["k"])
b3 = bt(lambda ix: drel(ix, "S_ADD8"))
res["X3"] = dict(d_rel=b3, d_rel_END=drel(allx, "E"),
                 verdict=(("relative: the offset is resolved against the visible end" if b3["lo"] > 0 else "absolute position" if b3["hi"] < 0 else "not separable")
                          if ok else "not read"))
desc = {}
for k in KS:
    ds = [d for d in REC if d["k"] == k]
    desc[f"k{k}"] = {X: dict(dist=dict(collections.Counter(d[X] for d in ds)), g=float(np.mean([d[X] == d["E"] for d in ds])),
                             h=float(np.mean([d[X] == d["S"] for d in ds]))) for X in ARMS}
res["descriptive"] = desc
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"]["all"], X1=[res["X1"]["verdict"], x1], X2=[res["X2"]["verdict"], b2["est"]], X3=[res["X3"]["verdict"], b3["est"]]), indent=1))
print("->", OUT)
