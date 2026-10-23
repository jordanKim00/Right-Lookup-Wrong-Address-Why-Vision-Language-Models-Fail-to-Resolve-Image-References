"""S892 analysis (rules fixed before the data; knowledge/DECISIONS.md S892 entry).  Input results/r3/s892_<tag>_natural.json.
Natural END(k) runs at N 8 (target 9 - k) with their natural L47 anchor a (2 <= a <= 8) and the references at a: R = START(a), E_BANDR (band <-
R), E_ALL47R (FA47 <- R), E_QM (MLP46 <- R in FA47's queries and gates).  T1_a = z(a - 1) - z(a) (matcher-head mean LSE block scores at L51,
rows B1 / B4 / answer); W = T1_a(E) - T1_a(E_BANDR); F = W - W_ALL47R; f_QM = (W - W_QM) / F (ratios of means).  Self-address heads
{1, 4, 12, 17, 20}: attention mass on block a, mean over the three rows and the five heads, in E and R.  Scene bootstrap 4,000 (seed 892).
I0: t1 >= .95; natural START(q) answers q in >= .95 of runs; the anchor reference holds (START(a)'s own L47 anchor = a in >= .90 of the used
  references); at least 60% of (scene, k) cells with k >= 3 have a valid anchor.  Nothing is read if I0 fails.
N1: mean over runs with k >= 3 of [a_SA(R) - a_SA(E)] > 0 with the interval above 0 -> 'natural END runs show the self-address heads' anchor
  re-affirmation weakening'; else 'not observed'.
N2: mean W(k >= 3) - mean W(k = 2) > 0 with the interval above 0 -> 'the writer's step is k-gated in natural runs'; else 'not observed'.
N3: f_QM pooled over k >= 3 >= .50 with the lower bound > .25 -> 'layer 46's MLP output, read through FA47's queries and gates, carries most of
  FA47's END part in natural runs'; <= .15 with the upper bound < .30 -> 'not'; else 'partial'.
Descriptive: per k — anchor distribution, answers (= a - 1, = a, correct), W, F, f_QM, the attention drop; the share of natural errors that land
on a - 1.
Usage: python3 s892_analyze.py TAG"""
from __future__ import annotations

import collections, json, sys
import numpy as np

TAG = sys.argv[1]
J = json.load(open(f"/vlm/results/r3/s892_{TAG}_natural.json"))
OUT = f"/vlm/results/r3/s892_analysis_{TAG}.json"
rng = np.random.default_rng(892); NB = 4000
rows = J["rows"]; S = len(rows); lm = J["late_mean"]; KS = J["ks"]; SA = J["selfaddr"]; N = J["N"]


def zv(r, key, L="51"):
    return np.asarray(r["zl"][key][L], dtype=float)[:, lm[L]].mean(1).mean(0)


def anchor(q):
    q = np.asarray(q); am = [int(np.argmax(q[i])) + 1 for i in (2, 5, 9)]
    c = collections.Counter(am).most_common(); return c[0][0] if c[0][1] >= 2 else am[2]


REC = []
for si, r in enumerate(rows):
    for k in KS:
        a_ = r["anchor"][str(k)]
        d = dict(si=si, k=k, a=a_, x=r["land"][f"N_E{k}"], target=N + 1 - k, valid=f"B{k}_E" in r["zl"])
        if d["valid"]:
            t = lambda key: (lambda z: z[a_ - 2] - z[a_ - 1])(zv(r, key))
            d["W"] = t(f"B{k}_E") - t(f"B{k}_E_BANDR"); d["W47"] = t(f"B{k}_ALL47R") - t(f"B{k}_E_BANDR"); d["WQM"] = t(f"B{k}_QM") - t(f"B{k}_E_BANDR")
            mE = np.asarray(r["mass47"][f"B{k}_E"], float); mR = np.asarray(r["mass47"][f"B{k}_R"], float)
            d["aE"] = float(mE[:, SA, a_ - 1].mean()); d["aR"] = float(mR[:, SA, a_ - 1].mean())
            d["Ranch"] = anchor(r["qk47"][f"B{k}_R"]); d["xB"] = r["land"][f"B{k}_E"]
        REC.append(d)
SC = list(range(S))


def bt(fn):
    est = fn(np.arange(S)); bs = [fn(rng.integers(0, S, S)) for _ in range(NB)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return dict(est=float(est), lo=float(lo), hi=float(hi))


def sel(ix, cond):
    s_ = set(int(i) for i in ix); cnt = collections.Counter(int(i) for i in ix)
    out = []
    for d in REC:
        if d["si"] in s_ and d["valid"] and cond(d):
            out += [d] * cnt[d["si"]]
    return out


res = dict(tag=TAG, n_scenes=S)
st = float(np.mean([r["land"][f"N_S{q}"] == q for r in rows for q in range(1, N + 1)]))
refs = [d for d in REC if d["valid"]]; rr = float(np.mean([d["Ranch"] == d["a"] for d in refs]))
cov = float(np.mean([d["valid"] for d in REC if d["k"] >= 3]))
I0 = dict(t1=bool((J["t1_menu"] or 0) >= .95), START=bool(st >= .95), reference=bool(rr >= .90), coverage=bool(cov >= .60))
res["I0"] = dict(**I0, all=bool(all(I0.values())), t1_value=J["t1_menu"], START_acc=st, reference_anchor=rr, coverage_value=cov)
ok = res["I0"]["all"]
n1 = bt(lambda ix: float(np.mean([d["aR"] - d["aE"] for d in sel(ix, lambda d: d["k"] >= 3)])))
res["N1"] = dict(drop=n1, verdict=(("natural END runs show the self-address heads' anchor re-affirmation weakening" if n1["lo"] > 0 else "not observed") if ok else "not read"))
n2 = bt(lambda ix: float(np.mean([d["W"] for d in sel(ix, lambda d: d["k"] >= 3)]) - np.mean([d["W"] for d in sel(ix, lambda d: d["k"] == 2)])))
res["N2"] = dict(diff=n2, verdict=(("the writer's step is k-gated in natural runs" if n2["lo"] > 0 else "not observed") if ok else "not read"))
def fqm(ix):
    ds = sel(ix, lambda d: d["k"] >= 3)
    return float(np.mean([d["W"] - d["WQM"] for d in ds]) / np.mean([d["W"] - d["W47"] for d in ds]))
n3 = bt(fqm)
res["N3"] = dict(f_QM=n3, verdict=(("layer 46's MLP output, read through FA47's queries and gates, carries most of FA47's END part in natural runs" if n3["est"] >= .50 and n3["lo"] > .25 else
                                    "not" if n3["est"] <= .15 and n3["hi"] < .30 else "partial") if ok else "not read"))
desc = {}
for k in KS:
    ds = [d for d in REC if d["k"] == k]; dv = [d for d in ds if d["valid"]]
    desc[f"k{k}"] = dict(n=len(ds), n_valid=len(dv), anchors=dict(collections.Counter(d["a"] for d in ds)),
                         P_correct=float(np.mean([d["x"] == d["target"] for d in ds])), P_a_minus_1=float(np.mean([d["x"] == d["a"] - 1 for d in dv])) if dv else None,
                         P_a=float(np.mean([d["x"] == d["a"] for d in dv])) if dv else None,
                         W=float(np.mean([d["W"] for d in dv])) if dv else None, F=float(np.mean([d["W"] - d["W47"] for d in dv])) if dv else None,
                         f_QM=(float(np.mean([d["W"] - d["WQM"] for d in dv]) / np.mean([d["W"] - d["W47"] for d in dv])) if dv else None),
                         attention_drop=float(np.mean([d["aR"] - d["aE"] for d in dv])) if dv else None)
errs = [d for d in REC if d["valid"] and d["x"] != d["target"]]
desc["errors_landing_a_minus_1"] = float(np.mean([d["x"] == d["a"] - 1 for d in errs])) if errs else None
res["descriptive"] = desc
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"]["all"], N1=[res["N1"]["verdict"], n1["est"]], N2=[res["N2"]["verdict"], n2["est"]], N3=[res["N3"]["verdict"], n3["est"]]), indent=1))
print("->", OUT)
