"""S902 analysis (rules fixed before the data; knowledge/DECISIONS.md S902 entry).  Input results/r3/s899_<tag>_natural.json (exp/s899_natural_n.py
run at N 8 with ks 1, 2, 3 on new scenes).  Question: S892 found the writer's step on at every k from 2 to 6 (W at k = 2 as large as at k >= 3).
Is it on at END(1) - the last picture, which the model answers well - and if it is, why is END(1) right?
Per natural END(k) run with its natural L47 anchor a (2 <= a <= N) and the references at a: T1_a = z(a - 1) - z(a) (matcher-head mean LSE
block scores at L51, rows B1 / B4 / answer); W = T1_a(E) - T1_a(E_BANDR); F = W - W_ALL47R; f_QM = (W - W_QM) / F; base margin
B = T1_a(E_BANDR) (the late address without END's band: negative = stays on a).  Scene bootstrap 4,000 (seed 902).
I0: t1 >= .95; natural START(q) answers q in >= .95 of runs with q < N; the anchor reference holds (START(a)'s own L47 anchor = a in >= .90 of
  the used references); >= 60% of the (scene, k) cells are valid for every k.  Nothing is read if I0 fails.
K1 (the step at END(1)): rho1 = mean W(k = 1) / mean W(k = 2): rho1 <= .25 with the upper bound < .50 -> 'the writer's step is gated off at
  END(1)'; rho1 >= .75 with the lower bound > .50 -> 'the writer steps at END(1) as at END(2)'; else 'reduced at END(1)'.
K2 (behaviour): P(answer = a - 1 | k = 1) - P(answer = a - 1 | k = 2) < 0 with the interval below 0 -> 'END(1) answers leave the anchor less
  than END(2)'; else 'not'.
K3 (only read if K1 is not 'gated off'): mean B(k = 1) - mean B(k = 2) < 0 with the interval below 0 -> 'END(1)'s writer-free address holds the
  anchor more firmly, so the same step does not move it'; else 'not'.
Descriptive: per k - anchor distribution, answers (= a - 1, = a, correct), W, F, f_QM, B, T1_a(E), the self-address attention drop; the same
restricted to runs whose anchor is the last picture (a = N).
Usage: python3 s902_analyze.py TAG"""
from __future__ import annotations

import collections, json, sys
import numpy as np

TAG = sys.argv[1]
J = json.load(open(f"/vlm/results/r3/s899_{TAG}_natural.json"))
OUT = f"/vlm/results/r3/s902_analysis_{TAG}.json"
rng = np.random.default_rng(902); NB = 4000
rows = J["rows"]; S = len(rows); lm = J["late_mean"]; KS = J["ks"]; SA = J["selfaddr"]; N = J["N"]
assert 1 in KS and 2 in KS


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
            d["B"] = t(f"B{k}_E_BANDR"); d["TE"] = t(f"B{k}_E")
            d["W"] = d["TE"] - d["B"]; d["W47"] = t(f"B{k}_ALL47R") - d["B"]; d["WQM"] = t(f"B{k}_QM") - d["B"]
            mE = np.asarray(r["mass47"][f"B{k}_E"], float); mR = np.asarray(r["mass47"][f"B{k}_R"], float)
            d["aE"] = float(mE[:, SA, a_ - 1].mean()); d["aR"] = float(mR[:, SA, a_ - 1].mean())
            d["Ranch"] = anchor(r["qk47"][f"B{k}_R"])
        REC.append(d)


def bt(fn):
    est = fn(np.arange(S)); bs = [fn(rng.integers(0, S, S)) for _ in range(NB)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return dict(est=float(est), lo=float(lo), hi=float(hi))


def sel(ix, cond):
    cnt = collections.Counter(int(i) for i in ix)
    out = []
    for d in REC:
        if cnt.get(d["si"]) and d["valid"] and cond(d):
            out += [d] * cnt[d["si"]]
    return out


mk = lambda key, k: (lambda ix: float(np.mean([d[key] for d in sel(ix, lambda d: d["k"] == k)])))
res = dict(tag=TAG, n_scenes=S)
st = float(np.mean([r["land"][f"N_S{q}"] == q for r in rows for q in range(1, N)]))
stN = float(np.mean([r["land"][f"N_S{N}"] == N for r in rows]))
refs = [d for d in REC if d["valid"]]; rr = float(np.mean([d["Ranch"] == d["a"] for d in refs]))
cov = {k: float(np.mean([d["valid"] for d in REC if d["k"] == k])) for k in KS}
I0 = dict(t1=bool((J["t1_menu"] or 0) >= .95), START=bool(st >= .95), reference=bool(rr >= .90), coverage=bool(min(cov.values()) >= .60))
res["I0"] = dict(**I0, all=bool(all(I0.values())), t1_value=J["t1_menu"], START_acc=st, START_N_acc=stN, reference_anchor=rr, coverage_value=cov)
ok = res["I0"]["all"]
rho = bt(lambda ix: mk("W", 1)(ix) / mk("W", 2)(ix))
k1 = ("the writer's step is gated off at END(1)" if rho["est"] <= .25 and rho["hi"] < .50 else
      "the writer steps at END(1) as at END(2)" if rho["est"] >= .75 and rho["lo"] > .50 else "reduced at END(1)")
res["K1"] = dict(rho1=rho, W1=bt(mk("W", 1)), W2=bt(mk("W", 2)), verdict=k1 if ok else "not read")
step = lambda k: (lambda ix: float(np.mean([d["x"] == d["a"] - 1 for d in sel(ix, lambda d: d["k"] == k)])))
k2 = bt(lambda ix: step(1)(ix) - step(2)(ix))
res["K2"] = dict(diff=k2, verdict=(("END(1) answers leave the anchor less than END(2)" if k2["hi"] < 0 else "not") if ok else "not read"))
k3 = bt(lambda ix: mk("B", 1)(ix) - mk("B", 2)(ix))
res["K3"] = dict(diff=k3, verdict=(("END(1)'s writer-free address holds the anchor more firmly, so the same step does not move it" if k3["hi"] < 0 else "not")
                                   if ok and not k1.startswith("the writer's step is gated off") else "not read"))
def block(ds):
    if not ds:
        return None
    return dict(n=len(ds), P_correct=float(np.mean([d["x"] == d["target"] for d in ds])), P_a_minus_1=float(np.mean([d["x"] == d["a"] - 1 for d in ds])),
                P_a=float(np.mean([d["x"] == d["a"] for d in ds])), W=float(np.mean([d["W"] for d in ds])), F=float(np.mean([d["W"] - d["W47"] for d in ds])),
                f_QM=float(np.mean([d["W"] - d["WQM"] for d in ds]) / np.mean([d["W"] - d["W47"] for d in ds])),
                B=float(np.mean([d["B"] for d in ds])), T1_E=float(np.mean([d["TE"] for d in ds])), attention_drop=float(np.mean([d["aR"] - d["aE"] for d in ds])))
desc = {}
for k in KS:
    ds = [d for d in REC if d["k"] == k]; dv = [d for d in ds if d["valid"]]
    desc[f"k{k}"] = dict(n=len(ds), anchors=dict(collections.Counter(d["a"] for d in ds)), P_correct_all=float(np.mean([d["x"] == d["target"] for d in ds])),
                         valid=block(dv), anchor_last=block([d for d in dv if d["a"] == N]))
res["descriptive"] = desc
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"]["all"], K1=[res["K1"]["verdict"], rho["est"]], K2=[res["K2"]["verdict"], k2["est"]], K3=[res["K3"]["verdict"], k3["est"]]), indent=1))
print("->", OUT)
