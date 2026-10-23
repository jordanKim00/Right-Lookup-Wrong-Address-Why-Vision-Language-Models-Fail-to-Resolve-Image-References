"""S911 analysis (rules fixed before the data; knowledge/DECISIONS.md S911 entry; measure revised before registration - see the entry).
Input results/r3/s911_<tag>_reach.json.  Natural runs at N 8.  For k = 1..6 and lambda in {0, 1, 2, 3}: S_k + lambda d_k (d_k = calibration
mean END(k) - START(k) at the layer-35 input, read rows; lambda 0 = START(k) itself).  r(k, lambda) = P(L47 anchor in {t, t + 1}) with
t = N + 1 - k (the end-route positions: the target or one after it; k = 5 is confounded - t + 1 = k = the copy address - and is left out of the
rules); c(k, lambda) = P(L47 anchor = k) (copy); g(k, lambda) = P(answer = END(k)'s own natural answer in the same scene).  END(2) arms:
E2_SUB (E2 - d2), E2_PIN2 (address part of the L35-47 read-row queries <- START(2)'s), E2_SUB_PIN2 (both).  Scene bootstrap 4,000 (seed 911).
I0: t1 >= .95; natural START(q) answers q >= .95; END(2) and START(2) differ (P(x(E2) != 2) >= .80); PIN2 reaches its anchor (E2_PIN2's L47
  anchor = 2 in >= .90).  Nothing is read if I0 fails.
R1 (reach shrinks with k): Spearman correlation between k in {1, 2, 3, 4, 6} and r(k, 1) <= -.80 -> 'the vector's pull of the early address to
  the end-route position shrinks with k'; else 'not monotone'.
R2 (reach extendable): mean over k in {4, 6} of r(k, 3) - r(k, 1) >= .30 with the lower bound > .15 -> 'amplifying the vector extends the end
  route's reach'; else 'not extendable'.
R3 (bypass): P(answer of E2_SUB_PIN2 >= N - 2) >= .50 with the lower bound > .35 -> 'END(2)'s answer bypasses both the read-row vector and the
  early address'; <= .20 with the upper bound < .35 -> 'no bypass: without both, END(2) leaves the end'; else 'partial'.
Descriptive: r, c and g for every (k, lambda); answer distributions; E2 arms' anchors and answers.
Usage: python3 s911_analyze.py TAG"""
from __future__ import annotations

import collections, json, sys
import numpy as np

TAG = sys.argv[1]
J = json.load(open(f"/vlm/results/r3/s911_{TAG}_reach.json"))
OUT = f"/vlm/results/r3/s911_analysis_{TAG}.json"
rng = np.random.default_rng(911); NB = 4000
rows = J["rows"]; S = len(rows); N = J["N"]; KS = J["ks"]; LAMS = (0, 1, 2, 3)


def anchor(q):
    q = np.asarray(q); am = [int(np.argmax(q[i])) + 1 for i in (2, 5, 9)]
    c = collections.Counter(am).most_common(); return c[0][0] if c[0][1] >= 2 else am[2]


REC = []
for si, r in enumerate(rows):
    d = dict(si=si)
    for k in KS:
        d[f"xE{k}"] = r["land"][f"E{k}"]
        for lam in LAMS:
            key = f"S{k}_L{lam}"
            d[f"x_{k}_{lam}"] = r["land"][key]; d[f"a_{k}_{lam}"] = anchor(r["qk47"][key])
    for key in ("E2_SUB", "E2_PIN2", "E2_SUB_PIN2"):
        d[f"x_{key}"] = r["land"][key]; d[f"a_{key}"] = anchor(r["qk47"][key])
    REC.append(d)


def bt(fn):
    est = fn(np.arange(S)); bs = [fn(rng.integers(0, S, S)) for _ in range(NB)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return dict(est=float(est), lo=float(lo), hi=float(hi))


def M(ix, fn):
    return float(np.mean([fn(REC[int(i)]) for i in ix]))


rr = lambda ix, k, lam: M(ix, lambda d: d[f"a_{k}_{lam}"] in (N + 1 - k, N + 2 - k))
cc = lambda ix, k, lam: M(ix, lambda d: d[f"a_{k}_{lam}"] == k)
KR = [k for k in KS if k != 5]
gg = lambda ix, k, lam: M(ix, lambda d: d[f"x_{k}_{lam}"] == d[f"xE{k}"])
res = dict(tag=TAG, n_scenes=S)
allx = np.arange(S)
st = float(np.mean([r["land"][f"START{q}"] == q for r in rows for q in range(1, N + 1)]))
diff = M(allx, lambda d: d["xE2"] != 2); reach = M(allx, lambda d: d["a_E2_PIN2"] == 2)
I0 = dict(t1=bool((J["t1_menu"] or 0) >= .95), START=bool(st >= .95), END2_differs=bool(diff >= .80), pin_reach=bool(reach >= .90))
res["I0"] = dict(**I0, all=bool(all(I0.values())), t1_value=J["t1_menu"], START_acc=st, END2_differs_value=diff, pin_reach_value=reach)
ok = res["I0"]["all"]


def spearman(ix):
    x = np.array(KR, float); y = np.array([rr(ix, k, 1) for k in KR])
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y + 1e-9 * np.arange(len(y))))
    return float(np.corrcoef(rx, ry)[0, 1])


b1 = bt(spearman)
res["R1"] = dict(spearman=b1, r_lambda1={str(k): rr(allx, k, 1) for k in KS},
                 verdict=(("the vector's pull of the early address to the end-route position shrinks with k" if b1["est"] <= -.80 else "not monotone") if ok else "not read"))
b2 = bt(lambda ix: float(np.mean([rr(ix, k, 3) - rr(ix, k, 1) for k in (4, 6)])))
res["R2"] = dict(diff=b2, verdict=(("amplifying the vector extends the end route's reach" if b2["est"] >= .30 and b2["lo"] > .15 else "not extendable")
                                   if ok else "not read"))
b3 = bt(lambda ix: M(ix, lambda d: d["x_E2_SUB_PIN2"] >= N - 2))
res["R3"] = dict(P_end=b3, verdict=(("END(2)'s answer bypasses both the read-row vector and the early address" if b3["est"] >= .50 and b3["lo"] > .35 else
                                     "no bypass: without both, END(2) leaves the end" if b3["est"] <= .20 and b3["hi"] < .35 else "partial") if ok else "not read"))
res["descriptive"] = dict(
    r={f"k{k}": {str(lam): rr(allx, k, lam) for lam in LAMS} for k in KS},
    c={f"k{k}": {str(lam): cc(allx, k, lam) for lam in LAMS} for k in KS},
    g={f"k{k}": {str(lam): gg(allx, k, lam) for lam in LAMS} for k in KS},
    answers={f"k{k}": {**{str(lam): dict(collections.Counter(d[f"x_{k}_{lam}"] for d in REC)) for lam in LAMS},
                        "E": dict(collections.Counter(d[f"xE{k}"] for d in REC))} for k in KS},
    E2_arms={key: dict(answers=dict(collections.Counter(d[f"x_{key}"] for d in REC)), anchors=dict(collections.Counter(d[f"a_{key}"] for d in REC)))
             for key in ("E2_SUB", "E2_PIN2", "E2_SUB_PIN2")})
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"]["all"], R1=[res["R1"]["verdict"], b1["est"]], R2=[res["R2"]["verdict"], b2["est"]], R3=[res["R3"]["verdict"], b3["est"]]), indent=1))
print("->", OUT)
