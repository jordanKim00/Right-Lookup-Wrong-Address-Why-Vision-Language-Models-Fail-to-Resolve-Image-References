"""S912 analysis (rules fixed before the data; knowledge/DECISIONS.md S912 entry).  Inputs results/r3/s912_<tag>_idrepair.json for the three runs
(N5 END(3), N7 END(4), N9 END(5)).  Identity runs: the natural L47 anchor a equals the target t = N + 1 - k (= k at these middle cells).
acc(X) = P(answer = t) over identity runs; Delta(X) = acc(X) - acc(E); rho_X = Delta(X) / Delta(E_BANDR), pooled over N7 and N9 identity runs.
T1_a = z(a - 1) - z(a) at L51 (matcher-head mean block scores, rows B1 / B4 / answer); writer-free margin B = T1_a(E_BANDR) (more negative =
the address holds the anchor more firmly); band effect W = T1_a(E) - B.  Scene bootstrap 4,000 within each N (seed 912).
I0: t1 >= .95 in every run; natural START(q) answers q >= .95 (q < N) in every run; >= 20 pooled N7 + N9 identity runs; the band repair is
  present (Delta(E_BANDR) >= .30 with the lower bound > .15, pooled).  Nothing is read if I0 fails.
C1 (per arm X in ALL47R, QM, QMg, M49R, DN49R): rho_X >= .50 with the lower bound > .30 -> 'X carries the identity collapse'; rho_X <= .15 with
  the upper bound < .30 -> 'no effective contribution of X observed'; else 'partial'.
C2 (why N5 survives): mean B(N5 identity runs) - mean B(N7 + N9 identity runs) <= -1 with the upper bound < 0 -> 'N5's identity cell holds by a
  deeper writer-free margin'; else 'not observed'.
Descriptive: per N - anchor distribution, identity share, acc of every arm (identity runs and all runs), mean W and B.
Usage: python3 s912_analyze.py TAG_N5 TAG_N7 TAG_N9"""
from __future__ import annotations

import collections, json, sys
import numpy as np

TAGS = sys.argv[1:4]
OUT = f"/vlm/results/r3/s912_analysis_{TAGS[1]}.json"
rng = np.random.default_rng(912); NB = 4000
ARMS = ("E", "R", "E_BANDR", "ALL47R", "QM", "QMg", "M49R", "DN49R")
RUNS = {}
REC = []
for tag in TAGS:
    J = json.load(open(f"/vlm/results/r3/s912_{tag}_idrepair.json")); RUNS[tag] = J
    N = J["N"]; k = J["ks"][0]; t = N + 1 - k; lm = J["late_mean"]
    def zv(r, key, L="51"):
        return np.asarray(r["zl"][key][L], dtype=float)[:, lm[L]].mean(1).mean(0)
    for si, r in enumerate(J["rows"]):
        a_ = r["anchor"][str(k)]
        if f"B{k}_E" not in r["land"]:
            continue
        d = dict(tag=tag, N=N, k=k, t=t, si=si, a=a_, ident=(a_ == t))
        for X in ARMS:
            key = f"B{k}_{X}"
            d[f"x_{X}"] = r["land"][key]
            z = zv(r, key); d[f"T_{X}"] = z[a_ - 2] - z[a_ - 1]
        d["B"] = d["T_E_BANDR"]; d["W"] = d["T_E"] - d["B"]
        REC.append(d)


def boot(fn, tags):
    est = fn({tg: np.arange(len(RUNS[tg]["rows"])) for tg in tags})
    bs = []
    for _ in range(NB):
        bs.append(fn({tg: rng.integers(0, len(RUNS[tg]["rows"]), len(RUNS[tg]["rows"])) for tg in tags}))
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return dict(est=float(est), lo=float(lo), hi=float(hi))


def sel(ixs, cond):
    out = []
    for tg, ix in ixs.items():
        cnt = collections.Counter(int(i) for i in ix)
        out += [d for d in REC if d["tag"] == tg and cond(d) for _ in range(cnt.get(d["si"], 0))]
    return out


def acc(ixs, X, cond=lambda d: d["ident"]):
    ds = sel(ixs, cond); return float(np.mean([d[f"x_{X}"] == d["t"] for d in ds])) if ds else float("nan")


def _div(a_, b_):
    return a_ / b_ if b_ not in (0, 0.0) and np.isfinite(b_) else float("nan")


P79 = TAGS[1:3]
res = dict(tags=TAGS)
t1s = {tg: RUNS[tg]["t1_menu"] for tg in TAGS}
sts = {tg: float(np.mean([r["land"][f"N_S{q}"] == q for r in RUNS[tg]["rows"] for q in range(1, RUNS[tg]["N"])])) for tg in TAGS}
nid = len([d for d in REC if d["tag"] in P79 and d["ident"]])
dB = boot(lambda ixs: acc(ixs, "E_BANDR") - acc(ixs, "E"), P79)
I0 = dict(t1=bool(all((v or 0) >= .95 for v in t1s.values())), START=bool(all(v >= .95 for v in sts.values())), identity_runs=bool(nid >= 20),
          band_repair=bool(dB["est"] >= .30 and dB["lo"] > .15))
res["I0"] = dict(**I0, all=bool(all(I0.values())), t1_values=t1s, START_acc=sts, n_identity_N7_N9=nid, delta_BANDR=dB)
ok = res["I0"]["all"]
res["C1"] = {}
for X in ("ALL47R", "QM", "QMg", "M49R", "DN49R"):
    b = boot(lambda ixs, X=X: _div(acc(ixs, X) - acc(ixs, "E"), acc(ixs, "E_BANDR") - acc(ixs, "E")), P79)
    res["C1"][X] = dict(rho=b, verdict=(("carries the identity collapse" if b["est"] >= .50 and b["lo"] > .30 else
                                         "no effective contribution observed" if b["est"] <= .15 and b["hi"] < .30 else "partial") if ok else "not read"))
mB = lambda ixs, tags: float(np.mean([d["B"] for d in sel({tg: ixs[tg] for tg in tags}, lambda d: d["ident"])]))
b2 = boot(lambda ixs: mB(ixs, [TAGS[0]]) - mB(ixs, P79), TAGS)
res["C2"] = dict(margin_diff=b2, verdict=(("N5's identity cell holds by a deeper writer-free margin" if b2["est"] <= -1 and b2["hi"] < 0 else "not observed")
                                          if ok else "not read"))
desc = {}
for tg in TAGS:
    ds = [d for d in REC if d["tag"] == tg]; di = [d for d in ds if d["ident"]]
    desc[tg] = dict(N=ds[0]["N"] if ds else None, n=len(ds), identity_share=float(np.mean([d["ident"] for d in ds])) if ds else None,
                    anchors=dict(collections.Counter(d["a"] for d in ds)),
                    acc_identity={X: float(np.mean([d[f"x_{X}"] == d["t"] for d in di])) if di else None for X in ARMS},
                    acc_all={X: float(np.mean([d[f"x_{X}"] == d["t"] for d in ds])) if ds else None for X in ARMS},
                    W_identity=float(np.mean([d["W"] for d in di])) if di else None, B_identity=float(np.mean([d["B"] for d in di])) if di else None)
res["descriptive"] = desc
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"]["all"], C1={X: [v["verdict"], v["rho"]["est"]] for X, v in res["C1"].items()}, C2=[res["C2"]["verdict"], b2["est"]]), indent=1))
print("->", OUT)
