"""S896 analysis (rules fixed before the data; knowledge/DECISIONS.md S896 entry).  Input results/r3/s896_<tag>_nonband.json.
T1 = z(p - 1) - z(p) at L51 (matcher-head mean LSE block scores, rows B1 / B4 / answer).  Non-band END shift NB = T1(E_BANDR) - T1(R); total END
shift TOT = T1(E) - T1(R).  Ratios of means over (scene, pin); scene bootstrap 4,000 (seed 896).
I0: t1 >= .95; START(p) answers p >= .95; L47 anchor = p for E and R >= .90; NB > 0 with the interval above 0; closure - with the band, the read
  rows and the text all <- START's from layer 46, the late address equals START's (median |T1(E_BANDR_RES46R_TXT46R) - T1(R)| <= .15).
X1 (pre-band read-row state): share_pre = mean[T1(E_BANDR) - T1(E_BANDR_RES46R)] / mean NB >= .60 with the lower bound > .40 -> 'the non-band END
  shift is carried by the read rows' pre-band state'; <= .20 with the upper bound < .35 -> 'not'; else 'partial'.
X2 (text positions): share_txt = mean[T1(E_BANDR) - T1(E_BANDR_TXT46R)] / mean NB with the same thresholds -> 'the non-band END shift is read
  from the text positions by the late layers' / 'not' / 'partial'.
X3 (entry depth): share35 = mean[T1(E) - T1(E_RES35R)] / mean TOT >= .60 with the lower bound > .40 -> 'the read rows carry END's late-address
  information by layer 35'; else 'not by 35'.
Descriptive: E_RES46R (band live on START's read-row input), E_RES50R, E_TXT35R, E_TXT46R as shares of TOT; L55-63; answers per arm.
Usage: python3 s896_analyze.py TAG"""
from __future__ import annotations

import collections, json, sys
import numpy as np

TAG = sys.argv[1]
J = json.load(open(f"/vlm/results/r3/s896_{TAG}_nonband.json"))
OUT = f"/vlm/results/r3/s896_analysis_{TAG}.json"
rng = np.random.default_rng(896); NB_ = 4000
rows = J["rows"]; S = len(rows); PINS = J["pins"]; lm = J["late_mean"]; LAYS = [str(l) for l in J["lateq"]]
ARMS = ("E", "R", "E_BANDR", "RES35R", "RES46R", "BANDR_RES46R", "TXT46R", "BANDR_TXT46R", "BANDR_RES46R_TXT46R", "RES50R", "TXT35R")


def key(a_, p):
    return {"E": f"Ep{p}", "R": f"Rp{p}", "E_BANDR": f"Ep{p}_E_BANDR"}.get(a_, f"Ep{p}_{a_}")


def T1(r, k, p, L="51"):
    z = np.asarray(r["zl"][k][L], dtype=float)[:, lm[L]].mean(1).mean(0); return z[p - 2] - z[p - 1]


def anchor(q):
    q = np.asarray(q); am = [int(np.argmax(q[i])) + 1 for i in (2, 5, 9)]
    c = collections.Counter(am).most_common(); return c[0][0] if c[0][1] >= 2 else am[2]


REC = {}
for si, r in enumerate(rows):
    for p in PINS:
        d = dict(si=si, p=p)
        for L in LAYS:
            for a_ in ARMS:
                d[f"{a_}_{L}"] = T1(r, key(a_, p), p, L)
        d["anch"] = (anchor(r["qk47"][f"Ep{p}"]), anchor(r["qk47"][f"Rp{p}"]))
        d["land"] = {a_: r["land"][key(a_, p)] for a_ in ARMS}
        REC[(si, p)] = d
SC = sorted({k[0] for k in REC})


def bt(fn):
    n = len(SC); est = fn(np.arange(n)); bs = [fn(rng.integers(0, n, n)) for _ in range(NB_)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return dict(est=float(est), lo=float(lo), hi=float(hi))


def M(ix, fn):
    return float(np.mean([fn(REC[(SC[i], p)]) for i in ix for p in PINS]))


NBf = lambda d, L="51": d[f"E_BANDR_{L}"] - d[f"R_{L}"]
TOTf = lambda d, L="51": d[f"E_{L}"] - d[f"R_{L}"]
res = dict(tag=TAG, n_scenes=S)
st = float(np.mean([r["land"][f"START{p}"] == p for r in rows for p in PINS]))
reach = float(np.mean([a_ == d["p"] for d in REC.values() for a_ in d["anch"]]))
bNB = bt(lambda ix: M(ix, NBf))
clo = float(np.median([abs(d["BANDR_RES46R_TXT46R_51"] - d["R_51"]) for d in REC.values()]))
I0 = dict(t1=bool((J["t1_menu"] or 0) >= .95), START=bool(st >= .95), pin_reach=bool(reach >= .90), nonband=bool(bNB["lo"] > 0), closure=bool(clo <= .15))
res["I0"] = dict(**I0, all=bool(all(I0.values())), t1_value=J["t1_menu"], START_acc=st, reach=reach, NB=bNB, closure_median=clo,
                 TOT=M(range(len(SC)), TOTf))
ok = res["I0"]["all"]
def verdict(sh, yes, ok_):
    return ((yes if sh["est"] >= .60 and sh["lo"] > .40 else "not" if sh["est"] <= .20 and sh["hi"] < .35 else "partial") if ok_ else "not read")
x1 = bt(lambda ix: M(ix, lambda d: d["E_BANDR_51"] - d["BANDR_RES46R_51"]) / M(ix, NBf))
res["X1"] = dict(share_pre=x1, verdict=verdict(x1, "the non-band END shift is carried by the read rows' pre-band state", ok))
x2 = bt(lambda ix: M(ix, lambda d: d["E_BANDR_51"] - d["BANDR_TXT46R_51"]) / M(ix, NBf))
res["X2"] = dict(share_txt=x2, verdict=verdict(x2, "the non-band END shift is read from the text positions by the late layers", ok))
x3 = bt(lambda ix: M(ix, lambda d: d["E_51"] - d["RES35R_51"]) / M(ix, TOTf))
res["X3"] = dict(share35=x3, verdict=(("the read rows carry END's late-address information by layer 35" if x3["est"] >= .60 and x3["lo"] > .40 else "not by 35") if ok else "not read"))
allx = range(len(SC))
res["descriptive"] = dict(
    shares_of_TOT={f"L{L}": {a_: M(allx, lambda d, a_=a_, L=L: d[f"E_{L}"] - d[f"{a_}_{L}"]) / M(allx, lambda d, L=L: TOTf(d, L)) for a_ in ARMS[2:]} for L in LAYS},
    mean_T1={a_: M(allx, lambda d, a_=a_: d[f"{a_}_51"]) for a_ in ARMS},
    answers={a_: dict(P_pm1=float(np.mean([d["land"][a_] == d["p"] - 1 for d in REC.values()])), P_p=float(np.mean([d["land"][a_] == d["p"] for d in REC.values()]))) for a_ in ARMS})
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"]["all"], X1=[res["X1"]["verdict"], x1["est"]], X2=[res["X2"]["verdict"], x2["est"]], X3=[res["X3"]["verdict"], x3["est"]]), indent=1))
print("->", OUT)
