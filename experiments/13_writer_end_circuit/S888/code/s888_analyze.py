"""S888 analysis (rules fixed before the data; knowledge/DECISIONS.md S888 entry).  Input results/r3/s888_<tag>_dnroute.json.
For START(p) runs with the anchor block's values at the read rows replaced by the far block f's at a layer set X: P_f(X) = P(answer = f | R_X);
late address shift s_X(L) = mean[(z_f - z_p)(R_X) - (z_f - z_p)(R)] / mean[(z_p - z_f)(R)] (the share of the f-vs-p gap closed; matcher-head mean
LSE block scores, rows B1 / B4 / answer).  Scene bootstrap 4,000 (seed 888) over (scene, pin).
I0: t1 >= .95; START(p) answers p >= .95 (natural donors) and R answers p >= .95; L47 anchor = p for R >= .90; the identity DeltaNet map R_DNID
  reproduces R (max late block-score difference <= 1e-3 in >= .99 of (scene, pin)).  Nothing is read if I0 fails.
D1 (wide redirect, FA + DeltaNet 24-50): P_f(WIDE) >= .50 with the lower bound > .30 -> 'the late lookup dereferences the picture content the
  read rows read through FA and DeltaNet layers 24-50'; <= .10 -> 'not'; else 'partial'.
D2 (route at 35-47): d = s_DN(L51) - s_FA(L51): >= .20 with the interval above 0 -> 'the DeltaNet route carries more of the late content signal
  than the FA route'; <= -.20 with the interval below 0 -> 'the FA route carries more'; else 'comparable'.
D4 (late readout, positive control): P_f(LATE) >= .50 with the lower bound > .30 -> 'the late heads read the answer content where they attend';
  else 'not'.
Descriptive: P_f and s_X at L51-63 for FA, DN, FADN, DNLO, WIDE, LATE; E_WIDE answers (P(p - 1), P(p), P(f), P(f - 1)).
Usage: python3 s888_analyze.py TAG"""
from __future__ import annotations

import collections, json, sys
import numpy as np

TAG = sys.argv[1]
J = json.load(open(f"/vlm/results/r3/s888_{TAG}_dnroute.json"))
OUT = f"/vlm/results/r3/s888_analysis_{TAG}.json"
rng = np.random.default_rng(888); NB = 4000
rows = J["rows"]; S = len(rows); PINS = J["pins"]; lm = J["late_mean"]; LAYS = [str(l) for l in J["lateq"]]
FAR = {int(k): v for k, v in J["far"].items()}
XS = ("FA", "DN", "FADN", "DNLO", "WIDE", "LATE")


def zv(r, key, L):
    return np.asarray(r["zl"][key][L], dtype=float)[:, lm[L]].mean(1).mean(0)


def anchor(q):
    q = np.asarray(q); am = [int(np.argmax(q[i])) + 1 for i in (2, 5, 9)]
    c = collections.Counter(am).most_common(); return c[0][0] if c[0][1] >= 2 else am[2]


REC = {}
for si, r in enumerate(rows):
    for p in PINS:
        f = FAR[p]; d = dict(si=si, p=p, f=f)
        for L in LAYS:
            zr = zv(r, f"Rp{p}", L); d[f"gap_{L}"] = zr[p - 1] - zr[f - 1]
            for X in XS:
                zx = zv(r, f"Rp{p}_{X}", L); d[f"sh_{X}_{L}"] = (zx[f - 1] - zx[p - 1]) - (zr[f - 1] - zr[p - 1])
        d["vid"] = max(float(np.abs(zv(r, f"Rp{p}_DNID", L) - zv(r, f"Rp{p}", L)).max()) for L in LAYS)
        d["land"] = {k: v for k, v in r["land"].items() if k.startswith(f"Rp{p}") or k.startswith(f"Ep{p}")}
        d["xR"] = r["land"][f"Rp{p}"]; d["anchR"] = anchor(r["qk47"][f"Rp{p}"])
        REC[(si, p)] = d
SC = sorted({k[0] for k in REC})


def bt(fn):
    n = len(SC); est = fn(np.arange(n)); bs = [fn(rng.integers(0, n, n)) for _ in range(NB)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return dict(est=float(est), lo=float(lo), hi=float(hi))


def M(ix, fn):
    return float(np.mean([fn(REC[(SC[i], p)]) for i in ix for p in PINS]))


def Pf(ix, X):
    return M(ix, lambda d: float(d["land"][f"Rp{d['p']}_{X}"] == d["f"]))


def sh(ix, X, L="51"):
    return M(ix, lambda d: d[f"sh_{X}_{L}"]) / M(ix, lambda d: d[f"gap_{L}"])


res = dict(tag=TAG, n_scenes=S)
st = float(np.mean([r["land"][f"START{p}"] == p for r in rows for p in PINS]))
rp = float(np.mean([d["xR"] == d["p"] for d in REC.values()])); reach = float(np.mean([d["anchR"] == d["p"] for d in REC.values()]))
vid_ok = float(np.mean([d["vid"] <= 1e-3 for d in REC.values()]))
I0 = dict(t1=bool((J["t1_menu"] or 0) >= .95), START=bool(st >= .95 and rp >= .95), pin_reach=bool(reach >= .90), identity_dn=bool(vid_ok >= .99))
res["I0"] = dict(**I0, all=bool(all(I0.values())), t1_value=J["t1_menu"], START_acc=st, R_acc=rp, reach=reach, vid_ok=vid_ok,
                 vid_max=float(max(d["vid"] for d in REC.values())))
ok = res["I0"]["all"]
d1 = bt(lambda ix: Pf(ix, "WIDE"))
res["D1"] = dict(P_f=d1, verdict=(("the late lookup dereferences the picture content the read rows read through FA and DeltaNet layers 24-50"
                                   if d1["est"] >= .50 and d1["lo"] > .30 else "not" if d1["est"] <= .10 else "partial") if ok else "not read"))
d2 = bt(lambda ix: sh(ix, "DN") - sh(ix, "FA"))
res["D2"] = dict(d=d2, s_DN=sh(range(len(SC)), "DN"), s_FA=sh(range(len(SC)), "FA"),
                 verdict=(("the DeltaNet route carries more of the late content signal than the FA route" if d2["est"] >= .20 and d2["lo"] > 0 else
                           "the FA route carries more" if d2["est"] <= -.20 and d2["hi"] < 0 else "comparable") if ok else "not read"))
d4 = bt(lambda ix: Pf(ix, "LATE"))
res["D4"] = dict(P_f=d4, verdict=(("the late heads read the answer content where they attend" if d4["est"] >= .50 and d4["lo"] > .30 else "not") if ok else "not read"))
allx = range(len(SC))
res["descriptive"] = dict(P_f={X: Pf(allx, X) for X in XS}, P_p={X: M(allx, lambda d, X=X: float(d["land"][f"Rp{d['p']}_{X}"] == d["p"])) for X in XS},
                          shift={f"L{L}": {X: sh(allx, X, L) for X in XS} for L in LAYS},
                          E_WIDE={k: M(allx, (lambda d, k=k: float(d["land"][f"Ep{d['p']}_WIDE"] == (d["p"] - 1 if k == "p-1" else d["p"] if k == "p" else d["f"] if k == "f" else d["f"] - 1))))
                                  for k in ("p-1", "p", "f", "f-1")},
                          E={k: M(allx, (lambda d, k=k: float(d["land"][f"Ep{d['p']}"] == (d["p"] - 1 if k == "p-1" else d["p"])))) for k in ("p-1", "p")})
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"]["all"], D1=[res["D1"]["verdict"], d1["est"]], D2=[res["D2"]["verdict"], d2["est"]], D4=[res["D4"]["verdict"], d4["est"]]), indent=1))
print("->", OUT)
