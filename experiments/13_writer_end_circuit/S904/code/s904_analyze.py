"""S904 analysis (rules fixed before the data; knowledge/DECISIONS.md S904 entry).  Input results/r3/s904_<tag>_modevec.json.
T1 = z_(p-1) - z_p of the matcher-head mean block scores at L51 (rows B1 / B4 / answer).  Frame-only shift FS = T1(E) - T1(S5) (END(5) vs
START(5), both pinned at p).  d = the calibration scenes' mean END(5) - START(5) difference at the layer-35 input, read rows.
suff = mean[T1(S5_ADD) - T1(S5)] / mean FS; nec = mean[T1(E) - T1(E_SUB)] / mean FS; rand = mean[T1(S5_RAND) - T1(S5)] / mean FS (ratios of
means over (scene, pin)); scene bootstrap 4,000 (seed 904).
I0: t1 >= .95; START(p) answers p >= .95; L47 anchor = p for E, S5 and R >= .90; S5_SELF reproduces S5 (max late block-score difference
  <= 1e-3 in >= .99); FS > 0 with the interval above 0.  Nothing is read if I0 fails.
Q1 (sufficiency): suff >= .50 with the lower bound > .30 -> 'one scene-independent mean direction per read row at layer 35 carries most of the
  frame's late shift when added to START'; suff <= .15 with the upper bound < .30 -> 'not'; else 'partial'.
Q2 (necessity): nec >= .50 with the lower bound > .30 -> 'removing that direction from END removes most of the shift'; nec <= .15 with the
  upper bound < .30 -> 'not'; else 'partial'.
Q3 (specificity): suff - rand >= .30 with the lower bound > .15 -> 'direction-specific'; else 'not specific'.
Descriptive: S5_ADD2 (dose), E_RES35S (the full state; S903's D35) as a share of FS, the shares at L55-63, answers (P(p - 1), P(p)) per arm,
the calibration energy share and mean cosine.
Usage: python3 s904_analyze.py TAG"""
from __future__ import annotations

import collections, json, sys
import numpy as np

TAG = sys.argv[1]
J = json.load(open(f"/vlm/results/r3/s904_{TAG}_modevec.json"))
OUT = f"/vlm/results/r3/s904_analysis_{TAG}.json"
rng = np.random.default_rng(904); NB = 4000
rows = J["rows"]; S = len(rows); PINS = J["pins"]; lm = J["late_mean"]; LAYS = ("51", "55", "59", "63")
ARMS = ("S5_ADD", "S5_ADD2", "E_SUB", "E_RES35S", "S5_RAND", "S5_SELF")


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
            d[f"E_{L}"] = T1(r, f"Ep{p}", p, L); d[f"S5_{L}"] = T1(r, f"S5p{p}", p, L)
            for X in ARMS:
                d[f"{X}_{L}"] = T1(r, f"Ep{p}_{X}", p, L)
        d["vid"] = max(float(np.abs(zv(r, f"Ep{p}_S5_SELF", L) - zv(r, f"S5p{p}", L)).max()) for L in LAYS)
        d["anch"] = tuple(anchor(r["qk47"][k_]) for k_ in (f"Ep{p}", f"S5p{p}", f"Rp{p}"))
        d["land"] = {k: v for k, v in r["land"].items() if f"p{p}" in k}
        REC[(si, p)] = d
SC = sorted({k[0] for k in REC})


def bt(fn):
    n = len(SC); est = fn(np.arange(n)); bs = [fn(rng.integers(0, n, n)) for _ in range(NB)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return dict(est=float(est), lo=float(lo), hi=float(hi))


def M(ix, fn):
    return float(np.mean([fn(REC[(SC[i], p)]) for i in ix for p in PINS]))


FS = lambda ix, L="51": M(ix, lambda d: d[f"E_{L}"] - d[f"S5_{L}"])
suff = lambda ix, X="S5_ADD", L="51": M(ix, lambda d: d[f"{X}_{L}"] - d[f"S5_{L}"]) / FS(ix, L)
nec = lambda ix, X="E_SUB", L="51": M(ix, lambda d: d[f"E_{L}"] - d[f"{X}_{L}"]) / FS(ix, L)
res = dict(tag=TAG, n_scenes=S, calibration=J.get("calibration"))
st = float(np.mean([r["land"][f"START{p}"] == p for r in rows for p in PINS]))
reach = float(np.mean([a_ == d["p"] for d in REC.values() for a_ in d["anch"]]))
vid_ok = float(np.mean([d["vid"] <= 1e-3 for d in REC.values()]))
bF = bt(lambda ix: FS(ix))
I0 = dict(t1=bool((J["t1_menu"] or 0) >= .95), START=bool(st >= .95), pin_reach=bool(reach >= .90), identity=bool(vid_ok >= .99), frame_shift=bool(bF["lo"] > 0))
res["I0"] = dict(**I0, all=bool(all(I0.values())), t1_value=J["t1_menu"], START_acc=st, reach=reach, vid_ok=vid_ok, FS=bF)
ok = res["I0"]["all"]
b1 = bt(lambda ix: suff(ix)); b2 = bt(lambda ix: nec(ix)); b3 = bt(lambda ix: suff(ix) - suff(ix, "S5_RAND"))
res["Q1"] = dict(suff=b1, verdict=(("one scene-independent mean direction per read row at layer 35 carries most of the frame's late shift when added to START"
                                    if b1["est"] >= .50 and b1["lo"] > .30 else "not" if b1["est"] <= .15 and b1["hi"] < .30 else "partial") if ok else "not read"))
res["Q2"] = dict(nec=b2, verdict=(("removing that direction from END removes most of the shift" if b2["est"] >= .50 and b2["lo"] > .30 else
                                   "not" if b2["est"] <= .15 and b2["hi"] < .30 else "partial") if ok else "not read"))
res["Q3"] = dict(diff=b3, verdict=(("direction-specific" if b3["est"] >= .30 and b3["lo"] > .15 else "not specific") if ok else "not read"))
allx = range(len(SC))
ans = lambda key: dict(P_pm1=float(np.mean([d["land"][key.format(p=d["p"])] == d["p"] - 1 for d in REC.values()])),
                       P_p=float(np.mean([d["land"][key.format(p=d["p"])] == d["p"] for d in REC.values()])))
res["descriptive"] = dict(
    shares={f"L{L}": dict(S5_ADD=suff(allx, "S5_ADD", L), S5_ADD2=suff(allx, "S5_ADD2", L), S5_RAND=suff(allx, "S5_RAND", L),
                          E_SUB=nec(allx, "E_SUB", L), E_RES35S=nec(allx, "E_RES35S", L)) for L in LAYS},
    FS_by_layer={L: FS(allx, L) for L in LAYS},
    answers={"E": ans("Ep{p}"), "S5": ans("S5p{p}"), **{X: ans(f"Ep{{p}}_{X}") for X in ARMS}})
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"]["all"], Q1=[res["Q1"]["verdict"], b1["est"]], Q2=[res["Q2"]["verdict"], b2["est"]], Q3=[res["Q3"]["verdict"], b3["est"]],
                      calibration=(J.get("calibration") or {}).get("energy_share")), indent=1))
print("->", OUT)
