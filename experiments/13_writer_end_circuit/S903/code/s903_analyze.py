"""S903 analysis (rules fixed before the data; knowledge/DECISIONS.md S903 entry).  Input results/r3/s903_<tag>_framewrite.json.
T1 = z_(p-1) - z_p of the matcher-head mean block scores at L51 (rows B1 / B4 / answer).  Frame part held by the read rows at layer 35:
D35 = T1(E) - T1(E_RES35S) (donor START(5): the same ordinal, the frame word 'first').  Share of an arm X: s_X = mean[T1(E) - T1(E_X)] / mean D35
(ratios of means over (scene, pin)); scene bootstrap 4,000 (seed 903).
I0: t1 >= .95; START(p) answers p >= .95; L47 anchor = p for E, S5 and R >= .90; the identity patch E_SELF reproduces E (max late block-score
  difference <= 1e-3 in >= .99); the frame part is present (mean D35 > 0 with the interval above 0).  Nothing is read if I0 fails.
G3 (closure first): s_C27_34 >= .70 with the lower bound > .50 -> 'the L27-34 components write the read rows' frame state'; else 'not only
  L27-34'.
G1 (which kind): among FA (FA27, FA31), DN (DeltaNet 28-30, 32-34) and MLP (27-34) the largest s; s >= .50 with the lower bound > .30 ->
  '<kind> at L27-34 writes most of it'; else 'distributed'.
G2 (which half): s_L31_34 >= .60 with the lower bound > .40 and s_L27_30 <= .30 -> 'L31-34 writes most of it'; s_L27_30 >= .60 with the lower
  bound > .40 and s_L31_34 <= .30 -> 'L27-30 writes most of it'; else 'both halves'.
Descriptive: D35 / (T1(E) - T1(S5)) (the read rows' share of the frame-only shift), s_RES27S (before 27), s_RES35R / (T1(E) - T1(R)) (S896
replication), the sum FA + DN + MLP against C27_34, every s at L55-63, answers per arm.
Usage: python3 s903_analyze.py TAG"""
from __future__ import annotations

import collections, json, sys
import numpy as np

TAG = sys.argv[1]
J = json.load(open(f"/vlm/results/r3/s903_{TAG}_framewrite.json"))
OUT = f"/vlm/results/r3/s903_analysis_{TAG}.json"
rng = np.random.default_rng(903); NB = 4000
rows = J["rows"]; S = len(rows); PINS = J["pins"]; lm = J["late_mean"]; LAYS = ("51", "55", "59", "63")
ARMS = ("RES35S", "RES27S", "C27_34S", "FAS", "DNS", "MLPS", "L27_30S", "L31_34S", "SELF", "RES35R")
KIND = {"FAS": "FA", "DNS": "DeltaNet", "MLPS": "MLP"}


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
            e = T1(r, f"Ep{p}", p, L)
            d[f"E_{L}"] = e; d[f"S5_{L}"] = T1(r, f"S5p{p}", p, L); d[f"R_{L}"] = T1(r, f"Rp{p}", p, L)
            for X in ARMS:
                d[f"d{X}_{L}"] = e - T1(r, f"Ep{p}_{X}", p, L)
        d["vid"] = max(float(np.abs(zv(r, f"Ep{p}_SELF", L) - zv(r, f"Ep{p}", L)).max()) for L in LAYS)
        d["anch"] = tuple(anchor(r["qk47"][k_]) for k_ in (f"Ep{p}", f"S5p{p}", f"Rp{p}"))
        d["land"] = {k: v for k, v in r["land"].items() if f"p{p}" in k}
        REC[(si, p)] = d
SC = sorted({k[0] for k in REC})


def bt(fn):
    n = len(SC); est = fn(np.arange(n)); bs = [fn(rng.integers(0, n, n)) for _ in range(NB)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return dict(est=float(est), lo=float(lo), hi=float(hi))


def M(ix, fn):
    return float(np.mean([fn(REC[(SC[i], p)]) for i in ix for p in PINS]))


sX = lambda ix, X, L="51": M(ix, lambda d: d[f"d{X}_{L}"]) / M(ix, lambda d: d[f"dRES35S_{L}"])
res = dict(tag=TAG, n_scenes=S)
st = float(np.mean([r["land"][f"START{p}"] == p for r in rows for p in PINS]))
reach = float(np.mean([a_ == d["p"] for d in REC.values() for a_ in d["anch"]]))
vid_ok = float(np.mean([d["vid"] <= 1e-3 for d in REC.values()]))
bD = bt(lambda ix: M(ix, lambda d: d["dRES35S_51"]))
I0 = dict(t1=bool((J["t1_menu"] or 0) >= .95), START=bool(st >= .95), pin_reach=bool(reach >= .90), identity=bool(vid_ok >= .99), frame_part=bool(bD["lo"] > 0))
res["I0"] = dict(**I0, all=bool(all(I0.values())), t1_value=J["t1_menu"], START_acc=st, reach=reach, vid_ok=vid_ok, D35=bD)
ok = res["I0"]["all"]
bs_ = {X: bt(lambda ix, X=X: sX(ix, X)) for X in ("C27_34S", "FAS", "DNS", "MLPS", "L27_30S", "L31_34S")}
c = bs_["C27_34S"]
res["G3"] = dict(s_C27_34=c, verdict=(("the L27-34 components write the read rows' frame state" if c["est"] >= .70 and c["lo"] > .50 else "not only L27-34") if ok else "not read"))
top = max(KIND, key=lambda X: bs_[X]["est"])
res["G1"] = dict(s={KIND[X]: bs_[X] for X in KIND}, top=KIND[top],
                 verdict=((f"{KIND[top]} at L27-34 writes most of it" if bs_[top]["est"] >= .50 and bs_[top]["lo"] > .30 else "distributed") if ok else "not read"))
h1, h0 = bs_["L31_34S"], bs_["L27_30S"]
res["G2"] = dict(s_L31_34=h1, s_L27_30=h0, verdict=(("L31-34 writes most of it" if h1["est"] >= .60 and h1["lo"] > .40 and h0["est"] <= .30 else
                                                     "L27-30 writes most of it" if h0["est"] >= .60 and h0["lo"] > .40 and h1["est"] <= .30 else "both halves")
                                                    if ok else "not read"))
allx = range(len(SC))
res["descriptive"] = dict(
    read_rows_share_of_frame_shift={L: M(allx, lambda d, L=L: d[f"dRES35S_{L}"]) / M(allx, lambda d, L=L: d[f"E_{L}"] - d[f"S5_{L}"]) for L in LAYS},
    s_RES27S=sX(allx, "RES27S"), S896_replication_share35=M(allx, lambda d: d["dRES35R_51"]) / M(allx, lambda d: d["E_51"] - d["R_51"]),
    sum_kinds=sum(sX(allx, X) for X in KIND), s={f"L{L}": {X: sX(allx, X, L) for X in ARMS if X != "SELF"} for L in LAYS},
    mean_T1={k_: M(allx, lambda d, k_=k_: d[f"{k_}_51"]) for k_ in ("E", "S5", "R")},
    answers={X: dict(P_pm1=float(np.mean([d["land"][f"Ep{d['p']}_{X}"] == d["p"] - 1 for d in REC.values()])),
                     P_p=float(np.mean([d["land"][f"Ep{d['p']}_{X}"] == d["p"] for d in REC.values()]))) for X in ARMS}
            | {"E": dict(P_pm1=float(np.mean([d["land"][f"Ep{d['p']}"] == d["p"] - 1 for d in REC.values()])),
                         P_p=float(np.mean([d["land"][f"Ep{d['p']}"] == d["p"] for d in REC.values()])))})
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"]["all"], G3=[res["G3"]["verdict"], c["est"]], G1=[res["G1"]["verdict"], {KIND[X]: bs_[X]["est"] for X in KIND}],
                      G2=[res["G2"]["verdict"], h1["est"], h0["est"]]), indent=1))
print("->", OUT)
