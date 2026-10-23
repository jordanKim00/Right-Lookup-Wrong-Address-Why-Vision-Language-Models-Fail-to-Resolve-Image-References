"""S860 analysis (rules fixed before the data; knowledge/DECISIONS.md S860 entry).  Input results/r3/s860_<tag>_heads.json.
Per run with pin p (N = 8, j* = 9 - k, d = p - j*): late profile z = head-mean (late_mean) block scores at L51, rows B1 / B4 / answer averaged
(L55 reported too); answer profile y = menu-word logits of pictures 1..8.  Indices (user review 10-20):
  U = (z[p-1] + z[p+1]) / 2 - z[p]   un-anchoring (the anchor's lead over its neighbours lost)
  T = z[p-1] - z[p+1]                predecessor selectivity (an odd measure: a start-ward tilt also raises it - V5 separates table from tilt)
  M = z[j*] - max_{i != j*} z[i]     target margin
and the answer shares back (= p - 1), stay (= p), target, below (< p - 1), forward (> p).  Removal effect of an arm a patched toward R:
Delta_a(X) = X(E) - X(a) (scene mean); relative f_a(X) = Delta_a(X) / Delta_BAND_R(X); k5 cells = pins 3, 4, 5, 7 of k 5.  Scene bootstrap 4,000.
Arms are compared with the E of their own sub-pass (passes are split into equal-size sub-passes, each holding every donor).
I0: SELF_FA, SELF_M, SELF_P bit-identical to their sub-pass E (answers and late scores); pin reach (head-mean argmax at L43 / L47 = p, rows B1 / B4 / answer) >= .95
  over E, R, EK; R answers p >= .90; t1 >= .95.
POS: Delta_BAND_R(back, k5) >= .20 and Delta_BAND_R(U, k5) > 0, both with intervals above 0; otherwise 'no valid positive control' (V1-V8 not read).
V1 (2 x 2, k5): V1a f_ANTI_R(U) >= .25 with the lower bound > 0 'the anti-self heads' END-specific part carries un-anchoring', else 'not';
  V1b f_M49_R(T) >= .25 with the lower bound > 0 'MLP49 carries predecessor selectivity (table or tilt - see V5)', else 'not';
  V1c I(X) = Delta_ANTI_M49_R(X) - Delta_ANTI_R(X) - Delta_M49_R(X) for X = U and T: 'interaction' if |I| / Delta_BAND_R >= .25 with the interval of I
  excluding 0 for either X, else 'additive'.
V2: ANTI_Z stay - E stay (all cells) >= .10 with the interval above 0 'the anti-self heads are causal for leaving the anchor', else 'not';
  ANTI43_Z the same with >= .05.
V3: f_H16_R(T) - mean f_RAND1a/b_R(T) (k5) >= .10 with the interval above 0 'h16 carries selective predecessor', else 'not above random';
  G16: back(E) - back(G16) (k5) >= .05 with the interval above 0 'k opens h16's gate causally', else 'not observed'.
V4 (chain): f_M4748_R(back) >= .25 with the lower bound > 0 and f_M49_R(back) <= .10: 'MLP47-48 carry the one-back, MLP49 does not'; f_M49_R(back)
  >= .25 with the lower bound > 0: 'MLP49 carries the one-back'; else 'neither alone'.  PATH47: f_PATH47(back) >= .15 with the lower bound > 0
  'the FA47 -> MLP47 path carries part of the one-back', else 'not'.
V5 (transfer, two-pin passes): for X in {M49, M47, M48, H16}, shift S_X = [z[p'-1] - z[p-1]](X_PX) - [z[p'-1] - z[p-1]](E), p' = donor pin; ratio
  r_X = S_X / S_BAND (pooled over the three two-pin passes).  BAND_PX valid if S_BAND > 0 with the interval above 0.  'X carries an
  anchor-keyed address' if r_X >= .5 with the lower bound > 0; 'anchor-independent' if r_X <= .2 with the upper bound <= .35; else 'mixed'.
V6: SELFADDR_R stay - E stay (all cells) >= .05 with the interval above 0 'the self-address heads' END down-weighting is causal for leaving'.
V9 (organisation of E's behaviour): back share k5 - k3 at d = -1 ((5, 3) vs (3, 5)) and d = +1 ((5, 5) vs (3, 7)): 'k-organised' if both >= .15
  with intervals above 0; 'd-organised' if both |.| <= .10 with intervals inside +-.15; else 'mixed'.
Descriptive: every arm's indices and shares per cell (late and answer profiles), f values, random-set spread, MLP3 parts, k3 cells.
Usage: python3 s860_analyze.py TAG"""
from __future__ import annotations

import json, sys
import numpy as np

TAG = sys.argv[1]
J = json.load(open(f"/vlm/results/r3/s860_{TAG}_heads.json"))
OUT = f"/vlm/results/r3/s860_analysis_{TAG}.json"
rng = np.random.default_rng(860)
NB = 4000
N = 8
rows = J["rows"]; S = len(rows)
PASSES = [(int(k), tuple(ps)) for k, ps in J["passes"]]
CELLS = [(k, p) for k, ps in PASSES for p in ps]
K5 = [c for c in CELLS if c[0] == 5]
TWO = [(k, ps) for k, ps in PASSES if len(ps) == 2]
LATE = [str(l) for l in J["late"]][:2]


def z_late(r, key, L):
    lm = J["late_mean"][L]
    return np.asarray(r["zl"][key][L], dtype=float)[:, lm].mean(1).mean(0)               # rows averaged -> [N]


def z_ans(r, key):
    return np.asarray(r["mlog"][key][:N], dtype=float)                                     # pictures 1..8


def idx(z, p, js):
    U = (z[p - 2] + z[p]) / 2 - z[p - 1]; T = z[p - 2] - z[p]
    M = z[js - 1] - np.max(np.delete(z, js - 1))
    return U, T, M


TAB = {}
SP = rows[0].get("subpass", {})
for k, ps in PASSES:
    js = N + 1 - k
    for p in ps:
        names = sorted({key.split("_", 1)[1] for key in rows[0]["land"] if key.startswith(f"k{k}p{p}_")})
        for a in names:
            key = f"k{k}p{p}_{a}"
            L_ = np.array([r["land"][key] for r in rows])
            d = dict(back=(L_ == p - 1).astype(float), stay=(L_ == p).astype(float), tgt=(L_ == js).astype(float),
                     below=((L_ >= 1) & (L_ < p - 1)).astype(float), fwd=(L_ > p).astype(float))
            for L in LATE:
                v = np.array([idx(z_late(r, key, L), p, js) for r in rows])
                d[f"U{L}"], d[f"T{L}"], d[f"M{L}"] = v[:, 0], v[:, 1], v[:, 2]
            v = np.array([idx(z_ans(r, key), p, js) for r in rows])
            d["Uans"], d["Tans"], d["Mans"] = v[:, 0], v[:, 1], v[:, 2]
            TAB[(k, p, a)] = d


def eref(k, p, a):
    """the E run of the sub-pass that holds arm a (arms are compared with the E of their own pass)."""
    return f"E@{SP[f'k{k}p{p}_{a}']}" if f"k{k}p{p}_{a}" in SP and a not in ("E", "R", "EK") else "E"


L0 = LATE[0]
U_, T_, M_ = f"U{L0}", f"T{L0}", f"M{L0}"


def bt(fn):
    est = fn(np.arange(S)); bs = [fn(rng.integers(0, S, S)) for _ in range(NB)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(est=float(est), lo=float(lo), hi=float(hi))


def delta(a, X, cells):
    return lambda ix: float(np.mean([TAB[(k, p, eref(k, p, a))][X][ix].mean() - TAB[(k, p, a)][X][ix].mean() for k, p in cells]))


def frac(a, X, cells):
    return lambda ix: delta(a, X, cells)(ix) / (delta("BAND_R", X, cells)(ix) + 1e-12)


res = dict(tag=TAG, n_scenes=S)
# ---------------------------------------------------------------- I0
self_ok = True
for r in rows:
    for k, p in CELLS:
        for s_ in ("SELF_FA", "SELF_M", "SELF_P"):
            a_, e_ = f"k{k}p{p}_{s_}", f"k{k}p{p}_{eref(k, p, s_)}"
            if r["land"][a_] != r["land"][e_] or not all(np.array_equal(np.asarray(r["zl"][a_][L]), np.asarray(r["zl"][e_][L])) for L in r["zl"][e_]):
                self_ok = False
reach = []
for r in rows:
    for k, p in CELLS:
        for a in ("E", "R", "EK"):
            for L in (J["early"][-2], J["early"][-1]):
                qk = np.asarray(r["qk"][f"k{k}p{p}_{a}"][str(L)])
                reach += [int(np.argmax(qk[i])) + 1 == p for i in (2, 5, 9)]
r_ok = float(np.mean([r["land"][f"k{k}p{p}_R"] == p for r in rows for k, p in CELLS]))
don_same = True                                                                              # donors bit-identical across the sub-passes of a pass
for r in rows:
    for k, p in CELLS:
        sps = sorted({int(key.split("@")[1]) for key in r["land"] if key.startswith(f"k{k}p{p}_E@")})
        for sp_ in sps[1:]:
            for dn in ("E", "R", "EK"):
                a_, b_ = f"k{k}p{p}_{dn}@0", f"k{k}p{p}_{dn}@{sp_}"
                if not all(np.array_equal(np.asarray(r["zl"][a_][L]), np.asarray(r["zl"][b_][L])) for L in r["zl"][a_]):
                    don_same = False
I0 = dict(self_bit_identical=bool(self_ok), t1=bool((J["t1_menu"] or 0) >= .95), pin_reach=bool(np.mean(reach) >= .95), R_pointer=bool(r_ok >= .90))
res["I0"] = dict(**I0, all=bool(all(I0.values())), t1_value=J["t1_menu"], pin_reach_value=float(np.mean(reach)), R_at_p=r_ok,
                 donors_identical_across_subpasses=bool(don_same))
# ---------------------------------------------------------------- POS
pb, pu = bt(delta("BAND_R", "back", K5)), bt(delta("BAND_R", U_, K5))
res["POS"] = dict(delta_back=pb, delta_U=pu, delta_T=bt(delta("BAND_R", T_, K5)), valid=bool(pb["est"] >= .20 and pb["lo"] > 0 and pu["lo"] > 0))
# ---------------------------------------------------------------- V1
v1a = bt(frac("ANTI_R", U_, K5)); v1b = bt(frac("M49_R", T_, K5))
res["V1"] = dict(f_ANTI_U=v1a, f_ANTI_T=bt(frac("ANTI_R", T_, K5)), f_M49_T=v1b, f_M49_U=bt(frac("M49_R", U_, K5)),
                 verdict_a=("the anti-self heads' END-specific part carries un-anchoring" if v1a["est"] >= .25 and v1a["lo"] > 0 else "not"),
                 verdict_b=("MLP49 carries predecessor selectivity (table or tilt - see V5)" if v1b["est"] >= .25 and v1b["lo"] > 0 else "not"))
inter = {}
for X in (U_, T_):
    I_ = bt(lambda ix, X=X: delta("ANTI_M49_R", X, K5)(ix) - delta("ANTI_R", X, K5)(ix) - delta("M49_R", X, K5)(ix))
    rel = I_["est"] / (delta("BAND_R", X, K5)(np.arange(S)) + 1e-12)
    inter[X] = dict(I=I_, rel=float(rel), flag=bool(abs(rel) >= .25 and (I_["lo"] > 0 or I_["hi"] < 0)))
res["V1"]["interaction"] = inter
res["V1"]["verdict_c"] = "interaction" if any(v["flag"] for v in inter.values()) else "additive"
# ---------------------------------------------------------------- V2
v2 = bt(lambda ix: -delta("ANTI_Z", "stay", CELLS)(ix)); v2b = bt(lambda ix: -delta("ANTI43_Z", "stay", CELLS)(ix))
res["V2"] = dict(ANTI_Z_stay=v2, ANTI_Z_U=bt(delta("ANTI_Z", U_, CELLS)), ANTI43_Z_stay=v2b, ANTI43_Z_U=bt(delta("ANTI43_Z", U_, CELLS)),
                 verdict=("the anti-self heads are causal for leaving the anchor" if v2["est"] >= .10 and v2["lo"] > 0 else "not"),
                 verdict_43=("FA43 anti-self heads causal for leaving" if v2b["est"] >= .05 and v2b["lo"] > 0 else "not"))
# ---------------------------------------------------------------- V3
v3 = bt(lambda ix: frac("H16_R", T_, K5)(ix) - np.mean([frac(x, T_, K5)(ix) for x in ("RAND1a_R", "RAND1b_R")]))
g16 = bt(delta("G16", "back", K5))
res["V3"] = dict(diff=v3, f_H16_T=bt(frac("H16_R", T_, K5)), f_RAND1_T={x: bt(frac(x, T_, K5)) for x in ("RAND1a_R", "RAND1b_R")}, G16_back=g16,
                 G16_T=bt(delta("G16", T_, K5)),
                 verdict=("h16 carries selective predecessor" if v3["est"] >= .10 and v3["lo"] > 0 else "not above random"),
                 verdict_gate=("k opens h16's gate causally" if g16["est"] >= .05 and g16["lo"] > 0 else "not observed"))
# ---------------------------------------------------------------- V4
m4748, m49, p47 = bt(frac("M4748_R", "back", K5)), bt(frac("M49_R", "back", K5)), bt(frac("PATH47", "back", K5))
if m4748["est"] >= .25 and m4748["lo"] > 0 and m49["est"] <= .10:
    v4 = "MLP47-48 carry the one-back, MLP49 does not"
elif m49["est"] >= .25 and m49["lo"] > 0:
    v4 = "MLP49 carries the one-back"
else:
    v4 = "neither alone"
res["V4"] = dict(f_M4748_back=m4748, f_M49_back=m49, f_M47_back=bt(frac("M47_R", "back", K5)), f_M48_back=bt(frac("M48_R", "back", K5)),
                 f_PATH47_back=p47, f_PATH47_U=bt(frac("PATH47", U_, K5)), verdict=v4,
                 verdict_path=("the FA47 -> MLP47 path carries part of the one-back" if p47["est"] >= .15 and p47["lo"] > 0 else "not"))
# ---------------------------------------------------------------- V5
# cache late profiles for speed
ZC = {}
for k, ps in TWO:
    for p in ps:
        for a in ("BAND_PX", "M49_PX", "M47_PX", "M48_PX", "H16_PX"):
            ZC[(k, p, a)] = np.array([z_late(r, f"k{k}p{p}_{a}", L0) for r in rows])
            ZC[(k, p, a, "E")] = np.array([z_late(r, f"k{k}p{p}_{eref(k, p, a)}", L0) for r in rows])
def shift_c(a):
    def f(ix):
        vals = []
        for k, ps in TWO:
            for p in ps:
                pp = ps[1] if p == ps[0] else ps[0]
                za, ze = ZC[(k, p, a)][ix], ZC[(k, p, a, "E")][ix]
                vals.append(float(np.mean((za[:, pp - 2] - za[:, p - 2]) - (ze[:, pp - 2] - ze[:, p - 2]))))
        return float(np.mean(vals))
    return f
sb = bt(shift_c("BAND_PX"))
res["V5"] = dict(S_BAND=sb, valid=bool(sb["lo"] > 0))
for X in ("M49", "M47", "M48", "H16"):
    rX = bt(lambda ix, X=X: shift_c(f"{X}_PX")(ix) / (shift_c("BAND_PX")(ix) + 1e-12))
    v = ("X carries an anchor-keyed address".replace("X", X) if rX["est"] >= .5 and rX["lo"] > 0 else
         "anchor-independent" if rX["est"] <= .2 and rX["hi"] <= .35 else "mixed")
    res["V5"][X] = dict(S=bt(shift_c(f"{X}_PX")), ratio=rX, verdict=(v if res["V5"]["valid"] else "transfer test invalid"))
# answers moving to the donor's predecessor (descriptive)
res["V5"]["answers_at_donor_pred"] = {}
for a in ("BAND_PX", "M49_PX", "M47_PX", "M48_PX", "H16_PX"):
    vals = []
    for k, ps in TWO:
        for p in ps:
            pp = ps[1] if p == ps[0] else ps[0]
            la = np.array([r["land"][f"k{k}p{p}_{a}"] for r in rows]); le = np.array([r["land"][f"k{k}p{p}_{eref(k, p, a)}"] for r in rows])
            vals.append(float(np.mean(la == pp - 1) - np.mean(le == pp - 1)))
    res["V5"]["answers_at_donor_pred"][a] = float(np.mean(vals))
# ---------------------------------------------------------------- V6, V9
v6 = bt(lambda ix: -delta("SELFADDR_R", "stay", CELLS)(ix))
res["V6"] = dict(stay=v6, verdict=("the self-address heads' END down-weighting is causal for leaving" if v6["est"] >= .05 and v6["lo"] > 0 else "not"))
def org(fn):
    a1 = bt(lambda ix: fn(5, 3, ix) - fn(3, 5, ix)); a2 = bt(lambda ix: fn(5, 5, ix) - fn(3, 7, ix))
    if a1["est"] >= .15 and a1["lo"] > 0 and a2["est"] >= .15 and a2["lo"] > 0:
        v = "k-organised"
    elif abs(a1["est"]) <= .10 and abs(a2["est"]) <= .10 and a1["lo"] >= -.15 and a1["hi"] <= .15 and a2["lo"] >= -.15 and a2["hi"] <= .15:
        v = "d-organised"
    else:
        v = "mixed"
    return dict(d_minus1=a1, d_plus1=a2, verdict=v)
res["V9"] = org(lambda k, p, ix: float(TAB[(k, p, "E")]["back"][ix].mean()))
# ---------------------------------------------------------------- descriptive
res["table"] = {f"k{k}p{p}_{a}": {x: round(float(v.mean()), 3) for x, v in d.items()} for (k, p, a), d in TAB.items()}
skip = ("E", "R", "EK", "SELF_FA", "SELF_M", "SELF_P")
res["f_k5"] = {a: {X: bt(frac(a, X, K5)) for X in ("back", U_, T_)} for a in J["arms_r"] if a not in skip}
res["f_k3"] = {a: {X: round(frac(a, X, [c for c in CELLS if c[0] == 3])(np.arange(S)), 3) for X in ("back", U_, T_)} for a in J["arms_r"] if a not in skip}
json.dump(res, open(OUT, "w"), indent=1)
summ = dict(I0=res["I0"]["all"], POS=res["POS"]["valid"], V1=[res["V1"]["verdict_a"], res["V1"]["verdict_b"], res["V1"]["verdict_c"]],
            V2=[res["V2"]["verdict"], res["V2"]["verdict_43"]], V3=[res["V3"]["verdict"], res["V3"]["verdict_gate"]],
            V4=[res["V4"]["verdict"], res["V4"]["verdict_path"]], V5={X: res["V5"][X]["verdict"] for X in ("M49", "M47", "M48", "H16")},
            V6=res["V6"]["verdict"], V9=res["V9"]["verdict"])
print(json.dumps(summ, indent=1))
print("->", OUT)
