"""S873 analysis (rules fixed before the data; knowledge/DECISIONS.md S873 entry).  Inputs results/r3/s873_<tag>_transplant.json and S872 v4's
results (results/r3/s872_<s872>_jlens.json, s872_<s872>_scenes/scene<si>.npz: states and gradients of the same scenes and pins).
Objectives of a run (as S872): z_b = L51 block score of picture b (late_mean heads, rows B1 / B4 / answer); L_i = menu logit of picture i;
T1 = z_(p-1) - z_p; T3 = L_(p-1) - L_p (S869's d1 pair).  Writer effect of a run X: W(X) = T3(X) - T3(X_BANDR) (band read-row outputs <- R_c).
Natural W_c = T3(E_c) - T3(E_c_BANDR) (sub-pass 0).  Transplant W_cd = W(X_cd) for donors d1, d2 (context c, END - START difference of d).
Predictions (frozen procedure, computed here from S872 v4 only): lens_cd = <gband_c^S(T3), dh46_d> with gband_c^S = scene c's Simpson-averaged
read-row band pathway (S872: (g_R + 4 g_M + g_E) / 6) and dh46_d = h46(E_d) - h46(R_d) (S872 states); per pin p the calibration a_p + b_p lens
fitted on S872's own natural pairs (all 48 scenes: W_c from S872 vs lens_cc); mean-lens version with the pin's mean gband over S872's scenes.
Scene bootstrap 4,000 (seed 873) over context scenes.
I0: t1 >= .95; START(p) answers p >= .95; the control X_cc (R_c + (E_c - R_c)) answers as E_c in >= .95 and median |T3(X_cc) - T3(E_c)| <= .05;
  the refs E_c of sub-passes 0 and 1 have equal T3 in >= .95 of (scene, pin) (no thread race); L47 anchor = p for E_c and R_c >= .90.
A1 (state or context; review 3 'same address, different writer'): across transplant cells, r_state = Pearson(W_cd, W_d) (the donor's own
  natural writer effect) and r_ctx = Pearson(W_cd, W_c) (the context scene's), within-pin centred: r_state - r_ctx >= .20 with the interval
  above 0 -> 'the writer effect follows the transplanted END - START state difference'; <= -.20 with the interval below 0 -> 'it follows the
  context scene (downstream sensitivity)'; else 'both'.  Descriptive: an additive two-way fit W[c, d] = mu_p + alpha_c + beta_d (share of
  var(beta)).
A2 (the lens predicts transplants): Pearson(W_cd, a_p + b_p lens_cd) over transplant cells (within-pin centred) >= .50 with the lower bound
  > .30 -> 'the scene-specific Jacobian lens predicts the writer effect of transplanted states'; else 'not'.  Reported: the mean-lens
  prediction, the donor's natural W_d and the context's W_c as competing predictors (R^2 against W_cd about the pin mean).
A3 (does the lens span carry the writer's input): ratio_span = mean W(X_SPAN) / mean W_c and ratio_perp = mean W(X_PERP) / mean W_c (pooled
  over pins; ratio of means, scene bootstrap): ratio_span >= .70 and ratio_perp <= .30 -> 'the 16 lens directions carry the writer's input';
  ratio_perp >= .70 and ratio_span <= .30 -> 'the writer reads directions outside the lens span'; else 'split'.
A4 (address-lens coordinate swap, Gurnee et al.'s test): dT1(X) = T1(X) - T1(E_c).  Predicted dT1 of SWAP from S872: <g_T1,c^E(in46),
  a V (sigma(c) - c)> with c the lens coordinates of S872's h46(E_c) (V = other-half mean lens vectors as in the run).  Pearson(dT1_obs,
  dT1_pred) over SWAP runs >= .50 with the lower bound > .30, and mean |dT1(SWAP)| >= 2 x mean |dT1(CRAND)| -> 'the address-lens coordinates
  are causally used as the lens predicts'; else 'not'.  Reported: SWAP2 (dose), CFAR, R_SWAP, answer shifts (P(answer = p) and P(= p - 1)).
Usage: python3 s873_analyze.py TAG [--s872 s872v4_27b]"""
from __future__ import annotations

import argparse, collections, json
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("tag"); ap.add_argument("--s872", default="s872v4_27b"); A = ap.parse_args()
R3 = "/vlm/results/r3"
J = json.load(open(f"{R3}/s873_{A.tag}_transplant.json"))
J72 = json.load(open(f"{R3}/s872_{A.s872}_jlens.json")); D72 = f"{R3}/s872_{A.s872}_scenes"
OUT = f"{R3}/s873_analysis_{A.tag}.json"
rng = np.random.default_rng(873); NB = 4000
rows = J["rows"]; S = len(rows); PINS = J["pins"]; N = J["N"]; lm = J["late_mean"]
lm72 = J72["late_mean"]


def unbits(a):
    return (a.view(np.uint16).astype(np.uint32) << 16).view(np.float32)


def zvec(zl, heads):
    z = np.asarray(zl["51"], dtype=float); return z[:, heads].mean(1).mean(0)


def obj(r, key, p, heads):
    z = zvec(r["zl"][key], heads); L = np.asarray(r["mlog"][key], dtype=float)
    return dict(T1=z[p - 2] - z[p - 1], T3=L[p - 2] - L[p - 1], z=z)


def anchor(q):
    q = np.asarray(q); am = [int(np.argmax(q[i])) + 1 for i in (2, 5, 9)]
    c = collections.Counter(am).most_common(); return c[0][0] if c[0][1] >= 2 else am[2]


H51 = lm["51"]
REC = {}
for si, r in enumerate(rows):
    for p in PINS:
        g = lambda nm: obj(r, nm, p, H51)
        E0, E1, R0 = g(f"Ep{p}@0"), g(f"Ep{p}@1"), g(f"Rp{p}@0")
        d = dict(si=si, p=p, E=E0, E1=E1, R=R0, xE=r["land"][f"Ep{p}"], xR=r["land"][f"Rp{p}"], donors=r["donors"][f"p{p}"],
                 anch=(anchor(r["qk47"][f"Ep{p}"]), anchor(r["qk47"][f"Rp{p}"])))
        d["W_nat"] = E0["T3"] - g(f"Ep{p}_E_BANDR")["T3"]
        d["X_cc"] = g(f"Ep{p}_X_cc"); d["x_cc"] = r["land"][f"Ep{p}_X_cc"]
        d["W_t"] = [g(f"Ep{p}_X_d{j}")["T3"] - g(f"Ep{p}_X_d{j}_BANDR")["T3"] for j in (1, 2)]
        d["W_span"] = g(f"Ep{p}_X_SPAN")["T3"] - g(f"Ep{p}_X_SPAN_BANDR")["T3"]
        d["W_perp"] = g(f"Ep{p}_X_PERP")["T3"] - g(f"Ep{p}_X_PERP_BANDR")["T3"]
        for nm in ("SWAP", "SWAP2", "CFAR", "CRAND", "R_SWAP"):
            base = E1 if nm != "R_SWAP" else g(f"Rp{p}@1")
            o = g(f"Ep{p}_{nm}"); d[f"dT1_{nm}"] = o["T1"] - base["T1"]; d[f"x_{nm}"] = r["land"][f"Ep{p}_{nm}"]
        REC[(si, p)] = d
# ---------------------------------------------------------------- S872 lens quantities (frozen procedure)
def simpson(Z, key):
    return sum(w * unbits(Z[key.format(a=a)]).astype(np.float64) for a, w in (("R", 1 / 6), ("M", 4 / 6), ("E", 1 / 6)))
cT3 = {p: np.zeros(2 * N) for p in PINS}; cT1 = {p: np.zeros(2 * N) for p in PINS}
for p in PINS:
    cT3[p][N + p - 2] += 1; cT3[p][N + p - 1] -= 1; cT1[p][p - 2] += 1; cT1[p][p - 1] -= 1
GB, DH, HE, G1E = {}, {}, {}, {}
S72 = len(J72["rows"])
for s in range(S72):
    Z = np.load(f"{D72}/scene{s:02d}.npz")
    for p in PINS:
        GB[(s, p)] = np.tensordot(cT3[p], simpson(Z, f"p{p}_gband_{{a}}"), 1).astype(np.float32)
        hE = unbits(Z[f"p{p}_st_E_in46"]).astype(np.float32); hR = unbits(Z[f"p{p}_st_R_in46"]).astype(np.float32)
        DH[(s, p)] = hE - hR; HE[(s, p)] = hE
        gE = unbits(Z[f"p{p}_g_E_in46"]).astype(np.float64)
        G1E[(s, p)] = np.tensordot(cT1[p], gE, 1).astype(np.float32)
        if s == 0:
            pass
    del Z
GBM = {p: np.mean([GB[(s, p)] for s in range(S72)], 0) for p in PINS}
# lens vectors as in the run (other-half mean of the E-pass gradients of z_1..8 at in46)
half = lambda s: 0 if s < 24 else 1
LVZ = {}
for h in (0, 1):
    for p in PINS:
        src = [s for s in range(S72) if half(s) == 1 - h] or [s for s in range(S72) if half(s) == h]
        acc = None
        for s in src:
            Z = np.load(f"{D72}/scene{s:02d}.npz"); g = unbits(Z[f"p{p}_g_E_in46"])[:N].astype(np.float64); acc = g if acc is None else acc + g
        LVZ[(h, p)] = (acc / len(src)).astype(np.float32)
# S872 natural writer effects for the calibration (from S872's own causal arms)
def W72(s, p):
    r = J72["rows"][s]; o = lambda nm: obj(r, nm, p, lm72["51"])
    return o(f"Ep{p}")["T3"] - o(f"Ep{p}_E_BANDR")["T3"]
CAL = {}
for p in PINS:
    x = np.array([float(np.sum(GB[(s, p)] * DH[(s, p)])) for s in range(S72)]); y = np.array([W72(s, p) for s in range(S72)])
    xm = np.array([float(np.sum(GBM[p] * DH[(s, p)])) for s in range(S72)])
    CAL[p] = dict(scene=np.polyfit(x, y, 1).tolist(), mean=np.polyfit(xm, y, 1).tolist())
for (si, p), d in REC.items():
    b, a0 = CAL[p]["scene"]; bm, am = CAL[p]["mean"]
    d["pred_t"] = [a0 + b * float(np.sum(GB[(si, p)] * DH[(dd, p)])) for dd in d["donors"]]
    d["pred_m"] = [am + bm * float(np.sum(GBM[p] * DH[(dd, p)])) for dd in d["donors"]]
    # predicted dT1 of the swaps (first order, S872's E-pass gradient of T1 at in46)
    V = LVZ[(half(si), p)]; h = HE[(si, p)].reshape(-1).astype(np.float64)
    for nm, idx, al in (("SWAP", (p - 2, p - 1), 1.0), ("SWAP2", (p - 2, p - 1), 2.0), ("CFAR", (p, p + 1), 1.0)):
        Vf = V[list(idx)].reshape(2, -1).astype(np.float64); c = np.linalg.solve(Vf @ Vf.T, Vf @ h); cs = c[::-1]
        d[f"pred_dT1_{nm}"] = float(G1E[(si, p)].reshape(-1).astype(np.float64) @ (al * (cs - c) @ Vf))
# ---------------------------------------------------------------- helpers
SC = sorted({k[0] for k in REC})
def bt(fn):
    n = len(SC); est = fn(np.arange(n)); bs = [fn(rng.integers(0, n, n)) for _ in range(NB)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return dict(est=float(est), lo=float(lo), hi=float(hi))
def cells(ix):
    out = []
    for i in ix:
        s = SC[i]
        for p in PINS:
            d = REC[(s, p)]
            for j, dd in enumerate(d["donors"]):
                out.append((p, d["W_t"][j], REC[(dd, p)]["W_nat"], d["W_nat"], d["pred_t"][j], d["pred_m"][j]))
    return out
def wcorr(ix, a_, b_):
    C = cells(ix); P = np.array([c[0] for c in C]); X = np.array([c[a_] for c in C]); Y = np.array([c[b_] for c in C])
    for p in PINS:
        m = P == p; X[m] -= X[m].mean(); Y[m] -= Y[m].mean()
    return float(np.corrcoef(X, Y)[0, 1])
res = dict(tag=A.tag, n_scenes=S, s872=A.s872, calib=CAL)
# ---------------------------------------------------------------- I0
st = float(np.mean([rows[s]["land"][f"START{p}"] == p for s in range(S) for p in PINS]))
cc_same = float(np.mean([d["x_cc"] == d["xE"] for d in REC.values()])); cc_dT = float(np.median([abs(d["X_cc"]["T3"] - d["E"]["T3"]) for d in REC.values()]))
race = float(np.mean([d["E"]["T3"] == d["E1"]["T3"] for d in REC.values()]))
reach = float(np.mean([a_ == d["p"] for d in REC.values() for a_ in d["anch"]]))
I0 = dict(t1=bool((J["t1_menu"] or 0) >= .95), START=bool(st >= .95), Xcc_is_E=bool(cc_same >= .95 and cc_dT <= .05), no_race=bool(race >= .95),
          pin_reach=bool(reach >= .90))
res["I0"] = dict(**I0, all=bool(all(I0.values())), t1_value=J["t1_menu"], START_acc=st, Xcc_same=cc_same, Xcc_dT3=cc_dT, ref_equal=race, reach=reach)
ok = res["I0"]["all"]
# ---------------------------------------------------------------- A1
a1 = bt(lambda ix: wcorr(ix, 1, 2) - wcorr(ix, 1, 3))
def additive():
    out = {}
    for p in PINS:
        rows_ = [(s, s, REC[(s, p)]["W_nat"]) for s in SC] + [(s, dd, REC[(s, p)]["W_t"][j]) for s in SC for j, dd in enumerate(REC[(s, p)]["donors"])]
        X = np.zeros((len(rows_), 2 * S)); y = np.array([w for _, _, w in rows_])
        for i, (c, dd, _) in enumerate(rows_):
            X[i, c] = 1; X[i, S + dd] = 1
        coef, *_ = np.linalg.lstsq(X - X.mean(0), y - y.mean(), rcond=None)
        out[f"p{p}"] = dict(var_context=float(np.var(coef[:S])), var_state=float(np.var(coef[S:])), share_state=float(np.var(coef[S:]) / (np.var(coef[:S]) + np.var(coef[S:]))))
    return out
res["A1"] = dict(diff=a1, r_state=bt(lambda ix: wcorr(ix, 1, 2)), r_context=bt(lambda ix: wcorr(ix, 1, 3)), additive=additive(),
                 verdict=(("the writer effect follows the transplanted END - START state difference" if a1["est"] >= .20 and a1["lo"] > 0 else
                           "it follows the context scene (downstream sensitivity)" if a1["est"] <= -.20 and a1["hi"] < 0 else "both") if ok else "not read"))
# ---------------------------------------------------------------- A2
a2 = bt(lambda ix: wcorr(ix, 1, 4))
def r2_against(ix, k):
    C = cells(ix); P = np.array([c[0] for c in C]); Y = np.array([c[1] for c in C]); X = np.array([c[k] for c in C])
    out = 0.; den = 0.
    for p in PINS:
        m = P == p; y = Y[m] - Y[m].mean(); x = X[m] - X[m].mean(); b = (x @ y) / (x @ x) if x @ x > 0 else 0.
        out += np.sum((y - b * x) ** 2); den += np.sum(y ** 2)
    return float(1 - out / den)
res["A2"] = dict(corr_scene_lens=a2, corr_mean_lens=bt(lambda ix: wcorr(ix, 1, 5)),
                 R2={nm: bt(lambda ix, k=k: r2_against(ix, k)) for nm, k in (("scene_lens", 4), ("mean_lens", 5), ("donor_W", 2), ("context_W", 3))},
                 verdict=(("the scene-specific Jacobian lens predicts the writer effect of transplanted states" if a2["est"] >= .50 and a2["lo"] > .30 else "not")
                          if ok else "not read"))
# ---------------------------------------------------------------- A3
def ratio(ix, key):
    num = [REC[(SC[i], p)][key] for i in ix for p in PINS]; den = [REC[(SC[i], p)]["W_nat"] for i in ix for p in PINS]
    return float(np.mean(num) / np.mean(den))
rs, rp = bt(lambda ix: ratio(ix, "W_span")), bt(lambda ix: ratio(ix, "W_perp"))
res["A3"] = dict(ratio_span=rs, ratio_perp=rp, by_pin={f"p{p}": dict(span=float(np.mean([REC[(s, p)]["W_span"] for s in SC]) / np.mean([REC[(s, p)]["W_nat"] for s in SC])),
                                                                     perp=float(np.mean([REC[(s, p)]["W_perp"] for s in SC]) / np.mean([REC[(s, p)]["W_nat"] for s in SC])))
                                                   for p in PINS},
                 verdict=(("the 16 lens directions carry the writer's input" if rs["est"] >= .70 and rp["est"] <= .30 else
                           "the writer reads directions outside the lens span" if rp["est"] >= .70 and rs["est"] <= .30 else "split") if ok else "not read"))
# ---------------------------------------------------------------- A4
def swapcorr(ix):
    o = [REC[(SC[i], p)]["dT1_SWAP"] for i in ix for p in PINS]; pr = [REC[(SC[i], p)]["pred_dT1_SWAP"] for i in ix for p in PINS]
    return float(np.corrcoef(o, pr)[0, 1])
a4c = bt(swapcorr)
mag = {nm: float(np.mean([abs(d[f"dT1_{nm}"]) for d in REC.values()])) for nm in ("SWAP", "SWAP2", "CFAR", "CRAND", "R_SWAP")}
res["A4"] = dict(corr_obs_pred=a4c, mean_abs_dT1=mag, mean_dT1={nm: float(np.mean([d[f"dT1_{nm}"] for d in REC.values()])) for nm in mag},
                 answers={nm: dict(P_p=float(np.mean([d[f"x_{nm}"] == d["p"] for d in REC.values()])), P_pm1=float(np.mean([d[f"x_{nm}"] == d["p"] - 1 for d in REC.values()])))
                          for nm in mag},
                 natural=dict(P_p=float(np.mean([d["xE"] == d["p"] for d in REC.values()])), P_pm1=float(np.mean([d["xE"] == d["p"] - 1 for d in REC.values()]))),
                 verdict=(("the address-lens coordinates are causally used as the lens predicts" if a4c["est"] >= .50 and a4c["lo"] > .30 and mag["SWAP"] >= 2 * mag["CRAND"]
                           else "not") if ok else "not read"))
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"]["all"], A1=[res["A1"]["verdict"], a1["est"]], A2=[res["A2"]["verdict"], a2["est"]],
                      A3=[res["A3"]["verdict"], rs["est"], rp["est"]], A4=[res["A4"]["verdict"], a4c["est"], mag]), indent=1))
print("->", OUT)
