"""S872b analysis = S872's registered analysis (exp/s872_analyze.py, sha256 df24395fbe2d440d) with one instrument criterion replaced
(knowledge/DECISIONS.md S872b entry, registered before S872b's data): in S872 the gradient passes (batch 1) and the stored readout (batch 13)
differed by unbiased bf16 batch numerics (median |dT1| .068 = 5.7% of the across-run sd, mean +.010) and the absolute gate .05 (set from a
two-scene smoke) failed; here I0's gradient check is relative - median |T1_grad(E) - T1_cap(E)| <= .10 sd and |mean| <= .05 sd, sd = the
across-run standard deviation of T1_cap(E).  Everything else is S872's.
S872 analysis text follows.  Inputs results/r3/s872_<tag>_jlens.json and the per-scene
npz files of results/r3/s872_<tag>_scenes/.
Objective values of a run: z_b = L51 block score of picture b (late_mean heads, rows B1 / B4 / answer averaged, from the stored caps), L_i = the menu
  logit of picture i (stored readout).  T1 = z_(p-1) - z_p, T2 = z_p - z_(p+1), T3 = L_(p-1) - L_p (S869's d1 pair).
Causal effects per (scene, pin p): writer W(o) = o(E) - o(E_BANDR) (band read-row outputs <- R); W_POS(o) = o(E) - o(E_BANDR_POS); total
  TOT(o) = o(E) - o(R); arm effect o(E) - o(E_arm).
Lenses (gradients from the npz; Simpson path average g_S = (g_R + 4 g_M + g_E) / 6 along R -> E at the input of layer 46):
  total lens <g46, dh46>;  read-row writer lens <gband, dh46> (gband = J_B,rows^T g50(rows), the band pathway to the read rows' outputs);
  POS writer lens <g46 - g50, dh46>;  dh46 = h46(E) - h46(R) at every position from the first differing question token on (pre-writer).
  Mean-Jacobian lens: the gradient averaged over the calibration-half scenes of the same pin (scenes 0-23), applied to each test scene's own dh46
  (scenes 24-47); scene-specific lens: the scene's own gradients.  Calibrated = y ~ a + b * lens fitted on the calibration half (same pin).
R^2 (per-scene, beyond the cell) = 1 - sum (y - yhat)^2 / sum (y - c_p)^2 over test runs pooled over pins, c_p = the calibration-half mean of y
  for pin p (a cell constant scores 0); Corr = Pearson of y and yhat after centring both within each pin (within-pin correlation); the pooled
  R^2 about the test mean (S869 F4's .092 style) is reported as well; scene bootstrap 4,000 over test scenes (seed 872).
I0: t1 >= .95; START(p) answers p >= .95; R answers p >= .90; L47 anchor (majority of B1 / B4 / answer, answer row if all differ) = p for E and
  R >= .90; the residual control E_RES46POS answers as R in >= .95 and median |T3(E_RES46POS) - T3(R)| <= .05; every stored gradient finite;
  gradient passes reproduce the caps: median |T1_grad(E) - T1_cap(E)| <= .05.  Nothing is read if I0 fails.
J1 (primary; S869 F4's d1): W(T3) on test runs from the calibrated mean-Jacobian read-row writer lens: R^2 >= .30 with the lower bound > .092
  and Corr >= .55 with the lower bound > .30 -> 'the model's mean sensitivity computes the per-scene writer effect from the pre-writer state';
  R^2 >= .15 with the lower bound > 0 -> 'partial'; else 'not'.
J1z: the same rule for W(T1) (the late predecessor preference at L51).
J2 (logit vector; S871's target): the per-run residual R = dL^W - c(p) (dL^W = centred menu-logit writer effect, c(p) its calibration-half
  mean) from the calibrated mean-Jacobian read-row lens of the eight logits (deviation from its calibration-half mean, one scale per pin):
  R^2 (relative to zero) >= .20 with the lower bound > 0 -> 'the Jacobian lens predicts the per-run writer variation the linear probe missed';
  else 'not'.
J3 (total vs writer): the calibrated mean-Jacobian total lens for TOT(T3); if J1 is 'not' and its R^2 >= .30 -> 'the lens tracks the general
  late sensitivity, not the writer'; otherwise reported.
J4 (H3, state dependence): R^2 of the calibrated scene-specific read-row writer lens minus the mean-Jacobian one for W(T3) >= .20 with the
  interval above 0 -> 'H3: the per-scene writer size depends on scene-specific downstream sensitivity'; else 'not'.  Completeness (reported):
  uncalibrated R^2 of the scene-specific Simpson total lens for TOT(T1) and TOT(T3).
J5 (H4, address specificity; read only if the same-pin calibrated R^2 of J1 >= .15): mean-Jacobian lens of pin p' (relative coordinates:
  objective T3 of p') applied to pin p != p' and calibrated on p's calibration half; mean R^2 over ordered pairs vs same-pin R^2:
  difference >= .20 with the interval above 0 -> 'H4: the writer lens is address-specific'; else 'the relative-address lens generalises'.
J6 (H2, distributed): a ridge (inner 4-fold CV penalty on the calibration half) of W(T3) on the projections of dh46 onto the top 5 principal
  directions of the calibration-half per-scene Simpson gband(T3) vectors (same pin), minus the calibrated single mean-direction R^2 >= .15 with
  the interval above 0 -> 'H2: the writer reads several directions'; else 'not'.
J7 (H5, positions): the scene-specific read-row writer lens summed within position groups (rows = '?' + B0-B4 + A4, tail, menu); causal effects
  of E_RES46ROWS / TAIL / MENU on T3; the group with the largest |mean lens part| = the group with the largest |mean causal effect| ->
  'lens and patching put the writer's input at the same positions'; else 'not'.
J8 (components): scene-specific Simpson attributions C_m = <g at m's output, dO_m> at the read rows for the 8 band components (mix / mlp 46-49)
  and FA47 head groups (h16, anti-self, self-address, rest: <g at mid47, sum_h W_O dho_h>) vs the causal effect T3(E) - T3(E_arm) on test runs:
  Pearson over the 12 arm means >= .70 -> 'the lens attributes the writer to the same components as causal swaps'; else 'not'.
J9 (descriptive): the address trajectory - per saved point, the mean-Jacobian readout <g_zb, dh> (b relative to p) and of the E state centred
  over b, for p 4 and 5; the first point where z_(p-1) overtakes z_p.
Usage: python3 s872_analyze.py TAG"""
from __future__ import annotations

import collections, json, sys
import numpy as np

TAG = sys.argv[1]
J = json.load(open(f"/vlm/results/r3/s872_{TAG}_jlens.json"))
NPD = J["npz_dir"].replace("/vlm/", "/vlm/")
OUT = f"/vlm/results/r3/s872b_analysis_{TAG}.json"
rng = np.random.default_rng(872)
NB = 4000
rows = J["rows"]; S = len(rows); PINS = J["pins"]; N = J["N"]; lm = J["late_mean"]
CAL = [s for s in range(S) if s < S // 2]; TEST = [s for s in range(S) if s >= S // 2]
OBJ = J["objectives"]; GR_ = J["groups"]
SAVE = [tuple(q) for q in J["save"]]; SAVE_M = [tuple(q) for q in J["save_m"]]
ARMS = [x[0] for x in J["specs"]]
COMPARMS = ["E_MIX46", "E_MLP46", "E_FA47", "E_MLP47", "E_MIX48", "E_MLP48", "E_MIX49", "E_MLP49"]
OUTPT = {"E_MIX46": ("mid", 46, ("in", 46)), "E_MLP46": ("in", 47, ("mid", 46)), "E_FA47": ("mid", 47, ("in", 47)), "E_MLP47": ("in", 48, ("mid", 47)),
         "E_MIX48": ("mid", 48, ("in", 48)), "E_MLP48": ("in", 49, ("mid", 48)), "E_MIX49": ("mid", 49, ("in", 49)), "E_MLP49": ("in", 50, ("mid", 49))}
HEADARMS = {"E_H16": J["h16"], "E_ANTI": J["anti"], "E_SELFADDR": J["selfaddr"], "E_REST47": J["rest47"]}


def unbits(a):
    return (a.view(np.uint16).astype(np.uint32) << 16).view(np.float32)


def zvec(r, key):
    z = np.asarray(r["zl"][key]["51"], dtype=float)                       # [3, 24, N]
    return z[:, lm["51"]].mean(1).mean(0)


def objs(r, key, p):
    z = zvec(r, key); L = np.asarray(r["mlog"][key], dtype=float)
    return dict(T1=z[p - 2] - z[p - 1], T2=z[p - 1] - z[p], T3=L[p - 2] - L[p - 1], Lc=L - L.mean(), z=z)


def coef(p):
    """objective-axis weights of T1, T3 (and T2) over the 16 stored objectives (z1..z8, L1..L8)."""
    c = {k: np.zeros(2 * N) for k in ("T1", "T2", "T3")}
    c["T1"][p - 2] += 1; c["T1"][p - 1] -= 1
    c["T2"][p - 1] += 1; c["T2"][p] -= 1
    c["T3"][N + p - 2] += 1; c["T3"][N + p - 1] -= 1
    return c


def anchor(r, key):
    q = np.asarray(r["qk"][key]["47"]); am = [int(np.argmax(q[i])) + 1 for i in (2, 5, 9)]
    c = collections.Counter(am).most_common(); return c[0][0] if c[0][1] >= 2 else am[2]


res = dict(tag=TAG, n_scenes=S, n_runs=J["n_runs"], cal=CAL, test=TEST)
# ---------------------------------------------------------------- behaviour, causal effects
REC = {}
for si, r in enumerate(rows):
    for p in PINS:
        E, R = f"Ep{p}", f"Rp{p}"
        oE, oR = objs(r, E, p), objs(r, R, p)
        d = dict(si=si, p=p, xE=r["land"][E], xR=r["land"][R], oE=oE, oR=oR, arms={})
        for arm in ARMS:
            d["arms"][arm] = objs(r, f"{E}_{arm}", p); d["arms"][arm]["x"] = r["land"][f"{E}_{arm}"]
        d["W"] = {k: oE[k] - d["arms"]["E_BANDR"][k] for k in ("T1", "T2", "T3", "Lc")}
        d["WPOS"] = {k: oE[k] - d["arms"]["E_BANDR_POS"][k] for k in ("T1", "T2", "T3", "Lc")}
        d["TOT"] = {k: oE[k] - oR[k] for k in ("T1", "T2", "T3", "Lc")}
        g_ = r["gz"]; d["T1grad"] = {a: g_[f"p{p}_{a}"][p - 2] - g_[f"p{p}_{a}"][p - 1] for a in ("E", "R", "M")}
        d["oEg"], d["oRg"] = objs(r, f"{E}@g", p), objs(r, f"{R}@g", p)
        d["anchors"] = (anchor(r, E), anchor(r, R))
        REC[(si, p)] = d
# ---------------------------------------------------------------- lenses from the npz (two passes: calibration sums, then every scene)
def load(si):
    return np.load(f"{NPD}/scene{si:02d}.npz")
GS = {}; nfin = 0; nbad = 0
SUM = {p: None for p in PINS}; SUMZ = {p: None for p in PINS}; GB_T3 = {p: [] for p in PINS}
for si in CAL:
    Z = load(si)
    for p in PINS:
        g46 = sum(w * unbits(Z[f"p{p}_g_{a}_in46"]).astype(np.float64) for a, w in (("R", 1 / 6), ("M", 4 / 6), ("E", 1 / 6)))
        g50 = sum(w * unbits(Z[f"p{p}_g_{a}_in50"]).astype(np.float64) for a, w in (("R", 1 / 6), ("M", 4 / 6), ("E", 1 / 6)))
        gb = sum(w * unbits(Z[f"p{p}_gband_{a}"]).astype(np.float64) for a, w in (("R", 1 / 6), ("M", 4 / 6), ("E", 1 / 6)))
        cur = dict(g46=g46, g50=g50, gb=gb)
        SUM[p] = cur if SUM[p] is None else {k: SUM[p][k] + cur[k] for k in cur}
        GB_T3[p].append(np.tensordot(coef(p)["T3"], gb, 1).ravel().astype(np.float32))
        zz = {q: unbits(Z[f"p{p}_g_E_{q[0]}{q[1]}"])[:N].astype(np.float64) for q in SAVE}
        SUMZ[p] = zz if SUMZ[p] is None else {q: SUMZ[p][q] + zz[q] for q in SAVE}
    del Z
MEAN = {p: {k: v / len(CAL) for k, v in SUM[p].items()} for p in PINS}
MEANZ = {p: {q: v / len(CAL) for q, v in SUMZ[p].items()} for p in PINS}
PCS = {}
for p in PINS:
    X = np.stack(GB_T3[p]); X = X - X.mean(0)
    U, Sv, Vt = np.linalg.svd(X, full_matrices=False); PCS[p] = Vt[:5]
LENS = {}
TRAJ = {p: collections.defaultdict(list) for p in PINS}
for si in range(S):
    Z = load(si); r = rows[si]
    rix = r["groups"]["qm"] + r["groups"]["B"] + r["groups"]["A"]
    G = {"rows": rix, "tail": r["groups"]["tail"], "menu": r["groups"]["menu"]}
    for p in PINS:
        c = coef(p)
        st = {q: (unbits(Z[f"p{p}_st_E_{q[0]}{q[1]}"]).astype(np.float64), unbits(Z[f"p{p}_st_R_{q[0]}{q[1]}"]).astype(np.float64)) for q in SAVE}
        dh = st[("in", 46)][0] - st[("in", 46)][1]
        gS = {}
        for nm, key in (("g46", "g_{a}_in46"), ("g50", "g_{a}_in50"), ("gb", "gband_{a}")):
            arrs = {a: unbits(Z[f"p{p}_" + key.format(a=a)]) for a in ("R", "M", "E")}
            for a in arrs:
                nfin += arrs[a].size; nbad += int((~np.isfinite(arrs[a])).sum())
            gS[nm] = sum(w * arrs[a].astype(np.float64) for a, w in (("R", 1 / 6), ("M", 4 / 6), ("E", 1 / 6)))
        L = dict(p=p, si=si)
        for k in ("T1", "T2", "T3"):
            L[f"mean_band_{k}"] = float(np.sum(np.tensordot(c[k], MEAN[p]["gb"], 1) * dh))
            L[f"mean_tot_{k}"] = float(np.sum(np.tensordot(c[k], MEAN[p]["g46"], 1) * dh))
            L[f"mean_pos_{k}"] = float(np.sum(np.tensordot(c[k], MEAN[p]["g46"] - MEAN[p]["g50"], 1) * dh))
            L[f"scene_band_{k}"] = float(np.sum(np.tensordot(c[k], gS["gb"], 1) * dh))
            L[f"scene_tot_{k}"] = float(np.sum(np.tensordot(c[k], gS["g46"], 1) * dh))
            for gname, gi in G.items():
                L[f"scene_band_{k}_{gname}"] = float(np.sum(np.tensordot(c[k], gS["gb"], 1)[gi] * dh[gi]))
            for pp in PINS:                                                                          # cross-pin lens, relative coordinates
                L[f"cross{pp}_band_{k}"] = float(np.sum(np.tensordot(coef(pp)[k], MEAN[pp]["gb"], 1) * dh))
        L["mean_band_L"] = [float(np.sum(MEAN[p]["gb"][N + i] * dh)) for i in range(N)]
        L["pcs_T3"] = [float(np.sum(PCS[p][j].reshape(dh.shape) * dh)) for j in range(len(PCS[p]))]
        # component attributions at the read rows (scene-specific Simpson gradient at the component's output point)
        L["comp"] = {}
        for arm, (kd, l, prev) in OUTPT.items():
            q = (kd, l)
            gq = sum(w * unbits(Z[f"p{p}_g_{a}_{kd}{l}"]).astype(np.float64) for a, w in (("R", 1 / 6), ("M", 4 / 6), ("E", 1 / 6)))
            dO = (st[q][0] - st[q][1]) - (st[prev][0] - st[prev][1])
            L["comp"][arm] = {k: float(np.sum(np.tensordot(c[k], gq, 1)[rix] * dO[rix])) for k in ("T1", "T3")}
        gq = sum(w * unbits(Z[f"p{p}_g_{a}_mid47"]).astype(np.float64) for a, w in (("R", 1 / 6), ("M", 4 / 6), ("E", 1 / 6)))
        dho = unbits(Z[f"p{p}_dho47"]).astype(np.float64)                                      # [10, 24, E] in rowsC order
        for arm, hs in HEADARMS.items():
            L["comp"][arm] = {k: float(np.sum(np.tensordot(c[k], gq, 1)[rix] * dho[:, hs].sum(1))) for k in ("T1", "T3")}
        # address trajectory (mean-Jacobian readout of z_b at every saved point), relative b = p-2 .. p+1 (1-indexed p-1 .. p+2)
        for q in SAVE:
            dq = st[q][0] - st[q][1]
            gz = MEANZ[p][q]                                                                        # [8, P, E]
            D = np.tensordot(gz, dq, axes=([1, 2], [0, 1]))                                        # [8]
            A_ = np.tensordot(gz, st[q][0], axes=([1, 2], [0, 1])); A_ = A_ - A_.mean()
            if si in TEST:
                TRAJ[p][f"{q[0]}{q[1]}_D"].append(D.tolist()); TRAJ[p][f"{q[0]}{q[1]}_A"].append(A_.tolist())
        LENS[(si, p)] = L
    del Z
res["gradients_finite"] = dict(n=nfin, nonfinite=nbad)
# ---------------------------------------------------------------- I0
t1 = J["t1_menu"] or 0
st_acc = float(np.mean([rows[s]["land"][f"START{p}"] == p for s in range(S) for p in PINS]))
r_acc = float(np.mean([REC[k]["xR"] == k[1] for k in REC]))
reach = float(np.mean([a == k[1] for k in REC for a in REC[k]["anchors"]]))
ctl_same = float(np.mean([REC[k]["arms"]["E_RES46POS"]["x"] == REC[k]["xR"] for k in REC]))
ctl_dT = float(np.median([abs(REC[k]["arms"]["E_RES46POS"]["T3"] - REC[k]["oR"]["T3"]) for k in REC]))
gcon = float(np.median([abs(REC[k]["T1grad"]["E"] - REC[k]["oEg"]["T1"]) for k in REC]))
gconR = float(np.median([abs(REC[k]["T1grad"]["R"] - REC[k]["oRg"]["T1"]) for k in REC]))
dlist = np.array([REC[k]["T1grad"]["E"] - REC[k]["oEg"]["T1"] for k in REC]); sdT1 = float(np.std([REC[k]["oEg"]["T1"] for k in REC]))
I0 = dict(t1=bool(t1 >= .95), START_acc=bool(st_acc >= .95), R_acc=bool(r_acc >= .90), pin_reach=bool(reach >= .90),
          res46pos_is_R=bool(ctl_same >= .95 and ctl_dT <= .05), grads_finite=bool(nbad == 0),
          grad_matches_caps=bool(gcon <= .10 * sdT1 and abs(float(dlist.mean())) <= .05 * sdT1))
res["I0"] = dict(**I0, all=bool(all(I0.values())), t1_value=t1, START_acc_value=st_acc, R_acc_value=r_acc, pin_reach_value=reach,
                 res46pos_same_answer=ctl_same, res46pos_median_dT3=ctl_dT, grad_T1_median_absdiff_E=gcon, grad_T1_median_absdiff_R=gconR,
                 grad_T1_mean_diff_E=float(dlist.mean()), sd_T1_E=sdT1)
ok = res["I0"]["all"]
# ---------------------------------------------------------------- scoring helpers
BYS = collections.defaultdict(list)
for k in REC:
    BYS[k[0]].append(k)


def collect(xkey, yfn, cal=True, xfn=None, pins=PINS, pin_from=None):
    """test-run tuples (y, yhat, c_p, p) keyed by scene; calibrated per pin on the calibration half."""
    out = collections.defaultdict(list)
    for p in pins:
        xf = xfn if xfn else (lambda L: L[xkey])
        if pin_from is not None:
            xf = (lambda L, pf=pin_from[p]: L[f"cross{pf}_band_T3"])
        xs = np.array([xf(LENS[(s, p)]) for s in CAL]); ys = np.array([yfn(REC[(s, p)]) for s in CAL])
        b, a = np.polyfit(xs, ys, 1) if cal else (1.0, 0.0)
        for s in TEST:
            out[s].append((yfn(REC[(s, p)]), a + b * xf(LENS[(s, p)]), float(ys.mean()), p))
    return out


def r2c(pairs):
    def f(ix):
        P = [pq for s in ix for pq in pairs[TEST[s]]]
        y = np.array([u[0] for u in P]); yh = np.array([u[1] for u in P]); c = np.array([u[2] for u in P]); pp = np.array([u[3] for u in P])
        r2 = float(1 - np.sum((y - yh) ** 2) / np.sum((y - c) ** 2))
        yc, hc = y.copy(), yh.copy()
        for p in set(pp.tolist()):
            m = pp == p; yc[m] -= yc[m].mean(); hc[m] -= hc[m].mean()
        co = float(np.corrcoef(yc, hc)[0, 1]) if np.std(hc) > 0 else 0.0
        r2_pooled = float(1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2))
        return r2, co, r2_pooled
    return f


def bt(fn):
    n = len(TEST); est = fn(np.arange(n)); bs = [fn(rng.integers(0, n, n)) for _ in range(NB)]
    if isinstance(est, tuple):
        out = []
        for j in range(len(est)):
            v = [b[j] for b in bs]; lo, hi = np.nanpercentile(v, [2.5, 97.5]); out.append(dict(est=float(est[j]), lo=float(lo), hi=float(hi)))
        return out
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return dict(est=float(est), lo=float(lo), hi=float(hi))


def diff_bt(f1, f2):
    return bt(lambda ix: f1(ix)[0] - f2(ix)[0])


Y = {k: (lambda d, k=k: d["W"][k]) for k in ("T1", "T2", "T3")}
# ---------------------------------------------------------------- J1 / J1z
J1 = {}
for k in ("T3", "T1"):
    f = r2c(collect(f"mean_band_{k}", Y[k])); r2_, co, r2p = bt(f)
    raw_r2, raw_co, _ = bt(r2c(collect(f"mean_band_{k}", Y[k], cal=False)))
    byp = {f"p{p}": dict(zip(("R2", "Corr", "R2_pooled"), r2c(collect(f"mean_band_{k}", Y[k], pins=[p]))(np.arange(len(TEST))))) for p in PINS}
    v = ("the model's mean sensitivity computes the per-scene writer effect from the pre-writer state"
         if r2_["est"] >= .30 and r2_["lo"] > .092 and co["est"] >= .55 and co["lo"] > .30 else
         "partial" if r2_["est"] >= .15 and r2_["lo"] > 0 else "not")
    J1[k] = dict(R2=r2_, Corr=co, R2_pooled=r2p, raw_R2=raw_r2, raw_Corr=raw_co, by_pin=byp, verdict=v if ok else "not read",
                 pos_writer_lens=dict(zip(("R2", "Corr", "R2_pooled"), r2c(collect(f"mean_pos_{k}", lambda d, k=k: d["WPOS"][k]))(np.arange(len(TEST))))))
res["J1"] = dict(T3=J1["T3"], verdict=J1["T3"]["verdict"]); res["J1z"] = dict(T1=J1["T1"], verdict=J1["T1"]["verdict"])
# ---------------------------------------------------------------- J2 (the logit vector, S871's residual target)
cm = {p: np.mean([REC[(s, p)]["W"]["Lc"] for s in CAL], 0) for p in PINS}
lm_ = {p: np.mean([np.asarray(LENS[(s, p)]["mean_band_L"]) - np.mean(LENS[(s, p)]["mean_band_L"]) for s in CAL], 0) for p in PINS}
def lv(s, p):
    x = np.asarray(LENS[(s, p)]["mean_band_L"]); return (x - x.mean()) - lm_[p]
scale = {}
for p in PINS:
    xs = np.concatenate([lv(s, p) for s in CAL]); ys = np.concatenate([REC[(s, p)]["W"]["Lc"] - cm[p] for s in CAL])
    scale[p] = float(np.dot(xs, ys) / np.dot(xs, xs))
def j2(ix):
    num = den = 0.
    for i in ix:
        s = TEST[i]
        for p in PINS:
            y = REC[(s, p)]["W"]["Lc"] - cm[p]; yh = scale[p] * lv(s, p); num += np.sum((y - yh) ** 2); den += np.sum(y ** 2)
    return float(1 - num / den)
j2b = bt(j2)
res["J2"] = dict(R2=j2b, scale=scale, verdict=(("the Jacobian lens predicts the per-run writer variation the linear probe missed"
                                                 if j2b["est"] >= .20 and j2b["lo"] > 0 else "not") if ok else "not read"))
# ---------------------------------------------------------------- J3 total lens
ftot = r2c(collect("mean_tot_T3", lambda d: d["TOT"]["T3"])); t3r2, t3co, _ = bt(ftot)
res["J3"] = dict(R2=t3r2, Corr=t3co, verdict=(("the lens tracks the general late sensitivity, not the writer" if J1["T3"]["verdict"] == "not" and t3r2["est"] >= .30
                                                else "reported") if ok else "not read"))
# ---------------------------------------------------------------- J4 H3 + completeness
fs = r2c(collect("scene_band_T3", Y["T3"])); fm = r2c(collect("mean_band_T3", Y["T3"]))
j4 = diff_bt(fs, fm)
comp = {k: dict(zip(("R2", "Corr", "R2_pooled"), r2c(collect(f"scene_tot_{k}", lambda d, k=k: d["TOT"][k], cal=False))(np.arange(len(TEST))))) for k in ("T1", "T3")}
res["J4"] = dict(diff=j4, R2_scene=bt(fs)[0], R2_mean=bt(fm)[0], completeness_uncalibrated=comp,
                 verdict=(("H3: the per-scene writer size depends on scene-specific downstream sensitivity" if j4["est"] >= .20 and j4["lo"] > 0 else "not")
                          if ok else "not read"))
# ---------------------------------------------------------------- J5 H4
same = J1["T3"]["R2"]["est"]
if same >= .15:
    pairs = [(p, pf) for p in PINS for pf in PINS if pf != p]
    SAMEP = {p: r2c(collect("mean_band_T3", Y["T3"], pins=[p])) for p in PINS}
    CROSS = {(p, pf): r2c(collect(None, Y["T3"], pins=[p], pin_from={p: pf})) for p, pf in pairs}
    def j5(ix):
        return float(np.mean([SAMEP[p](ix)[0] for p in PINS]) - np.mean([CROSS[pq](ix)[0] for pq in pairs]))
    j5b = bt(j5)
    res["J5"] = dict(diff=j5b, verdict=(("H4: the writer lens is address-specific" if j5b["est"] >= .20 and j5b["lo"] > 0
                                         else "the relative-address lens generalises") if ok else "not read"))
else:
    res["J5"] = dict(verdict="not read (same-pin R^2 < .15)", same_pin_R2=same)
# ---------------------------------------------------------------- J6 H2
def ridge_pcs(ix):
    P = []
    for p in PINS:
        X = np.array([LENS[(s, p)]["pcs_T3"] for s in CAL]); yv = np.array([REC[(s, p)]["W"]["T3"] for s in CAL])
        mu, sd = X.mean(0), X.std(0) + 1e-9; Xs = (X - mu) / sd; ym = yv.mean()
        best, bl = None, None
        for lam in (1e-2, 1e-1, 1, 10, 100):
            err = 0.
            for f in range(4):
                te_ = np.arange(len(CAL)) % 4 == f; tr_ = ~te_
                w = np.linalg.solve(Xs[tr_].T @ Xs[tr_] + lam * np.eye(Xs.shape[1]), Xs[tr_].T @ (yv[tr_] - yv[tr_].mean()))
                err += np.sum((yv[te_] - yv[tr_].mean() - Xs[te_] @ w) ** 2)
            if best is None or err < best:
                best, bl = err, lam
        w = np.linalg.solve(Xs.T @ Xs + bl * np.eye(Xs.shape[1]), Xs.T @ (yv - ym))
        for i in ix:
            s = TEST[i]; x = (np.array(LENS[(s, p)]["pcs_T3"]) - mu) / sd
            P.append((REC[(s, p)]["W"]["T3"], ym + x @ w, ym))
    y = np.array([u[0] for u in P]); yh = np.array([u[1] for u in P]); c = np.array([u[2] for u in P])
    return float(1 - np.sum((y - yh) ** 2) / np.sum((y - c) ** 2)),
j6 = bt(lambda ix: ridge_pcs(ix)[0] - fm(ix)[0])
res["J6"] = dict(diff=j6, R2_pcs=bt(lambda ix: ridge_pcs(ix)[0]),
                 verdict=(("H2: the writer reads several directions" if j6["est"] >= .15 and j6["lo"] > 0 else "not") if ok else "not read"))
# ---------------------------------------------------------------- J7 H5 positions
lensg_signed = {g: float(np.mean([LENS[(s, p)][f"scene_band_T3_{g}"] for s in TEST for p in PINS])) for g in ("rows", "tail", "menu")}
causg = {g: float(np.mean([REC[(s, p)]["oE"]["T3"] - REC[(s, p)]["arms"][a]["T3"] for s in TEST for p in PINS]))
         for g, a in (("rows", "E_RES46ROWS"), ("tail", "E_RES46TAIL"), ("menu", "E_RES46MENU"))}
gl_ = max(lensg_signed, key=lambda g: abs(lensg_signed[g])); gc_ = max(causg, key=lambda g: abs(causg[g]))
res["J7"] = dict(lens_mean_signed=lensg_signed, causal_mean=causg, lens_top=gl_, causal_top=gc_,
                 verdict=(("lens and patching put the writer's input at the same positions" if gl_ == gc_ else "not") if ok else "not read"))
# ---------------------------------------------------------------- J8 components
armsJ8 = COMPARMS + list(HEADARMS)
mc = {a: float(np.mean([LENS[(s, p)]["comp"][a]["T3"] for s in TEST for p in PINS])) for a in armsJ8}
ca = {a: float(np.mean([REC[(s, p)]["oE"]["T3"] - REC[(s, p)]["arms"][a]["T3"] for s in TEST for p in PINS])) for a in armsJ8}
rr = float(np.corrcoef([mc[a] for a in armsJ8], [ca[a] for a in armsJ8])[0, 1])
per_run = float(np.corrcoef([LENS[(s, p)]["comp"][a]["T3"] for a in armsJ8 for s in TEST for p in PINS],
                            [REC[(s, p)]["oE"]["T3"] - REC[(s, p)]["arms"][a]["T3"] for a in armsJ8 for s in TEST for p in PINS])[0, 1])
res["J8"] = dict(lens_mean=mc, causal_mean=ca, corr_arm_means=rr, corr_runs=per_run,
                 sign_agree={a: bool(np.sign(mc[a]) == np.sign(ca[a])) for a in armsJ8},
                 verdict=(("the lens attributes the writer to the same components as causal swaps" if rr >= .70 else "not") if ok else "not read"))
# ---------------------------------------------------------------- J9 trajectory (descriptive)
traj = {}
for p in PINS:
    tp = {}
    for q in SAVE:
        D = np.mean(TRAJ[p][f"{q[0]}{q[1]}_D"], 0); A_ = np.mean(TRAJ[p][f"{q[0]}{q[1]}_A"], 0)
        tp[f"{q[0]}{q[1]}"] = dict(D=D.tolist(), A=A_.tolist(), D_pred_minus_anchor=float(D[p - 2] - D[p - 1]), A_argmax=int(np.argmax(A_)) + 1,
                                  A_pred_minus_anchor=float(A_[p - 2] - A_[p - 1]))
    traj[f"p{p}"] = tp
res["J9"] = traj
# ---------------------------------------------------------------- descriptive
res["descriptive"] = dict(
    answers={f"p{p}": {nm: {str(x): float(np.mean([REC[(s, p)]["arms"][nm]["x"] == x if nm != "E" else REC[(s, p)]["xE"] == x for s in range(S)]))
                             for x in range(1, N + 1)} for nm in ["E"] + ARMS} for p in PINS},
    writer_T3_mean={f"p{p}": float(np.mean([REC[(s, p)]["W"]["T3"] for s in range(S)])) for p in PINS},
    writer_T1_mean={f"p{p}": float(np.mean([REC[(s, p)]["W"]["T1"] for s in range(S)])) for p in PINS},
    writer_T3_sd={f"p{p}": float(np.std([REC[(s, p)]["W"]["T3"] for s in range(S)])) for p in PINS},
    arm_effect_T3={a: float(np.mean([REC[(s, p)]["oE"]["T3"] - REC[(s, p)]["arms"][a]["T3"] for s in range(S) for p in PINS])) for a in ARMS})
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"]["all"], J1=[res["J1"]["verdict"], J1["T3"]["R2"]["est"], J1["T3"]["Corr"]["est"]],
                      J1z=[res["J1z"]["verdict"], J1["T1"]["R2"]["est"], J1["T1"]["Corr"]["est"]], J2=[res["J2"]["verdict"], j2b["est"]],
                      J3=[res["J3"]["verdict"], t3r2["est"]], J4=[res["J4"]["verdict"], j4["est"]], J5=res["J5"]["verdict"], J6=[res["J6"]["verdict"], j6["est"]],
                      J7=res["J7"]["verdict"], J8=[res["J8"]["verdict"], rr]), indent=1))
print("->", OUT)
