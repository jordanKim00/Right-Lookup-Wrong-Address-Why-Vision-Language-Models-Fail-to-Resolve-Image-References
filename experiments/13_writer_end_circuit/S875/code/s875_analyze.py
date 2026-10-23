"""S875 analysis (descriptive; the candidate rules at the end are proposals to be registered in knowledge/DECISIONS.md before the data are
read - nothing printed here is a registered verdict).  Inputs results/r3/s875_<tag>_bridge.json and results/r3/s875_<tag>_scenes/scene<si>.npz.
Notation per run: shown labels i = 1..N (picture order), x = the chosen picture (runs whose answer is the menu word of a picture that is not
shown are excluded, counted), r = the runner-up (largest model logit among the other shown labels), L_i = the model's menu logit of label i
(max over the two spacing forms), Lc_i = L_i - mean over the shown labels (the nine-word menu mean is reported in (e) as well),
D[L, h, c, i] = the direct logit attribution of head h of late layer L through key bucket c (picture block b, other text, menu word j) at the
answer row, already divided by the run's final rms (logit units), onto label i's kept spacing form.
(a) copy    C[L, h] = mean over runs of sum_b (D[L, h, b, b] - mean_{i != b} D[L, h, b, i]) (b, i over the shown pictures); gain G[L, h] =
            C summed over runs / the answer-row attention mass on the shown pictures summed over runs (copy per unit of mass); the same for
            the menu-word buckets (C_menu: attending label j's menu word promotes label j).  Heads ranked by C (scene-bootstrap intervals).
(b) H1      s_i = sum_{L, h} sum_b D[L, h, b, i] (image-routed bridge); s^all = every bucket (the late heads' whole direct write); s^menu, and
            s^img+menu.  Per run Pearson r(s_c, Lc) over the shown labels (s_c centred like Lc); pooled R2_id = 1 - sum (Lc - s_c)^2 /
            sum Lc^2 (no fit: D is in logit units); R2_fit = the same with one pooled slope beta through the origin; pooled Pearson over every
            (run, label).  Accuracy of argmax s, argmax s^all, argmax s^img+menu, S869R's R1 (argmax over pictures of the matcher-head mean
            late block score: layers 51-63, rows B1 / B4 / answer) and of the answer-row attention mass (summed over late layers and heads)
            against x; the paired difference bridge - R1; bridge-R1 agreement; per late layer: only-this-layer and leave-this-layer-out.
(c) H2      m = L_x - L_r against m_s = s_x - s_r (and s^all): Pearson (bootstrap), Spearman, R2_id, ratio of means; the bridge's own
            top-second gap s_(1) - s_(2) against m.
(d) H3      per run c[L, h] = sum_b D[L, h, b, x] - mean_i sum_b D[L, h, b, i] (sum over (L, h) = s_c at x); signed shares of the top 1 / 3 / 10
            pairs (runs with s_c[x] > 0), |c| concentration of the top 1 / 3 / 10, participation ratio (sum |c|)^2 / sum c^2; a fixed global
            top-k set (pairs ranked by mean c over runs); per-layer shares (ratios of sums).
(e) parts   Lc_x decomposed exactly at the run's rms: the residual entering layer 51, the late FA attention outputs (split by D into picture
            blocks / menu words / other text), the Gated DeltaNet mixers, the MLPs 51-63, and the rounding residual (linearised - sum of
            parts; and model logit - linearised); shares = ratios of sums over runs (scene bootstrap) and per-run medians.
I0 (instrument; read first): t1 >= .95; START(1) / START(N) answer their picture in >= .90; the float32 recompute matches the module's own
output (median relative error <= .02 over runs, layers and rows); the bucket split sums to the recompute (max <= 1e-3); the linearised
logits match the model's logits (median |diff| <= .10) and the parts sum to the final residual's DLA (median <= .10); the recomposed FA
output's DLA matches the module output's (median <= .05); R1 on the END runs reproduces S869R (accuracy >= .60; S869R .71).
Scene bootstrap 4,000 (seed 875) for headline numbers (END runs primary; START runs and all runs reported); per (N, cell) point estimates.
Usage: python3 s875_analyze.py TAG [--indir /vlm/results/r3]"""
from __future__ import annotations

import argparse, collections, json
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("tag"); ap.add_argument("--indir", default="/vlm/results/r3"); ap.add_argument("--nb", type=int, default=4000)
A = ap.parse_args()
J = json.load(open(f"{A.indir}/s875_{A.tag}_bridge.json"))
NPD = f"{A.indir}/s875_{A.tag}_scenes"
OUT = f"{A.indir}/s875_analysis_{A.tag}.json"
rng = np.random.default_rng(875)
np.seterr(divide="ignore", invalid="ignore")                               # empty or zero-sum resamples give nan, dropped by nanpercentile
NB = A.nb
LATE = J["late"]; nL = len(LATE); NH = J["nh"]; PARTS = J["parts"]; LM = J["late_mean"]
IFA = [PARTS.index(f"fa{l}") for l in LATE]; IDN = [i for i, p in enumerate(PARTS) if p.startswith("dn")]
IMLP = [i for i, p in enumerate(PARTS) if p.startswith("mlp")]
nan = float("nan")


def cen(u):
    return u - u.mean()


def pearson(x, y):
    x, y = cen(np.asarray(x, float)), cen(np.asarray(y, float))
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else nan


def rankdata(x):
    o = np.argsort(x, kind="stable"); r = np.empty(len(x)); r[o] = np.arange(len(x)); return r


def nanmed(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else nan


def derive(n, D, M, Zs, logit, parts, final, farec, hchk, land):
    """per-run quantities (D [nL, H, nk, 9], M [nL, 3, H, nk], Zs [nL, 3, H, n], logit / final [9], parts [P, 9], farec [nL, 9])."""
    r = dict(n=n, x=land)
    Ds = D[..., :n]
    img = Ds[:, :, :n, :]                                                  # [nL, H, b, i]
    oth = Ds[:, :, n, :]                                                   # [nL, H, i]
    men = Ds[:, :, n + 1:, :]                                              # [nL, H, j (menu word of label j), i]
    s_img, s_oth, s_men = img.sum((0, 1, 2)), oth.sum((0, 1)), men.sum((0, 1, 2))
    S_ = dict(img=s_img, all=s_img + s_oth + s_men, menu=s_men, imgmenu=s_img + s_men)
    L = logit[:n].astype(float); Lc = cen(L)
    ar = np.arange(n)
    # (a) copy through the picture blocks and through the menu words of the shown labels
    dg = img[:, :, ar, ar]
    r["copy"] = (dg - (img.sum(-1) - dg) / (n - 1)).sum(-1)               # [nL, H] summed over the shown blocks
    r["copyb"] = r["copy"] / n                                             # the same as a mean over blocks (the per-(run, block) form)
    r["mimg"] = M[:, 2, :, :n].sum(-1)                                     # [nL, H] answer-row mass on the shown pictures
    ms = men[:, :, :n, :]; mdg = ms[:, :, ar, ar]
    r["mcopy"] = (mdg - (ms.sum(-1) - mdg) / (n - 1)).sum(-1)
    r["mmen"] = M[:, 2, :, n + 1:n + 1 + n].sum(-1)
    # (b) bridge
    r["syy"] = float((Lc * Lc).sum())
    for nm, sv in S_.items():
        c_ = cen(sv)
        r[f"sxy_{nm}"] = float((c_ * Lc).sum()); r[f"sxx_{nm}"] = float((c_ * c_).sum()); r[f"sse_{nm}"] = float(((Lc - c_) ** 2).sum())
        r[f"r_{nm}"] = pearson(c_, Lc); r[f"pred_{nm}"] = int(np.argmax(sv)) + 1
    zm = np.mean([Zs[li][:, LM[str(L_)]].mean(1).mean(0) for li, L_ in enumerate(LATE)], 0)
    r["pred_R1"] = int(np.argmax(zm)) + 1
    r["pred_mass"] = int(np.argmax(M[:, 2, :, :n].sum((0, 1)))) + 1
    lay_img = img.sum((1, 2))                                              # [nL, i] image-routed write of each late layer
    r["pred_only"] = [int(np.argmax(lay_img[li])) + 1 for li in range(nL)]
    r["pred_loo"] = [int(np.argmax(s_img - lay_img[li])) + 1 for li in range(nL)]
    r["sse_loo"] = [float(((Lc - cen(s_img - lay_img[li])) ** 2).sum()) for li in range(nL)]
    # checks
    r["chk_attn"] = hchk[:, :3]; r["chk_split"] = float(hchk[:, 3].max())
    r["chk_lin"] = float(np.abs(final - logit).max()); r["chk_close"] = float(np.abs(parts.sum(0) - final).max())
    r["chk_farec"] = float(np.abs(farec - parts[IFA]).max())
    r["valid"] = 1 <= land <= n
    if not r["valid"]:
        return r
    x = land - 1
    rr = [i for i in np.argsort(-L, kind="stable") if i != x][0]
    # (c) margin
    r["m_act"] = float(L[x] - L[rr]); r["m_img"] = float(s_img[x] - s_img[rr]); r["m_all"] = float(S_["all"][x] - S_["all"][rr])
    ss = np.sort(s_img)[::-1]; r["gap_img"] = float(ss[0] - ss[1])
    # (d) routes
    c = img[..., x].sum(2) - img.sum(2).mean(-1)                           # [nL, H]
    tot = float(c.sum()); cf = np.sort(c.ravel())[::-1]; ab = np.sort(np.abs(c.ravel()))[::-1]
    r["c"] = c; r["tot"] = tot
    r["top"] = {k: (float(cf[:k].sum() / tot) if tot > 0 else nan) for k in (1, 3, 10)}
    r["conc"] = {k: (float(ab[:k].sum() / ab.sum()) if ab.sum() > 0 else nan) for k in (1, 3, 10)}
    r["pr"] = float(ab.sum() ** 2 / (c * c).sum()) if (c * c).sum() > 0 else nan
    # (e) exact decomposition of the chosen label's centred logit
    P = parts[:, :n]; pcx = (P - P.mean(-1, keepdims=True))[:, x]; fc = cen(final[:n].astype(float))[x]
    r["e"] = dict(act=float(Lc[x]), lin=float(fc), in0=float(pcx[0]), fa=float(pcx[IFA].sum()), dn=float(pcx[IDN].sum()),
                  mlp=float(pcx[IMLP].sum()), round=float(fc - pcx.sum()), logit_res=float(Lc[x] - fc),
                  fa_img=float(cen(s_img)[x]), fa_menu=float(cen(s_men)[x]), fa_oth=float(cen(s_oth)[x]))
    r["e_fa"] = pcx[IFA]; r["e_mlp"] = pcx[IMLP]
    P9 = parts - parts.mean(-1, keepdims=True); L9 = cen(logit.astype(float))
    r["e9"] = dict(act=float(L9[x]), in0=float(P9[0, x]), fa=float(P9[IFA, x].sum()), dn=float(P9[IDN, x].sum()), mlp=float(P9[IMLP, x].sum()))
    return r


RUNS = []
for rec in J["rows"]:
    Z = np.load(f"{NPD}/scene{rec['si']:02d}.npz")
    for n_s, lay in rec["layout"].items():
        n = int(n_s)
        ar_ = {k: Z[f"N{n}_{k}"] for k in ("dla", "mass", "z", "hchk", "farec", "parts", "final", "logit")}
        for j, nm in enumerate(lay["keys"]):
            form = "END" if nm.startswith("END") else "START"; q = int(nm[len(form):]); key = f"G{n}_{nm}"
            r = derive(n, ar_["dla"][j].astype(np.float64), ar_["mass"][j].astype(np.float64), ar_["z"][j].astype(np.float64),
                       ar_["logit"][j].astype(np.float64), ar_["parts"][j].astype(np.float64), ar_["final"][j].astype(np.float64),
                       ar_["farec"][j].astype(np.float64), ar_["hchk"][j].astype(np.float64), rec["land"][key])
            r.update(si=rec["si"], form=form, q=q, tgt=(n + 1 - q if form == "END" else q), t1=rec["t1"][key], key=key)
            RUNS.append(r)
R_ = len(RUNS)
res = dict(tag=A.tag, n_scenes=len(J["rows"]), n_runs=R_, complete=J.get("complete"), late=LATE, parts=PARTS)
arr = lambda f: np.array([f(r) for r in RUNS], dtype=float)
END = arr(lambda r: r["form"] == "END"); STA = 1 - END
VAL = arr(lambda r: r["valid"])
res["excluded_unshown"] = dict(END=int(((1 - VAL) * END).sum()), START=int(((1 - VAL) * STA).sum()))
# ---------------------------------------------------------------- scene bootstrap (weights: row 0 = the estimate, rows 1.. = resamples)
SC = np.array([r["si"] for r in RUNS]); US = np.unique(SC); S = len(US); IX = np.searchsorted(US, SC)
CNT = np.stack([np.bincount(rng.integers(0, S, S), minlength=S) for _ in range(NB)]).astype(float)
W = np.vstack([np.ones(R_), CNT[:, IX]])


def summ(vals):
    vals = np.asarray(vals, float); lo, hi = np.nanpercentile(vals[1:], [2.5, 97.5])
    return dict(est=float(vals[0]), lo=float(lo), hi=float(hi))


def z0(x, m):
    return np.where(m > 0, np.nan_to_num(np.asarray(x, float), nan=0.0), 0.0)


def wmean(x, m):
    return (W @ z0(x, m)) / (W @ m)


def wratio(num, den, m):
    return (W @ z0(num, m)) / (W @ z0(den, m))


def wpear(x, y, m):
    x, y = z0(x, m), z0(y, m); Wm = W * m; s0 = Wm.sum(1)
    mx, my = Wm @ x / s0, Wm @ y / s0
    cxy = Wm @ (x * y) / s0 - mx * my; vx = Wm @ (x * x) / s0 - mx ** 2; vy = Wm @ (y * y) / s0 - my ** 2
    return cxy / np.sqrt(vx * vy)


SUBS = dict(END=END, START=STA, ALL=np.ones(R_))
# ---------------------------------------------------------------- I0
attn = np.concatenate([r["chk_attn"].ravel() for r in RUNS])
sta_ok = arr(lambda r: r["x"] == r["tgt"])
I0 = dict(t1=float(np.mean([r["t1"] for r in RUNS])), start_acc=float(sta_ok[STA > 0].mean()) if STA.sum() else nan,
          attn_rel_median=float(np.median(attn)), attn_rel_p99=float(np.percentile(attn, 99)), attn_rel_max=float(attn.max()),
          split_max=float(max(r["chk_split"] for r in RUNS)), lin_median=float(np.median([r["chk_lin"] for r in RUNS])),
          lin_max=float(max(r["chk_lin"] for r in RUNS)), closure_median=float(np.median([r["chk_close"] for r in RUNS])),
          closure_max=float(max(r["chk_close"] for r in RUNS)), farec_median=float(np.median([r["chk_farec"] for r in RUNS])),
          farec_max=float(max(r["chk_farec"] for r in RUNS)))
okR1 = arr(lambda r: r["pred_R1"] == r["x"])
I0["R1_END_acc"] = float(okR1[(END * VAL) > 0].mean()) if (END * VAL).sum() else nan
I0["pass"] = dict(t1=I0["t1"] >= .95, start=I0["start_acc"] >= .90, recompute=I0["attn_rel_median"] <= .02, split=I0["split_max"] <= 1e-3,
                  linearised=I0["lin_median"] <= .10, closure=I0["closure_median"] <= .10, farec=I0["farec_median"] <= .05,
                  R1_alive=I0["R1_END_acc"] >= .60)
I0["all_pass"] = bool(all(I0["pass"].values()))
res["I0"] = I0
# ---------------------------------------------------------------- (a) copy heads
HN = [(L_, h) for L_ in LATE for h in range(NH)]
COPY = np.stack([r["copy"].ravel() for r in RUNS]); MIMG = np.stack([r["mimg"].ravel() for r in RUNS])
COPYB = np.stack([r["copyb"].ravel() for r in RUNS])
MCOPY = np.stack([r["mcopy"].ravel() for r in RUNS]); MMEN = np.stack([r["mmen"].ravel() for r in RUNS])
res["a_copy"] = {}
for sub, m in SUBS.items():
    if m.sum() == 0:
        continue
    Wm = W * m; den = Wm.sum(1)[:, None]
    C = (Wm @ COPY) / den; G = (Wm @ COPY) / (Wm @ MIMG); CM = (Wm @ MCOPY) / den; GM = (Wm @ MCOPY) / (Wm @ MMEN); CB = (Wm @ COPYB) / den
    rk = np.argsort(-C[0])
    pct = lambda T, i: [float(x) for x in np.nanpercentile(T[1:, i], [2.5, 97.5])]
    top = [dict(layer=HN[i][0], head=HN[i][1], in_R1_mean=HN[i][1] in LM[str(HN[i][0])], C=float(C[0, i]), C_ci=pct(C, i),
                C_per_block=float(CB[0, i]), gain=float(G[0, i]), mass_pictures=float((m @ MIMG[:, i]) / m.sum()), C_menu=float(CM[0, i]),
                C_menu_ci=pct(CM, i), gain_menu=float(GM[0, i]), mass_menu=float((m @ MMEN[:, i]) / m.sum())) for i in rk[:15]]
    tot = C.sum(1); totm = CM.sum(1)
    res["a_copy"][sub] = dict(top15=top, sum_all_heads=summ(tot), sum_all_heads_menu=summ(totm),
                              n_heads_C_lo_pos=int(sum(np.nanpercentile(C[1:, i], 2.5) > 0 for i in range(len(HN)))),
                              n_heads_C_hi_neg=int(sum(np.nanpercentile(C[1:, i], 97.5) < 0 for i in range(len(HN)))),
                              per_layer=dict(zip([str(l) for l in LATE], [summ(C[:, li * NH:(li + 1) * NH].sum(1)) for li in range(nL)])),
                              all_heads=dict(C=C[0].reshape(nL, NH).tolist(), C_per_block=CB[0].reshape(nL, NH).tolist(),
                                             gain=G[0].reshape(nL, NH).tolist(), C_menu=CM[0].reshape(nL, NH).tolist(),
                                             gain_menu=GM[0].reshape(nL, NH).tolist()))
# ---------------------------------------------------------------- (b) H1 bridge
SYY = arr(lambda r: r["syy"])
res["b_H1"] = {}
for sub, m0 in SUBS.items():
    m = m0 * VAL
    if m.sum() == 0:
        continue
    out = dict(n=int(m.sum()))
    for nm in ("img", "all", "menu", "imgmenu"):
        sxy, sxx, sse = arr(lambda r: r[f"sxy_{nm}"]), arr(lambda r: r[f"sxx_{nm}"]), arr(lambda r: r[f"sse_{nm}"])
        Sxy, Sxx, Syy, Sse = W @ z0(sxy, m), W @ z0(sxx, m), W @ z0(SYY, m), W @ z0(sse, m)
        beta = Sxy / Sxx
        out[nm] = dict(R2_id=summ(1 - Sse / Syy), R2_fit=summ(1 - (Syy - 2 * beta * Sxy + beta ** 2 * Sxx) / Syy), slope=summ(beta),
                       pooled_r=summ(Sxy / np.sqrt(Sxx * Syy)), mean_run_r=summ(wmean(arr(lambda r: r[f"r_{nm}"]), m)),
                       median_run_r=nanmed(arr(lambda r: r[f"r_{nm}"])[m > 0]),
                       acc=summ(wmean(arr(lambda r: r[f"pred_{nm}"] == r["x"]), m)))
    okI = arr(lambda r: r["pred_img"] == r["x"]); okM = arr(lambda r: r["pred_mass"] == r["x"])
    out["acc_R1"] = summ(wmean(okR1, m)); out["acc_mass"] = summ(wmean(okM, m))
    out["acc_bridge_minus_R1"] = summ(wmean(okI - okR1, m)); out["acc_bridge_minus_mass"] = summ(wmean(okI - okM, m))
    out["agree_bridge_R1"] = summ(wmean(arr(lambda r: r["pred_img"] == r["pred_R1"]), m))
    out["per_layer"] = {}
    for li, L_ in enumerate(LATE):
        sse_l = arr(lambda r: r["sse_loo"][li])
        out["per_layer"][str(L_)] = dict(acc_only=summ(wmean(arr(lambda r: r["pred_only"][li] == r["x"]), m)),
                                         acc_loo=summ(wmean(arr(lambda r: r["pred_loo"][li] == r["x"]), m)),
                                         R2_id_loo=summ(1 - (W @ z0(sse_l, m)) / (W @ z0(SYY, m))))
    res["b_H1"][sub] = out
# ---------------------------------------------------------------- (c) H2 margin
res["c_H2"] = {}
for sub, m0 in SUBS.items():
    m = m0 * VAL
    if m.sum() == 0:
        continue
    ma, mi, mA, gp = arr(lambda r: r.get("m_act", nan)), arr(lambda r: r.get("m_img", nan)), arr(lambda r: r.get("m_all", nan)), arr(lambda r: r.get("gap_img", nan))
    sel = m > 0
    out = dict(n=int(m.sum()), mean_margin=summ(wmean(ma, m)))
    sst = W @ z0(ma * ma, m) - (W @ z0(ma, m)) ** 2 / (W @ m)                # about each resample's own mean margin
    for nm, pr in (("img", mi), ("all", mA), ("gap_img", gp)):
        out[nm] = dict(pearson=summ(wpear(pr, ma, m)), spearman=pearson(rankdata(pr[sel]), rankdata(ma[sel])),
                       R2_id=summ(1 - (W @ z0((ma - pr) ** 2, m)) / sst), ratio_of_means=summ(wratio(pr, ma, m)))
    res["c_H2"][sub] = out
# ---------------------------------------------------------------- (d) H3 routes
res["d_H3"] = {}
for sub, m0 in SUBS.items():
    m = m0 * VAL
    if m.sum() == 0:
        continue
    sel = np.where(m > 0)[0]
    Cm = np.stack([RUNS[i]["c"].ravel() for i in sel]); TOT = np.array([RUNS[i]["tot"] for i in sel])
    pos = TOT > 0
    out = dict(n=int(len(sel)), n_tot_pos=int(pos.sum()), mean_tot=summ(wmean(arr(lambda r: r.get("tot", nan)), m)))
    for k in (1, 3, 10):
        tk = arr(lambda r: r["top"][k] if r["valid"] else nan); ck = arr(lambda r: r["conc"][k] if r["valid"] else nan)
        mp = m * np.isfinite(tk)
        out[f"top{k}_signed"] = dict(mean=summ(wmean(tk, mp)), median=nanmed(tk[mp > 0]))
        out[f"top{k}_abs_conc"] = dict(mean=summ(wmean(ck, m)), median=nanmed(ck[m > 0]))
    out["participation_ratio_median"] = nanmed([RUNS[i]["pr"] for i in sel])
    # a fixed global set: (L, h) pairs ranked by their mean contribution over the subset's runs
    gm = Cm.mean(0); rk = np.argsort(-gm)
    out["global_rank"] = [dict(layer=HN[i][0], head=HN[i][1], mean_c=float(gm[i])) for i in rk[:10]]
    full = np.zeros((R_, Cm.shape[1])); full[sel] = Cm
    for k in (1, 3, 10):
        part = full[:, rk[:k]].sum(1)
        out[f"global_top{k}_share"] = summ(wratio(part, arr(lambda r: r.get("tot", nan)), m))
    lay = np.zeros((R_, nL)); lay[sel] = Cm.reshape(len(sel), nL, NH).sum(2)
    out["layer_share"] = {str(L_): summ(wratio(lay[:, li], arr(lambda r: r.get("tot", nan)), m)) for li, L_ in enumerate(LATE)}
    out["n_layers_ge_15pct"] = int(sum(out["layer_share"][str(L_)]["est"] >= .15 for L_ in LATE))
    res["d_H3"][sub] = out
# ---------------------------------------------------------------- (e) parts of the chosen label's centred logit
res["e_parts"] = {}
for sub, m0 in SUBS.items():
    m = m0 * VAL
    if m.sum() == 0:
        continue
    act = arr(lambda r: r["e"]["act"] if r["valid"] else nan)
    out = dict(n=int(m.sum()), mean_act=summ(wmean(act, m)))
    for g in ("in0", "fa", "dn", "mlp", "round", "logit_res", "fa_img", "fa_menu", "fa_oth"):
        v_ = arr(lambda r: r["e"][g] if r["valid"] else nan)
        sh = v_ / act; ok_ = m * np.isfinite(sh) * (np.abs(act) > 1e-6)
        out[g] = dict(share_ratio_of_sums=summ(wratio(v_, act, m)), median_run_share=nanmed(sh[ok_ > 0]),
                      mean=summ(wmean(v_, m)))
    for li, L_ in enumerate(LATE):
        out[f"fa{L_}"] = summ(wratio(arr(lambda r: r["e_fa"][li] if r["valid"] else nan), act, m))
    for k_, l_ in enumerate([p for p in PARTS if p.startswith("mlp")]):
        out[l_] = summ(wratio(arr(lambda r: r["e_mlp"][k_] if r["valid"] else nan), act, m))
    act9 = arr(lambda r: r["e9"]["act"] if r["valid"] else nan)
    out["menu9"] = {g: summ(wratio(arr(lambda r: r["e9"][g] if r["valid"] else nan), act9, m)) for g in ("in0", "fa", "dn", "mlp")}
    res["e_parts"][sub] = out
# ---------------------------------------------------------------- per (N, cell) point estimates
cells = collections.defaultdict(list)
for i, r in enumerate(RUNS):
    cells[(r["n"], r["form"], r["q"])].append(i)
res["cells"] = []
for (n, form, q), ix in sorted(cells.items()):
    rs = [RUNS[i] for i in ix]; vr = [r for r in rs if r["valid"]]
    sse = sum(r["sse_img"] for r in vr); syy = sum(r["syy"] for r in vr)
    res["cells"].append(dict(N=n, form=form, q=q, target=rs[0]["tgt"], n=len(rs), n_valid=len(vr),
                             acc_model=float(np.mean([r["x"] == r["tgt"] for r in rs])),
                             acc_bridge=float(np.mean([r["pred_img"] == r["x"] for r in vr])) if vr else nan,
                             acc_R1=float(np.mean([r["pred_R1"] == r["x"] for r in vr])) if vr else nan,
                             acc_mass=float(np.mean([r["pred_mass"] == r["x"] for r in vr])) if vr else nan,
                             R2_id_img=float(1 - sse / syy) if syy > 0 else nan,
                             mean_run_r_img=float(np.nanmean([r["r_img"] for r in vr])) if vr else nan,
                             margin_r_img=pearson([r["m_img"] for r in vr], [r["m_act"] for r in vr]) if len(vr) > 2 else nan,
                             fa_share=float(sum(r["e"]["fa"] for r in vr) / sum(r["e"]["act"] for r in vr)) if vr else nan,
                             mlp_share=float(sum(r["e"]["mlp"] for r in vr) / sum(r["e"]["act"] for r in vr)) if vr else nan,
                             top3_signed_median=nanmed([r["top"][3] for r in vr])))
# ---------------------------------------------------------------- candidate rules (proposals; NOT registered verdicts)
cand = {}
if "END" in res["b_H1"]:
    b_, a_, c_, d_, e_ = res["b_H1"]["END"], res["a_copy"]["END"], res["c_H2"]["END"], res["d_H3"]["END"], res["e_parts"]["END"]
    t0 = a_["top15"][0]
    cand["CH1a_copy"] = dict(rule="the top late head's mean copy C >= .10 logit with its interval above 0, and the sum of C over all late heads "
                                  "has its interval above 0 -> 'late heads write the attended picture's own label (OV copy)'; else 'not'",
                             value=dict(top=t0, sum=a_["sum_all_heads"]),
                             verdict=("late heads write the attended picture's own label (OV copy)"
                                      if t0["C"] >= .10 and t0["C_ci"][0] > 0 and a_["sum_all_heads"]["lo"] > 0 else "not"))
    acc, r1 = b_["img"]["acc"], b_["acc_R1"]
    cand["CH1b_choice"] = dict(rule="accuracy of argmax s (image-routed bridge) >= .60 and >= R1 - .05 -> 'the QK-OV-logit bridge reproduces the "
                                    "choice'; else 'not'", value=dict(acc=acc, R1=r1, diff=b_["acc_bridge_minus_R1"]),
                               verdict=("the QK-OV-logit bridge reproduces the choice" if acc["est"] >= .60 and acc["est"] >= r1["est"] - .05 else "not"))
    r2, pr = b_["img"]["R2_id"], b_["img"]["pooled_r"]
    cand["CH1c_logits"] = dict(rule="pooled R2_id of s_c against Lc >= .30 with the lower bound > 0 -> 'the direct image-routed write accounts for "
                                    "the label-logit pattern'; else pooled Pearson >= .50 -> 'the bridge tracks the pattern at another scale "
                                    "(later processing rescales it)'; else 'not'", value=dict(R2_id=r2, pooled_r=pr, slope=b_["img"]["slope"]),
                               verdict=("the direct image-routed write accounts for the label-logit pattern" if r2["est"] >= .30 and r2["lo"] > 0
                                        else "the bridge tracks the pattern at another scale" if pr["est"] >= .50 else "not"))
    pm = c_["img"]["pearson"]
    cand["CH2_margin"] = dict(rule="Pearson(s_x - s_r, L_x - L_r) >= .50 with the lower bound > .30 -> 'the direct late writes predict the "
                                   "decision margin'; else 'not'", value=pm,
                              verdict=("the direct late writes predict the decision margin" if pm["est"] >= .50 and pm["lo"] > .30 else "not"))
    t1s, t3s = d_["top1_signed"]["median"], d_["top3_signed"]["median"]
    cand["CH3_routes"] = dict(rule="median top-1 signed share >= .50 -> 'single head'; else median top-3 signed share <= .50 and >= 2 late layers "
                                   "each carrying >= 15% of the chosen label's bridge score (ratio of sums) -> 'multi-route (several heads and "
                                   "layers add)'; else 'few heads'", value=dict(top1=t1s, top3=t3s, layer_share=d_["layer_share"]),
                              verdict=("single head" if t1s >= .50 else "multi-route (several heads and layers add)"
                                       if t3s <= .50 and d_["n_layers_ge_15pct"] >= 2 else "few heads"))
    fa, ml = e_["fa"]["share_ratio_of_sums"], e_["mlp"]["share_ratio_of_sums"]
    cand["CE_writer"] = dict(rule="share of the chosen label's centred logit written directly by the late FA attention outputs (ratio of sums) "
                                  ">= .50 with the lower bound > .30 -> 'late attention writes the answer directly'; the same for the MLPs 51-63 "
                                  "-> 'the late MLPs write it'; else 'mixed'", value=dict(fa=fa, mlp=ml, dn=e_["dn"]["share_ratio_of_sums"],
                                                                                          in0=e_["in0"]["share_ratio_of_sums"]),
                             verdict=("late attention writes the answer directly" if fa["est"] >= .50 and fa["lo"] > .30 else
                                      "the late MLPs write it" if ml["est"] >= .50 and ml["lo"] > .30 else "mixed"))
res["candidate_rules"] = dict(status="CANDIDATE - proposals for registration, not registered verdicts", rules=cand)
json.dump(res, open(OUT, "w"), indent=1)
fmt = lambda d: f"{d['est']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]"
print(json.dumps(dict(I0={k: (round(v, 4) if isinstance(v, float) else v) for k, v in I0.items()}), indent=1))
if "END" in res["b_H1"]:
    b_ = res["b_H1"]["END"]
    print(f"END n {b_['n']} · bridge acc {fmt(b_['img']['acc'])} · R1 {fmt(b_['acc_R1'])} · mass {fmt(b_['acc_mass'])} · "
          f"R2_id {fmt(b_['img']['R2_id'])} · pooled r {fmt(b_['img']['pooled_r'])} · margin r {fmt(res['c_H2']['END']['img']['pearson'])}")
    e_ = res["e_parts"]["END"]
    print("END parts (ratio of sums): " + " · ".join(f"{g} {fmt(e_[g]['share_ratio_of_sums'])}" for g in ("in0", "fa", "dn", "mlp", "fa_img", "fa_menu")))
    print("candidate (unregistered): " + json.dumps({k: v["verdict"] for k, v in res["candidate_rules"]["rules"].items()}, ensure_ascii=False))
print("->", OUT)
