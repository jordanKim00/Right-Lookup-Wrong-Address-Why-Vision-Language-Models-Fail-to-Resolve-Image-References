"""S848 analysis (rules fixed in knowledge/DECISIONS.md before any S848 data).  Test A part 1: is goal attainment (pinned early position p equal
to the stated target t = N_s - 3) represented in the read rows' residual stream around the band 46-49?  8 pictures, k = 4, N_s in {6..9},
p in {3..6} (4 x 4 grid; the diagonal is p = t); E_EARLY_p residuals at the inputs of layers 44, 46, 47, 48, 49, 50, 52 (ten read rows).
I0 (per N_s, all four must pass): t1 >= .95; QNOOP_E = END in >= .90 of scenes with the 95th percentile of the largest menu-logit change <= .5;
START(4) -> 4 >= .90; donors START(p) -> p >= .90 pooled (>= .80 each); image keys identical (0.0).
REP (primary sites: inputs of layer 46 = band input and layer 50 = band output, boundary B4 row and answer row - four sites; Bonferroni):
  x = residual of one row, centred within each scene over its 16 cells.  Two folds by scene (even / odd index).  On the training fold: PCA
  (K = 50) of x; cell means in PCA space, double-centred over the 4 x 4 grid (removes every additive p and N_s effect); for each diagonal cell c
  the 'match direction' m_c = mean of the double-centred means of the OTHER three diagonal cells (leave-one-diagonal-out).  On the held-out fold,
  per scene: s = x . V m_c for its 16 cells and the held-out interaction I = s[c] - mean(row of c) - mean(column of c) + mean(all), averaged over
  the four diagonal cells (any additive p / N_s effect cancels exactly, so E[I] = 0 without a match signal).  Z = mean over the 40 scenes;
  d' = Z / within-cell SD of s.  Scene bootstrap 4,000 (seed 0), 98.75% interval (four sites).
  Instrument checks per site (all must pass for a verdict at that site): DECODE - nearest-centroid decoding of N_s (4-way, fold-held-out, same
  PCA space) >= .90 (p decoding is reported, not required - amendment 10-17 before data: the read rows carry the content of the pinned picture,
  which differs between scenes, so p need not be shared linearly across scenes; the smoke showed .38-.75 with two scenes); SENS - planted diagonal
  signal of d' = 1.0 along 20 random directions within the top-50 PCs is detected (interval above 0) in >= 16 / 20.
  Verdict: Z > 0 with the 98.75% interval above 0 at any primary site passing its checks -> 'goal attainment is represented at the band'
  (named sites); no such site but every primary site passes its checks -> 'no goal-attainment signal detected at the band' with the bound (the
  largest upper end of d' over the primary sites); otherwise 'uninterpretable (instrument)'.
Descriptive: the same statistic at every residual layer x row and with K = 20 / 200; the interaction profile along m by stated distance d = t - p
(binary 'reached' vs graded closeness); within the four diagonal cells, the projection of runs that stay at p vs runs that move (point-biserial r);
test-B development quantities (late query, positional-score and answer predictions of PRED_F / PRED_E against BANDF / BANDE; not verdicts).
Usage: python exp/s848_analyze.py [TAG]  ->  results/r3/s848_analysis.json"""
from __future__ import annotations

import json, sys
import numpy as np

V = "/home/storage2/ai26269345/vlm"
B = 4000
PS = (3, 4, 5, 6); NSS = (6, 7, 8, 9); K0 = 50
PRIMARY = [(46, 5), (46, 9), (50, 5), (50, 9)]


def rep_stat(X, sis, K, rng, plant=None, boot=True, lo=.625, hi=99.375):
    """X: dict si -> [4 (p), 4 (N_s), E] scene-centred features.  Returns Z, interval, d', per-scene I, decode accuracies."""
    folds = [[s for s in sis if s % 2 == 0], [s for s in sis if s % 2 == 1]]
    Iscene = {}; sw = []; dec_p = []; dec_n = []
    for f in (0, 1):
        tr, te = folds[1 - f], folds[f]
        A = np.stack([X[s] for s in tr])                                        # [n, 4, 4, E]
        T = np.stack([X[s] for s in te])
        if plant is not None:
            A = A + plant[None]; T = T + plant[None]
        flat = A.reshape(-1, A.shape[-1]); mu = flat.mean(0)
        U, S_, Vt = np.linalg.svd(flat - mu, full_matrices=False)
        Vk = Vt[:K].T                                                           # [E, K]
        Ap = (A - mu) @ Vk; Tp = (T - mu) @ Vk                                  # [n, 4, 4, K]
        M = Ap.mean(0)
        Y = M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + M.mean((0, 1), keepdims=True)
        # decoding (nearest centroid in PCA space, held-out fold)
        cp = Ap.mean((0, 2)); cn = Ap.mean((0, 1))                              # [4, K] centroids of p and of N_s
        for t in Tp:
            for i in range(4):
                for j in range(4):
                    dec_p.append(int(np.argmin(((cp - t[i, j]) ** 2).sum(1)) == i)); dec_n.append(int(np.argmin(((cn - t[i, j]) ** 2).sum(1)) == j))
        for si, t in zip(te, Tp):
            vals = []
            for c in range(4):
                m = np.mean([Y[d, d] for d in range(4) if d != c], 0); m = m / (np.linalg.norm(m) + 1e-12)
                s_ = t @ m                                                      # [4, 4]
                vals.append(s_[c, c] - s_[c, :].mean() - s_[:, c].mean() + s_.mean())
                sw.append(s_)
            Iscene[si] = float(np.mean(vals))
    Is = np.array([Iscene[s] for s in sis])
    sd_w = float(np.mean(np.std(np.stack(sw), 0)))
    Z = float(Is.mean())
    ci = None
    if boot:
        bs = [Is[rng.integers(0, len(Is), len(Is))].mean() for _ in range(B)]
        ci = [float(np.percentile(bs, lo)), float(np.percentile(bs, hi))]
    return dict(Z=Z, ci=ci, dprime=Z / sd_w if sd_w > 0 else float("nan"), dprime_ci=([c / sd_w for c in ci] if ci else None), sd_within=sd_w,
                decode_p=float(np.mean(dec_p)), decode_Ns=float(np.mean(dec_n)), per_scene=Iscene)


def main(tag="s848_27b"):
    d = json.load(open(f"{V}/results/r3/s848_{tag}_stop.json")); R = d["rows"]
    Z_ = np.load(f"{V}/results/r3/s848_{tag}_stop.npz")
    rng = np.random.default_rng(0)
    by = {ns: {r["si"]: r for r in R if r["Ns"] == ns} for ns in NSS}
    sis = sorted(set.intersection(*[set(x) for x in by.values()]))
    RES = d["res_layers"]; late = [str(x) for x in d["late"]]
    out = dict(tag=tag, t1=d["t1_menu"], key_rel_diff_max=d["key_rel_diff_max"], n=len(sis), I0={})
    ok_all = True
    for ns in NSS:
        rows = [by[ns][s] for s in sis]
        ag = float(np.mean([r["land"]["QNOOP_E"] == r["land"]["END"] for r in rows]))
        p95 = float(np.percentile([max(abs(x - y) for x, y in zip(r["mlog"]["QNOOP_E"], r["mlog"]["END"])) for r in rows], 95))
        st = float(np.mean([r["land"]["START"] == 4 for r in rows]))
        don = {p: float(np.mean([r["land"][f"START_p{p}"] == p for r in rows])) for p in PS}
        ok = bool(d["t1_menu"] >= .95 and ag >= .90 and p95 <= .5 and st >= .90 and np.mean(list(don.values())) >= .90 and min(don.values()) >= .80
                  and d["key_rel_diff_max"] == 0.0)
        ok_all &= ok
        out["I0"][str(ns)] = dict(qnoop=ag, p95=p95, start_acc=st, donors=don, ok=ok)
    out["I0"]["all"] = bool(ok_all)
    def feats(L, row):
        li = RES.index(L); X = {}
        for s in sis:
            a = np.stack([np.stack([Z_[f"Ns{ns}_s{s}_E_EARLY_p{p}_res"][li, row].astype(np.float64) for ns in NSS]) for p in PS])   # [4, 4, E]
            X[s] = a - a.mean((0, 1), keepdims=True)
        return X
    rep = {}
    for L in RES:
        for row in (5, 9):
            X = feats(L, row)
            prim = (L, row) in PRIMARY
            r0 = rep_stat(X, sis, K0, rng)
            res_ = dict(K50={k: v for k, v in r0.items() if k != "per_scene"})
            if prim:
                # planted sensitivity: d' = 1.0 along random directions in the top-50 PCs of the site's data
                allx = np.stack([X[s] for s in sis]).reshape(-1, X[sis[0]].shape[-1])
                U, S_, Vt = np.linalg.svd(allx - allx.mean(0), full_matrices=False)
                det = 0
                for t in range(20):
                    ke = min(K0, Vt.shape[0]); u = Vt[:ke].T @ rng.standard_normal(ke); u /= np.linalg.norm(u)
                    proj = np.stack([X[s] for s in sis]) @ u                    # [n, 4, 4]
                    sd_u = float(np.mean(np.std(proj, 0)))
                    plant = np.zeros((4, 4, allx.shape[1]));
                    for c in range(4):
                        plant[c, c] = 1.0 * sd_u * u
                    rp = rep_stat(X, sis, K0, rng, plant=plant)
                    det += int(rp["ci"][0] > 0)
                res_["SENS_detected_of_20"] = det
                res_["checks_ok"] = bool(r0["decode_Ns"] >= .90 and det >= 16)     # amendment 10-17 before data: p decoding reported, not required
                res_["per_scene"] = r0["per_scene"]
                for K in (20, 200):
                    rk = rep_stat(X, sis, K, rng)
                    res_[f"K{K}"] = {k: v for k, v in rk.items() if k != "per_scene"}
            rep[f"L{L}_row{row}"] = res_
    out["REP_sites"] = rep
    prim_keys = [f"L{L}_row{row}" for L, row in PRIMARY]
    hits = [k for k in prim_keys if rep[k]["checks_ok"] and rep[k]["K50"]["ci"][0] > 0]
    allok = all(rep[k]["checks_ok"] for k in prim_keys)
    if not ok_all:
        out["REP"] = "no valid run"
    elif hits:
        out["REP"] = dict(verdict="goal attainment is represented at the band", sites=hits)
    elif allok:
        out["REP"] = dict(verdict="no goal-attainment signal detected at the band",
                          bound_dprime=max(rep[k]["K50"]["dprime_ci"][1] for k in prim_keys))
    else:
        out["REP"] = dict(verdict="uninterpretable (instrument)", failed=[k for k in prim_keys if not rep[k]["checks_ok"]])
    # descriptive: interaction profile along the pooled match direction by d = t - p, and stay vs move on the diagonal
    prof = {}; stay_r = {}
    for L, row in PRIMARY:
        X = feats(L, row)
        A = np.stack([X[s] for s in sis]); flat = A.reshape(-1, A.shape[-1]); mu = flat.mean(0)
        U, S_, Vt = np.linalg.svd(flat - mu, full_matrices=False); Vk = Vt[:K0].T
        Ap = (A - mu) @ Vk; M = Ap.mean(0)
        Y = M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + M.mean((0, 1), keepdims=True)
        m = np.mean([Y[c, c] for c in range(4)], 0); m /= np.linalg.norm(m) + 1e-12
        s_ = Ap @ m                                                             # [n, 4, 4]
        Ys = s_.mean(0); Ys = Ys - Ys.mean(1, keepdims=True) - Ys.mean(0, keepdims=True) + Ys.mean()
        prof[f"L{L}_row{row}"] = {str(dd): float(np.mean([Ys[i, j] for i in range(4) for j in range(4) if (j + 6 - 3) - PS[i] == dd])) for dd in range(-3, 4)}
        xs, ys = [], []
        for si_, s in enumerate(sis):
            for c in range(4):
                p = PS[c]; ns = NSS[c]
                xs.append(s_[si_, c, c]); ys.append(int(by[ns][s]["land"][f"E_EARLY_p{p}"] == p))
        stay_r[f"L{L}_row{row}"] = float(np.corrcoef(xs, ys)[0, 1]) if np.std(ys) > 0 else float("nan")
    out["profile_by_d"] = prof; out["diag_stay_r"] = stay_r
    # test-B development quantities (descriptive)
    dev = {}
    PW = PS
    for kind, act, base, pred in (("F", "BANDF", "E_EARLY", "PRED_F"), ("E", "BANDE", "S_EARLY", "PRED_E")):
        cos, r2 = {L: [] for L in late}, {L: [] for L in late}
        zr = {L: ([], []) for L in late}
        for ns in NSS:
            for s in sis:
                for p in PW:
                    qa = Z_[f"Ns{ns}_s{s}_{act}_p{p}_q"].astype(np.float64); qb = Z_[f"Ns{ns}_s{s}_{base}_p{p}_q"].astype(np.float64)
                    qp = Z_[f"Ns{ns}_s{s}_{pred}_p{p}_qd"].astype(np.float64)
                    for li, L in enumerate(late):
                        da = (qa[li] - qb[li]).ravel(); dp = (qp[li] - qb[li]).ravel()
                        cos[L].append(float(da @ dp / (np.linalg.norm(da) * np.linalg.norm(dp) + 1e-12)))
                        r2[L].append(float(1 - ((da - dp) ** 2).sum() / ((da ** 2).sum() + 1e-12)))
                        za = np.array(by[ns][s]["qk"][f"{act}_p{p}"][L]); zb = np.array(by[ns][s]["qk"][f"{base}_p{p}"][L])
                        zp = np.array(by[ns][s]["qk"][f"{pred}_p{p}"][L])
                        for row in (5, 9):
                            zr[L][0].extend((za[row] - zb[row]).tolist()); zr[L][1].extend((zp[row] - zb[row]).tolist())
        dev[kind] = {L: dict(q_cos_median=float(np.median(cos[L])), q_r2_median=float(np.median(r2[L])),
                             z_change_r=float(np.corrcoef(zr[L][0], zr[L][1])[0, 1])) for L in late}
    out["B_dev"] = dev
    fn = f"{V}/results/r3/s848_analysis.json" if tag == "s848_27b" else f"{V}/results/r3/s848_analysis_{tag}.json"
    json.dump(out, open(fn, "w"), indent=1, default=float)
    print(json.dumps(out["I0"])[:600])
    print("REP", json.dumps(out["REP"]))
    for k, v in rep.items():
        print(k, json.dumps({kk: (vv if not isinstance(vv, dict) else {a: b for a, b in vv.items() if a in ("Z", "ci", "dprime", "dprime_ci", "decode_p", "decode_Ns")})
                             for kk, vv in v.items() if kk != "per_scene"})[:600])
    print("profile_by_d", json.dumps(prof)); print("diag_stay_r", json.dumps(stay_r)); print("B_dev", json.dumps(dev))


if __name__ == "__main__":
    main(*sys.argv[1:])
