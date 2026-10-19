"""S858 analysis (P3; every prediction, metric and threshold fixed in knowledge/DECISIONS.md before any S858 data).  Prior predictions from the
circuit fixed after S857 (pictures' count information reaches the late queries 51-63 through full-attention reads of the extra blocks in layers
35-50; the late queries carry it to the answer; END-relative inputs act on the late lookup as graded ramps, START's ordinal as a pointer), tested in
new N {9, 7}, k, p {2, 5} and new scenes.  Cells (N, k, p); targets jN = N + 1 - k (N pictures) and jm = N - 1 - k (N - 2 pictures), both among
the shared pictures 1..N-2.  Scene bootstrap 4,000 (seed 0); cells pooled.
  g    = l(w_jN) - l(w_jm)                       answer-side logit gap
  m51  = z_jN - z_jm at layer 51 (rows B1, B4, answer);  mL = the same, late mean
  zsh  = late block scores of the shared pictures (rows B1, B4, answer; mean over 51-63; centred over pictures 1..N-2)
  f(arm, x) = sum [x(arm) - x(Nm2)] / sum [x(Nn) - x(Nm2)]   (Nm2 = the same scene with pictures 1..N-2, same text)
I0 (all): t1 >= .95; shared-picture keys identical across the instrument configurations (0.0); LEAK: CAsw = CA (END and START; answers >= .99,
  p95 of the largest menu-logit difference <= .05); QSELF = Nn (same); INSTR: Nn vs Nstd answers >= .85 and no systematic shift of the gap
  (|mean g(Nn) - g(Nstd)| <= .15; set after the GPU smoke: .875 on 16 pairs at N 9 / 7, S857 .983 on 120); donors START(q) -> q >= .90 (Nn and
  Nm2, each N and q); PIN (layers 43 / 47, rows B1 / B4 maximum over the shared pictures = p) >= .90; POS: mean natural effect g(Nn) - g(Nm2)
  > 0 with lower bound > 0.
Level 1 - late queries (pre-RoPE, rows B1, B4, answer, every head):
  L1a  cut: cos(q_CA - q_Nn, q_Nm2 - q_Nn) at layer 51, median over scenes x cells x rows x heads >= .80 with the bootstrap lower bound > .70
       -> 'the late-query change under the cut is predicted (the cut makes the run an N-2 run)'; layers 55-63 reported.
  L1b  unperformed full-attention cut in 35-50: median cos(q_CF - q_Nn, q_Nm2 - q_Nn) at 51 >= .80 (lower > .70) and median projection share
       (q_CF - q_Nn).(q_Nm2 - q_Nn) / |q_Nm2 - q_Nn|^2 >= .70 (lower > .55) -> 'predicted'.
Level 2 - late picture scores:
  L2a  dz = zsh(CA) - zsh(Nn) against the prediction zsh(Nm2) - zsh(Nn): pooled r >= .80 (lower > .70) and explained share 1 - SSE / SS(dz) >= .60
       (lower > .45) -> 'predicted' (pure copy, positional geometry and a fixed one-step shift all predict no change from the cut: share 0).
  L2b  f(CF) on m51 and on g both <= .25 with upper < .40 -> 'the unperformed cut removes the count effect, as predicted'; both >= .60 ->
       'does not'; else 'partial'.
  L2c  shapes (R^2 of the scene-mean profile against a ramp i - mean and against a target shift e_new - e_old):
       visual zsh(Nn) - zsh(CA): ramp > shift(e_jN - e_jm) in >= 6 of 8 cells; END ordinal zsh(Nn, k_hi) - zsh(Nn, k_lo) at the same N, p:
       ramp > shift in >= 3 of 4; START ordinal zsh(S, k_hi) - zsh(S, k_lo): shift(e_khi - e_klo) > ramp in >= 3 of 4 -> all three:
       'END inputs act as ramps, the START ordinal as a pointer'; the first two both fail -> 'not ramps'; else 'partial'.
  L2d  writer: zsh(Nn) - zsh(BANDF): ramp > shift(e_jN - e_k) in >= 75% of the cells where the template is defined (jN != k; N 7 k 4 has
       jN = k) and a negative ramp slope in all 4 cells with k >= 5 -> 'the END write is a ramp, backward for larger k'; < 50% and not all
       negative -> 'not'; else 'partial' (k 4 slopes reported: the count-independent writer of S854-1 predicts about 0 at both N; a
       target-half rule would predict a positive slope at N 9).
Level 3 - answer logits and answers:
  L3a  predicted shares of the gap g: CA and CF 0 (= Nm2); RQ .80 and KQ .17 of the CA -> Nn difference (frozen from S857); per arm 'predicted'
       if |realized - predicted| <= .15 and the bootstrap interval of the realized share contains the prediction.
  L3b  scene-level answer predictions, smoothed (1 - .1) prediction + .1 uniform over the shown pictures and 'other': CA, CF -> the scene's Nm2
       answer; RQ -> .80 Nn answer + .20 CA answer; KQ -> .17 Nn + .83 CA.  Baselines (same smoothing): no intervention (Nn's answer; CA's for
       RQ), pure copy k, fixed one-step shift p - 1, actual target N + 1 - k, the N - 2 target N - 1 - k, uniform (positional geometry = no
       intervention for the cuts: positions are unchanged).  Per arm 'beats every baseline' if the per-answer log-likelihood gain is > 0 with
       lower bound > 0 against each; 'beats no-intervention only'; else 'does not'.
Descriptive (added 10-18 20:58 before any S858 result - user: compare effects with the instrument's scene-level NOOP variation): per scene and
cell |g(Nn) - g(Nstd)| (instrument on vs off) against |g(arm) - g(ref)| (CA, CF, KQ vs Nn; RQ vs CA): medians, the NOOP 95th percentile, the
share of (scene, cell) whose effect exceeds it, and answer-change rates.
Usage: python exp/s858_analyze.py [TAG]  ->  results/r3/s858_analysis.json"""
from __future__ import annotations

import json, os, sys
import numpy as np

V = "/home/storage2/ai26269345/vlm"
B = 4000
EPS = .1


def main(tag="s858_27b"):
    d = json.load(open(f"{V}/results/r3/s858_{tag}_pred.json"))
    Zp = f"{V}/results/r3/s858_{tag}_pred.npz"; Z = np.load(Zp) if os.path.exists(Zp) else None
    LATE = [str(l) for l in d["late"]]; L51 = LATE[0]; PINL = [str(l) for l in d["early"][-2:]]
    KN = {int(k_): tuple(v_) for k_, v_ in d["kn"].items()}; PS = tuple(d["ps"]); NS = tuple(d["ns"])
    by = {r["si"]: r for r in d["rows"]}; sis = sorted(by); n = len(sis)
    rng = np.random.default_rng(0); idx = rng.integers(0, n, (B, n))
    CELLS = [(N, k, p) for N in NS for k in KN[N] for p in PS]
    nm = lambda N, arm, k, p: f"N{N}_{arm}_k{k}p{p}"
    mld = lambda r, a1, a2: max(abs(x - y) for x, y in zip(r["mlog"][a1], r["mlog"][a2]))
    def same(pairs):
        return [float(np.mean([by[s]["land"][a1] == by[s]["land"][a2] for s in sis for a1, a2 in pairs])),
                float(np.percentile([mld(by[s], a1, a2) for s in sis for a1, a2 in pairs], 95))]
    out = dict(tag=tag, t1=d["t1_menu"], key_rel_diff_max=d["key_rel_diff_max"], n=n, n_runs=d.get("n_runs"), calls=d.get("calls"))
    i0 = dict(t1=d["t1_menu"] >= .95, keys=d["key_rel_diff_max"] == 0.0)
    lk = same([(nm(N, "CAsw", k, p), nm(N, "CA", k, p)) for N, k, p in CELLS] + [(f"N{N}_CAsw_START_q{q}", f"N{N}_CA_START_q{q}") for N in NS for q in PS])
    i0["leak"] = dict(v=lk, ok=bool(lk[0] >= .99 and lk[1] <= .05))
    qs_ = same([(nm(N, "QSELF", k, p), nm(N, "Nn", k, p)) for N, k, p in CELLS]); i0["qself"] = dict(v=qs_, ok=bool(qs_[0] >= .99 and qs_[1] <= .05))
    ins = same([(nm(N, "Nn", k, p), nm(N, "Nstd", k, p)) for N, k, p in CELLS])
    dn = {f"N{N}_{c}_q{q}": float(np.mean([by[s]["land"][f"N{N}_{c}_START_q{q}"] == q for s in sis])) for N in NS for c in ("Nn", "Nm2") for q in PS}
    i0["donors"] = dict(v=dn, ok=bool(min(dn.values()) >= .90))
    pin = float(np.mean([int(np.argmax(by[s]["qk"][nm(N, "Nn", k, p)][L][ri][:N - 2]) + 1 == p) for s in sis for N, k, p in CELLS for L in PINL for ri in (2, 5)]))
    i0["pin"] = dict(v=pin, ok=bool(pin >= .90))
    tg = lambda N, k: (N + 1 - k, N - 1 - k)
    def gfun(s, N, arm, k, p):
        jN, jm = tg(N, k); m = by[s]["mlog"][nm(N, arm, k, p)]; return float(m[jN - 1] - m[jm - 1])
    gsh = float(np.mean([gfun(s, N, "Nn", k, p) - gfun(s, N, "Nstd", k, p) for s in sis for N, k, p in CELLS]))
    i0["instr"] = dict(v=ins, gap_shift=gsh, ok=bool(ins[0] >= .85 and abs(gsh) <= .15))
    def mfun(layers):
        def f(s, N, arm, k, p):
            jN, jm = tg(N, k); q = by[s]["qk"][nm(N, arm, k, p)]
            return float(np.mean([q[L][ri][jN - 1] - q[L][ri][jm - 1] for L in layers for ri in (2, 5, 9)]))
        return f
    FUN = dict(g=gfun, m51=mfun([L51]), mL=mfun(LATE))
    def ratio(num, den):
        num, den = np.asarray(num), np.asarray(den); bs = num[idx].sum(1) / den[idx].sum(1)
        return [float(num.sum() / den.sum()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]]
    def share(arm, x, ref0="Nm2", ref1="Nn"):
        fun = FUN[x]; num, den = [], []
        for s in sis:
            a_ = b_ = 0.0
            for N, k, p in CELLS:
                r0 = fun(s, N, ref0, k, p); a_ += fun(s, N, arm, k, p) - r0; b_ += fun(s, N, ref1, k, p) - r0
            num.append(a_); den.append(b_)
        return ratio(num, den)
    G = np.asarray([np.mean([gfun(s, N, "Nn", k, p) - gfun(s, N, "Nm2", k, p) for N, k, p in CELLS]) for s in sis]); gb = G[idx].mean(1)
    i0["pos"] = dict(v=[float(G.mean()), [float(np.percentile(gb, 2.5)), float(np.percentile(gb, 97.5))]], ok=bool(np.percentile(gb, 2.5) > 0))
    ok_all = bool(i0["t1"] and i0["keys"] and all(v_["ok"] for v_ in i0.values() if isinstance(v_, dict)))
    out["I0"] = dict(i0, all=ok_all); nvt = "no valid test"
    out["shares"] = {arm: {x: share(arm, x) for x in FUN} for arm in ("CA", "CAsw", "CF", "RQ", "KQ", "QSELF", "BANDF", "Nstd")}
    # ---------- level 1
    L1 = {}
    if Z is not None:
        for L_i, L in enumerate(LATE):
            cosA, cosF, projF, per_scene = [], [], [], {}
            for s in sis:
                cs = []
                for N, k, p in CELLS:
                    qn = Z[f"s{s}_{nm(N, 'Nn', k, p)}_q"][L_i].astype(np.float64); qa = Z[f"s{s}_{nm(N, 'CA', k, p)}_q"][L_i].astype(np.float64)
                    qf = Z[f"s{s}_{nm(N, 'CF', k, p)}_q"][L_i].astype(np.float64); qm = Z[f"s{s}_{nm(N, 'Nm2', k, p)}_q"][L_i].astype(np.float64)
                    da, df, dp = qa - qn, qf - qn, qm - qn                                          # [3 rows, H, hd]
                    nrm = lambda x: np.linalg.norm(x, axis=-1)
                    c_a = (da * dp).sum(-1) / np.maximum(nrm(da) * nrm(dp), 1e-12); c_f = (df * dp).sum(-1) / np.maximum(nrm(df) * nrm(dp), 1e-12)
                    pr = (df * dp).sum(-1) / np.maximum((dp * dp).sum(-1), 1e-12)
                    cosA += c_a.ravel().tolist(); cosF += c_f.ravel().tolist(); projF += pr.ravel().tolist(); cs.append((c_a.ravel(), c_f.ravel(), pr.ravel()))
                per_scene[s] = [np.concatenate([x[j] for x in cs]) for j in range(3)]
            med = lambda j: [float(np.median(np.concatenate([per_scene[s][j] for s in sis]))),
                             [float(np.percentile([np.median(np.concatenate([per_scene[sis[i]][j] for i in row])) for row in idx[:1000]], q)) for q in (2.5, 97.5)]]
            L1[L] = dict(cos_CA=med(0), cos_CF=med(1), proj_CF=med(2))
    out["L1"] = L1
    if not ok_all or not L1:
        l1a = l1b = nvt
    else:
        c = L1[L51]
        l1a = "predicted" if c["cos_CA"][0] >= .80 and c["cos_CA"][1][0] > .70 else "not predicted"
        l1b = "predicted" if (c["cos_CF"][0] >= .80 and c["cos_CF"][1][0] > .70 and c["proj_CF"][0] >= .70 and c["proj_CF"][1][0] > .55) else "not predicted"
    out["L1a"] = l1a; out["L1b"] = l1b
    # ---------- level 2
    def zsh(s, N, arm, k, p):
        q = by[s]["qk"][nm(N, arm, k, p)]
        z = np.mean([[q[L][ri][:N - 2] for ri in (2, 5, 9)] for L in LATE], axis=(0, 1)); return z - z.mean()
    dz, dzh = [], []
    per = {s: ([], []) for s in sis}
    for s in sis:
        for N, k, p in CELLS:
            a_ = zsh(s, N, "CA", k, p) - zsh(s, N, "Nn", k, p); b_ = zsh(s, N, "Nm2", k, p) - zsh(s, N, "Nn", k, p)
            per[s][0].append(a_); per[s][1].append(b_)
    def l2stats(rows_):
        A = np.concatenate([np.concatenate(per[s][0]) for s in rows_]); Bp = np.concatenate([np.concatenate(per[s][1]) for s in rows_])
        r = float(np.corrcoef(A, Bp)[0, 1]); ex = float(1 - ((A - Bp) ** 2).sum() / (A ** 2).sum()); return r, ex
    r0, e0 = l2stats(sis); bsr = np.array([l2stats([sis[i] for i in row]) for row in idx[:1000]])
    out["L2a_stats"] = dict(r=[r0, [float(np.percentile(bsr[:, 0], 2.5)), float(np.percentile(bsr[:, 0], 97.5))]],
                            explained=[e0, [float(np.percentile(bsr[:, 1], 2.5)), float(np.percentile(bsr[:, 1], 97.5))]], alternatives_explained=0.0)
    l2a = nvt if not ok_all else ("predicted" if r0 >= .80 and np.percentile(bsr[:, 0], 2.5) > .70 and e0 >= .60 and np.percentile(bsr[:, 1], 2.5) > .45 else "not predicted")
    fm, fg = out["shares"]["CF"]["m51"], out["shares"]["CF"]["g"]
    l2b = (nvt if not ok_all else "the unperformed cut removes the count effect, as predicted" if (fm[0] <= .25 and fm[1][1] < .40 and fg[0] <= .25 and fg[1][1] < .40)
           else "does not" if (fm[0] >= .60 and fg[0] >= .60) else "partial")
    X6 = lambda m_: np.arange(1, m_ + 1) - (m_ + 1) / 2
    def r2s(M, t):
        M = M - M.mean(); v_ = M @ M; t = t - t.mean()
        if np.linalg.norm(t) == 0:
            return None                                                                   # undefined template (END target = literal k)
        t = t / np.linalg.norm(t); return float((M @ t) ** 2 / v_) if v_ > 0 else 0.0
    def onehot(m_, a_, b_):
        t = np.zeros(m_); t[a_ - 1] += 1; t[b_ - 1] -= 1; return t
    shp = dict(visual={}, end_ordinal={}, start_ordinal={}, writer={})
    for N, k, p in CELLS:
        jN, jm = tg(N, k); m_ = N - 2
        Mv = np.mean([zsh(s, N, "Nn", k, p) - zsh(s, N, "CA", k, p) for s in sis], 0)
        shp["visual"][f"N{N}k{k}p{p}"] = dict(r2_ramp=r2s(Mv, X6(m_)), r2_shift=r2s(Mv, onehot(m_, jN, jm)), slope=float(Mv @ X6(m_) / (X6(m_) @ X6(m_))))
        Mw = np.mean([zsh(s, N, "Nn", k, p) - zsh(s, N, "BANDF", k, p) for s in sis], 0)
        shp["writer"][f"N{N}k{k}p{p}"] = dict(r2_ramp=r2s(Mw, X6(m_)), r2_shift=r2s(Mw, onehot(m_, jN, min(k, m_))), slope=float(Mw @ X6(m_) / (X6(m_) @ X6(m_))),
                                             profile=[round(float(x), 3) for x in Mw])
    for N in NS:
        klo, khi = KN[N]; m_ = N - 2
        for p in PS:
            Me = np.mean([zsh(s, N, "Nn", khi, p) - zsh(s, N, "Nn", klo, p) for s in sis], 0)
            shp["end_ordinal"][f"N{N}p{p}"] = dict(r2_ramp=r2s(Me, X6(m_)), r2_shift=r2s(Me, onehot(m_, N + 1 - khi, N + 1 - klo)), slope=float(Me @ X6(m_) / (X6(m_) @ X6(m_))))
            Ms = np.mean([zsh(s, N, "S", khi, p) - zsh(s, N, "S", klo, p) for s in sis], 0)
            shp["start_ordinal"][f"N{N}p{p}"] = dict(r2_ramp=r2s(Ms, X6(m_)), r2_shift=r2s(Ms, onehot(m_, khi, klo)), slope=float(Ms @ X6(m_) / (X6(m_) @ X6(m_))))
    nv = sum(x["r2_ramp"] > x["r2_shift"] for x in shp["visual"].values()); ne = sum(x["r2_ramp"] > x["r2_shift"] for x in shp["end_ordinal"].values())
    ns_ = sum(x["r2_shift"] > x["r2_ramp"] for x in shp["start_ordinal"].values())
    l2c = (nvt if not ok_all else "END inputs act as ramps, the START ordinal as a pointer" if (nv >= 6 and ne >= 3 and ns_ >= 3)
           else "not ramps" if (nv < 6 and ne < 3) else "partial")
    wdef = [x for x in shp["writer"].values() if x["r2_shift"] is not None]
    nw = sum(x["r2_ramp"] > x["r2_shift"] for x in wdef); nwd = len(wdef)
    neg = all(shp["writer"][f"N{N}k{k}p{p}"]["slope"] < 0 for N, k, p in CELLS if k >= 5)
    l2d = nvt if not ok_all else ("the END write is a ramp, backward for larger k" if (nw >= .75 * nwd and neg) else "not" if (nw < .5 * nwd and not neg) else "partial")
    out["shapes"] = shp; out["shape_counts"] = dict(visual_ramp=nv, end_ordinal_ramp=ne, start_ordinal_pointer=ns_, writer_ramp=nw, writer_defined=nwd,
                                                    writer_negative_k5plus=bool(neg))
    out["L2a"] = l2a; out["L2b"] = l2b; out["L2c"] = l2c; out["L2d"] = l2d
    # ---------- level 3
    def share_g(arm, a0, a1):
        num, den = [], []
        for s in sis:
            x_ = y_ = 0.0
            for N, k, p in CELLS:
                g0 = gfun(s, N, a0, k, p); x_ += gfun(s, N, arm, k, p) - g0; y_ += gfun(s, N, a1, k, p) - g0
            num.append(x_); den.append(y_)
        return ratio(num, den)
    l3a = {}
    for arm, (a0, a1, pred) in {"CA": ("Nm2", "Nn", 0.0), "CF": ("Nm2", "Nn", 0.0), "RQ": ("CA", "Nn", .80), "KQ": ("CA", "Nn", .17)}.items():
        sh = share_g(arm, a0, a1)
        ok = abs(sh[0] - pred) <= .15 and sh[1][0] <= pred <= sh[1][1]
        l3a[arm] = dict(realized=sh, predicted=pred, verdict=(nvt if not ok_all else "predicted" if ok else "not predicted"))
    out["L3a"] = l3a
    def ll_of(s, N, k, p, arm, pred_dist):
        a = by[s]["land"][nm(N, arm, k, p)]; o = a if 1 <= a <= N else 0                              # 0 = other (a label not shown)
        return float(np.log(pred_dist[o]))
    def dist_from(points):
        """points: list of (weight, outcome) with outcome 0..N; smoothed."""
        def mk(N):
            pdist = np.full(N + 1, EPS / (N + 1))
            for w, o in points:
                pdist[o] += (1 - EPS) * w
            return pdist
        return mk
    def outc(s, N, arm, k, p):
        a = by[s]["land"][nm(N, arm, k, p)]; return a if 1 <= a <= N else 0
    l3b = {}
    for arm in ("CA", "CF", "RQ", "KQ"):
        rows_ = {}
        for s in sis:
            per_b = {}
            for N, k, p in CELLS:
                ans = outc(s, N, arm, k, p); a_nn, a_ca, a_m2 = outc(s, N, "Nn", k, p), outc(s, N, "CA", k, p), outc(s, N, "Nm2", k, p)
                preds = {"circuit": {"CA": [(1.0, a_m2)], "CF": [(1.0, a_m2)], "RQ": [(.80, a_nn), (.20, a_ca)], "KQ": [(.17, a_nn), (.83, a_ca)]}[arm],
                         "no_intervention": [(1.0, a_ca if arm == "RQ" else a_nn)], "pure_copy": [(1.0, k)], "one_step_shift": [(1.0, max(p - 1, 0))],
                         "actual_target": [(1.0, N + 1 - k)], "target_N_minus_2": [(1.0, N - 1 - k)], "uniform": []}
                if arm in ("RQ", "KQ"):
                    preds["full_restore" if arm == "RQ" else "full_removal"] = [(1.0, a_nn if arm == "RQ" else a_ca)]
                for nm_b, pts in preds.items():
                    pdist = dist_from(pts)(N); per_b.setdefault(nm_b, []).append(float(np.log(pdist[ans])))
            rows_[s] = {k_: float(np.mean(v_)) for k_, v_ in per_b.items()}
        base_names = [b_ for b_ in rows_[sis[0]] if b_ != "circuit"]
        res = {}
        for b_ in base_names:
            diff = np.array([rows_[s]["circuit"] - rows_[s][b_] for s in sis]); bs = diff[idx].mean(1)
            res[b_] = [float(diff.mean()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]]
        ll = {b_: float(np.mean([rows_[s][b_] for s in sis])) for b_ in ["circuit"] + base_names}
        beats_all = all(v_[1][0] > 0 for v_ in res.values()); beats_ni = res["no_intervention"][1][0] > 0
        l3b[arm] = dict(ll_per_answer=ll, gain_over=res, verdict=(nvt if not ok_all else "beats every baseline" if beats_all
                                                                  else "beats no-intervention only" if beats_ni else "does not"))
    out["L3b"] = l3b
    dno = np.array([abs(gfun(s, N, "Nn", k, p) - gfun(s, N, "Nstd", k, p)) for s in sis for N, k, p in CELLS]); p95n = float(np.percentile(dno, 95))
    noise = dict(noop_median=float(np.median(dno)), noop_p95=p95n,
                 noop_answer_change=float(np.mean([by[s]["land"][nm(N, "Nn", k, p)] != by[s]["land"][nm(N, "Nstd", k, p)] for s in sis for N, k, p in CELLS])))
    for arm, ref in (("CA", "Nn"), ("CF", "Nn"), ("KQ", "Nn"), ("RQ", "CA")):
        dd = np.array([abs(gfun(s, N, arm, k, p) - gfun(s, N, ref, k, p)) for s in sis for N, k, p in CELLS])
        noise[arm] = dict(median=float(np.median(dd)), share_above_noop_p95=float(np.mean(dd > p95n)),
                          answer_change=float(np.mean([by[s]["land"][nm(N, arm, k, p)] != by[s]["land"][nm(N, ref, k, p)] for s in sis for N, k, p in CELLS])))
    out["noise_vs_effect"] = noise
    fn = f"{V}/results/r3/s858_analysis.json" if tag == "s858_27b" else f"{V}/results/r3/s858_analysis_{tag}.json"
    json.dump(out, open(fn, "w"), indent=1)
    print("I0", json.dumps(out["I0"]))
    print("L1a", l1a, "| L1b", l1b, "|", json.dumps(L1.get(L51, {})))
    print("L2a", l2a, out["L2a_stats"], "| L2b", l2b, "| L2c", l2c, out["shape_counts"], "| L2d", l2d)
    print("L3a", json.dumps(l3a)); print("L3b", json.dumps({k_: v_["verdict"] for k_, v_ in l3b.items()}))
    print("noise_vs_effect", json.dumps(out["noise_vs_effect"]))
    for arm, x in out["shares"].items():
        print(f"  {arm:6s} " + "  ".join(f"{k_} {v_[0]:+.3f} [{v_[1][0]:+.3f}, {v_[1][1]:+.3f}]" for k_, v_ in x.items()))


if __name__ == "__main__":
    main(*sys.argv[1:])
