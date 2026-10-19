"""S827 predictions, fixed from STAGE 1 (natural runs) before STAGE 2 (the arms) runs; procedure registered in knowledge/DECISIONS.md before any
S827 data.  Writes research/ces/outputs/s827_predictions_<tag>.json and prints its sha256 (recorded in DECISIONS before stage 2).
Internal profile z_F(run) = mean over the rows of F of the index-matching-head block scores at F's layer, F one of {look layer} x {B1 + B4,
the five boundary rows, the question mark + the five boundary rows}.
(a) Readout law: P(land = i) proportional to exp(beta z_F,i), i = 1..N; F and beta by maximum likelihood on the non-identity natural runs only
    (START(k), START(k +- 1), END(k +- 1), both cases).  Its prediction for the held-out identity cell END(k) (each scene's own measured z).
(b) Held-out-scene END operator (internal and answer): per scene s, the END change pattern D_s = z_END(s) - z_START(s) (per head and for the
    matcher mean, centred over blocks) is predicted by the mean over the other scenes of the case; internal accuracy = sign agreement of the local
    coefficients (a_(k-1), a_k, a_(k+1); relative to the far blocks) for heads whose training mean is beyond 2 standard errors, and the pattern
    correlation; answer = log-probability of s's END landing under softmax(beta (z_START(s) + mean D_-s)) against the base rate (other scenes'
    END landings, +.5 smoothing).
(c) Every stage-2 arm, per scene: the direct change at F's layer of F's rows' block scores, linear in the block-mean keys
    (dz_b = scaling (dq . Kbar_b), dq = the arm's query change from the natural post-RoPE queries; local parts x_S = pinv(C) sum a_i e_i,
    U projector as exp/c_qpath.py), added to the recipient's natural z (matcher mean of B1 and B4), and the answer shares softmax(beta z_pred).
    This ignores what the edits at the other look layers do downstream - its error measures how much those compound.
Usage: python s827_predict.py TAG"""
from __future__ import annotations

import hashlib, json, sys
import numpy as np

NONID = ["START", "START_m1", "START_p1", "END_m1", "END_p1"]
ARMS = dict(A_FULL=("END", "U", -1.0), A_HALF=("END", "U", -0.5), A_DIP=("END", "k", -1.0), A_DIP50=("END", "k", -0.5),
            A_PREV=("END", "km1", -1.0), A_PREV50=("END", "km1", -0.5), A_NEXT=("END", "kp1", -1.0), A_FLANK=("END", "flank", -1.0),
            A_UK=("END", "uk", -1.0), R_FULL=("START", "U", 1.0), R_DIP=("START", "k", 1.0), R_PREV=("START", "km1", 1.0), R_NEXT=("START", "kp1", 1.0))


def main(tag):
    d = json.load(open(f"/vlm/results/r3/s827_{tag}_natural.json")); R = d["rows"]
    Z = np.load(f"/vlm/results/r3/s827_{tag}_natural.npz")
    look = d["look"]; scaling = d["scaling"]
    def lp(zz, x, beta):
        s = beta * zz; s = s - s.max()
        return float(s[x - 1] - np.log(np.exp(s).sum()))
    ROWSETS = {"B1B4": [2, 5], "B": [1, 2, 3, 4, 5], "QB": [0, 1, 2, 3, 4, 5]}
    betas = np.linspace(0.05, 12.0, 480); cand = []
    for L_ in look:
        if not d["matchers"].get(str(L_)):
            continue
        for rs, ix in ROWSETS.items():
            zf = lambda r, run, L_=L_, ix=ix: np.mean([np.array(r["qk"][run][str(L_)][i]) for i in ix], 0)
            fit_ = [(zf(r, run), r["land"][run]) for r in R for run in NONID if r["land"][run] >= 1]
            ll = [sum(lp(zz, x, b_) for zz, x in fit_) for b_ in betas]
            cand.append((max(ll), L_, rs, float(betas[int(np.argmax(ll))]), len(fit_)))
    best = max(cand); _, L51, RS, beta, nfit = best
    li = look.index(L51); MH = d["matchers"][str(L51)]; RIX = ROWSETS[RS]
    z = lambda r, run: np.mean([np.array(r["qk"][run][str(L51)][i]) for i in RIX], 0)
    fit = [None] * nfit
    soft = lambda zz: (lambda s: np.exp(s) / np.exp(s).sum())(beta * zz - (beta * zz).max())
    out = dict(tag=tag, layer=L51, rows=RS, beta=beta, fit_n=nfit, a={}, b={}, c={},
               candidates=[dict(ll=c_[0], layer=c_[1], rows=c_[2], beta=c_[3]) for c_ in sorted(cand, reverse=True)])
    for N, k in ((7, 4), (9, 5)):
        rows = [r for r in R if r["N"] == N]
        if not rows:
            continue
        cname = f"N{N}k{k}"
        pr = np.mean([soft(z(r, "END")) for r in rows], 0)
        out["a"][cname] = dict(pred={x: float(pr[x - 1]) for x in (k - 1, k, k + 1)}, obs={x: float(np.mean([r["land"]["END"] == x for r in rows])) for x in (k - 1, k, k + 1)})
        # (b) held-out-scene END operator, per head (all heads at L51, rows B1 / B4 mean) and matcher mean
        cap = lambda r, run: Z[f"N{N}_s{r['si']}_{run}_cap"][li].astype(np.float32)     # [rows, H, nb]
        Dh = np.stack([(cap(r, "END") - cap(r, "START"))[RIX].mean(0) for r in rows])     # [S, H, nb]
        Dh = Dh - Dh.mean(-1, keepdims=True)
        far = [i for i in range(N) if i not in (k - 2, k - 1, k)]
        loc = lambda D: np.stack([D[..., i] - D[..., far].mean(-1) for i in (k - 2, k - 1, k)], -1)   # [..., 3]
        A = loc(Dh)                                                                          # [S, H, 3]
        sign_hits, sign_n, corrs = 0, 0, []
        gains = []
        for s in range(len(rows)):
            tr = [i for i in range(len(rows)) if i != s]
            mu, se = A[tr].mean(0), A[tr].std(0, ddof=1) / np.sqrt(len(tr))
            ok = np.abs(mu) > 2 * se
            sign_hits += int((np.sign(A[s]) == np.sign(mu))[ok].sum()); sign_n += int(ok.sum())
            for h in range(Dh.shape[1]):
                pm = Dh[tr, h].mean(0)
                if np.std(pm) > 0 and np.std(Dh[s, h]) > 0:
                    corrs.append(float(np.corrcoef(pm, Dh[s, h])[0, 1]))
            Dm = np.stack([z(rows[i], "END") - z(rows[i], "START") for i in tr]); Dm = (Dm - Dm.mean(-1, keepdims=True)).mean(0)
            x = rows[s]["land"]["END"]
            if x >= 1:
                zp = z(rows[s], "START") + Dm
                base = np.bincount([rows[i]["land"]["END"] for i in tr if rows[i]["land"]["END"] >= 1], minlength=N + 1)[1:] + .5
                gains.append(lp(zp, x, beta) - float(np.log(base[x - 1] / base.sum())))
        out["b"][cname] = dict(sign_agree=(sign_hits / sign_n if sign_n else None), sign_n=sign_n, median_pattern_corr=float(np.median(corrs)) if corrs else None,
                               answer_gain_mean=float(np.mean(gains)) if gains else None, answer_gain_per_scene=gains,
                               head_means_L51={str(h): [float(x) for x in A[:, h].mean(0)] for h in range(A.shape[1])})
        # (c) every arm's direct change at L51
        C_ = {}
        for r in rows:
            key = f"N{N}_s{r['si']}"
            qE = Z[key + "_qeE"][li][RIX].astype(np.float64); qS = Z[key + "_qeS"][li][RIX].astype(np.float64)       # [rows, H, hd]
            kb = Z[key + "_kb"][li].astype(np.float64)                                                                   # [nb, H, hd]
            dz = {nm: np.zeros((len(RIX), len(MH), N)) for nm in ARMS}
            for hi, h in enumerate(MH):
                Kb = kb[:, h]; C = Kb - Kb.mean(0, keepdims=True)
                Uu, sv, _ = np.linalg.svd(C.T, full_matrices=False); Uu = Uu[:, sv > 1e-6 * sv.max()]
                Pi = np.linalg.pinv(C, rcond=1e-6)
                u = Kb[k - 1] - np.delete(Kb, k - 1, 0).mean(0)
                for ri in range(len(RIX)):
                    dd = qE[ri, h] - qS[ri, h]
                    c = C @ dd; a = {i: c[i] - c[far].mean() for i in (k - 2, k - 1, k)}
                    e = lambda i: np.eye(N)[i] - 1.0 / N
                    part = dict(U=Uu @ (Uu.T @ dd), k=Pi @ (a[k - 1] * e(k - 1)), km1=Pi @ (a[k - 2] * e(k - 2)), kp1=Pi @ (a[k] * e(k)),
                                uk=(dd @ u) / (u @ u) * u)
                    part["flank"] = part["km1"] + part["kp1"]
                    for nm, (rcp, pt, coef) in ARMS.items():
                        dz[nm][ri, hi] = scaling * (Kb @ (coef * part[pt]))
            for nm, (rcp, pt, coef) in ARMS.items():
                zp = z(r, rcp) + dz[nm].mean((0, 1))
                C_.setdefault(nm, []).append(dict(si=r["si"], dz=dz[nm].mean((0, 1)).tolist(), z_pred=zp.tolist(), p=soft(zp).tolist()))
        out["c"][cname] = {nm: dict(per_scene=v_, shares={x: float(np.mean([s_["p"][x - 1] for s_ in v_])) for x in (k - 1, k, k + 1)})
                           for nm, v_ in C_.items()}
    fn = f"/vlm/research/ces/outputs/s827_predictions_{tag}.json"
    js = json.dumps(out, indent=1, default=float); open(fn, "w").write(js)
    print(f"readout feature: layer {L51} rows {RS} beta {beta:.3f} fit_n {nfit}")
    for cname in out["a"]:
        print(cname, "(a) END law pred", {x: round(y, 3) for x, y in out["a"][cname]["pred"].items()}, "obs", out["a"][cname]["obs"])
        b = out["b"][cname]
        print(cname, f"(b) sign agree {b['sign_agree']} (n {b['sign_n']}) median corr {b['median_pattern_corr']} answer gain {b['answer_gain_mean']}")
        for nm, v_ in out["c"][cname].items():
            print(cname, "(c)", f"{nm:8s}", {x: round(y, 3) for x, y in v_["shares"].items()})
    print("sha256", hashlib.sha256(js.encode()).hexdigest(), fn)


if __name__ == "__main__":
    main(sys.argv[1])
