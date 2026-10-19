"""Bump + slope model of the END late lookup, and its calibration (descriptive; rules fixed 10-18 21:0x BEFORE any S858 result, logged in
knowledge/DECISIONS.md; user 10-18 20:5x: the next causal test of the slope combination is run only if a bump + slope model explains the
new-scene position-wise scores; the failure hypothesis is a miscalibrated combination - the required net slope (N + 1 - k - p) / sigma^2
depends on p and is 0 at identity, the realised slope may not).
Profiles: late block scores of the shared pictures (rows B1, B4, answer; mean over layers 51-63), centred.
Model per run: z(i) = a G(i; c, sigma) + gamma (i - mean) + const, G = exp(-(i - c)^2 / (2 sigma^2)) (centred), fitted by least squares for a
and gamma.  c = p (anchor bump; 'bump + slope'), c = j* (target bump), or the best c on a 0.25 grid over the pictures (free peak, upper bound).
sigma: fitted on development data only (S857 natural N8, N6 and cut CA runs, c = p; grid .5-3 by .05, maximising the mean R^2), then frozen.
Outputs on S858 (natural Nn runs of every cell; the S arm reported as the START contrast):
  R^2 of bump(p) + slope, of target-bump + slope, of the free peak; share of runs where bump(p) + slope beats target-bump + slope
  calibration: gamma regressed on the required (j* - p) (pooled; coefficient, bootstrap CI) and on k, N, p jointly (the required law has
  coefficients -1, +1, -1 on k, N, p in units of 1 / sigma^2)
Decision rule (fixed now; it gates only whether to propose the I_{Q,V} combination test): 'sufficient' if the median R^2 of bump(p) + slope
>= .70 and it beats target-bump + slope in >= 60% of the Nn runs; else 'not sufficient'.
Usage: python3 exp/p4_s858_slopefit.py [TAG]  ->  research/ces/outputs/s858_slopefit.json"""
from __future__ import annotations

import json, sys
import numpy as np

V = "/home/storage2/ai26269345/vlm"


def prof(qk, LATE, m):
    z = np.mean([[qk[L][ri][:m] for ri in (2, 5, 9)] for L in LATE], axis=(0, 1)); return z - z.mean()


def fit(z, c, sigma):
    m = len(z); i = np.arange(1, m + 1, dtype=float)
    G = np.exp(-(i - c) ** 2 / (2 * sigma ** 2)); G = G - G.mean(); x = i - i.mean()
    X = np.stack([G, x], 1); b = np.linalg.lstsq(X, z, rcond=None)[0]; r = z - X @ b
    ss = float(z @ z)
    return (1 - float(r @ r) / ss if ss > 0 else 0.0), float(b[0]), float(b[1])


def main(tag="s858_27b"):
    d0 = json.load(open(f"{V}/results/r3/s857_s857_27b_cut.json")); L0 = [str(l) for l in d0["late"]]
    dev = []
    for r in d0["rows"]:
        for k in (3, 5):
            for p in (3, 4):
                for arm, m in (("N8", 6), ("N6", 6), ("CA", 6)):
                    dev.append((prof(r["qk"][f"{arm}_k{k}p{p}"], L0, m), p))
    grid = np.round(np.arange(.5, 3.0001, .05), 2)
    score = [np.mean([fit(z, p, sg)[0] for z, p in dev]) for sg in grid]
    sigma = float(grid[int(np.argmax(score))])
    d = json.load(open(f"{V}/results/r3/s858_{tag}_pred.json")); L = [str(l) for l in d["late"]]
    KN = {int(k_): tuple(v_) for k_, v_ in d["kn"].items()}
    rows = []
    for r in d["rows"]:
        for N in d["ns"]:
            for k in KN[N]:
                for p in d["ps"]:
                    for arm in ("Nn", "S"):
                        z = prof(r["qk"][f"N{N}_{arm}_k{k}p{p}"], L, N - 2); m = N - 2
                        js = (N + 1 - k) if arm == "Nn" else k
                        r_bp, a_bp, g_bp = fit(z, p, sigma); r_bt, _, _ = fit(z, js, sigma)
                        r_free = max(fit(z, c, sigma)[0] for c in np.arange(1, m + .001, .25))
                        rows.append(dict(si=r["si"], arm=arm, N=N, k=k, p=p, jstar=js, r2_bp=r_bp, r2_bt=r_bt, r2_free=r_free, a=a_bp, gamma=g_bp))
    out = dict(sigma_dev=sigma, dev_mean_r2=float(max(score)))
    rng = np.random.default_rng(11); sis = sorted({x["si"] for x in rows})
    for arm in ("Nn", "S"):
        R = [x for x in rows if x["arm"] == arm]
        o = dict(median_r2_bump_p_slope=float(np.median([x["r2_bp"] for x in R])), median_r2_target_bump_slope=float(np.median([x["r2_bt"] for x in R])),
                 median_r2_free_peak=float(np.median([x["r2_free"] for x in R])), share_bp_beats_bt=float(np.mean([x["r2_bp"] > x["r2_bt"] for x in R])))
        if arm == "Nn":
            X = np.array([[1.0, x["jstar"] - x["p"]] for x in R]); y = np.array([x["gamma"] for x in R])
            X2 = np.array([[1.0, x["k"], x["N"], x["p"]] for x in R])
            b1 = np.linalg.lstsq(X, y, rcond=None)[0]; b2 = np.linalg.lstsq(X2, y, rcond=None)[0]
            bs1, bs2 = [], []
            per = {s_: [j for j, x in enumerate(R) if x["si"] == s_] for s_ in sis}
            for _ in range(2000):
                ii = [j for s_ in rng.choice(sis, len(sis)) for j in per[s_]]                    # scene bootstrap with repetition
                bs1.append(np.linalg.lstsq(X[ii], y[ii], rcond=None)[0][1]); bs2.append(np.linalg.lstsq(X2[ii], y[ii], rcond=None)[0][1:])
            bs2 = np.array(bs2)
            ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            o["calibration_gamma_on_required"] = [float(b1[1]), ci(bs1)]
            o["required_unit_per_picture"] = float(1 / sigma ** 2)
            o["gamma_on_k_N_p"] = {nm: [float(b2[i + 1]), ci(bs2[:, i])] for i, nm in enumerate(("k", "N", "p"))}
            o["cell_mean_gamma"] = {f"N{N}k{k}p{p}": float(np.mean([x["gamma"] for x in R if (x["N"], x["k"], x["p"]) == (N, k, p)]))
                                   for N in d["ns"] for k in KN[N] for p in d["ps"]}
            o["decision"] = ("sufficient" if o["median_r2_bump_p_slope"] >= .70 and o["share_bp_beats_bt"] >= .60 else "not sufficient")
        out[arm] = o
    json.dump(out, open(f"{V}/research/ces/outputs/s858_slopefit.json" if tag == "s858_27b" else f"{V}/research/ces/outputs/s858_slopefit_{tag}.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
