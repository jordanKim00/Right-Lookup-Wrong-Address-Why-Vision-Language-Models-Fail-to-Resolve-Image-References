"""S923 fit and freeze (registered in knowledge/DECISIONS.md before the S923 test data): the circuit-structured predictor AP - S913's
structured model M3 revised by two circuit findings, fitted on the S913 training runs only (N7 / N8, natural END(k) with E_BANDR):
  (1) no argmax features: the late support and the writer's update depend on the early profile through e(b - 1 .. b + 3), the last-picture
      and copy indicators only - no anchor indicator, no anchor-margin terms (the early reference is a graded preference; the argmax
      indicator absorbed the anchor's support in natural data and froze it under sharpening, S914 post hoc);
  (2) a two-temperature readout: answer logits = T_B,g * B^ + T_W,g * W^ per k group, T_B fitted on the writer-free runs' answers (x(E_BANDR)
      from B^ alone), then T_W on the natural answers with T_B fixed (the writer's effect reaches the answer through other late routes than
      the lookup support, S917 / S921; a single temperature mis-sized the writer-free readout, S914 post hoc).
Predictions: natural softmax(T_B B^ + T_W W^); writer removal softmax(T_B B^); a sharpened address the same on the run's own profile.
Original S913 doc follows.
S913 fit and freeze (registered in knowledge/DECISIONS.md before the S914 test data; the advisor's design, 10-22).  Trains every predictor on
the S913 training runs only (natural END(k), N 7 and 8, every k; results/r3/s913_<tag>_collect.json) and writes the frozen parameters with a
hash.  The test data (S914) are opened only by s914_analyze.py after this file's output is hashed in DECISIONS.
Per END run: k, N, its natural L47 anchor a, the answer x; early profiles at L47 (and L43): e_mean[b] = the matcher-head mean block score
(rows B1 / B4 / answer averaged) and e_head[h, b] for every head (rows averaged), centred over blocks only (magnitudes kept); late profiles at
L51 (matcher-head mean, rows B1 / B4 / answer): zE (the natural run of phase B) and zB (E_BANDR, the writer-free run) -> B = zB - mean(zB),
W = (zE - zB) - mean(zE - zB).  Positions outside 1..N are padded with min(profile) - 1.
Models (k groups: k = 1, 2, 3, >= 4):
  M0     behaviour prior: P(x - t | k group), t = N + 1 - k (add .5);
  M1     argmax: P(x - a | k group) (add .5);
  M2a    conditional logit over b with f(b) = [e(b-1), e(b), e(b+1), e(b+2), e(b+3), last, copy (b = k), anchor (b = a)] from e_mean, weights per k
         group, L2 1e-3;
  M2b    the same with every head's e_head(b), e_head(b+1) in place of the five mean terms, L2 1e-2;
  M3     structured: ridge regressions (per k group, lambda 1) of B(b) and W(b) on f(b) plus the anchor's early margin m47 = e(a) - max_{i != a} e(i)
         times 1(b = a) and 1(b = a - 1); late address P_late = softmax(T_late (B^ + W^)), answer P_ans = softmax(T_ans,g (B^ + W^)), where only the
         temperatures are fitted (T_late to the natural late argmax, T_ans per k group to the answers);
  M3noW  M3 without W^ (answer and late from B^ only, temperatures refitted);
  *_L43  M2a and M3 with the L43 early profile (pre-band boundary).
Every distribution is over {other, 1..N} (other = an answer outside 1..N) mixed with 1% uniform.
Usage: python3 s913_fit.py TRAIN_TAG_N7 TRAIN_TAG_N8"""
from __future__ import annotations

import collections, hashlib, json, sys
import numpy as np
from scipy.optimize import minimize

TAGS = sys.argv[1:3]
OUT = "/vlm/research/ces/outputs/writer_circuit_2026-10-20/main/s923_frozen_AP.json"
KG = lambda k: 0 if k == 1 else 1 if k == 2 else 2 if k == 3 else 3


def centre(v):
    return np.asarray(v, float) - np.mean(v)


def load_runs(tags, layers=(47, 43)):
    runs = []
    for tag in tags:
        J = json.load(open(f"/vlm/results/r3/s913_{tag}_collect.json")); N = J["N"]; lm = J["late_mean"]; em = J["early_mean"]
        for si, r in enumerate(J["rows"]):
            for k in J["ks"]:
                key = f"N_E{k}"; a = r["anchor"][str(k)]
                if f"B{k}_E" not in r["zl"]:
                    continue
                d = dict(tag=tag, N=N, k=k, si=si, a=a, x=int(r["land"][key]))
                for L in layers:
                    h = np.asarray(r[f"qk{L}h"][key], float).mean(0)                          # [H, N] (rows averaged)
                    d[f"e{L}"] = centre(h[em[str(L)]].mean(0)); d[f"eh{L}"] = np.stack([centre(x) for x in h])
                zl = lambda key_: np.asarray(r["zl"][key_]["51"], float)[:, lm["51"]].mean(1).mean(0)
                zE, zB = zl(f"B{k}_E"), zl(f"B{k}_E_BANDR")
                d["B"] = centre(zB); d["W"] = centre(zE - zB); d["late"] = int(np.argmax(zE)) + 1; d["xB"] = int(r["land"][f"B{k}_E_BANDR"])
                runs.append(d)
    return runs


def pad(e, b):
    return e[b - 1] if 1 <= b <= len(e) else float(np.min(e) - 1.0)


def fmean(d, L):
    e = d[f"e{L}"]; N = d["N"]; F = np.zeros((N, 8))
    for b in range(1, N + 1):
        F[b - 1, :5] = [pad(e, b + j) for j in (-1, 0, 1, 2, 3)]
        F[b - 1, 5:] = [float(b == N), float(b == d["k"]), float(b == d["a"])]
    return F


def fhead(d, L):
    eh = d[f"eh{L}"]; H = eh.shape[0]; N = d["N"]; F = np.zeros((N, 2 * H + 3))
    for b in range(1, N + 1):
        F[b - 1, :H] = [pad(eh[h], b) for h in range(H)]; F[b - 1, H:2 * H] = [pad(eh[h], b + 1) for h in range(H)]
        F[b - 1, 2 * H:] = [float(b == N), float(b == d["k"]), float(b == d["a"])]
    return F


def mix(p, N):
    p = np.clip(np.asarray(p, float), 0, None); p = p / p.sum(); return .99 * p + .01 / (N + 1)


def xi(d, x=None):
    x = d["x"] if x is None else x; return x if 1 <= x <= d["N"] else 0


def fit_logit(runs, fk, l2):
    nf = fk(runs[0]).shape[1]
    Fs = [fk(d) for d in runs]
    def nll(th):
        W = th.reshape(4, nf); s = 0.0; g = np.zeros_like(W)
        for d, F in zip(runs, Fs):
            lg = F @ W[KG(d["k"])]; lg = lg - lg.max(); p = np.exp(lg) / np.exp(lg).sum()
            pm = np.concatenate([[0.0], p]); pm = .99 * pm + .01 / (d["N"] + 1); j = xi(d)
            s -= np.log(pm[j])
            if j >= 1:                                                                         # gradient of -log(.99 p_j + .01/(N+1))
                w = .99 * p[j - 1] / pm[j]; gj = -(w * (F[j - 1] - p @ F))
            else:
                gj = np.zeros(nf)
            g[KG(d["k"])] += gj
        n = len(runs); return s / n + l2 * (th ** 2).sum(), g.ravel() / n + 2 * l2 * th
    r = minimize(nll, np.zeros(4 * nf), jac=True, method="L-BFGS-B", options=dict(maxiter=1000))
    return r.x.reshape(4, nf)


def ridge(X, y, lam=1.0):
    Xa = np.concatenate([X, np.ones((len(X), 1))], 1)
    A = Xa.T @ Xa + lam * np.eye(Xa.shape[1]); A[-1, -1] -= lam
    return np.linalg.solve(A, Xa.T @ y)


def f3(d, L):
    F = fmean(d, L); e = d[f"e{L}"]; a = d["a"]; m47 = float(e[a - 1] - np.max(np.delete(e, a - 1)))
    extra = np.zeros((d["N"], 2))
    extra[a - 1, 0] = m47
    if a >= 2:
        extra[a - 2, 1] = m47
    return np.concatenate([F, extra], 1)


def fit_m3(runs, L):
    out = {}
    for g in range(4):
        rs = [d for d in runs if KG(d["k"]) == g]
        X = np.concatenate([f3(d, L) for d in rs]); yB = np.concatenate([d["B"] for d in rs]); yW = np.concatenate([d["W"] for d in rs])
        out[g] = dict(B=ridge(X, yB), W=ridge(X, yW))
    return out


def m3_scores(d, P, L, useW=True):
    X = np.concatenate([f3(d, L), np.ones((d["N"], 1))], 1); c = P[KG(d["k"])]
    return X @ c["B"] + (X @ c["W"] if useW else 0.0)


def fit_temp(runs, sc, target):
    def nll(t):
        s = 0.0
        for d in runs:
            v = t[0] * sc(d); v = v - v.max(); p = np.exp(v) / np.exp(v).sum()
            s -= np.log(mix(np.concatenate([[0.0], p]), d["N"])[target(d)])
        return s / len(runs)
    return float(minimize(nll, np.array([1.0]), method="Nelder-Mead").x[0])


def offset_model(runs, ref):
    c = collections.defaultdict(collections.Counter)
    for d in runs:
        c[KG(d["k"])][d["x"] - ref(d)] += 1
    return {str(g): {str(o): n for o, n in v.items()} for g, v in c.items()}


def fap(d, L):
    """AP features: e(b - 1 .. b + 3), last picture, copy (b = k) - no anchor indicator, no anchor margin"""
    return fmean(d, L)[:, :7]


def fit_ap(runs, L):
    out = {}
    for g in range(4):
        rs = [d for d in runs if KG(d["k"]) == g]
        X = np.concatenate([fap(d, L) for d in rs]); yB = np.concatenate([d["B"] for d in rs]); yW = np.concatenate([d["W"] for d in rs])
        out[g] = dict(B=ridge(X, yB), W=ridge(X, yW))
    return out


def ap_scores(d, P, L):
    X = np.concatenate([fap(d, L), np.ones((d["N"], 1))], 1); c = P[KG(d["k"])]
    return X @ c["B"], X @ c["W"]


def fit_T2(runs, P, L):
    T = {}
    for g in range(4):
        rs = [d for d in runs if KG(d["k"]) == g]
        def nllB(t):
            s_ = 0.0
            for d in rs:
                b, _ = ap_scores(d, P, L); v = t[0] * b; v = v - v.max(); p = np.exp(v) / np.exp(v).sum()
                s_ -= np.log(mix(np.concatenate([[0.0], p]), d["N"])[xi(d, d["xB"])])
            return s_ / len(rs)
        tB = float(minimize(nllB, np.array([1.0]), method="Nelder-Mead").x[0])
        def nllW(t):
            s_ = 0.0
            for d in rs:
                b, w = ap_scores(d, P, L); v = tB * b + t[0] * w; v = v - v.max(); p = np.exp(v) / np.exp(v).sum()
                s_ -= np.log(mix(np.concatenate([[0.0], p]), d["N"])[xi(d)])
            return s_ / len(rs)
        T[str(g)] = dict(B=tB, W=float(minimize(nllW, np.array([1.0]), method="Nelder-Mead").x[0]))
    return T


if __name__ == "__main__":
    runs = load_runs(TAGS)
    out = dict(train_tags=TAGS, n_train=len(runs), cells=dict(collections.Counter(f"N{d['N']}_k{d['k']}" for d in runs)), note=__doc__)
    P = fit_ap(runs, 47)
    out["AP"] = {str(g): {kk: v.tolist() for kk, v in c.items()} for g, c in P.items()}
    out["AP_T"] = fit_T2(runs, P, 47)
    json.dump(out, open(OUT, "w"), indent=1)
    print("frozen ->", OUT, hashlib.sha256(open(OUT, "rb").read()).hexdigest()[:16], "n_train", len(runs), out["AP_T"])
