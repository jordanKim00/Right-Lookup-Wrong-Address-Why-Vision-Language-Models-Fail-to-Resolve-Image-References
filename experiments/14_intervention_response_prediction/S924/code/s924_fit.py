"""S924 fit and freeze (registered in knowledge/DECISIONS.md before the S924 test data; user 10-23: keep going until intervention prediction
clearly works) - the mechanism-calibrated predictor MC.  It encodes the circuit's state variables at the answer level and is fitted on natural
runs AND the interventions already observed (never on the S924 test arms):
  inputs per run: the early L47 profile e of the address the run reads (matcher-head mean, rows B1 / B4 / answer, centred over blocks) - for a
  writer-removed arm the profile of its writer-present twin (the address before the band), its anchor a (majority of the rows' argmax), the
  sharpness m = e(a) - max_{i != a} e(i), the writer flag w (1 natural / pinned, 0 band <- START's), k and N;
  answer logits over pictures b: theta_{G, w} . [e(b), 1(b = a), 1(b = a - 1), 1(b = a - 2), 1(b = a + 1), 1(b = k), 1(b = N), 1(b = 1),
  m 1(b = a), m 1(b = a - 1)] with G = k group {1, 2, 3, 4 .. N - 1, N} (S905: a k-set step back from any address for k >= 3; k = 2 holds a
  sharp address; k = 1 is pulled to the last picture; k = N jumps to the first picture, S913 / S914 / S922); 1% uniform mixing over
  {other, 1..N}; L2 1e-3; L-BFGS with the analytic gradient.
Training files: results/r3/s913_<tag>_collect.json for s913n7_27b, s913n8_27b (E, E_BANDR), s914n9_27b, s914n6_27b, s914e8_27b, s923A_27b
(E, E_BANDR, PIN, PIN_BANDR).  Baseline OFF (anchor-offset prior): P(x - a | G, w) from the same runs (add .5).
Usage: python3 s924_fit.py"""
from __future__ import annotations

import collections, hashlib, json, sys
import numpy as np
from scipy.optimize import minimize

TRAIN = ["s913n7_27b", "s913n8_27b", "s914n9_27b", "s914n6_27b", "s914e8_27b", "s923A_27b"]
OUT = "/vlm/research/ces/outputs/writer_circuit_2026-10-20/main/s924_frozen_MC.json"
NF = 10


def G(k, N):
    return 0 if k == 1 else 1 if k == 2 else 2 if k == 3 else 4 if k == N else 3


def anchor_of(q):
    q = np.asarray(q); am = [int(np.argmax(q[i])) + 1 for i in (2, 5, 9)]
    c = collections.Counter(am).most_common(); return c[0][0] if c[0][1] >= 2 else am[2]


def profile(r, key, em):
    h = np.asarray(r["qk47h"][key], float).mean(0); e = h[em].mean(0); return e - e.mean()


def unit(r, key_addr, key_ans, w, k, N, em):
    """one prediction unit: the address read (key_addr's L47 profile and anchor), the writer flag, the observed answer (key_ans)"""
    e = profile(r, key_addr, em); a = anchor_of(r["qk47"][key_addr])
    m = float(e[a - 1] - np.max(np.delete(e, a - 1)))
    return dict(e=e, a=a, m=m, w=w, k=k, N=N, g=G(k, N), x=int(r["land"][key_ans]))


def feats(u):
    N, a, k, e, m = u["N"], u["a"], u["k"], u["e"], u["m"]; F = np.zeros((N, NF))
    for b in range(1, N + 1):
        F[b - 1] = [e[b - 1], b == a, b == a - 1, b == a - 2, b == a + 1, b == k, b == N, b == 1, m * (b == a), m * (b == a - 1)]
    return F


def load(tag, arms):
    J = json.load(open(f"/vlm/results/r3/s913_{tag}_collect.json")) if tag.startswith(("s913", "s914", "s923")) else json.load(open(f"/vlm/results/r3/s924_{tag}_collect.json"))
    N = J["N"]; em = J["early_mean"]["47"]; out = []
    for si, r in enumerate(J["rows"]):
        for k in J["ks"]:
            if f"B{k}_E" not in r["land"]:
                continue
            for name, (ka, kx, w) in arms.items():
                ka_, kx_ = ka.format(k=k), kx.format(k=k)
                if ka_ in r["land"] and kx_ in r["land"]:
                    u = unit(r, ka_, kx_, w, k, N, em); u.update(arm=name, si=si, tag=tag, t=N + 1 - k); out.append(u)
    return out


ARMS_TRAIN = dict(E=("B{k}_E", "B{k}_E", 1), B=("B{k}_E", "B{k}_E_BANDR", 0), PIN=("B{k}_PIN", "B{k}_PIN", 1), PINB=("B{k}_PIN", "B{k}_PIN_BANDR", 0))


def mixp(lg, N):
    lg = lg - lg.max(); p = np.exp(lg) / np.exp(lg).sum(); pm = np.concatenate([[0.0], p]); return .99 * pm + .01 / (N + 1), p


def xi(u):
    return u["x"] if 1 <= u["x"] <= u["N"] else 0


def fit(units, l2=1e-3):
    Fs = [feats(u) for u in units]; idx = [u["g"] * 2 + u["w"] for u in units]; nP = 10
    def nll(th):
        W = th.reshape(nP, NF); s = 0.0; g = np.zeros_like(W)
        for u, F, ix in zip(units, Fs, idx):
            pm, p = mixp(F @ W[ix], u["N"]); j = xi(u); s -= np.log(pm[j])
            if j >= 1:
                wj = .99 * p[j - 1] / pm[j]; g[ix] -= wj * (F[j - 1] - p @ F)
        n = len(units); return s / n + l2 * (th ** 2).sum(), g.ravel() / n + 2 * l2 * th
    r = minimize(nll, np.zeros(nP * NF), jac=True, method="L-BFGS-B", options=dict(maxiter=2000))
    return r.x.reshape(nP, NF), float(r.fun)


def predict(u, W):
    pm, _ = mixp(feats(u) @ np.asarray(W)[u["g"] * 2 + u["w"]], u["N"]); return pm


def offset_prior(units):
    c = collections.defaultdict(collections.Counter)
    for u in units:
        c[f"{u['g']}_{u['w']}"][u["x"] - u["a"]] += 1
    return {kk: {str(o): n for o, n in v.items()} for kk, v in c.items()}


def predict_off(u, OFF):
    c = OFF.get(f"{u['g']}_{u['w']}", {}); p = np.array([c.get(str(b - u["a"]), 0) + .5 for b in range(1, u["N"] + 1)], float)
    pm = np.concatenate([[0.0], p / p.sum()]); return .99 * pm + .01 / (u["N"] + 1)


if __name__ == "__main__":
    units = []
    for tag in TRAIN:
        units += load(tag, ARMS_TRAIN)
    W, f = fit(units)
    out = dict(train=TRAIN, n_units=len(units), cells=dict(collections.Counter(f"{u['arm']}_g{u['g']}" for u in units)), note=__doc__,
               MC=W.tolist(), train_nll=f, OFF=offset_prior(units))
    json.dump(out, open(OUT, "w"), indent=1)
    print("frozen ->", OUT, hashlib.sha256(open(OUT, "rb").read()).hexdigest()[:16], "units", len(units), "train nll", round(f, 4))
