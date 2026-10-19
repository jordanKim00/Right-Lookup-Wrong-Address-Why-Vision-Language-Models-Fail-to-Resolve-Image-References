"""S803 (analysis only, no model call; registered in knowledge/DECISIONS.md before any Qwen3.5-27B S801 data): a two-route model of the natural
end-relative answer distribution, tested by predicting a held-out number of images.
Data: S801 natural END(k) answers (unedited arm), N in {6, 7, 8}, k in {1, 2, 3, 4, 5} (present at every N), 40 scenes per cell; outcome b in
{1..N} (the landed image) or 0 (a menu word not in the scene).
Models, parameters per k shared across N:
  D (copy + end-anchored route):   P(b) = eps [b = 0] + (1 - eps) (c [b = k] + e G_N(b; N + 1 - k + mu, sigma) + u / N),  c + e + u = 1
  S (copy + fixed-position route): the same with the kernel centred at m, independent of N
  K (end-anchored route only):     D with c = 0
  C (copy only):                   D with e = 0
G_N is a Gaussian over the blocks 1..N, normalized over them.  Maximum likelihood (L-BFGS-B, 12 starts, seed 80300).
Test: leave one N out; fit each model per k on the other two N, score the held-out cell; per held-out scene the log-likelihood summed over k;
paired differences D - X over the 120 held-out scenes, scene bootstrap 4,000 (seed 80301).
Registered rules (27B):
  X1  LL(D) - LL(S) > 0 with the interval above 0 (the answers move with N: the route is anchored at the end)
  X2  LL(D) - LL(C) > 0 with the interval above 0 (an end-anchored route is needed beyond the copy)
  X3  fitted on all three N: copy weight c(k) <= .15 for k in {1, 2, 3, 5} (27B's answers do not copy; k = 4 reported, its copy coincides with
      the target at N = 7)
  X4  fitted on all three N: sigma(5) - sigma(2) >= .5 (the route blurs with distance from the end); evaluated only when the route weight
      e >= .3 at both k = 2 and k = 5, otherwise "not evaluable" (amended before any 27B data, after the exploratory 9B fit where e(5) = .03)
Descriptive: every fitted parameter per k, and the per-k held-out differences.  Positive control (software, not evidence): synthetic answers
drawn from D and from S with known parameters must be assigned to their generating model by the same test.
Usage (docker): python /vlm/exp/s803_mixture.py TAG [TAG ...]   (TAG = S801 tag, e.g. s801_35_9b; --control runs the synthetic check)"""
from __future__ import annotations

import json, sys
import numpy as np
from scipy.optimize import minimize

NS, KS = (6, 7, 8), (1, 2, 3, 4, 5)
B = 4000


def kernel(N, centre, sigma):
    b = np.arange(1, N + 1); w = np.exp(-0.5 * ((b - centre) / sigma) ** 2)
    s = w.sum()
    return w / s if s > 0 else np.full(N, 1.0 / N)


def probs(model, th, N, k):
    """P over outcomes 0..N for one cell."""
    a1, a2, a3, a4, a5 = th
    eps = 0.5 / (1 + np.exp(-a3))
    if model == "D" or model == "S":
        z = np.array([a1, a2, 0.0]); w = np.exp(z - z.max()); c, e, u = w / w.sum()
    elif model == "K":
        z = np.array([a2, 0.0]); w = np.exp(z - z.max()); c, (e, u) = 0.0, w / w.sum()
    else:                                                                       # C
        z = np.array([a1, 0.0]); w = np.exp(z - z.max()); (c, u), e = w / w.sum(), 0.0
    sigma = 0.25 + np.log1p(np.exp(a5))
    centre = (N + 1 - k + 4 * np.tanh(a4)) if model in ("D", "K", "C") else (0.5 + 8 / (1 + np.exp(-a4)))
    p = np.zeros(N + 1)
    p[1:] = e * kernel(N, centre, sigma) + u / N
    p[k] += c
    p[1:] *= (1 - eps); p[0] = eps
    return np.clip(p, 1e-12, 1.0)


def nll(model, th, cells):
    return -sum(float(np.dot(cnt, np.log(probs(model, th, N, k)))) for (N, k, cnt) in cells)


def fit(model, cells, seed=80300, starts=12):
    rng = np.random.default_rng(seed); best = None
    for _ in range(starts):
        x0 = rng.normal(0, 1.5, 5)
        r = minimize(lambda th: nll(model, th, cells), x0, method="L-BFGS-B")
        if best is None or r.fun < best.fun:
            best = r
    return best.x


def params(model, th, N=7, k=1):
    a1, a2, a3, a4, a5 = th
    p = dict(eps=float(0.5 / (1 + np.exp(-a3))), sigma=float(0.25 + np.log1p(np.exp(a5))))
    if model in ("D", "S"):
        z = np.array([a1, a2, 0.0]); w = np.exp(z - z.max()); w = w / w.sum(); p.update(c=float(w[0]), e=float(w[1]), u=float(w[2]))
    if model == "D":
        p["mu"] = float(4 * np.tanh(a4))
    if model == "S":
        p["m"] = float(0.5 + 8 / (1 + np.exp(-a4)))
    return p


def counts(rows, N, k):
    cnt = np.zeros(N + 1)
    for r in rows:
        if r["N"] == N and r["k"] == k:
            cnt[int(r["land"])] += 1
    return cnt


def evaluate(rows):
    """rows: dicts with N, k, si, land.  Returns the held-out comparison and the all-N fits."""
    models = ("D", "S", "K", "C"); per_scene = {m: {} for m in models}; per_k = {m: {k: 0.0 for k in KS} for m in models}
    for hold in NS:
        train = [n for n in NS if n != hold]
        for k in KS:
            cells = [(N, k, counts(rows, N, k)) for N in train]
            for m in models:
                th = fit(m, cells)
                lp = np.log(probs(m, th, hold, k))
                for r in rows:
                    if r["N"] == hold and r["k"] == k:
                        key = (hold, r["si"]); per_scene[m][key] = per_scene[m].get(key, 0.0) + float(lp[int(r["land"])])
                        per_k[m][k] += float(lp[int(r["land"])])
    keys = sorted(per_scene["D"]); rng = np.random.default_rng(80301); out = dict(n_scenes=len(keys), total={m: float(sum(per_scene[m].values())) for m in models})
    for x in ("S", "K", "C"):
        d = np.array([per_scene["D"][q] - per_scene[x][q] for q in keys])
        bs = [d[rng.integers(0, len(d), len(d))].sum() for _ in range(B)]
        out[f"D-{x}"] = dict(diff=float(d.sum()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                             per_k={str(k): per_k["D"][k] - per_k[x][k] for k in KS})
    out["fit_all"] = {m: {str(k): params(m, fit(m, [(N, k, counts(rows, N, k)) for N in NS])) for k in KS} for m in ("D", "S")}
    return out


def rules(ev):
    fd = ev["fit_all"]["D"]
    return dict(X1=bool(ev["D-S"]["diff"] > 0 and ev["D-S"]["ci"][0] > 0), X2=bool(ev["D-C"]["diff"] > 0 and ev["D-C"]["ci"][0] > 0),
                X3=bool(all(fd[str(k)]["c"] <= .15 for k in (1, 2, 3, 5))), X4=(bool(fd["5"]["sigma"] - fd["2"]["sigma"] >= .5) if (fd["2"]["e"] >= .3 and fd["5"]["e"] >= .3) else "not evaluable"))


def control():
    """Software check: synthetic data from D (end-anchored) and from S (fixed position) with known parameters."""
    rng = np.random.default_rng(1); res = {}
    for gen in ("D", "S"):
        rows = []
        for N in NS:
            for k in KS:
                if gen == "D":
                    th = np.array([np.log(.2 / .1), np.log(.7 / .1), -3.0, 0.0, np.log(np.expm1(0.3 + 0.25 * k))])
                else:
                    m_k = 8 - k - 0.5                                            # fixed position, independent of N
                    a4 = np.log((m_k - 0.5) / 8 / (1 - (m_k - 0.5) / 8))
                    th = np.array([np.log(.2 / .1), np.log(.7 / .1), -3.0, a4, np.log(np.expm1(0.3 + 0.25 * k))])
                p = probs(gen, th, N, k)
                for si in range(40):
                    rows.append(dict(N=N, k=k, si=si, land=int(rng.choice(N + 1, p=p / p.sum()))))
        ev = evaluate(rows); res[gen] = dict(D_minus_S=ev["D-S"]["diff"], ci=ev["D-S"]["ci"])
    ok = res["D"]["ci"][0] > 0 and res["S"]["ci"][1] < 0
    return dict(res=res, ok=bool(ok))


if __name__ == "__main__":
    args = sys.argv[1:]; out = {}
    if "--control" in args:
        out["control"] = c = control(); print(f"control (synthetic; software check): {c}")
        args = [a for a in args if a != "--control"]
    for tag in args:
        d = json.load(open(f"/vlm/results/r3/s801_{tag}_grounding.json"))
        rows = [dict(N=r["N"], k=r["k"], si=r["si"], land=r["END"]["land"]) for r in d["rows"] if r["k"] in KS]
        ev = evaluate(rows); ev["rules"] = rules(ev); ev["model"] = d["model"]; out[tag] = ev
        print(f"== {tag} ({d['model']}) scenes {ev['n_scenes']} held-out LL " + " ".join(f"{m} {v:.1f}" for m, v in ev["total"].items()))
        for x in ("S", "K", "C"):
            print(f"   D-{x} {ev[f'D-{x}']['diff']:.1f} {np.round(ev[f'D-{x}']['ci'], 1).tolist()} per k " + " ".join(f"{k}:{v:.1f}" for k, v in ev[f"D-{x}"]["per_k"].items()))
        for k, p in ev["fit_all"]["D"].items():
            print(f"   D k{k}: " + " ".join(f"{a} {b:.2f}" for a, b in p.items()) + " | S: " + " ".join(f"{a} {b:.2f}" for a, b in ev["fit_all"]["S"][k].items()))
        print(f"   rules {ev['rules']}")
    json.dump(out, open("/vlm/results/r3/s803_analysis.json", "w"), indent=1, default=float)
