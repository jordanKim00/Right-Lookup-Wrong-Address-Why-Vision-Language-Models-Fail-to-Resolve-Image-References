"""S847 descriptive check (post hoc, NOT registered): do the layer 59 / 63 positional margins add predictive power for the answer shares?
Split-half by scene (fit on scenes 0-14, test on 15-29 and the reverse), logistic map from positional margins (z[k] - max other; B4 and answer
rows) to answering k, features from layers 51 / 55 only vs 51-63; cell calibration (case x arm, |pred - obs| <= .15) on the held-out half.
Usage: python3 exp/p4_s847_cvmap.py (inside the research container: scipy)  ->  research/ces/outputs/s847_cvmap.json"""
import json
import numpy as np
from scipy.optimize import minimize

V = "/home/storage2/ai26269345/vlm"
d = json.load(open(f"{V}/results/r3/s847_s847_27b_close.json")); R = d["rows"]
ARMS = sorted(R[0]["land"].keys())
def zm(r, a, L, row):
    z = r["qk"][a][L][row]; k = r["k"]
    return z[k - 1] - max(z[i] for i in range(len(z)) if i != k - 1)
def X(rows, layers):
    return np.array([[1.0] + [zm(r, a, L, row) for L in layers for row in (5, 9)] for r in rows for a in ARMS])
def Y(rows):
    return np.array([float(r["land"][a] == r["k"]) for r in rows for a in ARMS])
def fit(x, y):
    f = lambda w: float(np.sum(np.logaddexp(0, x @ w) - y * (x @ w))) + .5e-3 * float(w[1:] @ w[1:])
    return minimize(f, np.zeros(x.shape[1]), method="BFGS").x
out = {}
for name, layers in (("51_55", ("51", "55")), ("51_63", ("51", "55", "59", "63"))):
    cal = []
    for tr, te in ((lambda r: r["si"] < 15, lambda r: r["si"] >= 15), (lambda r: r["si"] >= 15, lambda r: r["si"] < 15)):
        rtr = [r for r in R if tr(r)]; rte = [r for r in R if te(r)]
        w = fit(X(rtr, layers), Y(rtr))
        for (N, k, j) in d["cases"]:
            rows = [r for r in rte if r["N"] == N and r["k"] == k]
            p = 1 / (1 + np.exp(-(X(rows, layers) @ w))); y = Y(rows)
            for ai, a in enumerate(ARMS):
                idx = list(range(ai, len(y), len(ARMS)))
                cal.append(abs(p[idx].mean() - y[idx].mean()) <= .15)
    out[name] = float(np.mean(cal))
json.dump(dict(note="post hoc split-half, not registered", calibration_15=out), open(f"{V}/research/ces/outputs/s847_cvmap.json", "w"), indent=1)
print(out)
