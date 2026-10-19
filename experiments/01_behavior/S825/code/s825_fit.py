"""S825 fit (pre-data for S825): an END error law fitted on N = 7 and N = 8 natural END landings (27B, clean readouts: the pooled clean runs of
research/ces/outputs/agent_math_1015/pool_end_kernel.json and S824's matched conditions a7s7 / a8s8), used to predict N = 6 and N = 9 before any
N = 6 / 9 run.  For 1 < k < N, j = N + 1 - k: P(p) = w_t [p = j] + w_o [p = j - 1] (one past the target toward the start) + w_s [p = k - sgn(k - j)]
(one step from the literal toward the target; for k = j the step toward the start, k - 1) + w_l [p = k] + w_u / N (spread); weights per class
c = 'near' (k in {2, N - 1}) or 'mid' (otherwise), fitted by maximum likelihood (softmax-parametrised, coinciding candidates add); anchors k = 1 ->
P(N), k = N -> P(1) fitted as one share each with the rest spread.  Output: research/ces/outputs/s825_predictions.json (predicted distributions
for N = 6, 9 and the fitted weights)."""
import json, numpy as np
from scipy.optimize import minimize
pk = json.load(open("/home/storage2/ai26269345/vlm/research/ces/outputs/agent_math_1015/pool_end_kernel.json"))["pooled"]
s824 = json.load(open("/home/storage2/ai26269345/vlm/results/r3/s824_analysis.json"))["s824_27b"]["landing"]
data = {}                                                            # (N, k) -> counts over positions 1..N (0 = not a picture dropped)
for N in (7, 8):
    for k in range(1, N + 1):
        c = np.zeros(N)
        key = f"N{N}_k{k}"
        if key in pk:
            c += np.array(pk[key][1:N + 1], float)
        sk = f"a{N}_s{N}_k{k}_E"
        if sk in s824:
            c += np.array(s824[sk][1:N + 1], float)
        if c.sum() > 0:
            data[(N, k)] = c
def cands(N, k):
    j = N + 1 - k; s = k - int(np.sign(k - j)) if k != j else k - 1
    return dict(t=j, o=j - 1, s=s, l=k)
def dist(N, k, w):                                                   # w: dict t, o, s, l, u (non-negative, sum 1)
    p = np.full(N, w["u"] / N)
    for nm, pos in cands(N, k).items():
        if 1 <= pos <= N:
            p[pos - 1] += w[nm]
    return p / p.sum()
def unpack(th):
    e = np.exp(th - th.max()); e /= e.sum()
    return dict(zip(("t", "o", "s", "l", "u"), e))
fit = {}
for cls in ("near", "mid"):
    cells = [(N, k) for (N, k) in data if 1 < k < N and ((k in (2, N - 1)) == (cls == "near"))]
    nll = lambda th: -sum(float((data[c] * np.log(dist(c[0], c[1], unpack(th)) + 1e-9)).sum()) for c in cells)
    r = minimize(nll, np.zeros(5), method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-6, fatol=1e-6))
    fit[cls] = dict(weights=unpack(r.x), nll=float(r.fun), cells=[f"N{N}k{k}" for N, k in cells])
anch = {}
for k_is in ("first", "last"):
    cs = [(N, k) for (N, k) in data if (k == 1 if k_is == "first" else k == N)]
    hit = sum(data[c][(c[0] - 1) if k_is == "first" else 0] for c in cs); tot = sum(data[c].sum() for c in cs)
    anch[k_is] = float(hit / tot)
pred = {}
for N in (6, 9):
    for k in range(1, N + 1):
        if k == 1:
            p = np.full(N, (1 - anch["first"]) / (N - 1)); p[N - 1] = anch["first"]
        elif k == N:
            p = np.full(N, (1 - anch["last"]) / (N - 1)); p[0] = anch["last"]
        else:
            p = dist(N, k, fit["near" if k in (2, N - 1) else "mid"]["weights"])
        pred[f"N{N}_k{k}"] = [round(float(x), 4) for x in p]
out = dict(fit=fit, anchors=anch, predictions=pred, n_cells_used={f"N{N}k{k}": float(data[(N, k)].sum()) for N, k in data})
json.dump(out, open("/home/storage2/ai26269345/vlm/research/ces/outputs/s825_predictions.json", "w"), indent=1)
print(json.dumps(fit, indent=0)[:1500]); print("anchors", anch)
for N in (6, 9):
    for k in range(2, N):
        print(f"N{N} k{k} pred", pred[f"N{N}_k{k}"])
