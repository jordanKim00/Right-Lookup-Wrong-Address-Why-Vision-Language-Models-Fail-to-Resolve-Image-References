"""S871 analysis (rules fixed before the data; knowledge/DECISIONS.md S871 entry).  Inputs results/r3/s871_<tag>_writer.json and _hpre.npz.
Per E run (N 8, END(k) pinned at p; k 3-6, p 3-5): writer effect dL = menu logits(E) - menu logits(E_BANDR) over pictures 1..8, centred over
pictures; cell mean c(k, p) from the training scenes; per-run residual R = dL - c(k, p) (what a constant writer per cell cannot carry).
Training scenes = the first half, test scenes = the second half (fixed); scene bootstrap 4,000 over test scenes (seed 871).
Predictors of R (fitted on training runs only; no post-writer activation and no logit of a test run is an input):
  A  E's L47 per-head block scores (rows B1 / B4 / answer, 24 heads, 8 blocks, centred per head) - the address scores (S869 F4's input);
  B  the residual at the input of layer 46 at the ten read rows (10 x 5120);
  C  B + the residual at the question tail (the ordinal token .. before '?');
  each a linear kernel ridge on centred features, the penalty chosen by 4-fold scene cross-validation inside the training half from
  {1e-3, 1e-2, 1e-1, 1, 10} x trace(K) / n.
R^2(X) = 1 - sum ||R - R_hat_X||^2 / sum ||R||^2 over test runs.
I0: E_BANDR answers stay at the pin p in >= .80 of runs (the writer's END part removed); pin reach (E, R: L43 / L47 argmax at B1 / B4 / answer = p)
  >= .95; R answers p >= .90; t1 >= .95; START(3..5) >= .95; every E run has its saved pre-writer state.  Nothing is read if I0 fails.
W1: R^2(B) - R^2(A) >= .15 with the interval above 0 -> 'the full pre-writer residual carries writer-effect information beyond the address
  scores'; else 'not'.
W2: the top two singular directions of R (training runs) explain >= .70 of its variance -> 'the per-run writer variation is low-dimensional';
  else 'not'.
W3: answer prediction on test runs: logits_hat = logits(E_BANDR) + c(k, p) + R_hat_B; mean NLL (softmax over the eight pictures, 1% uniform
  mixed) of the actual E answer, minus that of logits(E_BANDR) + c(k, p): improvement >= .10 nat with the interval above 0 -> 'the pre-writer
  state predicts per-run answers beyond a constant writer effect'; else 'not'.
W4: R^2(B) >= .20 with the lower bound > 0 -> 'the pre-writer residual predicts the per-run writer variation'; else 'not'.
Descriptive: R^2(C), R^2 per k and pin, test accuracy of the three answer rules, the variance of R relative to dL, 'same address,
different writer' pairs (test pairs of the same (k, p) whose A features have cosine >= .95: the spread of their R).
Usage: python3 s871_analyze.py TAG"""
from __future__ import annotations

import collections, json, sys
import numpy as np

TAG = sys.argv[1]
J = json.load(open(f"/vlm/results/r3/s871_{TAG}_writer.json"))
H = np.load(f"/vlm/results/r3/s871_{TAG}_hpre.npz")
OUT = f"/vlm/results/r3/s871_analysis_{TAG}.json"
rng = np.random.default_rng(871)
NB = 4000
rows = J["rows"]; S = len(rows); PINS = J["pins"]; KS = J["ks"]
SP = rows[0]["subpass"]
E47, E43 = str(J["early"][-1]), str(J["early"][-2])
TRAIN = set(range(S // 2)); TEST = [s for s in range(S) if s not in TRAIN]


def ref_of(armkey):
    return f"{armkey.split('_')[0]}@{SP[armkey]}"


res = dict(tag=TAG, n_scenes=S, n_runs=J["n_runs"], train_scenes=len(TRAIN), test_scenes=len(TEST))
RUNS = []
missing = 0
for si, r in enumerate(rows):
    for p in PINS:
        for k in KS:
            arm = f"E{k}p{p}_E_BANDR"; ek = ref_of(arm)
            hk = f"{si}_E{k}p{p}"
            if hk not in H:
                missing += 1; continue
            lE = np.array(r["mlog"][ek], dtype=float); lB = np.array(r["mlog"][arm], dtype=float)
            dL = (lE - lE.mean()) - (lB - lB.mean())
            z = np.asarray(r["z47"][ek], dtype=float); z = z - z.mean(-1, keepdims=True)
            h = H[hk].astype(np.float32)
            RUNS.append(dict(si=si, k=k, p=p, dL=dL, lB=lB, x=r["land"][ek], xB=r["land"][arm], A=z.ravel(), B=h[:10].ravel(), C=h.ravel()))
# ---------------------------------------------------------------- I0
stayB = float(np.mean([q["xB"] == q["p"] for q in RUNS]))
reach = []
for p in PINS:
    for rk in [f"E{k}p{p}" for k in KS] + [f"Rp{p}"]:
        for r in rows:
            for L in (E43, E47):
                q = np.asarray(r["qk"][rk][L]); reach += [int(np.argmax(q[i])) + 1 == p for i in (2, 5, 9)]
racc = float(np.mean([r["land"][f"Rp{p}"] == p for r in rows for p in PINS]))
st = float(np.mean([r["land"][f"START{p}"] == p for r in rows for p in PINS]))
I0 = dict(bandr_stays=bool(stayB >= .80), pin_reach=bool(np.mean(reach) >= .95), R_acc=bool(racc >= .90), t1=bool((J["t1_menu"] or 0) >= .95),
          START_acc=bool(st >= .95), hpre_complete=bool(missing == 0))
res["I0"] = dict(**I0, all=bool(all(I0.values())), bandr_stay_value=stayB, pin_reach_value=float(np.mean(reach)), R_acc_value=racc, t1_value=J["t1_menu"],
                 START_acc_value=st, hpre_missing=missing)
ok = res["I0"]["all"]
# ---------------------------------------------------------------- cell means and residuals
tr = [q for q in RUNS if q["si"] in TRAIN]; te = [q for q in RUNS if q["si"] not in TRAIN]
cell = {c: np.mean([q["dL"] for q in tr if (q["k"], q["p"]) == c], 0) for c in {(q["k"], q["p"]) for q in RUNS}}
for q in RUNS:
    q["R"] = q["dL"] - cell[(q["k"], q["p"])]
Rtr = np.stack([q["R"] for q in tr]); Rte = np.stack([q["R"] for q in te])


def kridge(feat):
    Xtr = np.stack([q[feat] for q in tr]).astype(np.float64); Xte = np.stack([q[feat] for q in te]).astype(np.float64)
    mu = Xtr.mean(0); Xtr = Xtr - mu; Xte = Xte - mu; ym = Rtr.mean(0)
    K = Xtr @ Xtr.T; scale = np.trace(K) / len(K)
    # inner 4-fold scene CV for the penalty
    trs = np.array([q["si"] for q in tr]); scs = sorted(set(trs.tolist())); folds = [set(scs[i::4]) for i in range(4)]
    best, bl = None, None
    for lam in (1e-3, 1e-2, 1e-1, 1.0, 10.0):
        err = 0.
        for f in folds:
            a = np.array([s not in f for s in trs]); b = ~a
            Ka = K[np.ix_(a, a)]; al = np.linalg.solve(Ka + lam * scale * np.eye(a.sum()), Rtr[a] - Rtr[a].mean(0))
            pr = K[np.ix_(b, a)] @ al + Rtr[a].mean(0); err += np.sum((Rtr[b] - pr) ** 2)
        if best is None or err < best:
            best, bl = err, lam
    al = np.linalg.solve(K + bl * scale * np.eye(len(K)), Rtr - ym)
    return (Xte @ Xtr.T) @ al + ym, bl


PRED = {}; LAM = {}
for nm in ("A", "B", "C"):
    PRED[nm], LAM[nm] = kridge(nm)
te_si = np.array([q["si"] for q in te])
BY = collections.defaultdict(list)
for i, s in enumerate(te_si):
    BY[int(s)].append(i)
def r2_fn(nm):
    def f(ix):
        idx = [i for s in ix for i in BY[TEST[s]]]
        return float(1 - np.sum((Rte[idx] - PRED[nm][idx]) ** 2) / np.sum(Rte[idx] ** 2))
    return f
def bt(fn):
    n = len(TEST); est = fn(np.arange(n)); bs = [fn(rng.integers(0, n, n)) for _ in range(NB)]
    lo, hi = np.percentile(bs, [2.5, 97.5]); return dict(est=float(est), lo=float(lo), hi=float(hi))
R2 = {nm: bt(r2_fn(nm)) for nm in PRED}
w1 = bt(lambda ix: r2_fn("B")(ix) - r2_fn("A")(ix))
res["R2"] = R2; res["penalty"] = LAM
res["W1"] = dict(diff=w1, verdict=(("the full pre-writer residual carries writer-effect information beyond the address scores" if w1["est"] >= .15 and w1["lo"] > 0 else "not")
                                   if ok else "not read"))
U, Sv, Vt = np.linalg.svd(Rtr - Rtr.mean(0), full_matrices=False); ev = Sv ** 2 / np.sum(Sv ** 2)
res["W2"] = dict(explained=ev[:4].tolist(), verdict=(("the per-run writer variation is low-dimensional" if ev[:2].sum() >= .70 else "not") if ok else "not read"))
def nll(logit, x):
    if not 1 <= x <= len(logit):
        return np.nan
    p = np.exp(logit - logit.max()); p /= p.sum(); p = .99 * p + .01 / len(p); return -np.log(p[x - 1])
def w3_fn(ix):
    idx = [i for s in ix for i in BY[TEST[s]]]
    a = [nll(te[i]["lB"] - te[i]["lB"].mean() + cell[(te[i]["k"], te[i]["p"])] + PRED["B"][i], te[i]["x"]) for i in idx]
    b = [nll(te[i]["lB"] - te[i]["lB"].mean() + cell[(te[i]["k"], te[i]["p"])], te[i]["x"]) for i in idx]
    return float(np.nanmean(b) - np.nanmean(a))
w3 = bt(w3_fn)
res["W3"] = dict(nll_gain=w3, verdict=(("the pre-writer state predicts per-run answers beyond a constant writer effect" if w3["est"] >= .10 and w3["lo"] > 0 else "not")
                                       if ok else "not read"))
res["W4"] = dict(R2_B=R2["B"], verdict=(("the pre-writer residual predicts the per-run writer variation" if R2["B"]["est"] >= .20 and R2["B"]["lo"] > 0 else "not")
                                        if ok else "not read"))
# ---------------------------------------------------------------- descriptive
acc = {}
for nm, add in (("constant", None), ("B", "B"), ("oracle_dL", "oracle")):
    hits = []
    for i, q in enumerate(te):
        base = q["lB"] - q["lB"].mean() + cell[(q["k"], q["p"])]
        lg = base if add is None else (base + PRED["B"][i] if add == "B" else q["lB"] - q["lB"].mean() + q["dL"])
        hits.append(int(np.argmax(lg)) + 1 == q["x"])
    acc[nm] = float(np.mean(hits))
pairs = []
for c in cell:
    idx = [i for i, q in enumerate(te) if (q["k"], q["p"]) == c]
    for a_ in range(len(idx)):
        for b_ in range(a_ + 1, len(idx)):
            qa, qb = te[idx[a_]], te[idx[b_]]
            cs = float(qa["A"] @ qb["A"] / (np.linalg.norm(qa["A"]) * np.linalg.norm(qb["A"]) + 1e-9))
            if cs >= .95:
                pairs.append(float(np.linalg.norm(qa["R"] - qb["R"])))
res["descriptive"] = dict(test_accuracy=acc, var_R_over_dL=float(np.sum(Rte ** 2) / np.sum(np.stack([q["dL"] for q in te]) ** 2)),
                          R2_by_k={str(k): float(1 - np.sum((Rte[[i for i, q in enumerate(te) if q["k"] == k]] - PRED["B"][[i for i, q in enumerate(te) if q["k"] == k]]) ** 2)
                                                 / np.sum(Rte[[i for i, q in enumerate(te) if q["k"] == k]] ** 2)) for k in KS},
                          same_address_pairs=dict(n=len(pairs), median_R_distance=(float(np.median(pairs)) if pairs else None),
                                                  median_R_norm=float(np.median(np.linalg.norm(Rte, axis=1)))))
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"]["all"], R2={k: round(v["est"], 3) for k, v in R2.items()}, W1=res["W1"]["verdict"], W2=res["W2"]["verdict"],
                      W3=res["W3"]["verdict"], W4=res["W4"]["verdict"], acc=acc), indent=1))
print("->", OUT)
