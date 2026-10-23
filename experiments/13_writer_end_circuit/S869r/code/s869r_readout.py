"""S869R (registered analysis of stored data; knowledge/DECISIONS.md S869R entry, rules fixed before this analysis was computed) - late readout
sufficiency: can the head-specific late attention scores predict the image each run selects, and does the head-mean readout lose that
information?  Data: S862 G natural END runs (N 6-9, every k, 24 scenes); intervention generalisation on S868 and S865 arm runs.
Per run, per image i: f_i = [z_{L,row,h}(i) - mean_i' z_{L,row,h}(i')] over L in {51, 55, 59, 63}, rows B1 / B4 / answer, 24 heads (288).
  R0a: x_hat = the L47 head-mean argmax anchor a;  R0b: a - s(k) (s = 1 for k >= 3);  R1: argmax_i of the matcher-head mean late score
  (mean over layers and rows; no training);  R2: softmax_i(w . f_i), w shared over images, L2 1e-2, full-batch gradient descent (fixed
  procedure);  R3: R2 + per (L, row, h) softmax_i shares (576);  random 6-head R2 (20 draws).
Splits: 4-fold scene cross-validation (every run predicted by a model not trained on its scene), leave-one-k-out, leave-one-N-out.
Runs whose answer is not a shown picture are excluded (counted).  Scene bootstrap 4,000 (seed 8690).
Usage: python3 s869r_readout.py"""
from __future__ import annotations

import collections, json
import numpy as np

R3D = "/vlm/results/r3"
OUT = f"{R3D}/s869r_analysis_s869r_27b.json"
rng = np.random.default_rng(8690)
NB = 4000
LATE = ["51", "55", "59", "63"]


def load_runs(path, keyspec):
    J = json.load(open(path)); rows = J["rows"]; lm = J["late_mean"]; E47 = str(J["early"][-1])
    runs = []
    for si, r in enumerate(rows):
        for key, N, k, tag in keyspec(r):
            if key not in r["land"]:
                continue
            x = r["land"][key]
            z = np.stack([np.asarray(r["zl"][key][L], dtype=float) for L in LATE])          # [4, 3, 24, N]
            zm = np.mean([np.asarray(r["zl"][key][L], dtype=float)[:, lm[L]].mean(1).mean(0) for L in LATE], 0)
            q = np.asarray(r["qk"][key][E47]); am = [int(np.argmax(q[i])) + 1 for i in (2, 5, 9)]
            c = collections.Counter(am).most_common(); a = c[0][0] if c[0][1] >= 2 else am[2]
            runs.append(dict(src=path.split("/")[-1], si=si, N=N, k=k, tag=tag, x=x, z=z, zm=zm, a=a))
    return runs


def feats(z, heads=None, comp=False):
    zz = z if heads is None else z[:, :, heads, :]
    zc = zz - zz.mean(-1, keepdims=True)
    F = zc.reshape(-1, zz.shape[-1]).T                                                     # [N, F]
    if comp:
        sm = np.exp(zz - zz.max(-1, keepdims=True)); sm = sm / sm.sum(-1, keepdims=True)
        F = np.concatenate([F, sm.reshape(-1, zz.shape[-1]).T], 1)
    return F


def stack(runs, heads=None, comp=False):
    Fs = [feats(r["z"], heads, comp) for r in runs]; D = Fs[0].shape[1]
    F = np.zeros((len(runs), 9, D)); M = np.zeros((len(runs), 9), dtype=bool)
    for i, f in enumerate(Fs):
        F[i, :len(f)] = f; M[i, :len(f)] = True
    return F, M


def train(runs, heads=None, comp=False, l2=1e-2, it=400, lr=.5):
    F, M = stack(runs, heads, comp); y = np.array([r["x"] - 1 for r in runs])
    mu = F[M].mean(0); sd = F[M].std(0) + 1e-6; F = (F - mu) / sd; F[~M] = 0
    w = np.zeros(F.shape[-1]); R = len(runs); ar = np.arange(R)
    for _ in range(it):
        s = F @ w; s[~M] = -1e9; s = s - s.max(1, keepdims=True); p = np.exp(s) * M; p /= p.sum(1, keepdims=True)
        g = l2 * w + (np.einsum("ri,rid->d", p, F) - F[ar, y].sum(0)) / R
        w -= lr * g
    return dict(w=w, mu=mu, sd=sd, heads=heads, comp=comp)


def predict(m, r):
    F = (feats(r["z"], m["heads"], m["comp"]) - m["mu"]) / m["sd"]; s = F @ m["w"]
    p = np.exp(s - s.max()); return p / p.sum()


def nll(p, x, N):
    q = .99 * p + .01 / N; return -np.log(q[x - 1])


S862 = load_runs(f"{R3D}/s862_s862_27b_predict.json",
                 lambda r: [(f"G{N}_END{k}", N, k, "nat") for N in (6, 7, 8, 9) for k in range(1, N + 1)])
excl = sum(1 for r in S862 if not 1 <= r["x"] <= r["N"])
S862 = [r for r in S862 if 1 <= r["x"] <= r["N"]]
SC = sorted({r["si"] for r in S862})
res = dict(n_runs=len(S862), excluded_other=excl)
# ---------------------------------------------------------------- out-of-fold predictions
def oof(split, heads=None, comp=False):
    pred = {}
    for test_keys in split:
        tr = [r for r in S862 if not test_keys(r)]; te = [r for r in S862 if test_keys(r)]
        m = train(tr, heads, comp)
        for r in te:
            pred[id(r)] = predict(m, r)
    return pred
folds = [set(SC[i::4]) for i in range(4)]
scene_split = [(lambda r, f=f: r["si"] in f) for f in folds]
k_split = [(lambda r, k=k: r["k"] == k) for k in range(1, 10)]
N_split = [(lambda r, n=n: r["N"] == n) for n in (6, 7, 8, 9)]
P2 = oof(scene_split); P3 = oof(scene_split, comp=True); P2k = oof(k_split); P2N = oof(N_split)
# cell-distribution baseline (scene folds)
Pc = {}
for f in folds:
    tr = [r for r in S862 if r["si"] not in f]
    cnt = collections.defaultdict(lambda: None)
    for r in tr:
        key = (r["N"], r["k"])
        if cnt[key] is None:
            cnt[key] = np.full(r["N"], .5)
        cnt[key][r["x"] - 1] += 1
    for r in S862:
        if r["si"] in f:
            c = cnt[(r["N"], r["k"])]; Pc[id(r)] = c / c.sum()
def acc_fn(pred_of):
    by = collections.defaultdict(list)
    for r in S862:
        by[r["si"]].append(int(pred_of(r) == r["x"]))
    return lambda ix: float(np.mean([v for s in ix for v in by[SC[s]]]))
def nll_fn(prob_of):
    by = collections.defaultdict(list)
    for r in S862:
        by[r["si"]].append(nll(prob_of(r), r["x"], r["N"]))
    return lambda ix: float(np.mean([v for s in ix for v in by[SC[s]]]))
def bt(fn):
    S = len(SC); est = fn(np.arange(S)); bs = [fn(rng.integers(0, S, S)) for _ in range(NB)]
    lo, hi = np.percentile(bs, [2.5, 97.5]); return dict(est=float(est), lo=float(lo), hi=float(hi))
PRED = dict(R0a=lambda r: r["a"], R0b=lambda r: r["a"] - (1 if r["k"] >= 3 else 0), R1=lambda r: int(np.argmax(r["zm"])) + 1,
            R2=lambda r: int(np.argmax(P2[id(r)])) + 1, R3=lambda r: int(np.argmax(P3[id(r)])) + 1,
            R2_kout=lambda r: int(np.argmax(P2k[id(r)])) + 1, R2_Nout=lambda r: int(np.argmax(P2N[id(r)])) + 1,
            cell_mode=lambda r: int(np.argmax(Pc[id(r)])) + 1)
res["accuracy"] = {m: bt(acc_fn(f)) for m, f in PRED.items()}
res["NLL"] = dict(R2=bt(nll_fn(lambda r: P2[id(r)])), R3=bt(nll_fn(lambda r: P3[id(r)])), cell=bt(nll_fn(lambda r: Pc[id(r)])))
v1 = bt(lambda ix: acc_fn(PRED["R2"])(ix) - acc_fn(PRED["R1"])(ix))
res["V1"] = dict(diff=v1, verdict=("head-mean late scores lose decision-relevant information" if v1["est"] >= .10 and v1["lo"] > 0
                                   else "the head-mean readout retains most of it"))
a2 = res["accuracy"]["R2"]["est"]
res["V2"] = dict(scene=a2, k_out=res["accuracy"]["R2_kout"]["est"], N_out=res["accuracy"]["R2_Nout"]["est"],
                 verdict=("the head-wise readout is a general (k- and N-invariant) decoder"
                          if res["accuracy"]["R2_kout"]["est"] >= a2 - .10 and res["accuracy"]["R2_Nout"]["est"] >= a2 - .10 else "condition-specific"))
v3 = bt(lambda ix: nll_fn(lambda r: Pc[id(r)])(ix) - nll_fn(lambda r: P2[id(r)])(ix))
res["V3"] = dict(nll_gain=v3, verdict=("late head-wise states predict individual runs beyond cell-level behaviour" if v3["est"] >= .10 and v3["lo"] > 0 else "not"))
# random 6-head subsets (scene folds, descriptive)
rh = []
for d in range(20):
    hs = sorted(rng.choice(24, 6, replace=False).tolist()); Pr = oof(scene_split, heads=hs)
    rh.append(float(np.mean([int(np.argmax(Pr[id(r)])) + 1 == r["x"] for r in S862])))
res["random6_R2_accuracy"] = dict(mean=float(np.mean(rh)), min=float(np.min(rh)), max=float(np.max(rh)))
# ---------------------------------------------------------------- V4: intervention generalisation
mfull = train(S862)
INT = []
INT += load_runs(f"{R3D}/s868_s868_27b_nearend.json",
                 lambda r: [(f"N{N}_END{k}_{arm}", N, k, arm) for N in (8, 9) for k in (1, 2, 3) for arm in ("PIN_NM1", "PIN_TGT", "TGT_BANDR", "EMASK", "LMASK")])
INT += load_runs(f"{R3D}/s865_s865_27b_xlayout.json",
                 lambda r: [(f"X{N}_{nm}{nd}_{k}", N, k, nm) for N in (7, 8, 9) for k in (4, 5) for nd in (7, 8, 9) if nd != N for nm in ("U_FROM", "Q_FROM")])
INT = [r for r in INT if 1 <= r["x"] <= r["N"]]
acc_int = collections.defaultdict(list)
for r in INT:
    acc_int[r["tag"]].append(int(np.argmax(predict(mfull, r)) + 1 == r["x"]))
acc_int_all = float(np.mean([v for vs in acc_int.values() for v in vs]))
res["V4"] = dict(acc_interventions=acc_int_all, by_arm={k: float(np.mean(v)) for k, v in acc_int.items()}, n=len(INT), natural_heldout=a2,
                 R1_on_interventions=float(np.mean([int(np.argmax(r["zm"]) + 1 == r["x"]) for r in INT])),
                 verdict=("the head-wise late state is a sufficient statistic for the answer under interventions" if acc_int_all >= a2 - .10 else "correlational"))
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(acc={k: round(v["est"], 3) for k, v in res["accuracy"].items()}, NLL={k: round(v["est"], 3) for k, v in res["NLL"].items()},
                      V1=res["V1"]["verdict"], V1d=res["V1"]["diff"], V2=res["V2"], V3=res["V3"]["verdict"], V4=[res["V4"]["verdict"], res["V4"]["acc_interventions"], res["V4"]["R1_on_interventions"]],
                      random6=res["random6_R2_accuracy"]), indent=1))
print("->", OUT)
