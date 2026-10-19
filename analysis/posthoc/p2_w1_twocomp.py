"""Phase-2 W1 (post hoc, not registered; 10-17 00:3x): one two-component law for END answers, fitted on NATURAL runs only and asked to predict
the INTERVENTION runs (S830 / S831 E_EARLY_p, early lookup forced to p).
  P(x | a, j) = ws * Step(x | a, j) + we * End(x | j) + wu / N
  Step: offsets o = (x - a) * dir(j - a) from the early anchor a in the target direction, o in {-1, 0, 1, 2, 3}; when a = j the direction is
        +-1 with equal weight (the step moves even at the target).  End: offsets d = x - j in {-2, ..., 2} around the end-relative target.
  In natural runs the early anchor is the literal copy a = k (k >= 3; k = 1, 2 and k = N use other anchors and are left out).
Training: natural END(k) landings, N = 6-9, 3 <= k <= N - 1 (S824 matched-count cells, S825), plus the natural END(k) of S827-S831.
Test: S830 E_EARLY_p (identity cells) and S831 E_EARLY_p (non-identity N7 k5 j3; N9 k7 j3 is descriptive since its I0 failed).
Baselines on the test runs: 'no anchor' (the cell's own natural END distribution), 'copy' (answer = p), the law with the anchor left at the copy.
Usage: python3 exp/p2_w1_twocomp.py  ->  research/ces/outputs/p2_w1_twocomp.json"""
from __future__ import annotations

import collections, json
import numpy as np
from scipy.optimize import minimize

import os
V = os.environ.get("VLM", "/home/storage2/ai26269345/vlm")
OFF_S = [-1, 0, 1, 2, 3]; OFF_E = [-2, -1, 0, 1, 2]


def load_natural():
    obs = []                                                         # (N, a, j, x)
    d = json.load(open(f"{V}/results/r3/s824_s824_27b_count.json"))
    for r in d["rows"]:
        for N in (7, 8):
            for k in d["ks"]:
                if 3 <= k <= N - 1:
                    obs.append((N, k, N + 1 - k, r["land"][f"a{N}_s{N}_k{k}_E"]))
    d = json.load(open(f"{V}/results/r3/s825_s825_27b_newN.json"))
    for r in d["rows"]:
        for N in (6, 9):
            for k in range(3, N):
                w = r["land"][f"N{N}_k{k}_E"]
                x = r["labs"][str(N)].index(w) + 1 if isinstance(w, str) and w in r["labs"][str(N)] else 0
                obs.append((N, k, N + 1 - k, x))
    for f, key in (("s827_s827_27b_natural", "END"), ("s828_s828_27b_natural", "END"), ("s829_s829_27b_natural", "END"),
                   ("s830_s830_27b_anchor", "END"), ("s831_s831_27b_anchor", "END")):
        for r in json.load(open(f"{V}/results/r3/{f}.json"))["rows"]:
            N, k = r["N"], r["k"]
            if 3 <= k <= N - 1:
                obs.append((N, k, N + 1 - k, r["land"][key]))
    return [o for o in obs if o[3] >= 1]


def load_test():
    obs = []                                                         # (cell, N, a, j, x, scene, natural END answer of that scene)
    for f, cname in (("s830_s830_27b_anchor", "S830"), ("s831_s831_27b_anchor", "S831")):
        d = json.load(open(f"{V}/results/r3/{f}.json"))
        for r in d["rows"]:
            N, k, j = r["N"], r["k"], r["j"]
            for key, x in r["land"].items():
                if key.startswith("E_EARLY_p") and x >= 1:
                    obs.append((f"{cname}_N{N}k{k}", N, int(key.split("_p")[1]), j, x, r["si"], r["land"]["END"]))
    return obs


def unpack(th):
    s = np.exp(th[:5]); s = s / s.sum(); e = np.exp(th[5:10]); e = e / e.sum()
    w = np.exp(th[10:13]); w = w / w.sum()
    return s, e, w


def dist(N, a, j, th):
    s, e, w = unpack(th)
    p = np.zeros(N + 1)
    dirs = [np.sign(j - a)] if j != a else [1, -1]
    for dr in dirs:
        for o, so in zip(OFF_S, s):
            x = a + o * dr
            if 1 <= x <= N:
                p[x] += w[0] * so / len(dirs)
    for dd, ed in zip(OFF_E, e):
        x = j + dd
        if 1 <= x <= N:
            p[x] += w[1] * ed
    p[1:] += w[2] / N
    p[1:] = p[1:] / p[1:].sum()
    return p


def nll(th, obs):
    return -sum(np.log(max(dist(N, a, j, th)[x], 1e-9)) for N, a, j, x in obs)


def main():
    train = load_natural(); test = load_test()
    th0 = np.zeros(13)
    res = minimize(nll, th0, args=(train,), method="Nelder-Mead", options=dict(maxiter=20000, xatol=1e-4, fatol=1e-4))
    res = minimize(nll, res.x, args=(train,), method="Powell", options=dict(maxiter=20000))
    th = res.x; s, e, w = unpack(th)
    out = dict(train_n=len(train), test_n=len(test), nll_train=float(res.fun), step=dict(zip(map(str, OFF_S), s.tolist())),
               end=dict(zip(map(str, OFF_E), e.tolist())), weights=dict(step=float(w[0]), end=float(w[1]), uniform=float(w[2])))
    print(f"train n {len(train)}  nll/obs {res.fun / len(train):.3f}")
    print("  step offsets toward the target", {k_: round(v_, 3) for k_, v_ in out["step"].items()})
    print("  end offsets around the target ", {k_: round(v_, 3) for k_, v_ in out["end"].items()})
    print("  weights", {k_: round(v_, 3) for k_, v_ in out["weights"].items()})
    # natural-cell fit quality (per cell shares)
    cells = collections.defaultdict(list)
    for N, a, j, x in train:
        cells[(N, a)].append(x)
    fit_hits = []
    for (N, a), xs in sorted(cells.items()):
        p = dist(N, a, N + 1 - a, th); o = np.bincount(xs, minlength=N + 1) / len(xs)
        fit_hits.extend([abs(p[x] - o[x]) <= .15 for x in range(1, N + 1)])
    out["natural_fit_within_.15"] = float(np.mean(fit_hits))
    # test: intervention runs
    by = collections.defaultdict(list)
    for t in test:
        by[(t[0], t[2])].append(t)
    rows = []; lp = dict(law=[], law_copyanchor=[], nat=[], copy=[])
    for (cell, p_), ts in sorted(by.items()):
        N, j = ts[0][1], ts[0][3]
        k = int(cell.split("k")[1])
        pr = dist(N, p_, j, th); pr_c = dist(N, k, j, th)                    # law with the forced anchor / with the copy anchor
        nat = np.bincount([t[6] for t in ts if t[6] >= 1], minlength=N + 1)[1:] + .5; nat = np.concatenate([[0], nat / nat.sum()])
        ob = np.bincount([t[4] for t in ts], minlength=N + 1) / len(ts)
        for t in ts:
            x = t[4]
            lp["law"].append(np.log(max(pr[x], 1e-9))); lp["law_copyanchor"].append(np.log(max(pr_c[x], 1e-9)))
            lp["nat"].append(np.log(max(nat[x], 1e-9))); lp["copy"].append(np.log(.85 if x == p_ else .15 / (N - 1)))
        top = int(np.argmax(pr[1:])) + 1
        rows.append(dict(cell=cell, p=p_, pred={x: round(float(pr[x]), 3) for x in range(1, N + 1)}, obs={x: round(float(ob[x]), 3) for x in range(1, N + 1)},
                         within=float(np.mean([abs(pr[x] - ob[x]) <= .15 for x in range(1, N + 1)])), pred_mode=top, obs_mode=int(np.argmax(ob[1:])) + 1))
    out["test"] = rows
    out["test_logp"] = {k_: float(np.mean(v_)) for k_, v_ in lp.items()}
    out["test_within_.15"] = float(np.mean([r_["within"] for r_ in rows]))
    out["test_mode_agree"] = float(np.mean([r_["pred_mode"] == r_["obs_mode"] for r_ in rows]))
    print(f"natural fit within .15: {out['natural_fit_within_.15']:.3f}")
    print(f"TEST (interventions, never fitted): mean log p  law {out['test_logp']['law']:.3f} | law with copy anchor {out['test_logp']['law_copyanchor']:.3f} | "
          f"cell's natural {out['test_logp']['nat']:.3f} | copy {out['test_logp']['copy']:.3f}")
    print(f"  shares within .15: {out['test_within_.15']:.3f}   predicted mode = observed mode: {out['test_mode_agree']:.3f}")
    for r_ in rows:
        print(f"   {r_['cell']:12s} p {r_['p']}  pred mode {r_['pred_mode']} obs mode {r_['obs_mode']}  within {r_['within']:.2f}")
    json.dump(out, open(f"{V}/research/ces/outputs/p2_w1_twocomp.json", "w"), indent=1)


if __name__ == "__main__":
    main()
