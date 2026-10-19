"""S803p (registered in knowledge/DECISIONS.md before any S807 / S808 27B data): predict unseen interventions from a model fitted only on
natural end-relative answers of Qwen3.5-27B.
Model M (behavioural approximation, not a claim about an internal linear mixture; s803_mixture.py model D): per k,
    P(b | k, N, s) = eps [b = 0] + (1 - eps) (c [b = k] + e G_N(b; N + 1 - k + mu + s, sigma) + u / N),
fitted on the natural END(k) answers at N = 7 (S786, S791, S793, S797) and, when present, S801's natural END answers at N = 6, 7, 8.
A source hypothesis H gives every S807 arm X a shift s_H(X, k) of the end-relative centre (CODE6 / CNT6: -1, CODE8 / CNT8 / BOTH8: +1 when that
source is read at that k; 0 otherwise):
  H_code        code read at every k              H_count       count read at every k
  H_nc_fcount   code near (k = 2), count far      H_nc_fnone    code near, nothing far
  H_n6_fnone    near: only the lowered code (CODE6) moves (FINDINGS §867), nothing far
  H_none        nothing moves
  H_mixed_half  code and count read with equal weight (each moves the centre half a block; BOTH8 a full block) - added before S807 data
Score: the log-likelihood of the observed S807 answers of CNT6, CNT8, CODE6, CODE8, BOTH8 (all k) under each H with M fixed.  Rule P1: the best H
is named only if it beats the second by >= 5 nats; otherwise the top two are reported as tied.
S808: the smeared address (SMEAR, equal thirds on j - 1, j, j + 1) is predicted to give the window shares of a natural uniform smear, the natural
END(5) answers at N = 7 (2 / 3 / 4 = .307 / .260 / .360, 150 answers pooled from S786, S791, S793, S797): rule P2 = every observed SMEAR share within .15.
Usage (docker): python /vlm/exp/s803p_predict.py [--s801 s801_27b] [--s807 s807_27b] [--s808 s808_27b]"""
from __future__ import annotations

import argparse, json, os, sys
import numpy as np

sys.path.insert(0, "/vlm/exp")
import s803_mixture as M

R3 = "/vlm/results/r3"
KS = (2, 3, 5, 6)
HYP = {"H_code": lambda X, k: {"CODE6": -1, "CODE8": 1, "BOTH8": 1}.get(X, 0),
       "H_count": lambda X, k: {"CNT6": -1, "CNT8": 1, "BOTH8": 1}.get(X, 0),
       "H_nc_fcount": lambda X, k: ({"CODE6": -1, "CODE8": 1, "BOTH8": 1} if k == 2 else {"CNT6": -1, "CNT8": 1, "BOTH8": 1}).get(X, 0),
       "H_nc_fnone": lambda X, k: ({"CODE6": -1, "CODE8": 1, "BOTH8": 1}.get(X, 0) if k == 2 else 0),
       "H_n6_fnone": lambda X, k: (-1 if (k == 2 and X == "CODE6") else 0),
       "H_none": lambda X, k: 0,
       # amended 2026-10-15 06:1x, before any S807 data (S787 27B I1 with the count stated: all codes +2 moved answers by about -1, half of
       # the -2 a pure count reading predicts): both sources read with equal weight, each moving the centre by half a block
       "H_mixed_half": lambda X, k: {"CODE6": -.5, "CODE8": .5, "CNT6": -.5, "CNT8": .5, "BOTH8": 1}.get(X, 0)}
ARMS = ("CNT6", "CNT8", "CODE6", "CODE8", "BOTH8")


def natural_rows(s801):
    rows = []
    for f, get in (("s730_s786_27b_stagewise.json", lambda r: r["land_END"]), ("s791_s791_27b_chain.json", lambda r: r["land_END"]),
                   ("s793_s793_27b_transplant.json", lambda r: r["land"]["END"]), ("s797_s797_27b_writer.json", lambda r: r["land"]["END"])):
        for r in json.load(open(f"{R3}/{f}"))["rows"]:
            rows.append(dict(N=7, k=r["k"], land=int(get(r))))
    if s801 and os.path.exists(f"{R3}/s801_{s801}_grounding.json"):
        for r in json.load(open(f"{R3}/s801_{s801}_grounding.json"))["rows"]:
            rows.append(dict(N=r["N"], k=r["k"], land=int(r["END"]["land"])))
    return rows


def fit_M(rows):
    th = {}
    for k in KS:
        cells = [(N, k, M.counts(rows, N, k)) for N in sorted({r["N"] for r in rows if r["k"] == k})]
        th[k] = M.fit("D", cells)
    return th


def probs_shift(th, N, k, s):
    t = np.array(th, float).copy()
    mu = 4 * np.tanh(t[3]) + s                                                    # move the centre by s blocks
    t[3] = np.arctanh(np.clip(mu / 4, -0.999, 0.999))
    return M.probs("D", t, N, k)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--s801", default="s801_27b"); ap.add_argument("--s807", default="s807_27b"); ap.add_argument("--s808", default="s808_27b")
    a = ap.parse_args(); rep = {}
    nat = natural_rows(a.s801); TH = fit_M(nat)
    rep["fit"] = {str(k): M.params("D", TH[k]) for k in KS}; rep["n_natural"] = len(nat)
    print(f"model M fitted on {len(nat)} natural END answers: " + " | ".join(f"k{k}: " + " ".join(f"{x} {v:.2f}" for x, v in rep['fit'][str(k)].items()) for k in KS))
    p807 = f"{R3}/s807_{a.s807}_source.json"
    if os.path.exists(p807):
        R = json.load(open(p807))["rows"]; LL = {}
        for h, f in HYP.items():
            LL[h] = float(sum(np.log(probs_shift(TH[r["k"]], 7, r["k"], f(X, r["k"]))[int(r["land"][X])]) for r in R for X in ARMS if r["k"] in KS))
        order = sorted(LL, key=LL.get, reverse=True)
        rep["S807"] = dict(LL=LL, best=order[0], margin=LL[order[0]] - LL[order[1]], P1=(order[0] if LL[order[0]] - LL[order[1]] >= 5 else f"tie {order[0]} / {order[1]}"))
        print("S807 log-likelihood by hypothesis: " + " · ".join(f"{h} {v:.1f}" for h, v in sorted(LL.items(), key=lambda t: -t[1])) + f" -> P1 {rep['S807']['P1']}")
    else:
        print("S807: no data yet")
    p808 = f"{R3}/s808_{a.s808}_smear.json"
    if os.path.exists(p808):
        R = json.load(open(p808))["rows"]; L = np.array([r["land"]["SMEAR"] for r in R]); J = np.array([r["j"] for r in R])
        obs = [float(np.mean(L == J - 1)), float(np.mean(L == J)), float(np.mean(L == J + 1))]
        pred = [.307, .260, .360]
        rep["S808"] = dict(observed=obs, predicted=pred, P2=bool(all(abs(o - p) <= .15 for o, p in zip(obs, pred))))
        print(f"S808 SMEAR shares j-1 / j / j+1: observed {np.round(obs, 2).tolist()} predicted {pred} -> P2 {rep['S808']['P2']}")
    else:
        print("S808: no data yet")
    json.dump(rep, open(f"{R3}/s803p_prediction.json", "w"), indent=1, default=float)
