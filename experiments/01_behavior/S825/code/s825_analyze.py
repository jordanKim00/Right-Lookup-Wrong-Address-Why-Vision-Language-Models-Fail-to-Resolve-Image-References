"""S825 analysis (rules registered in knowledge/DECISIONS.md with the run, before any S825 data; predictions frozen in
research/ces/outputs/s825_predictions.json by exp/s825_fit.py).  Cells N in {6, 9}, k = 1..N.  Scene bootstrap, 4,000 draws, seed 0.
  I0  t1 >= .95; START(k) lands on k in >= .90 of scenes in every cell.
  R1  calibration (middle and near-end cells, 1 < k < N): for each cell and each candidate (target, one past, one step), |observed - predicted|
      <= .15; 'calibrated' if >= 70% of the checks pass.
  R2  information: mean log-probability of the observed END landings (1 < k < N) under the law minus under the baseline 'target + spread'
      (P(target) = the law's mean target weight over those cells, the rest uniform) >= .05 nats per landing with the interval above 0.
  R3  anchors: END(1) -> N and END(N) -> 1 each >= .80 in both N.
Usage: python s825_analyze.py TAG [TAG ...]"""
from __future__ import annotations

import json, sys
import numpy as np

B = 4000


def analyze(tag):
    d = json.load(open(f"/vlm/results/r3/s825_{tag}_newN.json")); R = d["rows"]; rng = np.random.default_rng(0)
    pred = json.load(open("/vlm/research/ces/outputs/s825_predictions.json"))["predictions"]
    rep = dict(tag=tag, model=d["model"], t1=d["t1_menu"], n=len(R))
    P = lambda rows, c, x: float(np.mean([r["land"][c] == x for r in rows]))
    st = {f"N{N}_k{k}": P(R, f"N{N}_k{k}_S", k) for N in (6, 9) for k in range(1, N + 1)}
    rep["I0"] = dict(start_min=min(st.values()), ok=bool((d["t1_menu"] or 0) >= .95 and min(st.values()) >= .90))
    checks = []; cells = {}
    for N in (6, 9):
        for k in range(2, N):
            j = N + 1 - k; s = k - int(np.sign(k - j)) if k != j else k - 1
            pr = pred[f"N{N}_k{k}"]; ob = {x: P(R, f"N{N}_k{k}_E", x) for x in range(0, N + 1)}
            row = {}
            for nm, pos in (("target", j), ("one_past", j - 1), ("one_step", s)):
                if 1 <= pos <= N:
                    ok = abs(ob[pos] - pr[pos - 1]) <= .15; checks.append(ok); row[nm] = dict(pos=pos, obs=ob[pos], pred=pr[pos - 1], ok=bool(ok))
            cells[f"N{N}_k{k}"] = row
    rep["R1"] = dict(cells=cells, share_ok=float(np.mean(checks)), calibrated=bool(np.mean(checks) >= .70))
    mid = [(N, k) for N in (6, 9) for k in range(2, N)]
    tw = float(np.mean([pred[f"N{N}_k{k}"][N - k] for N, k in mid]))      # law's target probability, mean over cells
    def gain(rows):
        g = []
        for r in rows:
            for N, k in mid:
                x = r["land"][f"N{N}_k{k}_E"]
                if x < 1:
                    continue
                pl = pred[f"N{N}_k{k}"][x - 1]; j = N + 1 - k
                pb = tw if x == j else (1 - tw) / (N - 1)
                g.append(np.log(pl) - np.log(pb))
        return float(np.mean(g)) if g else float("nan")
    gv = gain(R); v = []
    for _ in range(B):
        pick = rng.choice(len(R), len(R)); v.append(gain([R[i] for i in pick]))
    ci = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    rep["R2"] = dict(gain_nats=gv, ci=ci, informative=bool(gv >= .05 and ci[0] > 0))
    an = {f"N{N}": dict(first=P(R, f"N{N}_k1_E", N), last=P(R, f"N{N}_k{N}_E", 1)) for N in (6, 9)}
    rep["R3"] = dict(anchors=an, holds=bool(all(v_["first"] >= .80 and v_["last"] >= .80 for v_ in an.values())))
    rep["landing"] = {f"N{N}_k{k}_{f}": np.bincount([r["land"][f"N{N}_k{k}_{f}"] for r in R], minlength=N + 1).tolist() for N in (6, 9) for k in range(1, N + 1) for f in ("E", "S")}
    return rep


if __name__ == "__main__":
    allr = {}
    for tag in sys.argv[1:]:
        try:
            rep = analyze(tag)
        except FileNotFoundError:
            print(f"[{tag}] missing"); continue
        allr[tag] = rep
        print(f"== {tag} ({rep['model']}) t1 {rep['t1']} n {rep['n']} I0 {rep['I0']}")
        for key in ("R1", "R2", "R3"):
            print(f"   {key} {rep[key]}")
    json.dump(allr, open("/vlm/results/r3/s825_analysis.json", "w"), indent=1, default=float)
