"""S833 analysis (rules registered in knowledge/DECISIONS.md with the run, before any S833 data).  Cells (N, k, j): identity N7 (4, 4), N9 (5, 5);
non-identity N7 (5, 3), N9 (7, 3).  Late picture mass m(run) = index-matching-head mean picture attention mass of B1, B4 and the answer row, mean
of L51 and L55; w(run) = (m(run) - m(START)) / (m(END) - m(START)) per scene (0 = START-like, 1 = END-like).  Scene bootstrap, 4,000 draws, seed 0.
  I0  t1 >= .95; WNOOP = END in >= .90 of scenes with the 95th percentile of the largest menu-logit difference <= .5; START(k) -> k >= .90.
  R1  O_FR (every cell) and O_ORD_FR (non-identity): mean w <= .30 with the interval's upper end < .50 -> 'the frame rebind turns off the late
      re-reading'.
  R2  identity cells: O_FR -> j >= .85 -> 'the frame write alone repairs the identity failure'.
  R3  non-identity cells: O_ORD lands on j + sign(k - j) (one step from the written j toward the reflected target k) in >= .30 and exceeds the share
      on j - sign(k - j) by >= .15 (interval above 0) -> 'the late step moves the written address toward the reflected target'; and mean
      w(O_ORD) >= .60 -> 'the late re-reading stays on'.
  R4  non-identity cells: O_ORD_FR -> j >= .85 -> 'both writes repair'.
  R5  non-identity cells: O_FR -> k >= .70 -> 'with the START frame the ordinal is read forward'.
Descriptive: landings, w per run, P(j), P(k) per run.  Usage: python s833_analyze.py TAG"""
from __future__ import annotations

import json, sys
import numpy as np

B = 4000
CASES = ((7, 4, 4), (9, 5, 5), (7, 5, 3), (9, 7, 3))


def analyze(tag):
    d = json.load(open(f"/vlm/results/r3/s833_{tag}_r3mode.json")); R = d["rows"]; rng = np.random.default_rng(0)
    look = d["look"]; L1 = str(min(look, key=lambda l: abs(l - 51))); L2 = str(min(look, key=lambda l: abs(l - 55)))
    rep = dict(tag=tag, model=d["model"], t1=d["t1_menu"], n=len(R), Lb=d.get("Lb"), cases={})
    m = lambda r, run: float(np.mean([np.mean([r["mass"][run][L][i] for i in (2, 5, 9)]) for L in (L1, L2)]))
    def wv(r, run):
        mE, mS = m(r, "END"), m(r, "START")
        return (m(r, run) - mS) / (mE - mS) if abs(mE - mS) > 1e-6 else float("nan")
    def boot(rows, f):
        v = []
        for _ in range(B):
            pick = rng.choice(len(rows), len(rows)); v.append(f([rows[i] for i in pick]))
        v = np.array(v, float); v = v[np.isfinite(v)]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else None
    P = lambda rows, run, x: float(np.mean([r["land"][run] == x for r in rows]))
    for N, k, j in CASES:
        rows = [r for r in R if r["N"] == N and r["k"] == k]
        if not rows:
            continue
        p95 = float(np.percentile([max(abs(x - y) for x, y in zip(r["mlog"]["WNOOP"], r["mlog"]["END"])) for r in rows], 95))
        i0 = dict(wnoop_agree=float(np.mean([r["land"]["WNOOP"] == r["land"]["END"] for r in rows])), wnoop_p95=p95, start_acc=P(rows, "START", k))
        i0["ok"] = bool((d["t1_menu"] or 0) >= .95 and i0["wnoop_agree"] >= .90 and p95 <= .5 and i0["start_acc"] >= .90)
        c = dict(I0=i0)
        runs = [x for x in ("END", "START", "WNOOP", "O_FR", "O_ORD", "O_ORD_FR") if x in rows[0]["land"]]
        c["w"] = {x: (float(np.nanmean([wv(r, x) for r in rows])), boot(rows, lambda q, x=x: float(np.nanmean([wv(r, x) for r in q])))) for x in runs}
        c["P_j"] = {x: P(rows, x, j) for x in runs}; c["P_k"] = {x: P(rows, x, k) for x in runs}
        c["landing"] = {x: np.bincount([r["land"][x] for r in rows], minlength=N + 1).tolist() for x in runs}
        offs = ["O_FR"] + (["O_ORD_FR"] if j != k else [])
        c["R1"] = dict(runs={x: c["w"][x] for x in offs},
                       verdict=("the frame rebind turns off the late re-reading" if all(c["w"][x][0] <= .30 and c["w"][x][1] and c["w"][x][1][1] < .50 for x in offs) else "not shown"))
        if j == k:
            c["R2"] = dict(P=c["P_j"]["O_FR"], verdict=("the frame write alone repairs the identity failure" if c["P_j"]["O_FR"] >= .85 else "not shown"))
        else:
            sg = 1 if k > j else -1
            f3 = lambda q: P(q, "O_ORD", j + sg) - P(q, "O_ORD", j - sg)
            v3, ci3 = f3(rows), boot(rows, f3)
            c["R3"] = dict(toward=P(rows, "O_ORD", j + sg), away=P(rows, "O_ORD", j - sg), ci=ci3, w=c["w"]["O_ORD"],
                           step=("the late step moves the written address toward the reflected target" if (P(rows, "O_ORD", j + sg) >= .30 and v3 >= .15 and ci3 and ci3[0] > 0) else "not shown"),
                           reread=("the late re-reading stays on" if c["w"]["O_ORD"][0] >= .60 else "not shown"))
            c["R4"] = dict(P=c["P_j"]["O_ORD_FR"], verdict=("both writes repair" if c["P_j"]["O_ORD_FR"] >= .85 else "not shown"))
            c["R5"] = dict(P=c["P_k"]["O_FR"], verdict=("with the START frame the ordinal is read forward" if c["P_k"]["O_FR"] >= .70 else "not shown"))
        rep["cases"][f"N{N}k{k}j{j}"] = c
    return rep


if __name__ == "__main__":
    tag = sys.argv[1]
    rep = analyze(tag)
    print(f"== {tag} ({rep['model']}) t1 {rep['t1']} n {rep['n']} Lb {rep['Lb']}")
    for c, v in rep["cases"].items():
        print(f"   {c} I0 {v['I0']}")
        print(f"   {c} w {{{', '.join(f'{x}: {y[0]:.2f}' for x, y in v['w'].items())}}}  P_j {v['P_j']}  P_k {v['P_k']}")
        for key in ("R1", "R2", "R3", "R4", "R5"):
            if key in v:
                print(f"   {c} {key} {v[key]}")
        print(f"   {c} landing {v['landing']}")
    json.dump(rep, open("/vlm/results/r3/s833_analysis.json" if tag == "s833_27b" else f"/vlm/results/r3/s833_analysis_{tag}.json", "w"), indent=1, default=float)
