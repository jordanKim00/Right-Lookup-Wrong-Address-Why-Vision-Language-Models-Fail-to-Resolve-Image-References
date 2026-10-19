"""S801 analysis (rules registered in knowledge/DECISIONS.md with the run, before any S801 data; amended before data to cover both ends).
Among cells a question answers correctly (unswapped answer on j), follow(T) = share whose answer becomes the new picture's word when the picture
at j is replaced; change(C) = share whose answer leaves j when a picture elsewhere (c not in {j, k}) is replaced.  Groups: near the end
(k in {1, 2}); middle (3 <= j <= N - 2, k >= 3, not identity); start end (k in {N - 1, N}, targets 2 and 1); identity cells (k = (N + 1) / 2) apart.
  I0  t1 >= .95 and START follow(T) >= .90 (pooled): the instrument reads content at a stated address.
  G1  near the end: END follow(T) >= .80.
  G2  middle: END follow(T) <= .50 and at least .30 below START follow(T) on the same cells, cell-bootstrap interval above 0.
  G3  middle: END change(C) exceeds START change(C) by >= .10, interval above 0.
  U   the far end is sharp again (H5, boundary-anchored lookup): acc(END, k = N) - acc(END, middle) >= .20 with the interval above 0, and END
      follow(T) for k = N >= .80.
Descriptive: every rate per (N, k) with its n.
Usage: python s801_analyze.py TAG [TAG ...]"""
from __future__ import annotations

import json, sys
import numpy as np

B = 4000


def analyze(tag):
    d = json.load(open(f"/vlm/results/r3/s801_{tag}_grounding.json")); R = d["rows"]; rng = np.random.default_rng(0)
    ident = lambda r: r["k"] == (r["N"] + 1) / 2
    def rates(rows, nm):
        cor = [r for r in rows if r[nm]["land"] == r["j"]]
        fT = [r[f"{nm}|T"]["ans"] == r["nw"] for r in cor]; cC = [r[f"{nm}|C"]["land"] != r["j"] for r in cor]
        return dict(n=len(rows), acc=float(np.mean([r[nm]["land"] == r["j"] for r in rows])) if rows else None, n_correct=len(cor),
                    follow_T=float(np.mean(fT)) if fT else None, change_C=float(np.mean(cC)) if cC else None)
    rep = dict(tag=tag, model=d["model"], t1=d["t1_menu"])
    cells = {}
    for (N, k) in sorted({(r["N"], r["k"]) for r in R}):
        rr = [r for r in R if r["N"] == N and r["k"] == k]
        cells[f"{N}|{k}"] = dict(identity=bool(ident(rr[0])), END=rates(rr, "END"), START=rates(rr, "START"))
    rep["cells"] = cells
    pool = lambda pred, nm: rates([r for r in R if pred(r)], nm)
    rep["START_all"] = pool(lambda r: True, "START")
    rep["I0"] = bool((d["t1_menu"] or 0) >= .95 and (rep["START_all"]["follow_T"] or 0) >= .90)
    near = pool(lambda r: r["k"] in (1, 2), "END"); rep["near_END"] = near
    rep["G1"] = bool(near["follow_T"] is not None and near["follow_T"] >= .80)
    mid = lambda r: (3 <= r["j"] <= r["N"] - 2) and r["k"] >= 3 and not ident(r)
    far = [r for r in R if mid(r)]
    fe, fs = rates(far, "END"), rates(far, "START"); rep["middle_END"], rep["middle_START"] = fe, fs
    def boot(rows_, f):
        v = [f([rows_[i] for i in rng.integers(0, len(rows_), len(rows_))]) for _ in range(B)]
        v = np.array([x for x in v if x is not None and np.isfinite(x)])
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else None
    dfol = lambda rows: (lambda a, b: (b - a) if (a is not None and b is not None) else None)(rates(rows, "END")["follow_T"], rates(rows, "START")["follow_T"])
    dchg = lambda rows: (lambda a, b: (a - b) if (a is not None and b is not None) else None)(rates(rows, "END")["change_C"], rates(rows, "START")["change_C"])
    rep["G2"] = dict(diff=dfol(far) if far else None, ci=boot(far, dfol) if far else None)
    rep["G2"]["ok"] = bool(fe["follow_T"] is not None and fe["follow_T"] <= .50 and rep["G2"]["diff"] is not None and rep["G2"]["diff"] >= .30
                           and rep["G2"]["ci"] and rep["G2"]["ci"][0] > 0)
    rep["G3"] = dict(diff=dchg(far) if far else None, ci=boot(far, dchg) if far else None)
    rep["G3"]["ok"] = bool(rep["G3"]["diff"] is not None and rep["G3"]["diff"] >= .10 and rep["G3"]["ci"] and rep["G3"]["ci"][0] > 0)
    se = [r for r in R if r["k"] == r["N"]]
    rep["start_end_END"] = rates(se, "END")
    if se and far:
        both = se + far
        def dU(rows):
            a_ = [r for r in rows if r["k"] == r["N"]]; b_ = [r for r in rows if mid(r)]
            return (rates(a_, "END")["acc"] - rates(b_, "END")["acc"]) if (a_ and b_) else None
        rep["U"] = dict(diff=dU(both), ci=boot(both, dU))
        rep["U"]["ok"] = bool(rep["U"]["diff"] is not None and rep["U"]["diff"] >= .20 and rep["U"]["ci"] and rep["U"]["ci"][0] > 0
                              and (rep["start_end_END"]["follow_T"] or 0) >= .80)
    return rep


if __name__ == "__main__":
    allr = {}
    for tag in sys.argv[1:]:
        try:
            rep = analyze(tag)
        except FileNotFoundError:
            print(f"[{tag}] missing"); continue
        allr[tag] = rep
        print(f"== {tag} ({rep['model']}) t1 {rep['t1']} I0 {rep['I0']} G1 {rep['G1']} G2 {rep['G2']} G3 {rep['G3']} U {rep.get('U')}")
        print(f"   near END {rep['near_END']} | middle END {rep['middle_END']} | middle START {rep['middle_START']} | k = N END {rep['start_end_END']}")
        for c, v in rep["cells"].items():
            print(f"   {c}{' (identity)' if v['identity'] else ''}: END {v['END']} | START {v['START']}")
    json.dump(allr, open("/vlm/results/r3/s801_analysis.json", "w"), indent=1, default=float)
