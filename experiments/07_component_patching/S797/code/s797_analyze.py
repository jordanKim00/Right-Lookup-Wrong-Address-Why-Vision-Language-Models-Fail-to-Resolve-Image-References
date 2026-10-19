"""S797 analysis (rules registered in knowledge/DECISIONS.md with the run, before any S797 data).
acc(X): share of (scene, k) cells whose answer lands on the target j; gap G = acc(START) - acc(END).
  I0   NOOP reproduces END's landed image in >= 95% of cells, and t1 >= .95; else no verdict.
  W1   the writing band carries the address: RF(ALL) = (acc(E<-S:ALL) - acc(END)) / G >= .8.
  W2   the writer: RF(X) for X in {DN, MLP, FA}; X writes the address if RF(X) >= .5 and RF(X) exceeds every other single component by >= .2;
       "distributed" if no single component reaches .5 while W1 holds; "not localized to 32-35" if W1 fails (amended before data: the address may
       be written before 32, after 35, or by interactions across layers, so W1 failing does not by itself place the writer upstream; the
       reader-layer shares of each arm are then read to see whether the exchange moved the query at all).
  W3   necessity of the writer named by W2 (or ALL if distributed): S<-E drops START's accuracy by >= .30 and, in the cells END answers wrongly,
       reproduces END's wrong answer in >= .5.
  W4   (descriptive) the frame-read band: RF(FB); block shares at the reader layers per arm.
Paired scene bootstrap (4,000 draws) for every RF and drop.
Usage: python s797_analyze.py TAG [TAG ...]"""
from __future__ import annotations

import json, sys
import numpy as np

B = 4000


def analyze(tag):
    d = json.load(open(f"/vlm/results/r3/s797_{tag}_writer.json"))
    R = d["rows"]; si = np.array([r["si"] for r in R]); us = np.unique(si); rng = np.random.default_rng(0)
    arms = sorted({a for r in R for a in r["land"]})
    L = {a: np.array([r["land"].get(a, -1) for r in R]) for a in arms}
    J = np.array([r["j"] for r in R])
    acc = lambda a, i=None: float(np.mean((L[a] if i is None else L[a][i]) == (J if i is None else J[i])))
    def boot(f):
        v = []
        for _ in range(B):
            pick = rng.choice(us, len(us)); idx = np.concatenate([np.flatnonzero(si == s) for s in pick]); v.append(f(idx))
        v = np.array(v, float); v = v[np.isfinite(v)]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else None
    rep = dict(tag=tag, model=d["model"], readout=d["readout"], t1=d["t1_menu"], n=len(R), acc={a: acc(a) for a in arms})
    noop = float(np.mean(L["NOOP"] == L["END"]))
    rep["I0"] = dict(noop=noop, ok=bool(noop >= .95 and (d["t1_menu"] or 0) >= .95))
    G = acc("START") - acc("END")
    rf = lambda a, i=None: ((acc(a, i) - acc("END", i)) / (acc("START", i) - acc("END", i))) if (acc("START", i) - acc("END", i)) > 0 else np.nan
    rep["G"] = G
    rep["RF"] = {a.split(":")[1]: dict(rf=rf(a), ci=boot(lambda i, a=a: rf(a, i))) for a in arms if a.startswith("E<-S:")}
    rep["W1"] = bool(rep["RF"].get("ALL", {}).get("rf", 0) >= .8)
    singles = {x: rep["RF"][x]["rf"] for x in ("DN", "MLP", "FA") if x in rep["RF"]}
    writer = None
    for x, v in singles.items():
        if v >= .5 and all(v - w >= .2 for y, w in singles.items() if y != x):
            writer = x
    rep["W2"] = writer if writer else ("distributed" if rep["W1"] else "not localized to 32-35")
    tgt = writer if writer in ("DN", "MLP") else "ALL"
    wrong = L["END"] != J
    if f"S<-E:{tgt}" in L:
        drop = acc("START") - acc(f"S<-E:{tgt}")
        repro = float(np.mean(L[f"S<-E:{tgt}"][wrong] == L["END"][wrong])) if wrong.any() else None
        rep["W3"] = dict(arm=tgt, drop=drop, drop_ci=boot(lambda i: acc("START", i) - acc(f"S<-E:{tgt}", i)), reproduce=repro,
                         ok=bool(drop >= .30 and repro is not None and repro >= .5))
    rep["per_k"] = {str(k): {a: float(np.mean(L[a][np.array([r["k"] for r in R]) == k] == J[np.array([r["k"] for r in R]) == k])) for a in arms}
                    for k in sorted({r["k"] for r in R})}
    sh = {}
    for r in R:
        for a, layers in r["share"].items():
            for li, dd in layers.items():
                for kind, vec in dd.items():
                    sh.setdefault(f"{a}|{li}|{kind}|tgt", []).append(vec[r["j"] - 1]); sh.setdefault(f"{a}|{li}|{kind}|lit", []).append(vec[r["k"] - 1])
    rep["share"] = {k_: round(float(np.mean(v_)), 3) for k_, v_ in sh.items()}
    return rep


if __name__ == "__main__":
    allr = {}
    for tag in sys.argv[1:]:
        try:
            rep = analyze(tag)
        except FileNotFoundError:
            print(f"[{tag}] missing"); continue
        allr[tag] = rep
        print(f"== {tag} ({rep['model']}) readout {rep['readout']} t1 {rep['t1']} n {rep['n']}  I0 {rep['I0']}")
        print("   acc " + " ".join(f"{a} {v:.2f}" for a, v in rep["acc"].items()))
        print(f"   G {rep['G']:.3f}  RF " + " ".join(f"{x} {v['rf']:.2f} {v['ci']}" for x, v in rep["RF"].items()))
        print(f"   W1 {rep['W1']}  W2 {rep['W2']}  W3 {rep.get('W3')}")
    json.dump(allr, open("/vlm/results/r3/s797_analysis.json", "w"), indent=1, default=float)
