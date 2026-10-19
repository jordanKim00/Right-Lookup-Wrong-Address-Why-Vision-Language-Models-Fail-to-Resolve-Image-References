"""S827 analysis (rules registered in knowledge/DECISIONS.md with the run, before any S827 data).  Cases N7 (k = 4) and N9 (k = 5), identity
references: j = k.  For an arm: dP(x) = P(arm -> x) - P(recipient natural -> x); neighbour mass nb = P(j - 1) + P(j + 1); logit margin
lm = logit(j) - max(logit(j - 1), logit(j + 1)) (two-form menu logits); internal margin im(row, layer) = block score(j) - max(block score(j - 1),
block score(j + 1)) (index-matching heads).  Scene bootstrap, 4,000 draws, seed 0.
  I0  t1 >= .95; NOOP_E lands as END and NOOP_S as START in >= .95 of scenes per case, 95th percentile of the largest menu-logit difference <= .5;
      START(k) lands on k in >= .90 per case.
  T1  (correction) an arm 'corrects the identity error' in a case if dP(j) >= .15 (interval above 0) and d nb <= -.15 (interval below 0) and
      the logit margin rises (interval above 0); tested for X_QT31 and X_B39 (and reported for every arm).
  T2  (internal signal) for a correcting arm: the answer row's L51 internal margin rises (interval above 0) and the boundary newline rows' (B1, B4
      mean) L51 internal margin rises (interval above 0) -> 'the internal error signal is corrected'.
  T3  (reverse) R_QT31 or R_B39 'produces the error in START' if dP(j) <= -.15 (interval below 0) and d nb >= .10 (interval above 0).
  T4  (transmission of the question-tail correction) X_QT31 raises the B1/B4 internal margin at L43, L47 or L51 (interval above 0).
  Generality: the verdicts per case, and whether T1-T4 hold in both cases.
Descriptive: every arm's landing distribution, logit margin and internal margins per row and layer; the components (mixers / MLPs).
Usage: python s827_analyze.py TAG [TAG ...]"""
from __future__ import annotations

import json, sys
import numpy as np

B = 4000


def analyze(tag):
    d = json.load(open(f"/vlm/results/r3/s827_{tag}_identity.json")); R = d["rows"]; rng = np.random.default_rng(0)
    rep = dict(tag=tag, model=d["model"], t1=d["t1_menu"], n=len(R))
    def boot(rows, f):
        v = []
        for _ in range(B):
            pick = rng.choice(len(rows), len(rows)); v.append(f([rows[i] for i in pick]))
        v = np.array(v, float); v = v[np.isfinite(v)]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else None
    P = lambda rows, arm, x: float(np.mean([r["land"][arm] == x for r in rows]))
    def lm(r, arm):
        L = r["mlog"][arm]; j = r["j"]
        return L[j - 1] - max(L[j - 2], L[j])
    def im(r, arm, row, li):
        q = r["qk"][arm].get(str(li))
        if q is None:
            return float("nan")
        s = q[row]; j = r["j"]
        return s[j - 1] - max(s[j - 2], s[j])
    rep["cases"] = {}
    for N, k in ((7, 4), (9, 5)):
        rows = [r for r in R if r["N"] == N]; j = k
        if not rows:
            continue
        p95 = lambda a1, a2: float(np.percentile([max(abs(x - y) for x, y in zip(r["mlog"][a1], r["mlog"][a2])) for r in rows], 95))
        i0 = dict(noopE=float(np.mean([r["land"]["NOOP_E"] == r["land"]["END"] for r in rows])), noopS=float(np.mean([r["land"]["NOOP_S"] == r["land"]["START"] for r in rows])),
                  p95E=p95("NOOP_E", "END"), p95S=p95("NOOP_S", "START"), start_acc=P(rows, "START", k), end_acc=P(rows, "END", k),
                  end_nb=P(rows, "END", j - 1) + P(rows, "END", j + 1))
        i0["ok"] = bool((d["t1_menu"] or 0) >= .95 and min(i0["noopE"], i0["noopS"]) >= .95 and max(i0["p95E"], i0["p95S"]) <= .5 and i0["start_acc"] >= .90)
        arms = [a for a in rows[0]["land"] if a not in ("END", "START", "NOOP_E", "NOOP_S")]
        def stats(a):
            base = "START" if a.startswith("R_") else "END"
            dPj = lambda q: P(q, a, j) - P(q, base, j)
            dnb = lambda q: (P(q, a, j - 1) + P(q, a, j + 1)) - (P(q, base, j - 1) + P(q, base, j + 1))
            dlm = lambda q: float(np.mean([lm(r, a) - lm(r, base) for r in q]))
            return dict(dPj=dPj(rows), ci_dPj=boot(rows, dPj), dnb=dnb(rows), ci_dnb=boot(rows, dnb), dlm=dlm(rows), ci_dlm=boot(rows, dlm), base=base)
        S = {a: stats(a) for a in arms}
        corrects = lambda a: bool(S[a]["dPj"] >= .15 and S[a]["ci_dPj"] and S[a]["ci_dPj"][0] > 0 and S[a]["dnb"] <= -.15 and S[a]["ci_dnb"] and S[a]["ci_dnb"][1] < 0
                                 and S[a]["ci_dlm"] and S[a]["ci_dlm"][0] > 0)
        t1 = {a: corrects(a) for a in ("X_QT31", "X_B39")}
        def dim(a, row, li):
            f = lambda q: float(np.nanmean([im(r, a, row, li) - im(r, "END", row, li) for r in q]))
            return f(rows), boot(rows, f)
        t2 = {}
        for a in [x for x, ok in t1.items() if ok]:
            ans = dim(a, 6, 51)
            bnl = (lambda q, a=a: float(np.nanmean([np.nanmean([im(r, a, 2, 51), im(r, a, 5, 51)]) - np.nanmean([im(r, "END", 2, 51), im(r, "END", 5, 51)]) for r in q])))
            bc = (bnl(rows), boot(rows, bnl))
            t2[a] = dict(answer_L51=ans, B1B4_L51=bc, corrected=bool(ans[1] and ans[1][0] > 0 and bc[1] and bc[1][0] > 0))
        t3 = {}
        for a in ("R_QT31", "R_B39"):
            s_ = S[a]; t3[a] = bool(s_["dPj"] <= -.15 and s_["ci_dPj"] and s_["ci_dPj"][1] < 0 and s_["dnb"] >= .10 and s_["ci_dnb"] and s_["ci_dnb"][0] > 0)
        t4 = {}
        for li in (43, 47, 51):
            f = lambda q, li=li: float(np.nanmean([np.nanmean([im(r, "X_QT31", 2, li), im(r, "X_QT31", 5, li)]) - np.nanmean([im(r, "END", 2, li), im(r, "END", 5, li)]) for r in q]))
            t4[f"L{li}"] = (f(rows), boot(rows, f))
        t4_ok = any(v_[1] and v_[1][0] > 0 for v_ in t4.values())
        rep["cases"][f"N{N}k{k}"] = dict(I0=i0, arms=S, T1=t1, T2=t2, T3=t3, T4=dict(margins=t4, holds=t4_ok),
                                        landing={a: np.bincount([r["land"][a] for r in rows], minlength=N + 1).tolist() for a in rows[0]["land"]})
    cs = rep["cases"]
    rep["generality"] = {key: all(any(cs[c][key].values()) if isinstance(cs[c][key], dict) and key != "T4" else cs[c]["T4"]["holds"] for c in cs)
                         for key in ("T1", "T3", "T4")} if cs else None
    return rep


if __name__ == "__main__":
    allr = {}
    for tag in sys.argv[1:]:
        try:
            rep = analyze(tag)
        except FileNotFoundError:
            print(f"[{tag}] missing"); continue
        allr[tag] = rep
        print(f"== {tag} ({rep['model']}) t1 {rep['t1']} n {rep['n']} generality {rep['generality']}")
        for c, v in rep["cases"].items():
            print(f"   {c} I0 {v['I0']}")
            print(f"   {c} T1 {v['T1']} T2 {v['T2']} T3 {v['T3']} T4 {v['T4']}")
            for a, s_ in v["arms"].items():
                print(f"      {a:10s} dPj {s_['dPj']:+.3f} dnb {s_['dnb']:+.3f} dlm {s_['dlm']:+.2f}")
    json.dump(allr, open("/vlm/results/r3/s827_analysis.json", "w"), indent=1, default=float)
