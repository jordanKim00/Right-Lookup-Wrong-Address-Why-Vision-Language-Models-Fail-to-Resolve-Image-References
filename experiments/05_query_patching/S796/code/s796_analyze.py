"""S796 analysis (rules registered in knowledge/DECISIONS.md with the run, before any S796 data).
m(X): q = 4, the share of scenes whose answer lands on image 4 (target = literal); q = 5, the share landing on the literal image 5 (the image the
start-relative frame selects), with the share on the target image 3 reported beside it.
  I0   NOOPF and NOOPQ each reproduce E's landed image in >= 95% of (scene, q) cells; else no verdict.
  M1   the frame read: m(FR) - m(E) >= .15 with the 95% paired scene-bootstrap interval above 0, per q; else M2 / M3 have no verdict for that q.
  M2   mediation through the queries, under one recompute: MF = (m(FRNQ) - m(FRQB)) / (m(FRNQ) - m(NOOPQ)).  Supported when MF >= .5 and the
       interval of m(FRNQ) - m(FRQB) lies above 0; refuted (the frame reaches the answer by other routes) when MF < .25; partial otherwise.
  M3   sufficiency of the queries: SF = (m(QS) - m(NOOPQ)) / (m(FRNQ) - m(NOOPQ)) >= .5.
  M4   the answer follows the picture: over arms E, S, FR, FRQB, QS and swap blocks p, among cells whose unswapped answer lands on p, the swapped
       answer is the new picture's word in >= 80% (pooled); among cells whose unswapped answer lands elsewhere, <= 20%.
Descriptive: t1 (free-generation validity), block shares at the READ layers (all heads; index-matching heads at their layers).
Usage: python s796_analyze.py TAG [TAG ...]"""
from __future__ import annotations

import json, sys
import numpy as np

B = 4000


def boot(si, fn, rng):
    u = np.unique(si); out = []
    for _ in range(B):
        pick = rng.choice(u, len(u))
        idx = np.concatenate([np.flatnonzero(si == s) for s in pick])
        out.append(fn(idx))
    return np.array(out, float)


def analyze(tag):
    d = json.load(open(f"/vlm/results/r3/s796_{tag}_mediate.json"))
    R = d["rows"]; rep = dict(tag=tag, model=d["model"], readout=d["readout"], t1_menu=d["t1_menu"], t1_n=d["t1_n"], P1=d["P1"], READ=d["READ"])
    nat = lambda r, X: r["nat"]["res"][X]["land"]
    cells = len(R)
    agreeF = np.mean([nat(r, "NOOPF") == nat(r, "E") for r in R]); agreeQ = np.mean([nat(r, "NOOPQ") == nat(r, "E") for r in R])
    agreeFR = np.mean([nat(r, "FRNQ") == nat(r, "FR") for r in R])
    rep["I0"] = dict(noopf=agreeF, noopq=agreeQ, frnq_vs_fr=agreeFR, cells=cells, ok=bool(agreeF >= .95 and agreeQ >= .95))
    rng = np.random.default_rng(0)
    ARMS = ["E", "S", "NOOPQ", "NOOPF", "FR", "FRNQ", "FRQB", "QS"]
    rep["q"] = {}
    for q in sorted({r["q"] for r in R}):
        Rq = [r for r in R if r["q"] == q]
        si = np.array([r["si"] for r in Rq]); key = q                    # q = 4: image 4; q = 5: the literal image 5
        L = {X: np.array([nat(r, X) for r in Rq]) for X in ARMS}
        m = lambda X, idx=None: float(np.mean((L[X] if idx is None else L[X][idx]) == key))
        tgt = {X: float(np.mean(L[X] == (8 - q))) for X in ARMS}
        rq = dict(n=len(Rq), m={X: m(X) for X in ARMS}, on_target={X: tgt[X] for X in ARMS})
        d1 = m("FR") - m("E"); b1 = boot(si, lambda i: m("FR", i) - m("E", i), rng)
        rq["M1"] = dict(d=d1, ci=[float(np.percentile(b1, 2.5)), float(np.percentile(b1, 97.5))])
        rq["M1"]["ok"] = bool(d1 >= .15 and rq["M1"]["ci"][0] > 0)
        den = m("FRNQ") - m("NOOPQ")
        d2 = m("FRNQ") - m("FRQB"); b2 = boot(si, lambda i: m("FRNQ", i) - m("FRQB", i), rng)
        MF = d2 / den if den > 0 else None
        bmf = boot(si, lambda i: ((m("FRNQ", i) - m("FRQB", i)) / (m("FRNQ", i) - m("NOOPQ", i))) if (m("FRNQ", i) - m("NOOPQ", i)) > 0 else np.nan, rng)
        ok = np.isfinite(bmf)
        rq["M2"] = dict(blocked=d2, blocked_ci=[float(np.percentile(b2, 2.5)), float(np.percentile(b2, 97.5))], effect_recomputed=den, MF=MF,
                        MF_ci=([float(np.percentile(bmf[ok], 2.5)), float(np.percentile(bmf[ok], 97.5))] if ok.any() else None),
                        MF_defined=float(ok.mean()))
        if not rq["M1"]["ok"] or MF is None:
            rq["M2"]["verdict"] = "no verdict (no frame effect to mediate)"
        elif MF >= .5 and rq["M2"]["blocked_ci"][0] > 0:
            rq["M2"]["verdict"] = "supported: the frame read reaches the answer through the READ-layer queries"
        elif MF < .25:
            rq["M2"]["verdict"] = "refuted: the frame read reaches the answer by other routes"
        else:
            rq["M2"]["verdict"] = "partial"
        SF = (m("QS") - m("NOOPQ")) / den if den > 0 else None
        rq["M3"] = dict(SF=SF, verdict=("no verdict" if (not rq["M1"]["ok"] or SF is None) else ("supported" if SF >= .5 else "not supported")))
        rep["q"][str(q)] = rq
    # M4: picture following
    fol, leak, per = [], [], {}
    for r in R:
        for p, c in r["swap"].items():
            p = int(p)
            for X in ["E", "S", "FR", "FRQB", "QS"]:
                hit = int(c["res"][X]["ans"] == r["nw"])
                (fol if nat(r, X) == p else leak).append(hit)
                per.setdefault(f"{X}|p{p}|q{r['q']}", []).append((int(nat(r, X) == p), hit))
    rep["M4"] = dict(follow=float(np.mean(fol)) if fol else None, n_follow=len(fol), leak=float(np.mean(leak)) if leak else None, n_leak=len(leak),
                     per={k: dict(n_on=sum(a for a, _ in v), follow=(float(np.mean([h for a, h in v if a])) if any(a for a, _ in v) else None),
                                  leak=(float(np.mean([h for a, h in v if not a])) if any(not a for a, _ in v) else None)) for k, v in per.items()})
    rep["M4"]["verdict"] = ("supported" if fol and leak and rep["M4"]["follow"] >= .8 and rep["M4"]["leak"] <= .2 else "not supported")
    # descriptive block shares (natural cells)
    sh = {}
    for r in R:
        for X, layers in r["nat"]["share"].items():
            for li, dd in layers.items():
                for kind, vec in dd.items():
                    sh.setdefault(f"q{r['q']}|{X}|{li}|{kind}", []).append(vec)
    rep["share"] = {k: np.mean(v, 0).round(3).tolist() for k, v in sh.items()}
    if not rep["I0"]["ok"]:
        rep["verdict"] = "instrument invalid (I0)"
    return rep


if __name__ == "__main__":
    allr = {}
    for tag in sys.argv[1:]:
        try:
            rep = analyze(tag)
        except FileNotFoundError:
            print(f"[{tag}] missing"); continue
        allr[tag] = rep
        print(f"== {tag} ({rep['model']}) readout {rep['readout']} t1 {rep['t1_menu']:.3f} (n {rep['t1_n']}) P1 {rep['P1'][0]}-{rep['P1'][-1]} READ {rep['READ']}")
        i0 = rep["I0"]; print(f"  I0 NOOPF {i0['noopf']:.3f} NOOPQ {i0['noopq']:.3f} (FRNQ=FR {i0['frnq_vs_fr']:.3f}) -> {'ok' if i0['ok'] else 'INVALID'}")
        for q, rq in rep["q"].items():
            mm = " ".join(f"{X} {rq['m'][X]:.2f}" for X in rq["m"])
            print(f"  q={q} n={rq['n']} m: {mm}")
            print(f"     on target: " + " ".join(f"{X} {rq['on_target'][X]:.2f}" for X in rq["on_target"]))
            print(f"     M1 d {rq['M1']['d']:+.3f} {rq['M1']['ci']} ok {rq['M1']['ok']}")
            m2 = rq["M2"]; print(f"     M2 blocked {m2['blocked']:+.3f} {m2['blocked_ci']} MF {m2['MF']} {m2['MF_ci']} -> {m2['verdict']}")
            print(f"     M3 SF {rq['M3']['SF']} -> {rq['M3']['verdict']}")
        m4 = rep["M4"]; print(f"  M4 follow {m4['follow']} (n {m4['n_follow']}) leak {m4['leak']} (n {m4['n_leak']}) -> {m4['verdict']}")
    json.dump(allr, open("/vlm/results/r3/s796_analysis.json", "w"), indent=1, default=float)
