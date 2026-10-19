"""S705b analysis, frozen with the pre-registration (knowledge/DECISIONS.md, 2026-10-09) before any S705b data.
I0: P(t+1 | PTR_W) >= .40 pooled over t and NOOP == NATIVE in >= .95 of runs.
On NATIVE-wrong items whose START(t) is correct: PTR_RESCUE P(t | PTR) >= .60; COND_CARRIES P(t | PTR_C) >= .5 P(t | PTR);
MATCHER_SUFFICIENT P(t | PTR_M) >= .5 P(t | PTR); QK_ROUTE P(t | Q_ONLY) >= .5 P(t | QG) and P(t | G_ONLY) < .5 P(t | QG); GATE_ROUTE mirrored.
Usage: python s720_analyze.py TAG   (S720 = S705b logic on Qwen3.5-4B / 9B records)
"""
import json, sys
import numpy as np

GE = lambda a, b: a >= b - 1e-9
ARMS = ("NATIVE", "NOOP", "PTR", "PTR_C", "PTR_W", "PTR_M", "Q_ONLY", "G_ONLY", "QG")


def analyze(tag):
    d = json.load(open(f"/vlm/results/r3/s720_{tag}_select.json")); rows = d["rows"]
    print(f"[{tag}] n {len(rows)} · PTR {d['PTR']} · MATCHER {d['MATCHER']}")
    mloc = lambda ll, t: ll[t - 1] - max(x for j, x in enumerate(ll) if j != t - 1)
    for t in (2, 3, 4):
        cells = []
        for arm in ARMS:
            tgt = t + 1 if arm == "PTR_W" else t
            p = np.mean([r["t"][str(t)][arm]["land"] == tgt for r in rows]); m = np.mean([mloc(r["t"][str(t)][arm]["ll"], t) for r in rows])
            cells.append(f"{arm} {p:.2f}({m:+.1f})")
        print(f"[{tag}]  t={t}: " + " | ".join(cells))
    selw = float(np.mean([r["t"][str(t)]["PTR_W"]["land"] == t + 1 for r in rows for t in (2, 3, 4)]))
    noop = float(np.mean([r["t"][str(t)]["NOOP"]["land"] == r["t"][str(t)]["NATIVE"]["land"] for r in rows for t in (2, 3, 4)]))
    fail = [(r, t) for r in rows for t in (2, 3, 4) if r["t"][str(t)]["NATIVE"]["land"] != t and r["t"][str(t)]["START"]["land"] == t]
    P = lambda arm: float(np.mean([r["t"][str(t)][arm]["land"] == t for r, t in fail])) if fail else float("nan")
    i0 = GE(selw, .40) and GE(noop, .95)
    ptr, ptrc, ptrm, qo, go, qg = P("PTR"), P("PTR_C"), P("PTR_M"), P("Q_ONLY"), P("G_ONLY"), P("QG")
    checks = dict(I0=i0, selw=round(selw, 3), noop=round(noop, 3), n_fail=len(fail),
                  P_fail={k: round(v, 3) for k, v in dict(PTR=ptr, PTR_C=ptrc, PTR_M=ptrm, Q_ONLY=qo, G_ONLY=go, QG=qg).items()},
                  PTR_RESCUE=GE(ptr, .60), COND_CARRIES=GE(ptrc, .5 * ptr), MATCHER_SUFFICIENT=GE(ptrm, .5 * ptr),
                  QK_ROUTE=GE(qo, .5 * qg) and go < .5 * qg - 1e-9, GATE_ROUTE=GE(go, .5 * qg) and qo < .5 * qg - 1e-9)
    print(f"[{tag}] checks {checks}")
    return checks


if __name__ == "__main__":
    res = {t: analyze(t) for t in sys.argv[1:]}
    json.dump(res, open("/vlm/results/r3/s720_analysis.json", "w"), indent=1, default=float)
