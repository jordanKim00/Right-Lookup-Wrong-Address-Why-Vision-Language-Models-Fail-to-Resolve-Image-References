"""S710 analysis, frozen with the pre-registration (knowledge/DECISIONS.md, 2026-10-09) before any S710 data.
Rescued items = R3 correct, NATIVE wrong.
MEDIATION_NECESSARY: rescued P(j | R3_QBACK) <= .70 and P(j | R3_RBACK) >= .90
MEDIATION_SUFFICIENT: rescued P(j | END_QR3) >= .20, >= P(j | END_GR3) + .10, >= P(j | END_RR3) + .10
Usage: python s710_analyze.py TAG
"""
import json, sys
import numpy as np

GE = lambda a, b: a >= b - 1e-9
ARMS = ("R3_QBACK", "R3_GBACK", "R3_RBACK", "END_QR3", "END_GR3", "END_RR3")


def analyze(tag):
    d = json.load(open(f"/vlm/results/r3/s710_{tag}_mediation.json")); rows = d["rows"]
    acted = [r for r in rows if r["acted"]]
    resc = [r for r in acted if r["land"]["R3"] == r["j"] and r["land"]["NATIVE"] != r["j"]]
    P = lambda arm, rs: float(np.mean([r["land"][arm] == r["j"] for r in rs])) if rs else float("nan")
    print(f"[{tag}] items {len(rows)} · acted {len(acted)} · rescued {len(resc)} | NATIVE {P('NATIVE', rows):.2f} · R3 {P('R3', rows):.2f}")
    print(f"[{tag}]  rescued: " + " | ".join(f"{a_} {P(a_, resc):.2f}" for a_ in ARMS))
    print(f"[{tag}]  all acted: " + " | ".join(f"{a_} {P(a_, acted):.2f}" for a_ in ARMS))
    checks = dict(n_rescued=len(resc), rescued={a_: round(P(a_, resc), 3) for a_ in ARMS},
                  MEDIATION_NECESSARY=bool(resc) and P("R3_QBACK", resc) <= .70 + 1e-9 and GE(P("R3_RBACK", resc), .90),
                  MEDIATION_SUFFICIENT=bool(resc) and GE(P("END_QR3", resc), .20) and GE(P("END_QR3", resc), P("END_GR3", resc) + .10)
                  and GE(P("END_QR3", resc), P("END_RR3", resc) + .10))
    print(f"[{tag}] checks {checks}")
    return checks


if __name__ == "__main__":
    res = {t: analyze(t) for t in sys.argv[1:]}
    json.dump(res, open("/vlm/results/r3/s710_analysis.json", "w"), indent=1, default=float)
