"""S722 analysis, frozen with the pre-registration (knowledge/DECISIONS.md, 2026-10-09) before any S722 data.
Rescued items = R3 correct, NATIVE wrong.  QUERY_COMPLETE: P(j | R3_QBACK_ALL) <= .15.  QUERY_SUFFICIENT: P(j | END_QALL) >= .60.
Reported: R3_QBACK and END_Q (matcher heads only), per cell.
Usage: python s722_analyze.py TAG
"""
import json, sys
from collections import defaultdict
import numpy as np

ARMS = ("R3_QBACK", "R3_QBACK_ALL", "END_Q", "END_QALL")


def analyze(tag):
    d = json.load(open(f"/vlm/results/r3/s722_{tag}_mediation.json")); rows = d["rows"]
    acted = [r for r in rows if r["acted"]]
    resc = [r for r in acted if r["land"]["R3"] == r["j"] and r["land"]["NATIVE"] != r["j"]]
    P = lambda arm, rs: float(np.mean([r["land"][arm] == r["j"] for r in rs])) if rs else float("nan")
    print(f"[{tag}] items {len(rows)} · acted {len(acted)} · rescued {len(resc)} | NATIVE {P('NATIVE', rows):.2f} · R3 {P('R3', rows):.2f}")
    print(f"[{tag}]  rescued: " + " | ".join(f"{a_} {P(a_, resc):.2f}" for a_ in ARMS))
    g = defaultdict(list)
    for r in resc:
        g[(r["N"], r["k"])].append(r)
    for c, rs in sorted(g.items()):
        print(f"[{tag}]   N{c[0]} k{c[1]} n {len(rs)}: " + " | ".join(f"{a_} {P(a_, rs):.2f}" for a_ in ARMS))
    back = [r for r in resc if r["land"]["R3_QBACK_ALL"] != r["j"]]
    print(f"[{tag}]  R3_QBACK_ALL failures returning to the original answer: {sum(r['land']['R3_QBACK_ALL'] == r['land']['NATIVE'] for r in back)} / {len(back)}")
    checks = dict(n_rescued=len(resc), rescued={a_: round(P(a_, resc), 3) for a_ in ARMS},
                  QUERY_COMPLETE=bool(resc) and P("R3_QBACK_ALL", resc) <= .15 + 1e-9, QUERY_SUFFICIENT=bool(resc) and P("END_QALL", resc) >= .60 - 1e-9)
    print(f"[{tag}] checks {checks}")
    return checks


if __name__ == "__main__":
    res = {t: analyze(t) for t in sys.argv[1:]}
    json.dump(res, open("/vlm/results/r3/s722_analysis.json", "w"), indent=1, default=float)
