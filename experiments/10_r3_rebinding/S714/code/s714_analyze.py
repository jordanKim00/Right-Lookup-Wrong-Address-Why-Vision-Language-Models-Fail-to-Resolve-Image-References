"""S714 analysis, frozen with the pre-registration (knowledge/DECISIONS.md, 2026-10-09) before any S714 data.
I0 HIDE_INSTRUMENT: P(2 | START(1), HIDE1_ALL) >= .50
KN_COUNT:         P(1 | END(N), COUNT_UP) <= P(1 | END(N), BASE) - .30
KN_FIRST_VISIBLE: P(2 | END(N), HIDE1_ALL) >= .50
KN_LATE:          P(1 | END(N), HIDE1_LATE) <= P(1 | END(N), BASE) - .30 and P(1 | END(N), HIDE1_EARLY) >= P(1 | END(N), BASE) - .10
Pooled over N in {5, 7}; per-N values and the landing distributions are reported.
Usage: python s714_analyze.py TAG
"""
import json, sys
from collections import Counter
import numpy as np

GE = lambda a, b: a >= b - 1e-9


def analyze(tag):
    d = json.load(open(f"/vlm/results/r3/s714_{tag}_kn.json")); rows = d["rows"]
    P = lambda part, arm, y, rs=rows: float(np.mean([r[part][arm] == y for r in rs]))
    for N in sorted({r["N"] for r in rows}) + [None]:
        rs = [r for r in rows if N is None or r["N"] == N]
        nm = f"N{N}" if N else "pooled"
        print(f"[{tag}] {nm} (n {len(rs)})")
        print("   END(N) → 1/2: " + " · ".join(f"{a_} {P('EN', a_, 1, rs):.2f}/{P('EN', a_, 2, rs):.2f}" for a_ in rs[0]["EN"]))
        print("   START(1) → 1/2: " + " · ".join(f"{a_} {P('S1', a_, 1, rs):.2f}/{P('S1', a_, 2, rs):.2f}" for a_ in rs[0]["S1"]))
        print("   END(N-1) → 2/3: " + " · ".join(f"{a_} {P('EN1', a_, 2, rs):.2f}/{P('EN1', a_, 3, rs):.2f}" for a_ in rs[0]["EN1"]))
        for a_ in rs[0]["EN"]:
            print(f"     END(N) {a_}: {dict(sorted(Counter(r['EN'][a_] for r in rs).items()))}")
    b = P("EN", "BASE", 1)
    checks = dict(I0_HIDE_INSTRUMENT=GE(P("S1", "HIDE1_ALL", 2), .50),
                  KN_COUNT=P("EN", "COUNT_UP", 1) <= b - .30 + 1e-9,
                  KN_FIRST_VISIBLE=GE(P("EN", "HIDE1_ALL", 2), .50),
                  KN_LATE=P("EN", "HIDE1_LATE", 1) <= b - .30 + 1e-9 and GE(P("EN", "HIDE1_EARLY", 1), b - .10))
    print(f"[{tag}] checks {checks}")
    return dict(checks=checks, END_N={a_: dict(p1=P("EN", a_, 1), p2=P("EN", a_, 2)) for a_ in rows[0]["EN"]},
                START1={a_: dict(p1=P("S1", a_, 1), p2=P("S1", a_, 2)) for a_ in rows[0]["S1"]},
                END_N1={a_: dict(p2=P("EN1", a_, 2), p3=P("EN1", a_, 3)) for a_ in rows[0]["EN1"]})


if __name__ == "__main__":
    res = {t: analyze(t) for t in sys.argv[1:]}
    json.dump(res, open("/vlm/results/r3/s714_analysis.json", "w"), indent=1, default=float)
