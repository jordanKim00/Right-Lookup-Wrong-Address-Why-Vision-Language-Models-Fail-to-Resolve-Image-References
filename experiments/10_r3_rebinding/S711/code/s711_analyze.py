"""S711 analysis, frozen with the pre-registration (knowledge/DECISIONS.md, 2026-10-09) before any S711 data.
V2_FIXES_MIDDLE: P(j | R3V2) >= .90 on middle cells (N7 k4, N5 k3) and, if P(j | R3) < .60 there, >= P(j | R3) + .30
V2_NO_HARM: P(j | R3V2) >= P(j | R3) - .02 on the other cells;  V2_OVERALL: P(j | R3V2) >= .95 overall
Usage: python s711_analyze.py TAG
"""
import json, sys
import numpy as np

GE = lambda a, b: a >= b - 1e-9
ARMS = ("NATIVE", "O_ORD_FR", "R3", "R3V2")


def analyze(tag):
    d = json.load(open(f"/vlm/results/r3/s711_{tag}_eval.json")); rows = d["rows"]
    P = lambda arm, rs: float(np.mean([r["land"][arm] == r["j"] for r in rs])) if rs else float("nan")
    print(f"[{tag}] n {len(rows)}")
    for N in (7, 5):
        for k in range(1, N + 1):
            rs = [r for r in rows if r["N"] == N and r["k"] == k]
            print(f"[{tag}]  N{N} k{k}: " + " | ".join(f"{a_} {P(a_, rs):.2f}" for a_ in ARMS))
    mid = [r for r in rows if r["j"] == r["k"]]; oth = [r for r in rows if r["j"] != r["k"]]
    fix = GE(P("R3V2", mid), .90) and (P("R3", mid) >= .60 or GE(P("R3V2", mid), P("R3", mid) + .30))
    checks = dict(overall={a_: round(P(a_, rows), 3) for a_ in ARMS}, middle={a_: round(P(a_, mid), 3) for a_ in ARMS},
                  other={a_: round(P(a_, oth), 3) for a_ in ARMS}, V2_FIXES_MIDDLE=fix,
                  V2_NO_HARM=GE(P("R3V2", oth), P("R3", oth) - .02), V2_OVERALL=GE(P("R3V2", rows), .95))
    print(f"[{tag}] checks {checks}")
    return checks


if __name__ == "__main__":
    res = {t: analyze(t) for t in sys.argv[1:]}
    json.dump(res, open("/vlm/results/r3/s711_analysis.json", "w"), indent=1, default=float)
