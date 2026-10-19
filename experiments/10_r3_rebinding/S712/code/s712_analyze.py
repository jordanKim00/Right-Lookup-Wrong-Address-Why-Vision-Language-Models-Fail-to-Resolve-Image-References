"""S712 analysis, frozen with the pre-registration (knowledge/DECISIONS.md, 2026-10-09) before any S712 data.
GEN_REPAIR:      END P(j | R3V2) >= .90 overall and >= P(j | NATIVE) + .20
GEN_FACTORS:     END P(j | R3V2) >= .85 within every N, every layout and every end word
ORACLE_FRACTION: END P(j | R3V2) >= .90 P(j | O_ORD_FR)
START_NO_HARM:   START P(k | R3V2) >= P(k | NATIVE) - .02, and write rate on START items <= .05
Also reported (no rule): per-(N, k) END table, stage ledger (detected, k_hat, anchor, N_hat, t, write), START write sources.
Usage: python s712_analyze.py TAG
"""
import json, sys
from collections import defaultdict
import numpy as np

GE = lambda a, b: a >= b - 1e-9


def analyze(tag):
    d = json.load(open(f"/vlm/results/r3/s712_{tag}_general.json")); rows = d["rows"]
    E = [r for r in rows if r["kind"] == "END"]; S = [r for r in rows if r["kind"] == "START"]
    P = lambda arm, rs: float(np.mean([r["land"][arm] == r["j"] for r in rs])) if rs else float("nan")
    print(f"[{tag}] END {len(E)} · START {len(S)} | END NATIVE {P('NATIVE', E):.3f} · R3V2 {P('R3V2', E):.3f} · O_ORD_FR {P('O_ORD_FR', E):.3f}"
          f" | START NATIVE {P('NATIVE', S):.3f} · R3V2 {P('R3V2', S):.3f} · write rate {np.mean([r['wrote'] for r in S]):.3f}")
    fac = {}
    for name, key in (("N", "N"), ("layout", "layout"), ("word", "word")):
        g = defaultdict(list)
        for r in E:
            g[r[key]].append(r)
        fac[name] = {str(x): dict(n=len(rs), NATIVE=round(P("NATIVE", rs), 3), R3V2=round(P("R3V2", rs), 3), O_ORD_FR=round(P("O_ORD_FR", rs), 3))
                     for x, rs in sorted(g.items())}
        print(f"[{tag}]  {name}: " + " | ".join(f"{x} n{v_['n']} {v_['NATIVE']:.2f}→{v_['R3V2']:.2f} (상한 {v_['O_ORD_FR']:.2f})" for x, v_ in fac[name].items()))
    g = defaultdict(list)
    for r in E:
        g[(r["N"], r["k"])].append(r)
    for N in sorted({r["N"] for r in E}):
        print(f"[{tag}]  N{N} k: " + " · ".join(f"k{k} {P('NATIVE', g[(N, k)]):.2f}→{P('R3V2', g[(N, k)]):.2f}" for k in range(1, N + 1)))
    led = defaultdict(int)
    for r in E:
        L = r["ledger"]
        led["detected"] += bool(L)
        src = [x for x in L if x["anchor"] is not None]
        led["anchor"] += bool(src)
        if src:
            x = src[0]
            led["kh_ok"] += x["kh"] == r["k"]; led["Nhat_ok"] += x["N_hat"] == r["N"]; led["t_ok"] += x["t"] == r["j"]
        led["wrote"] += r["wrote"]
        led["wrote_wrong"] += r["wrote"] and r["land"]["R3V2"] != r["j"]
    print(f"[{tag}]  END 원장: " + " · ".join(f"{k_} {v_}" for k_, v_ in led.items()) + f" (항목 {len(E)})")
    sw = [r for r in S if r["wrote"]]
    print(f"[{tag}]  START 쓰기 {len(sw)}: " + "; ".join(f"N{r['N']} k{r['k']} {r['word']} {r['layout']} {[(x['kh'], x['anchor'], x['act'], x['fonly']) for x in r['ledger']]}" for r in sw[:12]))
    checks = dict(
        GEN_REPAIR=GE(P("R3V2", E), .90) and GE(P("R3V2", E), P("NATIVE", E) + .20),
        GEN_FACTORS=all(GE(v_["R3V2"], .85) for f_ in fac.values() for v_ in f_.values()),
        ORACLE_FRACTION=GE(P("R3V2", E), .90 * P("O_ORD_FR", E)),
        START_NO_HARM=GE(P("R3V2", S), P("NATIVE", S) - .02) and float(np.mean([r["wrote"] for r in S])) <= .05 + 1e-9)
    print(f"[{tag}] checks {checks}")
    return dict(END=dict(n=len(E), NATIVE=P("NATIVE", E), R3V2=P("R3V2", E), O_ORD_FR=P("O_ORD_FR", E)),
                START=dict(n=len(S), NATIVE=P("NATIVE", S), R3V2=P("R3V2", S), write_rate=float(np.mean([r["wrote"] for r in S]))),
                factors=fac, ledger=dict(led), checks=checks)


if __name__ == "__main__":
    res = {t: analyze(t) for t in sys.argv[1:]}
    json.dump(res, open("/vlm/results/r3/s712_analysis.json", "w"), indent=1, default=float)
