"""S707 analysis, frozen with the pre-registration (knowledge/DECISIONS.md, 2026-10-09) before any S707 data.  Pooled over k in {2, 3}, m in {1, 2}.
END_IN_MATCHERS: P(T_m | MAT_4347) >= .5 P(T_m | ALL_4347) and >= P(T_m | NONMAT_4347) + .15
END_OUTSIDE_MATCHERS: P(T_m | NONMAT_4347) >= .5 P(T_m | ALL_4347) and >= P(T_m | MAT_4347) + .15
Usage: python s707_analyze.py TAG
"""
import json, sys
import numpy as np

GE = lambda a, b: a >= b - 1e-9


def analyze(tag):
    d = json.load(open(f"/vlm/results/r3/s707_{tag}_endreaders.json")); rows = d["rows"]; arms = d["arms"]
    cells = [(k, m) for k in (2, 3) for m in (1, 2)]
    P = {a_: dict(Tm=float(np.mean([r["arm"][f"{a_}|{m}|{k}"]["land"] == 8 - m - k for r in rows for k, m in cells])),
                  T0=float(np.mean([r["arm"][f"{a_}|{m}|{k}"]["land"] == 8 - k for r in rows for k, m in cells]))) for a_ in arms}
    print(f"[{tag}] n {len(rows)} · ref T_m {np.mean([r['ref'][f'{m}|{k}'] == 8 - m - k for r in rows for k, m in cells]):.2f} · unedited T_0 {np.mean([r['base'][str(k)]['land'] == 8 - k for r in rows for k, m in cells]):.2f}")
    for k, m in cells:
        print(f"[{tag}]  k{k} m{m}: " + " | ".join(f"{a_} Tm {np.mean([r['arm'][f'{a_}|{m}|{k}']['land'] == 8 - m - k for r in rows]):.2f} T0 {np.mean([r['arm'][f'{a_}|{m}|{k}']['land'] == 8 - k for r in rows]):.2f}" for a_ in arms))
    checks = dict(pooled={a_: {kk: round(vv, 3) for kk, vv in P[a_].items()} for a_ in arms},
                  END_IN_MATCHERS=GE(P["MAT_4347"]["Tm"], .5 * P["ALL_4347"]["Tm"]) and GE(P["MAT_4347"]["Tm"], P["NONMAT_4347"]["Tm"] + .15),
                  END_OUTSIDE_MATCHERS=GE(P["NONMAT_4347"]["Tm"], .5 * P["ALL_4347"]["Tm"]) and GE(P["NONMAT_4347"]["Tm"], P["MAT_4347"]["Tm"] + .15))
    print(f"[{tag}] checks {checks}")
    return checks


if __name__ == "__main__":
    res = {t: analyze(t) for t in sys.argv[1:]}
    json.dump(res, open("/vlm/results/r3/s707_analysis.json", "w"), indent=1, default=float)
