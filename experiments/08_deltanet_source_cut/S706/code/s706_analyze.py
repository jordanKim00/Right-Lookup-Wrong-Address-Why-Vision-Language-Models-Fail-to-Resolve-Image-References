"""S706 analysis, frozen with the pre-registration (knowledge/DECISIONS.md, 2026-10-09) before any S706 data.  Pooled over k in {2, 3}, m in {1, 2}.
ANCHORS_SET_END: P(T_m | ANC) >= .5 P(T_m | FA_ALL) and >= P(T_m | ANC_RAND) + .20
PURE_SUFFICE (reported): P(T_m | ANC_PURE) >= .5 P(T_m | ANC)
END_VIA_MATCHER_Q: P(T_0 | ANC_Q) >= P(T_0 | ANC) + .20 and P(T_0 | ANC_Q) > P(T_0 | ANC_G)
Usage: python s706_analyze.py TAG
"""
import json, sys
import numpy as np

GE = lambda a, b: a >= b - 1e-9
ARMS = ("FA_ALL", "ANC", "ANC_PURE", "ANC_RAND", "ANC_Q", "ANC_G")


def analyze(tag):
    d = json.load(open(f"/vlm/results/r3/s706_{tag}_anchorpath.json")); rows = d["rows"]
    print(f"[{tag}] n {len(rows)} · sets {d['sets']}")
    cells = [(k, m) for k in (2, 3) for m in (1, 2)]
    P = {}
    for arm in ARMS:
        P[arm] = dict(Tm=float(np.mean([r["arm"][f"{arm}|{m}|{k}"]["land"] == 8 - m - k for r in rows for k, m in cells])),
                      T0=float(np.mean([r["arm"][f"{arm}|{m}|{k}"]["land"] == 8 - k for r in rows for k, m in cells])))
    ref = float(np.mean([r["ref"][f"{m}|{k}"] == 8 - m - k for r in rows for k, m in cells]))
    une_T0 = float(np.mean([r["base"][str(k)]["land"] == 8 - k for r in rows for k, m in cells]))
    for k, m in cells:
        line = " | ".join(f"{arm} Tm {np.mean([r['arm'][f'{arm}|{m}|{k}']['land'] == 8 - m - k for r in rows]):.2f} T0 {np.mean([r['arm'][f'{arm}|{m}|{k}']['land'] == 8 - k for r in rows]):.2f}" for arm in ARMS)
        print(f"[{tag}]  k{k} m{m}: ref Tm {np.mean([r['ref'][f'{m}|{k}'] == 8 - m - k for r in rows]):.2f} | {line}")
    checks = dict(pooled={a_: {kk: round(vv, 3) for kk, vv in P[a_].items()} for a_ in ARMS}, ref_Tm=round(ref, 3), unedited_T0=round(une_T0, 3),
                  ANCHORS_SET_END=GE(P["ANC"]["Tm"], .5 * P["FA_ALL"]["Tm"]) and GE(P["ANC"]["Tm"], P["ANC_RAND"]["Tm"] + .20),
                  PURE_SUFFICE=GE(P["ANC_PURE"]["Tm"], .5 * P["ANC"]["Tm"]),
                  END_VIA_MATCHER_Q=GE(P["ANC_Q"]["T0"], P["ANC"]["T0"] + .20) and P["ANC_Q"]["T0"] > P["ANC_G"]["T0"] + 1e-9)
    print(f"[{tag}] checks {checks}")
    return checks


if __name__ == "__main__":
    res = {t: analyze(t) for t in sys.argv[1:]}
    json.dump(res, open("/vlm/results/r3/s706_analysis.json", "w"), indent=1, default=float)
