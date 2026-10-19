"""S708 analysis, frozen with the pre-registration (knowledge/DECISIONS.md, 2026-10-09) before any S708 data.  Per write layer, means over k.
REBIND_REPAIRS: P(r | ORD_FR) >= .80 and >= P(r | BASE) + .30;  REREFLECT: P(k | ORD) >= .40;  ADDRESS_SPECIFIC: P(w | WRONG) >= .70
Usage: python s708_analyze.py TAG
"""
import json, sys
import numpy as np

GE = lambda a, b: a >= b - 1e-9


def analyze(tag):
    d = json.load(open(f"/vlm/results/r3/s708_{tag}_rebind.json")); rows = d["rows"]; ks = ("2", "3", "4", "5", "6")
    P = lambda arm, who: float(np.mean([np.mean([r["k"][k]["land"][arm] == (r["k"][k][who] if who in ("r", "w") else int(k)) for r in rows]) for k in ks]))
    out = {}
    print(f"[{tag}] n {len(rows)} · LW {d['LW']} · BASE r {P('BASE', 'r'):.2f} k {P('BASE', 'k'):.2f}")
    for li in d["LW"]:
        line = []
        for arm in ("ORD", "FR", "ORD_FR", "WRONG", "RAND_FR"):
            a_ = f"{arm}@{li}"
            line.append(f"{arm} r {P(a_, 'r'):.2f} k {P(a_, 'k'):.2f} w {P(a_, 'w'):.2f}")
        print(f"[{tag}]  L{li}: " + " | ".join(line))
        byk = {k: round(float(np.mean([r['k'][k]['land'][f'ORD_FR@{li}'] == r['k'][k]['r'] for r in rows])), 2) for k in ks}
        print(f"[{tag}]  L{li} ORD_FR target by k {byk} · BASE by k {{" + ", ".join(f"{k}: {np.mean([r['k'][k]['land']['BASE'] == r['k'][k]['r'] for r in rows]):.2f}" for k in ks) + "}")
        out[str(li)] = dict(REBIND_REPAIRS=GE(P(f"ORD_FR@{li}", "r"), .80) and GE(P(f"ORD_FR@{li}", "r"), P("BASE", "r") + .30),
                            REREFLECT=GE(P(f"ORD@{li}", "k"), .40), ADDRESS_SPECIFIC=GE(P(f"WRONG@{li}", "w"), .70))
    print(f"[{tag}] checks {out}")
    return out


if __name__ == "__main__":
    res = {t: analyze(t) for t in sys.argv[1:]}
    json.dump(res, open("/vlm/results/r3/s708_analysis.json", "w"), indent=1, default=float)
