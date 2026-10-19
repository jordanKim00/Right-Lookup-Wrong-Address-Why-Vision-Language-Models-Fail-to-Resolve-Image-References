"""S703 analysis, frozen with the pre-registration (knowledge/DECISIONS.md, 2026-10-09) before any S703 data.
T_m = 8 - m - k (moved target), T_0 = 8 - k (unmoved).  "kN" denotes k = 7 - m (moved target 1).
END_MOVES(arm, k) = P(T_m | arm) >= .5 P(T_m | prefix reference) and P(T_m | arm) >= P(T_m | unedited) + .20, pooled over m in {1, 2}
Predictions: END_MOVES(FA_ALL, 2), END_MOVES(FA_4347, 2), not END_MOVES(DN_ALL, 2).
Usage: python s703_analyze.py TAG
"""
import json, sys
import numpy as np

GE = lambda a, b: a >= b - 1e-9
NR = 7


def analyze(tag):
    d = json.load(open(f"/vlm/results/r3/s703_{tag}_moveend.json")); rows = d["rows"]; arms = d["arms"]
    print(f"[{tag}] n {len(rows)} · L43/L47 {d['L43']}/{d['L47']}")
    checks = {}
    for kk in ("1", "2", "3", "kN"):
        print(f"[{tag}] == k {kk}")
        stats = {a_: [] for a_ in arms + ["ref", "unedited"]}
        for m in (1, 2):
            k = NR - m if kk == "kN" else int(kk)
            Tm, T0 = 8 - m - k, 8 - k
            P = lambda xs: float(np.mean(xs))
            ref = P([r["ref"][f"{m}|{k}"] == Tm for r in rows]); une = P([r["base"][str(k)] == Tm for r in rows])
            une0 = P([r["base"][str(k)] == T0 for r in rows])
            stats["ref"].append(ref); stats["unedited"].append(une)
            line = [f"m{m} k{k}: T_m={Tm} ref {ref:.2f} · unedited T_m {une:.2f} (T_0 {une0:.2f})"]
            for a_ in arms:
                pm = P([r["hide"][f"{a_}|{m}|{k}"] == Tm for r in rows]); p0 = P([r["hide"][f"{a_}|{m}|{k}"] == T0 for r in rows])
                pl = P([r["hide"][f"{a_}|{m}|{k}"] == k for r in rows])
                stats[a_].append(pm); line.append(f"{a_} T_m {pm:.2f} T_0 {p0:.2f} k {pl:.2f}")
            print(f"[{tag}]  " + " | ".join(line))
        mean = {a_: float(np.mean(v)) for a_, v in stats.items()}
        checks[kk] = {a_: bool(GE(mean[a_], .5 * mean["ref"]) and GE(mean[a_], mean["unedited"] + .20)) for a_ in arms}
        checks[kk]["means"] = {a_: round(v, 3) for a_, v in mean.items()}
        print(f"[{tag}]  END_MOVES k {kk}: {checks[kk]}")
    return checks


if __name__ == "__main__":
    res = {t: analyze(t) for t in sys.argv[1:]}
    json.dump(res, open("/vlm/results/r3/s703_analysis.json", "w"), indent=1, default=float)
