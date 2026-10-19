"""S741 analysis, frozen with the pre-registration (knowledge/DECISIONS.md, 2026-10-11) before any S741 data.
Per model, over forward questions with k >= 3 (the shifted target k - 2 exists):
    I0              NONE answers on k >= .80; APP leaves >= .95 of answers as NONE; FULL sends >= .50 of answers to k - 2
    CODE_SUFFICIENT P(k - 2 | CODE) >= .5 x P(k - 2 | FULL)
    REST_INERT      P(k - 2 | REST) <= .10
    RAND_INERT      P(k - 2 | RAND) <= .10
Reported: P(k | arm), P(k - 2 | arm) with scene-bootstrap 95% intervals (unit = scene, 2,000 draws), variance share of the code subspace.
Predictions: all four rules in both models.  Usage: python s741_analyze.py TAG [TAG ...]
"""
import json, sys
import numpy as np

ARMS = ("NONE", "FULL", "CODE", "REST", "RAND", "APP")


def main(tags):
    allr = {}
    for tag in tags:
        d = json.load(open(f"/vlm/results/r3/s741_{tag}_codesub.json")); rows = [r for r in d["rows"] if all(r[a] != -1 for a in ARMS)]
        units = sorted({(r["t"], r["N"]) for r in rows}); rng = np.random.default_rng(74103)
        def share(rs, arm, off):
            return float(np.mean([r[arm] == r["k"] - off for r in rs])) if rs else float("nan")
        res = {}
        for arm in ARMS:
            pt = (share(rows, arm, 0), share(rows, arm, 2)); bs = []
            for _ in range(2000):
                pick = [units[i] for i in rng.integers(0, len(units), len(units))]
                rs = [r for u in pick for r in rows if (r["t"], r["N"]) == u]
                bs.append((share(rs, arm, 0), share(rs, arm, 2)))
            bs = np.array(bs)
            res[arm] = dict(k=pt[0], km2=pt[1], k_ci=np.percentile(bs[:, 0], [2.5, 97.5]).tolist(), km2_ci=np.percentile(bs[:, 1], [2.5, 97.5]).tolist())
        same_app = float(np.mean([r["APP"] == r["NONE"] for r in rows]))
        chk = dict(I0=res["NONE"]["k"] >= .80 - 1e-12 and same_app >= .95 - 1e-12 and res["FULL"]["km2"] >= .50 - 1e-12,
                   CODE_SUFFICIENT=res["CODE"]["km2"] >= .5 * res["FULL"]["km2"] - 1e-12,
                   REST_INERT=res["REST"]["km2"] <= .10 + 1e-12, RAND_INERT=res["RAND"]["km2"] <= .10 + 1e-12)
        allr[tag] = dict(n=len(rows), scenes=len(units), res=res, app_same=same_app, checks=chk, d=d["d"], D=d["D"],
                         var_explained_mean=float(np.mean(list(d["var_explained"].values()))))
        print(f"[{tag}] n {len(rows)} ({len(units)} scenes) · d {d['d']} of D {d['D']}")
        for arm in ARMS:
            x = res[arm]; print(f"   {arm:5s} on k {x['k']:.3f} [{x['k_ci'][0]:.2f}, {x['k_ci'][1]:.2f}] · on k-2 {x['km2']:.3f} [{x['km2_ci'][0]:.2f}, {x['km2_ci'][1]:.2f}]")
        print(f"   APP = NONE {same_app:.3f} · checks {chk}")
    json.dump(allr, open("/vlm/results/r3/s741_analysis.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
