"""S741b analysis, frozen with the pre-registration (knowledge/DECISIONS.md, 2026-10-11) before any S741b data.
Per model, forward questions with k >= 3:
    I0              NONE answers on k >= .80; FULL sends >= .50 to k - 2; AKEY leaves >= .95 of answers as NONE
    KEY_SUFFICIENT  P(k - 2 | KEY) >= .5 x P(k - 2 | FULL)
    VAL_INERT       P(k - 2 | VAL) <= .10
Reported: P(k | arm), P(k - 2 | arm) with scene-bootstrap 95% intervals (2,000 draws); KV.
Predictions: all three in both models.  Usage: python s741b_analyze.py TAG [TAG ...]
"""
import json, sys
import numpy as np

ARMS = ("NONE", "FULL", "KEY", "VAL", "KV", "AKEY")


def main(tags):
    allr = {}
    for tag in tags:
        rows = [r for r in json.load(open(f"/vlm/results/r3/s741b_{tag}_kv.json"))["rows"] if all(r[a] != -1 for a in ARMS)]
        units = sorted({(r["t"], r["N"]) for r in rows}); rng = np.random.default_rng(74104)
        sh = lambda rs, arm, off: float(np.mean([r[arm] == r["k"] - off for r in rs])) if rs else float("nan")
        res = {}
        for arm in ARMS:
            bs = []
            for _ in range(2000):
                pick = [units[i] for i in rng.integers(0, len(units), len(units))]
                rs = [r for u in pick for r in rows if (r["t"], r["N"]) == u]
                bs.append((sh(rs, arm, 0), sh(rs, arm, 2)))
            bs = np.array(bs)
            res[arm] = dict(k=sh(rows, arm, 0), km2=sh(rows, arm, 2), k_ci=np.percentile(bs[:, 0], [2.5, 97.5]).tolist(), km2_ci=np.percentile(bs[:, 1], [2.5, 97.5]).tolist())
        same = float(np.mean([r["AKEY"] == r["NONE"] for r in rows]))
        chk = dict(I0=res["NONE"]["k"] >= .80 - 1e-12 and res["FULL"]["km2"] >= .50 - 1e-12 and same >= .95 - 1e-12,
                   KEY_SUFFICIENT=res["KEY"]["km2"] >= .5 * res["FULL"]["km2"] - 1e-12, VAL_INERT=res["VAL"]["km2"] <= .10 + 1e-12)
        allr[tag] = dict(n=len(rows), scenes=len(units), res=res, akey_same=same, checks=chk)
        print(f"[{tag}] n {len(rows)} ({len(units)} scenes)")
        for arm in ARMS:
            x = res[arm]; print(f"   {arm:5s} on k {x['k']:.3f} [{x['k_ci'][0]:.2f}, {x['k_ci'][1]:.2f}] · on k-2 {x['km2']:.3f} [{x['km2_ci'][0]:.2f}, {x['km2_ci'][1]:.2f}]")
        print(f"   AKEY = NONE {same:.3f} · checks {chk}")
    json.dump(allr, open("/vlm/results/r3/s741b_analysis.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
