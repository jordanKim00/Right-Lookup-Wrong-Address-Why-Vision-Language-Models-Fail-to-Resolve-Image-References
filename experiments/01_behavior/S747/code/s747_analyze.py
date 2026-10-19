"""S747a analysis, frozen with the pre-registration (knowledge/DECISIONS.md, 2026-10-11) before any S747 data.
Per model (unedited, later MMIU split, 500 items; cloze reported):
    END_DROP        start-relative accuracy - end-relative accuracy >= .20
    LITERAL_ERRORS  among end-relative errors on items whose lure differs from the answer, the lure share >= .50 (uniform ~ .35)
Reported: accuracies with paired item-bootstrap intervals (4,000 draws), the lure share and its uniform reference, cloze start / end.
Predictions: both rules in every model.  Usage: python s747_analyze.py TAG [TAG ...]
"""
import json, os, subprocess, sys
import numpy as np

R = "/vlm/results"


def main(tags):
    out = {}
    for tag in tags:
        d = json.load(open(f"{R}/s582_evalrepair_{tag}.json")); rows = [r for r in d["rows"] if r["suite"] == "mmiu_pos"]
        F = np.array([r["F|base"] == r["gold"] for r in rows], float); E_ = np.array([r["R|base"] == r["gold"] for r in rows], float)
        rng = np.random.default_rng(74701); idx = [rng.integers(0, len(rows), len(rows)) for _ in range(4000)]
        ci = lambda x: [float(np.percentile([x[i].mean() for i in idx], 2.5)), float(np.percentile([x[i].mean() for i in idx], 97.5))]
        lure = subprocess.run(["python3", "/vlm/exp/lure_latest.py", f"{tag}:fresh", "--out", f"/tmp/lure_{tag}.json"], capture_output=True, text=True).stdout
        L = json.load(open(f"/tmp/lure_{tag}.json")) if os.path.exists(f"/tmp/lure_{tag}.json") else None
        Ld = (L[0] if isinstance(L, list) else (L.get(tag) or L.get(f"{tag}:fresh") or next(iter(L.values())))) if L else {}
        cz = json.load(open(f"{R}/s582_evalrepair_{tag}_cz.json"))["rows"] if os.path.exists(f"{R}/s582_evalrepair_{tag}_cz.json") else []
        res = dict(n=len(rows), start=float(F.mean()), start_ci=ci(F), end=float(E_.mean()), end_ci=ci(E_), drop=float((F - E_).mean()), drop_ci=ci(F - E_),
                   lure_share=Ld.get("lure_share"), lure_chance=Ld.get("chance"), lure_errors=Ld.get("errors"),
                   cloze_start=float(np.mean([r["F|base"] == r["gold"] for r in cz])) if cz else None,
                   cloze_end=float(np.mean([r["R|base"] == r["gold"] for r in cz])) if cz else None)
        chk = dict(END_DROP=res["drop"] >= .20 - 1e-12, LITERAL_ERRORS=(res["lure_share"] or 0) >= .50 - 1e-12)
        out[tag] = dict(res=res, checks=chk)
        print(f"[{tag}] later MMIU start {res['start']:.3f} end {res['end']:.3f} drop {res['drop']:+.3f} [{res['drop_ci'][0]:+.3f}, {res['drop_ci'][1]:+.3f}] · "
              f"lure share {res['lure_share']} (uniform {res['lure_chance']}) · cloze start {res['cloze_start']} end {res['cloze_end']} · checks {chk}")
    json.dump(out, open(f"{R}/r3/s747_analysis.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
