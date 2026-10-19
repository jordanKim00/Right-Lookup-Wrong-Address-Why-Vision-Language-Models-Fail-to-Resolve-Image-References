"""S746b analysis, frozen with the pre-registration (knowledge/DECISIONS.md, 2026-10-11) before any S746b data.
    THINK_HELPS_BENCH   end-relative option accuracy with reasoning on >= with reasoning skipped + .20
Reported: start- and end-relative accuracy OFF / ON, unanswered share ON, lure share of end-relative errors OFF / ON (lure = the option
naming forward image N + 1 - p for gold image p), mean characters generated, item-bootstrap intervals (4,000 draws).
Prediction: THINK_HELPS_BENCH in both checkpoints.  Usage: python s746b_analyze.py TAG [TAG ...]
"""
import json, sys
import numpy as np


def main(tags):
    out = {}
    for tag in tags:
        R = json.load(open(f"/vlm/results/r3/s746b_{tag}_mmiu.json"))["rows"]; rng = np.random.default_rng(74602)
        res = {}
        for form in ("F", "R"):
            rs = [r for r in R if r["form"] == form]; idx = [rng.integers(0, len(rs), len(rs)) for _ in range(4000)]
            for mode in ("off", "on"):
                x = np.array([r[mode] == r["gold"] for r in rs], float)
                res[f"{form}|{mode}"] = dict(v=float(x.mean()), ci=[float(np.percentile([x[i].mean() for i in idx], q)) for q in (2.5, 97.5)])
            res[f"{form}|on|unanswered"] = float(np.mean([r["on"] is None for r in rs]))
            res[f"{form}|on|chars"] = float(np.mean([r["n_chars"] for r in rs]))
        def lure_share(mode):
            n = l = 0
            for r in R:
                if r["form"] != "R":
                    continue
                img = dict((L, k) for L, k in r["opts"]); p = img.get(r["gold"])
                lure = [L for L, k in r["opts"] if p is not None and k == r["N"] + 1 - p]
                if not lure or lure[0] == r["gold"] or r[mode] in (None, r["gold"]):
                    continue
                n += 1; l += r[mode] == lure[0]
            return (l / n if n else None), n
        res["R|off|lure"], res["R|off|errors"] = lure_share("off")
        res["R|on|lure"], res["R|on|errors"] = lure_share("on")
        chk = dict(THINK_HELPS_BENCH=res["R|on"]["v"] >= res["R|off"]["v"] + .20 - 1e-12)
        out[tag] = dict(n=len(R), res=res, checks=chk)
        print(f"[{tag}] items {len(R)//2} · start OFF {res['F|off']['v']:.3f} ON {res['F|on']['v']:.3f} · end OFF {res['R|off']['v']:.3f} ON {res['R|on']['v']:.3f} "
              f"[{res['R|on']['ci'][0]:.2f}, {res['R|on']['ci'][1]:.2f}] · unanswered ON {res['R|on|unanswered']:.3f} · lure share OFF {res['R|off|lure']} ON {res['R|on|lure']} · "
              f"chars ON start {res['F|on|chars']:.0f} end {res['R|on|chars']:.0f} · checks {chk}")
    json.dump(out, open("/vlm/results/r3/s746b_analysis.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
