"""S746 analysis, frozen with the pre-registration (knowledge/DECISIONS.md, 2026-10-11) before any S746 data.
Per Thinking checkpoint:
    THINK_HELPS   end-relative accuracy with reasoning on >= with reasoning skipped + .20
    SAY_NOT_SEE   among end-relative questions whose trace settles on the correct forward position j (n >= 10), the final answer is not
                  the target in >= .25 (an address stated in words but not used for selection)
Reported: accuracy, literal answers and truncation for START / END x OFF / ON; trace position (j, k, other, none) and its cross-tab with
the answer; scene-bootstrap intervals (2,000 draws).  Predictions: THINK_HELPS yes; SAY_NOT_SEE no prediction.
Usage: python s746_analyze.py TAG [TAG ...]
"""
import json, sys
import numpy as np


def main(tags):
    out = {}
    for tag in tags:
        R = json.load(open(f"/vlm/results/r3/s746_{tag}_think.json"))["rows"]; sc = sorted({r["si"] for r in R}); rng = np.random.default_rng(74601)
        def stat(rs, form, mode, what):
            rs = [r for r in rs if r["form"] == form]
            tgt = lambda r: r["j"] if form == "END" else r["j"]
            if what == "acc": return float(np.mean([r[mode] == tgt(r) for r in rs]))
            if what == "lit": return float(np.mean([r[mode] == r["k"] for r in rs]))
            if what == "trunc": return float(np.mean([r[mode] is None for r in rs]))
        res = {}
        for form in ("START", "END"):
            for mode in ("off", "on"):
                for what in (("acc", "lit") if mode == "off" else ("acc", "lit", "trunc")):
                    if form == "START" and what == "lit":
                        continue
                    pt = stat(R, form, mode, what)
                    bs = [stat([r for s in rng.choice(sc, len(sc)) for r in R if r["si"] == s], form, mode, what) for _ in range(2000)]
                    res[f"{form}|{mode}|{what}"] = dict(v=pt, ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
        E = [r for r in R if r["form"] == "END"]
        cat = lambda r: "j" if r["trace_pos"] == r["j"] else ("k" if r["trace_pos"] == r["k"] else ("none" if r["trace_pos"] is None else "other"))
        cross = {}
        for c in ("j", "k", "other", "none"):
            rs = [r for r in E if cat(r) == c]
            cross[c] = dict(n=len(rs), ans_target=float(np.mean([r["on"] == r["j"] for r in rs])) if rs else None,
                            ans_literal=float(np.mean([r["on"] == r["k"] for r in rs])) if rs else None)
        g = lambda key: res[key]["v"]
        chk = dict(THINK_HELPS=g("END|on|acc") >= g("END|off|acc") + .20 - 1e-12,
                   SAY_NOT_SEE=(cross["j"]["n"] >= 10 and 1 - cross["j"]["ans_target"] >= .25 - 1e-12) if cross["j"]["n"] else False)
        out[tag] = dict(res=res, trace=cross, checks=chk, n=len(R))
        print(f"[{tag}] questions {len(R)}")
        for key, x in res.items():
            print(f"   {key:16s} {x['v']:.3f} [{x['ci'][0]:.2f}, {x['ci'][1]:.2f}]")
        print(f"   END trace position -> answer: " + " · ".join(f"{c}: n {d['n']}, target {d['ans_target'] if d['ans_target'] is None else round(d['ans_target'],3)}, literal {d['ans_literal'] if d['ans_literal'] is None else round(d['ans_literal'],3)}" for c, d in cross.items()))
        print(f"   checks {chk}")
    json.dump(out, open("/vlm/results/r3/s746_analysis.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
