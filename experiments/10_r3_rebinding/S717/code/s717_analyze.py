"""S717 analysis, frozen with the pre-registration (knowledge/DECISIONS.md, 2026-10-09) before any S717 data.
R³-v4 = R³-v3 + frame anchors for "-to-last", "in reverse order", "counting backwards" (exp/s717_framecal.py, base r327b4).
Each v4 run's own paired base (unedited) and repaired answers on the development items (MMIU 550, MuirBench 350):
    V4_REVERSE:   MMIU end-relative form (reverse wording): rep - base >= .25 and the paired 95% bootstrap interval excludes 0
    V4_BACKWARDS: same (counting backwards wording)
    V4_TOLAST:    MMIU (-to-last wording): rep - base >= .10 and the interval excludes 0
    V4_LOCALITY:  forward answers changed by v4 on <= .01 of the items in every run, and the canonical end form under v4 gives the
                  same repaired answers as under v3 on >= .99 of the items
MuirBench is reported.  Also reported: anchors found / frame writes per wording from the saved plans.
Usage: python s717_analyze.py --reverse T --backwards T --tolast T --end4 T --end3 T
"""
import argparse, json
import numpy as np


def rows(tag):
    return json.load(open(f"/vlm/results/s582_evalrepair_{tag}.json"))["rows"]


def paired(rs, form):
    x0 = np.array([r[f"{form}|base"] == r["gold"] for r in rs], float); x1 = np.array([r[f"{form}|rep"] == r["gold"] for r in rs], float)
    dd = x1 - x0; rng = np.random.default_rng(0)
    bs = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(4000)]
    return dict(n=len(rs), base=float(x0.mean()), rep=float(x1.mean()), d=float(dd.mean()), lo=float(np.percentile(bs, 2.5)),
                hi=float(np.percentile(bs, 97.5)), changed=float(np.mean([r[f"{form}|base"] != r[f"{form}|rep"] for r in rs])))


def main(a):
    res, checks = {}, {}
    for name in ("reverse", "backwards", "tolast", "end4"):
        R = rows(getattr(a, name)); res[name] = {}
        for s in ("mmiu_pos", "muir"):
            S = [r for r in R if r["suite"] == s]
            res[name][s] = {f: paired(S, f) for f in ("F", "R")}
            det = [x for r in S for x in (r.get("R|det") or [])]
            res[name][s]["plans"] = dict(sources=len(det), anchored=sum(x[3] for x in det), frame_writes=sum(1 for x in det if x[6] is not None))
            p = res[name][s]
            print(f"[{name} {s}] F {p['F']['base']:.3f} (바뀜 {p['F']['changed']:.3f}) | R {p['R']['base']:.3f} → {p['R']['rep']:.3f}  Δ {p['R']['d']:+.3f} "
                  f"[{p['R']['lo']:+.3f}, {p['R']['hi']:+.3f}] | 계획: 원천 {p['plans']['sources']} · 기준점 {p['plans']['anchored']} · 틀 쓰기 {p['plans']['frame_writes']}")
    m = lambda n: res[n]["mmiu_pos"]["R"]
    checks["V4_REVERSE"] = m("reverse")["d"] >= .25 - 1e-9 and m("reverse")["lo"] > 0
    checks["V4_BACKWARDS"] = m("backwards")["d"] >= .25 - 1e-9 and m("backwards")["lo"] > 0
    checks["V4_TOLAST"] = m("tolast")["d"] >= .10 - 1e-9 and m("tolast")["lo"] > 0
    e4, e3 = rows(a.end4), rows(a.end3)
    common = {r["suite"] for r in e4} & {r["suite"] for r in e3}                     # S734: the comparison run may carry more suites
    e4 = [r for r in e4 if r["suite"] in common]; e3 = [r for r in e3 if r["suite"] in common]
    assert len(e4) == len(e3) and all(x["gold"] == y["gold"] for x, y in zip(e4, e3))
    same = float(np.mean([x["R|rep"] == y["R|rep"] for x, y in zip(e4, e3)]))
    fch = max(res[n][s]["F"]["changed"] for n in res for s in res[n])
    checks["V4_LOCALITY"] = fch <= .01 + 1e-9 and same >= .99 - 1e-9
    print(f"end form: v4 vs v3 repaired answers identical on {same:.3f} · max forward change {fch:.3f}")
    print(f"checks {checks}")
    json.dump(dict(res=res, end_same=same, checks=checks), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("reverse", "backwards", "tolast", "end4", "end3"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--out", default="/vlm/results/r3/s717_analysis.json")
    main(ap.parse_args())
