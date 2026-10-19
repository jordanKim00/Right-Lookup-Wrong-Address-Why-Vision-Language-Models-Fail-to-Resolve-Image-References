"""S716 analysis, frozen with the pre-registration (knowledge/DECISIONS.md, 2026-10-09) before any S716 (R³-v3) data.
R³-v3 = R³-v2 with the READ rule chosen on the sealed development measurement (S715): an ordinal source is a token whose chart distance
to the nearest ordinal prototype is at most half of that to the second nearest (margin .5), and frame anchors are accepted within 1.2x
their calibrated radius.  Paired with the R³-v2 records on the same items (same harness, same order).
    V3_FRESH_GAIN:    later MMIU split, end-relative form: P(v3) - P(v2) >= .03 and the paired 95% bootstrap interval excludes 0
    V3_FRESH_CEILING: P(R | v3) >= .93 P(F | base) on the later split (text normalization = the forward form)
    V3_LOCALITY:      forward-form answers changed by v3 on <= .01 of the later-split items
    V3_CLOZE (report): visual_cloze end-relative options, v3 vs v2
Usage: python s716_analyze.py --fresh V2TAG V3TAG [--cloze V2TAG V3TAG] [--extra NAME V2TAG V3TAG ...]
"""
import argparse, json
import numpy as np

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]


def load(tag):
    return json.load(open(f"/vlm/results/s582_evalrepair_{tag}.json"))["rows"]


def pair(t2, t3, suite):
    a = [r for r in load(t2) if r["suite"] == suite]; b = [r for r in load(t3) if r["suite"] == suite]
    assert len(a) == len(b) and all(x["gold"] == y["gold"] and x["task"] == y["task"] for x, y in zip(a, b)), "records not aligned"
    agree = np.mean([x["F|base"] == y["F|base"] and x["R|base"] == y["R|base"] for x, y in zip(a, b)])
    print(f"  ({suite}: base answers identical across the two runs on {agree:.3f} of items)")
    return a, b


def boot(dd):
    rng = np.random.default_rng(0)
    bs = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(4000)]
    return float(dd.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def compare(name, t2, t3, suite):
    a, b = pair(t2, t3, suite)
    g = lambda rs, k: np.array([r[k] == r["gold"] for r in rs], float)
    base, v2, v3 = g(a, "R|base"), g(a, "R|rep"), g(b, "R|rep")
    F = g(a, "F|base"); Fch = float(np.mean([r["F|rep"] != r["F|base"] for r in b]))
    d, lo, hi = boot(v3 - v2)
    print(f"[{name} {suite}] n {len(a)}  F {F.mean():.3f} | R base {base.mean():.3f} · v2 {v2.mean():.3f} · v3 {v3.mean():.3f}  "
          f"v3−v2 {d:+.3f} [{lo:+.3f}, {hi:+.3f}]  (v3 구출 {int(((v2 == 0) & (v3 == 1)).sum())} · 손상 {int(((v2 == 1) & (v3 == 0)).sum())})  F 바뀜 {Fch:.3f}")
    return dict(n=len(a), F=float(F.mean()), base=float(base.mean()), v2=float(v2.mean()), v3=float(v3.mean()), d=d, lo=lo, hi=hi, F_changed=Fch)


def main(a):
    res = {"fresh": compare("fresh", a.fresh[0], a.fresh[1], "mmiu_pos")}
    f = res["fresh"]
    checks = dict(V3_FRESH_GAIN=f["d"] >= .03 - 1e-9 and f["lo"] > 0, V3_FRESH_CEILING=f["v3"] >= .93 * f["F"] - 1e-9, V3_LOCALITY=f["F_changed"] <= .01 + 1e-9)
    if a.cloze:
        res["cloze"] = compare("cloze", a.cloze[0], a.cloze[1], "cloze")
    for i in range(0, len(a.extra or []), 3):
        nm, t2, t3 = a.extra[i:i + 3]
        for s in ("mmiu_pos", "muir"):
            try:
                res[f"{nm}|{s}"] = compare(nm, t2, t3, s)
            except (AssertionError, ZeroDivisionError, ValueError, KeyError):
                pass
    print(f"checks {checks}")
    json.dump(dict(res=res, checks=checks), open("/vlm/results/r3/s716_analysis.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", nargs=2, required=True); ap.add_argument("--cloze", nargs=2); ap.add_argument("--extra", nargs="*")
    main(ap.parse_args())
