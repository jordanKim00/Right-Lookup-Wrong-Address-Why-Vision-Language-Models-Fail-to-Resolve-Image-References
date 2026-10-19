"""S755 analysis, frozen with the pre-registration (knowledge/DECISIONS.md, 2026-10-13 02:xx) before any S755 data.
acc(form, d) = answers on image 7 - d.
FORM_FREE       for each d in {2, 3, 4}, the accuracies of F1, F2, F3, F5 span <= .15, and in each of these forms acc(d = 1) - acc(d = 3) >= .30
FORM_DEPENDENT  some form (F2, F3, F4, F5) differs from F1 by >= .20 at some d >= 2
ABSTRACT_CODE   at layer 34, each cross-form transplant (F4, F2, F3 d = 1 -> F1 d = 4) sends the recipient to image 6 in >= .5 x the same-form
                control (F1 d = 1 -> F1 d = 4)
Descriptive: every accuracy with scene-bootstrap 95% intervals (4,000 draws), answer distributions, START and "last" controls, layer 30.
Usage: python s755_analyze.py TAG
"""
import json, sys
import numpy as np


def boot(x, B=4000, seed=0):
    x = np.asarray(x, float); r = np.random.default_rng(seed)
    m = [x[r.integers(0, len(x), len(x))].mean() for _ in range(B)]
    return [round(float(x.mean()), 3), round(float(np.percentile(m, 2.5)), 3), round(float(np.percentile(m, 97.5)), 3)]


def analyze(tag):
    d_ = json.load(open(f"/vlm/results/r3/s755_{tag}_forms.json")); M = d_["meta"]; LAY = d_["LAY"]; LAY27 = d_["LAY27"]
    L = lambda key: np.array([m["land"][key] for m in M])
    acc = {}
    for key in M[0]["land"]:
        if key.startswith("X"):
            continue
        form, d = key.split("|"); d = int(d)
        acc[key] = boot(L(key) == (7 - d))
    out = dict(n=len(M), acc=acc, checks={}, dist={k: {str(b): round(float(np.mean(L(k) == b)), 3) for b in range(0, 8)} for k in M[0]["land"]})
    A = lambda f, d: acc[f"{f}|{d}"][0]
    forms = ["F1", "F2", "F3", "F5"]
    band = {d: max(A(f, d) for f in forms) - min(A(f, d) for f in forms) for d in (2, 3, 4)}
    law = {f: A(f, 1) - A(f, 3) for f in forms}
    out["checks"]["FORM_FREE"] = bool(all(b <= .15 + 1e-9 for b in band.values()) and all(x >= .30 - 1e-9 for x in law.values()))
    dev = {}
    for f in ("F2", "F3", "F5"):
        for d in (2, 3, 4):
            dev[f"{f}|{d}"] = round(A(f, d) - A("F1", d), 3)
    dev["F4|2"] = round(A("F4", 2) - A("F1", 2), 3)
    out["checks"]["FORM_DEPENDENT"] = bool(any(abs(x) >= .20 - 1e-9 for x in dev.values()))
    out["form_detail"] = dict(band={str(k): round(v, 3) for k, v in band.items()}, one_step_law={f: round(v, 3) for f, v in law.items()}, dev_from_F1=dev)
    top = LAY[LAY27.index(34)]
    ctrl = float(np.mean(L(f"X{top}|F1|1>F1|4") == 6))
    xs = {src: float(np.mean(L(f"X{top}|{src}>F1|4") == 6)) for src in ("F4|1", "F2|1", "F3|1")}
    out["checks"]["ABSTRACT_CODE"] = bool(all(x >= .5 * ctrl - 1e-9 for x in xs.values()))
    out["cross"] = {str(l27): {src: round(float(np.mean(L(f"X{l}|{src}>F1|4") == 6)), 3) for src in ("F1|1", "F4|1", "F2|1", "F3|1")}
                    for l27, l in zip(LAY27, LAY)}
    out["cross"]["base_F1_4_on6"] = round(float(np.mean(L("F1|4") == 6)), 3)
    print(f"[{tag}] checks {out['checks']}")
    print(f"[{tag}] form detail {out['form_detail']}")
    print(f"[{tag}] cross {out['cross']}")
    for f in ("F1", "F2", "F3", "F5", "S"):
        print(f"   {f}: " + " ".join(f"d{d} {A(f, d):.2f}" for d in (1, 2, 3, 4)))
    print(f"   F4: d1 {A('F4', 1):.2f} d2 {A('F4', 2):.2f} · LAST {A('LAST', 0):.2f}")
    return out


if __name__ == "__main__":
    res = {t: analyze(t) for t in sys.argv[1:]}
    json.dump(res, open("/vlm/results/r3/s755_analysis.json", "w"), indent=1)
