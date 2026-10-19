"""S724 selection, frozen with the pre-registration (knowledge/DECISIONS.md, 2026-10-10) before any S724 data.
Per model, from the calibration measurements of exp/s724_readcal.py (no end-relative answer, no benchmark item):
    detector:  among abs (calibrated threshold in the model's detection space) and margin m in {.4, .5, .6} (chart), the highest recall of
               reference ordinals with the correct k_hat subject to false detections on non-ordinal, non-frame tokens <= .005 (ties: abs)
    binder:    among prototype frame tests within the clause at radius x1.0 / x1.2 / x1.5 and the earlier learned frame gate (g > .5, smaller
               models only), the highest balanced accuracy (END-bound recall + 1 - non-END false binding) / 2 subject to false binding <= .02
               (binding is scored on references the respective detector found; ties: prototype x1.0)
    count:     max code or last-block code, the higher accuracy of N_hat = N (ties: last block)
Reported separately for procedural and natural images.  Writes /vlm/results/r3/s724_selection.json.
Amendment (before data): the component metrics and choice use single-frame inputs; mixed-frame inputs are reported and enter exp/s724_plan.py.
Usage: python s724_select.py TAG [TAG ...]
"""
import json, sys
from collections import defaultdict
import numpy as np

ORD = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
FRAMEW = {"end", "last", "start", "beginning"}


def is_boundary(s):
    st = s.strip()
    return "\n" in s or (st != "" and st[0] in ".?!")


def window(toks, i, w=8):
    L, q = [], i - 1
    while q >= 0 and len(L) < w and not is_boundary(toks[q]):
        L.append(q); q -= 1
    R, q = [], i + 1
    while q < len(toks) and len(R) < w and not is_boundary(toks[q]):
        R.append(q); q += 1
    return L[::-1] + R


def refs(rec):
    if "gold" in rec:                                                              # amendment: per-reference gold (mixed-frame items)
        return [(g["pos"], g["k"]) for g in rec["gold"]]
    t = rec["toks"]; out = []
    for i, w in enumerate(t):
        ww = w.strip().lower(); nx = t[i + 1].strip().lower() if i + 1 < len(t) else ""
        if ww in ORD and nx == "image":
            out.append((i, ORD.index(ww) + 1))
    return out


def is_end(rec, i):
    if "gold" in rec:
        return any(g["pos"] == i and g["frame"] == "end" for g in rec["gold"])
    return rec["kind"] == "end"


def detect(rec, rule):
    n = len(rec["toks"])
    if rule == "abs":
        if rec["det_space"] == "full":
            return [rec["df"][i] <= rec["th_full"] for i in range(n)], rec["kh_f"]
        return [rec["dc"][i] <= rec["th_chart"] for i in range(n)], rec["kh_c"]
    m = float(rule.split(":")[1])
    return [rec["dc"][i] / max(rec["dc2"][i], 1e-6) <= m for i in range(n)], rec["kh_c"]


def analyze(tag):
    d = json.load(open(f"/vlm/results/r3/s724_{tag}_readcal.json")); R = d["rec"]
    res = {}
    S1 = [r for r in R if r.get("kind") != "mixed"]                               # component rules: single-frame inputs (as registered);
    MX = [r for r in R if r.get("kind") == "mixed"]                                # mixed-frame inputs enter the plan-level rule (s724_plan.py)
    for subset, rs in (("all", S1), ("procedural", [r for r in S1 if not r["natural"]]), ("natural", [r for r in S1 if r["natural"]]), ("mixed", MX)):
        if not rs:
            continue
        det = {}
        for rule in ("abs", "margin:.4", "margin:.5", "margin:.6"):
            hit = n = fp = no = 0
            for r in rs:
                m, kh = detect(r, rule); rf = dict(refs(r))
                for i, w in enumerate(r["toks"]):
                    if i in rf:
                        n += 1; hit += bool(m[i]) and kh[i] == rf[i]
                    elif w.strip().lower() not in ORD and w.strip().lower() not in FRAMEW:
                        no += 1; fp += bool(m[i])
            det[rule] = dict(recall=hit / max(n, 1), fp=fp / max(no, 1), n=n)
        bind = {}
        for s in (1.0, 1.2, 1.5):
            er = en = nf = nn = 0
            for r in rs:
                for i, k in refs(r):
                    anch = any(r["d_end"][q] <= s * r["thr"]["end"] or r["d_last"][q] <= s * r["thr"]["last"] for q in window(r["toks"], i))
                    if is_end(r, i):
                        en += 1; er += anch
                    else:
                        nn += 1; nf += anch
            bind[f"proto:{s}"] = dict(end_recall=er / max(en, 1), false_bind=nf / max(nn, 1))
        if all("v1" in r for r in rs) and rs:
            er = en = nf = nn = 0
            for r in rs:
                g = {x["pos"]: x["g"] for x in r["v1"]}
                for i, k in refs(r):
                    b = g.get(i) is not None and g[i] > .5
                    if is_end(r, i):
                        en += 1; er += b
                    else:
                        nn += 1; nf += b
            bind["v1_gate"] = dict(end_recall=er / max(en, 1), false_bind=nf / max(nn, 1))
        for b_ in bind.values():
            b_["balanced"] = (b_["end_recall"] + 1 - b_["false_bind"]) / 2
        cnt = dict(max=float(np.mean([max(r["codes"]) == r["N"] for r in rs])), last=float(np.mean([r["codes"][-1] == r["N"] for r in rs])))
        res[subset] = dict(detector=det, binder=bind, count=cnt)
    A = res["all"]
    ok = {k: v for k, v in A["detector"].items() if v["fp"] <= .005 + 1e-12}
    det_choice = max(ok, key=lambda k: (round(ok[k]["recall"], 6), k == "abs")) if ok else "abs"
    okb = {k: v for k, v in A["binder"].items() if v["false_bind"] <= .02 + 1e-12}
    bind_choice = max(okb, key=lambda k: (round(okb[k]["balanced"], 6), k == "proto:1.0")) if okb else "proto:1.0"
    cnt_choice = "last" if A["count"]["last"] >= A["count"]["max"] - 1e-12 else "max"
    print(f"[{tag}] detector: " + " · ".join(f"{k} rec {v['recall']:.3f} fp {v['fp']:.4f}" for k, v in A["detector"].items()) + f"  -> {det_choice}")
    print(f"[{tag}] binder:   " + " · ".join(f"{k} end {v['end_recall']:.3f} false {v['false_bind']:.3f}" for k, v in A["binder"].items()) + f"  -> {bind_choice}")
    print(f"[{tag}] count:    max {A['count']['max']:.3f} · last {A['count']['last']:.3f}  -> {cnt_choice}")
    for sub in ("procedural", "natural", "mixed"):
        if sub not in res:
            continue
        S = res[sub]
        print(f"[{tag}]   {sub}: det(choice) rec {S['detector'][det_choice]['recall']:.3f} fp {S['detector'][det_choice]['fp']:.4f} · bind(choice) end {S['binder'][bind_choice]['end_recall']:.3f} false {S['binder'][bind_choice]['false_bind']:.3f} · count {S['count'][cnt_choice]:.3f}")
    return dict(metrics=res, choice=dict(detector=det_choice, binder=bind_choice, count=cnt_choice))


if __name__ == "__main__":
    import os
    P = "/vlm/results/r3/s724_selection.json"
    out = json.load(open(P)) if os.path.exists(P) else {}                          # one file for all models (runs finish at different times)
    out.update({t: analyze(t) for t in sys.argv[1:]})
    json.dump(out, open(P, "w"), indent=1)
