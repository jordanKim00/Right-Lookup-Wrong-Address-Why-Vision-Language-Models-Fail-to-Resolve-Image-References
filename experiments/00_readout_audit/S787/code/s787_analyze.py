"""S787 analysis, small models (registration knowledge/DECISIONS.md 2026-10-14 20:3x: the original analysis rules, re-run on the S787 re-measure).
Four parts, each the original rule unchanged; only the input tags change.
  1. S652 fit (exp/s652_fit.py): I1 all codes +2, I2 last code -1, I3 last code +1, I4 forward-ordinal transplant; A_copy vs A_comp landing rates,
     item bootstrap 2,000, seed 65200.  The loop below is s652_fit.main's per-model block with a parameterized tag map; positive control: on the
     original tags it must reproduce research/ces/outputs/s652_fit/s652_fit.json exactly.
  2. S658 f rule (DECISIONS 2026-10-07 13:36): f = (acc(QRP) - acc(QR)) / (acc(QFP) - acc(QR)) per preposed wording; f >= .70 -> the postposed
     frame's causal unavailability explains most of the failure; f <= .30 and surf(QRP) >= .5 surf(QR) -> early dereferencing persists with the
     frame visible; in between mixed; claimed only when both wordings fall on the same side; withheld if acc(QFP) < acc(QF) - .15.
     Scene bootstrap 10,000, seed 65800.
  3. S736 checks (exp/s736_analyze.py, imported): FRAME_NEEDED_NEAR, FRAME_SEMANTIC, FRAME_IDLE_FAR.
  4. Re-reflection = composition law (exp/reflect_rate.py, from_rows copied; §773/§780): per (model, N, t) compose(t) vs reflect(t); pearson r,
     mean / max |d|, and the inner cells (t not 1 or N).  Positive control: on the original fr736_* files the copied function must reproduce the
     S736 cells of research/ces/outputs/r3_lure_latest/reflect_rate.json.
Readout: R3_READ=max (two-form, S794); these instruments do not record the vocabulary top-1 check (t1), so the results are safe estimates (B),
not in-run verified (checklist item 1).
Usage (docker, /vlm mount): python /vlm/exp/s787_analyze.py -> /vlm/results/r3/s787_analysis.json"""
from __future__ import annotations

import json, os, random, sys
import numpy as np

sys.path.insert(0, "/vlm/exp")
import s652_fit as F
import s736_analyze as A736

R3 = "/vlm/results/r3"
OUTS = "/vlm/research/ces/outputs"
SMALL = ["4b", "8b", "35_4b", "35_9b"]


def fit(models):                                                     # s652_fit.main, per-model block unchanged, tag map as argument
    rng = random.Random(65200); out = {}
    boot, load, R = F.boot, F.load, F.R
    for m, (t1, t2, t4) in models.items():
        res = {}
        rows = load(f"s545_indexshift_{t1}.json") if t1 else None
        if rows:
            sel = [r for r in rows if r["k"] >= 3 and r["k"] - 2 != r["j"]]
            res["I1"] = dict(copy=boot([r["R|pre"] == r["k"] - 2 for r in sel], rng), comp=boot([r["R|pre"] == r["j"] for r in sel], rng),
                             count=boot([r["R|pre"] == r["j"] - 2 for r in sel], rng),
                             base_copy=boot([r["R|none"] == r["k"] for r in sel], rng), base_comp=boot([r["R|none"] == r["j"] for r in sel], rng))
        rows = load(f"s549_endshift_{t2}.json")
        if rows:
            sel = [r for r in rows if r["j"] != r["k"] and r["j"] - 1 != r["k"]]
            res["I2"] = dict(copy=boot([r["R|m1"] == r["k"] for r in sel], rng), comp=boot([r["R|m1"] == r["j"] - 1 for r in sel], rng),
                             count=boot([r["R|m1"] == r["j"] for r in sel], rng), base_comp_target=boot([r["R|none"] == r["j"] - 1 for r in sel], rng))
            sel = [r for r in rows if r["j"] != r["k"] and r["j"] + 1 != r["k"] and r["j"] + 1 <= r["N"]]
            res["I3"] = dict(copy=boot([r["R|p1"] == r["k"] for r in sel], rng), comp=boot([r["R|p1"] == r["j"] + 1 for r in sel], rng),
                             jm1=boot([r["R|p1"] == r["j"] - 1 for r in sel], rng), base_comp_target=boot([r["R|none"] == r["j"] + 1 for r in sel], rng))
            sel = [r for r in rows if r["j"] != r["k"]]
            res["I0"] = dict(copy=boot([r["R|none"] == r["k"] for r in sel], rng), comp=boot([r["R|none"] == r["j"] for r in sel], rng))
        d = json.load(open(f"{R}/s474b_fwddonor_{t4}.json")) if os.path.exists(f"{R}/s474b_fwddonor_{t4}.json") else None
        if d:
            L = d["res"]["layers"]; D = max(L) + 1
            nl = 64 if m == "27b" else (32 if m.startswith("35") else 36)
            lay = min(L, key=lambda l: abs(l / (nl - 1) - .13))
            sel = [r for r in d["rows"] if r["donor_ok"] and r["x"] != r["N"] + 1 - r["x"] and f"ord|{lay}" in r["grid"]]
            res["I4"] = dict(layer=lay, copy=boot([r["grid"][f"ord|{lay}"] == r["x"] for r in sel], rng),
                             comp=boot([r["grid"][f"ord|{lay}"] == r["N"] + 1 - r["x"] for r in sel], rng))
        out[m] = res
    return out


def s658(tag, B=10000):
    rows = json.load(open(f"{R3}/s658_{tag}_preposed.json"))["rows"]
    acc = lambda rr, f: float(np.mean([r[f]["ok"] for r in rr]))
    surf = lambda rr, f: float(np.mean([r[f]["surf"] for r in rr]))
    def fval(rr, p):
        den = acc(rr, "QFP") - acc(rr, "QR")
        return (acc(rr, p) - acc(rr, "QR")) / den if den > 0 else float("nan")
    rng = np.random.default_rng(65800); n = len(rows)
    out = dict(n=n, acc={f: acc(rows, f) for f in ("QF", "QFP", "QR", "QRP1", "QRP2", "OR", "ORP")},
               surf={f: surf(rows, f) for f in ("QF", "QFP", "QR", "QRP1", "QRP2")},
               lure={f: float(np.mean([r[f]["lure"] for r in rows])) for f in ("OR", "ORP")})
    out["hold"] = bool(out["acc"]["QFP"] < out["acc"]["QF"] - .15)
    side = {}
    for p in ("QRP1", "QRP2"):
        f = fval(rows, p); bs = []
        for _ in range(B):
            rr = [rows[i] for i in rng.integers(0, n, n)]
            x = fval(rr, p)
            if np.isfinite(x):
                bs.append(x)
        ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if bs else None
        persists = f <= .30 and out["surf"][p] >= .5 * out["surf"]["QR"]
        side[p] = "mask" if f >= .70 else ("persists" if persists else ("low_f_surf_fails" if f <= .30 else "mixed"))
        out[f"f_{p}"] = dict(f=float(f), ci=ci, side=side[p])
    out["verdict"] = "withheld (QFP hard)" if out["hold"] else (side["QRP1"] if side["QRP1"] == side["QRP2"] else "no claim (wordings differ)")
    return out


def reflect_cells(R, src, model, by_n=False):                       # exp/reflect_rate.py from_rows, unchanged, returning the cells
    cells = []
    for N in sorted({r.get("N", 7) for r in R}) if by_n else [None]:
        rows = [r for r in R if (N is None or r.get("N", 7) == N)]
        for t in sorted({r["j"] for r in rows}):
            comp = [r["land_END"] == r["j"] for r in rows if r["k"] == t and r["j"] != r["k"]]
            refl = [r["land_ADDR"] == r["k"] for r in rows if r["j"] == t and r["j"] != r["k"]]
            acc = [r["land_ADDR"] == r["j"] for r in rows if r["j"] == t and r["j"] != r["k"]]
            if comp and refl:
                cells.append(dict(source=src, model=model, N=N or 7, t=t, compose=float(np.mean(comp)), reflect=float(np.mean(refl)),
                                  address_acc=float(np.mean(acc)), n=len(refl)))
    return cells


def law(cells):
    if len(cells) < 3:
        return None
    x = np.array([c["compose"] for c in cells]); y = np.array([c["reflect"] for c in cells])
    return dict(n_cells=len(cells), pearson=float(np.corrcoef(x, y)[0, 1]), mean_abs_diff=float(np.mean(np.abs(x - y))),
                max_abs_diff=float(np.max(np.abs(x - y))))


if __name__ == "__main__":
    rep = {}
    # positive controls first
    stored = json.load(open(f"{OUTS}/s652_fit/s652_fit.json"))
    rep["control_s652"] = bool(json.loads(json.dumps(fit(F.MODELS))) == stored)
    names = {"4b": "Qwen3-VL-4B", "8b": "Qwen3-VL-8B", "35_4b": "Qwen3.5-4B", "35_9b": "Qwen3.5-9B", "27b": "Qwen3.5-27B"}
    old = []
    for m in ("4b", "8b", "35_4b", "35_9b", "27b"):
        p = f"{R3}/s736_fr736_{m}_frame.json"
        if os.path.exists(p):
            old += reflect_cells(json.load(open(p))["rows"], "S736", names[m], by_n=True)
    ref = [c for c in json.load(open(f"{OUTS}/r3_lure_latest/reflect_rate.json"))["cells"] if c["source"] == "S736"]
    rep["control_reflect"] = bool(json.loads(json.dumps(old)) == ref)
    print(f"positive controls: S652 fit reproduces the stored fit {rep['control_s652']} · reflect cells reproduce the stored S736 cells {rep['control_reflect']}")
    # 1. S652 fit on the S787 re-measure
    tags = {m: (f"s787_{m}", f"s787_{m}", f"s787_{m}") for m in SMALL}
    rep["fit"] = fit(tags)
    for m, res in rep["fit"].items():
        print(f"[fit {m}] " + " | ".join(f"{i}: copy {v['copy'][0]} [{v['copy'][1]}, {v['copy'][2]}] comp {v['comp'][0]} [{v['comp'][1]}, {v['comp'][2]}] (n {v['comp'][3]})"
                                        + (f" count {v['count'][0]}" if v.get('count') else "") + (f" j-1 {v['jm1'][0]}" if v.get('jm1') else "")
                                        + (f" L{v['layer']}" if 'layer' in v else "") for i, v in sorted(res.items())))
    # 2. S658 f rule
    rep["s658"] = {}
    for m in SMALL:
        try:
            o = rep["s658"][m] = s658(f"s787_{m}")
        except FileNotFoundError:
            print(f"[s658 {m}] missing"); continue
        print(f"[s658 {m}] n {o['n']} acc " + " ".join(f"{k} {v:.2f}" for k, v in o["acc"].items()) + " | surf " + " ".join(f"{k} {v:.2f}" for k, v in o["surf"].items())
              + f" | f1 {o['f_QRP1']['f']:.2f} {o['f_QRP1']['ci']} {o['f_QRP1']['side']} · f2 {o['f_QRP2']['f']:.2f} {o['f_QRP2']['ci']} {o['f_QRP2']['side']} -> {o['verdict']}")
    # 3. S736 checks
    rep["s736"] = {}
    for m in SMALL:
        try:
            rep["s736"][m] = A736.analyze(f"s787_{m}")
        except FileNotFoundError:
            print(f"[s736 s787_{m}] missing")
    # 4. re-reflection = composition
    cells = []
    for m in SMALL:
        p = f"{R3}/s736_s787_{m}_frame.json"
        if os.path.exists(p):
            cells += reflect_cells(json.load(open(p))["rows"], "S787/S736", names[m], by_n=True)
    inner = [c for c in cells if c["t"] not in (1, c["N"])]
    rep["reflect"] = dict(cells=cells, all=law(cells), inner=law(inner),
                          per_model={names[m]: law([c for c in inner if c["model"] == names[m]]) for m in SMALL})
    print(f"[reflect] all {rep['reflect']['all']} · inner {rep['reflect']['inner']}")
    for k_, v_ in rep["reflect"]["per_model"].items():
        print(f"   {k_} inner {v_}")
    json.dump(rep, open(f"{R3}/s787_analysis.json", "w"), indent=1, default=float)
    print(f"-> {R3}/s787_analysis.json")
