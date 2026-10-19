"""Benchmark table and figure for the revised paper, built only from saved S582-format records.
Rows: five models; sets: MMIU first-use (sealed), MuirBench first-use, MMIU later split, MMIU cloze.
Columns: forward (= text normalization of the same items), original end-relative, R³ v1 (paper module, small models), R³ latest
(R³-v3; for 27B v3 with v2 reported in the appendix).  Writes /vlm/results/figures/bench/{bench_main.tex, bench_values.json, fig_bench.*}.
Usage: python figs/make_bench.py
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/vlm/exp/figs")
import style

R = "/vlm/results"
OUT = "/vlm/results/figures/bench"
MODELS = [("Qwen3-VL-4B", "s609c_4b", "4b"), ("Qwen3-VL-8B", "s609c_8b", "8b"), ("Qwen3.5-4B", "s619g_35_4b", "35_4b"),
          ("Qwen3.5-9B", "s619g_35_9b", "35_9b"), ("Qwen3.5-27B", None, "27b")]
SETS = [("MMIU first-use", "sealed", "mmiu_pos"), ("MUIR first-use", "sealed", "muir"), ("MMIU later", "later", "mmiu_pos"), ("MMIU cloze", "cloze", "cloze")]


def tag(kind, set_, v1, b):
    if b == "27b":
        return {("latest", "sealed"): "m3sl3_27b", ("latest", "later"): "m3fr3_27b", ("latest", "cloze"): "m3cz3_27b",
                ("v2", "sealed"): "m3sl_27b", ("v2", "later"): "m3fr_27b", ("v2", "cloze"): "m3cz_27b"}.get((kind, set_))
    if kind == "latest":
        return {"sealed": f"lmsl_{b}", "later": f"lmfr_{b}", "cloze": f"lmcz_{b}"}[set_]
    if kind == "v1":
        return {"sealed": f"s601_{v1}", "later": f"fresh_{v1}", "cloze": (f"e582_{v1}" if v1.startswith("s609c") else f"cloze_{v1}")}[set_]


def acc(t, suite):
    p = f"{R}/s582_evalrepair_{t}.json"
    if not t or not os.path.exists(p):
        return None
    rows = [r for r in json.load(open(p))["rows"] if r["suite"] == suite]
    g = lambda k: float(np.mean([r[k] == r["gold"] for r in rows]))
    return dict(n=len(rows), F=g("F|base"), R=g("R|base"), rep=g("R|rep"), Fch=float(np.mean([r["F|rep"] != r["F|base"] for r in rows])))


def main():
    os.makedirs(OUT, exist_ok=True)
    vals = {}
    for name, v1, b in MODELS:
        for sname, set_, suite in SETS:
            lat = acc(tag("latest", set_, v1, b), suite)
            old = acc(tag("v1" if b != "27b" else "v2", set_, v1, b), suite)
            vals[f"{name}|{sname}"] = dict(latest=lat, v1=old)
    json.dump(vals, open(f"{OUT}/bench_values.json", "w"), indent=1)
    pct = lambda x: "--" if x is None else f"{100 * x:.1f}"
    lines = ["\\begin{tabular}{llrrrr}", "\\toprule", "Model & Set & Fwd. & Rev. & \\rthree{} v1 & \\rthree{} \\\\", "\\midrule"]
    for name, v1, b in MODELS:
        for sname, set_, suite in SETS:
            v = vals[f"{name}|{sname}"]; L, O = v["latest"], v["v1"]
            base = L or O
            if base is None:
                continue
            lines.append(f"{name} & {sname} & {pct(base['F'])} & {pct(base['R'])} & {pct(O['rep'] if O else None)} & {pct(L['rep'] if L else None)} \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"; lines.append("\\end{tabular}")
    open(f"{OUT}/bench_main.tex", "w").write("\n".join(lines) + "\n")
    style.apply()
    fig, ax = plt.subplots(1, 3, figsize=(style.TWO, 1.9), sharey=True, constrained_layout=True)
    for a, (sname, set_, suite) in zip(ax, SETS[:3]):
        for i, (name, v1, b) in enumerate(MODELS):
            v = vals[f"{name}|{sname}"]; L = v["latest"] or v["v1"]
            if L is None:
                continue
            a.plot([L["R"], L["rep"]], [i, i], "-", color=style.RULE, lw=2.2, zorder=1)
            a.scatter([L["R"]], [i], color=style.SURFACE, s=14, zorder=2)
            a.scatter([L["rep"]], [i], color=style.TARGET, s=14, zorder=3)
            a.scatter([L["F"]], [i], marker="|", color=style.INK, s=60, zorder=4)
            if v["v1"] is not None and v["latest"] is not None:
                a.scatter([v["v1"]["rep"]], [i], marker="o", facecolors="none", edgecolors=style.ACCENT, s=16, lw=.8, zorder=3)
        a.set_title(sname); a.set_xlim(.15, .95); a.set_xlabel("accuracy")
        a.set_yticks(range(len(MODELS))); a.set_yticklabels([m for m, _, _ in MODELS])
    ax[0].invert_yaxis()
    style.save(fig, f"{OUT}/fig_bench"); plt.close(fig)
    print("ok", json.dumps({k: (v["latest"]["rep"] if v["latest"] else None) for k, v in vals.items()})[:600])


if __name__ == "__main__":
    main()
