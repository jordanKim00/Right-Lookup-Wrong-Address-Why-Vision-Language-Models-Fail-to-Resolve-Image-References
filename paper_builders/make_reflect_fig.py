"""fig_c7_reflect: an address written without its frame returns to the literal image as often as the model composes that reference itself.
Data: research/ces/outputs/r3_lure_latest/reflect_rate.json (exp/reflect_rate.py; S730, S736, address test).  Style: exp/figs/style.py.
Usage: python figs/make_reflect_fig.py"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/vlm/exp/figs")
import style

OUT = "/vlm/results/figures/c_circuit"
d = json.load(open("/vlm/research/ces/outputs/r3_lure_latest/reflect_rate.json"))
COL = {"Qwen3-VL-4B": style.ACCENT, "Qwen3-VL-8B": "#7f93e0", "Qwen3.5-4B": style.MUTED, "Qwen3.5-9B": style.INK, "Qwen3.5-27B": style.TARGET}
MK = {"S730": "o", "S736": "s", "address test": "^"}
SRC = {"S730": "stated-count question", "S736": "stated-count question, $N$=5, 7, 9", "address test": "address-test question"}
style.apply()
fig, a = plt.subplots(figsize=(style.ONE, 2.25), constrained_layout=True)
a.plot([0, 1], [0, 1], ls="--", lw=.6, color=style.RULE, zorder=0)
for c in d["cells"]:
    a.scatter(c["compose"], c["reflect"], s=14, marker=MK[c["source"]], facecolor=COL[c["model"]], edgecolor="white", linewidth=.4, zorder=2)
for m, col in COL.items():
    a.scatter([], [], s=14, marker="o", facecolor=col, edgecolor="white", linewidth=.4, label=m)
for s, mk in MK.items():
    a.scatter([], [], s=14, marker=mk, facecolor="none", edgecolor=style.INK, linewidth=.6, label=SRC[s])
a.set_xlim(-.03, 1.03); a.set_ylim(-.03, 1.03)
a.set_xlabel("unedited accuracy on \u201cthe $t$-th image from the end\u201d")
a.set_ylabel("address $t$ alone: answers on literal image")
a.text(.97, .05, f"$r$ = {d['pearson']:.3f}, {d['n_cells']} cells", ha="right", va="bottom", fontsize=6.5)
a.legend(loc="upper left", fontsize=5.6, handletextpad=.2, labelspacing=.25, borderaxespad=.2)
style.save(fig, f"{OUT}/fig_c7_reflect"); plt.close(fig)
print("saved", f"{OUT}/fig_c7_reflect")
