"""fig_recompose (S786 records, bare-token readout = generation; FINDINGS §852): an address written under the end frame is interpreted as a
natural end-relative ordinal.  For each model and k in {2, 3, 5, 6} (j = N + 1 - k), the share of address-only END(k) answers on image x
against the share of natural END(j) answers on image x of the same scenes, for x = the literal image k and x = the target j.
Style: exp/figs/style.py.  Output: /vlm/results/figures/c_circuit/fig_recompose.{pdf,png}"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/vlm/exp/figs")
import style

R, OUT, N = "/vlm/results/r3", "/vlm/results/figures/c_circuit", 7
M = (("s786_4b", "Qwen3-VL-4B", "s"), ("s786_8b", "Qwen3-VL-8B", "D"), ("s786_35_9b", "Qwen3.5-9B", "^"), ("s786_27b", "Qwen3.5-27B", "o"))
style.apply()
fig, ax = plt.subplots(figsize=(style.ONE, 2.25))
ax.plot([0, 1], [0, 1], color=style.RULE, lw=.8, zorder=0)
agree = {}
for tag, name, mk in M:
    rows = json.load(open(f"{R}/s730_{tag}_stagewise.json"))["rows"]; by = {(r["si"], r["k"]): r for r in rows}
    for xkey, col in (("k", style.SURFACE), ("j", style.TARGET)):
        xs, ys = [], []
        for k in (2, 3, 5, 6):
            j = N + 1 - k; img = k if xkey == "k" else j
            rk = [r for r in rows if r["k"] == k]; rj = [r for r in rows if r["k"] == j]
            ys.append(np.mean([r["land_ADDR"] == img for r in rk])); xs.append(np.mean([r["land_END"] == img for r in rj]))
        ax.scatter(xs, ys, s=16, marker=mk, facecolor=col, edgecolor="white", lw=.4, zorder=3)
    a = [by[(si, k)]["land_ADDR"] == by[(si, N + 1 - k)]["land_END"] for (si, k) in by if k != 4 and (si, N + 1 - k) in by]
    agree[name] = np.mean(a)
ax.set_xlim(-.03, 1.03); ax.set_ylim(-.03, 1.03); ax.set_xticks([0, .5, 1]); ax.set_yticks([0, .5, 1])
ax.set_xlabel("natural “the $j$-th image from the end”", labelpad=1)
ax.set_ylabel("address $j$ written into “the $k$-th …”", labelpad=1)
from matplotlib.lines import Line2D
h = [Line2D([], [], marker=mk, ls="", color=style.INK, ms=4, label=name) for _, name, mk in M]
h += [Line2D([], [], marker="o", ls="", color=style.SURFACE, ms=4, label="answers on the literal image $k$"),
      Line2D([], [], marker="o", ls="", color=style.TARGET, ms=4, label="answers on the target $j$")]
ax.legend(handles=h, loc="upper left", fontsize=5.4, handletextpad=.3, labelspacing=.25)
os.makedirs(OUT, exist_ok=True); style.save(fig, f"{OUT}/fig_recompose"); plt.close(fig)
print("ok fig_recompose", {k: round(v, 3) for k, v in agree.items()})
