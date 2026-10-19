"""fig_natmed (S740, MMIU development photographs, option letters; FINDINGS §784 / §786): the repair runs through the full-attention queries.
Items repaired by R3 (R3 correct, unedited wrong; 150 per model).  Left: the rebound run with the queries of every full-attention head (dark) or
of the index-matching heads only (light) put back to the unrepaired run's, share still correct.  Right: the unrepaired run given the rebound
queries, share repaired.  Style: exp/figs/style.py.  Output: /vlm/results/figures/c_circuit/fig_natmed.{pdf,png}"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/vlm/exp/figs")
import style

R, OUT = "/vlm/results/r3", "/vlm/results/figures/c_circuit"
M = (("35_4b", "3.5-4B"), ("35_9b", "3.5-9B"), ("27b", "3.5-27B"))
V = {}
for tag, _ in M:
    rows = [r for r in json.load(open(f"{R}/s740_nm740_{tag}_natmed.json"))["rows"] if "pick_R3_QBACK_ALL" in r]
    resc = [r for r in rows if r["pick_R3"] == r["gold"] and r["pick_END"] != r["gold"]]
    V[tag] = {k: float(np.mean([r[k] == r["gold"] for r in resc])) for k in ("pick_R3_QBACK_ALL", "pick_R3_QBACK_M", "pick_END_QALL", "pick_END_QM")}
style.apply()
LIGHT = "#9fd3cb"
fig, axes = plt.subplots(1, 2, figsize=(style.ONE, 1.6), sharey=True, gridspec_kw=dict(wspace=.08))
x = np.arange(len(M)); w = .36
for ax, (title, ka, km) in zip(axes, (("rebound run, queries put back", "pick_R3_QBACK_ALL", "pick_R3_QBACK_M"),
                                      ("unrepaired run, rebound queries given", "pick_END_QALL", "pick_END_QM"))):
    for dx, key, col, lab in ((-w / 2, ka, style.TARGET, "every full-attention head"), (w / 2, km, LIGHT, "index-matching heads only")):
        vals = [100 * V[t][key] for t, _ in M]
        ax.bar(x + dx, vals, w, color=col, label=lab)
        for xi, v in zip(x + dx, vals):
            ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=5.2)
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in M], fontsize=5.8); ax.tick_params(axis="x", length=0)
    ax.set_title(title, fontsize=6.2, pad=2); ax.set_ylim(0, 108); ax.set_yticks([0, 25, 50, 75, 100])
axes[0].set_ylabel("repaired items correct (%)", fontsize=6.0, labelpad=1)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", bbox_to_anchor=(.53, .96), ncol=2, fontsize=5.6, handlelength=1.2, columnspacing=1.)
os.makedirs(OUT, exist_ok=True); style.save(fig, f"{OUT}/fig_natmed"); plt.close(fig)
print("ok fig_natmed", {t: {k: round(v, 3) for k, v in d.items()} for t, d in V.items()})
