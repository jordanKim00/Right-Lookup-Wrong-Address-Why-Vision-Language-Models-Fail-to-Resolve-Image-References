"""fig_restore (main-text Figure 6): the rebound reference reads the intended image, through the queries.
(a, b) S780 re-measured with the bare-token readout (S783a; Qwen3.5-27B, 80 new scenes at N = 7 dark / 8 light): share of answers that change
to a new picture placed at the named position, and share changed by reversing the options.  (c, d) S740 (MMIU development photographs,
option letters): items repaired by R3; the rebound run with every full-attention query (dark) or only the index-matching heads' queries
(light) put back, and the unrepaired run given them.  Style: exp/figs/style.py.  Output: /vlm/results/figures/c_circuit/fig_restore.{pdf,png}"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/vlm/exp/figs")
import style
from make_fig_frame_bind import wilson, pick

R, OUT = "/vlm/results/r3", "/vlm/results/figures/c_circuit"
LIGHT_T, LIGHT_A, LIGHT_S = "#9fd3cb", "#aab5e6", "#efb894"
style.apply()
fig = plt.figure(figsize=(style.TWO, 1.6))
gs = fig.add_gridspec(1, 4, width_ratios=[3.2, 1.25, 1.75, 1.75], wspace=.42)
ax, bx, cx, dx = (fig.add_subplot(gs[0, i]) for i in range(4))
B = json.load(open(pick("s780_analysis"))); c7, c8 = (B["s783_27b7"], B["s783_27b8"])
groups = [("R³:\nwritten\naddress", "visual_r3", "j", style.TARGET, LIGHT_T), ("END(2):\ntarget", "visual_e2", "j", style.ACCENT, LIGHT_A),
          ("END(5):\nimage it\nnamed", "visual_end", "a", style.SURFACE, LIGHT_S), ("END(5):\ntarget", "visual_end", "j", style.SURFACE, LIGHT_S)]
w = .36
for i, (lab, key, f, c1, c2) in enumerate(groups):
    for d, c, col in ((-w / 2, c7, c1), (w / 2, c8, c2)):
        p, n = c[key][f], c[key]["n"]; lo, hi = wilson(p, n)
        ax.bar(i + d, 100 * p, w, color=col, yerr=[[100 * lo], [100 * hi]], error_kw=dict(lw=.5, capsize=1.2, ecolor=style.INK))
        ax.text(i + d, 100 * (p + hi) + 2, f"{100 * p:.0f}", ha="center", va="bottom", fontsize=5.2)
ax.set_xticks(range(4)); ax.set_xticklabels([g[0] for g in groups], fontsize=5.2); ax.tick_params(axis="x", length=0)
ax.set_ylim(0, 112); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_ylabel("answer follows the new picture (%)", fontsize=5.9, labelpad=1)
ax.set_title("(a) a new picture at the address", fontsize=6.3, pad=2)
for i, (lab, key, col, c2) in enumerate((("END(5)", "END", style.SURFACE, LIGHT_S), ("R³", "R3", style.TARGET, LIGHT_T))):
    for d, c, cc in ((-w / 2, c7, col), (w / 2, c8, c2)):
        p = c["menu_reversal_change"][key]; lo, hi = wilson(p, c["n"])
        bx.bar(i + d, 100 * p, w, color=cc, yerr=[[100 * lo], [100 * hi]], error_kw=dict(lw=.5, capsize=1.2, ecolor=style.INK))
        bx.text(i + d, 100 * (p + hi) + 2, f"{100 * p:.0f}", ha="center", va="bottom", fontsize=5.2)
bx.set_xticks([0, 1]); bx.set_xticklabels(["END(5)", "R³"], fontsize=5.4); bx.tick_params(axis="x", length=0)
bx.set_ylim(0, 112); bx.set_yticks([0, 25, 50, 75, 100]); bx.set_ylabel("answer changes (%)", fontsize=5.9, labelpad=1)
bx.set_title("(b) options reversed", fontsize=6.3, pad=2)
M = (("35_4b", "3.5-4B"), ("35_9b", "3.5-9B"), ("27b", "3.5-27B")); V = {}
for tag, _ in M:
    rows = [r for r in json.load(open(f"{R}/s740_nm740_{tag}_natmed.json"))["rows"] if "pick_R3_QBACK_ALL" in r]
    resc = [r for r in rows if r["pick_R3"] == r["gold"] and r["pick_END"] != r["gold"]]
    V[tag] = {k: float(np.mean([r[k] == r["gold"] for r in resc])) for k in ("pick_R3_QBACK_ALL", "pick_R3_QBACK_M", "pick_END_QALL", "pick_END_QM")}
x = np.arange(3)
for axx, title, ka, km in ((cx, "(c) rebound, queries put back", "pick_R3_QBACK_ALL", "pick_R3_QBACK_M"),
                           (dx, "(d) unrepaired, queries given", "pick_END_QALL", "pick_END_QM")):
    for d, key, col, lab in ((-w / 2, ka, "#4d5650", "every full-attention head"), (w / 2, km, "#b4bbb6", "index-matching heads")):
        vals = [100 * V[t][key] for t, _ in M]
        axx.bar(x + d, vals, w, color=col, label=lab)
        for xi, v in zip(x + d, vals):
            axx.text(xi, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=5.0)
    axx.set_xticks(x); axx.set_xticklabels([n for _, n in M], fontsize=5.4); axx.tick_params(axis="x", length=0)
    axx.set_title(title, fontsize=6.3, pad=2); axx.set_ylim(0, 112); axx.set_yticks([0, 25, 50, 75, 100])
cx.set_ylabel("repaired items correct (%)", fontsize=5.9, labelpad=1)
h, l = cx.get_legend_handles_labels()
fig.legend(h, l, loc="upper right", bbox_to_anchor=(.905, 1.06), ncol=2, fontsize=5.5, handlelength=1.2, columnspacing=1.)
os.makedirs(OUT, exist_ok=True); style.save(fig, f"{OUT}/fig_restore"); plt.close(fig); print("ok fig_restore")
