"""fig_end4b (S769 re-measured with the bare readout, S783e; FINDINGS §847): the end frame redirects the lookup even where no position has
to be converted.  Qwen3.5-27B, 40 scenes per N.  (a) seven images: END(4) targets the literal image 4; (b) eight images: END(4) targets image 5
and image 4 is the literal one.  Per question: START(4), END(4), and END(4) reading "first" in place of "last" at layers 27-34 (F4:P1).
Bars: answers on image 4 / on image 5, and image 4's share of the image attention at layer 39 (all heads, last two rows).
Style: exp/figs/style.py.  Output: /vlm/results/figures/c_circuit/fig_end4b.{pdf,png}"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/vlm/exp/figs")
import style

R, OUT = "/vlm/results/r3", "/vlm/results/figures/c_circuit"
D = json.load(open(f"{R}/s769_analysis_bare.json"))
style.apply()
fig, axes = plt.subplots(1, 2, figsize=(style.ONE, 1.72), sharey=True, gridspec_kw=dict(wspace=.08))
CONDS = [("S4", "START(4)"), ("E4", "END(4)"), ("F4:P1", "END(4),\n“first”\nat 27–34")]
LIGHT = "#c9cec9"
for ax, N, title in ((axes[0], "N7", "(a) seven images\nEND(4) target: image 4"), (axes[1], "N8", "(b) eight images\nEND(4) target: image 5")):
    c = D[N]; x = np.arange(len(CONDS))
    ans = lambda nm, img: 100 * c["dist"][nm].get(str(img), 0.0)
    sh = [100 * c["share4_L39"][nm] for nm, _ in CONDS]
    if N == "N7":
        w = .36
        b1 = ax.bar(x - w / 2, sh, w, color=LIGHT, lw=0, label="image 4's share of image attention (layer 39)")
        b2 = ax.bar(x + w / 2, [ans(nm, 4) for nm, _ in CONDS], w, color=style.TARGET, lw=0, label="answers on END(4)'s target")
        for xi, v in zip(x - w / 2, sh):
            ax.text(xi, v + 2, f"{v:.0f}", ha="center", va="bottom", fontsize=5.6, color=style.MUTED)
        for xi, (nm, _) in zip(x + w / 2, CONDS):
            v = ans(nm, 4); ax.text(xi, v + 2, f"{v:.0f}", ha="center", va="bottom", fontsize=5.6, color=style.INK)
    else:
        w = .27
        ax.bar(x - w, sh, w, color=LIGHT, lw=0)
        b3 = ax.bar(x, [ans(nm, 4) for nm, _ in CONDS], w, color=style.SURFACE, lw=0, label="answers on image 4 (literal)")
        ax.bar(x + w, [ans(nm, 5) for nm, _ in CONDS], w, color=style.TARGET, lw=0)
        for xi, v in zip(x - w, sh):
            ax.text(xi, v + 2, f"{v:.0f}", ha="center", va="bottom", fontsize=5.6, color=style.MUTED)
        for xi, (nm, _) in zip(x, CONDS):
            v = ans(nm, 4); ax.text(xi, v + 2, f"{v:.0f}", ha="center", va="bottom", fontsize=5.6, color=style.INK)
        for xi, (nm, _) in zip(x + w, CONDS):
            v = ans(nm, 5); ax.text(xi, v + 2, f"{v:.0f}", ha="center", va="bottom", fontsize=5.6, color=style.INK)
    ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in CONDS], fontsize=5.8)
    ax.set_ylim(0, 112); ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title(title, fontsize=6.3, pad=3, linespacing=1.05)
    ax.tick_params(axis="x", length=0, pad=2)
axes[0].set_ylabel("%", labelpad=1)
h1, l1 = axes[0].get_legend_handles_labels(); h2, l2 = axes[1].get_legend_handles_labels()
fig.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(.5, 1.2), ncol=2, fontsize=5.6, handlelength=1.1, columnspacing=.9, handletextpad=.4)
os.makedirs(OUT, exist_ok=True); style.save(fig, f"{OUT}/fig_end4b"); plt.close(fig)
print("ok fig_end4b")
