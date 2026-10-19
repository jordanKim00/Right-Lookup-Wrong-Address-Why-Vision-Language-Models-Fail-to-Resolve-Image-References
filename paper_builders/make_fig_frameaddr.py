"""fig_frameaddr (main Figure 5, review revision): in Qwen3.5-27B the frame redirects the lookup, and the address it redirects is written late.
(a, b) S769 re-measured with the bare readout (S783e; 40 scenes per N): START(4), END(4), and END(4) reading "first" in place of "last" at layers
       27-34; bars: image 4's share of image attention at layer 39 (grey), answers on END(4)'s target (teal) and on the literal image 4 (orange).
(c-e)  S795 address lens (30 scenes, seven images): block shares that the index-matching heads of layer 51 would give if their query were computed
       from the answer row's stream after layer l -- end-relative target (teal), literal block (orange, dashed), start-relative question for the same
       image (grey).  Shaded: the frame-read band 27-34.
Style: exp/figs/style.py.  Output: /vlm/results/figures/c_circuit/fig_frameaddr.{pdf,png}"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/vlm/exp/figs")
import style

R, OUT, N = "/vlm/results/r3", "/vlm/results/figures/c_circuit", 7
D = json.load(open(f"{R}/s769_analysis_bare.json"))
A = json.load(open(f"{R}/s795_analysis.json"))["s795_27b"]["per_Ls"]["51"]
LIGHT = "#c9cec9"
style.apply()
fig = plt.figure(figsize=(style.TWO, 1.58))
gs = fig.add_gridspec(1, 6, width_ratios=[1.25, 1.25, .1, 1, 1, 1], wspace=.16, left=.045, right=.995, bottom=.24, top=.8)
CONDS = [("S4", "START(4)"), ("E4", "END(4)"), ("F4:P1", "END(4),\n“first” at\n27–34")]
axa = fig.add_subplot(gs[0, 0]); axb = fig.add_subplot(gs[0, 1], sharey=axa)
for ax, N_, title in ((axa, "N7", "(a) seven images: target 4"), (axb, "N8", "(b) eight images: target 5")):
    c = D[N_]; x = np.arange(3)
    ans = lambda nm, img: 100 * c["dist"][nm].get(str(img), 0.0)
    sh = [100 * c["share4_L39"][nm] for nm, _ in CONDS]
    if N_ == "N7":
        w = .36
        ax.bar(x - w / 2, sh, w, color=LIGHT, lw=0, label="image 4's share of image attention (layer 39)")
        ax.bar(x + w / 2, [ans(nm, 4) for nm, _ in CONDS], w, color=style.TARGET, lw=0, label="answers on the target")
        for xi, v in zip(x - w / 2, sh):
            ax.text(xi, v + 2, f"{v:.0f}", ha="center", va="bottom", fontsize=5.2, color=style.MUTED)
        for xi, (nm, _) in zip(x + w / 2, CONDS):
            v = ans(nm, 4); ax.text(xi, v + 2, f"{v:.0f}", ha="center", va="bottom", fontsize=5.2)
    else:
        w = .27
        ax.bar(x - w, sh, w, color=LIGHT, lw=0)
        ax.bar(x, [ans(nm, 4) for nm, _ in CONDS], w, color=style.SURFACE, lw=0, label="answers on the literal image 4")
        ax.bar(x + w, [ans(nm, 5) for nm, _ in CONDS], w, color=style.TARGET, lw=0)
        for xi, v in zip(x - w, sh):
            ax.text(xi, v + 2, f"{v:.0f}", ha="center", va="bottom", fontsize=5.2, color=style.MUTED)
        for xi, (nm, _) in zip(x, CONDS):
            v = ans(nm, 4); ax.text(xi, v + 2, f"{v:.0f}", ha="center", va="bottom", fontsize=5.2)
        for xi, (nm, _) in zip(x + w, CONDS):
            v = ans(nm, 5); ax.text(xi, v + 2, f"{v:.0f}", ha="center", va="bottom", fontsize=5.2)
    ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in CONDS], fontsize=5.3, linespacing=1.0)
    ax.set_ylim(0, 112); ax.set_yticks([0, 25, 50, 75, 100]); ax.tick_params(axis="x", length=0, pad=2); ax.tick_params(axis="y", labelsize=5.4)
    ax.set_title(title, fontsize=6.1, pad=2, loc="left")
plt.setp(axb.get_yticklabels(), visible=False)
axa.set_ylabel("%", labelpad=1, fontsize=6)
axes = [fig.add_subplot(gs[0, 3 + i]) for i in range(3)]
for i, (ax, k) in enumerate(zip(axes, (2, 4, 5))):
    j = N + 1 - k; r = A[str(k)]; ls = np.arange(len(r["end_target"])); sel = ls >= 20
    ax.axvspan(27, 34, color=style.RULE, alpha=.5, lw=0)
    ax.plot(ls[sel], np.array(A[str(j)]["start_target"])[sel], color=style.MUTED, lw=1.0, label="start-relative question, same image")
    ax.plot(ls[sel], np.array(r["end_target"])[sel], color=style.TARGET, lw=1.2, label="end-relative: target")
    if k != j:
        ax.plot(ls[sel], np.array(r["end_literal"])[sel], color=style.SURFACE, lw=1.0, ls=(0, (2.5, 1.2)), label="end-relative: literal block")
    ax.axhline(1 / N, color=style.MUTED, lw=.5, ls=":")
    ax.set_xlim(20, 50); ax.set_xticks([20, 30, 40]); ax.tick_params(labelsize=5.4)
    ax.set_ylim(0, 1); ax.set_yticks([0, .5, 1])
    if i:
        plt.setp(ax.get_yticklabels(), visible=False)
    ax.set_title(("(c) " if i == 0 else "(d) " if i == 1 else "(e) ") + f"END({k})" + (", target = literal" if k == j else f", target {j}"), fontsize=6.1, pad=2, loc="left")
axes[0].set_ylabel("share via the lookup's query", fontsize=5.6, labelpad=1)
axes[0].text(30.5, .95, "frame\nread", ha="center", va="top", fontsize=4.8, color=style.MUTED)
fig.text(.72, .04, "layer whose output stream is read (lookup heads of layer 51)", ha="center", fontsize=5.6)
h1, l1 = axa.get_legend_handles_labels(); h2, l2 = axb.get_legend_handles_labels(); h3, l3 = axes[2].get_legend_handles_labels()
fig.legend(h1 + h2, l1 + l2, loc="upper left", bbox_to_anchor=(.03, 1.03), ncol=3, fontsize=5.0, handlelength=1.0, columnspacing=.7, handletextpad=.3)
fig.legend(h3, l3, loc="upper right", bbox_to_anchor=(1.0, 1.03), ncol=3, fontsize=5.0, handlelength=1.4, columnspacing=.7, handletextpad=.3)
os.makedirs(OUT, exist_ok=True); style.save(fig, f"{OUT}/fig_frameaddr"); plt.close(fig); print("ok fig_frameaddr")
