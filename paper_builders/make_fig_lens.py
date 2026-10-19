"""fig_lens (S795 address lens, Qwen3.5-27B, FINDINGS §855): where the end-relative address goes wrong.
For "the k-th image from the end" (k = 2, 4, 5), the block shares that the index-matching heads of lookup layer L* = 51 would give if their query
were computed from the answer row's stream after layer l: on the target j = 8 - k (teal) and the literal block k (orange, dashed), against the
start-relative question for the same target, START(j) (grey).  Shaded: the frame-read band (layers 27-34).  30 scenes, seven images.
Style: exp/figs/style.py.  Output: /vlm/results/figures/c_circuit/fig_lens.{pdf,png}"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/vlm/exp/figs")
import style

R, OUT, N = "/vlm/results/r3", "/vlm/results/figures/c_circuit", 7
A = json.load(open(f"{R}/s795_analysis.json"))["s795_27b"]["per_Ls"]["51"]
style.apply()
fig, axes = plt.subplots(1, 3, figsize=(style.ONE, 1.45), sharey=True, gridspec_kw=dict(wspace=.08, left=.11, right=.99, bottom=.24, top=.8))
lo = 20
for ax, k in zip(axes, (2, 4, 5)):
    j = N + 1 - k; r = A[str(k)]; ls = np.arange(len(r["end_target"]))
    sel = ls >= lo
    ax.axvspan(27, 34, color=style.RULE, alpha=.5, lw=0)
    ax.plot(ls[sel], np.array(A[str(j)]["start_target"])[sel], color=style.MUTED, lw=1.0, label=f"start-relative, same image")
    ax.plot(ls[sel], np.array(r["end_target"])[sel], color=style.TARGET, lw=1.2, label="end-relative: target")
    if k != j:
        ax.plot(ls[sel], np.array(r["end_literal"])[sel], color=style.SURFACE, lw=1.0, ls=(0, (2.5, 1.2)), label="end-relative: literal block")
    ax.axhline(1 / N, color=style.MUTED, lw=.5, ls=":")
    ax.set_title(f"END({k})" + (": target = literal" if k == j else f", target {j}"), fontsize=6.0, pad=2)
    ax.set_xlim(lo, 50); ax.set_xticks([20, 30, 40]); ax.tick_params(labelsize=5.4)
    ax.set_ylim(0, 1); ax.set_yticks([0, .5, 1])
axes[0].set_ylabel("share through the lookup’s query", fontsize=5.6, labelpad=1)
fig.text(.55, .03, "layer $l$ whose output stream is read", ha="center", fontsize=5.8)
axes[0].text(30.5, .93, "frame\nread", ha="center", va="top", fontsize=4.8, color=style.MUTED)
h, l = axes[2].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", bbox_to_anchor=(.55, 1.02), ncol=3, fontsize=5.0, handlelength=1.4, columnspacing=.8, handletextpad=.35)
os.makedirs(OUT, exist_ok=True); style.save(fig, f"{OUT}/fig_lens"); plt.close(fig); print("ok fig_lens")
