"""v2 Figure (fig:rule): moving the early lookup moves the END answer one step toward the target; START ignores the same edit; a late edit
sets the answer.  Saved data only: S830 (identity N9 END(5), §924), S831b (N9 END(7), target 3, §927), S831 (N7 END(5), target 3, §925).
Bubbles: share of scenes answering image y when the early lookup (layers 35-47) is set to START(p) - teal END, grey START control; open
squares: late edit (51-55).  Dashed: y = p; dotted: target j.
Usage (container): python /vlm/exp/figs/make_v2_fig_rule.py -> research/ces/paper_v2_20261017/figures/fig_rule.{pdf,png}"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/vlm/exp/figs")
import style

style.apply()
R = "/vlm/results/r3"; OUT = "/vlm/research/ces/paper_v2_20261017/figures"
os.makedirs(OUT, exist_ok=True)
PANELS = [("s830_s830_27b_anchor", 9, 5, 5, [2, 3, 4, 6, 7, 8, 9], "identity: 5th from the end, $N=9$"),
          ("s831_s831b_27b_anchor", 9, 7, 3, [2, 3, 4, 5, 6, 8, 9], "7th from the end, $N=9$ (target 3)"),
          ("s831_s831_27b_anchor", 7, 5, 3, [2, 3, 4, 6, 7], "5th from the end, $N=7$ (target 3)")]


def shares(rows, key, N):
    c = np.bincount([r["land"][key] for r in rows], minlength=N + 1)[1:]
    return c / max(len(rows), 1)


fig, axes = plt.subplots(1, 3, figsize=(style.TWO, 2.15), sharey=False)
for ax, (fn, N, k, j, ps, title) in zip(axes, PANELS):
    d = json.load(open(f"{R}/{fn}.json")); rows = [r for r in d["rows"] if r["N"] == N and r["k"] == k]
    ax.plot([1, N], [1, N], ls="--", lw=.6, color=style.MUTED, zorder=1)
    ax.axhline(j, ls=":", lw=.8, color=style.TARGET, zorder=1)
    ax.text(1.45, j + .18, "target $j$", color=style.TARGET, fontsize=6, va="bottom")
    for p in ps:
        e = shares(rows, f"E_EARLY_p{p}", N); s_ = shares(rows, f"S_EARLY_p{p}", N); l_ = shares(rows, f"E_LATE_p{p}", N)
        ys = np.arange(1, N + 1)
        ax.scatter(np.full(N, p - .17), ys, s=e * 120, color=style.TARGET, alpha=.85, lw=0, zorder=3)
        ax.scatter(np.full(N, p + .17), ys, s=s_ * 120, color=style.GREYS[0], alpha=.8, lw=0, zorder=2)
        for st_ in ([p + (1 if j > p else -1)] if p != j else [p - 1, p + 1]):
            if 1 <= st_ <= N:
                ax.scatter([p - .17], [st_], marker="D", s=16, facecolor="none", edgecolor=style.TARGET, lw=.6, zorder=6)
        top = int(np.argmax(l_)) + 1
        ax.scatter([p], [top], marker="s", s=14, facecolor="none", edgecolor=style.INK, lw=.5, zorder=4)
        mean_e = float(np.sum(ys * e) / max(e.sum(), 1e-9))
        ax.plot([p - .17], [mean_e], marker="_", ms=6, color=style.INK, mew=.9, zorder=5)
    ax.set_xlim(1.3, N + .6); ax.set_ylim(.5, N + .5); ax.set_xticks(ps); ax.set_yticks(range(1, N + 1))
    ax.set_title(title, fontsize=7.2)
    ax.set_xlabel("early lookup moved to image $p$")
axes[0].set_ylabel("answered image")
h = [plt.Line2D([], [], marker="o", ls="", color=style.TARGET, ms=5, label="end-relative question"),
     plt.Line2D([], [], marker="o", ls="", color=style.GREYS[0], ms=5, label="start-relative control"),
     plt.Line2D([], [], marker="s", ls="", mfc="none", mec=style.INK, ms=4, label="late lookup moved instead (mode)"),
     plt.Line2D([], [], marker="_", ls="", color=style.INK, ms=7, label="mean end-relative answer"),
     plt.Line2D([], [], marker="D", ls="", mfc="none", mec=style.TARGET, ms=3.5, label="one step toward the target")]
fig.legend(handles=h, loc="upper center", ncol=5, bbox_to_anchor=(.5, 1.07), handletextpad=.3, columnspacing=.9)
fig.tight_layout(w_pad=.8)
style.save(fig, f"{OUT}/fig_rule")
print("saved", f"{OUT}/fig_rule.pdf")
