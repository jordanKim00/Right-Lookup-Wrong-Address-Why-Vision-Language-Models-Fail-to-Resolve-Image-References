"""Method figure: R³ = model-specific READ + fixed RESOLVE + REBIND, feeding the model's own index-matching lookup.
Output: /vlm/results/figures/c_circuit/fig_method.{pdf,png}.  Usage: python figs/make_fig_method.py"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

sys.path.insert(0, "/vlm/exp/figs")
import style

OUT = "/vlm/results/figures/c_circuit"
READC = "#5b6bbf"


def box(ax, x, y, w, h, ec, title, lines, sub):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.018", fc="white", ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h - .075, title, ha="center", va="center", fontsize=7.4, color=ec, fontweight="bold")
    for i, t in enumerate(lines):
        ax.text(x + w / 2, y + h - .2 - i * .115, t, ha="center", va="center", fontsize=6.1, color=style.INK)
    ax.text(x + w / 2, y - .065, sub, ha="center", va="center", fontsize=5.9, color=style.MUTED, style="italic")


def arrow(ax, x0, x1, y):
    ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="-|>", lw=.8, color=style.INK))


def main():
    style.apply()
    fig = plt.figure(figsize=(style.TWO, 1.35))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    # input
    for i in range(7):
        ax.add_patch(Rectangle((.012 + i * .019, .58), .016, .1, fc="white", ec=style.INK, lw=.6))
        ax.text(.02 + i * .019, .63, str(i + 1), ha="center", va="center", fontsize=4.8)
    ax.text(.077, .44, "“What is in the third\nimage from the end?”", ha="center", va="center", fontsize=6.0, color=style.INK, linespacing=1.0)
    ax.text(.077, .78, "input", ha="center", va="center", fontsize=6.4, color=style.MUTED)
    y, h = .2, .66
    box(ax, .17, y, .2, h, READC, "READ", [r"ordinal $\hat{k}=3$ (“third”)", "frame owner: “end” → END", r"count $\hat{N}=7$ (image codes)"],
        "calibrated per model")
    box(ax, .405, y, .14, h, style.TARGET, "RESOLVE", [r"$t=\hat{N}+1-\hat{k}$", r"$=5$"], "fixed rule")
    box(ax, .58, y, .19, h, style.TARGET, "REBIND", [r"early layer $L_w$:", "“third” ← index 5", "“end” ← START"], "fixed rule, model's vectors")
    box(ax, .805, y, .185, h, style.INK, "own lookup", ["matcher-head query", "→ image 5", "(QK index match)"], "unchanged")
    for x0, x1 in ((.145, .17), (.37, .405), (.545, .58), (.77, .805)):
        arrow(ax, x0, x1, y + h / 2)
    os.makedirs(OUT, exist_ok=True)
    style.save(fig, f"{OUT}/fig_method"); plt.close(fig)
    print("ok")


if __name__ == "__main__":
    main()
