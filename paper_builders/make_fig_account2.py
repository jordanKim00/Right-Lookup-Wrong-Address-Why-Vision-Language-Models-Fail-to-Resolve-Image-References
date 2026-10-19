"""Figure 1 (account): one lookup, three addresses.  (a) start-relative: the query names code 5 and the lookup matches the index code of image 5;
(b) smaller models: "the third image from the end" sends the literal code 3; (c) Qwen3.5-27B: the query counts back from the last visible
image, sharp for one step and blurred beyond; (d) R³ writes the address and the frame, and the same lookup returns image 5.
Schematic (no data).  Style: exp/figs/style.py.  Output: /vlm/results/figures/c_circuit/fig_account.{pdf,png}.  Usage: python figs/make_fig_account.py"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

sys.path.insert(0, "/vlm/exp/figs")
import style

OUT = "/vlm/results/figures/c_circuit"
N, K, J = 7, 3, 5
FS = 6.0


def tokens(ax, y, words, hi=None, edits=None):
    ws = [.029 * len(w) + .03 for w in words]
    gap = .01
    x = .5 - (sum(ws) + gap * (len(words) - 1)) / 2
    for w, width in zip(words, ws):
        fc = (hi or {}).get(w, "white")
        ax.add_patch(FancyBboxPatch((x, y), width, .1, boxstyle="round,pad=0.004,rounding_size=0.015", fc=fc, ec=style.RULE, lw=.5))
        ax.text(x + width / 2, y + .05, w, ha="center", va="center", fontsize=FS, color=style.INK)
        if edits and w in edits:
            ax.text(x + width / 2, y + .155, edits[w], ha="center", va="bottom", fontsize=FS, color=style.TARGET, fontweight="bold")
            ax.annotate("", xy=(x + width / 2, y + .102), xytext=(x + width / 2, y + .15), arrowprops=dict(arrowstyle="-|>", lw=.5, color=style.TARGET))
        x += width + gap


def images(ax, y, shade=None, sel=None, sel_col=None, anchor=None, mark=None):
    w, gap = .1, .035
    x0 = .5 - (N * w + (N - 1) * gap) / 2
    xs = []
    for i in range(1, N + 1):
        x = x0 + (i - 1) * (w + gap); xs.append(x + w / 2)
        fc = "white"
        if shade is not None and shade[i - 1] > 0:
            a = shade[i - 1]; fc = (1 - .55 * a, 1 - .35 * a, 1 - .45 * a)
        ax.add_patch(Rectangle((x, y), w, .15, fc=fc, ec=style.MUTED if i == anchor else style.INK, lw=1.2 if i == anchor else .6))
        ax.text(x + w / 2, y + .075, str(i), ha="center", va="center", fontsize=FS + .4)
        if i == J:
            ax.add_patch(Rectangle((x - .01, y - .012), w + .02, .174, fill=False, ec=style.TARGET, lw=.8, ls=(0, (2, 1.2))))
        if sel and i == sel:
            ax.add_patch(Rectangle((x - .016, y - .018), w + .032, .186, fill=False, ec=sel_col, lw=1.3))
    if mark:
        ax.text(xs[(sel or J) - 1] + .075, y + .15, mark[0], ha="left", va="bottom", fontsize=FS + 2, color=mark[1], fontweight="bold", fontfamily="DejaVu Sans")
    return xs


def query(ax, y, text, col):
    ax.add_patch(FancyBboxPatch((.08, y), .84, .105, boxstyle="round,pad=0.005,rounding_size=0.025", fc="white", ec=col, lw=.9))
    ax.text(.5, y + .052, text, ha="center", va="center", fontsize=FS, color=col)


def panel(ax, title, mode):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0, 1.0, title, ha="left", va="top", fontsize=FS + 1, fontweight="bold")
    yI, yQ, yT = .06, .42, .66
    if mode == "start":
        tokens(ax, yT, ["the", "fifth", "image"], hi={"fifth": "#d9efe9"})
        query(ax, yQ, "query: index code 5", style.TARGET)
        xs = images(ax, yI, sel=J, sel_col=style.TARGET, mark=("✓", style.TARGET))
        ax.add_patch(FancyArrowPatch((.5, yQ), (xs[J - 1], yI + .2), arrowstyle="-|>", mutation_scale=6, lw=.8, color=style.TARGET))
        ax.text(.02, yI + .25, "keys carry the code", ha="left", va="center", fontsize=FS - .5, color=style.MUTED)
        ax.text(.68, yQ - .055, "QK match", ha="left", va="center", fontsize=FS - .5, color=style.MUTED)
        ax.annotate("", xy=(.5, yQ + .105), xytext=(.5, yT - .005), arrowprops=dict(arrowstyle="-|>", lw=.6, color=style.MUTED))
        ax.text(.53, (yQ + .105 + yT) / 2, "read", ha="left", va="center", fontsize=FS - .5, color=style.MUTED)
    elif mode == "small":
        tokens(ax, yT, ["the", "third", "image", "from", "the", "end"], hi={"third": "#f6e2d3"})
        query(ax, yQ, "query: code 3, the ordinal as written", style.SURFACE)
        xs = images(ax, yI, sel=K, sel_col=style.SURFACE, mark=("✗", style.SURFACE))
        ax.add_patch(FancyArrowPatch((.5, yQ), (xs[K - 1], yI + .2), arrowstyle="-|>", mutation_scale=6, lw=.8, color=style.SURFACE))
    elif mode == "27b":
        tokens(ax, yT, ["the", "third", "image", "from", "the", "end"], hi={"third": "#e3e7f6", "end": "#e3e7f6"})
        query(ax, yQ, "query: anchored at the last image", style.ACCENT)
        xs = images(ax, yI, shade=np.array([0, 0, .12, .45, .8, .45, 0]), anchor=N)
        for a_, b_, ls in ((N - 1, N - 2, "-"), (N - 2, N - 3, (0, (2, 1.5)))):
            ax.add_patch(FancyArrowPatch((xs[a_], yI + .2), (xs[b_], yI + .2), connectionstyle="arc3,rad=-.8", arrowstyle="-|>", mutation_scale=5, lw=.7, color=style.ACCENT, ls=ls))
    else:
        tokens(ax, yT, ["the", "third", "image", "from", "the", "end"], hi={"third": "#d9efe9", "end": "#d9efe9"}, edits={"third": "fifth", "end": "start"})
        query(ax, yQ, "query: code 5, the written address", style.TARGET)
        xs = images(ax, yI, sel=J, sel_col=style.TARGET, mark=("✓", style.TARGET))
        ax.add_patch(FancyArrowPatch((.5, yQ), (xs[J - 1], yI + .2), arrowstyle="-|>", mutation_scale=6, lw=.8, color=style.TARGET))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); style.apply()
    fig, ax = plt.subplots(2, 2, figsize=(style.ONE, 2.45))
    panel(ax[0, 0], "(a) start-relative: right lookup", "start")
    panel(ax[0, 1], "(b) smaller models: literal address", "small")
    panel(ax[1, 0], "(c) 27B: anchored at the end", "27b")
    panel(ax[1, 1], "(d) R$^3$: address and frame rewritten", "r3")
    fig.subplots_adjust(wspace=.06, hspace=.18, left=.005, right=.995, top=.985, bottom=.01)
    style.save(fig, f"{OUT}/fig_account2"); plt.close(fig); print("ok fig_account")
