"""Figure 1 (concept): how "the third image from the end" becomes a query of the shared index-matching lookup.
(a) smaller models: the query carries the literal ordinal -> image 3; (b) Qwen3.5-27B: the query is an offset from the last visible image,
exact for one step and blurred beyond; (c) R³ writes the address and the frame early -> the query names image 5.
Output: /vlm/results/figures/c_circuit/fig1_concept.{pdf,png}.  Usage: python figs/make_fig1.py"""
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


def tokens(ax, y, words, hi=None, edits=None):
    ws = [.019 * len(w) + .026 for w in words]
    x = .5 - (sum(ws) + .012 * (len(words) - 1)) / 2
    for w, width in zip(words, ws):
        col = style.INK
        fc = "white"
        if hi and w in hi:
            fc = hi[w]
        ax.add_patch(FancyBboxPatch((x, y), width, .075, boxstyle="round,pad=0.004,rounding_size=0.012", fc=fc, ec=style.RULE, lw=.6))
        ax.text(x + width / 2, y + .037, w, ha="center", va="center", fontsize=6.4, color=col)
        if edits and w in edits:
            ax.text(x + width / 2, y + .115, edits[w], ha="center", va="bottom", fontsize=6.2, color=style.TARGET, fontweight="bold")
            ax.annotate("", xy=(x + width / 2, y + .078), xytext=(x + width / 2, y + .11), arrowprops=dict(arrowstyle="-|>", lw=.6, color=style.TARGET))
        x += width + .012
    return x


def images(ax, y, shade=None, sel=None, sel_col=None, target=J, anchor=None):
    w, gap = .1, .03
    x0 = .5 - (N * w + (N - 1) * gap) / 2
    xs = []
    for i in range(1, N + 1):
        x = x0 + (i - 1) * (w + gap); xs.append(x + w / 2)
        fc = "white"
        if shade is not None:
            a = shade[i - 1]
            fc = (1 - .55 * a, 1 - .35 * a, 1 - .45 * a) if a > 0 else "white"
        ax.add_patch(Rectangle((x, y), w, .13, fc=fc, ec=style.INK if i != anchor else style.MUTED, lw=.7 if i != anchor else 1.4))
        ax.text(x + w / 2, y + .065, str(i), ha="center", va="center", fontsize=7)
        if i == target:
            ax.add_patch(Rectangle((x - .008, y - .008), w + .016, .146, fill=False, ec=style.TARGET, lw=.9, ls=(0, (2, 1.2))))
        if sel and i == sel:
            ax.add_patch(Rectangle((x - .012, y - .012), w + .024, .154, fill=False, ec=sel_col, lw=1.5))
    return xs


def query(ax, x, y, text, col):
    ax.add_patch(FancyBboxPatch((x - .43, y), .86, .09, boxstyle="round,pad=0.005,rounding_size=0.02", fc="white", ec=col, lw=1.0))
    ax.text(x, y + .045, text, ha="center", va="center", fontsize=6.2, color=col)


def panel(ax, title, mode):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(.0, .97, title, ha="left", va="top", fontsize=7.5, fontweight="bold")
    words = ["the", "third", "image", "from", "the", "end"]
    if mode == "r3":
        tokens(ax, .66, words, hi={"third": "#d9efe9", "end": "#d9efe9"}, edits={"third": "fifth", "end": "start"})
    else:
        tokens(ax, .66, words, hi={"third": "#f6e2d3"})
    if mode == "small":
        xs = images(ax, .08, sel=K, sel_col=style.SURFACE)
        query(ax, .5, .40, "query: block 3 (literal ordinal)", style.SURFACE)
        ax.add_patch(FancyArrowPatch((.5, .40), (xs[K - 1], .225), arrowstyle="-|>", mutation_scale=7, lw=.9, color=style.SURFACE))
    elif mode == "27b":
        shade = np.array([0, 0, .1, .45, .75, .45, 0])
        xs = images(ax, .08, shade=shade, anchor=N)
        query(ax, .5, .40, "query: two steps before the last visible block", style.ACCENT)
        ax.add_patch(FancyArrowPatch((xs[N - 1], .27), (xs[N - 2], .27), connectionstyle="arc3,rad=-.7", arrowstyle="-|>", mutation_scale=6, lw=.8, color=style.ACCENT))
        ax.add_patch(FancyArrowPatch((xs[N - 2], .27), (xs[N - 3], .27), connectionstyle="arc3,rad=-.7", arrowstyle="-|>", mutation_scale=6, lw=.8, color=style.ACCENT, ls=(0, (2, 1.5))))
        ax.text(xs[N - 1], .03, "end anchor", ha="center", va="center", fontsize=5.6, color=style.MUTED)
        ax.text(.5, .335, "exact for one step, blurred beyond", ha="center", va="center", fontsize=5.8, color=style.MUTED)
    else:
        xs = images(ax, .08, sel=J, sel_col=style.TARGET)
        query(ax, .5, .40, "query: block 5 (written early)", style.TARGET)
        ax.add_patch(FancyArrowPatch((.5, .40), (xs[J - 1], .225), arrowstyle="-|>", mutation_scale=7, lw=.9, color=style.TARGET))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); style.apply()
    fig, ax = plt.subplots(1, 3, figsize=(style.TWO, 1.45))
    panel(ax[0], "(a) smaller models: literal query", "small")
    panel(ax[1], "(b) 27B: offset from the visible end", "27b")
    panel(ax[2], "(c) with R$^3$: address and frame rewritten", "r3")
    fig.subplots_adjust(wspace=.08, left=.01, right=.99, top=.98, bottom=.02)
    style.save(fig, f"{OUT}/fig1_concept"); plt.close(fig); print("ok fig1")
