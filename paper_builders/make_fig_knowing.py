"""Figure 1 (paper revision, 2026-10-14): knowing is not using.  Schematic in the style of attribution-graph figures
(flat; sans labels; monospace prompt tokens with tinted key tokens at the bottom; beige stacked feature cards; thick maroon used paths;
faded unused paths; intervention icons; output at the top).  Scene: seven images, "the third image from the end" = image 5.
(a) Qwen3.5-27B names the index (95%; appendix table "Which image is third from the end? (number)") but answers the content question less
    often (65%; "third image from the end"); the ordinal, its frame and the count are readable from its states (the Read of R3).
(b) smaller models: the ordinal alone becomes the address (literal image 3); the frame and the count go unused.
(c) Qwen3.5-27B: the frame switches the lookup to count back from the last visible image; sharp for one step, spread beyond.
(d) R3: read the three pieces, resolve 7 + 1 - 3 = 5, rebind ("fifth", "start") in the states at layer 3; the lookup returns image 5.
Numbers are from the paper's tables; everything else is schematic.  Colours follow the paper: teal = target, orange = literal.
Output: /vlm/results/figures/c_circuit/fig_knowing.{pdf,png}.  Usage: python figs/make_fig_knowing.py"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D

sys.path.insert(0, "/vlm/exp/figs")
import style

OUT = "/vlm/results/figures/c_circuit"
SANS, MONO = "DejaVu Sans", "DejaVu Sans Mono"
MAROON, FADE, BEIGE, BEIGE_E, GREY = "#8f2f2f", "#cfcfca", "#f4ecdc", "#d8c6a3", "#6b706d"
T_ORD, T_FRAME = "#f7e0cf", "#dde6f5"
TEAL, ORANGE, BLUE = style.TARGET, style.SURFACE, style.ACCENT
SHAPES = ["^", "o", "*", "s", "D", "p", "h"]
N, K, J = 7, 3, 5
Y_TOK, Y_CARD, Y_ADDR, Y_IMG = .035, .27, .505, .745


def card(ax, x, y, w, h, text, fc=BEIGE, ec=BEIGE_E, faded=False, fs=5.4, weight="normal", sub=None):
    a = .38 if faded else 1.
    for d in (.016, .008):
        ax.add_patch(FancyBboxPatch((x + d, y + d * 1.25), w, h, boxstyle="round,pad=0,rounding_size=0.02", fc=fc, ec=ec, lw=.45, alpha=a * .85, zorder=2))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.02", fc=fc, ec=ec, lw=.6, alpha=a, zorder=3))
    col = style.INK if not faded else "#a3a39e"
    if sub is None:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, family=SANS, color=col, fontweight=weight, zorder=4)
    else:
        ax.text(x + w / 2, y + h * .68, text, ha="center", va="center", fontsize=fs, family=SANS, color=col, fontweight="bold", zorder=4)
        ax.text(x + w / 2, y + h * .3, sub, ha="center", va="center", fontsize=fs - .3, family=SANS, color=col, zorder=4)


def tokens(ax, words, y=Y_TOK, tint=None, fs=5.2):
    tint = tint or {}
    ws = [.0255 * len(t) + .026 for t in words]; gap = .011
    x = .5 - (sum(ws) + gap * (len(words) - 1)) / 2; cx = {}
    for t, w in zip(words, ws):
        ax.add_patch(FancyBboxPatch((x, y), w, .068, boxstyle="round,pad=0,rounding_size=0.014", fc=tint.get(t, "white"), ec="#cfcfca", lw=.5, zorder=3))
        ax.text(x + w / 2, y + .034, t, ha="center", va="center", fontsize=fs, family=MONO, color=style.INK, zorder=4)
        cx.setdefault(t, x + w / 2); x += w + gap
    return cx


def strip(ax, y=Y_IMG, sel=None, col=TEAL, spread=None, mark=None, w=.098, gap=.027):
    x0 = .5 - (N * w + (N - 1) * gap) / 2; xs = []
    for i in range(1, N + 1):
        x = x0 + (i - 1) * (w + gap); xs.append(x + w / 2)
        fc = "white"
        if spread is not None and spread[i - 1] > 0:
            s = spread[i - 1]; fc = (1 - .9 * s, 1 - .35 * s, 1 - .45 * s)
        ax.add_patch(FancyBboxPatch((x, y), w, .085, boxstyle="round,pad=0,rounding_size=0.012", fc=fc, ec="#b9b9b4", lw=.5, zorder=3))
        ax.plot(x + w / 2, y + .045, marker=SHAPES[i - 1], ms=3.2 if SHAPES[i - 1] != "*" else 4.2, color="#4f5452", mec="none", zorder=4)
        ax.text(x + w / 2, y - .028, str(i), ha="center", va="center", fontsize=4.7, family=SANS, color="#8a8f8c")
        if sel == i:
            ax.add_patch(FancyBboxPatch((x - .012, y - .012), w + .024, .109, boxstyle="round,pad=0,rounding_size=0.016", fill=False, ec=col, lw=1.4, zorder=5))
    if mark:
        i, m, c = mark
        ax.text(xs[i - 1], y + .112, m, ha="center", va="bottom", fontsize=7.2, color=c, family=SANS, fontweight="bold", zorder=6)
    return xs


def arrow(ax, p, q, col=MAROON, lw=1.5, cs="arc3,rad=0", faded=False, ls="-", head=True):
    ax.add_patch(FancyArrowPatch(p, q, connectionstyle=cs, arrowstyle="-|>" if head else "-", mutation_scale=6.2 if head else 1,
                                 lw=lw, color=FADE if faded else col, ls=ls, shrinkA=0, shrinkB=0, zorder=1, capstyle="round", joinstyle="round"))


def title(ax, s, sub):
    ax.text(0, 1.0, s, ha="left", va="top", fontsize=6.3, family=SANS, fontweight="bold", color=style.INK)
    ax.text(0, .948, sub, ha="left", va="top", fontsize=5.0, family=SANS, color=GREY)


def magnifier(ax, x, y, r=.022, col=style.INK):
    ax.add_patch(Circle((x, y), r, fill=False, ec=col, lw=.8, zorder=7))
    ax.add_line(Line2D([x + r * .72, x + r * 1.75], [y - r * .72 * 1.23, y - r * 1.75 * 1.23], color=col, lw=1.1, zorder=7, solid_capstyle="round"))


def pencil(ax, x, y, s=.02, col=style.INK):
    ax.add_line(Line2D([x - s, x + s], [y - s * 1.1, y + s * 1.1], color=col, lw=2.0, zorder=7, solid_capstyle="butt"))
    ax.add_line(Line2D([x - s, x - s * 1.4], [y - s * 1.1, y - s * 1.55], color=col, lw=.8, zorder=7))


def panel_a(ax):
    title(ax, "(a) It knows the address", "Qwen3.5-27B, scenes of five to seven images")
    for y, l1, l2, out, pct, c in ((.70, "Which image is the", "third from the end?", "5", "95%", TEAL),
                                  (.475, "What is in the third", "image from the end?", "◆", "65%", ORANGE)):
        ax.add_patch(FancyBboxPatch((.0, y), .62, .145, boxstyle="round,pad=0,rounding_size=0.02", fc="white", ec="#cfcfca", lw=.5, zorder=3))
        ax.text(.03, y + .098, l1, ha="left", va="center", fontsize=4.95, family=MONO, color=style.INK, zorder=4)
        ax.text(.03, y + .045, l2, ha="left", va="center", fontsize=4.95, family=MONO, color=style.INK, zorder=4)
        arrow(ax, (.635, y + .072), (.72, y + .072), lw=1.2)
        ax.add_patch(FancyBboxPatch((.735, y + .026), .095, .092, boxstyle="round,pad=0,rounding_size=0.015", fc="#fff5dc", ec="#e3c98b", lw=.6, zorder=3))
        ax.text(.7825, y + .072, out, ha="center", va="center", fontsize=6.0 if out == "5" else 5.6, family=MONO if out == "5" else SANS, color=style.INK, zorder=4)
        ax.text(.85, y + .072, pct, ha="left", va="center", fontsize=6.0, family=SANS, color=c, fontweight="bold")
    ax.add_patch(FancyBboxPatch((0, .085), 1., .3, boxstyle="round,pad=0,rounding_size=0.02", fc="#f6f6f2", ec="none", zorder=1))
    for x, t in ((.035, "ordinal: 3"), (.355, "frame: end"), (.675, "count: 7")):
        card(ax, x, .14, .29, .095, t)
    magnifier(ax, .07, .33, r=.018)
    ax.text(.115, .33, "each piece can be read from its states", ha="left", va="center", fontsize=4.8, family=SANS, color=GREY)


def panel_b(ax):
    title(ax, "(b) Smaller models", "the ordinal alone becomes the address")
    cx = tokens(ax, ["the", "third", "image", "from", "the", "end"], tint={"third": T_ORD, "end": T_FRAME})
    card(ax, .03, Y_CARD, .29, .095, "ordinal: 3")
    card(ax, .355, Y_CARD, .29, .095, "frame: end", faded=True)
    card(ax, .68, Y_CARD, .29, .095, "count: 7", faded=True)
    arrow(ax, (cx["third"], Y_TOK + .07), (cx["third"], Y_CARD), lw=1.3)
    arrow(ax, (cx["end"], Y_TOK + .07), (.5, Y_CARD), faded=True, lw=.9)
    card(ax, .03, Y_ADDR, .44, .095, "address = 3", fc="#fbe5d6", ec="#e2a77f")
    arrow(ax, (.175, Y_CARD + .115), (.175, Y_ADDR), lw=1.6)
    for x in (.5, .825):
        arrow(ax, (x, Y_CARD + .115), (x, Y_ADDR - .02), faded=True, lw=.9, ls=(0, (2, 1.4)), head=False)
    ax.text(.66, Y_ADDR + .047, "never used", ha="center", va="center", fontsize=4.9, family=SANS, color="#a3a39e", style="italic")
    xs = strip(ax, sel=K, col=ORANGE, mark=(K, "✗", ORANGE))
    arrow(ax, (xs[K - 1], Y_ADDR + .115), (xs[K - 1], Y_IMG - .045), lw=1.6)


def panel_c(ax):
    title(ax, "(c) Qwen3.5-27B", "the frame switches the lookup to count back")
    cx = tokens(ax, ["the", "third", "image", "from", "the", "end"], tint={"third": T_ORD, "end": T_FRAME})
    card(ax, .03, Y_CARD, .29, .095, "ordinal: 3")
    card(ax, .355, Y_CARD, .29, .095, "frame: end")
    card(ax, .68, Y_CARD, .29, .095, "last image: 7")
    arrow(ax, (cx["third"], Y_TOK + .07), (cx["third"], Y_CARD), lw=1.3)
    arrow(ax, (cx["end"], Y_TOK + .07), (.5, Y_CARD), lw=1.3)
    card(ax, .03, Y_ADDR, .94, .095, "count back from the last image", fc="#e5ebf8", ec="#a3b5dd")
    for x in (.175, .5, .825):
        arrow(ax, (x, Y_CARD + .115), (x, Y_ADDR), lw=1.4)
    xs = strip(ax, spread=np.array([0, 0, .12, .45, .75, .45, .1]))
    for i, a_ in ((4, .55), (5, 1.), (6, .55)):
        arrow(ax, (.5, Y_ADDR + .115), (xs[i - 1], Y_IMG - .045), lw=.6 + .9 * a_)
    ax.text(.0, Y_ADDR + .165, "sharp one step from\nthe end, spread beyond", ha="left", va="center", fontsize=4.6, family=SANS, color=GREY, linespacing=1.1)


def panel_d(ax):
    title(ax, "(d) With R³", "the missing step is supplied, the lookup does the rest")
    cx = tokens(ax, ["the", "third", "image", "from", "the", "end"], tint={"third": T_ORD, "end": T_FRAME})
    w, h = .27, .13; y = Y_CARD - .02; X = (.015, .365, .715)
    card(ax, X[0], y, w, h, "Read", sub="3, end, 7", fs=5.2)
    card(ax, X[1], y, w, h, "Resolve", sub="7+1−3 = 5", fs=5.2)
    card(ax, X[2], y, w, h, "Rebind", sub="fifth, start", fs=5.2)
    magnifier(ax, X[0] + w - .02, y + h + .045); pencil(ax, X[2] + w - .02, y + h + .05)
    arrow(ax, (cx["third"], Y_TOK + .07), (X[0] + w / 2, y), lw=1.3)
    arrow(ax, (X[0] + w + .02, y + h / 2), (X[1] - .004, y + h / 2), lw=1.2)
    arrow(ax, (X[1] + w + .02, y + h / 2), (X[2] - .004, y + h / 2), lw=1.2)
    card(ax, .53, Y_ADDR + .02, .44, .095, "address = 5", fc="#d9efea", ec="#86c4b7")
    arrow(ax, (X[2] + w / 2, y + h + .02), (X[2] + w / 2, Y_ADDR + .02), lw=1.6)
    xs = strip(ax, sel=J, col=TEAL, mark=(J, "✓", TEAL))
    arrow(ax, (xs[J - 1], Y_ADDR + .135), (xs[J - 1], Y_IMG - .045), lw=1.6)
    ax.text(.03, Y_ADDR + .067, "written into the\nstates at layer 3,\ntext unchanged", ha="left", va="center", fontsize=4.6, family=SANS, color=GREY, linespacing=1.1)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); style.apply()
    fig, axs = plt.subplots(1, 4, figsize=(style.TWO, 1.95))
    for ax in axs:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.axis("off")
    panel_a(axs[0]); panel_b(axs[1]); panel_c(axs[2]); panel_d(axs[3])
    fig.subplots_adjust(wspace=.12, left=.004, right=.996, top=.985, bottom=.012)
    style.save(fig, f"{OUT}/fig_knowing"); plt.close(fig); print("ok fig_knowing")
