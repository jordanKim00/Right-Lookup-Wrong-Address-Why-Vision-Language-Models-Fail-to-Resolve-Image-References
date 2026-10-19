"""v6 figures (no model calls).  Output: research/ces/paper_v6_20261019/figures/.
  fig_account   Figure 2, schematic (no data): one lookup, three addresses.  (a) start-relative: the query names code 5 and the lookup
                matches the index code of image 5; (b) smaller models: "the third image from the end" sends the copied code 3; (c) Qwen3.5-27B
                on the identity reference "the fourth image from the end" (target = copy = image 4): the early query copies code 4, a late
                update moves the address to a neighbour and is not stopped at the target; (d) R3 writes the address and the frame.  Panels
                (a, b, d) are those of the old v3 Figure 2 (exp/figs/make_fig_account2.py); panel (c) is redrawn for the copy-then-update
                account (S844-S858), replacing v3's "anchored at the end".
  fig_lookup, fig_selnn, fig_frameaddr, fig_restore2
                the old v3 builders (exp/figs/make_fig_lookup.py, make_fig_frameaddr.py, make_fig_restore2.py) run unchanged on the current
                saved results, with only the output folder redirected here.  Their data are QK captures (S705a, S790c, S795), the semantic
                audit v1b (option letters), and the clean-readout re-runs S783a (S769, S780) and S740 (option letters).
  phenomenon    copied from research/ces/outputs/paper_r3_v2/figures/phenomenon.png (illustration, no data).
  v5 figures    fig_scale, fig_stop, fig_visual, fig_heldout, fig_repair, fig_writer, fig_grid, fig_stages, fig_predict are copied from
                research/ces/paper_v5_20261018/figures/ (built by exp/figs/make_v5_figs.py and the v4/v2 builders it names).
Usage (container with matplotlib; /vlm = the repository root): python /vlm/exp/figs/make_v6_figs.py"""
from __future__ import annotations

import os, shutil, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

sys.path.insert(0, "/vlm/exp/figs"); sys.path.insert(0, "/vlm/exp")
import style
import make_fig_account2 as acc

V = "/vlm"
OUT = f"{V}/research/ces/paper_v6_20261019/figures"
V5 = f"{V}/research/ces/paper_v5_20261018/figures"
os.makedirs(OUT, exist_ok=True)


def panel_27b(ax):
    """Identity reference, seven images: copy = target = image 4; the late update moves the address to a neighbour."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0, 1.0, "(c) 27B: copy first, update late", ha="left", va="top", fontsize=acc.FS + 1, fontweight="bold")
    yI, yQ, yT = .06, .42, .66
    acc.tokens(ax, yT, ["the", "fourth", "image", "from", "the", "end"], hi={"fourth": "#f6e2d3", "end": "#e3e7f6"})
    ax.add_patch(matplotlib.patches.FancyBboxPatch((.08, yQ), .84, .105, boxstyle="round,pad=0.005,rounding_size=0.025", fc="white",
                                                   ec=style.ACCENT, lw=.9))
    ax.text(.5, yQ + .052, "copy: code 4, then a late update", ha="center", va="center", fontsize=acc.FS, color=style.ACCENT)
    w, gap, N = .1, .035, acc.N
    x0 = .5 - (N * w + (N - 1) * gap) / 2
    xs = [x0 + i * (w + gap) + w / 2 for i in range(N)]
    for i in range(1, N + 1):
        x = x0 + (i - 1) * (w + gap)
        ax.add_patch(Rectangle((x, yI), w, .15, fc="#d6dcf4" if i in (3, 5) else "white", ec=style.INK, lw=.6))
        ax.text(x + w / 2, yI + .075, str(i), ha="center", va="center", fontsize=acc.FS + .4)
    xq = x0 + 3 * (w + gap)
    ax.add_patch(Rectangle((xq - .01, yI - .012), w + .02, .174, fill=False, ec=style.TARGET, lw=.8, ls=(0, (2, 1.2))))
    ax.add_patch(Rectangle((xq - .018, yI - .02), w + .036, .19, fill=False, ec=style.SURFACE, lw=.8, ls=(0, (1, 1))))
    ax.add_patch(FancyArrowPatch((.5, yQ), (xs[3], yI + .2), arrowstyle="-|>", mutation_scale=6, lw=.8, color=style.SURFACE))
    for b in (2, 4):
        ax.add_patch(FancyArrowPatch((xs[3], yI + .19), (xs[b], yI + .19), connectionstyle="arc3,rad=-.9" if b == 4 else "arc3,rad=.9",
                                     arrowstyle="-|>", mutation_scale=5, lw=.75, color=style.ACCENT))
    ax.text(xs[4] + .075, yI + .15, "✗", ha="left", va="bottom", fontsize=acc.FS + 2, color=style.ACCENT, fontweight="bold",
            fontfamily="DejaVu Sans")
    ax.text(.995, yI + .175, "not stopped\nat the target", ha="right", va="bottom", fontsize=acc.FS - .6, color=style.MUTED, linespacing=.95)


def fig_account():
    style.apply()
    fig, ax = plt.subplots(2, 2, figsize=(style.ONE, 2.45))
    acc.panel(ax[0, 0], "(a) start-relative: right lookup", "start")
    acc.panel(ax[0, 1], "(b) smaller models: the ordinal copied", "small")
    panel_27b(ax[1, 0])
    acc.panel(ax[1, 1], "(d) R$^3$: address and frame rewritten", "r3")
    fig.subplots_adjust(wspace=.06, hspace=.18, left=.005, right=.995, top=.985, bottom=.01)
    style.save(fig, f"{OUT}/fig_account"); plt.close(fig)


LABELS = {  # v6 calls image k "the copy" throughout; only legend wording changes, never data
    "make_fig_lookup.py": [('"image $k$ (literal)"', '"image $k$ (copy)"'), ('label="literal, $t=k$"', 'label="copy, $t=k$"')],
    "make_fig_frameaddr.py": [('label="answers on the literal image 4"', 'label="answers on the copy, image 4"'),
                              ('label="end-relative: literal block"', 'label="end-relative: copy"')],
    "make_fig_restore2.py": [],
}


def run_old(path):
    """Run an old builder unchanged except for its output folder (style.save(fig, f"{OUT}/...") -> this folder) and legend wording."""
    src = open(path).read()
    assert 'style.save(fig, f"{OUT}/' in src, path
    src = src.replace('style.save(fig, f"{OUT}/', 'style.save(fig, f"' + OUT + '/')
    for old, new in LABELS.get(os.path.basename(path), []):
        assert old in src, (path, old)
        src = src.replace(old, new)
    g = {"__name__": "__main__", "__file__": path}
    exec(compile(src, path, "exec"), g)


if __name__ == "__main__":
    fig_account(); print("ok fig_account")
    for p in ("make_fig_lookup.py", "make_fig_frameaddr.py", "make_fig_restore2.py"):
        run_old(f"{V}/exp/figs/{p}")
    shutil.copy(f"{V}/research/ces/outputs/paper_r3_v2/figures/phenomenon.png", f"{OUT}/phenomenon.png")
    for f in ("fig_scale", "fig_stop", "fig_visual", "fig_heldout", "fig_repair", "fig_writer", "fig_grid", "fig_stages", "fig_predict", "fig_phenomenon"):
        for ext in ("pdf", "png"):
            shutil.copy(f"{V5}/{f}.{ext}", f"{OUT}/{f}.{ext}")
    print("ok v6 figures")
