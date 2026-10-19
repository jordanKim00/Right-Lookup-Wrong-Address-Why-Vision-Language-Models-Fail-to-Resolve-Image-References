"""fig_circuit: schematic of the reference-to-image circuit the paper traces (no data).  Three rows share one lookup: start-relative
(the address is the ordinal), end-relative (the frame acts as a switch to an end anchor, never combined with N), and rebound (R3 writes
the resolved ordinal and a start frame early).  Style: exp/figs/style.py.  Output: /vlm/results/figures/c_circuit/fig_circuit.{pdf,png}"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, "/vlm/exp/figs")
import style

OUT = "/vlm/results/figures/c_circuit"
style.apply()
fig, ax = plt.subplots(figsize=(style.TWO, 1.95))
ax.set_xlim(0, 100); ax.set_ylim(0, 42); ax.axis("off")


def box(x, y, w, h, txt, fc="#ffffff", ec=style.INK, fs=6.2, lw=.7, color=style.INK, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8", fc=fc, ec=ec, lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, color=color, linespacing=1.15)


def arrow(x0, y0, x1, y1, col=style.INK, lw=.8, ls="-", txt=None, tx=0, ty=1.2, fs=5.6, rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6, lw=lw, color=col, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}"))
    if txt:
        ax.text((x0 + x1) / 2 + tx, (y0 + y1) / 2 + ty, txt, ha="center", va="bottom", fontsize=fs, color=col)


# shared lookup (right): images with index codes, matched by the query of the index-matching heads
for i in range(7):
    x = 70 + i * 3.9
    ax.add_patch(FancyBboxPatch((x, 33.0), 3.2, 3.2, boxstyle="round,pad=0.1,rounding_size=0.3", fc="#eef1ee", ec=style.MUTED, lw=.5))
    ax.text(x + 1.6, 34.6, str(i + 1), ha="center", va="center", fontsize=5.4, color=style.INK)
ax.text(83.6, 38.6, "image blocks carry an index code (keys)", ha="center", fontsize=5.8, color=style.INK)
ax.text(83.6, 30.6, "index-matching heads: query $\\cdot$ code", ha="center", fontsize=5.8, color=style.INK)
rows = [  # y, label, tokens, address text, colour, fill, picked block (1-based) or None, note
    (23.5, "start-relative", "“fifth … from the first”", "address 5", style.TARGET, "#e3f3f0", 5, None),
    (13.5, "end-relative", "“third … from the end”", "smaller: 3 (copy)\n27B: end anchor − 1 step", style.SURFACE, "#fbeee5", 3, None),
    (3.5, "rebound (R³)", "“third … from the end”\n+ writes: ordinal 5, frame START", "address 5", style.TARGET, "#e3f3f0", 5, None),
]
for y, lab, tok, addr, col, fill, pick, note in rows:
    ax.text(0.5, y + 3.1, lab, fontsize=6.4, fontweight="bold", color=style.INK, va="center")
    box(0.5, y - 2.6, 22, 4.6, tok, fc="#ffffff", fs=5.8)
    box(31, y - 2.6, 20, 4.6, addr, fc=fill, ec=col, fs=5.8)
    arrow(22.9, y - 0.3, 30.6, y - 0.3, col=style.INK, txt="ordinal + frame" if lab != "rebound (R³)" else "early write", fs=5.2)
    arrow(51.4, y - 0.3, 62.5, y - 0.3, col=col, txt="query", fs=5.2)
    box(63, y - 2.6, 5.4, 4.6, str(pick), fc=fill, ec=col, fs=6.6)
    xb = 70 + (pick - 1) * 3.9 + 1.6
    ax.add_patch(FancyArrowPatch((68.6, y - 0.3), (xb, 32.6), arrowstyle="-|>", mutation_scale=5, lw=.6, color=col, ls=(0, (2, 1.2)),
                                 connectionstyle="arc3,rad=-0.12"))
ax.text(41, 8.8, "frame never combined with $N$", ha="center", fontsize=5.4, color=style.SURFACE, style="italic")
ax.text(83.6, 22.5, "start: block 5", ha="center", fontsize=5.6, color=style.TARGET)
ax.text(83.6, 14.6, "end-relative: literal block 3\n(smaller) or a blurred count (27B)", ha="center", fontsize=5.6, color=style.SURFACE)
ax.text(83.6, 4.4, "rebound: block 5, and the answer\nfollows a new picture placed there", ha="center", fontsize=5.6, color=style.TARGET)
os.makedirs(OUT, exist_ok=True); style.save(fig, f"{OUT}/fig_circuit"); plt.close(fig); print("ok fig_circuit")
