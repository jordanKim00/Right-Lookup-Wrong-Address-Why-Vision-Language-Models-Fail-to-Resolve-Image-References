"""Figure 1 (paper overhaul, 2026-10-14): knowing is not using.  The paradox of the earlier "main figure" (phenomenon.png: F = index, H = retrieval,
H(F(X)) = the composed reference), redrawn flat with real photographs (Imagenette validation images) and a fourth row with R3.
Seven images; "the third image from the end" is image 5 (truck); the literal image 3 is the parachute.  Rates are Qwen3.5-27B's: index 95% (greedy
number; appendix table), retrieval of the named image 99%, composed reference 65% (S643; bare-token re-measurement S784 pending), with R3 99%
(procedural scenes).  The wrong answer drawn in the third row (golf ball, image 4) is schematic: 27B's errors fall near the target.
Output: /vlm/results/figures/c_circuit/fig_paradox.{pdf,png}.  Usage: python figs/make_fig_paradox.py [--composed 65]"""
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image

sys.path.insert(0, "/vlm/exp/figs")
import style

OUT = "/vlm/results/figures/c_circuit"
ROOT = "/vlm/assets/imagenette2-160/val"
PICS = [("n02102040", 4, "dog"), ("n03028079", 2, "church"), ("n03888257", 2, "parachute"), ("n03445777", 7, "golf ball"),
        ("n03417042", 6, "truck"), ("n03394916", 0, "horn"), ("n01440764", 5, "fish")]
SANS, MONO = "DejaVu Sans", "DejaVu Sans Mono"
TEAL, ORANGE, GREY = style.TARGET, style.SURFACE, "#6b706d"


def load(c, i, size=96):
    f = sorted(os.listdir(f"{ROOT}/{c}"))[i]
    im = Image.open(f"{ROOT}/{c}/{f}").convert("RGB"); w, h = im.size; s = min(w, h)
    return np.asarray(im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((size, size)))


def main(composed):
    style.apply()
    ims = [load(c, i) for c, i, _ in PICS]
    H = 2.3; ar = style.ONE / H
    fig = plt.figure(figsize=(style.ONE, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    w, gap = .104, .02; h = w * ar; y0 = .985 - .055 - h
    x0 = .5 - (7 * w + 6 * gap) / 2
    for i, im in enumerate(ims):
        x = x0 + i * (w + gap)
        ax.imshow(im, extent=(x, x + w, y0, y0 + h), zorder=2, aspect="auto")
        ax.add_patch(Rectangle((x, y0), w, h, fill=False, ec="#b9b9b4", lw=.5, zorder=3))
        ax.text(x + w / 2, y0 - .03, str(i + 1), ha="center", va="center", fontsize=5.6, family=SANS, color=GREY)
        if i == 4:
            ax.add_patch(Rectangle((x - .008, y0 - .011), w + .016, h + .022, fill=False, ec=TEAL, lw=1.1, ls=(0, (2.2, 1.2)), zorder=4))
    ax.text(x0 + 4 * (w + gap) + w / 2, y0 + h + .018, "the third image from the end", ha="center", va="bottom", fontsize=5.3, family=SANS, color=TEAL)
    rows = [("F", "index", "Which image is the third from the end?", "5", None, True, "95%"),
            ("H", "retrieval", "What is in image 5?", "truck", 4, True, "99%"),
            ("H\u2218F", "composed", "What is in the third image from the end?", "golf ball", 3, False, f"{composed}%"),
            ("+R\u00b3", "rebound", "What is in the third image from the end?", "truck", 4, True, "99%")]
    top = y0 - .085; dy = .165; bh = .1
    ax.text(.995, top + .012, "Qwen3.5-27B", ha="right", va="bottom", fontsize=4.9, family=SANS, color=GREY)
    for r, (sym, lab, q, ans, img, ok, pct) in enumerate(rows):
        y = top - (r + 1) * dy + (.03 if r == 3 else 0) - (.0 if r < 3 else .0); c = TEAL if ok else ORANGE
        y = top - bh - r * dy - (.025 if r == 3 else 0)
        if r == 3:
            ax.plot([.01, .99], [y + bh + .028, y + bh + .028], color="#d8ddd8", lw=.6)
        ym = y + bh / 2
        ax.text(.01, ym + .022, sym, ha="left", va="center", fontsize=7.0, family=SANS, fontweight="bold", color=style.INK)
        ax.text(.01, ym - .03, lab, ha="left", va="center", fontsize=4.7, family=SANS, color=GREY)
        ax.add_patch(FancyBboxPatch((.105, y), .56, bh, boxstyle="round,pad=0,rounding_size=0.012", fc="white", ec="#cfcfca", lw=.5))
        ax.text(.118, ym, q, ha="left", va="center", fontsize=4.4, family=MONO, color=style.INK)
        ax.annotate("", xy=(.696, ym), xytext=(.671, ym), arrowprops=dict(arrowstyle="-|>", lw=.9, color="#8f2f2f", mutation_scale=5))
        if img is not None:
            tw = .062; th = tw * ar
            ax.imshow(ims[img], extent=(.703, .703 + tw, ym - th / 2, ym + th / 2), zorder=2, aspect="auto")
            ax.add_patch(Rectangle((.703, ym - th / 2), tw, th, fill=False, ec=c, lw=.9, zorder=3))
            ax.add_patch(FancyBboxPatch((.772, ym - .033), .1, .066, boxstyle="round,pad=0,rounding_size=0.01", fc="#fff5dc", ec="#e3c98b", lw=.5))
            ax.text(.822, ym, ans, ha="center", va="center", fontsize=4.9, family=SANS, color=style.INK)
        else:
            ax.add_patch(FancyBboxPatch((.703, ym - .033), .045, .066, boxstyle="round,pad=0,rounding_size=0.01", fc="#fff5dc", ec="#e3c98b", lw=.5))
            ax.text(.7255, ym, ans, ha="center", va="center", fontsize=5.4, family=MONO, color=style.INK)
        ax.text(.895, ym, "\u2713" if ok else "\u2717", ha="center", va="center", fontsize=7.2, color=c, family=SANS, fontweight="bold")
        ax.text(.995, ym, pct, ha="right", va="center", fontsize=6.0, family=SANS, fontweight="bold", color=c)
    ax.text(.118, y - .035, "address and start frame rewritten in the model's states; the text is unchanged", ha="left", va="center", fontsize=4.2, family=SANS, color=GREY)
    os.makedirs(OUT, exist_ok=True); style.save(fig, f"{OUT}/fig_paradox"); plt.close(fig); print("ok fig_paradox")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--composed", type=int, default=65)
    main(ap.parse_args().composed)
