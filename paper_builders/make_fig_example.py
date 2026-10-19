"""One MMIU photograph item, end-relative options: the index-matching heads of Qwen3.5-27B at each option's ordinal token, before and
after R³ (saved S737 records, final Read; no model call).  Writes results/figures/c_circuit/fig_example.{pdf,png}.
Usage: python figs/make_fig_example.py [ITEM]   (default 308, the dev-split item with the clearest contrast among lure-picked items)"""
import ast, json, os, sys
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

sys.path.insert(0, "/vlm/exp"); sys.path.insert(0, "/vlm/exp/figs")
import style
import s737_natqk as Q

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth"]
OUT = "/vlm/results/figures/c_circuit"
lv = lambda x: ast.literal_eval(x) if isinstance(x, str) else x


def main(idx):
    rec = json.load(open("/vlm/results/r3/s737_nq737f_27b_natqk.json"))["rows"][idx]
    task, item, opts = Q.dev_items()[idx]
    N = len(item["input"]["input_image_path"]); gold = rec["gold"]
    E, R = lv(rec["END"]), lv(rec["R3"])
    img = dict(opts); p = img[gold]; lure = [L for L, k in opts if k == N + 1 - p][0]
    style.apply()
    fig = plt.figure(figsize=(style.ONE, 2.25))
    gs = fig.add_gridspec(2, N, height_ratios=[1.0, 1.55], hspace=.28, wspace=.08, left=.03, right=.97, top=.93, bottom=.02)
    for b in range(N):
        a = fig.add_subplot(gs[0, b])
        im = Image.open(f"/vlm/assets/mmiu/{task}/{item['input']['input_image_path'][b].split('/')[-1]}").convert("RGB")
        a.imshow(im); a.set_xticks([]); a.set_yticks([])
        col = style.TARGET if b + 1 == p else style.RULE
        for s_ in a.spines.values():
            s_.set_visible(True); s_.set_edgecolor(col); s_.set_linewidth(1.8 if b + 1 == p else .6)
        a.set_title(f"image {b + 1}" + (" (answer)" if b + 1 == p else ""), fontsize=5.6, pad=2, color=style.INK)
    ax = fig.add_subplot(gs[1, :])
    ax.set_xlim(0, 1); ax.set_ylim(-1.5, len(opts) + .2); ax.invert_yaxis(); ax.axis("off")
    desc = item["input"]["question"].split("Description:")[-1].strip().split(".")[0]
    ax.text(0.0, -1.25, f"Which image fits \u201c{desc}\u201d?", fontsize=5.4, color=style.INK, va="center", style="italic")
    for i, ((L, k), e, r) in enumerate(zip(opts, E, R)):
        ref = f"{L}: the {ORDW[N - k]} image from the end"
        tag = " (answer)" if L == gold else (" (lure)" if L == lure else "")
        ax.text(0.0, i, ref + tag, fontsize=5.4, va="center", color=style.INK, fontweight="bold" if L in (gold, lure) else "normal")
        for x0, d, lab in ((.56, e, "unedited"), (.79, r, r"with R$^3$")):
            lit, nam = d.get("literal", d.get("named")), d["named"]
            ax.add_patch(Rectangle((x0, i - .3), .18 * lit, .26, color=style.SURFACE, lw=0))
            ax.add_patch(Rectangle((x0, i + .02), .18 * nam, .26, color=style.TARGET, lw=0))
            if i == 0:
                ax.text(x0, -.55, lab, fontsize=5.4, color=style.INK, va="bottom")
        for x0, pick in ((.545, rec["pick_END"]), (.775, rec["pick_R3"])):
            if pick == L:
                ax.text(x0, i, "▶", fontsize=5, color=style.INK, va="center", ha="right")
    ax.text(.0, len(opts) - .05, "attention share at the option's ordinal:", fontsize=4.8, color=style.MUTED, va="top")
    ax.text(.56, len(opts) - .05, "literal image", fontsize=4.8, color=style.SURFACE, va="top")
    ax.text(.79, len(opts) - .05, "named image", fontsize=4.8, color=style.TARGET, va="top")
    os.makedirs(OUT, exist_ok=True)
    style.save(fig, f"{OUT}/fig_example"); plt.close(fig)
    print("ok fig_example", task, "gold", gold, "lure", lure, "END", rec["pick_END"], "R3", rec["pick_R3"])


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 308)
