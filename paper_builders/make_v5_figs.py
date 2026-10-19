"""v5 figures from saved registered results only (no model calls).  Output: research/ces/paper_v5_20261018/figures/.
  fig_writer   Act III  (a) S853 (registered): answers on discordant runs after replacing rows B1, B4 at the writer input (layer 46) with a
                        donor run that differs in the early position p or in the ordinal k (share following the donor / the recipient);
                        (b) S854-2 (registered): Shapley shares of the ordinal-condition effect carried by four input-row groups at layer 46,
                        on the late image scores and on the menu logits.
  fig_visual   Act IV   (a) S855 (registered): share of the 6 -> 8 picture effect on the logits of pictures 1-6 produced by the 8-picture
                        position geometry alone (all layers / late layers), and kept by pictures 7-8 under the 6-picture geometry;
                        (b) S856b (registered): share of the count effect on the two-target logit gap moved by swapping the late queries,
                        both directions, four arms; (c) S857 (registered): share of the count effect kept under source cuts and restores.
  fig_heldout  Act V    S858 (registered, held-out input configurations): (a) predicted vs observed change of the late image scores under the
                        complete cut (per scene, cell and picture); (b) predicted vs realized shares of the count effect on the logit gap for
                        the cut, the late-query restore and removal, and the never-run full-attention-only cut.
Usage: python3 exp/figs/make_v5_figs.py"""
from __future__ import annotations

import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import style

V = "/home/storage2/ai26269345/vlm"
R = f"{V}/results/r3"
OUT = f"{V}/research/ces/paper_v5_20261018/figures"
style.apply()


def errbar_h(ax, i, v, lo, hi, color, h=.62):
    ax.barh(i, v, color=color, height=h, lw=0); ax.plot([lo, hi], [i, i], color=style.INK, lw=.6)


def fig_writer():
    a = json.load(open(f"{R}/s853_analysis.json"))["TYPES"]; b = json.load(open(f"{R}/s854b_analysis.json"))["SHAPLEY"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(style.ONE, 1.6), gridspec_kw=dict(width_ratios=[1.0, 1.25]))
    items = [("P", "early position $p$"), ("K", "ordinal $k$")]
    for i, (key, lab) in enumerate(items):
        fd, fr = a[key]["f_don"][0], a[key]["f_rec"][0]
        ax1.bar(i - .18, fd, width=.34, color=style.ACCENT, lw=0); ax1.bar(i + .18, fr, width=.34, color=style.GREYS[1], lw=0)
        for x, (lo, hi) in ((i - .18, a[key]["f_don"][1]), (i + .18, a[key]["f_rec"][1])):
            ax1.plot([x, x], [lo, hi], color=style.INK, lw=.6)
    ax1.set_xticks(range(2)); ax1.set_xticklabels([l for _, l in items], fontsize=6); ax1.set_ylim(0, 1); ax1.set_ylabel("answer share")
    ax1.bar(0, 0, color=style.ACCENT, label="donor's answer"); ax1.bar(0, 0, color=style.GREYS[1], label="recipient's")
    ax1.legend(fontsize=5.2, loc="upper right", handlelength=.8)
    ax1.set_title("(a) B1, B4 from a donor", fontsize=7, loc="left")
    groups = [("Q", "question tail"), ("A", "B1, B4"), ("M", "menu"), ("O", "B0, B2, B3")]
    for j, (key, meas, col, lab) in enumerate((("Z", "Z", style.ACCENT, "late image scores"), ("L", "L", style.TARGET, "menu logits"))):
        for i, (g, _) in enumerate(groups):
            ax2.barh(i + (j - .5) * .36, b[key][g], height=.34, color=col, lw=0, label=lab if i == 0 else None)
    ax2.set_yticks(range(4)); ax2.set_yticklabels([l for _, l in groups], fontsize=6); ax2.invert_yaxis(); ax2.set_xlim(-.02, .6)
    ax2.axvline(0, color=style.MUTED, lw=.5); ax2.set_xlabel("share of the ordinal's effect"); ax2.legend(fontsize=5.2, loc="lower right", handlelength=.8)
    ax2.set_title("(b) four input-row groups", fontsize=7, loc="left")
    fig.tight_layout(w_pad=.8)
    style.save(fig, f"{OUT}/fig_writer")


def fig_visual():
    g = json.load(open(f"{R}/s855_analysis.json"))["shares"]["L6"]
    s6 = json.load(open(f"{R}/s856b_analysis.json"))["RESULTS"]
    s7 = json.load(open(f"{R}/s857_analysis.json"))["shares"]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(style.TWO, 2.05), gridspec_kw=dict(width_ratios=[.8, 1.0, 1.35]))
    items = [("G_all", "8-picture geometry,\nall layers"), ("G_late", "8-picture geometry,\nlate layers"), ("X_pic", "pictures 7–8 under\n6-picture geometry")]
    for i, (key, lab) in enumerate(items):
        v, (lo, hi) = g[key][0], g[key][1]
        errbar_h(a1, i, v, lo, hi, style.TARGET if key == "X_pic" else style.GREYS[1])
        a1.text(max(v, hi) + .04, i, f"{v:.3f}".lstrip("0") if abs(v) < .1 else f"{v:.2f}".lstrip("0"), va="center", fontsize=5.6)
    a1.set_yticks(range(3)); a1.set_yticklabels([l for _, l in items], fontsize=5.6); a1.invert_yaxis(); a1.set_xlim(-.05, 1.05)
    a1.set_xlabel("share of the 6→8 effect"); a1.set_title("(a) geometry or pictures", fontsize=7, loc="left")
    arms = [("A_QX", "whole query"), ("NC_QX", "whole query,\nno count sentence"), ("A_QXA", "rows B1, B4, answer"), ("A_QXU", "address part")]
    for i, (key, lab) in enumerate(arms):
        for j, (d_, col) in enumerate((("T_6from8", style.ACCENT), ("T_8from6", style.SURFACE))):
            v, (lo, hi) = s6[key]["g"][d_][0], s6[key]["g"][d_][1]
            y = i + (j - .5) * .36
            a2.barh(y, v, height=.34, color=col, lw=0, label=("8-picture queries into the 6-picture run" if j == 0 else "6-picture queries into the 8-picture run") if i == 0 else None)
            a2.plot([lo, hi], [y, y], color=style.INK, lw=.6)
    a2.axvline(1, color=style.MUTED, lw=.5, ls="--"); a2.set_yticks(range(4)); a2.set_yticklabels([l for _, l in arms], fontsize=5.6)
    a2.invert_yaxis(); a2.set_xlim(0, 1.1); a2.set_xlabel("share of the count effect moved")
    a2.legend(fontsize=5.0, loc="upper center", bbox_to_anchor=(.42, -.25), handlelength=.8, ncol=1)
    a2.set_title("(b) swapping the late queries", fontsize=7, loc="left")
    arms7 = [("CA", "cut everywhere", style.GREYS[0]), ("CQ", "cut before layer 51", style.GREYS[0]), ("CR", "cut in layers 51–63", style.TARGET),
             ("CQF", "cut full attention, 0–50", style.GREYS[0]), ("CQD", "cut DeltaNet, 0–50", style.ACCENT),
             ("O1", "open only layers 0–34", style.GREYS[0]), ("O2", "open only layers 35–50", style.ACCENT),
             ("RW", "cut + restore B1, B4 at 46", style.GREYS[0]), ("RQ", "cut + restore late queries", style.TARGET),
             ("KQ", "remove late queries", style.SURFACE)]
    for i, (key, lab, col) in enumerate(arms7):
        v, (lo, hi) = s7[key]["g"][0], s7[key]["g"][1]
        errbar_h(a3, i, v, lo, hi, col, h=.66)
    a3.axvline(0, color=style.MUTED, lw=.5); a3.axvline(1, color=style.MUTED, lw=.5, ls="--")
    a3.set_yticks(range(len(arms7))); a3.set_yticklabels([l for _, l, _ in arms7], fontsize=5.6); a3.invert_yaxis(); a3.set_xlim(-.25, 1.1)
    a3.set_xlabel("share of the count effect kept"); a3.set_title("(c) cutting pictures 7–8 from the text", fontsize=7, loc="left")
    fig.tight_layout(w_pad=.8)
    style.save(fig, f"{OUT}/fig_visual")


def fig_heldout():
    an = json.load(open(f"{R}/s858_analysis.json")); d = json.load(open(f"{R}/s858_s858_27b_pred.json"))
    LATE = [str(l) for l in d["late"]]; KN = {int(k): tuple(v) for k, v in d["kn"].items()}
    def zsh(r, N, arm, k, p):
        q = r["qk"][f"N{N}_{arm}_k{k}p{p}"]
        z = np.mean([[q[L][ri][:N - 2] for ri in (2, 5, 9)] for L in LATE], axis=(0, 1)); return z - z.mean()
    xs, ys = [], []
    for r in d["rows"]:
        for N in d["ns"]:
            for k in KN[N]:
                for p in d["ps"]:
                    base = zsh(r, N, "Nn", k, p)
                    xs += list(zsh(r, N, "Nm2", k, p) - base); ys += list(zsh(r, N, "CA", k, p) - base)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(style.ONE, 1.7), gridspec_kw=dict(width_ratios=[1.0, 1.15]))
    a1.scatter(xs, ys, s=1.2, color=style.ACCENT, alpha=.35, lw=0)
    lim = max(np.abs(xs).max(), np.abs(ys).max()) * 1.05
    a1.plot([-lim, lim], [-lim, lim], color=style.MUTED, lw=.5); a1.set_xlim(-lim, lim); a1.set_ylim(-lim, lim)
    a1.set_xlabel("predicted change"); a1.set_ylabel("observed change")
    a1.set_title(f"(a) image scores, $r$={an['L2a_stats']['r'][0]:.2f}", fontsize=7, loc="left")
    items = [("CA", "cut", 0.0), ("RQ", "restore\nqueries", .80), ("KQ", "remove\nqueries", .17), ("CF", "FA-only\ncut 35–50", 0.0)]
    for i, (key, lab, pred) in enumerate(items):
        v, (lo, hi) = an["L3a"][key]["realized"][0], an["L3a"][key]["realized"][1]
        col = style.SURFACE if key == "CF" else style.TARGET
        a2.bar(i, v, width=.6, color=col, lw=0); a2.plot([i, i], [lo, hi], color=style.INK, lw=.6)
        a2.plot([i - .36, i + .36], [pred, pred], color=style.INK, lw=1.0, ls=(0, (2, 1)))
    a2.set_xticks(range(len(items))); a2.set_xticklabels([l for _, l, _ in items], fontsize=5.4); a2.set_ylim(-.05, 1.0)
    a2.set_ylabel("share of the count effect"); a2.set_title("(b) logit gap", fontsize=7, loc="left")
    a2.text(.02, .97, "dashes: predicted", transform=a2.transAxes, fontsize=5.2, va="top")
    fig.tight_layout(w_pad=.7)
    style.save(fig, f"{OUT}/fig_heldout")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for f_ in (fig_writer, fig_visual, fig_heldout):
        f_(); print("wrote", f_.__name__)
