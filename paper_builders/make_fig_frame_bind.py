"""Two main-text figures for the 27B account (paper v3 revision, 2026-10-14).
fig_end4: S769 (N = 7) - answers on image 4 and the layer-39 lookup share of image 4 for END(4) (target = literal image 4) and START(4), after
          reading the other question's frame word at one band (exp/s769_analyze.py: acc, share4_L39; all heads, rows tail1 / answer).
fig_bind: S780 (N = 7, 8; 80 new scenes each) - share of answers that follow a new picture placed at the named position (R3 at its written
          address on the scenes it answers; natural END(2) at its target on the scenes it answers; natural END(5) at the image it answered and at
          its target on the scenes it answers wrongly), and the share changed by reversing the option order (exp/s779_analyze.py, run by S780).
Wilson 95% intervals from the per-cell scene counts.  Style: exp/figs/style.py.  Output: /vlm/results/figures/c_circuit/fig_{end4,bind}.{pdf,png}.
Usage: python figs/make_fig_frame_bind.py"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/vlm/exp/figs")
import style

OUT, R = "/vlm/results/figures/c_circuit", "/vlm/results/r3"
pick = lambda name: f"{R}/{name}_bare.json" if os.path.exists(f"{R}/{name}_bare.json") else f"{R}/{name}.json"   # S783: bare-token readout when available
LIGHT_T, LIGHT_A = "#9fd3cb", "#aab5e6"


def wilson(p, n, z=1.96):
    if not n:
        return 0., 0.
    d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p - (c - h), (c + h) - p


def fig_end4():
    a = json.load(open(pick("s769_analysis")))["N7"]
    rows = [("S4", "start-relative (``first'')".replace("``", "“").replace("''", "”"), 0),
            ("R4:P1", "+ “last” read at 27–34", 1),
            ("E4", "end-relative (“last”)", 0),
            ("F4:P1", "+ “first” read at 27–34", 1),
            ("F4:P1conv", "only via DeltaNet convolutions", 2),
            ("F4:P1fa", "only via attention 27, 31", 2),
            ("F4:P23", "+ “first” read at 35–63", 1)]
    acc = [a["acc"][k]["4"] for k, _, _ in rows]; sh = [a["share4_L39"][k] for k, _, _ in rows]
    fig, ax = plt.subplots(figsize=(style.ONE, 1.85))
    y = np.arange(len(rows))[::-1].astype(float); h = .36
    ax.barh(y + h / 2, [100 * v[0] for v in acc], h, color=style.TARGET, label="answers on image 4",
            xerr=[[100 * (v[0] - v[1]) for v in acc], [100 * (v[2] - v[0]) for v in acc]], error_kw=dict(lw=.5, capsize=1.2, ecolor=style.INK))
    ax.barh(y - h / 2, [100 * v for v in sh], h, color=LIGHT_A, label="image 4's share of image attention, layer 39")
    for yi, v in zip(y, acc):
        ax.text(100 * v[2] + 1.5, yi + h / 2, f"{100 * v[0]:.0f}", va="center", fontsize=5.6, color=style.INK)
    for yi, v in zip(y, sh):
        ax.text(100 * v + 1.5, yi - h / 2, f"{100 * v:.0f}", va="center", fontsize=5.6, color=style.MUTED)
    ax.set_yticks(y); ax.set_yticklabels([("    " * ind) + lab for _, lab, ind in rows], fontsize=6.3)
    for t, (_, _, ind) in zip(ax.get_yticklabels(), rows):
        if ind == 0:
            t.set_fontweight("bold")
    ax.axhline(4.5, color=style.RULE, lw=.6)
    ax.set_xlim(0, 112); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xlabel("%", labelpad=1)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(.42, 1.0), ncol=2, fontsize=5.6, handlelength=1.2, borderaxespad=.2, columnspacing=1.)
    style.save(fig, f"{OUT}/fig_end4"); plt.close(fig)


def fig_bind():
    B = json.load(open(pick("s780_analysis"))); c7, c8 = (B["s783_27b7"], B["s783_27b8"]) if "s783_27b7" in B else (B["s780_27b7"], B["s780_27b8"])
    groups = [("\\rthree", "R³:\nwritten\naddress", "visual_r3", "j", style.TARGET, LIGHT_T),
              ("E2", "END(2):\ntarget", "visual_e2", "j", style.ACCENT, LIGHT_A),
              ("E5a", "END(5):\nimage it\nnamed", "visual_end", "a", style.SURFACE, "#efb894"),
              ("E5j", "END(5):\ntarget", "visual_end", "j", style.SURFACE, "#efb894")]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(style.ONE, 2.05), gridspec_kw=dict(width_ratios=[3.3, 1.25], wspace=.32))
    x = np.arange(len(groups)); w = .36
    for i, (_, lab, key, f, c1, c2) in enumerate(groups):
        for dx, c, col in ((-w / 2, c7, c1), (w / 2, c8, c2)):
            p, n = c[key][f], c[key]["n"]; lo, hi = wilson(p, n)
            ax.bar(x[i] + dx, 100 * p, w, color=col, yerr=[[100 * lo], [100 * hi]], error_kw=dict(lw=.5, capsize=1.2, ecolor=style.INK))
            ax.text(x[i] + dx, 100 * (p + hi) + 2, f"{100 * p:.0f}", ha="center", va="bottom", fontsize=5.4)
            ax.text(x[i] + dx, 2.5, "7" if c is c7 else "8", ha="center", va="bottom", fontsize=5, color="white" if c is c7 else style.INK)
    ax.set_xticks(x); ax.set_xticklabels([g[1] for g in groups], fontsize=5.4)
    ax.set_ylim(0, 112); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_ylabel("answer follows the new picture (%)", fontsize=6.2, labelpad=1)
    ax.tick_params(axis="x", length=0)
    for i, (lab, key, col, c2) in enumerate((("END(5)", "END", style.SURFACE, "#efb894"), ("R³", "R3", style.TARGET, LIGHT_T))):
        for dx, c, cc in ((-w / 2, c7, col), (w / 2, c8, c2)):
            p = c["menu_reversal_change"][key]; lo, hi = wilson(p, c["n"])
            bx.bar(i + dx, 100 * p, w, color=cc, yerr=[[100 * lo], [100 * hi]], error_kw=dict(lw=.5, capsize=1.2, ecolor=style.INK))
            bx.text(i + dx, 100 * (p + hi) + 2, f"{100 * p:.0f}", ha="center", va="bottom", fontsize=5.4)
            bx.text(i + dx, 2.5, "7" if c is c7 else "8", ha="center", va="bottom", fontsize=5, color="white" if (c is c7 and p > .1) else style.INK)
    bx.set_xticks([0, 1]); bx.set_xticklabels(["END(5)", "R³"], fontsize=5.4); bx.tick_params(axis="x", length=0)
    bx.set_ylim(0, 112); bx.set_yticks([0, 25, 50, 75, 100]); bx.set_ylabel("answer changes when the\noptions are reversed (%)", fontsize=6.2, labelpad=1)
    style.save(fig, f"{OUT}/fig_bind"); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); style.apply()
    fig_end4(); fig_bind(); print("ok fig_end4 fig_bind")
