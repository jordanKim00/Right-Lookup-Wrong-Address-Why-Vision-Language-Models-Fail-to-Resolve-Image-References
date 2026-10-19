"""v3 figures from saved data only (no model calls).  Output: research/ces/paper_v3_20261017/figures/.
  fig_hero     Figure 1 as v2 (same measured bars, exp/figs/make_v2_figs.py), relabelled: the late update moves the copied address and is not
               stopped at the target (v2's 'one step' / 'needed: 4 steps' wording is retired, S844-S847).
  fig_stop     (a) S846 (registered, fresh scenes): share of scenes staying at the pinned early position p, per p and stated target t = N_s - 3,
               the bar with t = p marked; (b) S846 band swap at the same pinned position (p in {3, 5, 6}): shares answering the ordinal k = 4, the early
               position p, or another image (END's departure), for END pinned, END with START's band, START pinned, START with END's band;
               (c) S847: share of the band's correction restored by band-corrected late queries (51 + 55, 59 + 63, all four, address part only) and
               removed by END's late queries in the corrected run, with 95% intervals.
  fig_predict  predicted vs observed shares for predictions fixed in advance: (a) answer-distribution laws that passed (S828 two-route law, S832
               two-component law, S844 S841-fitted law, S846 position law); (b) the late-score readout frozen on S845 applied to S847 (failed, 60%).
  other figures (grid, stages, phenomenon, repair, scale) are the v2 functions written to the v3 folder.
Usage (container): python /vlm/exp/figs/make_v3_figs.py"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/vlm/exp/figs")
import style
import make_v2_figs as v2

style.apply()
R = "/vlm/results/r3"; O = "/vlm/research/ces/outputs"; OUT = "/vlm/research/ces/paper_v3_20261017/figures"
os.makedirs(OUT, exist_ok=True)
v2.OUT = OUT


def fig_hero():
    from matplotlib.patches import Rectangle, FancyArrowPatch
    rows = []
    for fn in ("s827_s827_27b_natural", "s828_s828_27b_natural", "s829_s829_27b_natural", "s830_s830_27b_anchor", "s833_s833_27b_r3mode"):
        try:
            rows += [r for r in json.load(open(f"{R}/{fn}.json"))["rows"] if r["N"] == 9 and r["k"] == 5]
        except FileNotFoundError:
            pass
    e9 = np.bincount([r["land"]["END"] for r in rows], minlength=10)[1:] / len(rows)
    d7 = json.load(open(f"{R}/s831_s831b_27b_anchor.json")); r7 = [r for r in d7["rows"] if r["N"] == 9 and r["k"] == 7]
    e7 = np.bincount([r["land"]["END"] for r in r7], minlength=10)[1:] / len(r7)
    fig = plt.figure(figsize=(style.ONE, 2.05))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 10.6); ax.set_ylim(0, 6.05); ax.axis("off")
    def strip(y0, k, j, dist, title):
        ax.text(.25, y0 + 1.55, title, fontsize=6.6, color=style.INK, va="bottom")
        for i in range(1, 10):
            x0 = .25 + (i - 1) * 1.1
            ax.add_patch(Rectangle((x0, y0), .95, .72, facecolor=style.TARGET if i == j else "#eef1ee", edgecolor=style.TARGET if i == j else style.MUTED,
                                   lw=.9 if i == j else .4))
            ax.text(x0 + .475, y0 + .36, str(i), ha="center", va="center", fontsize=6.4, color="white" if i == j else style.INK)
            if dist[i - 1] > 0:
                ax.add_patch(Rectangle((x0 + .2, y0 - .05 - 1.1 * dist[i - 1]), .55, 1.1 * dist[i - 1], facecolor=style.SURFACE if i != j else style.TARGET, lw=0))
        cx = lambda i: .25 + (i - 1) * 1.1 + .475
        ax.add_patch(FancyArrowPatch((cx(k), y0 + 1.45), (cx(k), y0 + .78), arrowstyle="-|>", mutation_scale=6, color=style.SURFACE, lw=.9))
        ax.text(cx(k) + .1, y0 + 1.15, "1 copy", fontsize=5.8, color=style.SURFACE, ha="left")
        return cx
    cx = strip(4.05, 7, 3, e7, "“the 7th image from the end”  (target 3)")
    ax.add_patch(FancyArrowPatch((cx(7) - .1, 4.05 + .9), (cx(6) + .1, 4.05 + .9), arrowstyle="-|>", mutation_scale=6, color=style.ACCENT, lw=.9,
                                 connectionstyle="arc3,rad=-.6"))
    ax.text(cx(6.4), 4.05 + 1.45, "2 late update", fontsize=5.8, color=style.ACCENT, ha="center")
    ax.add_patch(FancyArrowPatch((cx(6) - .1, 4.05 + .9), (cx(3) + .15, 4.05 + .9), arrowstyle="-|>", mutation_scale=5, color=style.MUTED, lw=.5, ls=(0, (2, 2)),
                                 connectionstyle="arc3,rad=-.25"))
    ax.text(cx(4.2), 4.05 + 1.12, "target not reached", fontsize=5.2, color=style.MUTED, ha="center")
    cx = strip(1.15, 5, 5, e9, "“the 5th image from the end”  (target 5 = copy)")
    for t in (4, 6):
        ax.add_patch(FancyArrowPatch((cx(5) + (.15 if t > 5 else -.15), 1.15 + .9), (cx(t) + (-.1 if t > 5 else .1), 1.15 + .9), arrowstyle="-|>", mutation_scale=6,
                                     color=style.ACCENT, lw=.9, connectionstyle=f"arc3,rad={-.6 if t < 5 else .6}"))
    ax.text(cx(7.1), 1.15 + 1.2, "2 not stopped at the target", fontsize=5.8, color=style.ACCENT, ha="left")
    ax.plot([.5, .5], [1.10, 1.10 - 1.1 * .5], color=style.INK, lw=.6, solid_capstyle="butt")
    for yy in (1.10, 1.10 - 1.1 * .5):
        ax.plot([.44, .56], [yy, yy], color=style.INK, lw=.6)
    ax.text(.62, 1.10 - 1.1 * .25, "50%", fontsize=5.0, color=style.INK, va="center", ha="left")
    ax.text(10.4, .15, "bars below the images: measured answers of Qwen3.5-27B", fontsize=5.2, color=style.MUTED, ha="right")
    style.save(fig, f"{OUT}/fig_hero")


def fig_stop():
    d = json.load(open(f"{R}/s846_s846_27b_swap.json")); Rr = d["rows"]
    a = json.load(open(f"{R}/s846_analysis.json")); sd = json.load(open(f"{O}/s846_swapdesc.json")); f7 = json.load(open(f"{R}/s847_analysis.json"))["f"]
    PS = (3, 4, 5, 6); NSS = (6, 7, 8, 9)
    fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(style.TWO, 1.95), gridspec_kw=dict(width_ratios=[1.35, 1.0, 1.05]))
    # (a) staying at p, per p and stated target
    wbar = .19
    for i, p in enumerate(PS):
        for jx, ns in enumerate(NSS):
            t = ns - 3
            v = float(np.mean([r["land"][f"E_EARLY_p{p}"] == p for r in Rr if r["Ns"] == ns]))
            x = i + (jx - 1.5) * wbar
            axa.bar(x, v, width=wbar * .9, color=style.TARGET if t == p else style.GREYS[1], lw=0)
    axa.set_xticks(range(4)); axa.set_xticklabels([f"$p$={p}" for p in PS]); axa.set_ylim(0, .8); axa.set_ylabel("stays at $p$")
    axa.set_title("(a) staying at the early position", fontsize=7, loc="left")
    axa.text(.02, .93, "teal: stated target $=p$", transform=axa.transAxes, fontsize=5.6, color=style.TARGET, va="top")
    D, ci = a["NOSTOP"]["D"], a["NOSTOP"]["ci"]
    sg = lambda x: ("+" if x >= 0 else "\u2212") + f"{abs(100 * x):.1f}"
    axa.text(.02, .82, f"target $=p$ vs other: {sg(D)} pts [{sg(ci[0])}, {sg(ci[1])}]", transform=axa.transAxes, fontsize=5.4,
             color=style.INK, va="top")
    # (b) band swap outcomes
    arms = [("E_EARLY", "END"), ("BANDF", "END +\nSTART band"), ("S_EARLY", "START"), ("BANDE", "START +\nEND band")]
    cols = [("k", "ordinal $k$", style.SURFACE), ("p", "early pos. $p$", style.ACCENT), ("other", "other image", style.GREYS[0])]
    for i, (arm, lab) in enumerate(arms):
        bottom = 0
        for key, _, col in cols:
            v = sd["outcome"][arm][key][0]
            axb.bar(i, v, bottom=bottom, color=col, width=.7, lw=0); bottom += v
    axb.set_xticks(range(4)); axb.set_xticklabels([l for _, l in arms], fontsize=5.6); axb.set_ylim(0, 1); axb.set_ylabel("answer share")
    axb.set_title("(b) swapping layers 46–49", fontsize=7, loc="left")
    for key, lab, col in cols:
        axb.bar(0, 0, color=col, label=lab)
    axb.legend(fontsize=5.2, loc="upper center", bbox_to_anchor=(.5, -.28), ncol=3, handlelength=.8, columnspacing=.6)
    # (c) late-query path
    items = [("Q5155", "51+55", "f"), ("Q5963", "59+63", "f"), ("QALL", "51–63", "f"), ("QALL_U", "51–63 address", "f"), ("NEC", "51–63 needed", "g")]
    for i, (arm, lab, kind) in enumerate(items):
        v = f7[arm][kind]; lo, hi = f7[arm]["ci"]
        axc.barh(i, v, color=style.ACCENT if kind == "f" else style.SURFACE, height=.62, lw=0)
        axc.plot([lo, hi], [i, i], color=style.INK, lw=.6)
    axc.set_yticks(range(len(items))); axc.set_yticklabels([l for _, l, _ in items], fontsize=5.8); axc.invert_yaxis(); axc.set_xlim(0, 1.05)
    axc.set_xlabel("share of the band's correction"); axc.set_title("(c) read by the late queries", fontsize=7, loc="left")
    fig.tight_layout(w_pad=1.0)
    style.save(fig, f"{OUT}/fig_stop")


def fig_predict():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(style.ONE, 1.75), sharey=True)
    for ax in (a1, a2):
        ax.fill_between([0, 1], [-.15, .85], [.15, 1.15], color=style.RULE, alpha=.6, lw=0)
        ax.plot([0, 1], [0, 1], color=style.MUTED, lw=.6); ax.set_xlim(-.02, 1); ax.set_ylim(-.02, 1)
    pf = json.load(open(f"{O}/s828_predictions_s828_27b.json")); d2 = json.load(open(f"{R}/s828_s828_27b_twoway.json"))
    xs, ys = [], []
    for cname, res in pf["cases"].items():
        N = int(cname.split("k")[0][1:]); j = int(cname.split("j")[1])
        rows = [r for r in d2["rows"] if r["N"] == N]
        for arm, v in res["arms"].items():
            for x in (j - 1, j, j + 1):
                xs.append(v["shares"][str(x)]); ys.append(float(np.mean([r["land"][arm] == x for r in rows])))
    a1.scatter(xs, ys, s=5, color=style.ACCENT, alpha=.8, lw=0, label="two-route (S828)")
    A = json.load(open(f"{R}/s832_analysis.json")); xs2, ys2 = [], []
    for cname, c in A["cases"].items():
        for p, x in c["probes"].items():
            xs2 += x["pred"]; ys2 += x["obs"]
    a1.scatter(xs2, ys2, s=5, color=style.TARGET, alpha=.75, lw=0, marker="D", label="two-component (S832)")
    for tag, fl, key, col, mk, lab in (("s844", "s844_frozen_laws.json", "mix", style.SURFACE, "s", "S841 law on S844"),
                                       ("s846", "s846_frozen_laws.json", "pos", style.INK, "^", "position law (S846)")):
        laws = json.load(open(f"{O}/{fl}"))["predictions_N8"]
        dd = json.load(open(f"{R}/{tag}_{tag}_27b_{'anchor' if tag == 's844' else 'swap'}.json"))["rows"]
        xs3, ys3 = [], []
        for ns in (6, 7, 8, 9):
            for p in (3, 4, 5, 6):
                obs = np.bincount([r["land"][f"E_EARLY_p{p}"] for r in dd if r["Ns"] == ns], minlength=9)[1:]
                obs = obs / max(1, sum(1 for r in dd if r["Ns"] == ns))
                xs3 += list(laws[f"Ns{ns}_p{p}"][key]); ys3 += list(obs)
        a1.scatter(xs3, ys3, s=4, color=col, alpha=.55, lw=0, marker=mk, label=lab)
    a1.set_xlabel("predicted share"); a1.set_ylabel("observed share"); a1.set_title("(a) answer laws", fontsize=7, loc="left")
    a1.legend(fontsize=4.6, loc="lower right", handletextpad=.1, borderaxespad=.1, markerscale=1.3)
    q = json.load(open(f"{R}/s847_analysis.json"))["QKPRED"]
    a2.scatter([c["pred"] for c in q["cells"].values()], [c["obs"] for c in q["cells"].values()], s=7, facecolors="none", edgecolors=style.SURFACE, lw=.6)
    a2.set_xlabel("predicted share"); a2.set_title(f"(b) late-score readout ({int(100 * q['calibration'] + .5)}%)", fontsize=7, loc="left")
    fig.tight_layout(w_pad=.6)
    style.save(fig, f"{OUT}/fig_predict")


if __name__ == "__main__":
    for f in (fig_hero, fig_stop, fig_predict, v2.fig_grid, v2.fig_phenomenon, v2.fig_stages, v2.fig_repair, v2.fig_scale):
        try:
            f(); print(f.__name__, "ok")
        except Exception as e:
            import traceback; traceback.print_exc(); print(f.__name__, "FAILED", e)
