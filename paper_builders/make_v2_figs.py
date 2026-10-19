"""v2 figures from saved data only (no model calls).  Output: research/ces/paper_v2_20261017/figures/{fig_phenomenon,fig_stages,fig_predict,
fig_repair}.{pdf,png}.  Sources:
  fig_phenomenon  (a) /vlm/results/figures/tables_r3/bench_values.json (make_r3_tables.bench, MMIU held-out); (b) natural END(5) / START(5) at
                  N = 9 pooled over S827-S833 (same scene recipe, new scenes each; results/r3/s82[7-9]*, s830, s833).
  fig_stages      (a) S827 head-level local coefficients (research/ces/outputs/s827_heads_desc_s827_27b.json): picture-output-weighted mean over
                  heads of the END - START change of the copied block and the flanks, relative to the far blocks, per look layer;
                  (b) late picture attention mass (index-matching heads, B1 / B4 / answer row) per look layer, END vs START (S829 natural runs);
                  (c) END - START block-score change of the boundary newline rows at L51 per image, identity N = 9 (S827 natural runs).
  fig_predict     predicted vs observed answer shares: S828 frozen two-route law (research/ces/outputs/s828_predictions_s828_27b.json vs
                  results/r3/s828_s828_27b_twoway.json) and S832 frozen two-component law (results/r3/s832_analysis.json).
  fig_repair      S833 (results/r3/s833_analysis.json): accuracy and rescaled late picture mass per write and cell.
Usage (container): python /vlm/exp/figs/make_v2_figs.py"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/vlm/exp/figs")
import style

style.apply()
R = "/vlm/results/r3"; O = "/vlm/research/ces/outputs"; OUT = "/vlm/research/ces/paper_v2_20261017/figures"
os.makedirs(OUT, exist_ok=True)
MODELS = [("4b", "VL-4B"), ("8b", "VL-8B"), ("35_4b", "3.5-4B"), ("35_9b", "3.5-9B"), ("27b", "3.5-27B")]


def fig_hero():
    """Figure 1: copy first, correct late - a schematic on the images themselves, with the measured identity outcome (saved data only)."""
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
            col = style.TARGET if i == j else "#eef1ee"
            ax.add_patch(Rectangle((x0, y0), .95, .72, facecolor=col if i == j else "#eef1ee", edgecolor=style.TARGET if i == j else style.MUTED, lw=.9 if i == j else .4))
            ax.text(x0 + .475, y0 + .36, str(i), ha="center", va="center", fontsize=6.4, color="white" if i == j else style.INK)
            if dist[i - 1] > 0:
                ax.add_patch(Rectangle((x0 + .2, y0 - .05 - 1.1 * dist[i - 1]), .55, 1.1 * dist[i - 1], facecolor=style.SURFACE if i != j else style.TARGET, lw=0))
        cx = lambda i: .25 + (i - 1) * 1.1 + .475
        ax.add_patch(FancyArrowPatch((cx(k), y0 + 1.45), (cx(k), y0 + .78), arrowstyle="-|>", mutation_scale=6, color=style.SURFACE, lw=.9))
        ax.text(cx(k) + .1, y0 + 1.15, "1 copy", fontsize=5.8, color=style.SURFACE, ha="left")
        return cx
    cx = strip(4.05, 7, 3, e7, "\u201cthe 7th image from the end\u201d  (target 3)")
    ax.add_patch(FancyArrowPatch((cx(7) - .1, 4.05 + .9), (cx(6) + .1, 4.05 + .9), arrowstyle="-|>", mutation_scale=6, color=style.ACCENT, lw=.9,
                                 connectionstyle="arc3,rad=-.6"))
    ax.text(cx(6.4), 4.05 + 1.45, "2 one step", fontsize=5.8, color=style.ACCENT, ha="center")
    ax.add_patch(FancyArrowPatch((cx(6) - .1, 4.05 + .9), (cx(3) + .15, 4.05 + .9), arrowstyle="-|>", mutation_scale=5, color=style.MUTED, lw=.5, ls=(0, (2, 2)),
                                 connectionstyle="arc3,rad=-.25"))
    ax.text(cx(4.2), 4.05 + 1.12, "needed: 4 steps", fontsize=5.2, color=style.MUTED, ha="center")
    cx = strip(1.15, 5, 5, e9, "\u201cthe 5th image from the end\u201d  (target 5 = copy)")
    for t in (4, 6):
        ax.add_patch(FancyArrowPatch((cx(5) + (.15 if t > 5 else -.15), 1.15 + .9), (cx(t) + (-.1 if t > 5 else .1), 1.15 + .9), arrowstyle="-|>", mutation_scale=6,
                                     color=style.ACCENT, lw=.9, connectionstyle=f"arc3,rad={-.6 if t < 5 else .6}"))
    ax.text(cx(7.1), 1.15 + 1.2, "2 moves anyway", fontsize=5.8, color=style.ACCENT, ha="left")
    ax.plot([.5, .5], [1.10, 1.10 - 1.1 * .5], color=style.INK, lw=.6, solid_capstyle="butt")
    for yy in (1.10, 1.10 - 1.1 * .5):
        ax.plot([.44, .56], [yy, yy], color=style.INK, lw=.6)
    ax.text(.62, 1.10 - 1.1 * .25, "50%", fontsize=5.0, color=style.INK, va="center", ha="left")
    ax.text(10.4, .15, "bars below the images: measured answers of Qwen3.5-27B", fontsize=5.2, color=style.MUTED, ha="right")
    style.save(fig, f"{OUT}/fig_hero")


def fig_grid():
    """Answer shares of END(k) for N = 6-9 (natural runs, 40 scenes per cell).  Data: research/ces/outputs/v2_grid_landings.json (S825 for
    N = 6, 9; S824 k = 3-7 + S834 k = 1, 2, 8 for N = 7, 8; written by exp/s834_analyze.py only if S834's I0 holds), else s827_pre_landings.json
    with the unrun rows hatched.  Teal outline: the end-relative target N + 1 - k; orange outline: the copy k."""
    from matplotlib.patches import Rectangle
    G = f"{O}/v2_grid_landings.json"
    T = json.load(open(G))["cells"] if os.path.exists(G) else json.load(open(f"{O}/s827_pre_landings.json"))
    fig, axes = plt.subplots(1, 4, figsize=(style.TWO, 2.0))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("v2", ["#ffffff", "#5c6b63", "#18201c"])
    any_missing = False
    for ax, N in zip(axes, (6, 7, 8, 9)):
        M = np.zeros((N, N)); missing = []
        for k in range(1, N + 1):
            c = T.get(f"N{N}_k{k}")
            if not c:
                missing.append(k)
            if c:
                for x, v in c["landing"].items():
                    if str(x).isdigit() and 1 <= int(x) <= N:
                        M[k - 1, int(x) - 1] = v
        ax.imshow(M, cmap=cmap, vmin=0, vmax=1, origin="upper", extent=(.5, N + .5, N + .5, .5))
        any_missing = any_missing or bool(missing)
        for k in missing:
            ax.add_patch(Rectangle((.5, k - .5), N, 1, facecolor="#f4f5f4", edgecolor=style.RULE, hatch="////", lw=0))
        for k in range(1, N + 1):
            ax.add_patch(Rectangle((N + 1 - k - .5, k - .5), 1, 1, fill=False, edgecolor=style.TARGET, lw=1.0))
            if k != N + 1 - k:
                ax.add_patch(Rectangle((k - .5, k - .5), 1, 1, fill=False, edgecolor=style.SURFACE, lw=.8, ls=(0, (2, 1))))
        ax.set_xticks(range(1, N + 1)); ax.set_yticks(range(1, N + 1)); ax.tick_params(labelsize=5.4)
        ax.set_title(f"$N={N}$", fontsize=6.8); ax.set_xlabel("answered image", fontsize=6)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_ylabel("$k$ (the $k$-th from the end)", fontsize=6)
    h = [Rectangle((0, 0), 1, 1, fill=False, edgecolor=style.TARGET, lw=1.0, label="target $N+1-k$"),
         Rectangle((0, 0), 1, 1, fill=False, edgecolor=style.SURFACE, lw=.8, ls=(0, (2, 1)), label="copy $k$")]
    if any_missing:
        h.append(Rectangle((0, 0), 1, 1, facecolor="#f4f5f4", edgecolor=style.RULE, hatch="////", lw=0, label="not run"))
    fig.tight_layout(w_pad=.4, rect=(0, 0, 1, .86))
    fig.legend(handles=h, loc="upper center", ncol=3, bbox_to_anchor=(.42, 1.0))
    cax = fig.add_axes([.835, .915, .12, .028])
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, 1), cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_ticks([0, .5, 1]); cb.set_ticklabels(["0", ".5", "1"]); cb.outline.set_linewidth(.3)
    cax.xaxis.set_ticks_position("top"); cax.tick_params(labelsize=5.2, length=1.5, pad=1)
    fig.text(.828, .929, "share of scenes", fontsize=5.8, ha="right", va="center")
    style.save(fig, f"{OUT}/fig_grid")


def fig_phenomenon():
    V = json.load(open("/vlm/results/figures/tables_r3/bench_values.json"))
    fig, (a, b) = plt.subplots(1, 2, figsize=(style.ONE, 1.55), gridspec_kw=dict(width_ratios=[1.15, 1]))
    x = np.arange(len(MODELS)); w = .38
    F = [100 * V[f"{m}|MMIU held-out"]["F"] for m, _ in MODELS]; E = [100 * V[f"{m}|MMIU held-out"]["R"] for m, _ in MODELS]
    a.bar(x - w / 2, F, w, color=style.GREYS[0], label="from the start")
    a.bar(x + w / 2, E, w, color=style.SURFACE, label="from the end")
    a.set_xticks(x); a.set_xticklabels([n for _, n in MODELS], rotation=40, ha="right", fontsize=5.6)
    a.set_ylabel("MMIU accuracy (%)"); a.set_ylim(0, 100); a.legend(loc="upper left", fontsize=5.6, handlelength=1)
    a.set_title("(a) same items, reversed references", fontsize=6.6)
    rows = []
    for fn, key in (("s827_s827_27b_natural", None), ("s828_s828_27b_natural", None), ("s829_s829_27b_natural", None), ("s830_s830_27b_anchor", None),
                    ("s833_s833_27b_r3mode", None)):
        try:
            rows += [r for r in json.load(open(f"{R}/{fn}.json"))["rows"] if r["N"] == 9 and r["k"] == 5]
        except FileNotFoundError:
            pass
    N = 9; xs = np.arange(1, N + 1)
    e = np.bincount([r["land"]["END"] for r in rows], minlength=N + 1)[1:] / len(rows)
    s_ = np.bincount([r["land"]["START"] for r in rows], minlength=N + 1)[1:] / len(rows)
    b.bar(xs, 100 * e, .7, color=[style.TARGET if i == 5 else style.SURFACE for i in xs])
    b.text(9.4, 92, f"from the start: {100 * s_[4]:.0f}% on image 5", ha="right", fontsize=5.6, color=style.INK)
    b.set_xticks(xs); b.set_xlabel("answered image"); b.set_ylabel("share of scenes (%)"); b.set_ylim(0, 100)
    b.set_title(f"(b) 5th of 9 from the end ($n$={len(rows)})", fontsize=6.6)
    b.text(5, 100 * e[4] + 3, "target", ha="center", fontsize=5.6, color=style.TARGET)
    fig.tight_layout(w_pad=.6)
    style.save(fig, f"{OUT}/fig_phenomenon")


def fig_stages():
    H = json.load(open(f"{O}/s827_heads_desc_s827_27b.json"))
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(style.TWO, 1.75), gridspec_kw=dict(width_ratios=[1, 1, 1.1]))
    for case, ls, lab in (("N9k5", "-", "$N=9$"), ("N7k4", "--", "$N=7$")):
        Ls, cp, fl = [], [], []
        for L, lay in sorted(H[case].items(), key=lambda t: int(t[0])):
            hs = list(lay.values()); w = np.array([x["img_E"] for x in hs]); A = np.array([x["a"] for x in hs])
            m = (A * w[:, None]).sum(0) / w.sum(); Ls.append(int(L)); cp.append(m[1]); fl.append((m[0] + m[2]) / 2)
        a.plot(Ls, cp, ls=ls, color=style.SURFACE, marker="o", ms=2.5, lw=1)
        a.plot(Ls, fl, ls=ls, color=style.TARGET, marker="o", ms=2.5, lw=1)
    a.axhline(0, color=style.RULE, lw=.6); a.set_xlabel("layer"); a.set_ylabel("end $-$ start score change")
    a.set_title("(a) copy lowered early, neighbours raised late", fontsize=6.6)
    a.text(36, -1.45, "copied image", color=style.SURFACE, fontsize=5.8); a.text(36, .75, "its neighbours", color=style.TARGET, fontsize=5.8)
    a.plot([], [], ls="-", color=style.INK, lw=.8, label="$N=9$"); a.plot([], [], ls="--", color=style.INK, lw=.8, label="$N=7$")
    a.legend(handles=a.get_lines()[-2:], fontsize=5.2, loc="center right", handlelength=1.8)
    d = json.load(open(f"{R}/s829_s829_27b_natural.json")); look = d["look"]
    for N, k, ls in ((9, 5, "-"), (7, 4, "--")):
        rows = [r for r in d["rows"] if r["N"] == N and r["k"] == k]
        for run, col, lab in (("END", style.SURFACE, "from the end"), ("START", style.GREYS[0], "from the start")):
            ms = [np.mean([np.mean([r["mass"][run][str(L)][i] for i in (2, 5, 9)]) for r in rows]) for L in look]
            b.plot(look, ms, ls=ls, color=col, marker="o", ms=2.5, lw=1)
    b.set_xlabel("layer"); b.set_ylabel("attention on image tokens"); b.set_title("(b) the end frame re-reads images late", fontsize=6.6); b.set_ylim(.2, .85)
    b.text(49.5, .64, "from the end", color=style.SURFACE, fontsize=5.8); b.text(44, .25, "from the start", color=style.MUTED, fontsize=5.8)
    d = json.load(open(f"{R}/s827_s827_27b_natural.json")); rows = [r for r in d["rows"] if r["N"] == 9]
    z = lambda r, run: np.mean([np.array(r["qk"][run]["51"][2]), np.array(r["qk"][run]["51"][5])], 0)
    D = np.array([z(r, "END") - z(r, "START") for r in rows]); D = D - D.mean(1, keepdims=True)
    mu, se = D.mean(0), D.std(0, ddof=1) / np.sqrt(len(D))
    xs = np.arange(1, 10)
    c.bar(xs, mu, .7, yerr=1.96 * se, error_kw=dict(lw=.5, capsize=1), color=[style.SURFACE if i == 5 else (style.TARGET if i in (4, 6) else style.GREYS[1]) for i in xs])
    c.axhline(0, color=style.INK, lw=.5); c.set_xticks(xs); c.set_xlabel("image (target = copy = 5)")
    c.set_ylabel("end $-$ start, layer 51"); c.set_title("(c) a displaced address, not a weakened one", fontsize=6.6)
    fig.tight_layout(w_pad=.8)
    style.save(fig, f"{OUT}/fig_stages")


def fig_predict():
    fig, ax = plt.subplots(figsize=(style.ONE, 2.0))
    ax.fill_between([0, 1], [-.15, .85], [.15, 1.15], color=style.RULE, alpha=.6, lw=0)
    ax.plot([0, 1], [0, 1], color=style.MUTED, lw=.6)
    pf = json.load(open(f"{O}/s828_predictions_s828_27b.json")); d2 = json.load(open(f"{R}/s828_s828_27b_twoway.json"))
    xs, ys = [], []
    for cname, res in pf["cases"].items():
        N = int(cname.split("k")[0][1:]); k = int(cname.split("k")[1].split("j")[0]); j = int(cname.split("j")[1])
        rows = [r for r in d2["rows"] if r["N"] == N]
        for arm, v in res["arms"].items():
            for x in (j - 1, j, j + 1):
                xs.append(v["shares"][str(x)]); ys.append(float(np.mean([r["land"][arm] == x for r in rows])))
    ax.scatter(xs, ys, s=9, color=style.ACCENT, alpha=.8, lw=0, label="two-route law, new scenes (S828)")
    A = json.load(open(f"{R}/s832_analysis.json")); xs2, ys2 = [], []
    for cname, c in A["cases"].items():
        for p, x in c["probes"].items():
            xs2 += x["pred"]; ys2 += x["obs"]
    ax.scatter(xs2, ys2, s=9, color=style.TARGET, alpha=.75, lw=0, marker="D", label="two-component law, new cells (S832)")
    ax.set_xlim(-.02, 1); ax.set_ylim(-.02, 1); ax.set_xlabel("predicted share (fixed before the run)"); ax.set_ylabel("observed share")
    ax.legend(fontsize=5.6, loc="upper left", handletextpad=.2)
    ax.text(.97, .05, "band: $\\pm$0.15", ha="right", fontsize=5.6, color=style.MUTED)
    fig.tight_layout()
    style.save(fig, f"{OUT}/fig_predict")


def fig_repair():
    A = json.load(open(f"{R}/s833_analysis.json"))
    cells = [("N7k4j4", "identity, $N=7$"), ("N9k5j5", "identity, $N=9$"), ("N7k5j3", "5th of 7 (target 3)"), ("N9k7j3", "7th of 9 (target 3)")]
    runs = [("END", "no write", style.SURFACE), ("O_ORD", "ordinal only", style.GREYS[0]), ("O_FR", "frame only", style.ACCENT), ("O_ORD_FR", "both", style.TARGET)]
    fig, (a, b) = plt.subplots(2, 1, figsize=(style.ONE, 2.45), sharex=True, gridspec_kw=dict(height_ratios=[1.25, 1]))
    x = np.arange(len(cells)); w = .2
    for i, (run, lab, col) in enumerate(runs):
        acc, wv = [], []
        for c, _ in cells:
            src = run if run in A["cases"][c]["P_j"] else ("O_FR" if run == "O_ORD_FR" else None)      # identity: the ordinal write is zero
            acc.append(A["cases"][c]["P_j"][src] if src else np.nan); wv.append(A["cases"][c]["w"][src][0] if src else np.nan)
        a.bar(x + (i - 1.5) * w, 100 * np.array(acc, float), w, color=col, label=lab)
        b.bar(x + (i - 1.5) * w, np.array(wv, float), w, color=col)
    a.set_ylabel("on the target (%)"); a.set_ylim(0, 108)
    a.legend(fontsize=5.4, ncol=4, loc="upper center", bbox_to_anchor=(.5, 1.25), handlelength=1, columnspacing=.7)
    b.axhline(1, color=style.SURFACE, lw=.5, ls=":"); b.axhline(0, color=style.INK, lw=.5)
    b.text(3.45, 1.08, "end level", color=style.SURFACE, fontsize=5.4, ha="right")
    b.set_ylabel("late re-reading"); b.set_ylim(-.35, 3.2)
    b.set_xticks(x); b.set_xticklabels([n for _, n in cells], fontsize=5.6)
    fig.tight_layout(h_pad=.3)
    style.save(fig, f"{OUT}/fig_repair")


def fig_scale():
    """Same scenes across models (S801, seed 80100, 40 scenes per cell, N = 6-8, k in {1..5, N-1, N}; natural runs, two-form readout): for
    non-identity cells, the share of END(k) answers on the target N + 1 - k and on the copy k; for the identity cell (N = 7, k = 4), accuracy.
    Raw rows: results/r3/s801_s801_{tag}_grounding.json (models that have run)."""
    order = [("4b", "VL-4B"), ("8b", "VL-8B"), ("35_4b", "3.5-4B"), ("35_9b", "3.5-9B"), ("27b", "3.5-27B")]
    rows = []
    for tag, nm in order:
        f = f"/vlm/results/r3/s801_s801_{tag}_grounding.json"
        if not os.path.exists(f):
            continue
        R = json.load(open(f))["rows"]
        non = [r for r in R if r["k"] != r["j"]]; idn = [r for r in R if r["k"] == r["j"]]
        rows.append(dict(model=nm, n_non=len(non), n_id=len(idn),
                         target=float(np.mean([r["END"]["land"] == r["j"] for r in non])),
                         copy=float(np.mean([r["END"]["land"] == r["k"] for r in non])),
                         identity=float(np.mean([r["END"]["land"] == r["j"] for r in idn])) if idn else float("nan"),
                         start=float(np.mean([r["START"]["land"] == r["j"] for r in R]))))
    fig, ax = plt.subplots(figsize=(style.ONE, 1.75))
    x = np.arange(len(rows))
    hy = [i for i, r in enumerate(rows) if r["model"].startswith("3.5")]
    spec = [("target", style.TARGET, "o", "-", "end-relative: target"), ("copy", style.SURFACE, "o", "-", "end-relative: copy $k$"),
            ("identity", style.INK, "D", (0, (2, 1.2)), "identity reference: target")]
    for key, col, mk, ls, lab in spec:
        v = [100 * r[key] for r in rows]
        ax.plot([x[i] for i in hy], [v[i] for i in hy], color=col, lw=.9, ls=ls, zorder=2)
        ax.scatter(x, v, s=14 if mk == "o" else 12, color=col if mk == "o" else "white", edgecolor=col, marker=mk, zorder=3, lw=.8)
        ax.text(x[-1] + .14, v[-1], lab, fontsize=5.5, color=col, va="center", ha="left")
    ax.set_xticks(x); ax.set_xticklabels([r["model"].replace("VL-", "").replace("3.5-", "") for r in rows], fontsize=5.8)
    ax.set_ylim(0, 100); ax.set_ylabel("share of answers (%)"); ax.set_xlim(-.4, len(rows) - .4 + 2.1)
    if hy and hy[0] > 0:
        ax.axvline(hy[0] - .5, color=style.RULE, lw=.6)
        tr = ax.get_xaxis_transform()
        ax.text((hy[0] - 1) / 2, -.2, "dense (Qwen3-VL)", fontsize=5.4, color=style.MUTED, ha="center", va="top", transform=tr)
        ax.text((hy[0] + hy[-1]) / 2, -.2, "hybrid (Qwen3.5)", fontsize=5.4, color=style.MUTED, ha="center", va="top", transform=tr)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    style.save(fig, f"{OUT}/fig_scale")
    json.dump(rows, open(f"{OUT}/fig_scale_values.json", "w"), indent=1)

if __name__ == "__main__":
    for f in (fig_hero, fig_grid, fig_phenomenon, fig_stages, fig_predict, fig_repair, fig_scale):
        try:
            f(); print(f.__name__, "ok")
        except Exception as e:
            import traceback; traceback.print_exc(); print(f.__name__, "FAILED", e)
