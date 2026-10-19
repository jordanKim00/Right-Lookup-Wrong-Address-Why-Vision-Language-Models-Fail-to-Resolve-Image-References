"""C-circuit figures for the revised main text (drafts; data = saved S703-S712 records, S705a QK arrays, S720 head sets).
Style: exp/figs/style.py (knowledge/FIGURE_CONVENTIONS.md).  Output: /vlm/results/figures/c_circuit/*.pdf|png
    fig_c1_qk_tables   START / END query x image-block compatibility of the index-matching heads, Qwen3.5-4B / 9B / 27B (S705a)
    fig_c2_anchor      27B matcher heads, END k = 2: the score profile follows the last image visible to full attention (S705a)
    fig_c3_steps       27B: END matching and END accuracy fall with steps from the visible end; START stays sharp (S705a, S711, S712)
    fig_c5_rereflect   27B: ordinal write alone is re-reflected, ordinal + frame write lands on the target, a wrong address is followed (S708)
Usage: python figs/make_c_figs.py
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, "/vlm/exp/figs")
import style

R = "/vlm/results/r3"
OUT = "/vlm/results/figures/c_circuit"
N = 7


def lse(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True); m = np.where(np.isfinite(m), m, 0.)
    return np.squeeze(m, axis) + np.log(np.sum(np.exp(x - m), axis=axis))


def qk_tables(tag, heads, ri=2, scenes=None):
    """mean block share over heads and scenes: START[t, j], END[k, j] (answer row)."""
    D = f"{R}/s705a_{tag}_qkanatomy"
    meta = json.load(open(D + ".json")); z = np.load(D + ".npz"); M = meta["meta"]
    scenes = scenes if scenes is not None else range(len(M) // 2, len(M))                # held-out half
    S = np.zeros((N, N)); E = np.zeros((N, N)); cS = np.zeros(N); cE = np.zeros(N)
    for si in scenes:
        own = np.array(M[si]["owner"])
        for fr, mat, cnt in (("S", S, cS), ("E", E, cE)):
            for q in range(1, N + 1):
                for (L, h) in heads:
                    key = f"{si}|{fr}{q}m0|{L}|limg"
                    if key not in z.files:
                        continue
                    sc = np.stack([lse(z[key][ri][h].astype(np.float64)[own == b]) for b in range(N)])
                    mat[q - 1] += np.exp(sc - lse(sc)); cnt[q - 1] += 1
    return S / cS[:, None], E / cE[:, None]


def fig_c1():
    """compact: START (27B; the diagonal is the same in every model) + END for each model, one row."""
    cols = [("qa_4b", f"{R}/s720_4b_heads.json", "Qwen3.5-4B"), ("qa_9b", f"{R}/s720_9b_heads.json", "Qwen3.5-9B"),
            ("qa_27b", f"{R}/s705a_qa_27b_matchers.json", "Qwen3.5-27B")]
    mats = {}
    for tag, hf, name in cols:
        heads = [tuple(x) for x in json.load(open(hf))["matchers"]]
        mats[name] = qk_tables(tag, heads)
    panels = [("start-relative, 27B", mats["Qwen3.5-27B"][0], 0)] + [(f"end-relative, {n.replace('Qwen3.5-', '')}", mats[n][1], 1) for _, _, n in cols]
    fig, ax = plt.subplots(1, 4, figsize=(style.TWO, 1.78), constrained_layout=True)
    for a, (title, mat, r) in zip(ax, panels):
        im = a.imshow(mat, cmap="Purples", vmin=0, vmax=.6, origin="upper")
        for q in range(N):
            j = q if r == 0 else N - 1 - q
            a.add_patch(Rectangle((j - .5, q - .5), 1, 1, fill=False, lw=.9, ec=style.TARGET))
            if r == 1 and j != q:
                a.add_patch(Rectangle((q - .5, q - .5), 1, 1, fill=False, lw=.7, ec=style.SURFACE, ls=(0, (2, 1))))
        a.set_xticks(range(N)); a.set_xticklabels(range(1, N + 1), fontsize=5.5); a.set_yticks(range(N)); a.set_yticklabels(range(1, N + 1), fontsize=5.5)
        a.set_title(title, fontsize=6.6, pad=2)
        a.set_xlabel("image block", fontsize=6, labelpad=1)
        a.tick_params(length=1.5, pad=1)
    ax[0].set_ylabel("ordinal in question", fontsize=6, labelpad=1)
    cb = fig.colorbar(im, ax=ax, shrink=.85, pad=.01, aspect=18)
    cb.set_label("share of image attention", fontsize=5.8); cb.ax.tick_params(labelsize=5.2, length=1.5)
    style.save(fig, f"{OUT}/fig_c1_qk_tables"); plt.close(fig)


def fig_c2():
    tag = "qa_27b"; heads = [tuple(x) for x in json.load(open(f"{R}/s705a_qa_27b_matchers.json"))["matchers"]]
    D = f"{R}/s705a_{tag}_qkanatomy"; meta = json.load(open(D + ".json")); z = np.load(D + ".npz"); M = meta["meta"]
    fig, a = plt.subplots(figsize=(style.ONE, 2.0), constrained_layout=True)
    cols = [style.TARGET, style.ACCENT, style.SURFACE]
    for m in (0, 1, 2):
        prof = []
        for si in range(len(M) // 2, len(M)):
            own = np.array(M[si]["owner"])
            for (L, h) in heads:
                key = f"{si}|E2m{m}|{L}|limg"
                if key not in z.files:
                    continue
                sc = np.stack([lse(z[key][2][h].astype(np.float64)[own == b]) for b in range(N - m)])
                p = np.zeros(N); p[:N - m] = np.exp(sc - lse(sc)); prof.append(p)
        prof = np.mean(prof, 0)
        a.plot(range(1, N + 1), prof, "-o", ms=3, lw=1.2, color=cols[m], label=f"last {m} hidden" if m else "unedited")
        a.axvline(N - m - 1, color=cols[m], lw=.6, ls=":")
    a.set_xlabel("image block $j$"); a.set_ylabel("matcher share (END, $k=2$)"); a.set_xticks(range(1, N + 1))
    a.legend(loc="upper left")
    style.save(fig, f"{OUT}/fig_c2_anchor"); plt.close(fig)


def fig_c3():
    heads = [tuple(x) for x in json.load(open(f"{R}/s705a_qa_27b_matchers.json"))["matchers"]]
    S, E = qk_tables("qa_27b", heads)
    steps = np.arange(0, N)                                                               # k - 1
    end_t = [E[k - 1, N - k] for k in range(1, N + 1)]; start_t = [S[N - k, N - k] for k in range(1, N + 1)]
    acc = {k: [] for k in range(1, N + 1)}
    for f in (f"{R}/s711_v2_27b_eval.json",):
        if os.path.exists(f):
            for r in json.load(open(f))["rows"]:
                if r["N"] == N:
                    acc[r["k"]].append(r["land"]["NATIVE"] == r["j"])
    fig, a = plt.subplots(figsize=(style.ONE, 2.0), constrained_layout=True)
    a.plot(steps, start_t, "-o", ms=3, color=style.MUTED, lw=1.0, label="START, same image (QK)")
    a.plot(steps, end_t, "-o", ms=3, color=style.TARGET, lw=1.2, label="END target (QK)")
    # S782 registry: the END accuracy line came from the space-token readout (S711) and is not drawn
    a.set_xlabel("steps from the visible end ($k-1$)"); a.set_ylabel("share of image attention"); a.set_ylim(0, 1.02)
    a.set_xticks(steps); a.legend(loc="upper right", ncol=2)
    style.save(fig, f"{OUT}/fig_c3_steps"); plt.close(fig)


def fig_c5():
    d = json.load(open(f"{R}/s708_rb_27b_rebind.json"))
    arms = [("BASE", "no write"), ("ORD@3", "ordinal only"), ("FR@3", "frame only"), ("ORD_FR@3", "ordinal + frame"),
            ("WRONG@3", "wrong address\n+ frame"), ("RAND_FR@3", "random + frame")]
    share = {a_: dict(target=[], literal=[], wrong=[]) for a_, _ in arms}
    for row in d["rows"]:
        for k, c in row["k"].items():
            k = int(k)
            for a_, _ in arms:
                x = c["land"][a_]
                share[a_]["target"].append(x == c["r"]); share[a_]["literal"].append(x == k); share[a_]["wrong"].append(x == c["w"])
    fig, a = plt.subplots(figsize=(style.ONE, 2.1), constrained_layout=True)
    xs = np.arange(len(arms)); w = .26
    for i, (key, col, lab) in enumerate((("target", style.TARGET, "target $j$"), ("literal", style.SURFACE, "literal $k$"),
                                         ("wrong", style.ACCENT, "written wrong $w$"))):
        a.bar(xs + (i - 1) * w, [np.mean(share[a_][key]) for a_, _ in arms], w, color=col, label=lab)
    a.set_xticks(xs); a.set_xticklabels([l for _, l in arms], rotation=35, ha="right"); a.set_ylabel("share of answers"); a.set_ylim(0, 1.05)
    a.legend(loc="upper left", ncol=1)
    style.save(fig, f"{OUT}/fig_c5_rereflect"); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    style.apply()
    for f in (fig_c1, fig_c2, fig_c3, fig_c5):
        try:
            f(); print("ok", f.__name__, flush=True)
        except Exception as e:
            print("FAIL", f.__name__, repr(e), flush=True)


def fig_c6_mediation():
    d = json.load(open(f"{R}/s710_mq_27b_mediation.json")); rows = d["rows"]
    resc = [r for r in rows if r["acted"] and r["land"]["R3"] == r["j"] and r["land"]["NATIVE"] != r["j"]]
    P = lambda arm: float(np.mean([r["land"][arm] == r["j"] for r in resc]))
    fig, ax = plt.subplots(1, 2, figsize=(style.ONE, 1.9), sharey=True, constrained_layout=True)
    for a, (title, arms) in zip(ax, (("repaired run, part reverted", (("R3", "none"), ("R3_QBACK", "matcher\nquery"), ("R3_GBACK", "matcher\ngate"), ("R3_RBACK", "random\nheads"))),
                                     ("original run, part supplied", (("NATIVE", "none"), ("END_QR3", "matcher\nquery"), ("END_GR3", "matcher\ngate"), ("END_RR3", "random\nheads"))))):
        vals = [P(k) for k, _ in arms]
        cols = [style.MUTED, style.TARGET, style.GREYS[0], style.GREYS[1]]
        a.bar(range(4), vals, .6, color=cols)
        for i, v_ in enumerate(vals):
            a.text(i, v_ + .02, f"{100 * v_:.0f}", ha="center", va="bottom", fontsize=6)
        a.set_xticks(range(4)); a.set_xticklabels([l for _, l in arms], fontsize=5.8); a.set_title(title, fontsize=7)
        a.set_ylim(0, 1.12)
    ax[0].set_ylabel(f"correct on repaired items (n = {len(resc)})")
    style.save(fig, f"{OUT}/fig_c6_mediation"); plt.close(fig)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "c6":
    os.makedirs(OUT, exist_ok=True); style.apply(); fig_c6_mediation(); print("ok fig_c6_mediation")
