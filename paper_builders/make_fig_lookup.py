"""fig_lookup (main Figure 3): a position-based lookup that can be controlled from outside.
(a) Qwen3.5-27B, start-relative questions: block shares of the index-matching heads at the answer position (S705a captures, held-out scenes,
    seven images) -- the lookup aims at the named block.
(b) Qwen3-VL-4B: which image the model names under six writes that share one frozen plan (semantic audit v1b, 86 scenes x 3 views) -- a written
    address, right or wrong, selects the image now at that address.
fig_selnn (main Figure 4, S790c re-analysis of the same captures): for "the k-th image from the end", the share of (scene, head) pairs whose block
    distribution is nearest (Jensen-Shannon) to that of "the t-th image from the first"; teal: the composed address 8 - k, orange: the copied k.
Style: exp/figs/style.py.  Output: /vlm/results/figures/c_circuit/fig_lookup.{pdf,png}, fig_selnn.{pdf,png}"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, "/vlm/exp/figs")
import style

R, OUT, N = "/vlm/results/r3", "/vlm/results/figures/c_circuit", 7
CACHE = f"{OUT}/qk_tables_cache.json"
style.apply()
os.makedirs(OUT, exist_ok=True)


def start_table_27b():
    if os.path.exists(CACHE):
        c = json.load(open(CACHE))
        if "qa_27b" in c:
            return np.array(c["qa_27b"]["S"]), np.array(c["qa_27b"]["E"])
    from make_c_figs import qk_tables
    heads = [tuple(x) for x in json.load(open(f"{R}/s705a_qa_27b_matchers.json"))["matchers"]]
    S, E = qk_tables("qa_27b", heads)
    c = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    c["qa_27b"] = dict(S=S.tolist(), E=E.tolist()); json.dump(c, open(CACHE, "w"))
    return S, E


def fig_lookup():
    S, _ = start_table_27b()
    a = json.load(open("/vlm/research/ces/outputs/r3_semantic_result_audit_v1b/frozen_analysis_science-002.json"))
    M = a["conditional_arm_destination_matrix"]
    val = lambda arm, k: M[arm][k]["complete_cluster_mean"] if isinstance(M[arm][k], dict) else float(M[arm][k])
    arms = [("00", "no write"), ("RJF", "random vector"), ("O", "address $j$"), ("OF", "$j$ + frame"), ("W", "wrong address $w$"), ("WF", "$w$ + frame")]
    cats = [("k", "image $k$ (literal)", style.SURFACE), ("j", "image $j$ (target)", style.TARGET), ("w", "image $w$ (written)", style.ACCENT),
            ("other", "other", "#c9cec9")]
    fig = plt.figure(figsize=(style.ONE, 1.62))
    ax1 = fig.add_axes([.075, .2, .3, .62]); ax2 = fig.add_axes([.625, .2, .36, .62])
    im = ax1.imshow(S, cmap="Purples", vmin=0, vmax=.7, origin="upper")
    for q in range(N):
        ax1.add_patch(Rectangle((q - .5, q - .5), 1, 1, fill=False, lw=.8, ec=style.TARGET))
    ax1.set_xticks(range(N)); ax1.set_xticklabels(range(1, N + 1), fontsize=5.4); ax1.set_yticks(range(N)); ax1.set_yticklabels(range(1, N + 1), fontsize=5.4)
    ax1.set_xlabel("image block", fontsize=5.9, labelpad=1); ax1.set_ylabel("$t$ in “the $t$-th image from the first”", fontsize=5.6, labelpad=1)
    ax1.tick_params(length=1.4, pad=1)
    ax1.set_title("(a) the lookup aims at\nthe named block (27B)", fontsize=6.2, pad=2.5, linespacing=1.0)
    cax = fig.add_axes([.39, .2, .012, .62]); cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=4.8, length=1.2, pad=1)
    cb.set_ticks([0, .35, .7])
    y = np.arange(len(arms))[::-1]
    for (arm, lab), yy in zip(arms, y):
        left = 0
        for c, _, col in cats:
            v = val(arm, c) + (val(arm, "wbar") + val(arm, "invalid") if c == "other" else 0)
            ax2.barh(yy, v, left=left, color=col, height=.66, edgecolor="white", lw=.5)
            if v > .15:
                ax2.text(left + v / 2, yy, f"{v * 100:.0f}", ha="center", va="center", fontsize=5.2, color="white")
            left += v
    ax2.set_yticks(y); ax2.set_yticklabels([l for _, l in arms], fontsize=5.5)
    ax2.set_xlim(0, 1); ax2.set_xticks([0, .5, 1]); ax2.set_xticklabels(["0", "50", "100%"], fontsize=5.4)
    ax2.set_xlabel("image the model names (VL-4B)", fontsize=5.9, labelpad=1)
    ax2.tick_params(axis="y", length=0, pad=1.5)
    ax2.set_title("(b) a written address\nselects its image", fontsize=6.2, pad=2.5, linespacing=1.0)
    hs = [Rectangle((0, 0), 1, 1, color=col, label=lab) for _, lab, col in cats]
    fig.legend(handles=hs, loc="lower center", bbox_to_anchor=(.6, -.06), ncol=4, fontsize=5.0, handlelength=.9, columnspacing=.6, handletextpad=.3)
    style.save(fig, f"{OUT}/fig_lookup"); plt.close(fig)


def fig_selnn():
    D = json.load(open(f"{R}/s790c_selnn_full.json"))
    MD = [("qa_4b", "Qwen3.5-4B"), ("qa_9b", "Qwen3.5-9B"), ("qa_27b", "Qwen3.5-27B")]
    fig, axes = plt.subplots(1, 3, figsize=(style.ONE, 1.32), gridspec_kw=dict(wspace=.1, left=.09, right=.88, bottom=.17, top=.82))
    for i, (ax, (t, nm)) in enumerate(zip(axes, MD)):
        Mx = np.array([D[t][str(k)]["nearest_t"] for k in range(1, N + 1)])
        im = ax.imshow(Mx, cmap="Purples", vmin=0, vmax=1, origin="upper")
        for k in range(1, N + 1):
            ax.add_patch(Rectangle((N - k - .5, k - 1 - .5), 1, 1, fill=False, lw=.8, ec=style.TARGET))
            if k != N + 1 - k:
                ax.add_patch(Rectangle((k - 1 - .5, k - 1 - .5), 1, 1, fill=False, lw=.7, ec=style.SURFACE, ls=(0, (2, 1))))
        ax.set_xticks(range(N)); ax.set_xticklabels(range(1, N + 1), fontsize=5.0); ax.set_yticks(range(N))
        ax.set_yticklabels(range(1, N + 1) if i == 0 else [], fontsize=5.0)
        ax.tick_params(length=1.2, pad=1)
        ax.set_title(nm, fontsize=6.0, pad=2)
    axes[0].set_ylabel("$k$ in \u201cthe $k$-th image from the end\u201d", fontsize=5.3, labelpad=1)
    fig.text(.485, .03, "$t$ of the start-relative question whose block distribution is nearest", ha="center", fontsize=5.4)
    hs = [Rectangle((0, 0), 1, 1, fill=False, ec=style.TARGET, lw=.9, label="target, $t=8-k$"),
          Rectangle((0, 0), 1, 1, fill=False, ec=style.SURFACE, lw=.8, ls=(0, (2, 1)), label="literal, $t=k$")]
    fig.legend(handles=hs, loc="upper center", bbox_to_anchor=(.485, 1.06), ncol=2, fontsize=5.3, handlelength=1.0, columnspacing=1.2, handletextpad=.4)
    cax = fig.add_axes([.9, .2, .014, .64]); cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=4.8, length=1.2, pad=1)
    cb.set_ticks([0, .5, 1]); cb.set_label("share of pairs", fontsize=5.0, labelpad=1)
    style.save(fig, f"{OUT}/fig_selnn"); plt.close(fig)


if __name__ == "__main__":
    fig_selnn(); print("ok fig_selnn")
    fig_lookup(); print("ok fig_lookup")
