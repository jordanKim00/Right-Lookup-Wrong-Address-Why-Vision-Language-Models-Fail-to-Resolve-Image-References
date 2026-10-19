"""fig_qident (S790, saved S705a captures, no model call): which start-relative query the end-relative query is.
For each Qwen3.5 model, the block-share-weighted mean block E[b] of the index-matching heads' answer-row query for "the k-th image from
the end" (solid), against the same quantity for the start-relative question with the same ordinal k (copy) and with the same image
N+1-k (composition).  N = 7, held-out scenes.  Style: exp/figs/style.py.  Output: /vlm/results/figures/c_circuit/fig_qident.{pdf,png}"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/vlm/exp/figs")
import style

R, OUT, N = "/vlm/results/r3", "/vlm/results/figures/c_circuit", 7
D = json.load(open(f"{R}/s790_qident.json"))
style.apply()
fig, ax = plt.subplots(figsize=(style.ONE, 1.55))
ks = [1, 2, 3, 5, 6, 7]                                                          # k = 4: the two references coincide
ax.axvspan(.5, 2.5, color=style.RULE, alpha=.45, lw=0)
ax.axhline(50, color=style.MUTED, lw=.6, ls=":")
for tag, name, col, mk in (("qa_4b", "Qwen3.5-4B", "#8c6bb1", "s"), ("qa_9b", "Qwen3.5-9B", style.ACCENT, "^"), ("qa_27b", "Qwen3.5-27B", style.INK, "o")):
    e = D[tag]["end"]
    ax.plot(ks[:2], [100 * e[str(k)]["closer_to_copy"] for k in ks[:2]], "-", marker=mk, color=col, lw=1.1, ms=3, label=name)
    ax.plot(ks[2:], [100 * e[str(k)]["closer_to_copy"] for k in ks[2:]], "-", marker=mk, color=col, lw=1.1, ms=3)
    ax.plot([2, 3], [100 * e["2"]["closer_to_copy"], 100 * e["3"]["closer_to_copy"]], "-", color=col, lw=1.1)
ax.set_xticks(range(1, 8)); ax.set_xticklabels(["1", "2", "3", "", "5", "6", "7"]); ax.set_xlim(.5, 7.5)
ax.set_ylim(-3, 103); ax.set_yticks([0, 25, 50, 75, 100])
ax.set_xlabel("$k$ in \u201cthe $k$-th image from the end\u201d", labelpad=1)
ax.set_ylabel("closer to the copied query (%)", labelpad=1)
ax.text(.55, 3, "composed", fontsize=5.6, color=style.MUTED, va="bottom")
ax.text(2.6, 99, "copied", fontsize=5.6, color=style.MUTED, ha="left", va="top")
ax.legend(loc="lower right", bbox_to_anchor=(1.0, .0), fontsize=5.8, handlelength=1.8)
os.makedirs(OUT, exist_ok=True); style.save(fig, f"{OUT}/fig_qident"); plt.close(fig)
print("ok fig_qident")
