"""Explanatory figure (not for the paper as is): matcher-head attention share by image vs the answer distribution, START vs END.
Data: saved block scores of S750 (END k), S757 (START(3), END(5)), S758 (START(2), START(5), END(2), END(5)); S705a matcher head-rows, L35-47, tail1 / answer.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/vlm/results/figures/explain"
os.makedirs(OUT, exist_ok=True)
D = json.load(open("/vlm/exp/figs/start_end_compare_data.json"))
panels = [("S757 START(3)", "START(3): third from the first", 3, 3), ("S758 START(5)", "START(5): fifth from the first", 5, 5),
          ("S758 START(2)", "START(2): second from the first", 2, 2),
          ("S757 END(5)", "END(5): fifth from the last", 3, 5), ("S750 END(4)", "END(4): fourth from the last", 4, 4),
          ("S750 END(3)", "END(3): third from the last", 5, 3), ("S758 END(2)", "END(2): second from the last", 6, 2)]
fig, axes = plt.subplots(2, 4, figsize=(13.5, 5.6), sharey=True)
axes = axes.ravel(); axes[3].axis("off")
x = np.arange(1, 8)
for (key, title, tgt, lit), ai in zip(panels, [0, 1, 2, 4, 5, 6, 7]):
    ax = axes[ai]; d = D[key]
    cols = ["#4caf50" if i == tgt else "#e0a040" if i == lit else "#b8bcc8" for i in x]
    ax.bar(x, d["share"], color=cols, width=.72, label="attention share (index-matching heads)")
    ax.plot(x, d["ans"], "o", color="#202020", ms=6, label="answer share")
    ax.set_xticks(x); ax.set_ylim(0, 1.05)
    ax.set_title(f"{title}\nattention on target {d['share'][tgt-1]:.2f} · correct {d['ans'][tgt-1]:.2f}", fontsize=9.5)
    ax.set_xlabel("image")
axes[0].set_ylabel("share"); axes[4].set_ylabel("share")
h, l = axes[0].get_legend_handles_labels()
axes[3].legend(h + [plt.Rectangle((0, 0), 1, 1, color="#4caf50"), plt.Rectangle((0, 0), 1, 1, color="#e0a040")],
               l + ["target image", "literal image (k-th from the first)"], loc="center", fontsize=9, frameon=False)
fig.suptitle("Qwen3.5-27B, seven images: where the index-matching heads look (bars) and what the model answers (dots)", fontsize=11)
fig.tight_layout()
fig.savefig(f"{OUT}/start_end_compare.png", dpi=160)
print("ok")
