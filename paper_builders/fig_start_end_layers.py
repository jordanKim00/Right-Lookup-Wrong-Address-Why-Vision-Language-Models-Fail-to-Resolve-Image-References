"""Explanatory figure from S764 (same scenes, same pictures, same menus): mean attention share (all full-attention heads, rows tail1 / answer)
on the target image and on the literal image, by layer, for START(t) and END(N+1-t) naming the same image.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("/vlm/results/r3/s764_analysis.json"))["desc"]
OUT = "/vlm/results/figures/explain"; os.makedirs(OUT, exist_ok=True)
panels = [("7", "S6", "E2", "image 6 of 7:  START(6) vs END(2)"), ("7", "S4", "E4", "image 4 of 7:  START(4) vs END(4)"),
          ("7", "S3", "E5", "image 3 of 7:  START(3) vs END(5)"), ("8", "S4", "E5", "image 4 of 8:  START(4) vs END(5)")]
fig, axes = plt.subplots(1, 4, figsize=(15, 3.9), sharey=True)
for ax, (N, s, e, title) in zip(axes, panels):
    Ls = sorted(d[N][s]["layers"], key=int); x = [int(l) for l in Ls]
    ax.plot(x, [d[N][s]["layers"][l]["target"] for l in Ls], "-o", color="#2e7d32", ms=4, label="START: target")
    ax.plot(x, [d[N][e]["layers"][l]["target"] for l in Ls], "-o", color="#1565c0", ms=4, label="END: target")
    ax.plot(x, [d[N][e]["layers"][l]["literal"] for l in Ls], "--s", color="#e08a20", ms=3.5, label="END: literal (k-th from the first)")
    ax.axvspan(35, 47, color="#000000", alpha=.06)
    ax.set_title(f"{title}\nanswer correct: START {d[N][s]['acc']:.2f} · END {d[N][e]['acc']:.2f}", fontsize=9.5)
    ax.set_xlabel("layer"); ax.set_xticks([27, 35, 43, 51, 59])
axes[0].set_ylabel("attention share (all heads)")
axes[0].legend(fontsize=8, loc="upper left", frameon=False)
fig.suptitle("Qwen3.5-27B, same scenes: START finds the image by L39; END finds it late (L51) for one step and never sharply for far steps "
             "(shaded: index-matching band L35-47)", fontsize=10.5)
fig.tight_layout()
fig.savefig(f"{OUT}/start_end_layers.png", dpi=160)
print("ok")
