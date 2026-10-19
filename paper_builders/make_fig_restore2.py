"""fig_restore2 (main Figure 6, review revision): the rebound reference reads the picture at its address, through the queries.
(a) An illustrative procedural scene (r3_proc pictures, not a measured scene): the picture at the written address, image 5 of seven for "the third
    image from the end", is replaced by a new object; the rebound answer names the new object.
(b) S780 re-measured with the bare readout (S783a; Qwen3.5-27B, 80 new scenes at N = 7 dark / 8 light, Wilson 95% intervals): answers that change to
    a new picture placed at the target -- rebound, and unedited one step (END(2)) and four steps (END(5)) from the end.
(c) S740 (MMIU development photographs, 150 items per model repaired by R3): every full-attention query put back to the unrepaired one (rebound run,
    items still correct) and given to the unrepaired run (items repaired).
Style: exp/figs/style.py.  Output: /vlm/results/figures/c_circuit/fig_restore2.{pdf,png}"""
import json, os, random, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

sys.path.insert(0, "/vlm/exp/figs"); sys.path.insert(0, "/vlm/exp")
import style
from make_fig_frame_bind import wilson, pick
from r3_proc import proc_image

R, OUT, N = "/vlm/results/r3", "/vlm/results/figures/c_circuit", 7
LIGHT_T, LIGHT_A, LIGHT_S = "#9fd3cb", "#aab5e6", "#efb894"
style.apply()
fig = plt.figure(figsize=(style.TWO, 1.5))
gs = fig.add_gridspec(1, 3, width_ratios=[2.35, 2.0, 1.75], wspace=.32, left=.01, right=.995, bottom=.2, top=.86)
ax_ex = fig.add_subplot(gs[0, 0]); bx = fig.add_subplot(gs[0, 1]); cx = fig.add_subplot(gs[0, 2])

# (a) illustrative scene
pids = sorted({int(k.split(":")[1]) for l in open("/vlm/assets/r3cal_a3/train.jsonl") for k in json.loads(l)["images"]})
rng = random.Random(3); pool = {}
for pid in pids[:1500]:
    im, lab = proc_image(pid)[:2]
    if lab not in pool:
        pool[lab] = im
labs = rng.sample(sorted(pool), N + 1); scene, new = labs[:N], labs[N]
ax_ex.set_xlim(0, 1); ax_ex.set_ylim(0, 1); ax_ex.axis("off")
w, g = .1, .012; x0 = .02
for row, (y, lab_row) in enumerate(((.55, "scene"), (.12, "image 5 replaced"))):
    for i in range(N):
        x = x0 + i * (w + g)
        lab = new if (row == 1 and i == 4) else scene[i]
        ax_ex.imshow(np.asarray(pool[lab].resize((96, 96))), extent=(x, x + w, y, y + .3), aspect="auto", zorder=1)
        ec = style.TARGET if i == 4 else "#c9cec9"; lw = 1.1 if i == 4 else .4
        ax_ex.add_patch(Rectangle((x, y), w, .3, fill=False, ec=ec, lw=lw, zorder=2))
        if row == 0:
            ax_ex.text(x + w / 2, y + .32, str(i + 1), ha="center", va="bottom", fontsize=5.2, color=style.MUTED)
    ax_ex.text(x0, y - .035, lab_row, ha="left", va="top", fontsize=5.2, color=style.MUTED)
xa = x0 + 4 * (w + g) + w / 2
ax_ex.text(.84, .7, f"R$^3$ answer:\n{scene[4]}", ha="left", va="center", fontsize=5.5, color=style.TARGET)
ax_ex.text(.84, .27, f"R$^3$ answer:\n{new}", ha="left", va="center", fontsize=5.5, color=style.TARGET, fontweight="bold")
ax_ex.set_title("(a) change the picture at the address", fontsize=6.3, pad=9, loc="left")

# (b) content following
B = json.load(open(pick("s780_analysis"))); c7, c8 = (B["s783_27b7"], B["s783_27b8"])
groups = [("R$^3$, written\naddress", "visual_r3", "j", style.TARGET, LIGHT_T), ("unedited,\none step (END(2))", "visual_e2", "j", style.ACCENT, LIGHT_A),
          ("unedited,\nfour steps (END(5))", "visual_end", "j", style.SURFACE, LIGHT_S)]
wd = .36
for i, (lab, key, f, c1, c2) in enumerate(groups):
    for d, c, col, nm in ((-wd / 2, c7, c1, "seven images"), (wd / 2, c8, c2, "eight images")):
        p, n = c[key][f], c[key]["n"]; lo, hi = wilson(p, n)
        bx.bar(i + d, 100 * p, wd, color=col, yerr=[[100 * lo], [100 * hi]], error_kw=dict(lw=.5, capsize=1.2, ecolor=style.INK))
        bx.text(i + d, 100 * (p + hi) + 2, f"{100 * p:.0f}", ha="center", va="bottom", fontsize=5.2)
bx.set_xticks(range(3)); bx.set_xticklabels([g_[0] for g_ in groups], fontsize=5.4); bx.tick_params(axis="x", length=0)
bx.set_ylim(0, 115); bx.set_yticks([0, 25, 50, 75, 100]); bx.set_ylabel("answer follows the new picture (%)", fontsize=5.8, labelpad=1)
bx.set_title("(b) the answer follows the picture", fontsize=6.3, pad=2, loc="left")
bx.text(.99, .95, "dark: seven images\nlight: eight images", transform=bx.transAxes, ha="right", va="top", fontsize=5.0, color=style.MUTED)

# (c) S740 mediation, every full-attention query
M = (("35_4b", "3.5-4B"), ("35_9b", "3.5-9B"), ("27b", "3.5-27B")); V = {}
for tag, _ in M:
    rows = [r for r in json.load(open(f"{R}/s740_nm740_{tag}_natmed.json"))["rows"] if "pick_R3_QBACK_ALL" in r]
    resc = [r for r in rows if r["pick_R3"] == r["gold"] and r["pick_END"] != r["gold"]]
    V[tag] = {k: float(np.mean([r[k] == r["gold"] for r in resc])) for k in ("pick_R3_QBACK_ALL", "pick_END_QALL")}
x = np.arange(3)
for d, key, col, lab in ((-wd / 2, "pick_R3_QBACK_ALL", "#b4bbb6", "rebound run, unrepaired queries"), (wd / 2, "pick_END_QALL", "#4d5650", "unrepaired run, rebound queries")):
    vals = [100 * V[t][key] for t, _ in M]
    cx.bar(x + d, vals, wd, color=col, label=lab)
    for xi, v in zip(x + d, vals):
        cx.text(xi, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=5.2)
cx.set_xticks(x); cx.set_xticklabels([n for _, n in M], fontsize=5.5); cx.tick_params(axis="x", length=0)
cx.set_ylim(0, 115); cx.set_yticks([0, 25, 50, 75, 100]); cx.set_ylabel("repaired items correct (%)", fontsize=5.8, labelpad=1)
cx.set_title("(c) photographs: swap the queries", fontsize=6.3, pad=2, loc="left")
cx.legend(loc="upper left", bbox_to_anchor=(-.02, 1.0), fontsize=5.0, handlelength=1.0, frameon=False, borderaxespad=.2)
os.makedirs(OUT, exist_ok=True); style.save(fig, f"{OUT}/fig_restore2"); plt.close(fig); print("ok fig_restore2", scene, new)
