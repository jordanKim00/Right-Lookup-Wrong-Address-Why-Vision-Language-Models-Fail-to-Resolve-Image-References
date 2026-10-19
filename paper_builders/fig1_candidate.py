"""Figure 1 candidate for the R3 paper (one column): the actual S626 input scene on top, measured answer-position arrival
curves below (exact patching, results/r3/s626_*_patch.json). No schematic boxes; every curve and count is read from saved JSON.
The scene is S626's first 4B scene, rebuilt with the same seed and sampling order as exp/s626_patch.py (tokenizer only, no model).
Usage (container): python /vlm/exp/figs/fig1_candidate.py
"""
import json, os, random, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, "/vlm/exp/figs"); sys.path.insert(0, "/vlm/exp")
import style as S
from r3_proc import proc_image
from transformers import AutoTokenizer

R3 = "/vlm/results/r3"; OUT = "/vlm/research/ces/paper_r3_20261006/evidence/figures"
N, K = 7, 3
S.apply()


def scene0(model="/dev/shm/vlm_models/Qwen3-VL-4B-Instruct", pixels=160):
    """Same as exp/s626_patch.py main(): label pool from single-token shape words, rng 62610, first draw."""
    tok = AutoTokenizer.from_pretrained(model)
    ids1 = lambda s: tok(" " + s, add_special_tokens=False)["input_ids"]
    pids = sorted({int(k.split(":")[1]) for l in open("/vlm/assets/r3cal_a3/train.jsonl") for k in json.loads(l)["images"]})
    pool = {}
    for pid in pids[:4000]:
        lab = proc_image(pid)[1]
        if len(ids1(lab)) == 1:
            pool.setdefault(lab, []).append(pid)
    labels = sorted(l for l, x in pool.items() if len(x) >= 20)
    rng = random.Random(62610)
    labs = rng.sample(labels, N + 2); scene_l = labs[:N]
    imgs = [proc_image(rng.choice(pool[l]))[0].resize((pixels, pixels)) for l in scene_l]
    return imgs, scene_l


def patch(tag):
    return json.load(open(f"{R3}/s626_{tag}_patch.json"))


def main():
    imgs, names = scene0()
    fig = plt.figure(figsize=(S.ONE, 2.45))
    gs = fig.add_gridspec(2, 1, height_ratios=[.62, 1.6], hspace=.55)
    top = gs[0].subgridspec(1, N, wspace=.08)
    for i in range(N):
        ax = fig.add_subplot(top[i]); ax.imshow(imgs[i]); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        c = S.SURFACE if i + 1 == K else S.TARGET if i + 1 == N + 1 - K else None
        if c:
            ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, ec=c, lw=1.6, clip_on=False))
        ax.text(.5, -.1, str(i + 1), transform=ax.transAxes, ha="center", va="top", fontsize=6.5,
                color=c or S.INK, weight="bold" if c else "normal")
        if i == 0:
            ax.text(0, 1.12, "“What is in the third image from the end?”", transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=7, style="italic")
    runs = [("patch_4b", "Qwen3-VL-4B", lambda b: f"img 3: {round(b['surf']*8)}/8"),
            ("patch_27b", "Qwen3.5-27B", lambda b: f"img 5: {round(b['rev']*6)}/6, img 3: {round(b['surf']*6)}/6"),
            ("patch_4b_r3", "Qwen3-VL-4B + R³", lambda b: f"img 5: {round(b['rev']*8)}/8")]
    bot = gs[1].subgridspec(1, 3, wspace=.38)
    axes = [fig.add_subplot(bot[i]) for i in range(3)]
    for ax, (tag, name, beh) in zip(axes, runs):
        d = patch(tag); z = np.array(d["layers"]) / (d["D"] - 1)
        top_ = max(d["full"]["A"], d["full"]["B"], 1.0)
        for m, c, mk, lab in (("B", S.SURFACE, "o", "image 3 (surface)"), ("A", S.TARGET, "D", "image 5 (target)")):
            if abs(d["full"][m]) < 1.0:                                           # whole-input swap has no effect: nothing to localize
                ax.text(.04, .96, "image 3:\nno effect", fontsize=5.2, color=S.MUTED, transform=ax.transAxes, va="top", linespacing=.95)
                continue
            y = np.array(d["maps"][m])[:, -1]                                     # logit effect of patching the answer position
            ax.plot(z, y, color=c, lw=1.2, marker=mk, ms=2.0, label=lab)
        ax.set_title(f"{name}\n{beh(d['behaviour'])}", fontsize=6.3, pad=3)
        ax.set_xlim(.45, 1.0); ax.set_ylim(-.04 * top_, 1.06 * top_); ax.grid(alpha=.3, lw=.4)
        ax.set_xticks([.5, .75, 1.0]); ax.set_xticklabels([".5", ".75", "1"])
    for ax, layer, D, txt in ((axes[0], 34, 36, "$N{+}1{-}k$ at L34\u2020"), (axes[1], 47, 64, "end read (L47)")):
        x = layer / (D - 1)
        ax.axvline(x, color=S.MUTED, lw=.6, ls=(0, (2, 1.5)), zorder=0)
        ax.text(x - .012, .22, txt, fontsize=4.9, color=S.MUTED, ha="right", va="bottom", rotation=90, transform=ax.get_xaxis_transform())
    axes[0].set_ylabel("logit effect at answer", fontsize=6.6)
    axes[1].set_xlabel("relative depth", fontsize=6.8, labelpad=1)
    # identity = frame colour of the images above; direct labels on the first panel only (no legend box over the curves)
    axes[0].text(.69, 15.0, "image 3", fontsize=5.4, ha="right", va="center")
    axes[0].text(.70, 4.2, "image 5", fontsize=5.4, ha="left", va="center")
    os.makedirs(OUT, exist_ok=True)
    S.save(fig, f"{OUT}/fig1_candidate"); plt.close(fig)
    json.dump({"scene_labels": names, "runs": {t: patch(t)["behaviour"] for t, _, _ in runs}},
              open(f"{OUT}/fig1_candidate_values.json", "w"), indent=1)


if __name__ == "__main__":
    main()
