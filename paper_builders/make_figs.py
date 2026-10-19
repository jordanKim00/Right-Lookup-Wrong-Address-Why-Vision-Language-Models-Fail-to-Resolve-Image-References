"""S626 paper figures (matplotlib/seaborn; conventions in knowledge/FIGURE_CONVENTIONS.md). Every number is read from result JSON.
  fig1_overview      retrieve-then-compute (4B) vs compute-then-retrieve (27B) vs completed (4B + R3)   [schematic; layers from S626 + §470 §597-§600]
  fig2_tracing       exact interchange maps, ROME layout (units x layers), sources A/B/C x {4B, 27B, 4B + R3}
  fig3_relay         share of the ordinal effect held by the question-end relay and by the answer position, per layer
  fig4_arrival       depth at which target vs surface image content first reaches the answer position
  fig5_results       sealed MMIU/MUIR dumbbell (base reverse -> R3 reverse, forward bound) + native visual_cloze
  tab_main.tex       booktabs table of the sealed results
Usage (container): python /vlm/exp/figs/make_figs.py [--korean]
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

sys.path.insert(0, "/vlm/exp/figs")
import style as S

R3 = "/vlm/results/r3"; RES = "/vlm/results"; OUT = "/vlm/results/figures/s626"
os.makedirs(OUT, exist_ok=True)
KO = "--korean" in sys.argv
S.apply(korean=KO)
sfx = "_ko" if KO else ""
L = (lambda en, ko: ko) if KO else (lambda en, ko: en)


def patch(tag):
    return json.load(open(f"{R3}/s626_{tag}_patch.json"))


def unit_index(d, key):
    """key: img3, img5, img7, third, image, end, q, nl (turn-end newline), ans (last position)."""
    u = d["units"]
    if key.startswith("img"):
        return u.index(key) if key in u else None
    if key == "ans":
        return len(u) - 1
    if key == "q":
        return next(i for i, x in enumerate(u) if x.strip() == "?")
    if key == "nl":
        q = unit_index(d, "q")
        return next(i for i in range(q + 1, len(u)) if u[i] == "\n")
    return next((i for i, x in enumerate(u) if x.strip() == key), None)


def frac(d, m):
    return np.array(d["maps"][m]) / d["full"][m]


# ---------------------------------------------------------------- fig 1: overview schematic
def fig1():
    fig, axes = plt.subplots(1, 3, figsize=(S.TWO, 3.0), gridspec_kw=dict(wspace=.1))
    X = dict(img=0.0, third=1.15, end=1.75, q=2.35, ans=3.5)

    def base(ax, D, title, tag, tagc):
        ax.set_xlim(-.45, 3.95); ax.set_ylim(-.42, 1.12); ax.spines["left"].set_bounds(0, 1)
        for k, lab in [("img", L("image", "그림")), ("third", "third"), ("end", "end"), ("q", "?"), ("ans", L("answer", "답 자리"))]:
            ax.plot([X[k]] * 2, [0, 1], color=S.RULE, lw=.6, ls=(0, (1.5, 2)), zorder=0)
            ax.text(X[k], 1.06, lab, ha="center", va="bottom", fontsize=6.8, weight="bold")
        ax.set_yticks([0, .5, 1]); ax.set_yticklabels([f"L0", f"L{round((D - 1) / 2)}", f"L{D - 1}"])
        ax.set_xticks([]); ax.spines["bottom"].set_visible(False)
        ax.set_title(title, loc="left", fontsize=7.6, pad=10, linespacing=1.25)
        ax.text(1.0, 1.035, tag, ha="right", va="bottom", fontsize=7, color=tagc, weight="bold", transform=ax.transAxes)

    y = lambda l, D: l / (D - 1)

    def bar(ax, x, l0, l1, D, c, a=.85, ls="-"):
        ax.add_patch(Rectangle((x - .09, y(l0, D)), .18, max(.012, y(l1, D) - y(l0, D)), fc=c, ec=c, alpha=a, lw=.8, ls=ls))

    def img(ax, l, D, n, c):
        ax.add_patch(FancyBboxPatch((X["img"] - .17, y(l, D) - .045), .34, .09, boxstyle="round,pad=0.01", fc=c, ec="none"))
        ax.text(X["img"], y(l, D), str(n), ha="center", va="center", color="white", fontsize=6.5, weight="bold")

    def arrow(ax, x0, y0, x1, y1, c, ls="-", lw=1.1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, color=c, lw=lw, ls=ls, shrinkA=0, shrinkB=1))

    # (a) Qwen3-VL-4B
    ax = axes[0]; D = 36
    base(ax, D, L("(a) Qwen3-VL-4B\nretrieve, then compute", "(a) Qwen3-VL-4B\n조회 뒤 계산"), L("answer: image 3", "답: 3번"), S.SURFACE)
    bar(ax, X["third"], 0, 5, D, S.INK); ax.text(X["third"] + .14, y(2, D), L("pointer $k{=}3$", "포인터 $k{=}3$"), fontsize=6, va="center")
    bar(ax, X["third"], 7, 17, D, S.INK, .3); bar(ax, X["end"], 7, 17, D, S.INK, .3)
    img(ax, 25, D, 3, S.SURFACE); arrow(ax, X["img"] + .18, y(25, D), X["ans"] - .1, y(25, D), S.SURFACE)
    ax.text((X["img"] + X["ans"]) / 2, y(25, D) + .035, L("lookup at L25 with raw $k$", "L25 조회: $k$ 그대로"), ha="center", fontsize=6, color=S.SURFACE, weight="bold")
    bar(ax, X["ans"], 32, 35, D, S.MUTED, .25, "--")
    ax.text(X["ans"] - .14, y(33.5, D), L("$j$ bound L32–34\n(value code, too late)", "$j$ 묶임 L32–34\n(값 부호, 늦음)"), ha="right", va="center", fontsize=5.6, color=S.MUTED)
    # (b) Qwen3.5-27B
    ax = axes[1]; D = 64
    base(ax, D, L("(b) Qwen3.5-27B\ncompute, then retrieve", "(b) Qwen3.5-27B\n계산 뒤 조회"), L("answer: image 5", "답: 5번"), S.TARGET)
    bar(ax, X["third"], 3, 23, D, S.INK); ax.text(X["third"] + .14, y(10, D), L("pointer $k{=}3$", "포인터 $k{=}3$"), fontsize=6, va="center")
    arrow(ax, X["third"] + .09, y(23, D), X["q"] - .1, y(27, D), S.INK, lw=.9)
    bar(ax, X["q"], 27, 47, D, S.INK, .55); ax.text(X["q"] + .14, y(36, D), L("hold\nL27–47", "붙잡기\nL27–47"), fontsize=6, va="center")
    img(ax, 45, D, 7, S.MUTED); arrow(ax, X["img"] + .18, y(45, D), X["q"] - .1, y(45, D), S.MUTED, ls=(0, (2, 1.5)), lw=.9)
    ax.text(X["img"] + .2, y(45, D) - .05, L("read end $N{=}7$ (L47)", "끝 읽기 $N{=}7$ (L47)"), fontsize=5.8, color=S.MUTED, va="top")
    bar(ax, X["q"], 48, 50, D, S.ACCENT, .95)
    ax.text(X["q"] + .14, y(49, D) - .005, L("compose L48–50\n$7{+}1{-}3{=}5$", "합성 L48–50\n$7{+}1{-}3{=}5$"), fontsize=5.8, color=S.ACCENT, va="top", weight="bold")
    img(ax, 54, D, 5, S.TARGET); arrow(ax, X["img"] + .18, y(54, D), X["ans"] - .1, y(54, D), S.TARGET)
    ax.text(X["img"] + .2, y(54, D) + .035, L("lookup L51", "L51 조회"), fontsize=6, color=S.TARGET, weight="bold")
    # (c) Qwen3-VL-4B + R3
    ax = axes[2]; D = 36
    base(ax, D, L("(c) Qwen3-VL-4B + R³\nmissing step inserted", "(c) Qwen3-VL-4B + R³\n빠진 단계 삽입"), L("answer: image 5", "답: 5번"), S.TARGET)
    bar(ax, X["third"], 0, 17, D, S.ACCENT, .75); ax.text(X["third"] + .14, y(9, D), L("pointer\n'fifth'", "포인터\n‘다섯째’"), fontsize=6, va="center", color=S.ACCENT, weight="bold")
    bar(ax, X["end"], 0, 2, D, S.ACCENT, .75); ax.text(X["end"] + .13, y(1, D) + .02, "end→start", fontsize=5.6, color=S.ACCENT)
    img(ax, 25, D, 5, S.TARGET); arrow(ax, X["img"] + .18, y(25, D), X["ans"] - .1, y(25, D), S.TARGET)
    ax.text((X["img"] + X["ans"]) / 2, y(25, D) + .035, L("same L25 lookup", "같은 L25 조회"), ha="center", fontsize=6, color=S.TARGET, weight="bold")
    ax.add_patch(FancyBboxPatch((-.2, -.38), 3.3, .18, boxstyle="round,pad=0.02", fc="white", ec=S.ACCENT, lw=.9))
    ax.text(1.45, -.265, L("R³ resolve: $t = N{+}1{-}k = 5$", "R³ 계산: $t = N{+}1{-}k = 5$"), ha="center", fontsize=6.2, color=S.ACCENT, weight="bold")
    ax.text(1.45, -.34, L("one integer op, outside the model", "정수 연산 하나, 모델 밖"), ha="center", fontsize=5.4, color=S.MUTED)
    arrow(ax, X["third"] - .07, -.01, X["third"] - .07, -.19, S.ACCENT, ls=(0, (2, 1.5)), lw=.8)
    arrow(ax, X["third"] + .07, -.19, X["third"] + .07, .005, S.ACCENT, lw=.8)
    ax.text(X["third"] - .13, -.11, L("read $k$, END", "읽기 $k$, END"), fontsize=5.4, color=S.ACCENT, ha="right")
    ax.text(X["third"] + .13, -.11, L("write at L1", "L1에 쓰기"), fontsize=5.4, color=S.ACCENT)
    for a in axes[1:]:
        a.set_yticklabels([L("L0", "L0"), "", ""]) if False else None
    axes[0].set_ylabel(L("layer (relative depth)", "층(상대 깊이)"))
    S.save(fig, f"{OUT}/fig1_overview{sfx}"); plt.close(fig)


# ---------------------------------------------------------------- fig 2: exact interchange maps
def fig2():
    runs = [("patch_4b", "Qwen3-VL-4B", 25), ("patch_27b", "Qwen3.5-27B", 51), ("patch_4b_r3", "Qwen3-VL-4B + R³", 25)]
    srcs = [("A", L("swap target image (5)", "정답 그림(5) 교체")), ("B", L("swap surface image (3)", "표면 그림(3) 교체")),
            ("Cc", L("swap ordinal (third→second)", "서수 교체(third→second)"))]
    keys = [("img3", L("image 3 (surface)", "그림 3(표면)")), ("img5", L("image 5 (target)", "그림 5(정답)")), ("img7", L("image 7 (last)", "그림 7(마지막)")),
            ("third", "third"), ("image", "image"), ("end", "end"), ("q", "?"), ("nl", L("turn-end ⏎", "턴 끝 ⏎")), ("ans", L("answer position", "답 자리"))]
    fig, axes = plt.subplots(3, 3, figsize=(S.TWO, 4.6), gridspec_kw=dict(hspace=.42, wspace=.08))
    im = None
    for c, (tag, name, rl) in enumerate(runs):
        d = patch(tag); layers = np.array(d["layers"]); step = layers[1] - layers[0]
        for r, (m, mname) in enumerate(srcs):
            ax = axes[r, c]
            if m == "B" and tag.endswith("_r3"):
                ax.text(.5, .5, L("full effect ≈ 0\n(surface image no longer used)", "전체 효과 ≈ 0\n(표면 그림을 더 쓰지 않음)"), ha="center", va="center", fontsize=6.5, color=S.MUTED, transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
                continue
            F = frac(d, m)
            rows, labs = [], []
            for k, lab in keys:
                i = unit_index(d, k)
                rows.append(F[:, i] if i is not None else np.full(len(layers), np.nan)); labs.append(lab)
            M = np.clip(np.array(rows), 0, 1)
            edges = np.concatenate([layers - step / 2, [layers[-1] + step / 2]])
            im = ax.pcolormesh(edges, np.arange(len(keys) + 1), np.ma.masked_invalid(M), cmap="Purples", vmin=0, vmax=1, shading="flat")
            ax.invert_yaxis()
            ax.axvline(rl, color=S.INK, lw=.6, ls=(0, (2, 1.5)))
            ax.set_yticks(np.arange(len(keys)) + .5)
            ax.set_yticklabels(labs if c == 0 else [""] * len(keys))
            for t, (k, _) in zip(ax.get_yticklabels(), keys):
                t.set_color(S.SURFACE if k == "img3" else S.TARGET if k == "img5" else S.INK)
            ax.set_xlim(0, d["D"] - 1)
            ax.tick_params(length=2)
            if r == 0:
                ax.set_title(name, fontsize=8)
            if r == 2:
                ax.set_xlabel(L("layer", "층"))
            if c == 0:
                ax.text(-0.62, 1.03, mname, transform=ax.transAxes, fontsize=7, weight="bold", va="bottom")
    cb = fig.colorbar(im, ax=axes, fraction=.018, pad=.012)
    cb.set_label(L("fraction of full-swap effect", "전체 교체 효과의 몫"), fontsize=6.5); cb.ax.tick_params(labelsize=6)
    S.save(fig, f"{OUT}/fig2_tracing{sfx}"); plt.close(fig)


# ---------------------------------------------------------------- fig 3: relay vs answer share of the ordinal effect
def fig3():
    runs = [("patch_4b", "Qwen3-VL-4B"), ("patch_8b", "Qwen3-VL-8B"), ("patch_35_4b", "Qwen3.5-4B"), ("patch2_35_9b", "Qwen3.5-9B"),
            ("patch_27b", "Qwen3.5-27B"), ("patch_4b_r3", "4B + R³")]
    col = {"Qwen3.5-27B": S.TARGET, "4B + R³": S.ACCENT}
    fig, axes = plt.subplots(1, 2, figsize=(S.TWO * .78, 2.0), sharey=True, gridspec_kw=dict(wspace=.08))
    for tag, name in runs:
        d = patch(tag); F = frac(d, "Cc"); z = np.array(d["layers"]) / (d["D"] - 1)
        relay = F[:, unit_index(d, "q")] + F[:, unit_index(d, "nl")]; ans = F[:, unit_index(d, "ans")]
        c = col.get(name, S.GREYS[runs.index((tag, name)) % len(S.GREYS)]); lw = 1.6 if name in col else .9
        axes[0].plot(z, relay, color=c, lw=lw, marker="o", ms=1.8, label=name)
        axes[1].plot(z, ans, color=c, lw=lw, marker="o", ms=1.8, label=name)
    axes[0].set_title(L("held at the question end (“?” + turn-end ⏎)", "질문 끝(“?” + 턴 끝 ⏎)이 쥔 몫"), fontsize=7.5)
    axes[1].set_title(L("arrived at the answer position", "답 자리에 도착한 몫"), fontsize=7.5)
    for a in axes:
        a.set_xlabel(L("relative depth", "상대 깊이")); a.grid(alpha=.3, lw=.4); a.set_ylim(-.1, 1.08); a.set_xlim(0, 1)
    axes[0].set_ylabel(L("share of ordinal-swap effect", "서수 교체 효과의 몫"))
    axes[1].legend(loc="upper left", fontsize=6, handlelength=1.4)
    S.save(fig, f"{OUT}/fig3_relay{sfx}"); plt.close(fig)


# ---------------------------------------------------------------- fig 4: arrival order (small multiples, Wu et al. Fig. 4 style)
def fig4():
    runs = [("patch_4b", "Qwen3-VL-4B"), ("patch_8b", "Qwen3-VL-8B"), ("patch_35_4b", "Qwen3.5-4B"), ("patch2_35_9b", "Qwen3.5-9B"),
            ("patch_27b", "Qwen3.5-27B"), ("patch_4b_r3", "Qwen3-VL-4B + R³")]
    fig, axes = plt.subplots(1, 6, figsize=(S.TWO, 1.55), sharey=True, gridspec_kw=dict(wspace=.12))
    for ax, (tag, name) in zip(axes, runs):
        d = patch(tag); z = np.array(d["layers"]) / (d["D"] - 1)
        for m, c, mk, lab in (("B", S.SURFACE, "o", L("surface image (3)", "표면 그림(3)")), ("A", S.TARGET, "D", L("target image (5)", "정답 그림(5)"))):
            if abs(d["full"][m]) < 1.0:
                ax.text(.05, .9 if m == "B" else .8, L("surface: no effect", "표면: 효과 없음"), fontsize=5.5, color=S.MUTED, transform=ax.transAxes)
                continue
            ax.plot(z, frac(d, m)[:, -1], color=c, lw=1.3, marker=mk, ms=2.2, label=lab)
        ax.set_title(name, fontsize=6.8); ax.set_xlim(.4, 1.0); ax.set_ylim(-.05, 1.05); ax.grid(alpha=.3, lw=.4)
        ax.set_xticks([.5, .75, 1.0]); ax.set_xticklabels([".5", ".75", "1"])
    axes[0].set_ylabel(L("share at answer pos.", "답 자리 몫"))
    fig.supxlabel(L("relative depth", "상대 깊이"), fontsize=7, y=-.06)
    axes[0].legend(loc="upper left", fontsize=5.5, handlelength=1.2)
    S.save(fig, f"{OUT}/fig4_arrival{sfx}"); plt.close(fig)


# ---------------------------------------------------------------- fig 5: results
def sealed():
    M = [("Qwen3-VL-4B", "s609c_4b"), ("Qwen3-VL-8B", "s609c_8b"), ("Qwen3.5-4B", "s619g_35_4b"), ("Qwen3.5-9B", "s619g_35_9b"),
         ("Qwen2.5-VL-7B", "s624rel_q25_7b"), ("InternVL3-8B", "s616b_ivl3_8b")]
    rows = []
    for name, tag in M:
        f = f"{RES}/s582_evalrepair_s601_{tag}.json"
        if not os.path.exists(f):
            continue
        r = json.load(open(f))["res"]
        rows.append(dict(name=name, mod=tag.split("_")[0].upper(),
                         **{s: (r[f"{s}|F"]["base"], r[f"{s}|R"]["base"], r[f"{s}|R"]["rep"], r[f"{s}|F"]["rep"] - r[f"{s}|F"]["base"]) for s in ("mmiu_pos", "muir")},
                         pres=[r[k]["rep"] - r[k]["base"] for k in ("mmiu_other|O", "muir_all|O") if k in r]))
    return rows


def cloze():
    out = []
    for name, tag in [("Qwen3-VL-4B", "e582_s609c_4b"), ("Qwen3-VL-8B", "e582_s609c_8b"), ("Qwen3.5-4B", "cloze_s619g_35_4b"),
                      ("Qwen3.5-9B", "cloze_s619g_35_9b"), ("Qwen2.5-VL-7B", "cloze_s624rel_q25_7b")]:
        f = f"{RES}/s582_evalrepair_{tag}.json"
        if not os.path.exists(f):
            continue
        r = json.load(open(f))["res"]
        if "cloze|R" in r:
            out.append((name, r["cloze|R"]["base"], r["cloze|R"]["rep"], r["cloze|F"]["base"]))
    return out


def fig5():
    rows, cz = sealed(), cloze()
    fig = plt.figure(figsize=(S.TWO, 2.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, .8], wspace=.12)
    for k, (s, title) in enumerate([("mmiu_pos", L("MMIU positional (sealed, 1,082)", "MMIU 위치(봉인, 1,082)")), ("muir", L("MUIR positional (sealed, 371)", "MUIR 위치(봉인, 371)"))]):
        ax = fig.add_subplot(gs[0, k])
        for i, r in enumerate(rows):
            up, b, rr, _ = r[s]; yv = len(rows) - 1 - i
            ax.plot([b, rr], [yv, yv], color=S.TARGET, lw=1.4, alpha=.5, zorder=1)
            ax.scatter(b, yv, s=18, facecolor="white", edgecolor=S.MUTED, lw=1, zorder=3)
            ax.scatter(rr, yv, s=20, color=S.TARGET, zorder=3)
            ax.plot([up, up], [yv - .28, yv + .28], color=S.INK, lw=1.3, zorder=2)
            ax.text(.9, yv, f"{(rr - b) / (up - b):.0%}", fontsize=5.8, color=S.MUTED, va="center", ha="right")
        ax.set_yticks(range(len(rows))); ax.set_yticklabels([r["name"] for r in rows][::-1] if k == 0 else [""] * len(rows))
        ax.set_xlim(.18, .92); ax.set_xlabel(L("reverse accuracy", "역순 정확도")); ax.set_title(title, fontsize=7.5)
        ax.grid(axis="x", alpha=.3, lw=.4)
    ax = fig.add_subplot(gs[0, 2])
    x = np.arange(len(cz)); w = .36
    ax.bar(x - w / 2, [c[1] for c in cz], w, color="#c9cec9", label=L("base", "원 모델"))
    ax.bar(x + w / 2, [c[2] for c in cz], w, color=S.TARGET, label="R³")
    for i, c in enumerate(cz):
        ax.plot([i - .42, i + .42], [c[3], c[3]], color=S.INK, lw=1.1)
    ax.axhline(.25, color=S.MUTED, lw=.6, ls=":")
    ax.set_xticks(x); ax.set_xticklabels([c[0].replace("Qwen", "Q").replace("-VL", "VL") for c in cz], rotation=30, ha="right", fontsize=6)
    ax.set_ylim(0, .7); ax.set_title(L("native reverse task:\nMMIU visual_cloze (200)", "원래 역방향 과제:\nMMIU visual_cloze(200)"), fontsize=7.5)
    ax.legend(fontsize=6, loc="upper right"); ax.grid(axis="y", alpha=.3, lw=.4)
    ax.text(-.45, .262, L("chance", "우연"), fontsize=5.5, color=S.MUTED, ha="left")
    S.save(fig, f"{OUT}/fig5_results{sfx}"); plt.close(fig)
    return rows, cz


def table(rows, cz):
    czd = {c[0]: c for c in cz}
    lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
             r" & \multicolumn{2}{c}{MMIU pos.\ (sealed)} & \multicolumn{2}{c}{MUIR pos.\ (sealed)} & visual\_cloze & $\Delta$ fwd / pres. \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-6}",
             r"Model & rev.\ base$\to$R$^3$ & rec. & rev.\ base$\to$R$^3$ & rec. & base$\to$R$^3$ & \\", r"\midrule"]
    for r in rows:
        cells = [r["name"]]
        for s in ("mmiu_pos", "muir"):
            up, b, rr, dF = r[s]
            cells += [f"{b:.3f}$\\to${rr:.3f}", f"{(rr - b) / (up - b) * 100:.0f}\\%"]
        c = czd.get(r["name"]); cells.append(f"{c[1]:.3f}$\\to${c[2]:.3f}" if c else "--")
        cells.append(f"{max(abs(r['mmiu_pos'][3]), abs(r['muir'][3])):.3f} / {max(abs(x) for x in r['pres']):.3f}")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT}/tab_main.tex", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); rows, cz = fig5()
    if not KO:
        table(rows, cz)
    print("->", OUT, sorted(os.listdir(OUT)))
