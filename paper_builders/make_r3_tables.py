"""R³ tables (paper v3), generated from saved records only (no model calls).  Missing inputs give "--".
Writes LaTeX tabular bodies to /vlm/results/figures/tables_r3/:
    read_choice.tex   per-model READ (S724 plan-level selection), PlanExact of the uniform READ, R³-v1 and the model's READ (validation split)
    stagewise.tex     S730: answers on the target and the matcher heads' query similarity to the forward wording, per condition
    reader.tex        S729: count accuracy of the end reader and the fresh synthetic draw, before / after re-calibration
    lexicon.tex       S734 / S717: other end-relative wordings before / after the frame-lexicon calibration
Usage: python figs/make_r3_tables.py
"""
import json, os
import numpy as np

R = "/vlm/results/r3"
OUT = "/vlm/results/figures/tables_r3"
P = lambda x: "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.0f}"
F3 = lambda x: "--" if x is None else f"{x:.3f}"
MODELS = [("4b", "Qwen3-VL-4B"), ("8b", "Qwen3-VL-8B"), ("35_4b", "Qwen3.5-4B"), ("35_9b", "Qwen3.5-9B"), ("27b", "Qwen3.5-27B")]


# S783 (FINDINGS §846): procedural answers re-read from the bare tokens.  bare(name, tag_old, tag_new) returns the re-measured record when its
# analysis exists, else the original one, so every 27B table moves to the generation-consistent readout as soon as its re-run is analysed.
BARE_TAGS = {"s758_27b": "s783r_27b", "s772_27b7A": "s783f_27b7A", "s772_27b8A": "s783f_27b8A", "s772_27b7B": "s783f_27b7B", "s772_27b8B": "s783f_27b8B",
             "s780_27b7": "s783_27b7", "s780_27b8": "s783_27b8", "s781_27b7": "s783p_27b7", "s781_27b8": "s783p_27b8"}


def bare(name):
    """name e.g. 's772_analysis.json' -> dict keyed by the ORIGINAL tags, filled from the *_bare.json analysis when present."""
    B = load(name.replace(".json", "_bare.json"))
    A = load(name) or {}
    if not B:
        return A
    out = dict(A)
    for old, new in BARE_TAGS.items():
        if new in B:
            out[old] = B[new]
    for k in ("N7", "N8"):                                                              # s769 keys its cells by N
        if k in B:
            out[k] = B[k]
    return out


def load(f):
    p = f if f.startswith("/") else f"{R}/{f}"
    return json.load(open(p)) if os.path.exists(p) else None


def _plan_in_force():
    """S738: the plan file named by /vlm/exp/r3_plan_default.txt (the clean-room plan), else the registered S724 plan."""
    try:
        p = open("/vlm/exp/r3_plan_default.txt").read().strip()
        return json.load(open(p)) if p else None
    except FileNotFoundError:
        return None


def _nm(x):
    if x.startswith("margin:"):
        return "margin " + x.split(":")[1].lstrip("0")
    if x.startswith("proto:"):
        return "clause anchor $\\times$" + x.split(":")[1]
    return {"abs": "absolute", "v1_gate": "learned frame gate", "v1": "absolute", "last": "last block", "max": "largest code"}.get(x, x)


def read_choice():
    plan = load("s724_plan.json") or {}
    dedup = load("/vlm/research/ces/outputs/r3_lure_latest/s738b_plan_val.json")     # S738b: confirmation split with the duplicate photograph swapped out
    if dedup:
        plan = {**plan, **dedup}
    force = _plan_in_force()
    if force:                                                                       # S738 rows: the clean-room choice and its PlanExact
        L = ["\\begin{tabular}{llllrrr}", "\\toprule",
             "& \\multicolumn{3}{c}{Read chosen on calibration data} & \\multicolumn{3}{c}{exact write plans (\\%)} \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}", "Model & ordinal detector & frame binder & image count & uniform & learned resolver & ours \\\\", "\\midrule"]
        for b, nm in MODELS:
            f = (force.get(f"rc_{b}") or {}).get("final"); p0 = plan.get(f"rc_{b}") or {}
            if not f:
                L.append(f"{nm} & -- & -- & -- & -- & -- & -- \\\\"); continue
            val = p0.get("val", {})
            unif = val.get("margin:.5 + proto:1.2 + max", {}).get("all", {}).get("exact")
            v1 = val.get("R3-v1", {}).get("all", {}).get("exact")
            L.append(f"{nm} & {_nm(f[1])} & {_nm(f[2])} & {_nm(f[3])} & {P(unif)} & {P(v1)} & {P(force[f'rc_{b}'].get('planexact_val'))} \\\\")
        L += ["\\bottomrule", "\\end{tabular}"]
        return "\n".join(L)
    names = {"abs": "absolute", "margin:.4": "margin .4", "margin:.5": "margin .5", "margin:.6": "margin .6",
             "proto:1.0": "clause anchor $\\times$1.0", "proto:1.2": "clause anchor $\\times$1.2", "proto:1.5": "clause anchor $\\times$1.5",
             "v1_gate": "learned frame gate", "v1": "learned (with gate)", "last": "last block", "max": "largest code"}
    L = ["\\begin{tabular}{llllrrr}", "\\toprule",
         "& \\multicolumn{3}{c}{READ chosen on calibration data} & \\multicolumn{3}{c}{exact write plans (\\%)} \\\\",
         "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}", "Model & ordinal detector & frame binder & image count & uniform & learned resolver & ours \\\\", "\\midrule"]
    for b, nm in MODELS:
        p = plan.get(f"rc_{b}")
        if not p:
            L.append(f"{nm} & -- & -- & -- & -- & -- & -- \\\\"); continue
        f = p["final"]; val = p["val"]
        unif = val.get("margin:.5 + proto:1.2 + max", {}).get("all", {}).get("exact")
        v1 = val.get("R3-v1", {}).get("all", {}).get("exact")
        fin = val.get(p["final_name"], {}).get("all", {}).get("exact")
        det = names.get(f[1], f[1]) if f[0] != "v1c" else names["abs"]                  # the gated module detects with the absolute threshold
        L.append(f"{nm} & {det} & {names.get(f[2], f[2])} & {names.get(f[3], f[3])} & {P(unif)} & {P(v1)} & {P(fin)} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L)


def stagewise():
    A = load("s730_analysis.json") or {}
    conds = [("START", "forward wording"), ("END", "end-relative, unedited"), ("ADDR", "address write only"), ("FRAME", "frame write only"), ("R3", "\\rthree{} (both writes)")]
    L = ["\\begin{tabular}{l" + "rr" * len(MODELS) + "}", "\\toprule",
         " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{nm.replace('Qwen3-VL-', 'VL-').replace('Qwen3.5-', '3.5-')}}}" for _, nm in MODELS) + " \\\\",
         "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(MODELS))),
         "Condition & " + " & ".join("acc & cos" for _ in MODELS) + " \\\\", "\\midrule"]
    for c, lab in conds:
        cells = []
        for b, _ in MODELS:
            s = (A.get(f"stw_{b}") or {}).get("summary", {}).get("k!=4", {}).get(c)
            cells += [P(s["acc"]) if s else "--", (f"{s['cos']:.2f}" if s else "--")]
        L.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    L += ["\\midrule", "matcher heads & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{(A.get(f'stw_{b}') or {}).get('heads', '--')}}}" for b, _ in MODELS) + " \\\\",
          "\\bottomrule", "\\end{tabular}"]
    return "\n".join(L)


def reader():
    S = load("s729_analysis.json") or {}
    L = ["\\begin{tabular}{lrrrrrr}", "\\toprule",
         "& \\multicolumn{2}{c}{image count, held-out (\\%)} & \\multicolumn{3}{c}{fresh synthetic draw, end-relative (\\%)} & \\\\",
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}", "Model / reader & low resolution & $N\\geq9$ & failing $N$ & other sets & R³-v1 & MMIU \\\\", "\\midrule"]
    for b, nm, fn in (("4b", "Qwen3-VL-4B", "N9"), ("35_9b", "Qwen3.5-9B", "N10")):
        rd = load(f"s729_rd_{b}_reader.json"); s = (S.get(b) or {}).get("out", {})
        fr = s.get("fresh", {}); mm = s.get("mmiu", {})
        for arm, lab in (("old", "procedural reader"), ("new", "re-calibrated")):
            r = rd["res"][arm] if rd else None
            oth = fr.get("other", {}).get(arm, {})
            o_ = [x for k, x in oth.items() if k not in (fn,) and x is not None]
            L.append(f"{nm} {lab} & {P(r['low']['count']) if r else '--'} & {P(r['N10']) if r else '--'} & {P(fr.get('end', {}).get(arm))} & "
                     f"{P(float(np.mean(o_)) if o_ else None)} & {P(fr.get('end', {}).get('v1')) if arm == 'new' else ''} & "
                     f"{P(mm.get('sealed', {}).get(arm if arm == 'new' else 'old'))} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L)


def lexicon():
    rows = [("Qwen3-VL-8B", "s734_8b_analysis.json"), ("Qwen3.5-9B", "s734_35_9b_analysis.json"), ("Qwen3.5-27B", "s717_analysis.json")]
    L = ["\\begin{tabular}{lrrrrrr}", "\\toprule",
         "& \\multicolumn{2}{c}{``$k$-th-to-last''} & \\multicolumn{2}{c}{``in reverse order''} & \\multicolumn{2}{c}{``counting backwards''} \\\\",
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}", "Model (MMIU, development) & original & \\rthree{} & original & \\rthree{} & original & \\rthree{} \\\\", "\\midrule"]
    for nm, f in rows:
        d = load(f)
        cells = []
        for fam in ("tolast", "reverse", "backwards"):
            x = (d or {}).get("res", {}).get(fam, {}).get("mmiu_pos", {}).get("R") if d else None
            cells += [P(x["base"]) if x else "--", P(x["rep"]) if x else "--"]
        L.append(f"{nm} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L)


# Benchmark rows: the latest saved configuration per model. S732 (f7*) supersedes the small-model tags once it is complete.
BENCH_TAGS = {"4b": dict(sealed="a6sl_4b", later="a6fr_4b", cloze="a5cz_4b"),
              "8b": dict(sealed="a5sl_8b", later="a5fr_8b", cloze="a5cz_8b"),
              "35_4b": dict(sealed="a5sl_35_4b", later="a5fr_35_4b", cloze="a5cz_35_4b"),
              "35_9b": dict(sealed="a6sl_35_9b", later="a6fr_35_9b", cloze="a5cz_35_9b"),
              "27b": dict(sealed="a5sl_27b", later="a5fr_27b", cloze="a5cz_27b")}
F732 = {b: dict(sealed=f"f7sl_{b}", later=f"f7fr_{b}", cloze=f"f7cz_{b}") for b in ("4b", "8b", "35_4b", "35_9b")}
F732["27b"] = dict(sealed="c8sl_27b", later="c8fr_27b", cloze="c8cz_27b")            # S738: 27B with the clean-room Read (c8 runs)
BENCH_SETS = [("sealed", "mmiu_pos", "MMIU held-out"), ("sealed", "muir", "MuirBench held-out"), ("later", "mmiu_pos", "MMIU later"),
              ("cloze", "cloze", "MMIU cloze")]


def bench_tags():
    tags = {b: dict(t) for b, t in BENCH_TAGS.items()}
    for b, t in F732.items():                                                      # switch a model to S732 only when all its runs exist
        if all(os.path.exists(f"/vlm/results/s582_evalrepair_{x}.json") for x in t.values()):
            tags[b] = dict(t)
    return tags


def bench_values():
    out = {}
    for b, nm in MODELS:
        tg = bench_tags()[b]
        for split, suite, lab in BENCH_SETS:
            d = load(f"/vlm/results/s582_evalrepair_{tg[split]}.json")
            if not d:
                continue
            rows = [r for r in d["rows"] if r["suite"] == suite]
            m = lambda k: float(np.mean([r[k] == r["gold"] for r in rows]))
            F, R_, rep = m("F|base"), m("R|base"), m("R|rep")
            out[f"{b}|{lab}"] = dict(tag=tg[split], n=len(rows), F=F, R=R_, rep=rep, closed=(rep - R_) / (F - R_) if F > R_ else None,
                                    F_changed=int(sum(r["F|rep"] != r["F|base"] for r in rows)))
        d = load(f"/vlm/results/s582_evalrepair_{tg['sealed']}.json")
        if d:
            o = [r for r in d["rows"] if "O|base" in r]
            out[f"{b}|position-free"] = dict(n=len(o), changed=int(sum(r["O|rep"] != r["O|base"] for r in o)))
    return out


def bench():
    V = bench_values()
    json.dump(V, open(f"{OUT}/bench_values.json", "w"), indent=1)
    p1 = lambda x: "--" if x is None else f"{100 * x:.1f}"
    L = ["\\begin{tabular}{l" + "rrr" * len(BENCH_SETS) + "}", "\\toprule",
         " & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{lab}}}" for _, _, lab in BENCH_SETS) + " \\\\",
         "".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(BENCH_SETS))),
         "Model & " + " & ".join("Start & End & +\\rthree" for _ in BENCH_SETS) + " \\\\", "\\midrule"]
    for b, nm in MODELS:
        cells = []
        for _, _, lab in BENCH_SETS:
            v = V.get(f"{b}|{lab}")
            cells += [p1(v["F"]), p1(v["R"]), "\\textbf{" + p1(v["rep"]) + "}"] if v else ["--"] * 3
        L.append(f"{nm} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L)


def lure():
    import subprocess
    tags = bench_tags()
    runs = [f"{tags[b]['sealed']}:sealed" for b, _ in MODELS]
    out = f"{OUT}/lure_values.json"
    subprocess.run(["python3", "/vlm/exp/lure_latest.py", *runs, "--out", out], check=True, capture_output=True)
    V = {d["tag"]: d for d in json.load(open(out))}
    L = ["\\begin{tabular}{lrrrrr}", "\\toprule",
         "& & \\multicolumn{2}{c}{errors on the lure} & \\multicolumn{2}{c}{corrected by \\rthree} \\\\",
         "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}", "Model & errors & before & after & lure & other \\\\", "\\midrule"]
    for b, nm in MODELS:
        d = V.get(tags[b]["sealed"])
        L.append(f"{nm} & {d['errors']} & {P(d['lure_share'])} & {P(d['lure_share_after'])} & {P(d['rescued_lure'])} & {P(d['rescued_other'])} \\\\"
                 if d else f"{nm} & -- & -- & -- & -- & -- \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L)


def stagewise_main():
    """S730 (S786 when present: the same runs read from the bare tokens, FINDINGS §846): the 2 x 2 of address and frame writes, k != 4, and the
    k = 4 cell (target = literal image), where only the frame is rewritten; query cosine to the start-relative wording."""
    A = load("s730_analysis.json") or {}
    def key(b):
        if f"s786_{b}" in A:
            return f"s786_{b}"
        # 27B: stw_27b's ADDR / FRAME arms are an instrument failure (FINDINGS §767); use the corrected stw2_27b once it exists
        return "stw2_27b" if (b == "27b" and "stw2_27b" in A) else f"stw_{b}"
    def get(b, c, sub="k!=4"):
        k = key(b)
        if k == "stw_27b" and c in ("ADDR", "FRAME"):
            return None
        return (A.get(k) or {}).get("summary", {}).get(sub, {}).get(c)
    short = lambda nm: nm.replace("Qwen3-VL-", "VL-").replace("Qwen3.5-", "3.5-")
    W = 1 + len(MODELS)
    L = ["\\begin{tabular}{l" + "r" * len(MODELS) + "}", "\\toprule",
         " & " + " & ".join(short(nm) for _, nm in MODELS) + " \\\\", "\\midrule",
         f"\\multicolumn{{{W}}}{{l}}{{\\textit{{Accuracy on the target (\\%), $k\\neq4$}}}} \\\\"]
    for c, lab in (("START", "start-relative"), ("END", "end-relative"), ("ADDR", "\\quad + address"),
                   ("FRAME", "\\quad + frame"), ("R3", "\\quad + both (\\rthree)")):
        L.append(f"{lab} & " + " & ".join(P(get(b, c)["acc"]) if get(b, c) else "--" for b, _ in MODELS) + " \\\\")
    L += ["\\addlinespace", f"\\multicolumn{{{W}}}{{l}}{{\\textit{{Accuracy, $k=4$ (target = literal image)}}}} \\\\"]
    for c, lab in (("END", "end-relative"), ("FRAME", "\\quad + frame")):
        L.append(f"{lab} & " + " & ".join(P(get(b, c, "k=4")["acc"]) if get(b, c, "k=4") else "--" for b, _ in MODELS) + " \\\\")
    L += ["\\midrule", f"\\multicolumn{{{W}}}{{l}}{{\\textit{{Query cosine to start-relative}}}} \\\\"]
    for c, lab in (("END", "end-relative"), ("ADDR", "\\quad + address"), ("R3", "\\quad + both (\\rthree)")):
        L.append(f"{lab} & " + " & ".join(f"{get(b, c)['cos']:.2f}" if get(b, c) else "--" for b, _ in MODELS) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L)


def rebind_main():
    """Table 1 (final form, needs s786_* for all five models): the 2 x 2 of address and frame writes (S786, bare readout), the k = 4 cell,
    the query cosine, and the photographs of the later MMIU split for Qwen3.5-27B (S721, option letters; FINDINGS §787)."""
    A = load("s730_analysis.json") or {}
    # S794: Qwen3.5-4B generates the space-prefixed form on this prompt, so its bare-read S786 run is not used; s794_35_4b (two-form readout) replaces it
    KEY = {b: (f"s794_{b}" if b == "35_4b" else f"s786_{b}") for b, _ in MODELS}
    MODELS_ = [(b, nm) for b, nm in MODELS if KEY[b] in A]
    if not MODELS_:
        raise FileNotFoundError("s786 / s794 missing")
    g = lambda b, c, sub="k!=4": A[KEY[b]]["summary"][sub][c]
    short = lambda nm: nm.replace("Qwen3-VL-", "VL-").replace("Qwen3.5-", "3.5-")
    W = 1 + len(MODELS_)
    L = ["\\begin{tabular}{@{}l" + "r" * len(MODELS_) + "@{}}", "\\toprule",
         " & " + " & ".join(short(nm) for _, nm in MODELS_) + " \\\\", "\\midrule",
         f"\\multicolumn{{{W}}}{{@{{}}l}}{{\\textit{{Accuracy on the target (\\%), $k\\neq4$}}}} \\\\"]
    for c, lab in (("START", "start-relative"), ("END", "end-relative"), ("ADDR", "\\quad + address"), ("FRAME", "\\quad + frame"), ("R3", "\\quad + both (\\rthree)")):
        L.append(f"{lab} & " + " & ".join(P(g(b, c)["acc"]) for b, _ in MODELS_) + " \\\\")
    L += ["\\addlinespace", f"\\multicolumn{{{W}}}{{@{{}}l}}{{\\textit{{Accuracy, $k=4$ (target = literal image)}}}} \\\\"]
    for c, lab in (("END", "end-relative"), ("FRAME", "\\quad + frame")):
        L.append(f"{lab} & " + " & ".join(P(g(b, c, "k=4")["acc"]) for b, _ in MODELS_) + " \\\\")
    L += ["\\addlinespace", f"\\multicolumn{{{W}}}{{@{{}}l}}{{\\textit{{Query cosine to start-relative}}}} \\\\"]
    for c, lab in (("END", "end-relative"), ("ADDR", "\\quad + address"), ("R3", "\\quad + both (\\rthree)")):
        L.append(f"{lab} & " + " & ".join(f"{g(b, c)['cos']:.2f}" for b, _ in MODELS_) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "", "\\vspace{4pt}", "\\begin{tabular}{@{}lrrrrr@{}}", "\\toprule",
          "\\multicolumn{6}{@{}l}{\\textit{Qwen3.5-27B, later MMIU split, accuracy (\\%)}} \\\\",
          "end & + address & + frame & + both & random & start \\\\", "\\midrule",
          "34.4 & 57.8 & 30.2 & 82.0 & 31.6 & 85.2 \\\\", "\\bottomrule", "\\end{tabular}"]
    return "\n".join(L)


TASK_NAMES = {"face_retrieval": "face retrieval", "handwritten_retrieval": "handwriting retrieval", "image2image_retrieval": "image-to-image retrieval",
              "person_reid": "person re-identification", "vehicle_retrieval": "vehicle retrieval", "sketch2image_retrieval": "sketch-to-image retrieval",
              "text2image_retrieval": "text-to-image retrieval", "forensic_detection_blink": "forensic detection (BLINK)",
              "forensic_detection_forgerynet": "forensic detection (ForgeryNet)", "visual_coherence": "visual coherence",
              "Icon_Question_Answering_with_Spatial_Context": "icon question answering"}


def bench_tasks():
    tags = bench_tags()
    rows = {b: [r for r in load(f"/vlm/results/s582_evalrepair_{tags[b]['sealed']}.json")["rows"] if r["suite"] == "mmiu_pos"] for b, _ in MODELS}
    short = lambda nm: nm.replace("Qwen3-VL-", "VL-").replace("Qwen3.5-", "3.5-")
    L = ["\\begin{tabular}{lr" + "rr" * len(MODELS) + "}", "\\toprule",
         " & & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{short(nm)}}}" for _, nm in MODELS) + " \\\\",
         "".join(f"\\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(MODELS))),
         "Task & $n$ & " + " & ".join("End & +\\rthree" for _ in MODELS) + " \\\\", "\\midrule"]
    for t, nm in TASK_NAMES.items():
        cells = []
        for b, _ in MODELS:
            rr = [r for r in rows[b] if r["task"] == t]
            cells += [P(np.mean([r["R|base"] == r["gold"] for r in rr])), P(np.mean([r["R|rep"] == r["gold"] for r in rr]))]
        L.append(f"{nm} & {len([r for r in rows[MODELS[0][0]] if r['task'] == t])} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L)


def bench_paired():
    tags = bench_tags(); rng = np.random.default_rng(0)
    sets = [("sealed", "mmiu_pos", "MMIU held-out"), ("sealed", "muir", "MuirBench held-out"), ("later", "mmiu_pos", "MMIU later"), ("cloze", "cloze", "MMIU cloze")]
    L = ["\\begin{tabular}{l" + "rrl" * len(sets) + "}", "\\toprule",
         " & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{lab}}}" for _, _, lab in sets) + " \\\\",
         "".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(sets))),
         "Model & " + " & ".join("res. & harm & $\\Delta$ [95\\% CI]" for _ in sets) + " \\\\", "\\midrule"]
    for b, nm in MODELS:
        cells = []
        for split, suite, _ in sets:
            d = load(f"/vlm/results/s582_evalrepair_{tags[b][split]}.json")
            rr = [r for r in d["rows"] if r["suite"] == suite] if d else []
            if not rr:
                cells += ["--"] * 3; continue
            x = np.array([(r["R|rep"] == r["gold"]) - (r["R|base"] == r["gold"]) for r in rr], float)
            bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(4000)]
            cells += [str(int((x > 0).sum())), str(int((x < 0).sum())),
                      f"{100 * x.mean():+.1f} [{100 * np.percentile(bs, 2.5):+.1f}, {100 * np.percentile(bs, 97.5):+.1f}]"]
        L.append(f"{nm} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L)


def generality():
    # Raw S718-format records per model: latest saved (a5 for the small models, lm for 27B); S732 (f7) once all its runs exist.
    tags = {b: (f"a5gen_{b}", f"a5rs_{b}") for b in ("4b", "8b", "35_4b", "35_9b")}
    tags["27b"] = ("c8gen_27b", "c8rs_27b") if all(os.path.exists(f"{R}/{t}_general.json") for t in ("c8gen_27b", "c8rs_27b")) else ("lmgen_27b", "lmrs_27b")
    for b in ("4b", "8b", "35_4b", "35_9b"):
        if all(os.path.exists(f"{R}/{t}_general.json") for t in (f"f7gen_{b}", f"f7rs_{b}")):
            tags[b] = (f"f7gen_{b}", f"f7rs_{b}")
    cols = [("blink", "BLINK"), ("qbench2", "Q-Bench2"), ("mibench", "MIBench"), ("mirb", "MIRB"), ("mantis", "Mantis")]
    L = ["\\begin{tabular}{l" + "rr" * len(cols) + "}", "\\toprule",
         " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{lab}}}" for _, lab in cols) + " \\\\",
         "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(cols))),
         "Model & " + " & ".join("End & +\\rthree" for _ in cols) + " \\\\", "\\midrule"]
    for b, nm in MODELS:
        g, r = (load(f"{t}_general.json") for t in tags[b])
        if not g or not r:
            L.append(f"{nm} & " + " & ".join(["--"] * (2 * len(cols))) + " \\\\"); continue
        cells = []
        for ds, _ in cols:
            v = g["res"].get(f"{ds}|R")
            cells += [P(v["base"]), P(v["rep"])] if v else ["--", "--"]
        keep = [(v["n"], v.get("changed") or 0) for d in (g, r) for k, v in d["res"].items() if k.endswith("|O")]
        ch = int(round(sum(n * c for n, c in keep))); tot = sum(n for n, _ in keep)
        L.append(f"{nm} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L)


def preserve():
    """Answers changed by R3 on items without an end-relative reference (the final configuration of every model): the start-relative wording of
    the held-out and later position items, the position-free items, the original-form items of five multi-image subsets and MMMU, and the
    reasoning suite.  Cells: changed answers (count); n: items per model."""
    tags = bench_tags(); gt = {}
    for b in ("4b", "8b", "35_4b", "35_9b"):
        gt[b] = (f"f7gen_{b}", f"f7rs_{b}") if all(os.path.exists(f"{R}/{t}_general.json") for t in (f"f7gen_{b}", f"f7rs_{b}")) else (f"a5gen_{b}", f"a5rs_{b}")
    gt["27b"] = ("c8gen_27b", "c8rs_27b")
    rows = [("start-relative, held-out MMIU and MuirBench", "sealedF"), ("start-relative, later MMIU", "laterF"), ("position-free MMIU and MuirBench", "free"),
            ("BLINK", "blink|O"), ("Q-Bench2", "qbench2|O"), ("Mantis", "mantis|O"), ("MMMU", "mmmu|O"), ("MIBench", "mibench|O"), ("MIRB", "mirb|O"),
            ("MIBench reasoning", "mibench_reason|O"), ("MIRB reasoning", "mirb_reason|O"), ("MathVista", "mathvista|O")]
    cell, nn = {}, {}
    for b, _ in MODELS:
        sl = (load(f"/vlm/results/s582_evalrepair_{tags[b]['sealed']}.json") or {}).get("rows", [])
        lt = (load(f"/vlm/results/s582_evalrepair_{tags[b]['later']}.json") or {}).get("rows", [])
        pos = [r for r in sl if r["suite"] in ("mmiu_pos", "muir")]; free = [r for r in sl if r["suite"] in ("mmiu_other", "muir_all")]
        lp = [r for r in lt if r["suite"] == "mmiu_pos"]
        for key, rr in (("sealedF", pos), ("laterF", lp)):
            cell[(key, b)] = sum(r["F|rep"] != r["F|base"] for r in rr); nn[key] = len(rr)
        cell[("free", b)] = sum(r["O|rep"] != r["O|base"] for r in free); nn["free"] = len(free)
        g, r_ = (load(f"{t}_general.json") or {"res": {}} for t in gt[b])
        for _, key in rows[3:]:
            v = g["res"].get(key) or r_["res"].get(key)
            if v:
                cell[(key, b)] = int(round(v["n"] * (v.get("changed") or 0))); nn[key] = v["n"]
    short = lambda nm: nm.replace("Qwen3-VL-", "VL-").replace("Qwen3.5-", "3.5-")
    L = ["\\begin{tabular}{@{}lr" + "r" * len(MODELS) + "@{}}", "\\toprule", "Items & $n$ & " + " & ".join(short(nm) for _, nm in MODELS) + " \\\\", "\\midrule"]
    for lab, key in rows:
        L.append(f"{lab} & {nn.get(key, 0):,} & " + " & ".join(str(cell.get((key, b), '--')) for b, _ in MODELS) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L)


def frame736():
    """S736: accuracy on the target by arm, split by the written address t = N + 1 - k (near t <= 3, far t >= 5); literal share for the address-only arm."""
    A = load("s736_analysis.json") or {}
    short = lambda nm: nm.replace("Qwen3-VL-", "VL-").replace("Qwen3.5-", "3.5-")
    arms = (("START", "start-relative"), ("END", "end-relative"), ("ADDR", "address only"), ("RANDF", "address + random frame"),
            ("R3", "address + frame (\\rthree)"), ("FRAME", "frame only"))
    L = ["\\begin{tabular}{l" + "rr" * len(MODELS) + "}", "\\toprule",
         " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{short(nm)}}}" for _, nm in MODELS) + " \\\\",
         "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(MODELS))),
         "Write & " + " & ".join("near & far" for _ in MODELS) + " \\\\", "\\midrule"]
    for c, lab in arms:
        cells = []
        for b, _ in MODELS:
            S = (A.get(f"fr736_{b}") or {}).get("summary")
            cells += [P(S["near"][c]), P(S["far"][c])] if S else ["--", "--"]
        L.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L)


def deref673():
    """S673b: the ordinal-token state of a forward question about position j, written at one layer into an end-relative question about the
    same images in another order (dense models): answers on position j, on the donor's named image y_j, on the literal image k."""
    short = lambda nm: nm.replace("Qwen3-VL-", "VL-")
    rows, LS = [], None
    for b, nm in (("4b", "Qwen3-VL-4B"), ("8b", "Qwen3-VL-8B")):
        d = load(f"s673b_db_{b}_deref.json")
        if not d:
            continue
        LS = d["LS"]; R_ = [r for r in d["rows"] if r["donor_ok"]]
        cat = lambda r, a: "j" if a == r["j"] else ("y" if a == r["yj"] else ("k" if a == r["k"] else "o"))
        sh = lambda key, c: float(np.mean([cat(r, r["land"][key]) == c for r in R_]))
        for c, lab in (("j", f"{short(nm)}, position $j$"), ("y", "\\quad image named by the donor"), ("k", "\\quad literal image $k$")):
            rows.append(f"{lab} & " + " & ".join(P(sh(f"ORD@{L_}", c)) for L_ in LS) + " \\\\")
    if not rows:
        raise FileNotFoundError("s673b records")
    L = ["\\begin{tabular}{l" + "r" * len(LS) + "}", "\\toprule", "Layer & " + " & ".join(str(x) for x in LS) + " \\\\", "\\midrule"] + rows + ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L)


def tokens739():
    """S739: forward retrieval, the end-relative failure and the repair with equal (UNI, 224 px) and unequal (MIX, 224/448/672 px) images."""
    A = load("s739_analysis.json") or {}
    L = ["\\begin{tabular}{lrrrrrrrr}", "\\toprule",
         " & \\multicolumn{2}{c}{start-relative} & \\multicolumn{2}{c}{end-relative} & \\multicolumn{2}{c}{literal answers} & \\multicolumn{2}{c}{\\rthree} \\\\",
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}",
         "Model & equal & unequal & equal & unequal & equal & unequal & equal & unequal \\\\", "\\midrule"]
    rows = 0
    for b, nm in MODELS:
        S = (A.get(f"tk739_{b}") or {}).get("summary")
        if not S:
            L.append(f"{nm} & " + " & ".join(["--"] * 8) + " \\\\"); continue
        U, M = S["UNI"], S["MIX"]; rows += 1
        L.append(f"{nm} & {P(U['acc_START'])} & {P(M['acc_START'])} & {P(U['acc_END'])} & {P(M['acc_END'])} & {P(U['lit_END'])} & {P(M['lit_END'])} & {P(U['acc_R3'])} & {P(M['acc_R3'])} \\\\")
    if not rows:
        raise FileNotFoundError("s739 analysis")
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L)


def xfam733():
    """S733: three other families with the paper models' calibrated pipeline.  Procedural scenes (S730 --discover, seven images, k != 4):
    accuracy, literal answers and the discovered index-matching heads' image-attention share on the literal and the target block, with and
    without R³, and the query cosine to the start-relative wording.  Later MMIU (500 end-relative items, never evaluated on these families):
    the unedited model on both wordings and R³."""
    A = load("s733_analysis.json") or {}; S7 = load("s730_analysis.json") or {}
    B = load("s733b_analysis.json") or {}                                         # final configuration under the amended reader rule (DECISIONS 2026-10-11 12:50)
    FAM = (("ivl3_8b", "InternVL3-8B"), ("q25_7b", "Qwen2.5-VL-7B"))                  # LLaVA-OV-7B removed from the paper (user, 2026-10-14)
    L = ["\\begin{tabular}{lrrrrrrrrrrr}", "\\toprule",
         " & & \\multicolumn{7}{c}{procedural scenes} & \\multicolumn{3}{c}{later MMIU} \\\\",
         "\\cmidrule(lr){3-9}\\cmidrule(lr){10-12}",
         " & & & \\multicolumn{4}{c}{end-relative} & \\multicolumn{2}{c}{\\rthree} & & & \\\\",
         "\\cmidrule(lr){4-7}\\cmidrule(lr){8-9}",
         "Model & heads & start & acc. & literal & share & cos & acc. & cos & end & \\rthree & start \\\\", "\\midrule"]
    rows = 0
    for f, nm in FAM:
        sfx = (B.get(f) or {}).get("suffix", "") if f in B else ""
        st = S7.get(f"stw{sfx}_{f}"); out = ((B if sfx else A).get(f) or {}).get("out", {}); lt = out.get("later|mmiu_pos")
        if not st and not lt:
            continue
        rows += 1
        if st:
            s = st["summary"]["k!=4"]
            pc = [str(st["heads"]), P(s["START"]["acc"]), P(s["END"]["acc"]), P(s["END"]["lit_ans"]),
                  f"{P(s['END']['lit'])}/{P(s['END']['tgt'])}", f"{s['END']['cos']:.2f}".lstrip("0"), P(s["R3"]["acc"]), f"{s['R3']['cos']:.2f}".lstrip("0")]
        else:
            pc = ["--"] * 8
        bc = [P(lt["original"]), P(lt["r3"]), P(lt["forward"])] if lt else ["--"] * 3
        L.append(f"{nm} & " + " & ".join(pc + bc) + " \\\\")
    if not rows:
        raise FileNotFoundError("s733 analysis")
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L)


def core745():
    """S745b / S747b: the core causal tests without R³ in InternVL2.5-8B (InternLM2.5), Qwen3-VL-32B and 30B-A3B -- the literal query, the
    start-relative query transplant and the early ordinal-state transplant, for the right (x = j) and a wrong (x = w) address."""
    A = load("s745b_analysis.json") or {}
    cols = [("s745b_ivl25_8b", "InternVL2.5-8B"), ("s747m_q3vl32b", "Qwen3-VL-32B"), ("s747m_q3vl30a3", "Qwen3-VL-30B-A3B")]
    data = []
    for tag, nm in cols:
        d = load(f"s745_{tag}_core.json")
        if not d or tag not in A:
            raise FileNotFoundError(tag)
        rows = d["rows"]; g = lambda k, A_=A[tag]: A_["res"][k]["v"]
        def e(nm_, what, rows=rows):
            tgt = lambda r: r[nm_] if what == "x" else (r[f"named_{nm_}"] if what == "named" else r["k"])
            return float(np.mean([np.mean([r[f"ADDR_{nm_}@{L}"] == tgt(r) for L in (2, 4, 6)]) for r in rows]))
        qk = lambda nm_, rows=rows: float(np.mean([r[f"Q_{nm_}"] == r["k"] for r in rows]))
        data.append((nm, g, e, qk))
    L = ["\\begin{tabular}{l" + "rr" * len(data) + "}", "\\toprule",
         " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{nm}}}" for nm, *_ in data) + " \\\\",
         "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(data))),
         " & " + " & ".join("$j$ & $w$" for _ in data) + " \\\\", "\\midrule"]
    def both(f):
        return " & ".join(f"\\multicolumn{{2}}{{c}}{{{f(g)}}}" for _, g, _, _ in data)
    L.append("start-relative accuracy & " + both(lambda g: P(g('START acc'))) + " \\\\")
    L.append("end-relative accuracy (literal) & " + both(lambda g: f"{P(g('END acc'))} ({P(g('END literal'))})") + " \\\\")
    L.append("heads' share, literal / target & " + both(lambda g: f"{P(g('END lit share'))} / {P(g('END tgt share'))}") + " \\\\")
    L.append("\\midrule")
    L.append("start-relative queries of $x$: on $x$ & " + " & ".join(f"{P(g('Q_j on x'))} & {P(g('Q_w on x'))}" for _, g, _, _ in data) + " \\\\")
    L.append("\\quad on the literal image & " + " & ".join(f"{P(qk('j'))} & {P(qk('w'))}" for _, _, _, qk in data) + " \\\\")
    L.append("ordinal state of $x$, layers 2--6: on $x$ & " + " & ".join(f"{P(e('j', 'x'))} & {P(e('w', 'x'))}" for _, _, e, _ in data) + " \\\\")
    L.append("\\quad on the donor's image / literal & " + " & ".join(f"{P(e('j', 'named'))}/{P(e('j', 'k'))} & {P(e('w', 'named'))}/{P(e('w', 'k'))}" for _, _, e, _ in data) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L)

def others():
    """Behaviour of the added models on the later MMIU split (500 end-relative rewrites, the evaluator's fresh selection): the unedited
    model on the start- and end-relative wording, and the share of end-relative errors on the lure among items whose lure is not the
    answer (exp/lure_latest.py, the definition of FINDINGS §659), with the uniform reference.  S733 runs (final configurations of
    InternVL3-8B, Qwen2.5-VL-7B; lure from results/r3/lure_xfam.json) and S747a (Qwen3-VL-32B, 30B-A3B, Kimi-VL-A3B,
    ERNIE-4.5-VL-28B-A3B when present)."""
    A = load("s733_analysis.json") or {}; B = load("s733b_analysis.json") or {}
    LX = {d["tag"]: d for d in (load("lure_xfam.json") or [])}
    S = load("s747_analysis.json") or {}
    rows = []
    for f, tag, nm, lm in (("ivl3_8b", "x5frn_ivl3_8b", "InternVL3-8B", "Qwen2.5-7B"), ("q25_7b", "x5fr_q25_7b", "Qwen2.5-VL-7B", "Qwen2.5-7B")):
        src = B if f in B else A
        lt = ((src.get(f) or {}).get("out") or {}).get("later|mmiu_pos")
        lx = LX.get(tag)
        if lt and lx:
            rows.append((nm, lm, lt["forward"], lt["original"], lx["lure_share"], lx["chance"]))
    for tag, nm, lm in (("s747_q3vl32b", "Qwen3-VL-32B", "Qwen3-32B"), ("s747_q3vl30a3", "Qwen3-VL-30B-A3B", "Qwen3-30B-A3B"),
                        ("s747_kimivl", "Kimi-VL-A3B", "Moonlight-16B-A3B"), ("s747_ernie", "ERNIE-4.5-VL-28B-A3B", "ERNIE-4.5-21B-A3B")):
        r = (S.get(tag) or {}).get("res")
        if r:
            rows.append((nm, lm, r["start"], r["end"], r["lure_share"], r["lure_chance"]))
    if not rows:
        raise FileNotFoundError("s733 / s747 analyses")
    L = ["\\begin{tabular}{llrrrr}", "\\toprule",
         "Model & language model & start & end & lure & uniform \\\\", "\\midrule"]
    for nm, lm, st, en, lu, ch in rows:
        L.append(f"{nm} & {lm} & {P(st)} & {P(en)} & {P(lu)} & {P(ch)} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L)


def where_end():
    """S748: what the end is.  MATCHER fixed-bank argmax for "the second image from the last" with blocks hidden from the question rows.
    QK only: the answer columns used the space-prefixed readout (S782) and are not shown."""
    A = (load("s748_analysis.json") or {}).get("s748_27b")
    if not A:
        raise FileNotFoundError("s748")
    rows = [("none", "NONE"), ("image 7 (last)", "LAST1"), ("image 3", "MID1"), ("image 1", "FIRST1"), ("images 1, 2", "FIRST2"), ("image 6", "HIDE6")]
    L = ["\\begin{tabular}{lrrr}", "\\toprule",
         "Hidden & block 5 & block 6 & block 7 \\\\", "\\midrule"]
    for nm, c in rows:
        qk = A["qk"][c]
        L.append(f"{nm} & {P(qk['5'])} & {P(qk['6'])} & {P(qk['7'])} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L)


def where_birth():
    """S750: answers on the donor's target after copying the residual after layer L at one group of rows after the ordinal."""
    A = (load("s750_analysis.json") or {}).get("s750_27b")
    if not A:
        raise FileNotFoundError("s750")
    layers = ["14", "18", "22", "26", "30", "34"]
    L = ["\\begin{tabular}{llr" + "r" * len(layers) + "}", "\\toprule",
         "Pair & rows & base & " + " & ".join(f"L{l}" for l in layers) + " \\\\", "\\midrule"]
    for pair, lab in (("5<2", "END(5) $\\leftarrow$ END(2)"), ("3<2", "END(3) $\\leftarrow$ END(2)"), ("5<4", "END(5) $\\leftarrow$ END(4)"), ("2<5", "END(2) $\\leftarrow$ END(5)")):
        for g, gl in (("ALL", "all"), ("QM", "``?''"), ("TAIL", "tail"), ("MID", "frame span")):
            f = A["F"][f"{pair}|{g}"]
            L.append(f"{lab if g == 'ALL' else ''} & {gl} & {P(A['base'][pair]) if g == 'ALL' else ''} & " + " & ".join(P(f[l]) for l in layers) + " \\\\")
        L.append("\\addlinespace")
    L[-1] = "\\bottomrule"; L.append("\\end{tabular}")
    return "\n".join(L)


def where_ops():
    """S751: share of the layer-26 residual effect carried by the outputs of single sublayers of layers 23-26 and their groups."""
    A = (load("s751_analysis.json") or {}).get("s751_27b")
    if not A:
        raise FileNotFoundError("s751")
    arms = ["A23", "A24", "A25", "A26", "M23", "M24", "M25", "M26", "ATT", "MLP", "BOTH", "RES22"]
    names = {"A23": "attn 23", "A24": "DN 24", "A25": "DN 25", "A26": "DN 26", "M23": "MLP 23", "M24": "MLP 24", "M25": "MLP 25", "M26": "MLP 26",
             "ATT": "attn + DN", "MLP": "MLPs", "BOTH": "all eight", "RES22": "residual 22"}
    L = ["\\begin{tabular}{lrr}", "\\toprule", "Copied & END(5) $\\leftarrow$ END(2) & END(3) $\\leftarrow$ END(2) \\\\", "\\midrule"]
    for a in arms:
        L.append(f"{names[a]} & {A['S'][a + '|5<2']:.2f} & {A['S'][a + '|3<2']:.2f} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L)


def where_forms():
    """S755: accuracy (%) of the same end-relative addresses asked in five wordings, d steps before the last image."""
    A = (load("s755_analysis.json") or {}).get("s755_27b")
    if not A:
        raise FileNotFoundError("s755")
    acc = A["acc"]
    rows = [("F1", "the $k$-th image from the last"), ("F5", "the $k$th (digits) image from the last"), ("F2", "the $k$-th-to-last image"),
            ("F3", "the image $d$ places before the last image"), ("F4", "penultimate, antepenultimate"), ("S", "the $(7-d)$-th image from the first")]
    L = ["\\begin{tabular}{lrrrr}", "\\toprule", "Wording & $d=1$ & 2 & 3 & 4 \\\\", "\\midrule"]
    for f, lab in rows:
        cells = [P(acc[f"{f}|{d}"][0]) if f"{f}|{d}" in acc else "--" for d in (1, 2, 3, 4)]
        L.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L)


def where_reads():
    """S758: answers (%) after exchanging what every later row reads from the ordinal or the frame word, by depth band and route."""
    A = (bare("s758_analysis.json") or {}).get("s758_27b")
    if not A:
        raise FileNotFoundError("s758")
    F = lambda a: A["F"][a][0]
    L = ["\\begin{tabular}{lr}", "\\toprule", "Ordinal ``second'' read at & END(5) on image 6 \\\\", "\\midrule"]
    for a, lab in (("K|E5", "none"), ("K:E", "layers 0--22"), ("K:L", "layers 23--26"), ("K:P", "layers 27--63"), ("K:EL", "0--26"),
                   ("K:EP", "0--22 and 27--63"), ("K:LP", "23--63"), ("K:ELP", "all")):
        L.append(f"{lab} & {P(F(a))} \\\\")
    L += ["\\addlinespace", "With layers 0--22, at & \\\\"]
    for a, lab in (("K:E+L23", "attention 23"), ("K:E+L24", "DN 24"), ("K:E+L25", "DN 25"), ("K:E+L26", "DN 26"), ("K:E+Lconv", "DN 24--26 conv only"),
                   ("K:E+Lmem", "DN 24--26 memory only")):
        L.append(f"{lab} & {P(F(a))} \\\\")
    L += ["\\midrule", "Frame ``first'' read at & END(2) / END(5) as START \\\\", "\\midrule"]
    for a2, a5, lab in (("F2|E2", "F5|E5", "none"), ("F2:E", "F5:E", "layers 0--22"), ("F2:L", "F5:L", "layers 23--26"), ("F2:EL", "F5:EL", "0--26"),
                        ("F2:P", "F5:P", "27--63")):
        L.append(f"{lab} & {P(F(a2))} / {P(F(a5))} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L)



def end4_paradox():
    """S769: END(4) answers (%) and the L39 lookup share of image 4 after reading START(4)'s frame word in one band (and the reverse)."""
    A = bare("s769_analysis.json")
    if not A or "N7" not in A or "N8" not in A:
        raise FileNotFoundError("s769")
    a7, a8 = A["N7"], A["N8"]
    acc = lambda a, arm, b: a["acc"][arm][str(b)][0]
    L = ["\\begin{tabular}{llrrrr}", "\\toprule", " & & \\multicolumn{2}{c}{$N=7$} & \\multicolumn{2}{c}{$N=8$} \\\\",
         "Question & frame word read at & image 4 & lookup & image 4 & image 5 \\\\", "\\midrule"]
    rows = (("E4", "END(4)", "``last'' (natural)"), ("F4:P1", "END(4)", "``first'' at 27--34"), ("F4:P1conv", "END(4)", "\\quad DeltaNet conv only"),
            ("F4:P1fa", "END(4)", "\\quad attention 27, 31 only"), ("F4:P23", "END(4)", "``first'' at 35--63"), ("F4:P", "END(4)", "``first'' at 27--63"),
            ("S4", "START(4)", "``first'' (natural)"), ("R4:P1", "START(4)", "``last'' at 27--34"), ("R4:P1conv", "START(4)", "\\quad DeltaNet conv only"),
            ("R4:P", "START(4)", "``last'' at 27--63"))
    for arm, q, lab in rows:
        L.append(f"{q} & {lab} & {P(acc(a7, arm, 4))} & {P(a7['share4_L39'][arm])} & {P(acc(a8, arm, 4))} & {P(acc(a8, arm, 5))} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L)


def readout_parts():
    """S770: share of the frame-driven recovery given (or removed) by each part of the L35-59 attention output, and START late control."""
    A = (load("s770_analysis.json") or {}).get("s770_27b")
    if not A:
        raise FileNotFoundError("s770")
    ph, rho, C6 = A["phi"], A["rho"], A["C6"]
    L = ["\\begin{tabular}{lrrrrrr}", "\\toprule", " & \\multicolumn{4}{c}{END(5): recovery given (\\%)} & removed & START(3)$\\to$6 \\\\",
         "Part & 43--47 & 51--59 & 35--59 & & 35--59 & 51--59 \\\\", "\\midrule"]
    names = (("D", "image distribution"), ("M", "image mass"), ("T", "text read"), ("Tw", "\\quad text weights only"), ("G", "output gate"),
             ("DM", "image distribution + mass"), ("DMT", "+ text read"), ("DMTG", "all (+ gate)"))
    for x, lab in names:
        r = rho.get(x); c = C6["late"].get(f"GIVE_{x}")
        L.append(f"{lab} & {P(ph['mid'][x])} & {P(ph['late'][x])} & {P(ph['all'][x])} & & {P(r) if r is not None else '--'} & {P(c) if c is not None else '--'} \\\\")
    L += ["\\addlinespace", f"query only & {P(A['phi_Q']['mid'])} & {P(A['phi_Q']['late'])} & {P(A['phi_Q']['all'])} & & {P(A['rho_Q'])} & {P(C6['late']['Q_DMT'])} \\\\",
          "\\bottomrule", "\\end{tabular}"]
    return "\n".join(L)


def sender_reader():
    """S771: what reaches the L43-59 readers - whole queries vs the message at a boundary read directly, per pair."""
    A = (load("s771_analysis.json") or {}).get("s771_27b")
    if not A:
        raise FileNotFoundError("s771")
    ph, g6, d6 = A["phi"], A["g6"], A["d6"]
    rep = {k: v[0] for k, v in A["rep"].items()}
    L = ["\\begin{tabular}{lrrrr}", "\\toprule",
         "Readers L43--59 get & frame recovery & END(2)$\\to$END(5) & END(5) errors in END(2) & END(2) lost to START(2) \\\\", "\\midrule"]
    L.append(f"whole donor query & {P(ph['A_QTOT@ML'])} & {P(g6['B_QTOT@ML'])} & {P(rep['C_QTOT@ML'])} & {P(d6['Nc_QTOT@ML'])} \\\\")
    for b, lab in (("M26", "message after layer 26"), ("M34", "message after layer 34"), ("M42", "message after layer 42")):
        a = ph.get(f"A_DIR_{b}")
        L.append(f"{lab} & {P(a) if a is not None else '--'} & {P(g6[f'B_DIR_{b}'])} & {P(rep[f'C_DIR_{b}'])} & {P(d6[f'Nc_DIR_{b}'])} \\\\")
    for w, lab in (("DN", "layers 27--34 DeltaNet only"), ("FA", "layers 27--34 attention only"), ("MLP", "layers 27--34 MLP only")):
        L.append(f"{lab} & {P(ph[f'A_DIR_P1{w}'])} & {P(g6[f'B_DIR_P1{w}'])} & {P(rep[f'C_DIR_P1{w}'])} & {P(d6[f'Nc_DIR_P1{w}'])} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L)


def faith():
    """S772: the circuit's predictions in four cells (image count x wording), 80 new scenes each, k = 2..6."""
    A = bare("s772_analysis.json")
    tags = [("s772_27b7A", "7, last"), ("s772_27b8A", "8, last"), ("s772_27b7B", "7, end"), ("s772_27b8B", "8, end")]
    if not A or any(t not in A for t, _ in tags):
        raise FileNotFoundError("s772")
    L = ["\\begin{tabular}{l" + "r" * len(tags) + "}", "\\toprule", "$N$, wording & " + " & ".join(lab for _, lab in tags) + " \\\\", "\\midrule"]
    get = lambda f: " & ".join(f(A[t]) for t, _ in tags)
    L.append("frame switch, far $k$ (+pts) & " + get(lambda c: P(c["frame_far_gain"])) + " \\\\")
    L.append("near $k$: frame at 27--34 / 27--47 & " + get(lambda c: f"{P(c['per_k']['2']['F_P1'])} / {P(c['per_k']['2']['F_P12'])}") + " \\\\")
    L.append("END errors reproduced by its queries & " + get(lambda c: P(c["repro"]["Q_ES"])) + " \\\\")
    L.append("\\quad chance (START($k'$) gives it) & " + get(lambda c: P(c["repro"]["natural_START_kp"])) + " \\\\")
    L.append("START($k'$) queries fix END($k$) (+pts) & " + get(lambda c: P(c["fix_gain"])) + " \\\\")
    L.append("END address after layer 42: $k=2$ / far & " + get(lambda c: f"{P(c['addr42']['near_k2'])} / {P(c['addr42']['far'])}") + " \\\\")
    L.append("mean position error: $k=2$ / $k=6$ & " + get(lambda c: f"{c['rule']['mae']['2']:.2f} / {c['rule']['mae']['6']:.2f}") + " \\\\")
    L.append("late heads' top option = answer: END / START & " + get(lambda c: f"{P(c['reader_top_option_is_answer']['E'])} / {P(c['reader_top_option_is_answer']['S'])}") + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L)


def validation():
    """S772 (faithfulness: 80 new scenes in each of four cells), S780 (the repair bridge: 80 new scenes at N = 7 and 8) and S781 (same-scene
    positive control: 40 at N = 7 and 8), one table.  Columns: N and wording.  '--': not measured in that cell."""
    F = bare("s772_analysis.json"); B = bare("s780_analysis.json"); C = bare("s781_analysis.json")
    if not F or not B or not C:
        raise FileNotFoundError("s772 / s780 / s781")
    cells = [("7A", "7, last"), ("8A", "8, last"), ("7B", "7, end"), ("8B", "8, end")]
    f = lambda c: F.get(f"s772_27b{c}"); b = lambda c: B.get(f"s780_27b{c[0]}") if c.endswith("A") else None
    cc = lambda c: C.get(f"s781_27b{c[0]}") if c.endswith("A") else None
    def row(lab, fn, src):
        vals = []
        for c, _ in cells:
            d = src(c)
            vals.append(fn(d) if d else "--")
        return f"{lab} & " + " & ".join(vals) + " \\\\"
    L = ["\\begin{tabular}{l" + "r" * len(cells) + "}", "\\toprule", "$N$, wording & " + " & ".join(l for _, l in cells) + " \\\\", "\\midrule",
         "\\multicolumn{5}{l}{\\textit{Frame read, address and readers}} \\\\"]
    L.append(row("frame at 27--34 sends far END to the literal (+pts)", lambda d: P(d["frame_far_gain"]), f))
    L.append(row("END(2) turned start-relative: frame at 27--34 / 27--47 (\\%)", lambda d: f"{P(d['per_k']['2']['F_P1'])} / {P(d['per_k']['2']['F_P12'])}", f))
    L.append(row("START($k'$) queries fix END($k$) (+pts)", lambda d: P(d["fix_gain"]), f))
    L.append(row("END($k$) queries carry its errors (chance)", lambda d: f"{P(d['repro']['Q_ES'])} ({P(d['repro']['natural_START_kp'])})", f))
    L.append(row("END address after layer 42: $k=2$ / far (\\%)", lambda d: f"{P(d['addr42']['near_k2'])} / {P(d['addr42']['far'])}", f))
    L.append(row("mean position error: $k=2$ / $k=6$", lambda d: f"{d['rule']['mae']['2']:.2f} / {d['rule']['mae']['6']:.2f}", f))
    L.append(row("late heads' top option is the answer: END / START", lambda d: f"{P(d['reader_top_option_is_answer']['E'])} / {P(d['reader_top_option_is_answer']['S'])}", f))
    L += ["\\addlinespace", "\\multicolumn{5}{l}{\\textit{Readout locked to a position, and the repair}} \\\\"]
    L.append(row("answer follows a new picture at the address: \\rthree{}", lambda d: P(d["visual_r3"]["j"]), b))
    L.append(row("\\quad natural END(2)", lambda d: P(d["visual_e2"]["j"]), b))
    L.append(row("\\quad natural END(5), its answered / its target image", lambda d: f"{P(d['visual_end']['a'])} / {P(d['visual_end']['j'])}", b))
    L.append(row("reversing the menu changes the answer: END(5) / \\rthree{}", lambda d: f"{P(d['menu_reversal_change']['END'])} / {P(d['menu_reversal_change']['R3'])}", b))
    L.append(row("\\rthree{} queries at 35--59: removed / given (pts of the gain)", lambda d: f"{P(d['r3_removed_ALL59'])} / {P(d['r3_suf'])} of {P(d['i0']['G'])}", b))
    L.append(row("\\rthree{} queries made in a rearranged scene: its address (+pts)", lambda d: f"{P(d['gains']['ALL59']['R3same_B'])} / {P(d['gains']['ALL59']['R3same_B2'])}", cc))
    L.append(row("\\rthree{} queries carried from the original: its object / address (+pts)", lambda d: f"{P(d['gains']['ALL59']['R3A_B_cont'])} / {P(d['gains']['ALL59']['R3A_B_addr'])}", cc))
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L)


def skills():
    """S643 / S661 (S784 when present: the same items re-read from the bare tokens): each skill alone and composed, five models.  Accuracy (%) and,
    for rows with an ordinal or number, the share of answers on the start-relative image in parentheses; those rows exclude items whose
    start-relative and correct images coincide.  n/a: the model began a written derivation instead of a number in most items."""
    M = [("4b", "VL-4B"), ("8b", "VL-8B"), ("35_4b", "3.5-4B"), ("35_9b", "3.5-9B"), ("27b", "3.5-27B")]
    def rows(m):                                  # S782 registry: content answers only from the bare-token re-runs (S784); the
        out = []                                  # old space-token runs supply only the generated-number rows and Start (agreement .96-1.00)
        for new, old in ((f"s643_s784_{m}_relimg.json", f"s643_ri2_{m}_relimg.json"), (f"s643_s784x_{m}_relimg.json", f"s643_rx_{m}_relimg.json")):
            d = load(new)
            if d:
                out += d.get("rows", [])
            else:
                out += [r for r in ((load(old) or {}).get("rows", [])) if r["op"] in ("Start", "IdxX", "IdxEnd")]
        return out
    RW = {m: rows(m) for m, _ in M}
    if not any(RW.values()):
        raise FileNotFoundError("s643")
    lab = [("Start", "What is in the third image?"), ("IdxX", "Which image shows the X? (number)"),
           ("IdxEnd", "Which image is third from the end? (number)"), ("End", "third image from the end"),
           ("EndC", "image two images before the last image"), ("RelC+", "image two images after the X"), ("RelC-", "image two images before the X"),
           ("RelO+", "second image after the X"), ("RelO-", "second image before the X"),
           ("RelOx+", "start at the X, count forward: third image reached"), ("RelOx-", "start at the X, count backward: third image reached")]
    L = ["\\begin{tabular}{l" + "c" * len(M) + "}", "\\toprule", "Question & " + " & ".join(n for _, n in M) + " \\\\", "\\midrule"]
    for op, text in lab:
        cells = []
        for m, _ in M:
            R_ = [r for r in RW[m] if r["op"] == op]
            if not R_:
                cells.append("\\nan"); continue
            if op not in ("Start", "IdxX"):
                R_ = [r for r in R_ if r.get("lab_copy") != r["target"]]
            if op.startswith("Idx") and np.mean([r["kind"] == "invalid" for r in R_]) > .5:
                cells.append("n/a"); continue
            acc = 100 * np.mean([r["kind"] == "correct" for r in R_])
            cells.append(f"{acc:.0f}" if op in ("Start", "IdxX") else f"{acc:.0f} ({100 * np.mean([r['kind'] == 'copy' for r in R_]):.0f})")
        L.append(f"\\textit{{{text}}} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L)


def qident():
    """Appendix C (re-analysis of the S705a captures, FINDINGS §848 · §849): which start-relative query the end-relative query of the
    index-matching heads resembles, per model and k (N = 7, held-out scenes, answer row).  S790: share of (scene, head) pairs closer by cosine
    to START(k) (copied) than to START(N+1-k) (composed), and the two cosines; S790b: the start-relative question whose block distribution is
    nearest in Jensen-Shannon divergence -- the copied ordinal, the composed image, or another block."""
    A, B = load("s790_qident.json"), load("s790b_selnn.json")
    if not A or not B:
        raise FileNotFoundError("s790 / s790b missing")
    MD = [("qa_4b", "Qwen3.5-4B"), ("qa_9b", "Qwen3.5-9B"), ("qa_27b", "Qwen3.5-27B")]
    ks = [str(k) for k in range(1, 8)]
    L = ["\\begin{tabular}{@{}l" + "r" * len(ks) + "@{}}", "\\toprule", "$k$ & " + " & ".join(ks) + " \\\\", "\\midrule"]
    for i, (t, nm) in enumerate(MD):
        e, b = A[t]["end"], B[t]
        L.append(f"\\multicolumn{{{1 + len(ks)}}}{{@{{}}l}}{{\\textit{{{nm}}}}} \\\\")
        cell = lambda k, f: "--" if k == "4" else f()
        L.append("closer to copied (\\%) & " + " & ".join(cell(k, lambda k=k: P(e[k]["closer_to_copy"])) for k in ks) + " \\\\")
        L.append("cosine, copied & " + " & ".join(f"{e[k]['cos_copy']:.2f}" for k in ks) + " \\\\")
        L.append("cosine, composed & " + " & ".join(cell(k, lambda k=k: f"{e[k]['cos_comp']:.2f}") for k in ks) + " \\\\")
        L.append("nearest: copied (\\%) & " + " & ".join(cell(k, lambda k=k: P(b[k]["copy"])) for k in ks) + " \\\\")
        L.append("nearest: composed (\\%) & " + " & ".join(cell(k, lambda k=k: P(b[k]["comp"])) for k in ks) + " \\\\")
        L.append("nearest: other (\\%) & " + " & ".join(P(b[k]["other"]) for k in ks) + " \\\\")
        if i < len(MD) - 1:
            L.append("\\addlinespace")
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L)


def rebind_byk():
    """Appendix F (S786 / S794, bare or two-form readout): Table 1 by distance from the end -- accuracy on the target and, in parentheses, the
    share of answers on the literal image, per model, arm and k (N = 7, 30 scenes)."""
    KEY = {b: (f"s794_{b}" if b == "35_4b" else f"s786_{b}") for b, _ in MODELS}
    rows = {}
    for b, nm in MODELS:
        d = load(f"s730_{KEY[b]}_stagewise.json")
        if d:
            rows[b] = d["rows"]
    if not rows:
        raise FileNotFoundError("s786 missing")
    ks = [2, 3, 4, 5, 6]
    L = ["\\begin{tabular}{@{}l" + "r" * len(ks) + "@{}}", "\\toprule", "$k$ (target $8-k$) & " + " & ".join(f"{k} ({8 - k})" for k in ks) + " \\\\", "\\midrule"]
    MD = [(b, nm) for b, nm in MODELS if b in rows]
    for i, (b, nm) in enumerate(MD):
        R_ = rows[b]
        L.append(f"\\multicolumn{{{1 + len(ks)}}}{{@{{}}l}}{{\\textit{{{nm}}}}} \\\\")
        for c, lab in (("START", "start-relative"), ("END", "end-relative"), ("ADDR", "\\quad + address"), ("FRAME", "\\quad + frame"), ("R3", "\\quad + both")):
            cells = []
            for k in ks:
                rk = [r for r in R_ if r["k"] == k]
                if not rk:
                    cells.append("--"); continue
                acc = np.mean([r[f"land_{c}"] == r["j"] for r in rk])
                lit = np.mean([r[f"land_{c}"] == r["k"] for r in rk])
                cells.append(P(acc) if (k == 4 or c in ("START", "R3")) else f"{P(acc)} ({P(lit)})")
            L.append(f"{lab} & " + " & ".join(cells) + " \\\\")
        if i < len(MD) - 1:
            L.append("\\addlinespace")
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L)


def rebind_main2():
    """Table 1 (review revision): accuracy on the target only -- the 2 x 2 of address and frame writes for k != 4 and the k = 4 cell (S786;
    Qwen3.5-4B from S794 with the two-form readout when present).  Query cosines go to Appendix F (cos_table), photographs to the text."""
    A = load("s730_analysis.json") or {}
    KEY = {b: (f"s794_{b}" if b == "35_4b" else f"s786_{b}") for b, _ in MODELS}
    MODELS_ = [(b, nm) for b, nm in MODELS if KEY[b] in A]
    if not MODELS_:
        raise FileNotFoundError("s786 / s794 missing")
    g = lambda b, c, sub="k!=4": A[KEY[b]]["summary"][sub][c]
    short = lambda nm: nm.replace("Qwen3-VL-", "VL-").replace("Qwen3.5-", "3.5-")
    W = 1 + len(MODELS_)
    L = ["\\begin{tabular}{@{}l" + "r" * len(MODELS_) + "@{}}", "\\toprule",
         " & " + " & ".join(short(nm) for _, nm in MODELS_) + " \\\\", "\\midrule",
         f"\\multicolumn{{{W}}}{{@{{}}l}}{{\\textit{{The address must change ($k\\neq4$)}}}} \\\\"]
    for c, lab in (("END", "end-relative, unedited"), ("ADDR", "\\quad + address"), ("FRAME", "\\quad + frame"), ("R3", "\\quad + both (\\rthree)"), ("START", "start-relative wording")):
        L.append(f"{lab} & " + " & ".join(P(g(b, c)["acc"]) for b, _ in MODELS_) + " \\\\")
    L += ["\\addlinespace", f"\\multicolumn{{{W}}}{{@{{}}l}}{{\\textit{{The address needs no change ($k=4$)}}}} \\\\"]
    for c, lab in (("END", "end-relative, unedited"), ("FRAME", "\\quad + frame")):
        L.append(f"{lab} & " + " & ".join(P(g(b, c, "k=4")["acc"]) for b, _ in MODELS_) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L)


def cos_table():
    """Appendix F: query cosine of the index-matching heads at the answer position to the start-relative wording of the same target (S786)."""
    A = load("s730_analysis.json") or {}
    KEY = {b: (f"s794_{b}" if b == "35_4b" else f"s786_{b}") for b, _ in MODELS}
    MODELS_ = [(b, nm) for b, nm in MODELS if KEY[b] in A]
    g = lambda b, c: A[KEY[b]]["summary"]["k!=4"][c]
    short = lambda nm: nm.replace("Qwen3-VL-", "VL-").replace("Qwen3.5-", "3.5-")
    L = ["\\begin{tabular}{@{}l" + "r" * len(MODELS_) + "@{}}", "\\toprule", " & " + " & ".join(short(nm) for _, nm in MODELS_) + " \\\\", "\\midrule"]
    for c, lab in (("END", "end-relative, unedited"), ("ADDR", "\\quad + address"), ("FRAME", "\\quad + frame"), ("R3", "\\quad + both (\\rthree)")):
        L.append(f"{lab} & " + " & ".join(f"{g(b, c)['cos']:.2f}" for b, _ in MODELS_) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L)


def bench2():
    """Table 2 (review revision, one column): held-out MMIU and MuirBench, start-relative wording, end-relative wording and the end-relative
    wording with R3, and on MMIU the paired difference R3 - Start over the same items (item bootstrap, 4,000 draws).  Later split and cloze: text
    and Appendix G (bench_tasks, bench_paired)."""
    tags = bench_tags(); rng = np.random.default_rng(0)
    p1 = lambda x: f"{100 * x:.1f}"
    short = lambda nm: nm.replace("Qwen3-VL-", "VL-").replace("Qwen3.5-", "3.5-")
    L = ["\\begin{tabular}{@{}lrrrlrrr@{}}", "\\toprule",
         " & \\multicolumn{4}{c}{MMIU held-out} & \\multicolumn{3}{c}{MuirBench held-out} \\\\",
         "\\cmidrule(lr){2-5}\\cmidrule(l){6-8}",
         "Model & Start & End & +\\rthree & \\rthree$-$Start & Start & End & +\\rthree \\\\", "\\midrule"]
    for b, nm in MODELS:
        d = load(f"/vlm/results/s582_evalrepair_{tags[b]['sealed']}.json")
        if not d:
            continue
        cells = []
        for suite in ("mmiu_pos", "muir"):
            rr = [r for r in d["rows"] if r["suite"] == suite]
            m = lambda k: float(np.mean([r[k] == r["gold"] for r in rr]))
            cells += [p1(m("F|base")), p1(m("R|base")), "\\textbf{" + p1(m("R|rep")) + "}"]
            if suite == "mmiu_pos":
                x = np.array([(r["R|rep"] == r["gold"]) - (r["F|base"] == r["gold"]) for r in rr], float)
                bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(4000)]
                cells.append(f"{100 * x.mean():+.1f} [{100 * np.percentile(bs, 2.5):+.1f}, {100 * np.percentile(bs, 97.5):+.1f}]")
        L.append(f"{short(nm)} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L)


def readout_table():
    """Appendix A (W5 of the 2026-10-14 review): agreement of the menu readout with greedy generation, per run behind a reported accuracy.
    t1: share of readouts whose vocabulary-wide top-1 token is a menu word in the scored form (then the readout equals greedy generation's first
    word); S784: share of answers equal to the parsed free generation.  S780 re-read (S783a): 6,480 / 6,480 recorded in FINDINGS §847."""
    cols = [("4b", "VL-4B"), ("8b", "VL-8B"), ("35_4b", "3.5-4B"), ("35_9b", "3.5-9B"), ("27b", "3.5-27B")]
    pct = lambda x: "--" if x is None else f"{100 * x:.0f}"
    rows = []
    t = {b: (load(f"s730_s786_{b}_stagewise.json") or {}).get("t1_menu") for b, _ in cols}
    s794 = (load("s730_s794_35_4b_stagewise.json") or {}).get("t1_menu")
    rows.append(("Table 1 (S786)", {b: (pct(t[b]) + ("$^\\dagger$" if b == "35_4b" else "")) for b, _ in cols}))
    if s794 is not None:
        rows.append(("Table 1, two-form readout (S794)", {"35_4b": pct(s794)}))
    one = lambda f: (load(f) or {}).get("t1_menu")
    e7, e8 = one("s769_s783e_27b7_end4.json"), one("s769_s783e_27b8_end4.json")
    rows.append(("Figure 5 (S769, re-read)", {"27b": pct(min(x for x in (e7, e8) if x is not None)) if (e7 or e8) else "--"}))
    rows.append(("Figure 6b (S780, re-read)", {"27b": "100"}))
    rows.append(("Section 6 (S793)", {"27b": pct(one("s793_s793_27b_transplant.json"))}))
    rows.append(("Appendix F.5 (S791)", {"27b": pct(one("s791_s791_27b_chain.json"))}))
    agree = {}
    for b, _ in cols:
        d = load(f"s643_s784_{b}_relimg.json")
        if d:
            rr = [r for r in d["rows"] if r.get("ans_free") is not None]
            agree[b] = pct(float(np.mean([r["ans"] == r["ans_free"] for r in rr]))) if rr else "--"
    rows.append(("Appendix B (S784), vs.\\ free generation", agree))
    L = ["\\begin{tabular}{@{}l" + "r" * len(cols) + "@{}}", "\\toprule", "Run & " + " & ".join(n for _, n in cols) + " \\\\", "\\midrule"]
    for lab, v in rows:
        L.append(f"{lab} & " + " & ".join(v.get(b, "") for b, _ in cols) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L)

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for name, fn in (("read_choice", read_choice), ("stagewise", stagewise), ("reader", reader), ("lexicon", lexicon), ("bench", bench), ("lure", lure), ("stagewise_main", stagewise_main), ("bench_tasks", bench_tasks), ("bench_paired", bench_paired), ("generality", generality), ("frame736", frame736), ("deref673", deref673), ("tokens739", tokens739), ("xfam733", xfam733), ("core745", core745), ("others", others), ("where_end", where_end), ("where_birth", where_birth), ("where_ops", where_ops), ("where_forms", where_forms), ("where_reads", where_reads),
                     ("end4_paradox", end4_paradox), ("readout_parts", readout_parts), ("sender_reader", sender_reader), ("faith", faith), ("validation", validation), ("skills", skills), ("rebind_main", rebind_main), ("preserve", preserve), ("qident", qident), ("rebind_byk", rebind_byk), ("rebind_main2", rebind_main2), ("cos_table", cos_table), ("bench2", bench2), ("readout_table", readout_table)):
        try:
            txt = fn()
        except Exception as e:                                                      # a missing piece leaves the table out, loudly
            print(f"{name}: skipped ({e!r})"); continue
        open(f"{OUT}/{name}.tex", "w").write(txt + "\n")
        print(f"{name}: ok")
