"""Appendix tables for the mechanism and repair sections, generated from saved records only (no model calls).
Writes LaTeX tabular bodies to /vlm/results/figures/tables_mech/: visibility.tex, select.tex, rebind.tex, mediation.tex, boundary.tex,
validation.tex.  Missing inputs are skipped (rows marked --).
Usage: python figs/make_mech_tables.py
"""
import json, os
from collections import defaultdict
import numpy as np

R = "/vlm/results/r3"
OUT = "/vlm/results/figures/tables_mech"
P = lambda x: "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.0f}"


def load(f):
    p = f"{R}/{f}"
    return json.load(open(p)) if os.path.exists(p) else None


def visibility():
    a = load("s703_analysis.json")["me_27b"]; e6 = load("s706_analysis.json")["ap_27b"]["pooled"]; e7 = load("s707_analysis.json")["er_27b"]["pooled"]
    L = ["\\begin{tabular}{lrr}", "\\toprule", "Last $m$ images hidden from & $k=2$ & $k=3$ \\\\", "\\midrule"]
    for key, lab in (("FA_ALL", "full attention, all layers"), ("FA_4347", "full attention, two layers (68--75\\% depth)"),
                     ("DN_ALL", "recurrent memory, all layers"), ("ref", "images removed (reference)"), ("unedited", "no edit")):
        L.append(f"{lab} & {P(a['2']['means'][key])} & {P(a['3']['means'][key])} \\\\")
    L += ["\\midrule", "\\multicolumn{3}{l}{\\textit{Heads hidden at the two end-reading layers ($k\\in\\{2,3\\}$, pooled)}} \\\\"]
    for key, lab, src in (("ALL_4347", "all heads", e7), ("MAT_4347", "index-matching heads only", e7), ("NONMAT_4347", "other heads only", e7),
                          ("ANC", "end-attending heads only (11)", e6), ("ANC_RAND", "random heads (11)", e6)):
        L.append(f"{lab} & \\multicolumn{{2}}{{r}}{{{P(src[key]['Tm'])}}} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L)


def select():
    cols = [("27B", "s705b_analysis.json", "b5_27b"), ("9B", "s720_analysis.json", "sel_9b"), ("4B", "s720_analysis.json", "sel_4b")]
    data = {}
    for name, f, tag in cols:
        d = load(f)
        data[name] = d.get(tag) if d else None
    arms = [("PTR", "image-attention pattern, all heads"), ("PTR_C", "within-image distribution only"), ("PTR_M", "pattern, index-matching heads"),
            ("Q_ONLY", "query of index-matching heads"), ("G_ONLY", "output gate of index-matching heads"), ("QG", "query and gate")]
    L = ["\\begin{tabular}{l" + "r" * len(cols) + "}", "\\toprule", "Start-relative part given to failed end-relative runs & " + " & ".join(c for c, _, _ in cols) + " \\\\", "\\midrule"]
    for k, lab in arms:
        dead = lambda c: k.startswith("PTR") and data[c] is not None and data[c].get("I0") is False      # pattern arms are read only where the pattern edit validates
        L.append(lab + " & " + " & ".join(("n.v." if dead(c) else P(data[c]["P_fail"][k])) if data[c] else "--" for c, _, _ in cols) + " \\\\")
    L.append("failed items & " + " & ".join(str(data[c]["n_fail"]) if data[c] else "--" for c, _, _ in cols) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L)


def rebind():
    d = load("s708_rb_27b_rebind.json")
    acc = defaultdict(lambda: defaultdict(list))
    for row in d["rows"]:
        for k, c in row["k"].items():
            for arm, x in c["land"].items():
                acc[(arm, int(k))]["t"].append(x == c["r"]); acc[(arm, int(k))]["k"].append(x == int(k)); acc[(arm, int(k))]["w"].append(x == c["w"])
    ks = sorted({k for _, k in acc})
    arms = [("BASE", "t", "no write"), ("ORD@3", "t", "ordinal only (target)"), ("ORD@3", "k", "ordinal only (literal)"), ("FR@3", "k", "frame only (literal)"),
            ("ORD_FR@3", "t", "ordinal + frame (target)"), ("WRONG@3", "w", "wrong address + frame (written)"), ("RAND_FR@3", "k", "random + frame (literal)")]
    L = ["\\begin{tabular}{l" + "r" * len(ks) + "}", "\\toprule", "Write at layer 3 & " + " & ".join(f"$k={k}$" for k in ks) + " \\\\", "\\midrule"]
    for arm, key, lab in arms:
        L.append(lab + " & " + " & ".join(P(float(np.mean(acc[(arm, k)][key]))) for k in ks) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L)


def mediation():
    d = load("s710_mq_27b_mediation.json")
    g = defaultdict(list)
    for r in d["rows"]:
        if r["acted"] and r["land"]["R3"] == r["j"] and r["land"]["NATIVE"] != r["j"]:
            g[(r["N"], r["k"])].append(r)
    arms = ("R3_QBACK", "R3_GBACK", "R3_RBACK", "END_QR3", "END_GR3", "END_RR3")
    L = ["\\begin{tabular}{lr" + "r" * len(arms) + "}", "\\toprule",
         "Cell & $n$ & \\multicolumn{3}{c}{repaired run, put back} & \\multicolumn{3}{c}{original run, given} \\\\",
         " & & query & gate & random & query & gate & random \\\\", "\\midrule"]
    allr = []
    for (N, k), rs in sorted(g.items()):
        allr += rs
        L.append(f"$N={N}$, $k={k}$ & {len(rs)} & " + " & ".join(P(float(np.mean([r['land'][a] == r['j'] for r in rs]))) for a in arms) + " \\\\")
    L.append("\\midrule")
    L.append(f"all & {len(allr)} & " + " & ".join(P(float(np.mean([r['land'][a] == r['j'] for r in allr]))) for a in arms) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L)


def boundary():
    d = load("s714_analysis.json")["kn_27b"]
    L = ["\\begin{tabular}{lrrr}", "\\toprule", "Condition & END $k=N$ $\\to$ 1 & START first $\\to$ 1 & END $k=N-1$ $\\to$ 2 \\\\", "\\midrule"]
    for key, lab in (("BASE", "no edit"), ("NOCOUNT", "count sentence removed"), ("COUNT_UP", "count sentence states $N+1$"),
                     ("HIDE1_ALL", "image 1 hidden, all full-attention layers"), ("HIDE1_EARLY", "image 1 hidden, 55--75\\% depth"), ("HIDE1_LATE", "image 1 hidden, from 81\\% depth")):
        e = d["END_N"].get(key); s = d["START1"].get(key); e1 = d["END_N1"].get(key)
        L.append(f"{lab} & {P(e['p1']) if e else '--'} & {P(s['p1']) if s else '--'} & {P(e1['p2']) if e1 else '--'} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L)


def validation():
    v = load("s711_v2_27b_eval.json"); g = load("s712_gen_27b_general.json")
    L = ["\\begin{tabular}{lrrr}", "\\toprule", "Scenes & original & \\rthree{} & correct address \\\\", "\\midrule"]
    if v:
        rs = v["rows"]
        for N in (7, 5):
            x = [r for r in rs if r["N"] == N]
            L.append(f"$N={N}$ (seen sizes) & {P(np.mean([r['land']['NATIVE'] == r['j'] for r in x]))} & {P(np.mean([r['land']['R3V2'] == r['j'] for r in x]))} & {P(np.mean([r['land']['O_ORD_FR'] == r['j'] for r in x]))} \\\\")
    if g:
        E = [r for r in g["rows"] if r["kind"] == "END"]
        for N in sorted({r["N"] for r in E}):
            x = [r for r in E if r["N"] == N]
            L.append(f"$N={N}$ (new size) & {P(np.mean([r['land']['NATIVE'] == r['j'] for r in x]))} & {P(np.mean([r['land']['R3V2'] == r['j'] for r in x]))} & {P(np.mean([r['land']['O_ORD_FR'] == r['j'] for r in x]))} \\\\")
        S = [r for r in g["rows"] if r["kind"] == "START"]
        L.append(f"start-relative (control) & {P(np.mean([r['land']['NATIVE'] == r['j'] for r in S]))} & {P(np.mean([r['land']['R3V2'] == r['j'] for r in S]))} & -- \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for f in (visibility, select, rebind, mediation, boundary, validation):
        try:
            open(f"{OUT}/{f.__name__}.tex", "w").write(f() + "\n"); print("ok", f.__name__)
        except Exception as e:
            print("FAIL", f.__name__, repr(e))


def qk():
    import sys as _s
    _s.path.insert(0, "/vlm/exp/figs")
    import make_c_figs as F
    rows = []
    for name, tag, hf in (("Qwen3.5-4B", "qa_4b", f"{R}/s720_4b_heads.json"), ("Qwen3.5-9B", "qa_9b", f"{R}/s720_9b_heads.json"),
                          ("Qwen3.5-27B", "qa_27b", f"{R}/s705a_qa_27b_matchers.json")):
        heads = [tuple(x) for x in json.load(open(hf))["matchers"]]
        S, E = F.qk_tables(tag, heads, ri=2)
        N = 7
        rows.append((name, len(heads), float(np.mean([S[t, t] for t in range(N)])), [E[k - 1, N - k] for k in range(1, N + 1)], [E[k - 1, k - 1] for k in range(1, N + 1)]))
    L = ["\\begin{tabular}{lrr" + "r" * 7 + "}", "\\toprule", "Model & heads & START & " + " & ".join(f"$k={k}$" for k in range(1, 8)) + " \\\\", "\\midrule"]
    for name, n, sd, tg, lt in rows:
        L.append(f"{name} (target) & {n} & {P(sd)} & " + " & ".join(P(x) for x in tg) + " \\\\")
        L.append(f"\\quad literal block & & & " + " & ".join(P(x) for x in lt) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L)


if __name__ == "__main__":
    try:
        open(f"{OUT}/qk.tex", "w").write(qk() + "\n"); print("ok qk")
    except Exception as e:
        print("FAIL qk", repr(e))
