"""v2 tables, generated from saved records only through exp/figs/make_r3_tables.py (no hand-typed numbers).
Writes research/ces/paper_v2_20261017/tables/bench.tex (table float around make_r3_tables.bench()).  Usage (container): python make_v2_tables.py"""
import os, sys
sys.path.insert(0, "/vlm/exp/figs")
import make_r3_tables as T
OUT = "/vlm/research/ces/paper_v2_20261017/tables"; os.makedirs(OUT, exist_ok=True); os.makedirs(T.OUT, exist_ok=True)
T.bench()                                                                         # refreshes bench_values.json (the shared source of the numbers)
V = T.bench_values()
SETS = [("MMIU cloze", "MMIU visual cloze"), ("MMIU held-out", "MMIU held-out"), ("MuirBench held-out", "MuirBench held-out"), ("MMIU later", "MMIU later")]
p1 = lambda x: "--" if x is None else f"{100 * x:.1f}"
L = ["\\begin{tabular}{l" + "rrr" * len(SETS) + "}", "\\toprule",
     " & \\multicolumn{3}{c}{native end-relative} & \\multicolumn{9}{c}{rewritten to end-relative (controlled diagnostic)} \\\\",
     "\\cmidrule(lr){2-4}\\cmidrule(lr){5-13}",
     " & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{lab}}}" for _, lab in SETS) + " \\\\",
     "".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(SETS))),
     "Model & " + " & ".join("Start & End & +\\rthree" for _ in SETS) + " \\\\", "\\midrule"]
for bb, nm in T.MODELS:
    cells = []
    for key, _ in SETS:
        v = V.get(f"{bb}|{key}")
        cells += [p1(v["F"]), p1(v["R"]), "\\textbf{" + p1(v["rep"]) + "}"] if v else ["--"] * 3
    L.append(f"{nm} & " + " & ".join(cells) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
body = "\n".join(L)
tex = ("\\begin{table*}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{4.2pt}\n" + body + "\n\\caption{\\textbf{\\rthree{} closes the end-relative gap on benchmarks.} "
       "Accuracy (\\%) on the start-relative (Start) and end-relative (End) wordings of the same items, and on End with \\rthree{}. Visual-cloze "
       "options are end-relative in MMIU itself (200 development items; Start is their start-relative rewrite); the other sets rewrite "
       "start-relative position items to end-relative form. Images, options and answers are fixed; only the image references change.}\n"
       "\\label{tab:bench}\n\\end{table*}\n")
open(f"{OUT}/bench.tex", "w").write(tex); print(body)


# Table 2: registered experiments and verdicts, read from the saved analysis records (no hand-typed numbers).
import json
R = "/vlm/results/r3"
J = lambda f: json.load(open(f"{R}/{f}"))
pct = lambda x: str(int(100 * x + .5 + 1e-9))
a827 = J("s827_analysis.json")["s827_27b"]["cases"]; a828 = J("s828_analysis.json"); a829 = J("s829_analysis.json")["cases"]
a830 = J("s830_analysis.json")["cases"]; a831 = J("s831_analysis.json")["cases"]; a831b = J("s831_analysis_s831b_27b.json")["cases"]
a832 = J("s832_analysis.json"); a833 = J("s833_analysis.json")["cases"]
wr = lambda x: f"{abs(x):.2f}" if abs(x) < .005 else f"{x:.2f}"
rows = [
 ("Wrong or weakened address?", "neighbour answers whose late address left the target: "
  f"{pct(a827['N7k4']['M1']['moved'])}\% / {pct(a827['N9k5']['M1']['moved'])}\%; removing the END address change: "
  f"+{pct(a827['N7k4']['arms']['A_FULL']['dPk'])} / +{pct(a827['N9k5']['arms']['A_FULL']['dPk'])} pts"),
 ("Where does the error travel?", "both carriers restored: "
  f"{pct(a828['cases']['N7k4j4']['arms']['BA_FULL']['P_j'])}\% / {pct(a828['cases']['N9k5j5']['arms']['BA_FULL']['P_j'])}\% on target; "
  f"frozen two-route law within .15: {pct(a828['PL1']['share'])}\%"),
 ("Early or late address?", "late layers restored: "
  f"{pct(a829['N7k4j4']['arms']['U_LATE']['P_j'])}\% / {pct(a829['N9k5j5']['arms']['U_LATE']['P_j'])}\%; early layers: "
  f"{pct(a829['N7k4j4']['arms']['U_EARLY']['P_j'])}\% / {pct(a829['N9k5j5']['arms']['U_EARLY']['P_j'])}\%"),
 ("What does the late lookup compute?", "early lookup moved: answer toward target vs ordinal "
  f"{pct(a831['N7k5j3']['G1']['toward_target'])} vs {pct(a831['N7k5j3']['G1']['toward_ordinal'])}\% ($N$=7), "
  f"{pct(a831b['N9k7j3']['G1']['toward_target'])} vs {pct(a831b['N9k7j3']['G1']['toward_ordinal'])}\% ($N$=9); start keeps its ordinal "
  f"{pct(a830['N9k5']['F2']['shares']['unmoved'])}\%"),
 ("Does a fixed law predict new cells?", f"within .15: {pct(a832['P1']['share'])}\%; "
  f"+{a832['P3']['gain_vs_natural']:.2f} nats over natural, +{a832['P3']['gain_vs_copy_anchor']:.2f} over ignoring the shift; modes {pct(a832['P2']['share'])}\%"),
 ("Why must \\rthree{} rebind the frame?", "frame write: late re-reading "
  f"{wr(min(a833[c]['w']['O_FR'][0] for c in a833))}--{wr(max(a833[c]['w']['O_FR'][0] for c in a833))} (start 0, end 1), identity "
  f"{pct(a833['N7k4j4']['P_j']['O_FR'])}\% / {pct(a833['N9k5j5']['P_j']['O_FR'])}\%; transformations need both writes "
  f"({pct(a833['N7k5j3']['P_j']['O_ORD_FR'])}\% / {pct(a833['N9k7j3']['P_j']['O_ORD_FR'])}\%)"),
]
L = ["\\begin{table*}[t]", "\\centering", "\\small", "\\begin{tabular}{p{4.6cm}p{10.5cm}}", "\\toprule", "Registered question & Result (Qwen3.5-27B; identity cells $N$=7 / $N$=9 unless stated) \\\\", "\\midrule"]
for q, r in rows:
    L.append(f"{q} & {r} \\\\")
L += ["\\bottomrule", "\\end{tabular}",
      "\\caption{\\textbf{Each stage was tested by an experiment registered before its data.} Predictions for intervention runs were fixed, with a checksum, before those runs. Full rules, predictions and verdicts in Appendix~\\ref{app:registry}.}",
      "\\label{tab:registry}", "\\end{table*}"]
open(f"{OUT}/registry.tex", "w").write("\n".join(L) + "\n"); print("registry table ok")
