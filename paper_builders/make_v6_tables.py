"""v6 tables from saved records only (no hand-typed numbers).  Output: research/ces/paper_v6_20261019/tables/.
  rebind.tex      Table 1: the 2 x 2 of address and frame writes for k != 4 and the k = 4 cell, five models (make_r3_tables.rebind_main2:
                  S786 bare readout, Qwen3.5-4B from S794 with the two-form readout; S782 registry item 6).
  bench_main.tex  Table 2 (one column): native MMIU visual cloze and held-out MMIU position items, Start / End / End + R3
                  (make_r3_tables.bench_values: option-letter readout, unaffected by S782).
  bench_full.tex  Appendix: the four benchmark sets of v5's Table 2 (same values).
  app_*.tex, registry.tex
                  copied from research/ces/paper_v5_20261018/tables/ (built by exp/figs/make_v5_tables.py and make_v5_appendix.py).
Usage (container; /vlm = repository root): python /vlm/exp/figs/make_v6_tables.py"""
import os, shutil, sys
sys.path.insert(0, "/vlm/exp/figs")
import make_r3_tables as T

OUT = "/vlm/research/ces/paper_v6_20261019/tables"
V5 = "/vlm/research/ces/paper_v5_20261018/tables"
os.makedirs(OUT, exist_ok=True)
p1 = lambda x: "--" if x is None else f"{100 * x:.1f}"
short = lambda nm: nm.replace("Qwen3-VL-", "VL-").replace("Qwen3.5-", "3.5-")

body = T.rebind_main2()
open(f"{OUT}/rebind.tex", "w").write(
    "\\begin{table}[t]\n\\centering\\small\n\\setlength{\\tabcolsep}{4pt}\n\\resizebox{\\columnwidth}{!}{" + body + "}\n"
    "\\caption{\\textbf{Both writes are needed, each where predicted.} Answers on the target (\\%), seven images, 30 scenes per model and "
    "condition. \\rthree: both writes.}\n\\label{tab:rebind}\n\\end{table}\n")

V = T.bench_values()
L = ["\\begin{tabular}{@{}lrrrrrr@{}}", "\\toprule",
     " & \\multicolumn{3}{c}{MMIU visual cloze} & \\multicolumn{3}{c}{MMIU position} \\\\",
     " & \\multicolumn{3}{c}{(native)} & \\multicolumn{3}{c}{(rewritten)} \\\\",
     "\\cmidrule(lr){2-4}\\cmidrule(l){5-7}",
     "Model & Start & End & +\\rthree & Start & End & +\\rthree \\\\", "\\midrule"]
for b, nm in T.MODELS:
    cells = []
    for key in ("MMIU cloze", "MMIU held-out"):
        v = V[f"{b}|{key}"]
        cells += [p1(v["F"]), p1(v["R"]), "\\textbf{" + p1(v["rep"]) + "}"]
    L.append(f"{short(nm)} & " + " & ".join(cells) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
n_cloze = f'{V["27b|MMIU cloze"]["n"]:,}'; n_pos = f'{V["27b|MMIU held-out"]["n"]:,}'
open(f"{OUT}/bench_main.tex", "w").write(
    "\\begin{table}[t]\n\\centering\\small\n\\setlength{\\tabcolsep}{3.2pt}\n" + "\n".join(L) + "\n"
    "\\caption{\\textbf{End-relative accuracy with \\rthree{}} (\\%). Start, End: the unedited model on the start- and end-relative wording of "
    f"the same items. Visual cloze is end-relative in MMIU itself ({n_cloze} items), position items are rewritten ({n_pos} held-out items).}}\n"
    "\\label{tab:bench}\n\\end{table}\n")

SETS = [("MMIU cloze", "MMIU visual cloze"), ("MMIU held-out", "MMIU held-out"), ("MuirBench held-out", "MuirBench held-out"), ("MMIU later", "MMIU later")]
L = ["\\begin{tabular}{l" + "rrr" * len(SETS) + "}", "\\toprule",
     " & \\multicolumn{3}{c}{native end-relative} & \\multicolumn{9}{c}{rewritten to end-relative (controlled diagnostic)} \\\\",
     "\\cmidrule(lr){2-4}\\cmidrule(lr){5-13}",
     " & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{lab}}}" for _, lab in SETS) + " \\\\",
     "".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(SETS))),
     "Model & " + " & ".join("Start & End & +\\rthree" for _ in SETS) + " \\\\", "\\midrule"]
for b, nm in T.MODELS:
    cells = []
    for key, _ in SETS:
        v = V.get(f"{b}|{key}")
        cells += [p1(v["F"]), p1(v["R"]), "\\textbf{" + p1(v["rep"]) + "}"] if v else ["--"] * 3
    L.append(f"{nm} & " + " & ".join(cells) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
open(f"{OUT}/bench_full.tex", "w").write(
    "\\begin{table*}[h]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{4.2pt}\n" + "\n".join(L) + "\n"
    "\\caption{\\textbf{All benchmark sets.} Accuracy (\\%) on the start-relative (Start) and end-relative (End) wordings of the same items, "
    "and on End with \\rthree{}. Images, options and answers are fixed; only the image references change.}\n\\label{tab:bench-full}\n\\end{table*}\n")

for f in ("app_exact", "app_failures", "app_mmiusig", "app_registry", "app_transfer", "registry"):
    shutil.copy(f"{V5}/{f}.tex", f"{OUT}/{f}.tex")
shutil.copy("/vlm/research/ces/outputs/paper_r3_v3/tables/mech_qk.tex", f"{OUT}/mech_qk.tex")    # QK captures (S705a), old v3 builder
open(f"{OUT}/read_choice.tex", "w").write(T.read_choice() + "\n")
import json                                                                      # lure table from the saved lure values (make_r3_tables.lure's input)
LV = {d["tag"]: d for d in json.load(open("/vlm/results/figures/tables_r3/lure_values.json"))}
TG = T.bench_tags()
L = ["\\begin{tabular}{lrrrrr}", "\\toprule",
     "& & \\multicolumn{2}{c}{errors on the lure} & \\multicolumn{2}{c}{corrected by \\rthree} \\\\",
     "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}", "Model & errors & before & after & lure & other \\\\", "\\midrule"]
for b, nm in T.MODELS:
    d = LV[TG[b]["sealed"]]
    L.append(f"{nm} & {d['errors']} & {T.P(d['lure_share'])} & {T.P(d['lure_share_after'])} & {T.P(d['rescued_lure'])} & {T.P(d['rescued_other'])} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
open(f"{OUT}/lure.tex", "w").write("\n".join(L) + "\n")
open(f"{OUT}/cos.tex", "w").write(T.cos_table() + "\n")
print(body)

# Registry of the experiments restored from the old v3 line (metadata only; registration times from knowledge/DECISIONS.md, results in
# results/FINDINGS.md).  Each row: (experiment, what it re-measures or measures, readout, where used in v6, FINDINGS section).
REST = [
    ("S787 (S545)", "index-code shift +2, five models", "two-form", "Sec.~3, App.~D", "\\S865, \\S877--878"),
    ("S789 (S741b)", "keys / values of the image tokens", "two-form", "Sec.~3, App.~D", "--"),
    ("S789 (S739)", "equal and unequal image sizes, five models", "two-form", "Sec.~3, App.~D", "--"),
    ("S789 (S708)", "27B: written address with a start frame", "two-form", "Sec.~3, App.~D", "--"),
    ("S790c", "nearest start-relative lookup (QK captures)", "QK", "Sec.~4, App.~D", "\\S849"),
    ("S789 (S673b)", "ordinal state as a position, dense models", "two-form", "Sec.~4.1, App.~E", "--"),
    ("S789 (S720)", "query vs output gate, Qwen3.5-4B, 9B", "two-form", "Sec.~4.1, App.~E", "--"),
    ("S787 (S474b)", "forward-ordinal transplant, five models", "two-form", "App.~E", "\\S865, \\S878"),
    ("S783a (S769)", "27B: frame read at layers 27--34", "bare, t1 1.00", "Sec.~4.2, App.~F", "\\S847"),
    ("S795", "address lens (QK)", "QK", "Sec.~4.2, App.~F", "\\S855"),
    ("S796", "27B: frame read through the lookup queries", "two-form, t1 1.00", "Sec.~4.2, App.~F", "\\S856"),
    ("S748", "hiding images from full attention (QK)", "QK", "Sec.~4.3, App.~F", "\\S860"),
    ("S786 / S794", "address and frame writes, five models", "bare / two-form, t1 1.00", "Sec.~5, App.~J", "\\S851, \\S857"),
    ("S787 (S736)", "written address read as a natural ordinal", "two-form", "Sec.~5, App.~J", "\\S865, \\S878"),
    ("S721", "27B ablation on the later MMIU split", "option letters", "Sec.~5", "\\S787"),
    ("S793", "natural query exchange", "two-form, t1 1.00", "Sec.~6, App.~K", "\\S854"),
    ("S783a (S780)", "changing the picture at the address", "bare, t1 1.00", "Sec.~6, App.~K", "\\S847"),
    ("S791", "each write followed to its picture", "bare, t1 1.00", "Sec.~6, App.~K", "\\S853"),
    ("S740", "MMIU photographs: queries of the repair", "option letters", "Sec.~6", "--"),
]
L = ["\\begin{table*}[h]", "\\centering\\small", "\\begin{tabular}{llllp{2.2cm}}", "\\toprule",
     "Experiment & What it measures & Readout & Used in & Record \\\\", "\\midrule"]
L += [" & ".join(r) + " \\\\" for r in REST]
L += ["\\bottomrule", "\\end{tabular}",
      "\\caption{\\textbf{Experiments restored from earlier work.} Each answer-based value was re-measured with the two-form or bare readout "
      "(in parentheses: the original experiment), the others are query--key captures or option letters, which the earlier readout defect did "
      "not affect. t1: share of runs whose unrestricted top token is a menu word, where recorded.}", "\\label{tab:app-registry-v6}", "\\end{table*}"]
open(f"{OUT}/app_registry_v6.tex", "w").write("\n".join(L) + "\n")
