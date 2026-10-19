"""v7 tables (no model calls).  Output: research/ces/paper_v7_20261019/tables/.
Unchanged tables are copied from paper_v6_20261019/tables (built by exp/figs/make_v6_tables.py).  Rebuilt or new here:
  bench_main.tex    Table 2: same values as v6; the caption marks visual cloze as a development set, its option-A-biased start-relative
                    rewrite, and the option-letter readout (mock review W2).
  bench_paired.tex  Appendix: paired changes R3 - End and R3 - Start with item-bootstrap 95% intervals (4,000 draws, seed 0), all four sets.
  cos.tex           Appendix: as v6, fitted to the column width (mock review C4).
  repro.tex         Appendix: checkpoints, software and the scene seeds of the experiments cited (mock review W3).
Benchmark values come from results/s582_evalrepair_{f7*, c8*}.json through make_r3_tables (option-letter readout).
Usage (container; /vlm = repository root): python /vlm/exp/figs/make_v7_tables.py"""
import json, os, shutil, sys
import numpy as np
sys.path.insert(0, "/vlm/exp/figs")
import make_r3_tables as T

OUT = "/vlm/research/ces/paper_v7_20261019/tables"
V6 = "/vlm/research/ces/paper_v6_20261019/tables"
os.makedirs(OUT, exist_ok=True)
for f in os.listdir(V6):
    if f.endswith(".tex") and f not in ("bench_main.tex", "cos.tex"):
        shutil.copy(f"{V6}/{f}", f"{OUT}/{f}")
p1 = lambda x: f"{100 * x:.1f}"
short = lambda nm: nm.replace("Qwen3-VL-", "VL-").replace("Qwen3.5-", "3.5-")
V = T.bench_values()
TAGS = T.bench_tags()

L = ["\\begin{tabular}{@{}lrrrrrr@{}}", "\\toprule",
     " & \\multicolumn{3}{c}{MMIU visual cloze} & \\multicolumn{3}{c}{MMIU position} \\\\",
     " & \\multicolumn{3}{c}{(native, development)} & \\multicolumn{3}{c}{(rewritten, held out)} \\\\",
     "\\cmidrule(lr){2-4}\\cmidrule(l){5-7}", "Model & Start & End & +\\rthree & Start & End & +\\rthree \\\\", "\\midrule"]
for b, nm in T.MODELS:
    cells = []
    for key in ("MMIU cloze", "MMIU held-out"):
        v = V[f"{b}|{key}"]
        cells += [p1(v["F"]), p1(v["R"]), "\\textbf{" + p1(v["rep"]) + "}"]
    L.append(f"{short(nm)} & " + " & ".join(cells) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
n_cloze, n_pos = f'{V["27b|MMIU cloze"]["n"]:,}', f'{V["27b|MMIU held-out"]["n"]:,}'
open(f"{OUT}/bench_main.tex", "w").write(
    "\\begin{table}[t]\n\\centering\\small\n\\setlength{\\tabcolsep}{3.2pt}\n" + "\n".join(L) + "\n"
    "\\caption{\\textbf{End-relative accuracy with \\rthree{}} (\\%, option-letter readout). Start, End: the unedited model on the start- and "
    f"end-relative wording of the same items. Visual cloze is end-relative in MMIU itself ({n_cloze} development items, whose start-relative "
    f"rewrite is biased toward option A), position items are rewritten ({n_pos} held-out items). Paired intervals: Appendix~\\ref{{app:bench}}.}}\n"
    "\\label{tab:bench}\n\\end{table}\n")

# paired changes with item-bootstrap intervals
SETS = [("cloze", "cloze", "cloze"), ("sealed", "mmiu_pos", "MMIU held-out"), ("sealed", "muir", "MuirBench"), ("later", "mmiu_pos", "MMIU later")]
rng = np.random.default_rng(0)
sg = lambda v: f"{v:+.1f}".replace("-", "$-$")
fmt = lambda x, ci: f"{sg(100 * x)} [{sg(100 * ci[0])}, {sg(100 * ci[1])}]"
L = ["\\begin{tabular}{@{}lllll@{}}", "\\toprule", "Model & visual cloze & MMIU held-out & MuirBench & MMIU later \\\\", "\\midrule"]
PAIRED = {}
for part, key_ in (("\\rthree{} $-$ End", "R|base"), ("\\rthree{} $-$ Start", "F|base")):
    L.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\textit{{{part}}}}} \\\\")
    for b, nm in T.MODELS:
        cells = []
        for split, suite, lab in SETS:
            rows = [r for r in T.load(f"/vlm/results/s582_evalrepair_{TAGS[b][split]}.json")["rows"] if r["suite"] == suite]
            x = np.array([(r["R|rep"] == r["gold"]) - (r[key_] == r["gold"]) for r in rows], float)
            bs = x[rng.integers(0, len(x), (4000, len(x)))].mean(1)
            ci = np.percentile(bs, [2.5, 97.5])
            PAIRED[f"{b}|{lab}|{key_}"] = [float(x.mean()), float(ci[0]), float(ci[1]), len(x)]
            cells.append(fmt(x.mean(), ci))
        L.append(f"{short(nm)} & " + " & ".join(cells) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
open(f"{OUT}/bench_paired.tex", "w").write(
    "\\begin{table*}[h]\n\\centering\\small\n" + "\n".join(L) + "\n"
    "\\caption{\\textbf{Paired benchmark changes} (points, 95\\% item-bootstrap intervals). Visual cloze is a development set and its "
    "start-relative rewrite is biased toward option A, so its \\rthree{} $-$ Start row is not a comparison with a ceiling.}\n"
    "\\label{tab:bench-paired}\n\\end{table*}\n")
json.dump(PAIRED, open(f"{OUT}/bench_paired_values.json", "w"), indent=1)

open(f"{OUT}/cos.tex", "w").write("\\resizebox{\\columnwidth}{!}{" + T.cos_table() + "}\n")

# reproduction table: checkpoints, software, seeds (metadata only)
MODELS = [("Qwen3-VL-4B", "Qwen/Qwen3-VL-4B-Instruct", "edac77033291"), ("Qwen3-VL-8B", "Qwen/Qwen3-VL-8B-Instruct", "5cd452860dc1"),
          ("Qwen3.5-4B", "Qwen/Qwen3.5-4B", "ddc63e1c717a"), ("Qwen3.5-9B", "Qwen/Qwen3.5-9B", "d0883072e018"),
          ("Qwen3.5-27B", "Qwen/Qwen3.5-27B", "f8d190c5b89c")]
SEEDS = [("index-code shift (S787 re-run of S545)", "78700"), ("keys and values (S789 re-run of S741b)", "74100"),
         ("image sizes (S789 re-run of S739)", "73900"), ("ordinal state as position (S789 re-run of S673b)", "67350"),
         ("query vs gate (S789 re-run of S720)", "70510"), ("27B written address (S789 re-run of S708)", "70800--70802"),
         ("forward-ordinal transplant, frame writes (S787)", "78720, 78740"), ("QK captures (S705a)", "70500"),
         ("frame read (S783a re-run of S769)", "76900 ($N=7$), 76901 ($N=8$)"), ("picture replacement (S783a re-run of S780)", "78000, 78001"),
         ("address and frame writes (S786 / S794)", "73000"), ("hiding images (S748)", "74800"), ("each write to its picture (S791)", "79100"),
         ("natural query exchange (S793)", "79300"), ("address lens (S795)", "79500"), ("frame through the queries (S796)", "79600"),
         ("frame write turns the update off (S833)", "83400"), ("stop tests (S844, S846)", "84400, 84600"),
         ("late-update tests (S851--S853)", "85100, 85200, 85300"), ("count factorial, writer inputs (S854-1, S854-2)", "85400, 85500"),
         ("geometry (S855)", "85600"), ("late-query swap (S856b)", "85800"), ("source cut (S857)", "85900"), ("held-out predictions (S858)", "85980")]
L = ["\\begin{tabular}{@{}lll@{}}", "\\toprule", "Model & checkpoint & \\texttt{config.json} SHA-256 \\\\", "\\midrule"]
L += [f"{a} & {b} & \\texttt{{{c}}} \\\\" for a, b, c in MODELS]
L += ["\\midrule", "\\multicolumn{2}{@{}l}{Experiment} & scene seed \\\\", "\\midrule"]
L += [f"\\multicolumn{{2}}{{@{{}}l}}{{{a}}} & {b} \\\\" for a, b in SEEDS]
L += ["\\bottomrule", "\\end{tabular}"]
open(f"{OUT}/repro.tex", "w").write(
    "\\begin{table*}[h]\n\\centering\\small\n" + "\n".join(L) + "\n"
    "\\caption{\\textbf{Checkpoints and scene seeds.} Local snapshots of the public checkpoints, identified by the first twelve hex digits of "
    "their configuration file. Seeds fix the scenes of each experiment; re-runs use the seeds of the original.}\n"
    "\\label{tab:repro}\n\\end{table*}\n")
print("ok v7 tables")
