"""v2 paper appendix tables, from saved analysis JSONs only (verdict wording follows the verdict records, FINDINGS §916-§930 and
knowledge/DECISIONS.md).  Writes research/ces/paper_v2_20261017/tables/{app_exact, app_transfer, app_registry, app_failures, app_mmiusig}.tex.
Usage (container): python /vlm/exp/figs/make_v2_appendix.py"""
import json, os

R = "/vlm/results/r3"; OUT = "/vlm/research/ces/paper_v2_20261017/tables"
os.makedirs(OUT, exist_ok=True)
load = lambda f: json.load(open(f"{R}/{f}")) if os.path.exists(f"{R}/{f}") else None
p2 = lambda x: "--" if x is None else (f"{x:.2f}" if abs(x) >= .005 else "0.00")
pc = lambda x: "--" if x is None else str(int(100 * x + .5 + 1e-9))
CELL = {"N7k4": (7, 4), "N9k5": (9, 5), "N7k4j4": (7, 4), "N9k5j5": (9, 5), "N8k5j4": (8, 5), "N7k5j3": (7, 5), "N9k7j3": (9, 7),
        "N8k6j3": (8, 6), "N8k3j6": (8, 3)}
cell = lambda c: "$N{=}%d$, $k{=}%d$" % CELL[c]


def table(name, spec, head, rows, cap, label, size="\\small", star=True):
    env = "table*" if star else "table"
    L = [f"\\begin{{{env}}}[t]", "\\centering", size, f"\\begin{{tabular}}{{{spec}}}", "\\toprule", head + " \\\\", "\\midrule"]
    L += [r if r.startswith(("\\midrule", "\\cmidrule", "\\addlinespace")) else r + " \\\\" for r in rows]
    L += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{cap}}}", f"\\label{{{label}}}", f"\\end{{{env}}}"]
    open(f"{OUT}/{name}.tex", "w").write("\n".join(L) + "\n")
    print("wrote", name)


def exact():
    rows = []
    d = load("s826b_analysis.json")
    if d:
        I = d["s826b_27b"]["I0"]
        rows.append(f"S826b & $N{{=}}8$, $k\\in\\{{5,6,7\\}}$ & 40 & {pc(I['noop'])} / {pc(I['src0_is_start'])} & {p2(I['noop_menu_diff_p95'])} & -- & -- & yes")
    def add(exp, f, outer=None, ref=("QNOOP_E=END", "QNOOP_S=START")):
        d = load(f)
        if not d:
            return
        d = d[outer] if outer else d
        for c, v in d["cases"].items():
            I = v["I0"]; n = None
            if "qnoop_agree" in I:
                ag, p95 = f"{pc(I['qnoop_agree'])}", p2(I["qnoop_p95"])
            elif "wnoop_agree" in I:
                ag, p95 = f"{pc(I['wnoop_agree'])}", p2(I["wnoop_p95"])
            else:
                a, b = I[ref[0]], I[ref[1]]
                ag, p95 = f"{pc(a['agree'])} / {pc(b['agree'])}", p2(max(a["p95"], b["p95"]))
            don = pc(I["donor_pooled"]) if "donor_pooled" in I else "--"
            use = "yes" if I["ok"] else "no (descriptive)"
            rows.append(f"{exp} & {cell(c)} & {d['n'] // len(d['cases']) if isinstance(d.get('n'), int) else 40} & {ag} & {p95} & {pc(I['start_acc'])} & {don} & {use}")
    add("S827", "s827_analysis.json", "s827_27b"); add("S828", "s828_analysis.json"); add("S829", "s829_analysis.json")
    add("S830", "s830_analysis.json"); add("S831", "s831_analysis.json"); add("S831b", "s831_analysis_s831b_27b.json")
    add("S832", "s832_analysis.json"); add("S833", "s833_analysis.json")
    d = load("s834_analysis.json")
    if d:
        I = d["I0"]; rep = I["k3_end_reproduces_s824"]
        rows.append(f"S834 & $N{{=}}7, 8$ & 40 & S824 repeat {pc(min(rep.values()))} & -- & {pc(min(I['start_correct'].values()))} & -- & "
                    + ("yes" if I["holds"] else "no (descriptive)"))
    table("app_exact", "llrlrrrl", "Exp. & Cell & Scenes & Reference agree (\\%) & $p_{95}$ $|\\Delta z|$ & \\START{} acc. & Donors & Used", rows,
          "\\textbf{Exactness controls.} Reference: the same forward with the edit machinery and no edit (\\END{} / \\START{}), the residual "
          "write with zero vectors (S833), or a repeat of S824 cells (S834); agreement of the answered image with the unedited run, and the "
          "95th percentile of the absolute change of the menu logits. \\START{} acc.: start-relative question answered correctly; Donors: "
          "\\START{}$(p)$ runs used as address donors answered correctly. Image keys are identical between \\START{} and \\END{} in every "
          "query-editing experiment (largest difference 0.0). Cells that miss a registered threshold are reported but not used for verdicts.",
          "tab:app-exact", size="\\footnotesize")


def transfer():
    d = load("s829_analysis.json")
    if not d:
        return
    arms = [("U_FULL", "\\END{} (from \\START{})", "address, layers 35--55"), ("U_EARLY", "", "address, layers 35--47"),
            ("U_LATE", "", "address, layers 51--55"), ("P_FULL", "", "non-address, layers 35--55"),
            ("P_LATE", "", "non-address, layers 51--55"), ("MASS", "", "picture-vs-text only"),
            ("R_U", "\\START{} (from \\END{})", "address"), ("R_P", "", "non-address"), ("R_ALL", "", "whole query"),
            ("R_UM", "", "address + picture-vs-text")]
    rows = []
    for a, grp, lab in arms:
        cells = []
        for c in ("N7k4j4", "N9k5j5"):
            x = d["cases"][c]["arms"][a]; ci = x.get("ci")
            s = f"{x['dPj']:+.2f}".replace("+0.00", "0.00").replace("-0.00", "0.00")
            cells.append(s + (f" [{ci[0]:+.2f}, {ci[1]:+.2f}]".replace("-0.00", "0.00").replace("+0.00", "0.00") if ci else "") + f" & {p2(x['late_mass'])}")
        if grp and a == "R_U":
            rows.append("\\midrule")
        rows.append(f"{grp} & {lab} & " + " & ".join(cells))
    m = {c: d["cases"][c]["M0"] for c in ("N7k4j4", "N9k5j5")}
    rows.append("\\midrule")
    rows.append(f"natural & \\END{{}} / \\START{{}} & -- & {p2(m['N7k4j4']['END'])} / {p2(m['N7k4j4']['START'])} & -- & "
                f"{p2(m['N9k5j5']['END'])} / {p2(m['N9k5j5']['START'])}")
    table("app_transfer", "llrrrr", "Into & Part moved & \\multicolumn{2}{c}{$N{=}7$, $k{=}4$} & \\multicolumn{2}{c}{$N{=}9$, $k{=}5$} \\\\\n"
          " & & $\\Delta P(j)$ [95\\% CI] & late att. & $\\Delta P(j)$ [95\\% CI] & late att.", rows,
          "\\textbf{Query-part transfers on identity references (S829, 40 scenes per cell).} Upper block: a part of the \\START{} query is put "
          "into the \\END{} question at the lookup layers; lower block: a part of the \\END{} query is put into the \\START{} question. "
          "Late image attention: share of the attention of the index-matching heads of boundary rows 1 and 4 and the answer row on image "
          "tokens, mean of layers 51 and 55 (natural runs in the last row). The $N{=}8$ cell missed "
          "the registered exactness threshold and is omitted.", "tab:app-transfer", size="\\footnotesize\\setlength{\\tabcolsep}{4pt}")


REG = [
    ("S801", "Oct 15, 00:5x", "--", "Are correct end-relative answers reads of the target? Natural answers on the same scenes; extended before "
     "data to the four smaller models (three run on Oct 17).", "used descriptively (Figure~\\ref{fig:scale}): copy 46--84\\% and identity "
     "78--98\\% in the smaller models, 5\\% and 45\\% in 27B; VL-4B missed the grounding-instrument threshold."),
    ("S824", "Oct 16, 15:47", "--", "Which wrong answers are target-relative, literal-relative or absolute; does the stated count or the shown "
     "images set the target?", "one step past the target and one step from the copy both present; an absolute bias to image 2 not excluded "
     "(.075); stated count vs shown images mixed ($D$ = .06)."),
    ("S825", "Oct 16, 17:40", "348f3d60", "Does an error law fitted at $N$ = 7, 8 predict the answer distributions at $N$ = 6, 9?",
     "calibration failed (45\\% within .15); informative (+.20 nats); anchors END(1) $\\to N$ and END($N$) $\\to$ 1 held."),
    ("S826b", "Oct 16, 18:58", "--", "Where is the stated count read, and which rows hold it?",
     "absorbed by layer 31; held by question-side rows (.99 vs .01 at layer 35); continuous index reproduces S826 within .054."),
    ("S827", "Oct 16, 20:18", "dd4b360a", "On identity references, does \\END{} write a wrong address or weaken the right one, and does copy "
     "suppression (H-A) or a predecessor gain (H-B) cause the error?", "wrong address (73\\% / 94\\% of neighbour answers); boundary-row "
     "address operative (+.20 / +.38); neither H-A nor H-B alone; \\START{} withstands the same change. Predicted: natural distributions and "
     "internal changes; failed: answers under interventions (56\\% / 51\\% within .15)."),
    ("S828", "Oct 16, 21:18", "15db59d2", "Do the answer row's own address and the boundary rows' address carry the error together? Does a "
     "two-route law fixed after S827 predict interventions?", "answer-row address operative; both carriers restored: 80\\% / 93\\% on target; "
     "the two-route \\END{} address breaks \\START{} at $N$ = 9 ($-$.40); non-address parts carry part of the error; law within .15: 75\\%, "
     "informative, transfers to a new cell."),
    ("S829", "Oct 16, 21:59", "e396de30", "Is the error a wrong address combined with an \\END{} reading mode; is the early or the late address "
     "at fault; does a mixture law predict?", "first run invalid (stale module in the resident worker; kept, rerun); \\END{} re-reads images "
     "late; the late address carries the error; \\END{}'s whole query breaks \\START{} at $N$ = 9 ($-$.58), its reading mode alone does not; "
     "the picture-vs-text direction is not the reading mode; mixture law failed (67\\%)."),
    ("S830", "Oct 16, 22:40", "--", "Does the late lookup answer the position just before the early one?",
     "no; \\START{} keeps its ordinal; the late address sets the answer; the answer follows the early position (slope .44--.48) one step toward "
     "the target."),
    ("S831", "Oct 16, 23:38", "--", "Off the identity, is the step toward the target, toward the ordinal, or always back?",
     "$N$ = 7: toward the target (67\\% vs 3\\%), moves at the target (23/30), one step most often; $N$ = 9 missed exactness (descriptive)."),
    ("S831b", "Oct 17, 00:17", "--", "Replication of the $N$ = 9 cell of S831 with 60 scenes, same rules.",
     "toward the target (32\\% vs 4\\%), moves at the target; step size not one (0: .28, +1: .27, +2: .11)."),
    ("S832", "Oct 17, 00:27", "21ae68b8", "Does the two-component law fixed after S831 predict two unseen cells?",
     "within .15: 80\\%; +.25 nats over the natural distribution, +.12 over a copy anchor; modes failed (25\\%); natural END(3) failed."),
    ("S833", "Oct 17, 01:24", "--", "Does \\rthree{}'s frame write turn off the late re-lookup?",
     "yes in all four cells; frame write alone fixes identity references (100\\% / 100\\%); off the identity both writes are needed "
     "(100\\% / 97\\%); the ordinal write alone is re-reflected."),
    ("S834", "Oct 17, 02:32", "--", "Descriptive: the missing cells of the natural answer grid (Figure~\\ref{fig:grid}).", None),
    ("S835", "Oct 17, 02:39", "--", "Descriptive: signatures of wrong MMIU answers under a two-form readout.", None),
]


def registry():
    rows = []
    for exp, t, sha, q, v in REG:
        if v is None:
            d = load(f"{exp.lower()}_analysis.json")
            if not d:
                continue
            if exp == "S834":
                v = ("exactness holds; scenes identical to S824." if d["I0"]["holds"] else "missed exactness; not used.")
            if exp == "S835":
                a, b = d["all"], d["t1_only"]
                v = (("" if d["I0"]["holds"] else f"free-token check missed ({pc(d['t1']['F'])}\\% / {pc(d['t1']['R'])}\\% letters; two forensic tasks start "
                      "to reason), so descriptive; ") + f"literal {pc(a['literal'])}\\% vs chance {pc(a['literal_chance'])}\\% (letter answers "
                     f"{pc(b['literal'])}\\% vs {pc(b['literal_chance'])}\\%); step below chance; same answers as the earlier readout.")
        rows.append(f"{exp} & {t} & {sha} & {q} & {v}")
    table("app_registry", "lllp{4.9cm}p{5.9cm}", "Exp. & Registered & Frozen & Question & Verdicts", rows,
          "\\textbf{Registered experiments.} Each experiment's design, analysis rules, thresholds and the authors' predictions were recorded "
          "before its data; for intervention runs, predictions computed from earlier data were frozen with a SHA-256 checksum (first eight hex "
          "digits) before the run. All on Qwen3.5-27B, procedural scenes, new seeds per experiment.", "tab:app-registry", size="\\scriptsize")


FAIL = [("S825", "error law fitted at $N$ = 7, 8, predicting $N$ = 6, 9", "45\\% within .15 (needed 70\\%)",
         "the law lacked the early position; the step component came later"),
        ("S827", "boundary-row law predicting answers under interventions", "56\\% / 51\\% within .15",
         "the answer row carries its own address (S828)"),
        ("S829", "mixture of the two-route law by late image attention", "67\\% within .15; $-$.35 nats",
         "the reading mode is not a scalar weight on one law"),
        ("S831b", "step of exactly one image", "0 and +1 equally frequent (.28 / .27)", "the step is a distribution over 0--2 images"),
        ("S832", "most likely answer of the fixed law", "25\\% of modes correct", "the law predicts probabilities, not modes"),
        ("S832", "natural END(3) at $N$ = 8", "target .53 predicted as .21", "near the end, the end-anchored path is sharper than the law"),
        ("S833", "ordinal write alone moving the address one step", "$N$ = 9 lands on $k$ (.57)", "the written ordinal is re-reflected in full")]


def failures():
    rows = [f"{e} & {w} & {o} & {l}" for e, w, o, l in FAIL]
    table("app_failures", "lp{4.6cm}p{3.6cm}p{5.2cm}", "Exp. & Prediction & Outcome & What it changed", rows,
          "\\textbf{Registered predictions that failed.} Each failure is kept with its registered threshold; the last column states the "
          "account it led to.", "tab:app-failures", size="\\footnotesize")


def mmiusig():
    d = load("s835_analysis.json")
    if not d:
        return
    raw = json.load(open("/vlm/results/s835_s835_27b_mmiusig.json"))["rows"]
    old = {(r["task"], r["i"]) for r in json.load(open("/vlm/results/s489_mmiu_fix_489a27b.json"))["rows"]}
    sig = [r for r in raw if (r["task"], r["i"]) in old and r["gold"] in "ABCD" and len(r["ks"]) == 4]   # the analysis' signature set
    by = {}
    for r in sig:
        by.setdefault(r["task"], []).append(r)
    mean = lambda xs: sum(xs) / len(xs)
    TN = {"face_retrieval": "face retrieval", "handwritten_retrieval": "handwriting retrieval", "image2image_retrieval": "image-to-image retrieval",
          "person_reid": "person re-identification", "vehicle_retrieval": "vehicle retrieval", "sketch2image_retrieval": "sketch-to-image retrieval",
          "text2image_retrieval": "text-to-image retrieval", "forensic_detection_blink": "forensic detection (BLINK)",
          "forensic_detection_forgerynet": "forensic detection (ForgeryNet)", "visual_coherence": "visual coherence",
          "Icon_Question_Answering_with_Spatial_Context": "icon QA with spatial context"}
    stat = lambda R: (len(R), mean([r["F"] == r["gold"] for r in R]), mean([r["R"] == r["gold"] for r in R]), mean([r["R|t1"] for r in R]))
    rows = []
    for t, x in d["literal_by_task"].items():
        n, F, E, t1 = stat(by[t])
        rows.append(f"{TN.get(t, t)} & {n} & {pc(F)} & {pc(E)} & {pc(t1)} & {x['errors']} & {pc(x['literal']) if x['errors'] else '--'}")
    a, b = d["all"], d["t1_only"]
    n, F, E, t1 = stat(sig)
    rows.append("\\midrule")
    rows.append(f"all & {n} & {pc(F)} & {pc(E)} & {pc(t1)} & {a['errors']} & {pc(a['literal'])} (chance {pc(a['literal_chance'])})")
    rows.append(f"letter answers only & & & & & {b['errors']} & {pc(b['literal'])} (chance {pc(b['literal_chance'])})")
    table("app_mmiusig", "lrrrrrr", "Task & Items & Start & End & End $t_1$ & End errors & Literal", rows,
          "\\textbf{End-relative MMIU errors of Qwen3.5-27B read the option's ordinal from the start.} Items with four options. Start, End: "
          "accuracy (\\%) of the two wordings; End $t_1$: share (\\%) of items whose unrestricted next token is an option letter; Literal: share "
          "(\\%) of wrong end-relative answers whose ordinal, read from the start, names the gold image, with the mean over the three wrong options "
          "as chance.", "tab:app-mmiusig", size="\\footnotesize\\setlength{\\tabcolsep}{4pt}", star=True)

if __name__ == "__main__":
    exact(); transfer(); registry(); failures(); mmiusig()
