"""v4 appendix tables: the v3 builders (exp/figs/make_v3_appendix.py) with S850-S853 added to the registry (S853 failed predictions in the failures table) and the S849-U2 verdict
worded as the user corrected it (10-17 evening: decodable, but amplifying it at the tested locus does not alter the update).  Writes research/ces/paper_v4_20261017/tables/.
Usage (container): python /vlm/exp/figs/make_v4_appendix.py"""
import json, sys
sys.path.insert(0, "/vlm/exp/figs")
import make_v3_appendix as V3
A = V3.A
A.OUT = "/vlm/research/ces/paper_v4_20261017/tables"
a850 = json.load(open("/vlm/results/r3/s850_analysis.json")); ex = json.load(open("/vlm/research/ces/outputs/s850_explore.json"))
sp = json.load(open("/vlm/research/ces/outputs/s851_split.json"))   # post hoc split by cell type (exp/p4_s851_split.py)
sg = lambda x: ("$+$" if x >= 0 else "$-$") + f"{abs(x):.1f}"
A.REG += [("S850", "Oct 17, 21:27", "705fe80f", "Steering the boundary rows' target-related direction at the writer's input: does the update stop (writer, late "
           "address, answer; random, NOOP and whole-boundary controls)?",
           f"signal holds in new scenes ($d'$ {a850['REP_B']['sites']['B1_L46']['dprime']:.2f}); whole-boundary replacement changes "
           f"{100 * a850['POS']['changed']:.0f}\\% of answers; writer and late address move toward the start-relative state (predicted from the writer "
           f"change, sign {a850['PRED']['sign_agree']:.2f}); staying {sg(100 * a850['UPDATE']['all_16']['S_P'][0])} pts: no stop-specific effect; "
           f"exploratory selectivity for goal-matched cells (vs NOOP): +8 SD {sg(100 * ex['SELECT']['PLUS8']['Delta_match'][0])} "
           f"[{sg(100 * ex['SELECT']['PLUS8']['Delta_match'][1][0])}, {sg(100 * ex['SELECT']['PLUS8']['Delta_match'][1][1])}], +16 SD "
           f"{sg(100 * ex['SELECT']['PLUS16']['Delta_match'][0])} [{sg(100 * ex['SELECT']['PLUS16']['Delta_match'][1][0])}, "
           f"{sg(100 * ex['SELECT']['PLUS16']['Delta_match'][1][1])}] pts; answers changed in {100 * ex['CHAIN']['PLUS16']['answers_changed']:.0f}\\% "
           f"(random {100 * ex['CHAIN']['RAND16']['answers_changed']:.0f}\\%). Post hoc split: where $p$ is not the target, +16 SD lowers target answers "
           f"{sg(100 * sp['S850']['non']['contrast']['acc|PLUS16-NOOP_B'][0])} [{sg(100 * sp['S850']['non']['contrast']['acc|PLUS16-NOOP_B'][1][0])}, "
           f"{sg(100 * sp['S850']['non']['contrast']['acc|PLUS16-NOOP_B'][1][1])}] pts (random {sg(100 * sp['S850']['non']['contrast']['acc|RAND16-NOOP_B'][0])}).")]
a849u2 = json.load(open("/vlm/results/r3/s849u_analysis_s849u2_27b.json"))["USE"]
for i, row in enumerate(A.REG):
    if row[0] == "S849-U2":
        A.REG[i] = row[:4] + (f"a weak target-related signal is decodable, but amplifying it at the tested answer-row locus does not alter the update "
                              f"(stay vs random {sg(100 * a849u2['46']['D'][0])} / {sg(100 * a849u2['50']['D'][0])} pts, intervals within $\\pm$5.3); "
                              "the steering persists downstream (.72 / .66).",) + row[5:]
a851 = json.load(open("/vlm/results/r3/s851_analysis.json"))
ci2 = lambda c: f"[{c[0]:.2f}, {c[1]:.2f}]"
A.REG += [("S851", "Oct 17, 22:48", "--", "Minimal path test: does replacing the whole boundary state reach the answer through the late queries 51--63, "
           "and does the steered run (+16 SD, S850) reproduce its effect when given those queries?",
           f"the replacement changes staying ({sg(100 * a851['POS']['G'])} pts [{sg(100 * a851['POS']['ci'][0])}, {sg(100 * a851['POS']['ci'][1])}]); its "
           f"late queries give the steered run {a851['SUFF']['s']:.2f} {ci2(a851['SUFF']['ci'])} of that change, and end-relative late queries remove "
           f"{a851['NEC']['n']:.2f} {ci2(a851['NEC']['ci'])} from the replaced run: the change travels through the late queries, and the steered "
           "run's late queries are not changed enough (post hoc: B1's late address, late image re-reading). Post hoc: with the swapped queries "
           f"the steered run gives the replaced run's answer in {100 * sp['S851']['agree']['all|PLUSQ=FULLB'][0]:.0f}\\% of runs ("
           f"{100 * sp['S851']['agree']['all|PLUS16=FULLB'][0]:.0f}\\% before), the replaced run with end-relative queries the end-relative answer in "
           f"{100 * sp['S851']['agree']['all|FULLBQE=E_EARLY'][0]:.0f}\\%; the replacement is not goal-selective (where $p$ is not the target, target "
           f"answers {sg(100 * sp['S851']['non']['contrast']['acc|FULLB-E_EARLY'][0])} pts). Does not test whether the target signal is used in natural runs.")]

a852 = json.load(open("/vlm/results/r3/s852_analysis.json")); a853 = json.load(open("/vlm/results/r3/s853_analysis.json"))
f2 = lambda v: f"{v[0]:.2f} [{v[1][0]:.2f}, {v[1][1]:.2f}]"
s2 = lambda x: ("$+$" if x >= 0 else "$-$") + f"{abs(x):.2f}"
A.REG += [("S852", "Oct 18, 01:05", "--", "How is the late query formed: does the writer's end-specific output carry the anchor (that output computed at "
           "another anchor, or the whole band), and what sets the direction of the late move ($k$ = 3--5 at a fixed pinned position)?",
           f"direction depends on $k$ (registered label `toward the end-relative target', slope {s2(a852['DIR']['b'][0])} [{s2(a852['DIR']['b'][1][0])}, "
           f"{s2(a852['DIR']['b'][1][1])}]); per cell not a step toward the target: at $k$ = 5, $p$ = 3, 73\\% move back to image 2 (target 4: 13\\%); "
           f"the end-specific output is partly anchor-specific (another anchor's output: recipient's answer {a852['XD']['f_rec'][0]:.2f}, donor's "
           f"{a852['XD']['f_don'][0]:.2f}; paired difference $+$0.07 [$-$0.06, 0.20]); no transfer as a displacement (0.13 vs baseline 0.19).")]
T853 = a853["TYPES"]
A.REG += [("S853", "Oct 18, 02:14", "--", "What does the writer take in? Boundary rows B1 / B4 at layer 46 replaced by the state of another ordinal, "
           "early position, both, stated count, or the same stated target; additivity; same-target contrast.",
           f"all five types carried to the writer output, late scores and answer readout; early position: donor's answer "
           f"{T853['P']['f_don'][0]:.2f} vs recipient's {T853['P']['f_rec'][0]:.2f}; ordinal: {T853['K']['f_don'][0]:.2f} vs {T853['K']['f_rec'][0]:.2f} "
           f"(all ten read rows: 0.48--0.62); PRED-A and PRED-B not shown ({s2(a853['PRED_A']['diff'][0])}, {s2(a853['PRED_B']['diff'][0])}); "
           f"additive fit $R^2$ {a853['INT']['R2_add'][0]:.2f} (registered: not the sum of the single swaps); at the same stated target the predecessor step "
           f"follows the ordinal ({a853['NAT_J']['r']['E3']:.2f} vs {a853['NAT_J']['r']['E5']:.2f}; stated count alone {s2(a853['NAT_J']['D_cnt3'][0])}); "
           "pictures always 8, so the ordinal was not separated from the target counted from the visible end (S854-1: it is that target).")]
a854 = json.load(open("/vlm/results/r3/s854a_analysis.json")); ML = a854["MAIN_LL"]
A.REG += [("S854-1", "Oct 18, 13:23", "--", "Ordinal or image count? Actual pictures $N_a$ 6 / 8 $\\times$ stated count 6 / 8 $\\times$ $k$ 3 / 5 "
           "$\\times$ pinned $p$ 3 / 4: does the late update follow $k$, the target counted from the actual end, or from the stated count?",
           f"held-out log-likelihood: the actual-end target account is compatible with the full model ({s2(ML['per_H']['H_a']['vs_full'][0])} nats), "
           f"the ordinal-only ({s2(ML['per_H']['H_k']['vs_full'][0])}) and stated-target ({s2(ML['per_H']['H_s']['vs_full'][0])}) accounts are rejected; "
           f"the actual count matters, the stated count does not; at the same target the predecessor rates match ({a854['C1']['rates']['8_8_5_3']:.2f} vs "
           f"{a854['C1']['rates']['6_6_3_3']:.2f}; registered `partial'); the writer's end-specific output is organised by $k$, not by the target "
           f"(class error {a854['COMPACT']['D_write']['E']['H_k']:.2f} vs {a854['COMPACT']['D_write']['E']['H_a']:.2f}); the $k \\times p$ non-additivity "
           f"replicates at the writer output (FA47 $R^2$ {a854['INT_REP']['levels']['8_8']['FA47']['R2_add'][0]:.2f}) and does not grow downstream.")]
A.FAIL += [("S853", "$k$ = 5 state at B1 / B4 raises the predecessor answer at $p$ = 3 by $\\ge$ .15",
            f"{s2(a853['PRED_A']['diff'][0])} [{s2(a853['PRED_A']['diff'][1][0])}, {s2(a853['PRED_A']['diff'][1][1])}]",
            "the ordinal's state is spread over the read rows; all ten rows carry the step (answer 2: .53)"),
           ("S853", "$p$ = 4 state at B1 / B4 moves the landing toward 3 by $\\ge$ .15",
            f"{s2(a853['PRED_B']['diff'][0])} [{s2(a853['PRED_B']['diff'][1][0])}, {s2(a853['PRED_B']['diff'][1][1])}]",
            "answers moved to the donor's distribution (2: .60 $\\to$ .23) but split between 3 and 4")]


def split_registry(at=("S836", "S848")):
    """The registry outgrew one page (rows after S849-U were cut off in v3 and in the first v4 builds; again after S854-1): split it into
    floats at the experiments in `at`, in the same file, the later ones captioned as continuations."""
    fn = f"{A.OUT}/app_registry.tex"; L = open(fn).read().rstrip("\n").split("\n")
    i0 = L.index("\\midrule") + 1; i1 = L.index("\\bottomrule")
    head, rows, tail = L[:i0], L[i0:i1], L[i1:]
    cuts = [0] + [next(i for i, r in enumerate(rows) if r.startswith(x + " &")) for x in at] + [len(rows)]
    parts = []
    for j in range(len(cuts) - 1):
        lab = "tab:app-registry" if j == 0 else f"tab:app-registry-{'bcdefg'[j - 1]}"
        t2 = tail if j == 0 else ["\\bottomrule", "\\end{tabular}", "\\caption{\\textbf{Registered experiments (continued).}}", f"\\label{{{lab}}}",
                                  "\\end{table*}"]
        parts += head + rows[cuts[j]:cuts[j + 1]] + t2
    open(fn, "w").write("\n".join(parts) + "\n")
    print("split app_registry at", at, [cuts[j + 1] - cuts[j] for j in range(len(cuts) - 1)], "rows")

if __name__ == "__main__":
    for f in (A.exact, A.transfer, A.registry, A.failures, A.mmiusig):
        f()
    split_registry()
