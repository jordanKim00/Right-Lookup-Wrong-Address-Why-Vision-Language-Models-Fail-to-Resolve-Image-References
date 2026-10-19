"""v3 appendix tables: the v2 builders (exp/figs/make_v2_appendix.py) with S836-S847 added to the registry and to the failed-prediction table
(verdict wording follows FINDINGS §934-§945 and knowledge/DECISIONS.md; numbers from the saved analysis records).  S831 / S831b keep their
registered verdicts with a note on the later confound (user 10-17 11:48; S844, S846).
Writes research/ces/paper_v3_20261017/tables/{app_exact, app_transfer, app_registry, app_failures, app_mmiusig}.tex.
Usage (container): python /vlm/exp/figs/make_v3_appendix.py"""
import json, sys

sys.path.insert(0, "/vlm/exp/figs")
import make_v2_appendix as A

A.OUT = "/vlm/research/ces/paper_v3_20261017/tables"
R = "/vlm/results/r3"
J = lambda f: json.load(open(f"{R}/{f}"))
pc = A.pc
a844 = J("s844_analysis.json"); a845 = J("s845_analysis.json"); a846 = J("s846_analysis.json"); a847 = J("s847_analysis.json")
f5, f7 = a845["f"], a847["f"]; w6 = a846["WRITER"]; n6 = a846["NOSTOP"]

for i, row in enumerate(A.REG):
    if row[0] in ("S831", "S831b"):
        A.REG[i] = row[:4] + (row[4] + " Later registered tests show this ``toward the target'' contrast coincides with a move to the preceding image "
                              "(S844, S846).",)
A.REG += [
    ("S836", "Oct 17, 08:50", "--", "Up to which layer is the end-relative information of the late queries written?",
     "by the input of layer 50, mostly in layers 48--49 (restore .54 at 48, .99 at 50); the late image attention by layer 48 (.91)."),
    ("S838", "Oct 17, 09:00", "--", "Do Qwen3.5-9B and 4B have the late stage (the S830 / S831 designs)?",
     "no late stage observed: answers stay at the moved early position (.49--.63), the late address does not set the answer (.11--.53), no late image "
     "re-reading ($\\le$0 vs +.25--.30 in 27B)."),
    ("S839", "Oct 17, 09:04", "--", "Is the reading mode the picture-vs-text, menu-word or norm direction of the query; what protects \\START{}?",
     "none of the four (menu $-$.17, all text +.05, norm +.07); \\START{} is protected by its own queries at layers 59 / 63 (break .37 $\\to$ .65)."),
    ("S837", "Oct 17, 09:26", "--", "Which component of layers 46--49 writes the late update; does the dose of its write set the step size?",
     "first runs failed by device placement (kept), clean rerun; no single component (largest .17); direct paths of the band .46 (needed .60); "
     "doubling the write adds two-step answers +.07 (dose not shown); monotone."),
    ("S840", "Oct 17, 10:15", "--", "Do the token mixers or the MLPs carry the update; is there a bottleneck?",
     "token mixers (.37 vs .09); both halves; the band's full patch .99 vs direct paths .57 (later processing needed); no bottleneck "
     "(largest .23)."),
    ("S841", "Oct 17, 10:26", "--", "Does the update fire at a stated tie; does its side follow the stated count?",
     "fires at a tie (+.28); the side follows the stated count only partly ($D$ = .13, needed .30)."),
    ("S842", "Oct 17, 10:35", "--", "Where in the query's non-address part is the reading mode?",
     "text-key subspace (+.30); within-picture pattern +.23 (just below .24); the rest harmful ($-$.17); late question-word reads: nothing."),
    ("S843", "Oct 17, 11:45", "--", "Descriptive (27B only; the text arm was halted by decision): word lists instead of pictures.",
     "same pattern as pictures: target .68, copy .06, identity .15 / .43."),
    ("S844", "Oct 17, 12:03", "0ea4ada9", "With the early lookup pinned and the stated target left of, at or right of it: a predecessor bias, goal "
     "conditioning, a stop at the target? Does a law fitted on S841 predict the new grid?",
     f"no uniform predecessor bias ({a844['stats']['bias']['value']:.2f}, needed .20); conditioning by rule but carried by the early position "
     f"(post hoc within-position +.027, interval includes 0); no stop at the target (+{a844['stats']['stop']['value']:.2f}); the frozen law "
     f"{pc(a844['PRED']['calibration'])}\\% within .15, +.22 / +.23 nats over pure laws."),
    ("S845", "Oct 17, 12:03", "--", "Does the band's correction reach the answer through the 51 / 55 queries; address or reading part?",
     f"{pc(f5['BOTH_B']['f'])}\\% (needed 60\\%); address part {pc(f5['ADDR_B']['f'])}\\%, non-address $-${pc(-f5['READ_B']['f'])}\\%, text-key "
     f"{pc(f5['READT_B']['f'])}\\%."),
    ("S846", "Oct 17, 13:23", "8961408d", "Fresh scenes: does a stated target equal to the early position stop the update; does the band carry the "
     "frame's late rule?", f"no effective stop: +{100 * n6['D']:.1f} pts [$-${-100 * n6['ci'][0]:.1f}, {100 * n6['ci'][1]:.1f}], late preference "
     f"+{n6['Dh']:.3f} vs a gap of {n6['gap']:.2f}; staying depends on $p$ (+{a846['PDEP']['value']:.2f}); position law {pc(a846['PRED']['calibration'])}\\%; "
     f"\\END{{}}'s band takes \\START{{}} off its ordinal ({pc(w6['S_k'])}\\% $\\to$ {pc(w6['B_k'])}\\%); \\START{{}}'s band in \\END{{}} mixed "
     f"($k$ {pc(w6['A_k'])}\\%, $p$ {pc(w6['A_p'])}\\%); swap not shown."),
    ("S847", "Oct 17, 13:23", "290bf68a", "Do the late queries of layers 51--63 carry the band's correction (sufficient, necessary)? Text-key restore "
     "and removal; a late-score readout frozen on S845 (amended 13:29--13:35, before data).",
     f"sufficient {pc(f7['QALL']['f'])}\\%, necessary {pc(f7['NEC']['g'])}\\%; 51 / 55 {pc(f7['Q5155']['f'])}\\%, 59 / 63 {pc(f7['Q5963']['f'])}\\%; "
     f"address part {pc(f7['QALL_U']['f'])}\\%; text-key restore {pc(f7['TRES']['t'])}\\%, removal {pc(f7['TREM']['r'])}\\%; readout "
     f"{pc(a847['QKPRED']['calibration'])}\\% (needed 70\\%)."),
]
a848 = J("s848_analysis.json"); a849u = J("s849u_analysis.json"); a849u2 = J("s849u_analysis_s849u2_27b.json"); a849p = J("s849p_analysis.json")
rs = a848["REP_sites"]
A.REG += [
    ("S848", "Oct 17, 15:11", "--", "Is reaching the target (pinned early position = stated target) represented in the read rows around layers 46--49?",
     f"yes, weakly: answer row $d'$ {rs['L46_row9']['K50']['dprime']:.3f} at the band's input, {rs['L50_row9']['K50']['dprime']:.3f} at its output "
     "(held-out interaction; planted $d'$ = 1 found 20/20); boundary rows larger but missed the decode check; graded by the distance to the target."),
    ("S849-U", "Oct 17, 16:46", "39a138a3", "Is the signal used: does steering the answer row along it ($\\pm$8 SD at layer 46 or 50) stop the update?",
     "no valid run: the reference-run check missed in one stated count (.875 < .90); descriptive steering effects within $\\pm$.04; rerun."),
    ("S849-U2", "Oct 17, 18:29", "39a138a3", "Rerun of S849-U (same rules, new seed, 30 scenes).",
     f"represented but not used to stop the update at both layers (stay vs random {100 * a849u2['USE']['46']['D'][0]:+.1f} / "
     f"{100 * a849u2['USE']['50']['D'][0]:+.1f} pts, intervals within $\\pm$5.3); the steering persists (.72 / .66)."),
    ("S849-P", "Oct 17, 16:46", "32fd3e5e", "From measured changes of layers 46--49, do the model's weights predict the late query, image-score and answer "
     "changes in new scenes and conditions?", "query change at 51 / 55 (cosine .74 / .54; 59 / 63 .49 / .34 missed); image scores at 51 ($r$ .78; later "
     ".36--.52 missed); answer distribution held (calibration .875; +.41 nats over no change, close to the donor's observed answers)."),
]
A.FAIL += [
    ("S837", "doubling the band's write makes two-step answers", "+.07 two-step; mean step +.29 (needed .30)", "the update saturates near one image"),
    ("S839", "the reading mode is the picture-vs-text, menu or norm direction", "$-$.17 / +.05 / +.07", "a finer text-key alignment (S842)"),
    ("S841", "the step's side follows the stated count", "$D$ = .13 (needed .30)", "the move depends mainly on the early position (S844, S846)"),
    ("S844", "a predecessor bias the goal does not override", ".04 (needed .20)", "an anchor-dependent move, not a uniform bias"),
    ("S845", "the 51 / 55 queries carry the band's correction", ".57 (needed .60)", "layers 59 / 63 are also needed (S847: .96)"),
    ("S845", "the band's reading correction lies in the text-key subspace", ".01", "the band does not write the reading-mode difference"),
    ("S846", "\\START{}'s band brings \\END{} back to the ordinal", "$k$ .54, $p$ .36 (needed .60 / .30)", "\\START{} itself returns to $k$ only .66"),
    ("S847", "removing \\START{}'s text-key part breaks \\START{}", "0", "the text-key part is not needed by \\START{}"),
    ("S847", "a late-score readout frozen on S845 predicts new interventions", "60\\% within .15 (needed 70\\%)", "layers 59 / 63 carry part of the answer"),
    ("S849-P", "the late query change is predicted at all four late layers", "51 / 55 yes; 59 .49, 63 .34 (needed .50)", "later layers add processing beyond the direct path"),
    ("S849-P", "the image-score change is predicted at all four late layers", "51 yes; 55--63 .36--.52 (needed .60)", "the direct path predicts the first late layer only"),
]

if __name__ == "__main__":
    for f in (A.exact, A.transfer, A.registry, A.failures, A.mmiusig):
        try:
            f()
        except Exception as e:
            import traceback; traceback.print_exc(); print(f.__name__, "FAILED", e)
