"""v5 registry summary table: the v4 rows plus the image-count pathway (S854-1 to S857) and the held-out predictions (S858), from saved
analysis records only.  Writes research/ces/paper_v5_20261018/tables/registry.tex.  Usage (container): python /vlm/exp/figs/make_v5_tables.py"""
import json

R = "/vlm/results/r3"; OUT = "/vlm/research/ces/paper_v5_20261018/tables"
J = lambda f: json.load(open(f"{R}/{f}"))
pct = lambda x: str(int(100 * x + .5 + 1e-9))
pts = lambda x: ("$+$" if x >= 0 else "$-$") + f"{abs(100 * x):.1f}"
a827 = J("s827_analysis.json")["s827_27b"]["cases"]; a828 = J("s828_analysis.json"); a829 = J("s829_analysis.json")["cases"]
a830 = J("s830_analysis.json")["cases"]; a832 = J("s832_analysis.json"); a833 = J("s833_analysis.json")["cases"]
a846 = J("s846_analysis.json"); a847 = J("s847_analysis.json"); a844 = J("s844_analysis.json")
sd = json.load(open("/vlm/research/ces/outputs/s846_swapdesc.json"))
a848s = J("s848_analysis.json")["REP_sites"]; a850 = J("s850_analysis.json"); u2 = J("s849u_analysis_s849u2_27b.json")["USE"]; a849p = J("s849p_analysis.json"); a851 = J("s851_analysis.json"); a853 = J("s853_analysis.json")
a854 = J("s854a_analysis.json"); a855 = J("s855_analysis.json")["shares"]["L6"]; a856b = J("s856b_analysis.json")["RESULTS"]
a857 = J("s857_analysis.json")["shares"]; a858 = J("s858_analysis.json")
wr = lambda x: f"{abs(x):.2f}" if abs(x) < .005 else f"{x:.2f}"
ns = a846["NOSTOP"]; w = a846["WRITER"]; f7 = a847["f"]
rows = [
 ("Wrong or weakened address?", "neighbour answers whose late address left the target: "
  f"{pct(a827['N7k4']['M1']['moved'])}\\% / {pct(a827['N9k5']['M1']['moved'])}\\%; removing the END address change: "
  f"+{pct(a827['N7k4']['arms']['A_FULL']['dPk'])} / +{pct(a827['N9k5']['arms']['A_FULL']['dPk'])} pts"),
 ("Early or late address?", "late layers restored: "
  f"{pct(a829['N7k4j4']['arms']['U_LATE']['P_j'])}\\% / {pct(a829['N9k5j5']['arms']['U_LATE']['P_j'])}\\%; early layers: "
  f"{pct(a829['N7k4j4']['arms']['U_EARLY']['P_j'])}\\% / {pct(a829['N9k5j5']['arms']['U_EARLY']['P_j'])}\\%; start keeps its ordinal from a moved "
  f"early lookup {pct(a830['N9k5']['F2']['shares']['unmoved'])}\\%"),
 ("Does reaching the target stop the update?", "early lookup pinned to $p$ ($N$=8, fresh scenes): staying at $p$ when the stated target is $p$ "
  f"{pts(ns['D'])} pts [{pts(ns['ci'][0])}, {pts(ns['ci'][1])}]; late preference for $p$ $+${ns['Dh']:.3f} (START--END gap {ns['gap']:.2f}); "
  f"staying depends on $p$: {pts(a846['PDEP']['value'])} pts ($p$=6 vs 3)"),
 ("Is the update written by layers 46--49?", "band outputs swapped at the same early position: END's departures "
  f"{pct(sd['outcome']['E_EARLY']['other'][0])}\\%$\\to${pct(sd['outcome']['BANDF']['other'][0])}\\% with START's band, START's "
  f"{pct(sd['outcome']['S_EARLY']['other'][0])}\\%$\\to${pct(sd['outcome']['BANDE']['other'][0])}\\% with END's band; registered: START off its ordinal "
  f"{pct(w['S_k'])}\\%$\\to${pct(w['B_k'])}\\% (held), END to the ordinal {pct(w['A_k'])}\\% (mixed)"),
 ("Is it read by the late queries?", "band-corrected late queries (51--63) in END: "
  f"{pct(f7['QALL']['f'])}\\% of the correction; END's late queries in the corrected run: {pct(f7['NEC']['g'])}\\% removed; 51+55 alone "
  f"{pct(f7['Q5155']['f'])}\\%, 59+63 alone {pct(f7['Q5963']['f'])}\\%, address part {pct(f7['QALL_U']['f'])}\\%"),
 ("Do fixed laws predict new conditions?", f"gain in log-likelihood per answer: two-route law +0.13 nats over no change (new scenes); two-component law +0.25 over the "
  f"natural distribution, +0.12 over ignoring the shift (new cells; modes {pct(a832['P2']['share'])}\\%); S841-fitted law +0.29 over chance (S844); position law "
  f"+0.17 over a law without the early position (fresh scenes); late-score readout of new interventions: {pct(a847['QKPRED']['calibration'])}\\% of shares within .15 (failed)"),
 ("Does a target-related signal stop the update?", f"held-out signal: boundary B1 $d'$ {a850['REP_B']['sites']['B1_L46']['dprime']:.2f}, answer row "
  f"{a848s['L50_row9']['K50']['dprime']:.2f}; steering B1 / B4 at the writer's input by 16 SD: writer toward start-relative {a850['WRITER']['rho_PLUS16']:.2f} "
  f"(random {a850['WRITER']['rho_RAND16']:.2f}), late hold $+${a850['LATE']['dH_vs_base']:.2f}, predicted from the writer change (sign {a850['PRED']['sign_agree']:.2f}); "
  f"staying {pts(a850['UPDATE']['all_16']['S_P'][0])} pts, not goal-selective (whole boundary state: {pts(a850['POS']['stay_FULLB'] - a850['POS']['stay_E'])} pts); "
  f"the whole-state replacement acts through the late queries 51--63 (sufficient {a851['SUFF']['s']:.2f}, necessary {a851['NEC']['n']:.2f}; S851) but is not goal-selective either"),
 ("What does the writer take in?", f"boundary rows B1 / B4 at layer 46 from another early position: donor's answer {a853['TYPES']['P']['f_don'][0]:.2f} "
  f"(recipient's {a853['TYPES']['P']['f_rec'][0]:.2f}); from another ordinal: {a853['TYPES']['K']['f_don'][0]:.2f} ({a853['TYPES']['K']['f_rec'][0]:.2f}), "
  f"all ten read rows 0.48--0.62; additive fit $R^2$ {a853['INT']['R2_add'][0]:.2f} (S853); answers follow the target counted from the actual end, "
  f"not the stated count, while the writer's output is organised by $k$ (S854-1)"),
 ("Does the image count reach the late queries?", f"answers follow the target counted from the images shown (held-out log-likelihood "
  f"{a854['MAIN_LL']['per_H']['H_a']['vs_full'][0]:+.3f} nats against the full model; ordinal-only {a854['MAIN_LL']['per_H']['H_k']['vs_full'][0]:+.3f}, stated "
  f"count {a854['MAIN_LL']['per_H']['H_s']['vs_full'][0]:+.3f}); positional geometry {a855['G_all'][0]:.3f} of the effect, images 7--8 keep {a855['X_pic'][0]:.2f}; "
  f"swapping only the late queries moves {a856b['A_QX']['g']['T_6from8'][0]:.2f} / {a856b['A_QX']['g']['T_8from6'][0]:.2f} of the count effect"),
 ("Through which route?", f"cutting images 7--8 everywhere gives the six-image state ({a857['CA']['g'][0]:+.2f}); cutting only layers 51--63 keeps "
  f"{a857['CR']['g'][0]:.2f}; full attention needed (cut {a857['CQF']['g'][0]:+.2f}), DeltaNet alone {a857['CQF']['m51'][0]:.2f}, full attention alone "
  f"{a857['CQD']['m51'][0]:.2f} (mixed); window 35--50 {a857['O2']['m51'][0]:.2f}; writer-input rows {a857['RW']['m51A'][0]:+.2f}; late-query restore "
  f"{a857['RQ']['g'][0]:.2f}, removal leaves {a857['KQ']['g'][0]:.2f}"),
 ("Do the pathways predict held-out interventions?", f"complete cut = the $N{{-}}2$ run: late-query cosine {a858['L1']['51']['cos_CA'][0]:.3f}, image "
  f"scores $r$ {a858['L2a_stats']['r'][0]:.2f}; restore {a858['L3a']['RQ']['realized'][0]:.2f} (predicted .80), removal {a858['L3a']['KQ']['realized'][0]:.2f} "
  f"(.17); answers {a858['L3b']['CA']['gain_over']['no_intervention'][0]:+.2f} nats over no change; full-attention-only cut in 35--50 not predicted "
  f"({a858['shares']['CF']['g'][0]:.2f} of the logit-gap effect left)"),
 ("Does the writer's output predict the readers?", "from measured changes of layers 46--49, weights only, new scenes and conditions: late query change cosine "
  + " / ".join(f"{a849p['level1']['per_layer'][L]['median_cos']:.2f}" for L in ('51', '55', '59', '63'))
  + " (layers 51--63); image-score change $r$ " + " / ".join(f"{a849p['level2']['per_layer'][L]['r']:.2f}" for L in ('51', '55', '59', '63'))
  + "; answers +0.41 nats over no change"),
 ("Why must \\rthree{} rebind the frame?", "frame write: late re-reading "
  f"{wr(min(a833[c]['w']['O_FR'][0] for c in a833))}--{wr(max(a833[c]['w']['O_FR'][0] for c in a833))} (start 0, end 1), identity "
  f"{pct(a833['N7k4j4']['P_j']['O_FR'])}\\% / {pct(a833['N9k5j5']['P_j']['O_FR'])}\\%; transformations need both writes "
  f"({pct(a833['N7k5j3']['P_j']['O_ORD_FR'])}\\% / {pct(a833['N9k7j3']['P_j']['O_ORD_FR'])}\\%)"),
]
L = ["\\begin{table*}[t]", "\\centering", "\\small", "\\begin{tabular}{p{4.6cm}p{10.5cm}}", "\\toprule",
     "Registered question & Result (Qwen3.5-27B; identity cells $N$=7 / $N$=9 unless stated) \\\\", "\\midrule"]
for q, r in rows:
    L.append(f"{q} & {r} \\\\")
L += ["\\bottomrule", "\\end{tabular}",
      "\\caption{\\textbf{Each stage was tested by an experiment registered before its data.} Predictions were fixed, with a checksum, before the runs they predict. Full rules, predictions and verdicts, including failed ones, in Appendix~\\ref{app:registry}.}",
      "\\label{tab:registry}", "\\end{table*}"]
open(f"{OUT}/registry.tex", "w").write("\n".join(L) + "\n"); print("\n".join(L))
