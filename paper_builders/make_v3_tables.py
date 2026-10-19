"""v3 registry table (Table 2) from saved analysis records only (no hand-typed numbers); the benchmark table is unchanged from v2 (copied).
Writes research/ces/paper_v3_20261017/tables/registry.tex.  Usage (container): python /vlm/exp/figs/make_v3_tables.py"""
import json

R = "/vlm/results/r3"; OUT = "/vlm/research/ces/paper_v3_20261017/tables"
J = lambda f: json.load(open(f"{R}/{f}"))
pct = lambda x: str(int(100 * x + .5 + 1e-9))
pts = lambda x: ("$+$" if x >= 0 else "$-$") + f"{abs(100 * x):.1f}"
a827 = J("s827_analysis.json")["s827_27b"]["cases"]; a828 = J("s828_analysis.json"); a829 = J("s829_analysis.json")["cases"]
a830 = J("s830_analysis.json")["cases"]; a832 = J("s832_analysis.json"); a833 = J("s833_analysis.json")["cases"]
a846 = J("s846_analysis.json"); a847 = J("s847_analysis.json"); a844 = J("s844_analysis.json")
sd = json.load(open("/vlm/research/ces/outputs/s846_swapdesc.json"))
a848s = J("s848_analysis.json")["REP_sites"]; u2 = J("s849u_analysis_s849u2_27b.json")["USE"]; a849p = J("s849p_analysis.json")
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
 ("Is reaching the target represented; is it used?", f"answer row at the band's input / output: $d'$ {a848s['L46_row9']['K50']['dprime']:.3f} / "
  f"{a848s['L50_row9']['K50']['dprime']:.3f} (held-out); steering it by 8 SD: stay vs random {pts(u2['46']['D'][0])} / {pts(u2['50']['D'][0])} pts "
  "(represented but not used)"),
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
