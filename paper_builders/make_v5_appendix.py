"""v5 appendix tables: the v4 builders (exp/figs/make_v4_appendix.py) with S854-2 to S858 added to the registry and their failed registered
predictions added to the failures table.  Numbers come from the registered analysis files only.  Writes research/ces/paper_v5_20261018/tables/.
Usage (container): python /vlm/exp/figs/make_v5_appendix.py"""
import json, sys
sys.path.insert(0, "/vlm/exp/figs")
import make_v4_appendix as V4
A = V4.A
A.OUT = "/vlm/research/ces/paper_v5_20261018/tables"
J = lambda f: json.load(open(f"/vlm/results/r3/{f}"))
s2 = lambda x: ("$+$" if x >= 0 else "$-$") + f"{abs(x):.2f}"
f2 = lambda v: f"{v[0]:.2f} [{v[1][0]:.2f}, {v[1][1]:.2f}]"
b = J("s854b_analysis.json"); g5 = J("s855_analysis.json")["shares"]["L6"]; x5 = json.load(open("/vlm/research/ces/outputs/s855_xeval.json"))["shares"]["51"]
s6 = J("s856_analysis.json"); s6b = J("s856b_analysis.json")["RESULTS"]; a7 = J("s857_analysis.json"); a8 = J("s858_analysis.json")
sf = json.load(open("/vlm/research/ces/outputs/s858_slopefit.json"))
sh7 = a7["shares"]
A.REG += [
 ("S854-2", "Oct 18, 14:28", "--", "Which input rows carry the ordinal into the writer? Rows at layer 46 grouped into question tail, menu, B0 / B2 / B3, "
  "B1 / B4 and thinking / answer rows, every subset swapped between $k$ 3 and $k$ 5; counterfactual combination.",
  f"question tail and B1 / B4 carry the ordinal's effect on the late scores (Shapley {b['SHAPLEY']['Z']['Q']:.2f} / {b['SHAPLEY']['Z']['A']:.2f}) and "
  f"logits ({b['SHAPLEY']['L']['Q']:.2f} / {b['SHAPLEY']['L']['A']:.2f}), menu and B0 / B2 / B3 almost none; pairs additive (interaction at most "
  f"{max(v[m]['I'][0] for v in b['PAIRS'].values() for m in v):.2f}); counterfactual combination $R^2$ {b['CF']['R2_cf'][0]:.2f} (registered `partial')."),
 ("S855", "Oct 18, 15:13", "--", "Positional geometry or the extra images? Text after the images shifted by the two-image advance (20 positions) in "
  "all layers or in the late layers only, both directions.",
  f"not positional geometry (all layers {f2(g5['G_all'])}, reverse {g5['G_back'][0]:.2f}); late layers alone {g5['G_late'][0]:.3f}; images 7--8 under "
  f"the six-image geometry keep {f2(g5['X_pic'])}; post hoc, layer 51: {100 * x5['content_at_th8'][0]:.1f}\\% of the score change is query content."),
 ("S856", "Oct 18, 16:20", "--", "Do the late queries carry the image count? Late queries (51--63) swapped between the six- and eight-image runs of "
  "a scene, two prompt forms.",
  f"no valid test: the self-injection check missed by one run (answers {s6['I0']['self_6']['v'][0]:.3f} against .97, a pre-data amendment), "
  "because donor queries were captured and injected in batches of different shapes; rerun as S856b."),
 ("S856b", "Oct 18, 17:48", "--", "S856 rerun with equal batch shapes and the strict self check.",
  f"self-injection bit-identical; whole query carries the count effect in both directions ({f2(s6b['A_QX']['g']['T_6from8'])}, "
  f"{f2(s6b['A_QX']['g']['T_8from6'])}); without a count sentence {s6b['NC_QX']['g']['T_6from8'][0]:.2f} / {s6b['NC_QX']['g']['T_8from6'][0]:.2f}; "
  f"rows B1, B4, answer {s6b['A_QXA']['g']['T_6from8'][0]:.2f} / {s6b['A_QXA']['g']['T_8from6'][0]:.2f}; address part {s6b['A_QXU']['g']['T_6from8'][0]:.2f} / "
  f"{s6b['A_QXU']['g']['T_8from6'][0]:.2f}; rows only without count sentence partial ({s6b['NC_QXA']['g']['T_6from8'][0]:.2f} / {s6b['NC_QXA']['g']['T_8from6'][0]:.2f})."),
 ("S857", "Oct 18, 18:55", "--", "Through which route and when do images 7--8 reach the late queries? Source cuts by route (full attention, DeltaNet "
  "state and convolution) and window (0--34, 35--50, 51--63); restores of writer-input rows and late queries; content controls.",
  f"leak check bit-identical; complete cut {sh7['CA']['g'][0]:+.2f}; late-only cut {sh7['CR']['g'][0]:.2f}, pre-late cut {sh7['CQ']['g'][0]:+.2f} (through "
  f"text-row state); full attention only {sh7['CQD']['m51'][0]:.3f} on layer-51 scores (registered `mixed'), DeltaNet only {sh7['CQF']['m51'][0]:.2f}; "
  f"window 35--50 alone {sh7['O2']['m51'][0]:.2f}, 0--34 alone {sh7['O1']['m51'][0]:+.2f}; restores at layer 46 {sh7['RW']['m51A'][0]:+.2f} / "
  f"{sh7['RT']['m51A'][0]:+.2f}; late-query restore {sh7['RQ']['g'][0]:.2f}, removal leaves {sh7['KQ']['g'][0]:.2f} (sufficient and necessary); "
  f"grey blocks {sh7['G78']['g'][0]:.2f}, copies {sh7['D78']['g'][0]:.2f}, swap {sh7['SW']['g'][0]:.2f}."),
 ("S858", "Oct 18, 20:36", "c4256492", "Held-out prediction (nine and seven images, new $k$, $p$, scenes): complete cut = the $N{-}2$ run; late-query "
  "restore .80 and removal .17; full-attention-only cut in 35--50 = complete cut; shapes; answers against baselines.",
  f"late-query cosine {a8['L1']['51']['cos_CA'][0]:.3f}; image scores $r$ {a8['L2a_stats']['r'][0]:.2f}, explained {a8['L2a_stats']['explained'][0]:.2f}; "
  f"shares {a8['L3a']['CA']['realized'][0]:.2f} / {a8['L3a']['RQ']['realized'][0]:.2f} / {a8['L3a']['KQ']['realized'][0]:.2f}; answers beat every "
  f"baseline (no change {s2(a8['L3b']['CA']['gain_over']['no_intervention'][0])} nats); full-attention-only cut not predicted (cosine "
  f"{a8['L1']['51']['cos_CF'][0]:.2f}, projection {a8['L1']['51']['proj_CF'][0]:.2f}, shares {a8['shares']['CF']['m51'][0]:.2f} / "
  f"{a8['shares']['CF']['g'][0]:.2f}); shapes held; writer ramp partial; pre-fixed peak-plus-slope fit not sufficient (median $R^2$ "
  f"{sf['Nn']['median_r2_bump_p_slope']:.2f})."),
]
A.FAIL += [
 ("S856b", "rows B1, B4, answer, no count sentence: $\\ge$ .70 in both directions",
  f"{s6b['NC_QXA']['g']['T_6from8'][0]:.2f} / {s6b['NC_QXA']['g']['T_8from6'][0]:.2f}", "whole-query swaps carry the count in both forms"),
 ("S857", "full attention alone carries the count to the layer-51 scores ($\\ge$ .70)", f2(sh7['CQD']['m51']),
  "DeltaNet adds about 30\\% only together with full attention"),
 ("S858", "full-attention-only cut in 35--50 changes the late query like the complete cut (cosine $\\ge$ .80, projection $\\ge$ .70)",
  f"{a8['L1']['51']['cos_CF'][0]:.2f}, {a8['L1']['51']['proj_CF'][0]:.2f}", "part of the count's route lies outside full attention in 35--50"),
 ("S858", "the same cut removes the count effect ($\\le$ .25 on scores and logit gap)",
  f"{a8['shares']['CF']['m51'][0]:.2f} / {a8['shares']['CF']['g'][0]:.2f}", "partial: scores mostly, logit gap less"),
 ("S858", "the same cut: answers beat assuming no change", f2(a8['L3b']['CF']['gain_over']['no_intervention']) + " nats", "not predicted"),
 ("S858", "writer's end-specific write is a ramp in $\\ge$ 75\\% of configurations",
  f"{a8['shape_counts']['writer_ramp']} of {a8['shape_counts']['writer_defined']}", "slopes negative for $k \\ge 5$, shape not reliably a ramp"),
 ("S858", "pre-fixed: a peak at the anchor plus a slope explains the late scores (median $R^2 \\ge$ .70)",
  f"{sf['Nn']['median_r2_bump_p_slope']:.2f}", "no single-slope law; the combination intervention was not run"),
]

if __name__ == "__main__":
    for f in (A.exact, A.transfer, A.registry, A.failures, A.mmiusig):
        f()
    V4.split_registry(at=("S836", "S848", "S853"))
