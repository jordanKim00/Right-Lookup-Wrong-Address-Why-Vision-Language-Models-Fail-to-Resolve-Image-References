"""Named numbers for the v5 manuscript (Acts III-V), read from registered analysis files only (no model calls, no hand-typed values).
Every value records its source file and key path.  Writes research/ces/paper_v5_20261018/numbers_v5.json; check_v5_numbers.py compares the
manuscript against it.  Usage: python3 exp/figs/v5_numbers.py"""
import json

V = "/home/storage2/ai26269345/vlm"; R = f"{V}/results/r3"; O = f"{V}/research/ces/outputs"
OUT = f"{V}/research/ces/paper_v5_20261018/numbers_v5.json"
N = {}


def put(name, value, src, path):
    N[name] = dict(value=value, source=src, path=path)


def J(f):
    return json.load(open(f))


a3 = J(f"{R}/s853_analysis.json"); T = a3["TYPES"]
put("act3_anchor_rhoW", T["P"]["rho_W"][0], "s853_analysis.json", "TYPES.P.rho_W")
put("act3_anchor_rhoZ", T["P"]["rho_Z"][0], "s853_analysis.json", "TYPES.P.rho_Z")
put("act3_anchor_don", T["P"]["f_don"][0], "s853_analysis.json", "TYPES.P.f_don")
put("act3_anchor_rec", T["P"]["f_rec"][0], "s853_analysis.json", "TYPES.P.f_rec")
put("act3_ord_rhoW", T["K"]["rho_W"][0], "s853_analysis.json", "TYPES.K.rho_W")
put("act3_ord_don", T["K"]["f_don"][0], "s853_analysis.json", "TYPES.K.f_don")
put("act3_ord_rec", T["K"]["f_rec"][0], "s853_analysis.json", "TYPES.K.f_rec")
put("act3_R2add", a3["INT"]["R2_add"][0], "s853_analysis.json", "INT.R2_add")
b = J(f"{R}/s854b_analysis.json")
for g in ("Q", "A", "M", "O"):
    put(f"act3_shapZ_{g}", b["SHAPLEY"]["Z"][g], "s854b_analysis.json", f"SHAPLEY.Z.{g}")
    put(f"act3_shapL_{g}", b["SHAPLEY"]["L"][g], "s854b_analysis.json", f"SHAPLEY.L.{g}")
put("act3_pair_I_max", max(v[m]["I"][0] for v in b["PAIRS"].values() for m in v), "s854b_analysis.json", "PAIRS.*.*.I (max)")
put("act3_R2cf", b["CF"]["R2_cf"][0], "s854b_analysis.json", "CF.R2_cf")
a4 = J(f"{R}/s854a_analysis.json"); ML = a4["MAIN_LL"]["per_H"]
for h in ("H_a", "H_k", "H_s"):
    put(f"act4_LL_{h}_vs_full", ML[h]["vs_full"][0], "s854a_analysis.json", f"MAIN_LL.per_H.{h}.vs_full")
    put(f"act4_LL_{h}_vs_full_ci", ML[h]["vs_full"][1], "s854a_analysis.json", f"MAIN_LL.per_H.{h}.vs_full[1]")
put("act4_compact_Hk", a4["COMPACT"]["D_write"]["E"]["H_k"], "s854a_analysis.json", "COMPACT.D_write.E.H_k")
put("act4_compact_Ha", a4["COMPACT"]["D_write"]["E"]["H_a"], "s854a_analysis.json", "COMPACT.D_write.E.H_a")
g5 = J(f"{R}/s855_analysis.json")["shares"]["L6"]
for k_ in ("G_all", "G_late", "G_back", "X_pic"):
    put(f"act4_geom_{k_}", g5[k_][0], "s855_analysis.json", f"shares.L6.{k_}")
xe = J(f"{O}/s855_xeval.json")
put("act4_xeval", xe, "research/ces/outputs/s855_xeval.json", "(whole file; query-content share at layer 51)")
s6 = J(f"{R}/s856b_analysis.json")["RESULTS"]
for arm in ("A_QX", "NC_QX", "A_QXA", "A_QXU", "NC_QXA", "NC_QXU"):
    put(f"act4_swap_{arm}_68", s6[arm]["g"]["T_6from8"][0], "s856b_analysis.json", f"RESULTS.{arm}.g.T_6from8")
    put(f"act4_swap_{arm}_86", s6[arm]["g"]["T_8from6"][0], "s856b_analysis.json", f"RESULTS.{arm}.g.T_8from6")
    put(f"act4_swap_{arm}_verdict", s6[arm]["verdict"], "s856b_analysis.json", f"RESULTS.{arm}.verdict")
a7 = J(f"{R}/s857_analysis.json")
for arm, x in a7["shares"].items():
    for m in ("g", "m51"):
        put(f"act4_cut_{arm}_{m}", x[m][0], "s857_analysis.json", f"shares.{arm}.{m}")
put("act4_cut_INT_g", a7["INT"]["g"]["I_over_G"][0], "s857_analysis.json", "INT.g.I_over_G")
put("act4_cut_V2", a7["V2"], "s857_analysis.json", "V2")
put("act4_cut_answers_CA_N6", a7["answers"]["CA"]["N6_answer"], "s857_analysis.json", "answers.CA.N6_answer")
put("act4_cut_answers_RQ_N8", a7["answers"]["RQ"]["N8_answer"], "s857_analysis.json", "answers.RQ.N8_answer")
put("act4_cut_instr", a7["I0"]["instr"]["v"][0], "s857_analysis.json", "I0.instr.v[0]")
a8 = J(f"{R}/s858_analysis.json")
put("act5_L1_cos51", a8["L1"]["51"]["cos_CA"][0], "s858_analysis.json", "L1.51.cos_CA")
put("act5_L1_cosCF", a8["L1"]["51"]["cos_CF"][0], "s858_analysis.json", "L1.51.cos_CF")
put("act5_L1_projCF", a8["L1"]["51"]["proj_CF"][0], "s858_analysis.json", "L1.51.proj_CF")
put("act5_L2_r", a8["L2a_stats"]["r"][0], "s858_analysis.json", "L2a_stats.r")
put("act5_L2_expl", a8["L2a_stats"]["explained"][0], "s858_analysis.json", "L2a_stats.explained")
for arm in ("CA", "CF", "RQ", "KQ"):
    put(f"act5_L3a_{arm}", a8["L3a"][arm]["realized"][0], "s858_analysis.json", f"L3a.{arm}.realized")
    put(f"act5_L3a_{arm}_ci", a8["L3a"][arm]["realized"][1], "s858_analysis.json", f"L3a.{arm}.realized[1]")
    put(f"act5_L3b_{arm}_verdict", a8["L3b"][arm]["verdict"], "s858_analysis.json", f"L3b.{arm}.verdict")
for bn, v in a8["L3b"]["CA"]["gain_over"].items():
    put(f"act5_gainCA_{bn}", v[0], "s858_analysis.json", f"L3b.CA.gain_over.{bn}")
for arm in ("RQ", "KQ"):
    k_ = "full_restore" if arm == "RQ" else "full_removal"
    put(f"act5_gain{arm}_{k_}", a8["L3b"][arm]["gain_over"][k_][0], "s858_analysis.json", f"L3b.{arm}.gain_over.{k_}")
put("act5_gainCF_noint", a8["L3b"]["CF"]["gain_over"]["no_intervention"], "s858_analysis.json", "L3b.CF.gain_over.no_intervention")
put("act5_shareCF_m51", a8["shares"]["CF"]["m51"][0], "s858_analysis.json", "shares.CF.m51")
put("act5_shapes", a8["shape_counts"], "s858_analysis.json", "shape_counts")
put("act5_noise", a8["noise_vs_effect"], "s858_analysis.json", "noise_vs_effect")
put("act5_instr", a8["I0"]["instr"]["v"][0], "s858_analysis.json", "I0.instr.v[0]")
sf = J(f"{O}/s858_slopefit.json")
put("disc_slopefit_r2", sf["Nn"]["median_r2_bump_p_slope"], "research/ces/outputs/s858_slopefit.json", "Nn.median_r2_bump_p_slope")
put("disc_slopefit_r2_target", sf["Nn"]["median_r2_target_bump_slope"], "research/ces/outputs/s858_slopefit.json", "Nn.median_r2_target_bump_slope")
put("disc_slopefit_r2_free", sf["Nn"]["median_r2_free_peak"], "research/ces/outputs/s858_slopefit.json", "Nn.median_r2_free_peak")
put("disc_calibration", sf["Nn"]["calibration_gamma_on_required"][0], "research/ces/outputs/s858_slopefit.json", "Nn.calibration_gamma_on_required")
put("disc_required_unit", sf["Nn"]["required_unit_per_picture"], "research/ces/outputs/s858_slopefit.json", "Nn.required_unit_per_picture")
put("disc_slope_on_k", sf["Nn"]["gamma_on_k_N_p"]["k"][0], "research/ces/outputs/s858_slopefit.json", "Nn.gamma_on_k_N_p.k")
sh = J(f"{O}/why_shapes.json")
put("disc_start_shift_r2_p4", [sh["ordinal_k5_minus_k3"][f"START_N{n}_p4"]["r2_shift"] for n in (8, 6)], "research/ces/outputs/why_shapes.json", "ordinal_k5_minus_k3.START_N*_p4.r2_shift")
json.dump(N, open(OUT, "w"), indent=1)
print(f"{len(N)} values -> {OUT}")
for k_, v in N.items():
    val = v["value"]
    print(f"  {k_:28s} {json.dumps(val)[:110]}")
