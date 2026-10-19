"""Cross-check the v5 manuscript's new numbers (Acts III-V, discussion, new appendix sections) against the registered analysis values in
research/ces/paper_v5_20261018/numbers_v5.json (written by exp/figs/v5_numbers.py).  Each check names the sentence fragment, the value as
printed, and the source value; it fails if the source value does not round to the printed one.  Numbers carried over from v4 (Acts I-II,
repair, benchmarks) were checked when v4 was built and come from the v4 table and figure builders.
Usage: python3 exp/figs/check_v5_numbers.py"""
import json, re, sys

P = "/home/storage2/ai26269345/vlm/research/ces/paper_v5_20261018"
N = {k: v["value"] for k, v in json.load(open(f"{P}/numbers_v5.json")).items()}
TEXT = {f: open(f"{P}/sections/{f}.tex").read() for f in ("01_introduction", "05_inputs", "06_visual", "07_predict", "09_discussion", "11_appendix")}
ALL = "\n".join(TEXT.values()) + open(f"{P}/main.tex").read()
fails = []


def chk(fragment, printed, value, scale=1.0, nd=None):
    """fragment must occur in the text; value * scale must round to printed (nd decimals inferred from printed)."""
    if fragment not in ALL:
        fails.append(f"MISSING fragment: {fragment!r}")
        return
    s = printed.replace("$-$", "-").replace("−", "-")
    d = nd if nd is not None else (len(s.split(".")[1]) if "." in s else 0)
    if round(value * scale + 1e-12, d) != round(float(s), d) and abs(value * scale - float(s)) > 0.5 * 10 ** (-d) + 1e-9:
        fails.append(f"MISMATCH {fragment!r}: printed {printed}, source {value * scale:.5f}")


# Act III
chk("moves the writer's output 99\\% of the way", "99", N["act3_anchor_rhoW"], 100)
chk("late image scores 90\\%", "90", N["act3_anchor_rhoZ"], 100)
chk("follows the donor in 69\\%", "69", N["act3_anchor_don"], 100)
chk("the recipient in 22\\%", "22", N["act3_anchor_rec"], 100)
chk("moves the writer's output 76\\% of the way", "76", N["act3_ord_rhoW"], 100)
chk("(21\\% against 56\\%)", "21", N["act3_ord_don"], 100)
chk("(21\\% against 56\\%)", "56", N["act3_ord_rec"], 100)
chk("(0.44 and 0.47)", "0.44", N["act3_shapZ_Q"])
chk("(0.44 and 0.47)", "0.47", N["act3_shapZ_A"])
chk("at most 0.03 of the effect", "0.03", N["act3_pair_I_max"])
chk("$R^2=0.32$ for the late scores", "0.32", N["act3_R2add"])
# Act IV
chk("($-0.031$ nats per answer, $[-0.083, 0.015]$)", "-0.031", N["act4_LL_H_a_vs_full"])
chk("($-0.031$ nats per answer, $[-0.083, 0.015]$)", "-0.083", N["act4_LL_H_a_vs_full_ci"][0])
chk("($-0.031$ nats per answer, $[-0.083, 0.015]$)", "0.015", N["act4_LL_H_a_vs_full_ci"][1])
chk("lose 0.129 and 0.136 nats", "0.129", -N["act4_LL_H_k_vs_full"])
chk("lose 0.129 and 0.136 nats", "0.136", -N["act4_LL_H_s_vs_full"])
chk("(class error 0.49, against 1.04", "0.49", N["act4_compact_Hk"])
chk("(class error 0.49, against 1.04", "1.04", N["act4_compact_Ha"])
chk("moves 0.7\\% of the effect", "0.7", N["act4_geom_G_all"], 100)
chk("(0.6\\% when shifted in the late layers alone)", "0.6", N["act4_geom_G_late"], 100)
chk("keep 92\\% of it", "92", N["act4_geom_X_pic"], 100)
chk("99.7\\% of the six-to-eight change", "99.7", N["act4_xeval"]["shares"]["51"]["content_at_th8"][0], 100)
chk("moves 80\\% and 86\\% of this effect", "80", N["act4_swap_A_QX_68"], 100)
chk("moves 80\\% and 86\\% of this effect", "86", N["act4_swap_A_QX_86"], 100)
chk("78\\% and 73\\% when the question states no count", "78", N["act4_swap_NC_QX_68"], 100)
chk("78\\% and 73\\% when the question states no count", "73", N["act4_swap_NC_QX_86"], 100)
lo = min(N["act4_swap_A_QXA_68"], N["act4_swap_A_QXA_86"], N["act4_swap_A_QXU_68"], N["act4_swap_A_QXU_86"])
hi = max(N["act4_swap_A_QXA_68"], N["act4_swap_A_QXA_86"], N["act4_swap_A_QXU_68"], N["act4_swap_A_QXU_86"])
chk("move 74--82\\%", "74", lo, 100); chk("move 74--82\\%", "82", hi, 100)
chk("reproduces the six-image state ($-0.01$ of the effect)", "-0.01", N["act4_cut_CA_g"])
chk("keeps the whole effect (1.02)", "1.02", N["act4_cut_CR_g"])
chk("keeps none ($-0.01$)", "-0.01", N["act4_cut_CQ_g"])
chk("removes the effect ($-0.05$)", "-0.05", N["act4_cut_CQF_g"])
chk("cutting the DeltaNet route keeps 72\\%", "72", N["act4_cut_CQD_g"], 100)
chk("(68\\% of the effect on the layer-51 scores", "68", N["act4_cut_CQD_m51"], 100)
chk("which alone keep 78\\%", "78", N["act4_cut_O2_g"], 100)
chk("brings back 0.02 and $-0.04$", "0.02", N["act4_cut_RW_g"])
chk("brings back 0.02 and $-0.04$", "-0.04", N["act4_cut_RT_g"])
chk("recovers 80\\% of the effect and 66\\%", "80", N["act4_cut_RQ_g"], 100)
chk("recovers 80\\% of the effect and 66\\%", "66", N["act4_cut_answers_RQ_N8"], 100)
chk("queries leaves 17\\%", "17", N["act4_cut_KQ_g"], 100)
chk("keep 68\\% of the", "68", N["act4_cut_G78_g"], 100)
# Act V
chk("(median cosine 0.985 at layer 51)", "0.985", N["act5_L1_cos51"])
chk("($r=0.97$, 94\\% of its variance", "0.97", N["act5_L2_r"])
chk("($r=0.97$, 94\\% of its variance", "94", N["act5_L2_expl"], 100)
chk("0.01 for the cut, 0.84 for the restore and 0.12", "0.01", N["act5_L3a_CA"])
chk("0.01 for the cut, 0.84 for the restore and 0.12", "0.84", N["act5_L3a_RQ"])
chk("0.01 for the cut, 0.84 for the restore and 0.12", "0.12", N["act5_L3a_KQ"])
chk("the prediction gained 1.87 nats per answer", "1.87", N["act5_gainCA_no_intervention"])
chk("from the end (3.39)", "3.39", N["act5_gainCA_actual_target"])
chk("the $N-2$ target (2.77)", "2.77", N["act5_gainCA_target_N_minus_2"])
chk("copying the ordinal (3.66)", "3.66", N["act5_gainCA_pure_copy"])
chk("one-image shift (3.15)", "3.15", N["act5_gainCA_one_step_shift"])
noise = N["act5_noise"]
chk("variation in 76--81\\%", "76", min(noise[a]["share_above_noop_p95"] for a in ("CA", "RQ", "KQ")), 100)
chk("variation in 76--81\\%", "81", max(noise[a]["share_above_noop_p95"] for a in ("CA", "RQ", "KQ")), 100)
chk("removed 78\\% of the effect on the layer-51 scores", "78", 1 - N["act5_shareCF_m51"], 100)
chk("but only 59\\% of the logit-gap effect", "59", 1 - N["act5_L3a_CF"], 100)
chk("reproduced 45\\% of the late-query change", "45", N["act5_L1_projCF"], 100)
chk("(cosine 0.985), the change of the image scores ($r=0.97$) and the restored share of the logit difference (0.84", "0.84", N["act5_L3a_RQ"])
# discussion and appendix
chk("(median $R^2=0.52$)", "0.52", N["disc_slopefit_r2"])
chk("(a peak at the target plus a slope 0.55, a free peak 0.88)", "0.55", N["disc_slopefit_r2_target"])
chk("(a peak at the target plus a slope 0.55, a free peak 0.88)", "0.88", N["disc_slopefit_r2_free"])
chk("(coefficient 0.002, $[-0.018, 0.023]$, against 1.56 needed)", "0.002", N["disc_calibration"])
chk("(coefficient 0.002, $[-0.018, 0.023]$, against 1.56 needed)", "1.56", N["disc_required_unit"])
chk("($-0.081$ per step", "-0.081", N["disc_slope_on_k"])
chk("in 98\\% (eight images) and 96\\% (held-out configurations)", "98", N["act4_cut_instr"], 100)
chk("in 98\\% (eight images) and 96\\% (held-out configurations)", "96", N["act5_instr"], 100)
chk("($+0.38$ nats, $[-0.09, 0.86]$)", "0.38", N["act5_gainCF_noint"][0])
chk("ramp-like in 7 of 8", "7", N["act5_shapes"]["visual_ramp"])
chk("ramp-like in only 2 of 6", "2", N["act5_shapes"]["writer_ramp"])
print(f"{len(fails)} problems")
for f in fails:
    print("  ", f)
sys.exit(1 if fails else 0)
