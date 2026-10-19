"""Cross-check the numbers printed in paper v7 (the v6 checks with v7 wording, plus the mock-review additions) (main text and the new appendix pieces) against the raw / registered values in
research/ces/paper_v6_20261019/numbers_v6.json (written by exp/figs/v6_numbers.py; the v5 keys come from numbers_v5.json).
chk: the fragment must occur in the v6 text and the source value must round to the printed value.
carry: numbers carried from v5's verified text (v4 Acts I-II, repair) -- the v6 fragment must occur in v6 and every number in it must occur in
the named v5 fragment, which must occur in v5's text.
Usage: python3 exp/figs/check_v6_numbers.py"""
import json, re, sys
import numpy as np

P6 = "/home/storage2/ai26269345/vlm/research/ces/paper_v7_20261019"
P5 = "/home/storage2/ai26269345/vlm/research/ces/paper_v5_20261018"
N = {k: v["value"] for k, v in json.load(open(f"{P6}/numbers_v7.json")).items()}
MAIN = ["01_introduction", "02_setup", "03_lookup", "04_frame", "05_rebind", "06_restore", "07_related", "08_discussion", "09_limitations"]
APP = ["10_appendix"]  # the compiled appendix (assembled from v5 blocks and sections/app_*.tex)
T6 = "\n".join(open(f"{P6}/sections/{f}.tex").read() for f in MAIN + APP) + open(f"{P6}/main.tex").read()
T6 += "\n".join(open(f"{P6}/tables/{f}.tex").read() for f in ("rebind", "bench_main", "bench_full", "lure", "cos"))
T5 = "\n".join(open(f"{P5}/sections/{f}").read() for f in ("01_introduction.tex", "02_design.tex", "03_paradox.tex", "04_update.tex",
                                                          "07_predict.tex", "11_appendix.tex"))
fails, n = [], [0]


def chk(fragment, printed, value, scale=1.0):
    n[0] += 1
    if fragment not in T6:
        fails.append(f"MISSING fragment: {fragment!r}"); return
    s = printed.replace("$-$", "-").replace("−", "-")
    d = len(s.split(".")[1]) if "." in s else 0
    v = value * scale
    if abs(round(v + 1e-9, d) - float(s)) > 1e-9 and abs(v - float(s)) > 0.5 * 10 ** (-d) + 1e-9:
        fails.append(f"MISMATCH {fragment!r}: printed {printed}, source {v:.5f}")


def carry(f6, f5):
    n[0] += 1
    if f6 not in T6:
        fails.append(f"MISSING v6 fragment: {f6!r}"); return
    if f5 not in T5:
        fails.append(f"MISSING v5 fragment: {f5!r}"); return
    nums = lambda t: set(re.findall(r"\d+(?:\.\d+)?", t))
    extra = nums(f6) - nums(f5)
    if extra:
        fails.append(f"CARRY {f6!r}: numbers {sorted(extra)} not in the v5 fragment")


rng = lambda xs: (min(xs), max(xs))
S = {d["model"]: d for d in N["scale"]}
small = ["VL-4B", "VL-8B", "3.5-4B", "3.5-9B"]
B = N["bench"]
# ---- abstract and introduction
lo, hi = rng([B[f"{b}|MMIU held-out"]["F"] for b in ("4b", "8b", "35_4b", "35_9b", "27b")])
chk("from 66--86\\% to 26--32\\%", "66", lo, 100); chk("from 66--86\\% to 26--32\\%", "86", hi, 100)
lo, hi = rng([B[f"{b}|MMIU held-out"]["R"] for b in ("4b", "8b", "35_4b", "35_9b", "27b")])
chk("from 66--86\\% to 26--32\\%", "26", lo, 100); chk("from 66--86\\% to 26--32\\%", "32", hi, 100)
carry("start-relative number in 94\\% of items, yet answers the end-relative wording in 26\\%", "converts every end-relative option to its start-relative number in 94\\%, yet answers the end-relative wording in 26\\%")
chk("reaches 62\\% of end-relative targets", "62", S["3.5-27B"]["target"], 100)
chk("27B names it in 45\\% of scenes, against 98\\% for Qwen3.5-4B", "45", S["3.5-27B"]["identity"], 100)
chk("27B names it in 45\\% of scenes, against 98\\% for Qwen3.5-4B", "98", S["3.5-4B"]["identity"], 100)
chk("in 45\\% of scenes, against 98\\% for Qwen3.5-4B. With", "98", S["3.5-4B"]["identity"], 100)
# ---- Section 3
for m in ("4b", "8b", "35_4b", "35_9b", "27b"):
    chk("23 of 24 references with $k\\geq3$, in each of the five models", "23", N["idx_shift"][m][0])
    chk("23 of 24 references with $k\\geq3$, in each of the five models", "24", N["idx_shift"][m][1])
chk("reproduces it in 87\\% and 99\\% of answers", "87", N["kv"]["4b"]["KV"], 100)
chk("reproduces it in 87\\% and 99\\% of answers", "99", N["kv"]["9b"]["KV"], 100)
tk = N["tokens"]
chk("blocks of 64 to 441 tokens", "64", min(tk[m]["tokens"][0] for m in tk)); chk("blocks of 64 to 441 tokens", "441", max(tk[m]["tokens"][1] for m in tk))
chk("start-relative accuracy at 95--99\\%", "95", min(min(tk[m]["UNI"], tk[m]["MIX"]) for m in tk), 100)
chk("start-relative accuracy at 95--99\\%", "99", max(max(tk[m]["UNI"], tk[m]["MIX"]) for m in tk), 100)
hs = N["heads_start_share"]["table"]
chk("block 58--64\\% of their image attention", "58", min(hs.values())); chk("block 58--64\\% of their image attention", "64", max(hs.values()))
chk("block 58--64\\% of their image attention", "58", N["heads_start_share"]["qa_27b_from_capture"], 100)
chk("against 14\\% for a uniform spread", "14", 1 / 7, 100)
ac = N["address_control_vl4b"]
chk("selects the image now at that address in 252 of 258 cells", "252", ac["dest"]["OF"]["j"] * ac["cells"])
chk("a wrong address selects the image at the wrong address in 252 of 258", "252", ac["dest"]["WF"]["w"] * ac["cells"])
w = [v for L in N["wrong_addr_27b"].values() for v in L.values()]
chk("lands on the written image in 97--100\\% of questions for every $k$", "97", min(w), 100)
chk("lands on the written image in 97--100\\% of questions for every $k$", "100", max(w), 100)
chk("leaves 39--40 of 40 answers unchanged", "39", min(N["idx_shift_ctrl"][m][0] for m in N["idx_shift_ctrl"]))
chk("leaves 39--40 of 40 answers unchanged", "40", max(N["idx_shift_ctrl"][m][0] for m in N["idx_shift_ctrl"]))
# ---- Section 4
sn = N["selnn"]
chk("same image (37--82\\% of head--scene pairs)", "37", min(sn[m][k]["comp"] for m in sn for k in ("1", "2")), 100)
chk("same image (37--82\\% of head--scene pairs)", "82", max(sn[m][k]["comp"] for m in sn for k in ("1", "2")), 100)
cp = [sn[m][str(k)]["copy"] for m in ("qa_4b", "qa_9b") for k in range(3, 8)]
chk("copied ordinal (52--95\\%)", "52", min(cp), 100); chk("copied ordinal (52--95\\%)", "95", max(cp), 100)
chk("name the copy in 46--84\\% of non-identity cases", "46", min(S[m]["copy"] for m in small), 100)
chk("name the copy in 46--84\\% of non-identity cases", "84", max(S[m]["copy"] for m in small), 100)
chk("reach the target in 4--21\\%", "4", min(S[m]["target"] for m in small), 100)
chk("reach the target in 4--21\\%", "21", max(S[m]["target"] for m in small), 100)
dj = [N["deref"][m][L]["j"] for m in ("4b", "8b") for L in ("2", "5", "8", "10")]
chk("position $j$ in 73--85\\% of questions", "73", min(dj), 100); chk("position $j$ in 73--85\\% of questions", "85", max(dj), 100)
qg = N["query_vs_gate"]
chk("repairs 92\\% and 64\\% of failed", "92", qg["4b"]["Q_ONLY"], 100); chk("repairs 92\\% and 64\\% of failed", "64", qg["9b"]["Q_ONLY"], 100)
chk("only the output gate 1\\% and 2\\%", "1", qg["4b"]["G_ONLY"], 100); chk("only the output gate 1\\% and 2\\%", "2", qg["9b"]["G_ONLY"], 100)
chk("names the copy in 5\\% of non-identity cases", "5", S["3.5-27B"]["copy"], 100)
chk("reaches the target in 62\\%, yet", "62", S["3.5-27B"]["target"], 100)
chk("in only 45\\% of scenes, where the smaller models reach 78--98\\%", "45", S["3.5-27B"]["identity"], 100)
chk("in only 45\\% of scenes, where the smaller models reach 78--98\\%", "78", min(S[m]["identity"] for m in small), 100)
chk("in only 45\\% of scenes, where the smaller models reach 78--98\\%", "98", max(S[m]["identity"] for m in small), 100)
fs = N["frame_switch"]["N7"]["ans"]
chk("drops the answer from 100\\% to 52\\%", "100", fs["S4"]["4"], 100); chk("drops the answer from 100\\% to 52\\%", "52", fs["E4"]["4"], 100)
chk("restores it to 100\\%. The lookup queries", "100", fs["F4:P1"]["4"], 100)
chk("carry 95\\% and 97\\% of the frame read's effect", "95", N["frame_queries"]["4"]["MF"], 100)
chk("carry 95\\% and 97\\% of the frame read's effect", "97", N["frame_queries"]["5"]["MF"], 100)
carry("(slope 0.44--0.48), and from $p\\geq4$ it moves to the preceding image in 30--52\\% of scenes", "(slope 0.44--0.48), while \\START{} questions return to their ordinal under the same edit in 66\\% of scenes. How the end-relative answer moves depends on $p$ (Figure~\\ref{fig:stop}a). With eight images and $k=4$, it stays at $p=3$ in 3\\% of scenes and at $p=4$--6 in 38--44\\%, and from $p\\ge4$ it moves to the preceding image in 30--52\\% of scenes")
carry("by $+0.6$ points ($[-3.1, 3.9]$)", "by $+0.6$ points ($[-3.1, 3.9]$)")
carry("removes 99\\% of the update's effect, and no single component carries more than 23\\%", "removes 99\\% of the update's effect, and no single component carries more than 23\\%")
carry("reproduce 96\\% of the correction, and withholding them removes 93\\%", "reproduces 96\\% of the correction, and withholding them from the corrected run removes 93\\%")
carry("33--88\\% of the answers fall strictly between the two", "33--88\\% of the answers fall strictly between the two")
g9 = N["grid"]["N9_k5"]
chk("lands on image 4 in about half of the scenes", "0.5", g9["4"]); chk("on image 6 in a third", "0.33", round(g9["6"], 2))
for frag, pr, key, sc in (("($-0.031$ nats per answer, $[-0.083, 0.015]$)", "-0.031", "act4_LL_H_a_vs_full", 1),
                          ("lose 0.129 and 0.136 nats", "0.129", "act4_LL_H_k_vs_full", -1), ("lose 0.129 and 0.136 nats", "0.136", "act4_LL_H_s_vs_full", -1),
                          ("moves 80\\% and 86\\% of this effect", "80", "act4_swap_A_QX_68", 100), ("moves 80\\% and 86\\% of this effect", "86", "act4_swap_A_QX_86", 100),
                          ("Positional geometry contributes 0.7\\%", "0.7", "act4_geom_G_all", 100), ("geometry keep 92\\% of it", "92", "act4_geom_X_pic", 100),
                          ("reproduces the six-image state ($-0.01$ of the effect)", "-0.01", "act4_cut_CA_g", 1),
                          ("keeps the whole effect (1.02)", "1.02", "act4_cut_CR_g", 1), ("removes the effect ($-0.05$)", "-0.05", "act4_cut_CQF_g", 1),
                          ("DeltaNet route keeps 72\\%", "72", "act4_cut_CQD_g", 100), ("(68\\% on the layer-51 scores", "68", "act4_cut_CQD_m51", 100),
                          ("which alone keep 78\\%", "78", "act4_cut_O2_g", 100), ("recovers 80\\% of the effect, and removing them leaves 17\\%", "80", "act4_cut_RQ_g", 100),
                          ("recovers 80\\% of the effect, and removing them leaves 17\\%", "17", "act4_cut_KQ_g", 100)):
    chk(frag, pr, N[key], sc)
chk("($-0.031$ nats per answer, $[-0.083, 0.015]$)", "-0.083", N["act4_LL_H_a_vs_full_ci"][0]); chk("($-0.031$ nats per answer, $[-0.083, 0.015]$)", "0.015", N["act4_LL_H_a_vs_full_ci"][1])
# ---- Section 5
ra = N["rebind_acc"]
sm = ("4b", "8b", "35_4b", "35_9b")
chk("the frame alone moves nothing where the address must change (0\\% in every model)", "0", max(ra[b]["k!=4"]["FRAME"] for b in ra), 100)
chk("in the smaller models (59--77\\%)", "59", min(ra[b]["k!=4"]["ADDR"] for b in sm), 100)
chk("in the smaller models (59--77\\%)", "77", max(ra[b]["k!=4"]["ADDR"] for b in sm), 100)
chk("fails in Qwen3.5-27B (2\\%)", "2", ra["27b"]["k!=4"]["ADDR"], 100)
chk("(correlation .995 across cells in 27B", "0.995", N["reflect"]["27b"]["pearson"])
chk("in every model (96--100\\%)", "96", min(ra[b]["k!=4"]["R3"] for b in ra), 100); chk("in every model (96--100\\%)", "100", max(ra[b]["k!=4"]["R3"] for b in ra), 100)
chk("restored by the frame alone (97--100\\%)", "97", min(ra[b]["k=4"]["FRAME"] for b in ra), 100)
chk("restored by the frame alone (97--100\\%)", "100", max(ra[b]["k=4"]["FRAME"] for b in ra), 100)
L7 = N["s721_later"]
for pr, v in (("34.4", L7["full"]["base"]), ("57.8", L7["no_frame"]["rep"]), ("30.2", L7["no_ord"]["rep"]), ("82.0", L7["full"]["rep"])):
    chk("27B answers 34.4\\% unedited, 57.8\\% with the address alone, 30.2\\% with the frame alone and 82.0\\% with both (Appendix", pr, v, 100)
for pr, v in (("34.4", L7["full"]["base"]), ("57.8", L7["no_frame"]["rep"]), ("30.2", L7["no_ord"]["rep"]), ("82.0", L7["full"]["rep"]), ("31.6", L7["rand_target"]["rep"])):
    chk("Qwen3.5-27B answers 34.4\\% unedited, 57.8\\% with the address alone, 30.2\\% with the frame alone and 82.0\\% with both writes, against 85.2\\% for the start-relative wording and 31.6\\%", pr, v, 100)
chk("against 85.2\\% for the start-relative wording and 31.6\\%", "85.2", json.load(open("/home/storage2/ai26269345/vlm/results/r3/s721_analysis.json"))["ref"]["forward(normalization)"], 100)
# ---- Section 6
nq = N["natural_q"]
chk("raises its accuracy from 51\\% to 99\\%", "51", nq["END"], 100); chk("raises its accuracy from 51\\% to 99\\%", "99", nq["ES_look"], 100)
chk("lowers it from 98\\% to 62\\%", "98", nq["START"], 100); chk("lowers it from 98\\% to 62\\%", "62", nq["SE_look"], 100)
chk("reproducing the wrong end-relative answer in 58\\%", "58", nq["SE_reproduces_wrong"], 100)
rt = N["r3_through_queries"]
chk("removes 85--89\\% of its gain", "85", min(rt[x]["removed59"] for x in rt), 100); chk("removes 85--89\\% of its gain", "89", max(rt[x]["removed59"] for x in rt), 100)
chk("supplies 97--100\\%", "97", min(rt[x]["supplied"] for x in rt), 100); chk("supplies 97--100\\%", "100", max(rt[x]["supplied"] for x in rt), 100)
pq = N["photo_queries"]
chk("repair 63--93\\% of the items", "63", min(pq[t]["pick_END_QALL"] for t in pq), 100); chk("repair 63--93\\% of the items", "93", max(pq[t]["pick_END_QALL"] for t in pq), 100)
pf = N["picture_follow"]
chk("99\\% of rebound answers change to the new object at seven images and 100\\% at eight", "99", pf["N7"]["visual_r3"]["j"], 100)
chk("99\\% of rebound answers change to the new object at seven images and 100\\% at eight", "100", pf["N8"]["visual_r3"]["j"], 100)
chk("reversing the menu changes 1 of 80", "1", pf["N7"]["menu_reversal_change"]["R3"] * 80); chk("reversing the menu changes 1 of 80", "1", pf["N8"]["menu_reversal_change"]["R3"] * 80)
chk("(84\\% and 76\\%)", "84", pf["N7"]["visual_e2"]["j"], 100); chk("(84\\% and 76\\%)", "76", pf["N8"]["visual_e2"]["j"], 100)
chk("follow it in only 15\\% and 32\\%", "15", pf["N7"]["visual_end"]["j"], 100); chk("follow it in only 15\\% and 32\\%", "32", pf["N8"]["visual_end"]["j"], 100)
for frag, pr, key, sc in (("(median cosine 0.985 at layer 51)", "0.985", "act5_L1_cos51", 1), ("($r=0.97$, 94\\% of its variance", "0.97", "act5_L2_r", 1),
                          ("($r=0.97$, 94\\% of its variance", "94", "act5_L2_expl", 100), ("0.01 for the cut, 0.84 for the restore and 0.12", "0.01", "act5_L3a_CA", 1),
                          ("0.01 for the cut, 0.84 for the restore and 0.12", "0.84", "act5_L3a_RQ", 1), ("0.01 for the cut, 0.84 for the restore and 0.12", "0.12", "act5_L3a_KQ", 1),
                          ("the prediction gained 1.87 nats per answer", "1.87", "act5_gainCA_no_intervention", 1), ("from the end (3.39)", "3.39", "act5_gainCA_actual_target", 1),
                          ("the $N-2$ target (2.77)", "2.77", "act5_gainCA_target_N_minus_2", 1), ("copying the ordinal (3.66)", "3.66", "act5_gainCA_pure_copy", 1),
                          ("one-image shift (3.15)", "3.15", "act5_gainCA_one_step_shift", 1), ("reproduced 45\\% of the late-query change", "45", "act5_L1_projCF", 100)):
    chk(frag, pr, N[key], sc)
chk("removed 78\\% of the effect on the layer-51 scores", "78", 1 - N["act5_shareCF_m51"], 100)
chk("but only 59\\% of the logit-gap effect", "59", 1 - N["act5_L3a_CF"], 100)
lu = N["lure"]; four = ("f7sl_4b", "f7sl_8b", "f7sl_35_9b", "c8sl_27b")
chk("66--81\\% of the errors of four of the five models", "66", min(lu[t]["lure_share"] for t in four), 100)
chk("66--81\\% of the errors of four of the five models", "81", max(lu[t]["lure_share"] for t in four), 100)
chk("against 35\\% for a uniform choice", "35", np.mean([lu[t]["chance"] for t in lu]), 100)
cz = lambda b, k: B[f"{b}|MMIU cloze"][k]
chk("it raises 27B from 26.5\\% to 41.5\\% ($+15.0$ points, $[9.0, 21.0]$)", "26.5", cz("27b", "R"), 100)
chk("it raises 27B from 26.5\\% to 41.5\\% ($+15.0$ points, $[9.0, 21.0]$)", "41.5", cz("27b", "rep"), 100)
bp = N["bench_paired"]["27b|cloze|R|base"]
chk("($+15.0$ points, $[9.0, 21.0]$)", "15.0", bp[0], 100); chk("($+15.0$ points, $[9.0, 21.0]$)", "9.0", bp[1], 100); chk("($+15.0$ points, $[9.0, 21.0]$)", "21.0", bp[2], 100)
chk("Qwen3-VL-4B from 34.0\\% to 56.0\\%", "34.0", cz("4b", "R"), 100); chk("Qwen3-VL-4B from 34.0\\% to 56.0\\%", "56.0", cz("4b", "rep"), 100)
chk("8B from 32.0\\% to 46.0\\%", "32.0", cz("8b", "R"), 100); chk("8B from 32.0\\% to 46.0\\%", "46.0", cz("8b", "rep"), 100)
clos = [(B[f"{b}|MMIU held-out"]["rep"] - B[f"{b}|MMIU held-out"]["R"]) / (B[f"{b}|MMIU held-out"]["F"] - B[f"{b}|MMIU held-out"]["R"]) for b in ("4b", "8b", "35_4b", "35_9b", "27b")]
chk("closes at least 95\\% of the gap", "95", np.floor(min(clos) * 100) / 100, 100)
# ---- Discussion
chk("(median $R^2$ of 0.52 for a fit fixed in advance", "0.52", N["disc_slopefit_r2"])
# ---- appendix pieces (new in v6)
chk("leaves the answer as it was in 39 of 40 references in Qwen3-VL-4B", "39", N["idx_shift_ctrl"]["4b"][0])
for m in ("8b", "35_4b", "35_9b", "27b"):
    chk("in 40 of 40 in the other four models", "40", N["idx_shift_ctrl"][m][0])
k4, k9 = N["kv"]["4b"], N["kv"]["9b"]
for pr, v in (("43", k4["KEY"]), ("84", k9["KEY"])):
    chk("moves 43\\% of the answers two images back in Qwen3-VL-4B and 84\\% in Qwen3.5-9B", pr, v, 100)
chk("only their values moves 37\\% and 30\\%", "37", k4["VAL"], 100); chk("only their values moves 37\\% and 30\\%", "30", k9["VAL"], 100)
chk("as much as the full exchange (88\\% and 99\\%)", "88", k4["FULL"], 100); chk("as much as the full exchange (88\\% and 99\\%)", "99", k9["FULL"], 100)
chk("(.43 against .44)", "0.43", k4["KEY"]); chk("(.43 against .44)", "0.44", 0.5 * k4["FULL"])
for m, a, b in (("4b", "96", "95"), ("8b", "97", "99"), ("35_4b", "99", "97"), ("35_9b", "98", "99"), ("27b", "99", "99")):
    frag = {"4b": "96 and 95\\% (Qwen3-VL-4B)", "8b": "97 and 99\\% (8B)", "35_4b": "99 and 97\\% (Qwen3.5-4B)", "35_9b": "98 and 99\\% (9B)", "27b": "99 and 99\\% (27B)"}[m]
    chk(frag, a, tk[m]["UNI"], 100); chk(frag, b, tk[m]["MIX"], 100)
chk("the differences are 97.7 and 96.9 points", "97.7", ac["correct_minus_random"], 100); chk("the differences are 97.7 and 96.9 points", "96.9", ac["wrong_minus_random"], 100)
chk("leave 97\\% of the answers on the copy", "97", ac["dest"]["RJF"]["k"], 100)
chk("41--63\\% of pairs nearest to some other block", "41", min(sn["qa_27b"][str(k)]["other"] for k in range(3, 7)), 100)
chk("41--63\\% of pairs nearest to some other block", "63", max(sn["qa_27b"][str(k)]["other"] for k in range(3, 7)), 100)
chk("the first image (67\\%)", "67", sn["qa_27b"]["7"]["comp"], 100)
d4 = [N["deref"]["4b"][L]["j"] for L in ("2", "5", "8", "10")]; d8 = [N["deref"]["8b"][L]["j"] for L in ("2", "5", "8", "10")]
chk("in 73--79\\% of questions at layers 2--10 in Qwen3-VL-4B", "73", min(d4), 100); chk("in 73--79\\% of questions at layers 2--10 in Qwen3-VL-4B", "79", max(d4), 100)
chk("in 82--85\\% in 8B", "82", min(d8), 100); chk("in 82--85\\% in 8B", "85", max(d8), 100)
chk("on the donor's image in at most 3\\%", "3", max(N["deref"][m][L]["y"] for m in ("4b", "8b") for L in ("2", "5", "8", "10")), 100)
chk("(layer 16: 39\\% and 72\\%", "39", N["deref"]["4b"]["16"]["j"], 100); chk("(layer 16: 39\\% and 72\\%", "72", N["deref"]["8b"]["16"]["j"], 100)
chk("layer 19: 9\\% and 33\\%)", "9", N["deref"]["4b"]["19"]["j"], 100); chk("layer 19: 9\\% and 33\\%)", "33", N["deref"]["8b"]["19"]["j"], 100)
kk = [N["deref"][m][L]["k"] for m in ("4b", "8b") for L in ("22", "28")]
chk("as unedited (77--87\\%)", "77", min(kk), 100); chk("as unedited (77--87\\%)", "87", max(kk), 100)
I4 = N["transplant_I4"]
for m, pr in (("4b", "76"), ("8b", "83"), ("35_4b", "59"), ("35_9b", "46")):
    chk("The answers land on $x$ in 76\\% (Qwen3-VL-4B), 83\\% (8B), 59\\% (Qwen3.5-4B) and 46\\% (9B)", pr, I4[m]["copy"][0], 100)
for m, pr in (("4b", "4"), ("8b", "4"), ("35_4b", "10"), ("35_9b", "16")):
    chk("and on $N+1-x$ in 4\\%, 4\\%, 10\\% and 16\\%", pr, I4[m]["comp"][0], 100)
chk("4\\% on $x$ and 53\\% on $N+1-x$", "4", I4["27b"]["copy"][0], 100); chk("4\\% on $x$ and 53\\% on $N+1-x$", "53", I4["27b"]["comp"][0], 100)
chk("repairs 92\\% and 64\\% of the failures (76 and 91)", "76", qg["4b"]["n"]); chk("repairs 92\\% and 64\\% of the failures (76 and 91)", "91", qg["9b"]["n"])
chk("and both 93\\% and 64\\%", "93", qg["4b"]["QG"], 100); chk("and both 93\\% and 64\\%", "64", qg["9b"]["QG"], 100)
F7, F8 = N["frame_switch"]["N7"], N["frame_switch"]["N8"]
chk("from 24\\% to 64\\% of the image attention (start-relative: 69\\%)", "24", F7["share4_L39"]["E4"], 100)
chk("from 24\\% to 64\\% of the image attention (start-relative: 69\\%)", "64", F7["share4_L39"]["F4:P1"], 100)
chk("from 24\\% to 64\\% of the image attention (start-relative: 69\\%)", "69", F7["share4_L39"]["S4"], 100)
chk("leaves the layer-39 lookup at 24\\% and restores the answer in part (78\\%)", "24", F7["share4_L39"]["F4:P23"], 100)
chk("leaves the layer-39 lookup at 24\\% and restores the answer in part (78\\%)", "78", F7["ans"]["F4:P23"]["4"], 100)
chk("sends 98\\% of the answers to image 4 (Figure", "98", F8["ans"]["F4:P1"]["4"], 100)
D769 = json.load(open("/home/storage2/ai26269345/vlm/results/r3/s769_analysis_bare.json"))["N7"]
chk("the lookup returns most of the way (57\\%) and the answers rise to 78\\%", "57", D769["share4_L39"]["F4:P1conv"], 100)
chk("the lookup returns most of the way (57\\%) and the answers rise to 78\\%", "78", D769["dist"]["F4:P1conv"]["4"], 100)
chk("the answers rise to 68\\% while the lookup stays low (29\\%)", "68", D769["dist"]["F4:P1fa"]["4"], 100)
chk("the answers rise to 68\\% while the lookup stays low (29\\%)", "29", D769["share4_L39"]["F4:P1fa"], 100)
chk("halves the lookup (34\\%) but leaves 95\\% of the answers correct", "34", D769["share4_L39"]["R4:P1"], 100)
chk("halves the lookup (34\\%) but leaves 95\\% of the answers correct", "95", D769["dist"]["R4:P1"]["4"], 100)
Ln = N["lens"]
st32 = [Ln["39"][str(t)]["start_target"][32] for t in range(1, 8)]; st35 = [Ln["39"][str(t)]["start_target"][35] for t in range(1, 8)]
chk("(start-relative target shares .13--.28 after layer 32", "0.13", min(st32)); chk("(start-relative target shares .13--.28 after layer 32", "0.28", max(st32))
chk("rises to .44--.96 after layer 35", "0.44", min(st35)); chk("rises to .44--.96 after layer 35", "0.96", max(st35))
lit = [Ln["39"][str(k)]["end_literal"][35] for k in (3, 5, 6)]; tg = [Ln["39"][str(k)]["end_target"][35] for k in (3, 5, 6)]
chk("puts .27--.36 on the copy and .12--.14 on the target", "0.27", min(lit)); chk("puts .27--.36 on the copy and .12--.14 on the target", "0.36", max(lit))
chk("puts .27--.36 on the copy and .12--.14 on the target", "0.12", min(tg)); chk("puts .27--.36 on the copy and .12--.14 on the target", "0.14", max(tg))
chk("puts .27 on image 4 against .93", "0.27", Ln["39"]["4"]["end_target"][35]); chk("puts .27 on image 4 against .93", "0.93", Ln["39"]["4"]["start_target"][35])
chk("(.22 after layer 33)", "0.22", Ln["39"]["2"]["end_literal"][33]); chk("(.69 after layer 50 for $L^*=51$", "0.69", Ln["51"]["2"]["end_target"][50])
fq = N["frame_queries"]
chk("from 47.5\\% to 100\\%, and putting the queries back returns them to 47.5\\%", "47.5", fq["4"]["m"]["E"], 100)
chk("from 47.5\\% to 100\\%, and putting the queries back returns them to 47.5\\%", "47.5", fq["4"]["m"]["FRQB"], 100)
chk("95\\% of the effect ($[85, 100]$)", "95", fq["4"]["MF"], 100); chk("95\\% of the effect ($[85, 100]$)", "85", fq["4"]["MF_ci"][0], 100)
chk("sends 92.5\\% of the answers to the copy, image 5 (2.5\\% unedited)", "92.5", fq["5"]["m"]["FR"], 100)
chk("sends 92.5\\% of the answers to the copy, image 5 (2.5\\% unedited)", "2.5", fq["5"]["m"]["E"], 100)
chk("the queries put back return 5\\%, 97\\% of the effect ($[88, 106]$)", "5", fq["5"]["m"]["FRQB"], 100)
chk("the queries put back return 5\\%, 97\\% of the effect ($[88, 106]$)", "106", fq["5"]["MF_ci"][1], 100)
M796 = json.load(open("/home/storage2/ai26269345/vlm/results/r3/s796_analysis.json"))["s796_27b"]["M4"]
chk("changes 83\\% of the answers that landed there to the new picture and 12\\% of the others", "83", M796["follow"], 100)
chk("changes 83\\% of the answers that landed there to the new picture and 12\\% of the others", "12", M796["leak"], 100)
hd = N["hide"]
chk("puts .49 of the heads' image attention on image 6", "0.49", hd["NONE"]["E2"][5])
chk("Hiding image 7 moves it to image 5 (.41, image 6: .23)", "0.41", hd["LAST1"]["E2"][4]); chk("Hiding image 7 moves it to image 5 (.41, image 6: .23)", "0.23", hd["LAST1"]["E2"][5])
chk("keep .60 on their targets (.61 unedited)", "0.60", hd["LAST1"]["S5"][4]); chk("keep .60 on their targets (.61 unedited)", "0.60", hd["LAST1"]["S6"][5])
chk("keep .60 on their targets (.61 unedited)", "0.61", hd["NONE"]["S5"][4]); chk("keep .60 on their targets (.61 unedited)", "0.61", hd["NONE"]["S6"][5])
e6 = [hd[a]["E2"][5] for a in ("MID1", "FIRST1", "FIRST2")]; stt = [hd[a]["S5"][4] for a in ("MID1", "FIRST1", "FIRST2")] + [hd[a]["S6"][5] for a in ("MID1", "FIRST1", "FIRST2")]
chk("(.48--.49 on image 6, .59--.62", "0.48", min(e6)); chk("(.48--.49 on image 6, .59--.62", "0.49", max(e6))
chk("(.48--.49 on image 6, .59--.62", "0.59", min(stt)); chk("(.48--.49 on image 6, .59--.62", "0.62", max(stt))
chk("on the hidden block (.39, against .17 on image 5)", "0.39", hd["HIDE6"]["E2"][5]); chk("on the hidden block (.39, against .17 on image 5)", "0.17", hd["HIDE6"]["E2"][4])
chk("``the sixth image from the first'' as well (.41)", "0.41", hd["HIDE6"]["S6"][5])
rf = N["reflect"]
chk("(correlation .995 over 18 cells in Qwen3.5-27B, mean absolute difference .017", "0.995", rf["27b"]["pearson"])
chk("(correlation .995 over 18 cells in Qwen3.5-27B, mean absolute difference .017", "18", rf["27b"]["n_cells"])
chk("(correlation .995 over 18 cells in Qwen3.5-27B, mean absolute difference .017", "0.017", rf["27b"]["mean_abs_diff"])
chk(".981 over the 48 inner cells of the four smaller models, .917 over all 72", "0.981", rf["small_inner"]["pearson"])
chk(".981 over the 48 inner cells of the four smaller models, .917 over all 72", "48", rf["small_inner"]["n_cells"])
chk(".981 over the 48 inner cells of the four smaller models, .917 over all 72", "0.917", rf["small_all"]["pearson"])
chk(".981 over the 48 inner cells of the four smaller models, .917 over all 72", "72", rf["small_all"]["n_cells"])
chk("End-relative accuracy rises from 51\\% to 99\\%", "51", nq["END"], 100); chk("End-relative accuracy rises from 51\\% to 99\\%", "99", nq["ES_look"], 100)
chk("Among the 79 scenes", "79", nq["n_endwrong"])
chk("moves end-relative accuracy to 56\\% and leaves start-relative accuracy at 99\\%", "56", nq["ES_pre"], 100)
chk("moves end-relative accuracy to 56\\% and leaves start-relative accuracy at 99\\%", "99", nq["SE_pre"], 100)
chk("removes 89\\% and 85\\% of the gain", "89", rt["N7"]["removed59"], 100); chk("removes 89\\% and 85\\% of the gain", "85", rt["N8"]["removed59"], 100)
chk("supplies 100\\% and 97\\%", "100", rt["N7"]["supplied"], 100); chk("supplies 100\\% and 97\\%", "97", rt["N8"]["supplied"], 100)
chk("which a reversed menu changes in 38\\% and 29\\%", "38", pf["N7"]["menu_reversal_change"]["END"], 100)
chk("which a reversed menu changes in 38\\% and 29\\%", "29", pf["N8"]["menu_reversal_change"]["END"], 100)
cj = N["chain_jk"]; ch = N["chain"]
chk("With both writes, 98\\% of the answers", "98", cj["R3"]["acc"], 100)
chk("With the address alone 3\\% are correct", "3", cj["ADDR"]["acc"], 100)
chk("follow a new picture at the copy in 61\\% of scenes, 84\\% for the farther references", "61", cj["ADDR"]["swapk"], 100)
chk("follow a new picture at the copy in 61\\% of scenes, 84\\% for the farther references", "84", ch["ADDR"]["far"]["swapk"], 100)
chk("With the address alone, 27B follows a new picture at the copy (84\\%", "84", ch["ADDR"]["far"]["swapk"], 100) if "With the address alone, 27B follows" in T6 else None
chk("near references follow a new picture at the target in 95\\% of scenes, and far references in 41\\%", "95", ch["END"]["near"]["swapj"], 100)
chk("near references follow a new picture at the target in 95\\% of scenes, and far references in 41\\%", "41", ch["END"]["far"]["swapj"], 100)
chk("In four of the five models 66--81\\% of the errors select it", "66", min(lu[t]["lure_share"] for t in four), 100)
chk("(79--91\\% against 55--63\\%)", "79", min(lu[t]["rescued_lure"] for t in four), 100); chk("(79--91\\% against 55--63\\%)", "91", max(lu[t]["rescued_lure"] for t in four), 100)
chk("(79--91\\% against 55--63\\%)", "55", min(lu[t]["rescued_other"] for t in four), 100); chk("(79--91\\% against 55--63\\%)", "63", max(lu[t]["rescued_other"] for t in four), 100)
chk("whose errors are spread (40\\% on the lure)", "40", lu["f7sl_35_4b"]["lure_share"], 100)
idn = [S["3.5-27B"]["identity"], ra["27b"]["k=4"]["END"], F7["ans"]["E4"]["4"]]
chk("Figure~\\ref{fig:scale}: 45\\%", "45", S["3.5-27B"]["identity"], 100); chk("Table~\\ref{tab:rebind}: 53\\%", "53", ra["27b"]["k=4"]["END"], 100)
chk("Figure~\\ref{fig:frameaddr}: 52\\%", "52", F7["ans"]["E4"]["4"], 100)
chk("from 31.1\\% to 83.6\\% in Qwen3.5-27B against 86.0\\%", "31.1", B["27b|MMIU held-out"]["R"], 100)
chk("from 31.1\\% to 83.6\\% in Qwen3.5-27B against 86.0\\%", "83.6", B["27b|MMIU held-out"]["rep"], 100)
chk("from 31.1\\% to 83.6\\% in Qwen3.5-27B against 86.0\\%", "86.0", B["27b|MMIU held-out"]["F"], 100)
chk("repaired 42\\% of Qwen3.5-4B failures", "42", qg["4b"]["PTR"], 100)
chk("and there it repairs 42\\% of the failures", "42", qg["4b"]["PTR"], 100)
chk("moves answers to $t+1$ in 59\\% of Qwen3.5-4B questions but in only 9\\%", "59", qg["4b"]["selw"], 100)
chk("moves answers to $t+1$ in 59\\% of Qwen3.5-4B questions but in only 9\\%", "9", qg["9b"]["selw"], 100)
chk("A no-op recomputation matches the clean run in 99--100\\%", "99", min(qg["4b"]["noop"], qg["9b"]["noop"]), 100)

# ---- v7 additions (mock review)
sv = N["stop_visual"]; c4 = sv["s854_1"]
chk("staying did not change either ($+0.00$, $[-0.09, 0.10]$", "0.00", c4["contrast"]["pooled"])
chk("staying did not change either ($+0.00$, $[-0.09, 0.10]$", "-0.09", c4["ci"]["pooled"][0]); chk("staying did not change either ($+0.00$, $[-0.09, 0.10]$", "0.10", c4["ci"]["pooled"][1])
s846 = sv["s846"]["stay_by_p"]
chk("stays there in 38\\% of runs, against 44\\% and 39\\% at $p=4$ and 6", "38", s846["5"], 100)
chk("stays there in 38\\% of runs, against 44\\% and 39\\% at $p=4$ and 6", "44", s846["4"], 100)
chk("stays there in 38\\% of runs, against 44\\% and 39\\% at $p=4$ and 6", "39", s846["6"], 100)
chk("Staying is 33\\% in both, a difference of $+0.00$ ($[-0.09, 0.10]$", "33", c4["p4_reached_mean"], 100)
chk("Staying is 33\\% in both, a difference of $+0.00$ ($[-0.09, 0.10]$", "33", c4["p4_not_reached_mean"], 100)
chk("gives $+0.00$ ($[-0.11, 0.12]$)", "0.00", c4["contrast"]["did"]); chk("gives $+0.00$ ($[-0.11, 0.12]$)", "-0.11", c4["ci"]["did"][0])
chk("gives $+0.00$ ($[-0.11, 0.12]$)", "0.12", c4["ci"]["did"][1])
chk("($+0.12$ and $-0.12$)", "0.12", c4["contrast"]["k3"]); chk("($+0.12$ and $-0.12$)", "-0.12", c4["contrast"]["k5"])
chk("(30 scenes, six or eight images shown", "30", c4["scenes"])
carry("60\\% of its errors carry this literal signature against 26\\% by chance", "names the gold image in 60\\% of cases, against 26\\% expected by chance among wrong options")
carry("lies below chance (14\\% against 24\\%", "the step signature lies below chance (14\\% against 24\\%; Table")
mu = [N["bench_paired"][f"{b}|MuirBench|F|base"] for b in ("4b", "8b", "35_4b", "35_9b", "27b")]
chk("On MuirBench it stays below the start-relative wording in four models", "4", sum(1 for m in mu if m[2] < 0))
chk("by 11.1 points in 27B", "11.1", -N["bench_paired"]["27b|MuirBench|F|base"][0], 100)
cp = N["cloze_picks"]
for pr, k in (("46", "A"), ("49", "B"), ("55", "C"), ("50", "D")):
    chk("(46, 49, 55 and 50 items)", pr, cp["gold"][k])
chk("chooses option A for 155 of the 200 rewritten items and for 176 of the end-relative ones", "155", cp["F|base"]["A"])
chk("chooses option A for 155 of the 200 rewritten items and for 176 of the end-relative ones", "176", cp["R|base"]["A"])
chk("chooses option A for 155 of the 200 rewritten items and for 176 of the end-relative ones", "200", cp["n"])
la = N["latency_27b"]
chk("on 218 MMIU items Qwen3.5-27B takes 1,464 ms", "218", la["n"]); chk("takes 1,464 ms per item", "1464", la["r3_ms"])
chk("against 977 ms unedited", "977", la["base_ms"]); chk("1,955 ms on items that receive a write", "1955", la["r3_ms_written"])
chk("and 973 ms on items that do not", "973", la["r3_ms_unwritten"])
rg = N["read_grid"]
chk("on a 300-item selection split", "300", rg["sel_n"]); chk("confirmed on a 300-item split with disjoint images", "300", rg["val_n"])
chk("$\\rho\\in\\{.30,.35,\\dots,.70\\}$", "0.30", float(rg["grid"]["detectors"][1].split(":")[1]))
chk("$\\rho\\in\\{.30,.35,\\dots,.70\\}$", "0.70", float(rg["grid"]["detectors"][-1].split(":")[1]))
chk("$\\lambda\\in\\{1.0,1.1,\\dots,1.6\\}$", "1.0", float(rg["grid"]["binders"][0].split(":")[1]))
chk("$\\lambda\\in\\{1.0,1.1,\\dots,1.6\\}$", "1.6", float(rg["grid"]["binders"][-1].split(":")[1]))
TR = open(f"{P6}/tables/repro.tex").read()
for m, h in N["config_sha"].items():
    n[0] += 1
    if h not in TR:
        fails.append(f"CONFIG SHA {m}: {h} not in tables/repro.tex")
print(f"{n[0]} checks, {len(fails)} problems")
for f in fails:
    print("  ", f)
sys.exit(1 if fails else 0)
