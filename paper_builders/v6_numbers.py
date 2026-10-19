"""Numbers for paper v6 that do not come from v5 (the restored lookup, frame and repair results of the old v3 line), computed from the raw
and registered result files only.  S782 rule: answer-based values come only from clean-readout runs (S783a, S786/S794, S787, S789, S791,
S793, S796) or from option-letter / QK records that the readout defect did not touch.  Writes research/ces/paper_v6_20261019/numbers_v6.json
with {key: {"value": ..., "source": ...}}; the v5 numbers (numbers_v5.json) are merged in unchanged under their own keys.
Usage: python3 exp/figs/v6_numbers.py"""
import json, re
import numpy as np

V = "/home/storage2/ai26269345/vlm"
R = f"{V}/results/r3"
P6 = f"{V}/research/ces/paper_v6_20261019"
J = lambda p: json.load(open(p if p.startswith("/") else f"{R}/{p}"))
N = {}


def put(key, value, source):
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    N[key] = dict(value=value, source=source)


M5 = ["4b", "8b", "35_4b", "35_9b", "27b"]

# --- Section 3: the index code (S545 re-measured in S787, clean readout, all five models) ---
shift, ctrl = {}, {}
for m in M5:
    d = J(f"{V}/results/s545_indexshift_s787_{m}.json"); rows = d["rows"]; k3 = [r for r in rows if r["k"] - 2 >= 1]
    shift[m] = [int(sum(r["F|pre"] == r["k"] - 2 for r in k3)), len(k3)]
    ctrl[m] = [int(sum(r["F|app"] == r["F|none"] for r in rows)), len(rows)]
put("idx_shift", shift, "results/s545_indexshift_s787_*.json: forward answers on k-2 under the prepended donor, k>=3")
put("idx_shift_ctrl", ctrl, "same files: appended donor leaves the forward answer unchanged")
kv = {}
for m in ("4b", "9b"):
    rows = [r for r in J(f"s741b_s789kv_{m}_kv.json")["rows"] if r["k"] - 2 >= 1]
    kv[m] = {a: float(np.mean([r[a] == r["k"] - 2 for r in rows])) for a in ("FULL", "KEY", "VAL", "KV", "AKEY")}
    kv[m]["n"] = len(rows)
put("kv", kv, "results/r3/s741b_s789kv_{4b,9b}_kv.json (S789 confirm run of S741b): forward answers on k-2")
tok = {}
for m in M5:
    rows = J(f"s739_s789tk_{m}_tokens.json")["rows"]
    tok[m] = {reg: float(np.mean([r["land_START"] == r["j"] for r in rows if r["regime"] == reg])) for reg in ("UNI", "MIX")}
    tok[m]["tokens"] = [min(t for r in rows for t in r["block_tokens"]), max(t for r in rows for t in r["block_tokens"])]
put("tokens", tok, "results/r3/s739_s789tk_*_tokens.json (S789 confirm run of S739): start-relative accuracy, equal / unequal blocks")
S27 = np.array(J(f"{V}/results/figures/c_circuit/qk_tables_cache.json")["qa_27b"]["S"])
qk_tab = open(f"{V}/research/ces/outputs/paper_r3_v3/tables/mech_qk.tex").read()
st = {m: int(re.search(rf"{m} \(target\) & \d+ & (\d+)", qk_tab).group(1)) for m in ("Qwen3.5-4B", "Qwen3.5-9B", "Qwen3.5-27B")}
put("heads_start_share", dict(table=st, qa_27b_from_capture=float(np.diag(S27).mean())),
    "QK captures S705a: tables/mech_qk.tex of the old v3 line; 27B recomputed from results/figures/c_circuit/qk_tables_cache.json")
a = J("/home/storage2/ai26269345/vlm/research/ces/outputs/r3_semantic_result_audit_v1b/frozen_analysis_science-002.json")
Mx = a["conditional_arm_destination_matrix"]
put("address_control_vl4b", dict(cells=a["co_primary"]["correct_address_OF_minus_RJF_at_current_j"]["complete_variant_cells"],
                                 correct_minus_random=a["co_primary"]["correct_address_OF_minus_RJF_at_current_j"]["complete_cluster_mean"],
                                 wrong_minus_random=a["co_primary"]["wrong_address_WF_minus_RWF_at_current_w"]["complete_cluster_mean"],
                                 dest={arm: {c: Mx[arm][c]["complete_cluster_mean"] for c in Mx[arm]} for arm in ("00", "RJF", "O", "OF", "W", "WF")}),
    "research/ces/outputs/r3_semantic_result_audit_v1b/frozen_analysis_science-002.json (Qwen3-VL-4B, option letters, 86 scenes x 3 views)")
d = J("s708_s789rb_27b_rebind.json"); rows = d["rows"]; ks = ("2", "3", "4", "5", "6")
wrong = {str(L): {k: float(np.mean([r["k"][k]["land"][f"WRONG@{L}"] == r["k"][k]["w"] for r in rows])) for k in ks} for L in d["LW"]}
put("wrong_addr_27b", wrong, "results/r3/s708_s789rb_27b_rebind.json (S789 confirm run of S708): wrong address + start frame lands on w, per k")

# --- Section 4: which address the end-relative query carries (S790c, QK) ---
D = J("s790c_selnn_full.json")
put("selnn", {m: {str(k): {x: D[m][str(k)][x] for x in ("copy", "comp", "other")} for k in range(1, 8)} for m in ("qa_4b", "qa_9b", "qa_27b")},
    "results/r3/s790c_selnn_full.json (QK captures, held-out scenes)")

# --- Section 4.1: smaller models (S673b re-measured in S789, S720 re-measured in S789, S787 transplant I4) ---
db = {}
for m in ("4b", "8b"):
    d = J(f"s673b_s789db_{m}_deref.json"); rows = [r for r in d["rows"] if r["donor_ok"]]
    cat = lambda r, x: "j" if x == r["j"] else ("y" if x == r["yj"] else ("k" if x == r["k"] else "o"))
    db[m] = {str(L): {c: float(np.mean([cat(r, r["land"][f"ORD@{L}"]) == c for r in rows])) for c in "jyko"} for L in d["LS"]}
    db[m]["n"] = len(rows)
put("deref", db, "results/r3/s673b_s789db_{4b,8b}_deref.json (S789 confirm run of S673b): landing of the transplanted ordinal state by layer")
sel = {}
for m in ("4b", "9b"):
    rows = J(f"s720_s789sel_{m}_select.json")["rows"]
    fail = [(r, t) for r in rows for t in (2, 3, 4) if r["t"][str(t)]["NATIVE"]["land"] != t and r["t"][str(t)]["START"]["land"] == t]
    sel[m] = {arm: float(np.mean([r["t"][str(t)][arm]["land"] == t for r, t in fail])) for arm in ("Q_ONLY", "G_ONLY", "QG", "PTR")}
    sel[m]["n"] = len(fail)
    sel[m]["selw"] = float(np.mean([r["t"][str(t)]["PTR_W"]["land"] == t + 1 for r in rows for t in (2, 3, 4)]))
    sel[m]["noop"] = float(np.mean([r["t"][str(t)]["NOOP"]["land"] == r["t"][str(t)]["NATIVE"]["land"] for r in rows for t in (2, 3, 4)]))
put("query_vs_gate", sel, "results/r3/s720_s789sel_{4b,9b}_select.json (S789 confirm run of S720): failed END items repaired")
A7 = J("s787_analysis.json"); F7 = J("s787_27b_final.json")
put("transplant_I4", {**{m: A7["fit"][m]["I4"] for m in ("4b", "8b", "35_4b", "35_9b")}, "27b": F7["fit"]["I4"]},
    "results/r3/s787_analysis.json, s787_27b_final.json: forward-ordinal transplant, [mean, lo, hi, n] on x (copy) and N+1-x (comp)")
put("reflect", {"small_inner": A7["reflect"]["inner"], "small_all": A7["reflect"]["all"], "27b": F7["reflect"]["all"]},
    "same files: address-write answers vs the model's own END(j) answers, per (model, N, t) cell")

# --- Section 4.2: the frame switch (S769 re-measured, S783a), the lens (S795), frame -> queries (S796) ---
D = J("s769_analysis_bare.json")
put("frame_switch", {Nn: {"ans": {nm: D[Nn]["dist"][nm] for nm in ("S4", "E4", "F4:P1", "F4:P23")},
                          "share4_L39": {nm: D[Nn]["share4_L39"][nm] for nm in ("S4", "E4", "F4:P1", "F4:P23")}} for Nn in ("N7", "N8")},
    "results/r3/s769_analysis_bare.json (S783a clean readout)")
L = J("s795_analysis.json")["s795_27b"]["per_Ls"]
put("lens", {Ls: {str(t): {kk: L[Ls][str(t)][kk] for kk in ("start_target", "end_target", "end_literal")} for t in range(1, 8)} for Ls in ("39", "51")},
    "results/r3/s795_analysis.json (QK lens, 30 scenes)")
a = J("s796_analysis.json")["s796_27b"]
put("frame_queries", {q: {"m": a["q"][q]["m"], "MF": a["q"][q]["M2"]["MF"], "MF_ci": a["q"][q]["M2"]["MF_ci"], "SF": a["q"][q]["M3"]["SF"]} for q in ("4", "5")},
    "results/r3/s796_analysis.json (max readout, t1 1.000)")

# --- Section 4.3: hiding images from full attention (S748 captures, QK) ---
z = np.load(f"{R}/s748_s748_27b_whatend.npz")
H = [tuple(x) for x in J("s705a_qa_27b_matchers.json")["matchers"]]
scenes = sorted({k.split("|")[0] for k in z.files})


def share(arm, q, row=2):
    out = []
    for s in scenes:
        hs = []
        for (Ly, h) in H:
            x = z[f"{s}|{arm}|{q}|{Ly}"][row, h]; p = np.exp(x - x.max()); hs.append(p / p.sum())
        out.append(np.mean(hs, 0))
    return np.mean(out, 0).tolist()


put("hide", {arm: {q: share(arm, q) for q in ("E2", "S5", "S6")} for arm in ("NONE", "LAST1", "MID1", "FIRST1", "FIRST2", "HIDE6")},
    "results/r3/s748_s748_27b_whatend.npz: answer row, per-head softmax over the seven blocks, mean over the 73 index-matching heads and 40 scenes")

# --- Section 6: queries carry error and repair, the answer follows the picture ---
rows = J("s793_s793_27b_transplant.json")["rows"]
land = lambda r: eval(r["land"]) if isinstance(r["land"], str) else r["land"]
acc = lambda arm, tgt: float(np.mean([land(r)[arm] == tgt(r) for r in rows]))
endwrong = [r for r in rows if land(r)["END"] != r["j"]]
put("natural_q", dict(END=acc("END", lambda r: r["j"]), ES_look=acc("ES_look", lambda r: r["j"]), START=acc("START", lambda r: r["j"]),
                      SE_look=acc("SE_look", lambda r: r["j"]), ES_pre=acc("ES_pre", lambda r: r["j"]), SE_pre=acc("SE_pre", lambda r: r["j"]),
                      SE_reproduces_wrong=float(np.mean([land(r)["SE_look"] == land(r)["END"] for r in endwrong])), n_endwrong=len(endwrong), n=len(rows)),
    "results/r3/s793_s793_27b_transplant.json (S793, max readout): START(j) <-> END(k) queries of the lookup band (layers 35-55)")
B = J("s780_analysis_bare.json")
c7, c8 = B["s783_27b7"], B["s783_27b8"]
gain = lambda c: c["i0"]["R3"] - c["i0"]["E5"]
put("r3_through_queries", {"N7": dict(removed59=c7["r3_removed_ALL59"] / gain(c7), supplied=c7["r3_suf"] / gain(c7)),
                           "N8": dict(removed59=c8["r3_removed_ALL59"] / gain(c8), supplied=c8["r3_suf"] / gain(c8))},
    "results/r3/s780_analysis_bare.json (S783a): share of R3's END(5) gain removed by the unrepaired queries of layers 35-59 / supplied by the rebound ones")
put("picture_follow", {Nn: {k: c[k] for k in ("visual_r3", "visual_e2", "visual_end", "menu_reversal_change")} for Nn, c in (("N7", c7), ("N8", c8))},
    "results/r3/s780_analysis_bare.json (S783a)")
s = J("s791_analysis.json")["s791_27b"]["table"]
put("chain", {arm: {g: {kk: s[arm][g][kk] for kk in ("acc", "swapj", "swapk", "cos")} for g in ("k4", "near", "far")} for arm in ("END", "ADDR", "FRAME", "R3")},
    "results/r3/s791_analysis.json (S791, bare readout = greedy generation)")
put("s721_later", J("s721_analysis.json")["arms"], "results/r3/s721_analysis.json (option letters, later MMIU split, Qwen3.5-27B)")
put("lure", {d["tag"]: d for d in J(f"{V}/results/figures/tables_r3/lure_values.json")}, "results/figures/tables_r3/lure_values.json (option letters)")
import importlib.util, sys
sys.path.insert(0, f"{V}/exp/figs")
A30 = J("s730_analysis.json")
KEY = {b: (f"s794_{b}" if b == "35_4b" else f"s786_{b}") for b in M5}
put("rebind_cos", {b: {c: A30[KEY[b]]["summary"]["k!=4"][c]["cos"] for c in ("END", "ADDR", "FRAME", "R3")} for b in M5},
    "results/r3/s730_analysis.json, s786_* / s794_35_4b: query cosine to the start-relative wording, k != 4")
put("rebind_acc", {b: {sub: {c: A30[KEY[b]]["summary"][sub][c]["acc"] for c in A30[KEY[b]]["summary"][sub]} for sub in ("k!=4", "k=4")} for b in M5},
    "results/r3/s730_analysis.json (S786 / S794)")
G = J(f"{V}/research/ces/outputs/v2_grid_landings.json")["cells"]
put("grid", {k: G[k]["landing"] for k in G}, "research/ces/outputs/v2_grid_landings.json (S824/S825/S834, 40 scenes per cell)")

# --- benchmark values (make_r3_tables.bench_values logic, host paths), fig_scale values, S740 photographs, S791 jk cells ---
F732 = {b: dict(sealed=f"f7sl_{b}", later=f"f7fr_{b}", cloze=f"f7cz_{b}") for b in ("4b", "8b", "35_4b", "35_9b")}
F732["27b"] = dict(sealed="c8sl_27b", later="c8fr_27b", cloze="c8cz_27b")
SETS = [("sealed", "mmiu_pos", "MMIU held-out"), ("sealed", "muir", "MuirBench held-out"), ("later", "mmiu_pos", "MMIU later"), ("cloze", "cloze", "MMIU cloze")]
bv = {}
for b in M5:
    for split, suite, lab in SETS:
        rows = [r for r in J(f"{V}/results/s582_evalrepair_{F732[b][split]}.json")["rows"] if r["suite"] == suite]
        m = lambda k: float(np.mean([r[k] == r["gold"] for r in rows]))
        bv[f"{b}|{lab}"] = dict(n=len(rows), F=m("F|base"), R=m("R|base"), rep=m("R|rep"))
put("bench", bv, "results/s582_evalrepair_{f7*,c8*}.json (option letters; same tags as make_r3_tables.bench_tags)")
put("scale", J(f"{V}/research/ces/paper_v5_20261018/figures/fig_scale_values.json"), "research/ces/paper_v5_20261018/figures/fig_scale_values.json")
nm = {}
for tag in ("35_4b", "35_9b", "27b"):
    rows = [r for r in J(f"s740_nm740_{tag}_natmed.json")["rows"] if "pick_R3_QBACK_ALL" in r]
    resc = [r for r in rows if r["pick_R3"] == r["gold"] and r["pick_END"] != r["gold"]]
    nm[tag] = {k: float(np.mean([r[k] == r["gold"] for r in resc])) for k in ("pick_R3_QBACK_ALL", "pick_END_QALL")}
    nm[tag]["n"] = len(resc)
put("photo_queries", nm, "results/r3/s740_nm740_*_natmed.json (MMIU development photographs, option letters)")
s = J("s791_analysis.json")["s791_27b"]["table"]
put("chain_jk", {arm: {kk: s[arm]["jk"][kk] for kk in ("acc", "swapj", "swapk")} for arm in ("END", "ADDR", "FRAME", "R3")}, "results/r3/s791_analysis.json, j != k cells")

v5 = json.load(open(f"{V}/research/ces/paper_v5_20261018/numbers_v5.json"))
out = {**v5, **N}
json.dump(out, open(f"{P6}/numbers_v6.json", "w"), indent=1, default=float)
print(f"{len(N)} new keys, {len(v5)} v5 keys -> {P6}/numbers_v6.json")
