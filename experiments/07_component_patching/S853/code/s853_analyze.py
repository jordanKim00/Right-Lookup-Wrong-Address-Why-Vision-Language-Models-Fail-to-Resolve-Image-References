"""S853 analysis (rules fixed in knowledge/DECISIONS.md before any S853 data; user 10-18 'GO - writer input decomposition').  30 independent
scenes; scene bootstrap 4,000 (seed 0); every ratio is a ratio of sums within each scene resample.
Transfer toward the donor, for a transplant X[R <- D] against the recipient's natural run R and the donor's D (same pass):
  rho = sum (x - r).(d - r) / sum |d - r|^2   (1 = the donor's state, 0 = unchanged), on
  rho_W  writer outputs: token mixer of layer 47 (full attention) and of layer 49 (DeltaNet) at rows B1, B4
  rho_Z  late positional scores: block scores of rows B1, B4, answer at layers 51, 55, 59, 63 (each 8-vector centred)
  rho_L  answer readout: menu logits of the eight picture words (centred)
I0 (all must pass): t1 >= .95; image keys identical (0.0); QNOOP_E = END(5, 8) in >= .90 of scenes with the 95th percentile of the largest
  menu-logit change <= .5; START(k, N_s) -> k in >= .90 for each (k, N_s); pin donors START(q, N_s) -> q in >= .90 pooled (>= .80 each); PIN:
  in E1-E7 the block-score maximum of rows B1 and B4 at layers 43 and 47 is the pinned p in >= .95; SELF: XSELF_1 / XSELF_3 give E1's / E3's
  answer in >= .99 with the 95th percentile of the largest menu-logit difference <= .05; PASSB: the pass-B copies of E1-E7 likewise.
PC (positive control of the layer-46 residual patch in this grid): FULLB (B0-B4 <- START(k) pinned at the same p) changes the answer of E1 / E3
  in >= .25 of runs, lower bound > .10 (S850: .46).
Per transplant type (K = k only, P = p only, KP = both, N = stated count only, T = same target j* = 4), B1 / B4 arms (needs I0 and PC):
  CHAIN  rho_W >= .30, rho_Z >= .20 and rho_L >= .20, each with lower bound > 0 -> 'the donor's input state at B1 / B4 is converted by the
         writer and carried to the late address and the answer readout'
  ADDR   rho_W and rho_Z pass, rho_L does not -> 'carried to the late address but not to the answer readout'
  WRITER rho_W passes, rho_Z does not -> 'changes the writer output but not the late address'
  NONE   rho_W does not pass, and rho_Z < .10 and rho_L < .10, each with upper bound < .20 -> 'the B1 / B4 input carries no measurable share of
         the condition's effect'
  otherwise 'partial' (checked in this order).  Also reported: on runs where R's and D's natural answers differ, the shares that follow the donor / the recipient and
  their paired difference.
PRED-A  XK_1_3 (k 3 <- k 5 at p 3) raises P(answer = 2) over E1 by >= .15 with lower bound > 0 -> 'the k = 5 input state raises the
        predecessor answer at p = 3' (S852: .03 vs .73); otherwise 'not shown'.
PRED-B  XP_3_4 (p 3 <- p 4 at k 5) raises P(answer = 3) over E3 by >= .15 with lower bound > 0 -> 'the p = 4 input state moves the landing
        toward 3' (S852: E(5, 4) -> 3 .50); otherwise 'not shown'.
INT     for each recipient corner of the 2 x 2, D_K, D_P, D_KP = late-score change under the k, p and both transplants;
        R2_add = 1 - sum |D_KP - D_K - D_P|^2 / sum |D_KP|^2.  Only if KP's rho_Z >= .20 with lower bound > 0 ('uninterpretable' otherwise):
        R2_add >= .70 with lower bound > .50 -> 'additive in the tested range'; R2_add <= .40 with upper bound < .60 -> 'the both-swap is not
        the sum of the single swaps: an interaction term is needed'; otherwise 'partial'.
NAT-J   natural pinned runs at p 3, r = P(answer = 2): D_same = r(E3: k5 N8) - r(E5: k3 N6) (same j* = 4); D_cnt3 = r(E1: k3 N8) - r(E5)
        (same k, j* 6 vs 4); D_cnt5 = r(E3) - r(E6: k5 N6) (same k, j* 4 vs 2).  |D_same| < .10 with the interval inside (-.20, .20) and
        |D_cnt3| >= .20 with the interval excluding 0 -> 'the predecessor step follows the stated target position'; D_same >= .20 with lower
        bound > 0 and |D_cnt3|, |D_cnt5| < .10 with intervals inside (-.20, .20) -> 'the predecessor step follows the ordinal at a fixed
        target'; D_same >= .20 with lower bound > 0 and |D_cnt3| or |D_cnt5| >= .20 with the interval excluding 0 -> 'both the ordinal and
        the stated count contribute'; otherwise 'partial' (needs I0).
Descriptive: answer distributions per arm; all-read-rows transplants (ALLR) against the B1 / B4 ones; per-component writer transfer (all eight
outputs, rows B1 / B4 / answer); late-query transfer; the natural 2 x 2 analogue of INT.
Usage: python exp/s853_analyze.py [TAG]  ->  results/r3/s853_analysis.json"""
from __future__ import annotations

import json, os, sys
import numpy as np

V = "/home/storage2/ai26269345/vlm"
B = 4000
LATE = ("51", "55", "59", "63"); ZROWS = (2, 5, 9)
TYPES = {"K": "XK_", "P": "XP_", "KP": "XKP_", "N": "XN_", "T": "XT_"}


def main(tag="s853_27b"):
    d = json.load(open(f"{V}/results/r3/s853_{tag}_writer.json")); R = d["rows"]
    Z = np.load(f"{V}/results/r3/s853_{tag}_writer.npz")
    COND = {k: tuple(v) for k, v in d["cond"].items()}; XA = [tuple(x) for x in d["xa"]]; XB = [tuple(x) for x in d["xb"]]
    by = {r["si"]: r for r in R}; sis = sorted(by)
    rng = np.random.default_rng(0); idx = rng.integers(0, len(sis), (B, len(sis)))
    def ci(per):
        per = np.asarray(per, dtype=float); bs = per[idx].mean(1)
        return [float(per.mean()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]]
    def ci_ratio(num, den):
        num = np.asarray(num, dtype=float); den = np.asarray(den, dtype=float)
        bs = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-12)
        return [float(num.sum() / max(den.sum(), 1e-12)), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]]
    mld = lambda r, a1, a2: max(abs(x - y) for x, y in zip(r["mlog"][a1], r["mlog"][a2]))
    cen = lambda x: np.asarray(x, dtype=float) - np.mean(x)
    def zvec(r, nm):
        return np.concatenate([cen(r["qk"][nm][L][ri]) for L in LATE for ri in ZROWS])
    def lvec(r, nm):
        return cen(r["mlog"][nm][:8])
    ci_ = {nm: i for i, nm in enumerate(f"{l}{kd}" for l, kd in [tuple(c) for c in d["comps"]])}
    WC = [ci_["47mix"], ci_["49mix"]] if "47mix" in ci_ else [2, 6]
    def wvec(s, nm, comps=None, rows=(0, 1)):
        c = Z[f"s{s}_{nm}_comp"].astype(np.float64)                    # [8 comps, 3 rows (B1, B4, answer), E]
        return np.concatenate([c[cc, ri] for cc in (WC if comps is None else comps) for ri in rows])
    out = dict(tag=tag, t1=d["t1_menu"], key_rel_diff_max=d["key_rel_diff_max"], n=len(sis), n_runs=d.get("n_runs"), I0={})
    # ---------------- I0
    i0 = {}
    i0["t1"] = d["t1_menu"] >= .95; i0["keys"] = d["key_rel_diff_max"] == 0.0
    qn = float(np.mean([by[s]["land"]["QNOOP_E"] == by[s]["land"]["END_k5N8"] for s in sis]))
    qp = float(np.percentile([mld(by[s], "QNOOP_E", "END_k5N8") for s in sis], 95)); i0["qnoop"] = dict(agree=qn, p95=qp, ok=bool(qn >= .90 and qp <= .5))
    st = {f"{k}_{n}": float(np.mean([by[s]["land"][f"START_q{k}N{n}"] == k for s in sis])) for k, n in [(3, 8), (5, 8), (3, 6), (5, 6), (4, 7)]}
    i0["start_k"] = dict(acc=st, ok=bool(min(st.values()) >= .90))
    dn = {f"{q}_{n}": float(np.mean([by[s]["land"][f"START_q{q}N{n}"] == q for s in sis])) for q, n in [(3, 8), (4, 8), (3, 6), (3, 7)]}
    i0["donors"] = dict(acc=dn, ok=bool(np.mean(list(dn.values())) >= .90 and min(dn.values()) >= .80))
    pin = [int(np.argmax(by[s]["qk"][nm][L][ri]) + 1 == COND[nm][1]) for s in sis for nm in COND for L in ("43", "47") for ri in (2, 5)]
    i0["pin"] = dict(share=float(np.mean(pin)), ok=bool(np.mean(pin) >= .95))
    def same(a1, a2):
        ag = float(np.mean([by[s]["land"][a1] == by[s]["land"][a2] for s in sis])); p95 = float(np.percentile([mld(by[s], a1, a2) for s in sis], 95))
        return ag, p95, float(max(mld(by[s], a1, a2) for s in sis))
    sf = {nm: same(nm, r_) for nm, r_, d_ in XA if nm.startswith("XSELF")}
    i0["self"] = dict(v=sf, ok=bool(all(x[0] >= .99 and x[1] <= .05 for x in sf.values())))
    pb = {nm: same(nm + "_B", nm) for nm in COND}
    i0["passb"] = dict(v=pb, ok=bool(all(x[0] >= .99 and x[1] <= .05 for x in pb.values())))
    ok_all = bool(i0["t1"] and i0["keys"] and all(v["ok"] for k, v in i0.items() if isinstance(v, dict)))
    out["I0"] = dict(i0, all=ok_all)
    # ---------------- PC
    pc = ci([np.mean([by[s]["land"]["FB_1"] != by[s]["land"]["E1_B"], by[s]["land"]["FB_3"] != by[s]["land"]["E3_B"]]) for s in sis])
    out["PC"] = dict(changed=pc, ok=bool(pc[0] >= .25 and pc[1][0] > .10))
    valid = ok_all and out["PC"]["ok"]; nvt = "no valid test"
    # ---------------- transfer per type
    def transfer(arms, rsuf=""):
        nW, dW, nZ, dZ, nL, dL = ([] for _ in range(6)); fr, fd, fn = [], [], []
        for s in sis:
            r = by[s]; a_ = np.zeros(6); n_ = r_ = d_ = 0
            for nm, rc, dc in arms:
                R_, D_ = rc + rsuf, dc + rsuf
                for j, (fx, fr_, fd_) in enumerate([(wvec(s, nm), wvec(s, R_), wvec(s, D_)), (zvec(r, nm), zvec(r, R_), zvec(r, D_)),
                                                    (lvec(r, nm), lvec(r, R_), lvec(r, D_))]):
                    a_[2 * j] += float((fx - fr_) @ (fd_ - fr_)); a_[2 * j + 1] += float((fd_ - fr_) @ (fd_ - fr_))
                ar, ad, ax = r["land"][R_], r["land"][D_], r["land"][nm]
                if ar != ad:
                    n_ += 1; r_ += int(ax == ar); d_ += int(ax == ad)
            for lst, v_ in zip((nW, dW, nZ, dZ, nL, dL), a_):
                lst.append(v_)
            fn.append(n_); fr.append(r_); fd.append(d_)
        fn_, fr_, fd_ = map(np.asarray, (fn, fr, fd))
        dif = (fd_[idx].sum(1) - fr_[idx].sum(1)) / np.maximum(fn_[idx].sum(1), 1)
        return dict(rho_W=ci_ratio(nW, dW), rho_Z=ci_ratio(nZ, dZ), rho_L=ci_ratio(nL, dL), n_discordant=int(fn_.sum()),
                    f_don=ci_ratio(fd, fn), f_rec=ci_ratio(fr, fn),
                    don_minus_rec=[float((fd_.sum() - fr_.sum()) / max(fn_.sum(), 1)), [float(np.percentile(dif, 2.5)), float(np.percentile(dif, 97.5))]])
    def verdict(t):
        if not valid:
            return nvt
        pW = t["rho_W"][0] >= .30 and t["rho_W"][1][0] > 0; pZ = t["rho_Z"][0] >= .20 and t["rho_Z"][1][0] > 0; pL = t["rho_L"][0] >= .20 and t["rho_L"][1][0] > 0
        if pW and pZ and pL:
            return "the donor's input state at B1 / B4 is converted by the writer and carried to the late address and the answer readout"
        if pW and pZ:
            return "carried to the late address but not to the answer readout"
        if pW:
            return "changes the writer output but not the late address"
        if t["rho_Z"][0] < .10 and t["rho_Z"][1][1] < .20 and t["rho_L"][0] < .10 and t["rho_L"][1][1] < .20:
            return "the B1 / B4 input carries no measurable share of the condition's effect"
        return "partial"
    out["TYPES"] = {}
    for ty, pre in TYPES.items():
        arms = [x for x in XA if x[0].startswith(pre) and not x[0].startswith("XSELF") and not (ty == "K" and x[0].startswith("XKP_"))]
        t = transfer(arms); t["verdict"] = verdict(t); t["arms"] = [x[0] for x in arms]; out["TYPES"][ty] = t
    # ---------------- PRED-A / PRED-B
    def pred(arm, base, x):
        v_ = ci([float(by[s]["land"][arm] == x) - float(by[s]["land"][base] == x) for s in sis]); return v_
    pa = pred("XK_1_3", "E1", 2); pb_ = pred("XP_3_4", "E3", 3)
    out["PRED_A"] = dict(diff=pa, verdict=(nvt if not valid else "the k = 5 input state raises the predecessor answer at p = 3"
                                           if pa[0] >= .15 and pa[1][0] > 0 else "not shown"))
    out["PRED_B"] = dict(diff=pb_, verdict=(nvt if not valid else "the p = 4 input state moves the landing toward 3"
                                            if pb_[0] >= .15 and pb_[1][0] > 0 else "not shown"))
    # ---------------- INT
    corners = {"E1": ("XK_1_3", "XP_1_2", "XKP_1_4"), "E2": ("XK_2_4", "XP_2_1", "XKP_2_3"), "E3": ("XK_3_1", "XP_3_4", "XKP_3_2"),
               "E4": ("XK_4_2", "XP_4_3", "XKP_4_1")}
    nat = {"E1": ("E3", "E2", "E4"), "E2": ("E4", "E1", "E3"), "E3": ("E1", "E4", "E2"), "E4": ("E2", "E3", "E1")}
    def r2(src):
        num, den = [], []
        for s in sis:
            r = by[s]; a_ = b_ = 0.0
            for c, (k_, p_, kp_) in src.items():
                z0 = zvec(r, c); dk, dp, dkp = zvec(r, k_) - z0, zvec(r, p_) - z0, zvec(r, kp_) - z0
                a_ += float((dkp - dk - dp) @ (dkp - dk - dp)); b_ += float(dkp @ dkp)
            num.append(a_); den.append(b_)
        num, den = np.asarray(num), np.asarray(den)
        bs = 1 - num[idx].sum(1) / den[idx].sum(1)
        return [float(1 - num.sum() / den.sum()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]]
    ra = r2(corners); rn = r2(nat)
    kp = out["TYPES"]["KP"]["rho_Z"]
    if not valid:
        iv = nvt
    elif not (kp[0] >= .20 and kp[1][0] > 0):
        iv = "uninterpretable (no both-swap effect on the late address)"
    elif ra[0] >= .70 and ra[1][0] > .50:
        iv = "additive in the tested range"
    elif ra[0] <= .40 and ra[1][1] < .60:
        iv = "the both-swap is not the sum of the single swaps: an interaction term is needed"
    else:
        iv = "partial"
    out["INT"] = dict(R2_add=ra, R2_add_natural=rn, verdict=iv)
    # ---------------- NAT-J
    rp = lambda nm: [float(by[s]["land"][nm] == 2) for s in sis]
    dsame = ci(np.subtract(rp("E3"), rp("E5"))); dc3 = ci(np.subtract(rp("E1"), rp("E5"))); dc5 = ci(np.subtract(rp("E3"), rp("E6")))
    dk8 = ci(np.subtract(rp("E3"), rp("E1")))
    inside = lambda v_, w: v_[1][0] > -w and v_[1][1] < w
    excl0 = lambda v_: v_[1][0] > 0 or v_[1][1] < 0
    if not ok_all:
        nj = nvt
    elif abs(dsame[0]) < .10 and inside(dsame, .20) and abs(dc3[0]) >= .20 and excl0(dc3):
        nj = "the predecessor step follows the stated target position"
    elif dsame[0] >= .20 and dsame[1][0] > 0 and abs(dc3[0]) < .10 and abs(dc5[0]) < .10 and inside(dc3, .20) and inside(dc5, .20):
        nj = "the predecessor step follows the ordinal at a fixed target"
    elif dsame[0] >= .20 and dsame[1][0] > 0 and ((abs(dc3[0]) >= .20 and excl0(dc3)) or (abs(dc5[0]) >= .20 and excl0(dc5))):
        nj = "both the ordinal and the stated count contribute"
    else:
        nj = "partial"
    out["NAT_J"] = dict(r={nm: float(np.mean(rp(nm))) for nm in ("E1", "E3", "E5", "E6", "E7")}, D_same=dsame, D_cnt3=dc3, D_cnt5=dc5, D_k8=dk8,
                        verdict=nj)
    # ---------------- descriptive
    desc = {}
    allarms = [nm for nm, _, _ in XA] + [nm for nm, _, _ in XB] + list(COND) + ["FB_1", "FB_3", "S_E1", "S_E3"]
    desc["dist"] = {nm: {str(x): float(np.mean([by[s]["land"][nm] == x for s in sis])) for x in range(0, 9)
                         if any(by[s]["land"][nm] == x for s in sis)} for nm in allarms}
    desc["allr"] = {}
    for nm, rc, dc in XB:
        b14 = nm.replace("AK_", "XK_").replace("AP_", "XP_").replace("AKP_", "XKP_").replace("AN_", "XN_").replace("AT_", "XT_")
        tA = transfer([(nm, rc, dc)], rsuf="_B"); tB = transfer([(b14, rc, dc)])
        desc["allr"][nm] = dict(allrows=dict(rho_Z=tA["rho_Z"], rho_L=tA["rho_L"], f_don=tA["f_don"]),
                                b1b4=dict(rho_Z=tB["rho_Z"], rho_L=tB["rho_L"], f_don=tB["f_don"]))
    comp_names = [f"{l}{kd}" for l, kd in [tuple(c) for c in d["comps"]]]
    desc["writer_per_component"] = {}
    for ty, pre in TYPES.items():
        arms = [x for x in XA if x[0].startswith(pre) and not x[0].startswith("XSELF") and not (ty == "K" and x[0].startswith("XKP_"))]
        rowsd = {}
        for ci2, cn in enumerate(comp_names):
            for ri, rn in enumerate(("B1", "B4", "answer")):
                num = den = 0.0
                for s in sis:
                    for nm, rc, dc in arms:
                        x_, r_, d_ = (Z[f"s{s}_{a}_comp"][ci2, ri].astype(np.float64) for a in (nm, rc, dc))
                        num += float((x_ - r_) @ (d_ - r_)); den += float((d_ - r_) @ (d_ - r_))
                rowsd[f"{cn}_{rn}"] = num / max(den, 1e-12)
        desc["writer_per_component"][ty] = rowsd
        num = den = 0.0
        for s in sis:
            for nm, rc, dc in arms:
                x_, r_, d_ = (Z[f"s{s}_{a}_q"].astype(np.float64).ravel() for a in (nm, rc, dc))
                num += float((x_ - r_) @ (d_ - r_)); den += float((d_ - r_) @ (d_ - r_))
        desc.setdefault("late_query", {})[ty] = num / max(den, 1e-12)
    out["desc"] = desc
    fn = f"{V}/results/r3/s853_analysis.json" if tag == "s853_27b" else f"{V}/results/r3/s853_analysis_{tag}.json"
    json.dump(out, open(fn, "w"), indent=1)
    print("I0", json.dumps(out["I0"])[:1500]); print("PC", out["PC"])
    for ty, t in out["TYPES"].items():
        print(ty, {k: t[k] for k in ("rho_W", "rho_Z", "rho_L", "f_don", "f_rec", "don_minus_rec", "n_discordant", "verdict")})
    print("PRED_A", out["PRED_A"]); print("PRED_B", out["PRED_B"]); print("INT", out["INT"]); print("NAT_J", out["NAT_J"])


if __name__ == "__main__":
    main(*sys.argv[1:])
