"""S854-2 analysis (rules fixed in knowledge/DECISIONS.md before any S854-2 data; user 10-18 'S854-2 - input-row causal decomposition').
Ordinal swap at a fixed early position (p 3): 'up' = recipient E1 (k 3) <- donor E3 (k 5), 'dn' = E3 <- E1; input groups at the input of layer 46:
Q question tail, M menu, O boundary rows B0/B2/B3, A boundary rows B1/B4 (T = thinking/answer positions, a separate control).  30 independent
scenes; scene bootstrap 4,000 (seed 0); both directions pooled; every ratio is a ratio of sums within each scene resample.
Donor-direction transfer rho = sum (x - r).(d - r) / sum |d - r|^2 (1 = the donor's state, 0 = unchanged) on
  W     writer outputs: token mixer of layer 47 (FA47) and of layer 49 (DN49) at rows B1, B4 (primary, as the user asked; group A replaces these
        rows' own input, so A's W is partly its own input)
  Wans  the same outputs at the answer row (in no group: changes only through what the writer reads; registered secondary)
  Qy    late queries (51-63, rows B1, B4, answer);  Z  late positional scores (rows B1, B4, answer, 51-63, centred);  L  answer logits (centred)
I0 (all): t1 >= .95; image keys identical (0.0); QNOOP_E = END(5) >= .90 with the 95th percentile of the largest menu-logit change <= .5;
  START(k) -> k >= .90 (k 3, 5); pin donors START(3), START(4) -> q >= .90; PIN (layers 43 / 47, rows B1 / B4 maximum = p) >= .95; exactness
  (answers >= .99 and 95th percentile of the largest menu-logit difference <= .05): XSELF = E1, LINSELF = E1, up_QMOAT = E3, dn_QMOAT = E1, and
  the pass-B / pass-C copies of E1-E4 = pass A.
Per group G (needs I0):
  SUFF  rho_W(G) >= .50 with lower bound > .30 -> 'G alone carries the ordinal-condition state into the writer output'; <= .15 with upper bound
        < .25 -> 'G alone does not'; otherwise 'partial'.  The same rule on rho_Wans is reported as SUFF_ans.
  NEC   D = rho_W(QMOA) - rho_W(QMOA without G) >= .30 with lower bound > .15 -> 'G is needed for the four-group patch's writer effect';
        <= .10 with upper bound < .20 -> 'not needed'; otherwise 'partial' (necessity relative to the full four-group patch, not to the natural
        computation - user).  Same on rho_Wans as NEC_ans.
  DOWN  rho_Z(G) >= .20 with lower bound > 0 -> 'carried to the late address' (rho_Qy, rho_L, answers following donor / recipient reported).
PAIR  for each pair G, H: I = rho(G u H) - rho(G) - rho(H) on rho_W and rho_Wans: |I| >= .15 with the interval excluding 0 -> 'G and H combine
      non-additively' (sign reported); interval inside (-.15, .15) -> 'additive within .15'; otherwise 'partial'.
CF    (user's Gap 3) per corner c of the 2 x 2 (k 3 / 5 x p 3 / 4): I_nat = W(diag) - W(k-flip) - W(p-flip) + W(c); I_cf = W(CF_c) - W(k-flip) -
      W(p-flip) + W(c) with CF_c's input = h(k-flip) + h(p-flip) - h(c) at every position from the ordinal token on.  R2_cf = 1 - sum |I_cf|^2 /
      sum |W(CF_c) - W(c)|^2 (FA47 + DN49, rows B1, B4).  R2_cf <= .60 with upper bound < .70 -> 'the writer combines the two changes
      non-additively even for additive inputs'; R2_cf >= .85 with lower bound > .75 -> 'the writer is near-additive for additive inputs (the
      natural non-additivity comes from its input)'; otherwise 'partial'.  Reported with R2_nat, per component, and the distance of the
      additive input from the natural diagonal input (rows B1, B4) - the additive input may leave the natural activation range (user).
Descriptive: Shapley shares of the four groups (rho_W, rho_Wans, rho_Z, rho_L), T alone, p 4 (singles and all four), the images whose late scores
each group raises, answer distributions.
Usage: python exp/s854b_analyze.py [TAG]  ->  results/r3/s854b_analysis.json"""
from __future__ import annotations

import itertools, json, sys
from math import factorial
import numpy as np

V = "/home/storage2/ai26269345/vlm"
B = 4000
GROUPS = ("Q", "M", "O", "A")
def sub_name(gs):
    return "".join(g for g in GROUPS if g in gs)


def main(tag="s854b_27b"):
    d = json.load(open(f"{V}/results/r3/s854b_{tag}_rows.json")); Z = np.load(f"{V}/results/r3/s854b_{tag}_rows.npz")
    comps = [f"{l}{kd}" for l, kd in [tuple(c) for c in d["comps"]]]
    LATE = tuple(str(l) for l in d["late"]); PINL = tuple(str(l) for l in d["early"][-2:])
    FAc, DNc = f"{d['early'][-1]}mix", f"{max(d['band'])}mix"
    COND = {k: tuple(v) for k, v in d["cond"].items()}
    by = {r["si"]: r for r in d["rows"]}; sis = sorted(by); n = len(sis)
    rng = np.random.default_rng(0); idx = rng.integers(0, n, (B, n))
    cen = lambda x: np.asarray(x, dtype=float) - np.mean(x)
    def ci_ratio(num, den):
        num = np.asarray(num, dtype=float); den = np.asarray(den, dtype=float)
        bs = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-12)
        return [float(num.sum() / max(den.sum(), 1e-12)), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]]
    def ci_mean(per):
        per = np.asarray(per, dtype=float); bs = per[idx].mean(1)
        return [float(per.mean()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]]
    mld = lambda r, a1, a2: max(abs(x - y) for x, y in zip(r["mlog"][a1], r["mlog"][a2]))
    comp = lambda s, nm: Z[f"s{s}_{nm}_comp"].astype(np.float64)
    vec = {"W": lambda s, nm: np.concatenate([comp(s, nm)[comps.index(c), r] for c in (FAc, DNc) for r in (0, 1)]),
           "Wans": lambda s, nm: np.concatenate([comp(s, nm)[comps.index(c), 2] for c in (FAc, DNc)]),
           "Qy": lambda s, nm: Z[f"s{s}_{nm}_q"].astype(np.float64).ravel(),
           "Z": lambda s, nm: np.concatenate([cen(by[s]["qk"][nm][L][ri]) for L in LATE for ri in (2, 5, 9)]),
           "L": lambda s, nm: cen(by[s]["mlog"][nm][:8])}
    out = dict(tag=tag, t1=d["t1_menu"], key_rel_diff_max=d["key_rel_diff_max"], n=n, n_runs=d.get("n_runs"), group_sizes=d.get("group_sizes"), I0={})
    # ---------------- I0
    def same(a1, a2):
        ag = float(np.mean([by[s]["land"][a1] == by[s]["land"][a2] for s in sis])); p95 = float(np.percentile([mld(by[s], a1, a2) for s in sis], 95))
        return [ag, p95, float(max(mld(by[s], a1, a2) for s in sis))]
    i0 = dict(t1=d["t1_menu"] >= .95, keys=d["key_rel_diff_max"] == 0.0)
    qn = same("QNOOP_E", "END_k5"); i0["qnoop"] = dict(v=qn, ok=bool(qn[0] >= .90 and qn[1] <= .5))
    st = {k: float(np.mean([by[s]["land"][f"START_q{k}"] == k for s in sis])) for k in (3, 5)}; i0["start_k"] = dict(v=st, ok=bool(min(st.values()) >= .90))
    dn_ = {q: float(np.mean([by[s]["land"][f"START_q{q}"] == q for s in sis])) for q in (3, 4)}; i0["donors"] = dict(v=dn_, ok=bool(min(dn_.values()) >= .90))
    pin = float(np.mean([int(np.argmax(by[s]["qk"][nm][L][ri]) + 1 == COND[nm][1]) for s in sis for nm in COND for L in PINL for ri in (2, 5)]))
    i0["pin"] = dict(v=pin, ok=bool(pin >= .95))
    ex = {"XSELF": same("XSELF", "E1"), "LINSELF": same("LINSELF", "E1"), "up_QMOAT": same("up_QMOAT", "E3"), "dn_QMOAT": same("dn_QMOAT", "E1")}
    for nm in COND:
        for w in ("B", "C"):
            ex[f"{nm}_{w}"] = same(f"{nm}_{w}", nm)
    i0["exact"] = dict(v=ex, ok=bool(all(x[0] >= .99 and x[1] <= .05 for x in ex.values())))
    ok_all = bool(i0["t1"] and i0["keys"] and all(v["ok"] for k, v in i0.items() if isinstance(v, dict)))
    out["I0"] = dict(i0, all=ok_all); nvt = "no valid test"
    # ---------------- transfer per arm (both directions pooled)
    DIRS = (("up", "E1", "E3"), ("dn", "E3", "E1"))
    def transfer(arm_of_dir, measures=("W", "Wans", "Qy", "Z", "L"), dirs=DIRS):
        res = {}
        for mname in measures:
            f = vec[mname]; num, den = [], []
            for s in sis:
                a_ = b_ = 0.0
                for dn, r_, d_ in dirs:
                    x, r0, dd = f(s, arm_of_dir(dn)), f(s, r_), f(s, d_)
                    a_ += float((x - r0) @ (dd - r0)); b_ += float((dd - r0) @ (dd - r0))
                num.append(a_); den.append(b_)
            res[mname] = dict(ratio=ci_ratio(num, den), _num=np.asarray(num), _den=np.asarray(den))
        fn, fr, fd = [], [], []
        for s in sis:
            n_ = r_c = d_c = 0
            for dn, r_, d_ in dirs:
                L = by[s]["land"]; ar, ad, ax = L[r_], L[d_], L[arm_of_dir(dn)]
                if ar != ad:
                    n_ += 1; r_c += int(ax == ar); d_c += int(ax == ad)
            fn.append(n_); fr.append(r_c); fd.append(d_c)
        fn, fr, fd = map(np.asarray, (fn, fr, fd))
        dif = (fd[idx].sum(1) - fr[idx].sum(1)) / np.maximum(fn[idx].sum(1), 1)
        res["answers"] = dict(n_discordant=int(fn.sum()), f_don=float(fd.sum() / max(fn.sum(), 1)), f_rec=float(fr.sum() / max(fn.sum(), 1)),
                              don_minus_rec=[float((fd.sum() - fr.sum()) / max(fn.sum(), 1)), [float(np.percentile(dif, 2.5)), float(np.percentile(dif, 97.5))]])
        return res
    subsets = [s_ for s_ in d["subsets"]]
    T = {sub: transfer(lambda dn, sub=sub: f"{dn}_{sub}") for sub in subsets}
    T["T"] = transfer(lambda dn: f"{dn}_T")
    def rho(sub, m):
        return T[sub][m]["ratio"] if sub else [0.0, [0.0, 0.0]]
    def diffci(sub1, sub2, m):                                              # rho(sub1) - rho(sub2) with a joint scene bootstrap
        n1, d1 = T[sub1][m]["_num"], T[sub1][m]["_den"]
        if sub2:
            n2, d2 = T[sub2][m]["_num"], T[sub2][m]["_den"]
            bs = n1[idx].sum(1) / d1[idx].sum(1) - n2[idx].sum(1) / d2[idx].sum(1); pt = n1.sum() / d1.sum() - n2.sum() / d2.sum()
        else:
            bs = n1[idx].sum(1) / d1[idx].sum(1); pt = n1.sum() / d1.sum()
        return [float(pt), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]]
    def inter(g, h, m):
        gh = sub_name({g, h}); a_, b_, c_ = T[gh][m], T[g][m], T[h][m]
        bs = (a_["_num"][idx].sum(1) / a_["_den"][idx].sum(1) - b_["_num"][idx].sum(1) / b_["_den"][idx].sum(1)
              - c_["_num"][idx].sum(1) / c_["_den"][idx].sum(1))
        pt = a_["_num"].sum() / a_["_den"].sum() - b_["_num"].sum() / b_["_den"].sum() - c_["_num"].sum() / c_["_den"].sum()
        return [float(pt), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]]
    full = "QMOA"
    out["GROUPS"] = {}
    for g in GROUPS:
        rec = {}
        for m, tagm in (("W", ""), ("Wans", "_ans")):
            r_ = rho(g, m)
            rec["SUFF" + tagm] = dict(rho=r_, verdict=(nvt if not ok_all else "G alone carries the ordinal-condition state into the writer output"
                                                     if r_[0] >= .50 and r_[1][0] > .30 else "G alone does not" if r_[0] <= .15 and r_[1][1] < .25 else "partial"))
            dn_v = diffci(full, sub_name(set(GROUPS) - {g}), m)
            rec["NEC" + tagm] = dict(D=dn_v, verdict=(nvt if not ok_all else "G is needed for the four-group patch's writer effect"
                                                     if dn_v[0] >= .30 and dn_v[1][0] > .15 else "not needed" if dn_v[0] <= .10 and dn_v[1][1] < .20 else "partial"))
        rz = rho(g, "Z")
        rec["DOWN"] = dict(rho_Z=rz, rho_Qy=rho(g, "Qy"), rho_L=rho(g, "L"), answers=T[g]["answers"],
                           verdict=(nvt if not ok_all else "carried to the late address" if rz[0] >= .20 and rz[1][0] > 0 else "not shown"))
        out["GROUPS"][g] = rec
    out["PAIRS"] = {}
    for g, h in itertools.combinations(GROUPS, 2):
        rec = {}
        for m in ("W", "Wans", "Z"):
            iv = inter(g, h, m)
            rec[m] = dict(I=iv, verdict=(nvt if not ok_all else ("G and H combine non-additively (" + ("super" if iv[0] > 0 else "sub") + "-additive)")
                                         if abs(iv[0]) >= .15 and (iv[1][0] > 0 or iv[1][1] < 0) else "additive within .15"
                                         if iv[1][0] > -.15 and iv[1][1] < .15 else "partial"))
        out["PAIRS"][f"{g}{h}"] = rec
    out["FULL"] = {m: rho(full, m) for m in ("W", "Wans", "Qy", "Z", "L")}; out["FULL"]["answers"] = T[full]["answers"]
    out["T_alone"] = {m: T["T"][m]["ratio"] for m in ("W", "Wans", "Qy", "Z", "L")}; out["T_alone"]["answers"] = T["T"]["answers"]
    # ---------------- Shapley shares over the four-group game
    def shapley(m):
        val = {frozenset(): 0.0}
        for sub in subsets:
            val[frozenset(sub_name_set(sub))] = T[sub][m]["ratio"][0]
        ph = {}
        for g in GROUPS:
            others = [x for x in GROUPS if x != g]; tot = 0.0
            for r_ in range(0, 4):
                for S_ in itertools.combinations(others, r_):
                    w = factorial(len(S_)) * factorial(4 - len(S_) - 1) / factorial(4)
                    tot += w * (val[frozenset(S_) | {g}] - val[frozenset(S_)])
            ph[g] = float(tot)
        return ph
    def sub_name_set(sub):
        return {c for c in sub}
    out["SHAPLEY"] = {m: shapley(m) for m in ("W", "Wans", "Z", "L")}
    # ---------------- p 4 generalisation (descriptive)
    DIRS4 = (("up4", "E2", "E4"), ("dn4", "E4", "E2"))
    out["P4"] = {sub: {m: v["ratio"] for m, v in transfer(lambda dn, sub=sub: f"{dn}_{sub}", dirs=DIRS4).items() if m != "answers"}
                 for sub in ("Q", "M", "O", "A", "QMOA")}
    # ---------------- CF (Gap 3)
    KF, PF, DG = d["kflip"], d["pflip"], d["diag"]
    def r2(fvec, use_cf):
        num, den = [], []
        for s in sis:
            a_ = b_ = 0.0
            for c in COND:
                w0, wk, wp = fvec(s, c), fvec(s, KF[c]), fvec(s, PF[c]); wx = fvec(s, f"CF_{c}") if use_cf else fvec(s, DG[c])
                a_ += float(np.sum((wx - wk - wp + w0) ** 2)); b_ += float(np.sum((wx - w0) ** 2))
            num.append(a_); den.append(b_)
        num, den = np.asarray(num), np.asarray(den); bs = 1 - num[idx].sum(1) / den[idx].sum(1)
        return [float(1 - num.sum() / den.sum()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]]
    rcf = r2(vec["W"], True); rnat = r2(vec["W"], False)
    percomp = {cn: dict(cf=r2(lambda s, nm, cn=cn: np.concatenate([comp(s, nm)[comps.index(cn), r] for r in (0, 1)]), True),
                        nat=r2(lambda s, nm, cn=cn: np.concatenate([comp(s, nm)[comps.index(cn), r] for r in (0, 1)]), False)) for cn in comps}
    resv = lambda s, nm: Z[f"s{s}_{nm}_res"].astype(np.float64)[:2].ravel()
    din = [float(np.linalg.norm(resv(s, f"CF_{c}") - resv(s, DG[c])) / max(np.linalg.norm(resv(s, DG[c]) - resv(s, c)), 1e-9)) for s in sis for c in COND]
    dout = [float(np.linalg.norm(vec["W"](s, f"CF_{c}") - vec["W"](s, DG[c])) / max(np.linalg.norm(vec["W"](s, DG[c]) - vec["W"](s, c)), 1e-9))
            for s in sis for c in COND]
    cfv = (nvt if not ok_all else "the writer combines the two changes non-additively even for additive inputs" if rcf[0] <= .60 and rcf[1][1] < .70
           else "the writer is near-additive for additive inputs (the natural non-additivity comes from its input)" if rcf[0] >= .85 and rcf[1][0] > .75
           else "partial")
    out["CF"] = dict(R2_cf=rcf, R2_nat=rnat, per_component=percomp, input_dist_cf_vs_diag_median=float(np.median(din)),
                     output_dist_cf_vs_diag_median=float(np.median(dout)), downstream_cf={m: r2(vec[m], True) for m in ("Qy", "Z", "L")},
                     downstream_nat={m: r2(vec[m], False) for m in ("Qy", "Z", "L")}, verdict=cfv)
    # ---------------- images each group raises (late scores, rows B1/B4/answer, 51-63; up and dn separately)
    def raise_img(sub):
        o = {}
        for dn, r_, d_ in DIRS:
            dz = np.mean([np.mean([np.asarray(by[s]["qk"][f"{dn}_{sub}"][L][ri]) - np.asarray(by[s]["qk"][r_][L][ri]) for L in LATE for ri in (2, 5, 9)], 0)
                          for s in sis], 0)
            dzd = np.mean([np.mean([np.asarray(by[s]["qk"][d_][L][ri]) - np.asarray(by[s]["qk"][r_][L][ri]) for L in LATE for ri in (2, 5, 9)], 0)
                           for s in sis], 0)
            o[dn] = dict(delta=[float(x - dz.mean()) for x in dz], donor_delta=[float(x - dzd.mean()) for x in dzd], top=int(np.argmax(dz - dz.mean()) + 1))
        return o
    out["RAISE"] = {sub: raise_img(sub) for sub in ("Q", "M", "O", "A", "QMOA", "T")}
    out["dist"] = {nm: {str(x): float(np.mean([by[s]["land"][nm] == x for s in sis])) for x in range(9) if any(by[s]["land"][nm] == x for s in sis)}
                   for nm in ["E1", "E2", "E3", "E4"] + [f"{dn}_{sub}" for dn in ("up", "dn") for sub in subsets + ["T", "QMOAT"]] + [f"CF_{c}" for c in COND]}
    for g in T.values():
        for m in g.values():
            if isinstance(m, dict):
                m.pop("_num", None); m.pop("_den", None)
    fn = f"{V}/results/r3/s854b_analysis.json" if tag == "s854b_27b" else f"{V}/results/r3/s854b_analysis_{tag}.json"
    json.dump(out, open(fn, "w"), indent=1)
    print("I0", json.dumps(out["I0"])[:1200])
    for g, v in out["GROUPS"].items():
        print(g, "SUFF", v["SUFF"]["rho"][0], v["SUFF"]["verdict"], "| SUFF_ans", v["SUFF_ans"]["verdict"], "| NEC", v["NEC"]["verdict"], "| NEC_ans",
              v["NEC_ans"]["verdict"], "| DOWN", v["DOWN"]["verdict"])
    print("PAIRS", json.dumps({k: {m: x["verdict"] for m, x in v.items()} for k, v in out["PAIRS"].items()}))
    print("CF", out["CF"]["R2_cf"], out["CF"]["R2_nat"], out["CF"]["verdict"])


if __name__ == "__main__":
    main(*sys.argv[1:])
