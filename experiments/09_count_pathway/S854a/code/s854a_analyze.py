"""S854-1 analysis (rules fixed in knowledge/DECISIONS.md before any S854-1 data; user 10-18 'GO - separate the ordinal from the image counts').
Factorial per scene: actual pictures N_a {6, 8} x stated count N_s {6, 8} x ordinal k {3, 5} x pinned early position p {3, 4}; 30 independent
scenes; scene bootstrap 4,000 (seed 0).  j_a = N_a + 1 - k (target from the actual end), j_s = N_s + 1 - k (target from the stated count).
Three candidate computations (user): H_k ordinal-centred predecessor (the late update depends on (k, p), not on N_a or N_s); H_a actual-end
target (depends on (j_a, p)); H_s stated-count target (depends on (j_s, p)).  j_a and j_s are fixed by (N_a, N_s, k), so the design cannot prove
an independent ordinal register; it can show which computation fails to explain the observed changes.  N_s != N_a are counterfactual probes.
I0 (per N_a, all must pass): t1 >= .95; image keys identical within N_a (0.0); QNOOP_E = END(N_s = N_a, k 5) in >= .90 with the 95th
  percentile of the largest menu-logit change <= .5; START(k) -> k in >= .90 for every (N_s, k); pin donors START(p) -> p in >= .90 pooled
  (>= .80 each); PIN: the block-score maximum of rows B1, B4 at layers 43 and 47 is the pinned p in >= .95 of the E runs.
MAIN-LL (primary; answers of the 16 pinned END conditions, nine menu words): leave-one-scene-out held-out log-likelihood per answer of
  equivalence-class models (class distribution from the other 29 scenes, add-1/2 smoothing): full (each condition), H_k (k, p), H_a (j_a, p),
  H_s (j_s, p), and the auxiliary H_ak (N_a, k, p) and H_sk (N_s, k, p); baselines (user): empirical predecessor (P(p - 1) = training rate,
  the rest uniform), actual target (P(j_a) = training rate), stated target (P(j_s) = training rate), uniform.
  Per H in {H_k, H_a, H_s}: LL(H) - LL(full) < 0 with upper bound < 0 -> 'H is rejected (pooling its equivalent conditions hurts)', otherwise
  'H is compatible'.  Best: LL(H_x) - max of the other two > 0 with lower bound > 0 -> 'H_x explains the answers best', otherwise 'not separated'.
C1 (the user's pair; same actual and stated target 4, p 3): D = P(answer 2 | N_a 8, N_s 8, k 5) - P(answer 2 | N_a 6, N_s 6, k 3).  D >= .20 with
  lower bound > 0 -> 'at the same target the predecessor step follows the ordinal'; |D| < .10 with the interval inside (-.20, .20) -> 'at the same
  target the predecessor step is the same'; otherwise 'partial'.
C2 (k 5, p 3 in the four count cells): every P(answer 2) >= .30 -> 'the k = 5 predecessor bias persists across the actual and stated counts';
  any < .15 -> 'the k = 5 predecessor bias depends on the counts'; otherwise 'partial'.
INT-REP (registered replication of the post hoc S853 analysis, research/ces/outputs/s853_int.json): natural 2 x 2 (k x p) at N_a = N_s = 8 (and
  at 6 / 6, reported alongside); corner R2_add = 1 - sum |D_KP - D_K - D_P|^2 / sum |D_KP|^2 over the four corners, per level: W_in (residual
  at the input of layer 46, rows B1, B4), FA47 and DN49 outputs (rows B1, B4), late queries Q, late scores Z (pictures 1-N_a, centred), answer
  logits L (shown pictures, centred).  FA47 R2_add <= .60 with upper bound < .70 -> 'the non-additivity is present at the writer output';
  R2_add(L) - R2_add(FA47) >= 0 with lower bound > -.05 -> 'it does not grow downstream'; W_in R2_add <= .60 with upper bound < .70 -> 'already
  present at the writer input'; W_in >= .80 with lower bound > .70 while FA47 passes -> 'it arises in the writer'; otherwise 'partial'.
COMPACT (secondary): within each scene, each condition's vector is predicted by the mean of the other conditions in its class (same p) under
  H_k, H_a, H_s; error ratio E_H = sum |v - class mean|^2 / sum |v - mean of all other same-p conditions|^2, for (i) the END-specific write
  D = band_E - band_S (FA47 and DN49, rows B1, B4) and (ii) the late scores Z (pictures 1-6, centred).  E_Hx < the other two with the interval of
  the difference to the runner-up excluding 0 -> 'H_x's classes predict it best'; otherwise 'not separated'.
Usage: python exp/s854a_analyze.py [TAG]  ->  results/r3/s854a_analysis.json"""
from __future__ import annotations

import json, os, sys
import numpy as np

V = "/home/storage2/ai26269345/vlm"
B = 4000
NAS = (8, 6); NSS = (6, 8); KS = (3, 5); PS = (3, 4)
CONDS = [(na, ns, k, p) for na in NAS for ns in NSS for k in KS for p in PS]
en = lambda na, ns, k, p: f"E_s{ns}k{k}p{p}_a{na}"
sn = lambda na, ns, k, p: f"S_s{ns}k{k}p{p}_a{na}"


def main(tag="s854a_27b"):
    d = json.load(open(f"{V}/results/r3/s854a_{tag}_count.json")); R = d["rows"]
    Z = np.load(f"{V}/results/r3/s854a_{tag}_count.npz")
    comps = [f"{l}{kd}" for l, kd in [tuple(c) for c in d["comps"]]]
    LATE = tuple(str(l) for l in d["late"]); PINL = tuple(str(l) for l in d["early"][-2:])     # 51-63 and 43, 47 in the 27B
    FAc, DNc = f"{d['early'][-1]}mix", f"{max(d['band'])}mix"                                    # 47mix (FA47) and 49mix (DN49) in the 27B
    by = {r["si"]: r for r in R}; sis = sorted(by); n = len(sis)
    rng = np.random.default_rng(0); idx = rng.integers(0, n, (B, n))
    def ci_mean(per):
        per = np.asarray(per, dtype=float); bs = per[idx].mean(1)
        return [float(per.mean()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]]
    mld = lambda r, a1, a2: max(abs(x - y) for x, y in zip(r["mlog"][a1], r["mlog"][a2]))
    out = dict(tag=tag, t1=d["t1_menu"], key_rel_diff_max=d["key_rel_diff_max"], n=n, n_runs=d.get("n_runs"), I0={})
    ok_all = True
    for na in NAS:
        rows = [by[s] for s in sis]
        qn = float(np.mean([r["land"][f"QNOOP_E_a{na}"] == r["land"][f"END_s{na}k5_a{na}"] for r in rows]))
        qp = float(np.percentile([mld(r, f"QNOOP_E_a{na}", f"END_s{na}k5_a{na}") for r in rows], 95))
        st = {f"s{ns}k{k}": float(np.mean([r["land"][f"START_s{ns}q{k}_a{na}"] == k for r in rows])) for ns in NSS for k in KS}
        dn = {f"s{ns}q{p}": float(np.mean([r["land"][f"START_s{ns}q{p}_a{na}"] == p for r in rows])) for ns in NSS for p in PS}
        pin = float(np.mean([int(np.argmax(r["qk"][en(na, ns, k, p)][L][ri]) + 1 == p) for r in rows for ns in NSS for k in KS for p in PS
                             for L in PINL for ri in (2, 5)]))
        ok = bool(d["t1_menu"][str(na)] >= .95 and d["key_rel_diff_max"][str(na)] == 0.0 and qn >= .90 and qp <= .5 and min(st.values()) >= .90
                  and np.mean(list(dn.values())) >= .90 and min(dn.values()) >= .80 and pin >= .95)
        out["I0"][str(na)] = dict(t1=d["t1_menu"][str(na)], qnoop=qn, p95=qp, start_k=st, donors=dn, pin=pin, ok=ok); ok_all = ok_all and ok
    out["I0"]["all"] = bool(ok_all); nvt = "no valid test"
    # ---------------- MAIN-LL
    ans = {(s, c): by[s]["land"][en(*c)] for s in sis for c in CONDS}            # 0 = the extra word, 1-8 = picture labels (7, 8 unshown at N_a 6)
    keyf = {"full": lambda c: c, "H_k": lambda c: (c[2], c[3]), "H_a": lambda c: (c[0] + 1 - c[2], c[3]), "H_s": lambda c: (c[1] + 1 - c[2], c[3]),
            "H_ak": lambda c: (c[0], c[2], c[3]), "H_sk": lambda c: (c[1], c[2], c[3])}
    ll = {m: np.zeros(n) for m in list(keyf) + ["B_pred", "B_a", "B_s", "B_uni"]}
    for ti, s in enumerate(sis):
        train = [x for x in sis if x != s]
        for m, kf in keyf.items():
            cnt = {}
            for x in train:
                for c in CONDS:
                    cnt.setdefault(kf(c), np.zeros(9))[ans[(x, c)]] += 1
            for c in CONDS:
                v_ = cnt[kf(c)] + .5; ll[m][ti] += float(np.log(v_[ans[(s, c)]] / v_.sum()))
        for m, tgt in (("B_pred", lambda c: c[3] - 1), ("B_a", lambda c: c[0] + 1 - c[2]), ("B_s", lambda c: c[1] + 1 - c[2])):
            hit = np.mean([ans[(x, c)] == tgt(c) for x in train for c in CONDS]); hit = min(max(hit, 1e-3), 1 - 1e-3)
            for c in CONDS:
                ll[m][ti] += float(np.log(hit if ans[(s, c)] == tgt(c) else (1 - hit) / 8))
        ll["B_uni"][ti] = len(CONDS) * float(np.log(1 / 9))
    per = {m: v_ / len(CONDS) for m, v_ in ll.items()}                          # per answer, per scene
    out["MAIN_LL"] = dict(ll_per_answer={m: float(v_.mean()) for m, v_ in per.items()})
    diff = lambda a_, b_: ci_mean(per[a_] - per[b_])
    rej = {}
    for h in ("H_k", "H_a", "H_s"):
        dd = diff(h, "full")
        rej[h] = dict(vs_full=dd, verdict=(nvt if not ok_all else f"{h} is rejected (pooling its equivalent conditions hurts)"
                                            if dd[0] < 0 and dd[1][1] < 0 else f"{h} is compatible"))
    out["MAIN_LL"]["per_H"] = rej
    best = None
    for h in ("H_k", "H_a", "H_s"):
        others = [x for x in ("H_k", "H_a", "H_s") if x != h]
        ru = max(others, key=lambda x: per[x].mean()); dd = diff(h, ru)
        if dd[0] > 0 and dd[1][0] > 0:
            best = (h, ru, dd)
    out["MAIN_LL"]["best"] = (nvt if not ok_all else f"{best[0]} explains the answers best (vs {best[1]}: {best[2][0]:+.3f} "
                              f"[{best[2][1][0]:+.3f}, {best[2][1][1]:+.3f}])" if best else "not separated")
    out["MAIN_LL"]["pairwise"] = {f"{a_}-{b_}": diff(a_, b_) for a_, b_ in [("H_k", "H_a"), ("H_k", "H_s"), ("H_a", "H_s"), ("H_k", "H_ak"),
                                                                             ("H_k", "H_sk"), ("H_ak", "full"), ("H_sk", "full"), ("H_k", "B_pred"),
                                                                             ("H_a", "B_a"), ("H_s", "B_s")]}
    # ---------------- C1, C2
    r2 = lambda c: [float(ans[(s, c)] == 2) for s in sis]
    c1 = ci_mean(np.subtract(r2((8, 8, 5, 3)), r2((6, 6, 3, 3))))
    out["C1"] = dict(D=c1, rates={"8_8_5_3": float(np.mean(r2((8, 8, 5, 3)))), "6_6_3_3": float(np.mean(r2((6, 6, 3, 3))))},
                     verdict=(nvt if not ok_all else "at the same target the predecessor step follows the ordinal" if c1[0] >= .20 and c1[1][0] > 0
                              else "at the same target the predecessor step is the same" if abs(c1[0]) < .10 and c1[1][0] > -.20 and c1[1][1] < .20
                              else "partial"))
    c2 = {f"{na}_{ns}": float(np.mean(r2((na, ns, 5, 3)))) for na in NAS for ns in NSS}
    out["C2"] = dict(rates=c2, verdict=(nvt if not ok_all else "the k = 5 predecessor bias persists across the actual and stated counts"
                                        if min(c2.values()) >= .30 else "the k = 5 predecessor bias depends on the counts"
                                        if min(c2.values()) < .15 else "partial"))
    # ---------------- INT-REP
    cen = lambda x: np.asarray(x, dtype=float) - np.mean(x)
    def lvl(na, ns):
        W = lambda s, c, cn: np.concatenate([Z[f"s{s}_{en(*c)}_comp"].astype(np.float64)[comps.index(cn), r] for r in (0, 1)])
        return {"W_in": lambda s, c: Z[f"s{s}_{en(*c)}_res"].astype(np.float64)[:2].ravel(),
                "FA47": lambda s, c: W(s, c, FAc), "DN49": lambda s, c: W(s, c, DNc),
                "Q": lambda s, c: Z[f"s{s}_{en(*c)}_q"].astype(np.float64).ravel(),
                "Z": lambda s, c: np.concatenate([cen(by[s]["qk"][en(*c)][L][ri][:na]) for L in LATE for ri in (2, 5, 9)]),
                "L": lambda s, c: cen(by[s]["mlog"][en(*c)][:na])}
    def r2add(na, ns, f):
        cell = {(k, p): (na, ns, k, p) for k in KS for p in PS}
        nat = {(3, 3): ((5, 3), (3, 4), (5, 4)), (3, 4): ((5, 4), (3, 3), (5, 3)), (5, 3): ((3, 3), (5, 4), (3, 4)), (5, 4): ((3, 4), (5, 3), (3, 3))}
        num, den = [], []
        for s in sis:
            a_ = b_ = 0.0
            for c0, (ck, cp, ckp) in nat.items():
                z0 = f(s, cell[c0]); dk, dp, dkp = f(s, cell[ck]) - z0, f(s, cell[cp]) - z0, f(s, cell[ckp]) - z0
                a_ += float(np.sum((dkp - dk - dp) ** 2)); b_ += float(np.sum(dkp ** 2))
            num.append(a_); den.append(b_)
        return np.asarray(num), np.asarray(den)
    intrep = {}
    for na, ns in ((8, 8), (6, 6)):
        res = {}
        for lv, f in lvl(na, ns).items():
            nu, de = r2add(na, ns, f); bs = 1 - nu[idx].sum(1) / de[idx].sum(1)
            res[lv] = dict(R2_add=[float(1 - nu.sum() / de.sum()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]], _nu=nu, _de=de)
        nuL, deL, nuF, deF = res["L"]["_nu"], res["L"]["_de"], res["FA47"]["_nu"], res["FA47"]["_de"]
        dLF = (1 - nuL[idx].sum(1) / deL[idx].sum(1)) - (1 - nuF[idx].sum(1) / deF[idx].sum(1))
        res["L_minus_FA47"] = [float((1 - nuL.sum() / deL.sum()) - (1 - nuF.sum() / deF.sum())), [float(np.percentile(dLF, 2.5)), float(np.percentile(dLF, 97.5))]]
        for lv in list(res):
            if isinstance(res[lv], dict):
                res[lv].pop("_nu", None); res[lv].pop("_de", None)
        intrep[f"{na}_{ns}"] = res
    m8 = intrep["8_8"]; fa = m8["FA47"]["R2_add"]; wi = m8["W_in"]["R2_add"]; lf = m8["L_minus_FA47"]
    v1 = "the non-additivity is present at the writer output" if fa[0] <= .60 and fa[1][1] < .70 else "not shown at the writer output"
    v2 = "it does not grow downstream" if lf[0] >= 0 and lf[1][0] > -.05 else "it grows downstream" if lf[1][1] < 0 else "partial"
    v3 = ("already present at the writer input" if wi[0] <= .60 and wi[1][1] < .70 else
          "it arises in the writer" if wi[0] >= .80 and wi[1][0] > .70 and fa[0] <= .60 and fa[1][1] < .70 else "partial")
    out["INT_REP"] = dict(levels=intrep, verdicts=(nvt if not ok_all else dict(writer_output=v1, downstream=v2, origin=v3)))
    # ---------------- COMPACT
    def vecD(s, c):
        e_ = Z[f"s{s}_{en(*c)}_comp"].astype(np.float64); s_ = Z[f"s{s}_{sn(*c)}_comp"].astype(np.float64)
        return np.concatenate([(e_ - s_)[comps.index(cn), r] for cn in (FAc, DNc) for r in (0, 1)])
    vecZ = lambda s, c: np.concatenate([cen(by[s]["qk"][en(*c)][L][ri][:6]) for L in LATE for ri in (2, 5, 9)])
    comp = {}
    for nmv, f in (("D_write", vecD), ("Z_late", vecZ)):
        errs = {h: np.zeros(n) for h in ("H_k", "H_a", "H_s", "null")}
        for ti, s in enumerate(sis):
            vs_ = {c: f(s, c) for c in CONDS}
            for c in CONDS:
                same_p = [x for x in CONDS if x != c and x[3] == c[3]]
                errs["null"][ti] += float(np.sum((vs_[c] - np.mean([vs_[x] for x in same_p], 0)) ** 2))
                for h in ("H_k", "H_a", "H_s"):
                    cl = [x for x in same_p if keyf[h](x) == keyf[h](c)]
                    errs[h][ti] += float(np.sum((vs_[c] - np.mean([vs_[x] for x in cl], 0)) ** 2))
        ratio = {h: float(errs[h].sum() / errs["null"].sum()) for h in ("H_k", "H_a", "H_s")}
        bsr = {h: errs[h][idx].sum(1) / errs["null"][idx].sum(1) for h in ("H_k", "H_a", "H_s")}
        order = sorted(ratio, key=ratio.get); bst, ru = order[0], order[1]; dd = bsr[bst] - bsr[ru]
        comp[nmv] = dict(E=ratio, best_minus_runner=[float(ratio[bst] - ratio[ru]), [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]],
                         verdict=(nvt if not ok_all else f"{bst}'s classes predict it best" if np.percentile(dd, 97.5) < 0 else "not separated"))
    out["COMPACT"] = comp
    # ---------------- descriptive: answer distributions per condition
    out["dist"] = {f"{na}_{ns}_{k}_{p}": {str(x): float(np.mean([ans[(s, (na, ns, k, p))] == x for s in sis])) for x in range(9)
                                         if any(ans[(s, (na, ns, k, p))] == x for s in sis)} for na, ns, k, p in CONDS}
    fn = f"{V}/results/r3/s854a_analysis.json" if tag == "s854a_27b" else f"{V}/results/r3/s854a_analysis_{tag}.json"
    json.dump(out, open(fn, "w"), indent=1)
    print("I0", json.dumps(out["I0"])[:900]); print("MAIN_LL", json.dumps({k: v for k, v in out["MAIN_LL"].items() if k != "pairwise"})[:1500])
    print("C1", out["C1"]); print("C2", out["C2"]); print("INT_REP verdicts", out["INT_REP"]["verdicts"])
    print("COMPACT", json.dumps(out["COMPACT"]))


if __name__ == "__main__":
    main(*sys.argv[1:])
