"""S846 analysis (rules fixed in knowledge/DECISIONS.md before any S846 data).  8 pictures, k = 4, stated count N_s in {6, 7, 8, 9} (stated target
t = N_s - 3), early lookup pinned to p in {3, 4, 5, 6}; the same 30 scenes for every N_s; paired scene bootstrap (4,000 draws, seed 0).
I0 (per N_s; all four must pass): t1 >= .95; QNOOP_E = END in >= .90 of scenes with the 95th percentile of the largest menu-logit change <= .5;
START(4) -> 4 >= .90; donors START(p) -> p >= .90 pooled (>= .80 each); image keys identical (0.0).
PC (needed for WRITER): S_EARLY_p -> 4 for p in {3, 5, 6}, pooled, >= .55 (S830: START keeps the ordinal from a moved early position, .66-.75).
  NOSTOP   stay(p, N_s) = P(E_EARLY_p answers p).  D = mean over p of [stay(p, N_s = p + 3) - mean of stay(p, other N_s)]:
           |D| <= .10 with interval inside [-.15, .15] -> 'no stop signal at the goal'; D >= .15 with interval above 0 -> 'a stop signal at the
           goal'; otherwise 'unresolved'.  Internal: h = z[p] - max(z[p - 1], z[p + 1]) (matcher-head block scores), averaged over layers 51, 55
           and the boundary B4 and answer rows; the same contrast Dh on E_EARLY_p, scaled by gap = mean over conditions of h(START(p)) - h(E_EARLY_p):
           |Dh| <= .10 gap with interval inside [-.20 gap, .20 gap] -> 'no internal stop signal'; Dh >= .25 gap with interval above 0 -> 'an
           internal stop signal'; otherwise 'unresolved'.
  PDEP     stay(p = 6) - stay(p = 3), pooled over N_s, >= .25 with interval above 0 -> 'the departure depends on the early position'.
  WRITER   p in {3, 5, 6}, pooled over N_s.  BANDF (END pinned at p with S_EARLY_p's band outputs): A_k = P(4), A_p = P(p):
           A_k >= .60 and A_k - A_p >= .30 (interval above 0) -> 'START's band brings back the ordinal'; A_p >= .60 and A_p - A_k >= .30 (interval
           above 0) -> 'without END's band write the late lookup stays at the early position'; otherwise 'mixed'.
           BANDE (START(k) pinned at p with E_EARLY_p's band outputs): B_k = P(4), S_k = P(S_EARLY_p = 4): B_k <= .40 and S_k - B_k >= .25 (interval
           above 0) -> 'END's band takes START off the ordinal'; B_k >= .55 and S_k - B_k <= .15 -> 'START keeps the ordinal despite END's band';
           otherwise 'partial'.
           SWAP: both first verdicts -> 'the band carries the frame's late anchor rule'.
  PRED     frozen L_pos (goal-blind position law from S844; research/ces/outputs/s846_frozen_laws.json): share of (condition x picture) cells with
           |predicted - observed| <= .15 among E_EARLY_p >= .70 -> 'the goal-blind position law predicts fresh scenes'.
Descriptive: per-condition table; within-p (N_s) and across-p slopes of the answer margin and of z[p + 1] - z[p - 1]; layer 51 / 55 answer-row
maximum = 4 and = p shares for E_EARLY, S_EARLY, BANDF, BANDE; paired agreement BANDE = E_EARLY and BANDF = S_EARLY; log-likelihoods of L_pos,
L_cond and M_mix.
Usage: python exp/s846_analyze.py [TAG]  ->  results/r3/s846_analysis.json"""
from __future__ import annotations

import json, sys
import numpy as np

V = "/home/storage2/ai26269345/vlm"
B = 4000
PS = (3, 4, 5, 6); NSS = (6, 7, 8, 9); PW = (3, 5, 6)


def main(tag="s846_27b"):
    d = json.load(open(f"{V}/results/r3/s846_{tag}_swap.json")); R = d["rows"]
    laws = json.load(open(f"{V}/research/ces/outputs/s846_frozen_laws.json"))["predictions_N8"]
    rng = np.random.default_rng(0)
    by = {ns: {r["si"]: r for r in R if r["Ns"] == ns} for ns in NSS}
    sis = sorted(set.intersection(*[set(x) for x in by.values()]))
    late = [str(x) for x in d["late"]]
    k = 4
    out = dict(tag=tag, t1=d["t1_menu"], key_rel_diff_max=d["key_rel_diff_max"], n=len(sis), I0={})
    ok_all = True
    for ns in NSS:
        rows = [by[ns][s] for s in sis]
        ag = float(np.mean([r["land"]["QNOOP_E"] == r["land"]["END"] for r in rows]))
        p95 = float(np.percentile([max(abs(x - y) for x, y in zip(r["mlog"]["QNOOP_E"], r["mlog"]["END"])) for r in rows], 95))
        st = float(np.mean([r["land"]["START"] == k for r in rows]))
        don = {p: float(np.mean([r["land"][f"START_p{p}"] == p for r in rows])) for p in PS}
        ok = bool(d["t1_menu"] >= .95 and ag >= .90 and p95 <= .5 and st >= .90 and np.mean(list(don.values())) >= .90 and min(don.values()) >= .80
                  and d["key_rel_diff_max"] == 0.0)
        ok_all &= ok
        out["I0"][str(ns)] = dict(qnoop=ag, p95=p95, start_acc=st, donors=don, ok=ok)
    out["I0"]["all"] = bool(ok_all)
    A = lambda arm, ns, p, x, ss: float(np.mean([by[ns][s]["land"][f"{arm}_p{p}"] == x for s in ss]))
    pc = float(np.mean([A("S_EARLY", ns, p, k, sis) for ns in NSS for p in PW]))
    out["PC"] = dict(value=pc, ok=bool(pc >= .55))
    def h(arm, ns, p, ss):
        v = []
        for s in ss:
            for L in late:
                for row in (5, 9):
                    z = by[ns][s]["qk"][arm.format(p=p)][L][row]
                    v.append(z[p - 1] - max(z[p - 2], z[p] if p < len(z) else -1e9))
        return float(np.mean(v))
    def boot(f):
        v = [f([sis[i] for i in rng.integers(0, len(sis), len(sis))]) for _ in range(B)]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    def contrast(f):
        def g(ss):
            return float(np.mean([f(p + 3, p, ss) - np.mean([f(ns, p, ss) for ns in NSS if ns != p + 3]) for p in PS]))
        return g
    stay = lambda ns, p, ss: A("E_EARLY", ns, p, p, ss)
    Dst = contrast(stay); Dh = contrast(lambda ns, p, ss: h("E_EARLY_p{p}", ns, p, ss))
    gap = float(np.mean([h("START_p{p}", ns, p, sis) - h("E_EARLY_p{p}", ns, p, sis) for ns in NSS for p in PS]))
    v1, c1 = Dst(sis), boot(Dst); v2, c2 = Dh(sis), boot(Dh)
    nv = "no valid run"
    out["NOSTOP"] = dict(D=v1, ci=c1, Dh=v2, ci_h=c2, gap=gap,
                         answers=(nv if not ok_all else "no stop signal at the goal" if abs(v1) <= .10 and c1[0] >= -.15 and c1[1] <= .15
                                  else "a stop signal at the goal" if v1 >= .15 and c1[0] > 0 else "unresolved"),
                         internal=(nv if not ok_all else "no internal stop signal" if abs(v2) <= .10 * gap and c2[0] >= -.20 * gap and c2[1] <= .20 * gap
                                   else "an internal stop signal" if v2 >= .25 * gap and c2[0] > 0 else "unresolved"))
    pdep = lambda ss: float(np.mean([stay(ns, 6, ss) for ns in NSS]) - np.mean([stay(ns, 3, ss) for ns in NSS]))
    pv, pci = pdep(sis), boot(pdep)
    out["PDEP"] = dict(value=pv, ci=pci, verdict=(nv if not ok_all else "the departure depends on the early position" if pv >= .25 and pci[0] > 0 else "not shown"))
    pool = lambda arm, x_of_p, ss: float(np.mean([A(arm, ns, p, x_of_p(p), ss) for ns in NSS for p in PW]))
    Ak = lambda ss: pool("BANDF", lambda p: k, ss); Ap = lambda ss: pool("BANDF", lambda p: p, ss)
    Bk = lambda ss: pool("BANDE", lambda p: k, ss); Sk = lambda ss: pool("S_EARLY", lambda p: k, ss)
    dkp = lambda ss: Ak(ss) - Ap(ss); dpk = lambda ss: Ap(ss) - Ak(ss); dsb = lambda ss: Sk(ss) - Bk(ss)
    ck, cp, cs = boot(dkp), boot(dpk), boot(dsb)
    bf = ("START's band brings back the ordinal" if Ak(sis) >= .60 and dkp(sis) >= .30 and ck[0] > 0
          else "without END's band write the late lookup stays at the early position" if Ap(sis) >= .60 and dpk(sis) >= .30 and cp[0] > 0 else "mixed")
    be = ("END's band takes START off the ordinal" if Bk(sis) <= .40 and dsb(sis) >= .25 and cs[0] > 0
          else "START keeps the ordinal despite END's band" if Bk(sis) >= .55 and dsb(sis) <= .15 else "partial")
    ok_w = ok_all and pc >= .55
    out["WRITER"] = dict(A_k=Ak(sis), A_p=Ap(sis), A_k_minus_p=[dkp(sis), ck], B_k=Bk(sis), S_k=Sk(sis), S_minus_B=[dsb(sis), cs],
                         BANDF=(bf if ok_w else nv if not ok_all else "uninterpretable (PC failed)"),
                         BANDE=(be if ok_w else nv if not ok_all else "uninterpretable (PC failed)"))
    out["WRITER"]["SWAP"] = ("the band carries the frame's late anchor rule" if ok_w and bf.startswith("START's band") and be.startswith("END's band")
                             else "not shown")
    obs = {(ns, p): np.bincount([by[ns][s]["land"][f"E_EARLY_p{p}"] for s in sis], minlength=9)[1:] / len(sis) for ns in NSS for p in PS}
    cal = float(np.mean([abs(laws[f"Ns{ns}_p{p}"]["pos"][x] - obs[(ns, p)][x]) <= .15 for ns in NSS for p in PS for x in range(8)]))
    def ll(kind):
        v = [np.log(max(laws[f"Ns{ns}_p{p}"][kind][by[ns][s]["land"][f"E_EARLY_p{p}"] - 1], 1e-9)) for ns in NSS for p in PS for s in sis
             if 1 <= by[ns][s]["land"][f"E_EARLY_p{p}"] <= 8]
        return float(np.mean(v))
    out["PRED"] = dict(calibration=cal, verdict=("the goal-blind position law predicts fresh scenes" if cal >= .70 else "not shown"),
                       ll={kd: ll(kd) for kd in ("pos", "cond", "mix")})
    # descriptive
    out["table"] = {f"Ns{ns}_p{p}": {arm: np.bincount([by[ns][s]["land"][f"{arm}_p{p}"] for s in sis], minlength=9).tolist()
                                    for arm in ("E_EARLY", "S_EARLY", "BANDF", "BANDE")} for ns in NSS for p in PS}
    m = lambda ns, p, ss: A("E_EARLY", ns, p, p + 1, ss) - A("E_EARLY", ns, p, p - 1, ss)
    zd = lambda L, row: (lambda ns, p, ss: float(np.mean([by[ns][s]["qk"][f"E_EARLY_p{p}"][L][row][p] - by[ns][s]["qk"][f"E_EARLY_p{p}"][L][row][p - 2]
                                                            for s in ss])))
    within = lambda f: (lambda ss: float(np.mean([np.polyfit(NSS, [f(ns, p, ss) for ns in NSS], 1)[0] for p in PS])))
    across = lambda f: (lambda ss: float(np.mean([np.polyfit([-p for p in PS], [f(ns, p, ss) for p in PS], 1)[0] for ns in NSS])))
    out["slopes"] = {"answer_margin": dict(within=[within(m)(sis), boot(within(m))], across=[across(m)(sis), boot(across(m))])}
    for L in late:
        for row, nm in ((5, "B4"), (9, "answer")):
            f = zd(L, row)
            out["slopes"][f"z{L}_{nm}"] = dict(within=[within(f)(sis), boot(within(f))], across=[across(f)(sis), boot(across(f))])
    am = {}
    for arm in ("E_EARLY", "S_EARLY", "BANDF", "BANDE"):
        for L in late:
            a_ = [int(np.argmax(by[ns][s]["qk"][f"{arm}_p{p}"][L][9])) + 1 for ns in NSS for p in PW for s in sis]
            pp = [p for ns in NSS for p in PW for s in sis]
            am[f"{arm}_L{L}"] = dict(is_k=float(np.mean(np.array(a_) == k)), is_p=float(np.mean(np.array(a_) == np.array(pp))))
    out["late_argmax"] = am
    out["paired"] = dict(BANDE_eq_E=float(np.mean([by[ns][s]["land"][f"BANDE_p{p}"] == by[ns][s]["land"][f"E_EARLY_p{p}"] for ns in NSS for p in PW for s in sis])),
                         BANDF_eq_S=float(np.mean([by[ns][s]["land"][f"BANDF_p{p}"] == by[ns][s]["land"][f"S_EARLY_p{p}"] for ns in NSS for p in PW for s in sis])))
    out["natural_END"] = {str(ns): np.bincount([by[ns][s]["land"]["END"] for s in sis], minlength=9).tolist() for ns in NSS}
    fn = f"{V}/results/r3/s846_analysis.json" if tag == "s846_27b" else f"{V}/results/r3/s846_analysis_{tag}.json"
    json.dump(out, open(fn, "w"), indent=1)
    print(json.dumps(out["I0"])[:800]); print("PC", out["PC"])
    for c, v in out["table"].items():
        print(c, {a: x[1:] for a, x in v.items()})
    for k_ in ("NOSTOP", "PDEP", "WRITER", "PRED", "late_argmax", "paired", "slopes"):
        print(k_, json.dumps(out[k_]))


if __name__ == "__main__":
    main(*sys.argv[1:])
