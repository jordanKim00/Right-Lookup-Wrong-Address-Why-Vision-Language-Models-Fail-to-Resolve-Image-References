"""S851 analysis (rules fixed in knowledge/DECISIONS.md before any S851 data; user 10-17 late evening).  Minimal path test after S850.
8 pictures, k = 4, N_s 6-8, early lookup pinned to p 3-5 (3 x 3 grid); stay = P(answer = p); scene bootstrap 4,000 (seed 0), 30 scenes.
I0 (per N_s; all must pass): t1 >= .95; QNOOP_E = END in >= .90 of scenes, 95th percentile of the largest menu-logit change <= .5; START(4) -> 4
  >= .90; donors START(p) -> p >= .90 pooled (>= .80 each); image keys identical (0.0).
POS  G = stay(FULLB) - stay(E_EARLY) over the 9 cells >= .15 with interval above 0 (the boundary correction exists; S850: +.28).
SUFF s = [stay(PLUSQ) - stay(PLUS16)] / G (pooled as sums): s >= .70 with lower bound > .50 -> 'the late queries carry the boundary correction
  into the steered run'; s <= .30 -> 'late queries are not sufficient in the steered run'; otherwise 'partial'.
NEC  n = [stay(FULLB) - stay(FULLBQE)] / G: n >= .70 with lower bound > .50 -> 'the boundary correction needs the late queries'; n <= .30 ->
  'the boundary correction also travels outside the late queries'; otherwise 'partial'.
Reading (registered): SUFF and NEC both first verdicts -> 'the boundary correction travels through the late queries 51-63, and the steered run
  fails because its late queries are not corrected enough, not because a later stage blocks a corrected query'; SUFF second -> 'a state outside
  the late queries of the steered run blocks recovery (late reading, values or readout)'; NEC second -> 'the boundary correction has routes
  outside the late queries'.  This does not test whether the target-match signal is used in natural runs.
Descriptive: the same ratios on the menu-logit margin of p; late margins of p (layers 51-63) per arm; match vs non-match cells.
Usage: python exp/s851_analyze.py [TAG]  ->  results/r3/s851_analysis.json"""
from __future__ import annotations

import json, sys
import numpy as np

V = "/home/storage2/ai26269345/vlm"
B = 4000
PS = (3, 4, 5); NSS = (6, 7, 8)
ARMS = ("E_EARLY", "S_EARLY", "FULLB", "PLUS16", "PLUSQ", "FULLBQE")


def main(tag="s851_27b"):
    d = json.load(open(f"{V}/results/r3/s851_{tag}_path.json")); R = d["rows"]
    rng = np.random.default_rng(0)
    by = {ns: {r["si"]: r for r in R if r["Ns"] == ns} for ns in NSS}
    sis = sorted(set.intersection(*[set(x) for x in by.values()]))
    out = dict(tag=tag, t1=d["t1_menu"], key_rel_diff_max=d["key_rel_diff_max"], n=len(sis), I0={})
    ok_all = True
    for ns in NSS:
        rows = [by[ns][s] for s in sis]
        ag = float(np.mean([r["land"]["QNOOP_E"] == r["land"]["END"] for r in rows]))
        p95 = float(np.percentile([max(abs(x - y) for x, y in zip(r["mlog"]["QNOOP_E"], r["mlog"]["END"])) for r in rows], 95))
        st = float(np.mean([r["land"]["START"] == 4 for r in rows]))
        don = {p: float(np.mean([r["land"][f"START_p{p}"] == p for r in rows])) for p in PS}
        ok = bool(d["t1_menu"] >= .95 and ag >= .90 and p95 <= .5 and st >= .90 and np.mean(list(don.values())) >= .90 and min(don.values()) >= .80
                  and d["key_rel_diff_max"] == 0.0)
        ok_all &= ok
        out["I0"][str(ns)] = dict(qnoop=ag, p95=p95, start_acc=st, donors=don, ok=ok)
    out["I0"]["all"] = bool(ok_all)
    cells = [(ns, p) for ns in NSS for p in PS]
    def stay(arm, ss, cs=cells):
        return float(np.mean([by[ns][s]["land"][f"{arm}_p{p}"] == p for ns, p in cs for s in ss]))
    def lmar(arm, ss, cs=cells):
        v = []
        for ns, p in cs:
            for s in ss:
                r = by[ns][s]; ix = r["words"].index(r["labs"][p - 1]); lg = np.array(r["mlog"][f"{arm}_p{p}"])
                v.append(float(lg[ix] - max(lg[i] for i in range(len(lg)) if i != ix)))
        return float(np.mean(v))
    def boot(f):
        v = [f([sis[i] for i in rng.integers(0, len(sis), len(sis))]) for _ in range(B)]
        return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    G = lambda ss, m=stay: m("FULLB", ss) - m("E_EARLY", ss)
    Sf = lambda ss, m=stay: (m("PLUSQ", ss) - m("PLUS16", ss)) / G(ss, m) if G(ss, m) != 0 else float("nan")
    Nf = lambda ss, m=stay: (m("FULLB", ss) - m("FULLBQE", ss)) / G(ss, m) if G(ss, m) != 0 else float("nan")
    gv, gci = G(sis), boot(G); sv, sci = Sf(sis), boot(Sf); nv_, nci = Nf(sis), boot(Nf)
    out["stay"] = {a: stay(a, sis) for a in ARMS}
    out["POS"] = dict(G=gv, ci=gci, verdict=("the boundary correction exists" if gv >= .15 and gci[0] > 0 else "not shown"))
    nvr = "no valid run"
    out["SUFF"] = dict(s=sv, ci=sci, verdict=(nvr if not ok_all else "uninterpretable (no correction)" if out["POS"]["verdict"] == "not shown"
                                              else "the late queries carry the boundary correction into the steered run" if sv >= .70 and sci[0] > .50
                                              else "late queries are not sufficient in the steered run" if sv <= .30 else "partial"))
    out["NEC"] = dict(n=nv_, ci=nci, verdict=(nvr if not ok_all else "uninterpretable (no correction)" if out["POS"]["verdict"] == "not shown"
                                              else "the boundary correction needs the late queries" if nv_ >= .70 and nci[0] > .50
                                              else "the boundary correction also travels outside the late queries" if nv_ <= .30 else "partial"))
    s1, n1 = out["SUFF"]["verdict"], out["NEC"]["verdict"]
    if s1.startswith("the late queries carry") and n1.startswith("the boundary correction needs"):
        rd = "the boundary correction travels through the late queries 51-63, and the steered run fails because its late queries are not corrected enough"
    elif s1.startswith("late queries are not sufficient"):
        rd = "a state outside the late queries of the steered run blocks recovery (late reading, values or readout)"
    elif n1.startswith("the boundary correction also travels"):
        rd = "the boundary correction has routes outside the late queries"
    else:
        rd = "partial"
    out["READING"] = rd
    out["logit"] = dict(margin={a: lmar(a, sis) for a in ARMS}, G=G(sis, lmar), s=Sf(sis, lmar), n=Nf(sis, lmar))
    def hold(arm, s, ns, p, L):
        v = []
        for row in (5, 9):
            z = by[ns][s]["qk"][f"{arm}_p{p}"][L][row]
            v.append(z[p - 1] - max(z[i] for i in range(len(z)) if i != p - 1))
        return float(np.mean(v))
    out["late_margin"] = {a: {L: float(np.mean([hold(a, s, ns, p, L) for s in sis for ns, p in cells])) for L in ("51", "55", "59", "63")} for a in ARMS}
    match = [(p + 3, p) for p in PS]
    out["match_stay"] = {a: stay(a, sis, match) for a in ARMS}
    fn = f"{V}/results/r3/s851_analysis.json" if tag == "s851_27b" else f"{V}/results/r3/s851_analysis_{tag}.json"
    json.dump(out, open(fn, "w"), indent=1)
    print(json.dumps(out["I0"])[:600]); print("stay", json.dumps(out["stay"])); print("POS", out["POS"]); print("SUFF", out["SUFF"]); print("NEC", out["NEC"])
    print("READING", rd); print("logit", json.dumps(out["logit"])); print("late_margin", json.dumps(out["late_margin"])); print("match_stay", json.dumps(out["match_stay"]))


if __name__ == "__main__":
    main(*sys.argv[1:])
