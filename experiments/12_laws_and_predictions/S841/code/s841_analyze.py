"""S841 analysis (rules fixed in knowledge/DECISIONS.md before any S841 data).  7 pictures shown, END(4); stated count N_s in {6, 7, 8}; the same
30 scenes for every N_s (paired scene bootstrap, 4,000 draws, seed 0).
I0 (per N_s): t1 >= .95; QNOOP_E = END in >= .90 of scenes with the 95th percentile of the largest menu-logit change <= .5; START(4) -> 4 >= .90;
every donor START(p) -> p >= .90 pooled (>= .80 each); image keys identical (0.0).  All three N_s must pass for SIGN / TIE.
P_Ns(p, x) = share of E_EARLY_p answering image x under stated count N_s.
  SIGN  at the copy p = 4: side(N_s) = P(5) - P(3); D = side(8) - side(6): D >= .30 with interval above 0 -> 'the step's side follows the stated
        count'; D <= .10 -> 'the side does not follow the stated count'; otherwise 'partial'.
  TIE   pooled over the stated ties (N_s 6, p 3), (7, 4), (8, 5): P(answer in {p - 1, p + 1}) - P(answer = p) >= .15 with interval above 0 ->
        'the step fires at a stated tie'; otherwise 'not shown'.
Descriptive: the same side contrast at p = 3 (P(4) - P(2)) and p = 5 (P(6) - P(4)); at p = 2 with N_s 8 (stated target 5) the shares of 3, 4, 5
(step size against distance); natural END(4) per N_s.
Usage: python exp/s841_analyze.py [TAG]  ->  results/r3/s841_analysis.json"""
from __future__ import annotations

import json, sys
import numpy as np

V = "/home/storage2/ai26269345/vlm"
B = 4000


def main(tag="s841_27b"):
    d = json.load(open(f"{V}/results/r3/s841_{tag}_anchor.json")); R = d["rows"]
    rng = np.random.default_rng(0)
    by = {ns: {r["si"]: r for r in R if r["Ns"] == ns} for ns in (6, 7, 8)}
    sis = sorted(set.intersection(*[set(x) for x in by.values()]))
    P = lambda ns, p, x, ss=None: float(np.mean([by[ns][s]["land"][f"E_EARLY_p{p}"] == x for s in (ss or sis)]))
    out = dict(tag=tag, t1=d["t1_menu"], key_rel_diff_max=d["key_rel_diff_max"], n=len(sis), I0={})
    ok_all = True
    for ns in (6, 7, 8):
        rows = [by[ns][s] for s in sis]
        ag = float(np.mean([r["land"]["QNOOP_E"] == r["land"]["END"] for r in rows]))
        p95 = float(np.percentile([max(abs(x - y) for x, y in zip(r["mlog"]["QNOOP_E"], r["mlog"]["END"])) for r in rows], 95))
        st = float(np.mean([r["land"]["START"] == 4 for r in rows]))
        don = {p: float(np.mean([r["land"][f"START_p{p}"] == p for r in rows])) for p in (2, 3, 4, 5, 6)}
        ok = bool(d["t1_menu"] >= .95 and ag >= .90 and p95 <= .5 and st >= .90 and np.mean(list(don.values())) >= .90 and min(don.values()) >= .80
                  and d["key_rel_diff_max"] == 0.0)
        ok_all &= ok
        out["I0"][str(ns)] = dict(qnoop=ag, p95=p95, start_acc=st, donors=don, ok=ok)
    out["I0"]["all"] = bool(ok_all)
    out["natural_END"] = {str(ns): np.bincount([by[ns][s]["land"]["END"] for s in sis], minlength=8).tolist() for ns in (6, 7, 8)}
    out["E_EARLY"] = {str(ns): {str(p): np.bincount([by[ns][s]["land"][f"E_EARLY_p{p}"] for s in sis], minlength=8).tolist() for p in (2, 3, 4, 5, 6)}
                      for ns in (6, 7, 8)}
    def boot(f):
        v = []
        for _ in range(B):
            ss = [sis[i] for i in rng.integers(0, len(sis), len(sis))]; v.append(f(ss))
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    side = lambda ns, p, up, dn, ss=None: P(ns, p, up, ss) - P(ns, p, dn, ss)
    D = side(8, 4, 5, 3) - side(6, 4, 5, 3); ci = boot(lambda ss: side(8, 4, 5, 3, ss) - side(6, 4, 5, 3, ss))
    out["SIGN"] = dict(side={str(ns): side(ns, 4, 5, 3) for ns in (6, 7, 8)}, D=D, ci=ci,
                       verdict=("no valid run" if not ok_all else "the step's side follows the stated count" if D >= .30 and ci[0] > 0
                                else "the side does not follow the stated count" if D <= .10 else "partial"))
    out["SIGN_desc"] = dict(p3={str(ns): side(ns, 3, 4, 2) for ns in (6, 7, 8)}, p5={str(ns): side(ns, 5, 6, 4) for ns in (6, 7, 8)})
    ties = [(6, 3), (7, 4), (8, 5)]
    def tie(ss=None):
        mv = np.mean([P(ns, p, p - 1, ss) + P(ns, p, p + 1, ss) for ns, p in ties]); st_ = np.mean([P(ns, p, p, ss) for ns, p in ties])
        return float(mv - st_)
    tv = tie(); tci = boot(lambda ss: tie(ss))
    out["TIE"] = dict(diff=tv, ci=tci, per_tie={f"{ns}_{p}": dict(moved=P(ns, p, p - 1) + P(ns, p, p + 1), stay=P(ns, p, p)) for ns, p in ties},
                      verdict=("no valid run" if not ok_all else "the step fires at a stated tie" if tv >= .15 and tci[0] > 0 else "not shown"))
    out["MAG_desc"] = {str(x): P(8, 2, x) for x in (2, 3, 4, 5, 6)}
    fn = f"{V}/results/r3/s841_analysis.json" if tag == "s841_27b" else f"{V}/results/r3/s841_analysis_{tag}.json"
    json.dump(out, open(fn, "w"), indent=1)
    print(json.dumps(out["I0"])[:900]); print("natural END", out["natural_END"]); print("E_EARLY", json.dumps(out["E_EARLY"]))
    print("SIGN", json.dumps(out["SIGN"])); print("SIGN_desc", json.dumps(out["SIGN_desc"])); print("TIE", json.dumps(out["TIE"])); print("MAG", out["MAG_desc"])


if __name__ == "__main__":
    main(*sys.argv[1:])
