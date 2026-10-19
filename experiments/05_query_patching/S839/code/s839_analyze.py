"""S839 analysis (rules fixed in knowledge/DECISIONS.md before any S839 data).  Identity cases N7 END(4), N9 END(5); P_j = share answering the target
(= k); scene bootstrap 4,000 draws (seed 0); 'pooled' = both cases' scenes together.
I0 (per case): t1 >= .95; QNOOP_E = END and QNOOP_S = START in >= .90 of scenes, 95th percentile of the largest menu-logit change <= .5; START(k) -> k
>= .90; image keys identical (0.0); positive controls (S829 replications): P_j(P_FULL) - P_j(END) >= .15 (gap 1) and P_j(START) - P_j(R_ALL) >= .15
(gap 2).  A case failing a gap's positive control is not used for that gap's verdicts.
Late picture mass m = mean over rows B1, B4, answer and the late look layers (> .75 depth) of the matcher heads' attention on image tokens.
GAP 1 (END arms; d(arm) = P_j(arm) - P_j(END), dm(arm) = m(arm) - m(END)):
  MENU   d(MENU_FULL) >= .5 d(P_FULL) with interval of d(MENU_FULL) above 0, and dm(MENU_FULL) <= .5 dm(P_FULL) (the late picture mass moves at
         least half as far toward START; dm(P_FULL) < 0 expected) -> 'the reading mode is the picture-vs-menu-word preference'; else 'not shown'.
  NEC    d(PNM_FULL) <= .5 d(P_FULL) with interval of d(P_FULL) - d(PNM_FULL) above 0 -> 'the menu component is needed for the repair'.
  TEXTALL, NORM  the MENU rule with TEXTALL_FULL / NORM_FULL (descriptive verdicts).
GAP 2 (START arms; drop(arm) = P_j(START) - P_j(arm)):
  LATE63    drop(R_ALL63) - drop(R_ALL) >= .15, interval above 0 -> 'the last full-attention layers protect START'.
  MENUREAD  drop(R_ALL_KO) - drop(R_ALL) >= .15 (interval above 0) and drop(KO) <= .15 -> 'START is protected by its late menu read'; if drop(KO)
            > .15 -> 'the late menu read is needed by START itself (not specific)'.
  LATEONLY  drop(R_LATE63) >= .5 drop(R_ALL63) -> 'the late layers carry most of the break' (descriptive).
Usage: python exp/s839_analyze.py [TAG]  ->  results/r3/s839_analysis.json"""
from __future__ import annotations

import json, sys
import numpy as np

V = "/home/storage2/ai26269345/vlm"
B = 4000


def main(tag="s839_27b"):
    d = json.load(open(f"{V}/results/r3/s839_{tag}_mode2.json")); R = d["rows"]; look = d["look"]; nl = d["n_layer"]
    late = [str(l) for l in look if l / (nl - 1) > .75]
    rng = np.random.default_rng(0)
    def m(r, a):
        x = r.get("mass", {}).get(a, {})
        v = [np.mean([x[L][i] for i in (2, 5, 9)]) for L in late if L in x]
        return float(np.mean(v)) if v else np.nan
    Pj = lambda rows, a: float(np.mean([r["land"][a] == r["k"] for r in rows]))
    M = lambda rows, a: float(np.nanmean([m(r, a) for r in rows]))
    def boot(rows, f):
        v = []
        for _ in range(B):
            ix = rng.integers(0, len(rows), len(rows)); v.append(f([rows[i] for i in ix]))
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    out = dict(tag=tag, t1=d["t1_menu"], key_rel_diff_max=d["key_rel_diff_max"], cases={})
    g1, g2 = [], []
    for N, k, j in [(7, 4, 4), (9, 5, 5)]:
        rows = [r for r in R if r["N"] == N]
        if not rows:
            continue
        cname = f"N{N}k{k}"
        ag = lambda a1, a2: float(np.mean([r["land"][a1] == r["land"][a2] for r in rows]))
        p95 = lambda a1, a2: float(np.percentile([max(abs(x - y) for x, y in zip(r["mlog"][a1], r["mlog"][a2])) for r in rows], 95))
        i0 = dict(qnoop_e=ag("QNOOP_E", "END"), qnoop_s=ag("QNOOP_S", "START"), p95=max(p95("QNOOP_E", "END"), p95("QNOOP_S", "START")),
                  start_acc=Pj(rows, "START"), pos1=Pj(rows, "P_FULL") - Pj(rows, "END"), pos2=Pj(rows, "START") - Pj(rows, "R_ALL"))
        base = bool(d["t1_menu"] >= .95 and i0["qnoop_e"] >= .90 and i0["qnoop_s"] >= .90 and i0["p95"] <= .5 and i0["start_acc"] >= .90
                    and d["key_rel_diff_max"] == 0.0)
        i0["ok_gap1"] = bool(base and i0["pos1"] >= .15); i0["ok_gap2"] = bool(base and i0["pos2"] >= .15)
        arms = sorted(rows[0]["land"])
        c = dict(n=len(rows), I0=i0, P_j={a: Pj(rows, a) for a in arms}, mass={a: M(rows, a) for a in arms})
        out["cases"][cname] = c
        if i0["ok_gap1"]:
            g1 += rows
        if i0["ok_gap2"]:
            g2 += rows
    def gap1(rows):
        if not rows:
            return dict(verdict="no valid case")
        d_ = lambda a, q=None: Pj(q or rows, a) - Pj(q or rows, "END")
        dm = lambda a: M(rows, a) - M(rows, "END")
        res = dict(d={a: d_(a) for a in ("P_FULL", "MENU_FULL", "TEXTALL_FULL", "NORM_FULL", "PNM_FULL")},
                   dm={a: dm(a) for a in ("P_FULL", "MENU_FULL", "TEXTALL_FULL", "NORM_FULL", "PNM_FULL")})
        res["ci"] = {a: boot(rows, lambda q, a=a: Pj(q, a) - Pj(q, "END")) for a in res["d"]}
        def rule(a):
            return ("holds" if res["d"][a] >= .5 * res["d"]["P_FULL"] and res["ci"][a][0] > 0 and res["dm"][a] <= .5 * res["dm"]["P_FULL"]
                    else "not shown")
        res["MENU"] = "the reading mode is the picture-vs-menu-word preference" if rule("MENU_FULL") == "holds" else "not shown"
        res["TEXTALL"] = rule("TEXTALL_FULL"); res["NORM"] = rule("NORM_FULL")
        diff_ci = boot(rows, lambda q: (Pj(q, "P_FULL") - Pj(q, "END")) - (Pj(q, "PNM_FULL") - Pj(q, "END")))
        res["NEC"] = ("the menu component is needed for the repair" if res["d"]["PNM_FULL"] <= .5 * res["d"]["P_FULL"] and diff_ci[0] > 0
                      else "not shown"); res["NEC_ci"] = diff_ci
        return res
    def gap2(rows):
        if not rows:
            return dict(verdict="no valid case")
        drop = lambda a, q=None: Pj(q or rows, "START") - Pj(q or rows, a)
        res = dict(drop={a: drop(a) for a in ("R_ALL", "R_ALL63", "R_LATE63", "KO", "R_ALL_KO")})
        c63 = boot(rows, lambda q: drop("R_ALL63", q) - drop("R_ALL", q)); cko = boot(rows, lambda q: drop("R_ALL_KO", q) - drop("R_ALL", q))
        res["LATE63"] = ("the last full-attention layers protect START" if res["drop"]["R_ALL63"] - res["drop"]["R_ALL"] >= .15 and c63[0] > 0
                         else "not shown"); res["LATE63_ci"] = c63
        if res["drop"]["R_ALL_KO"] - res["drop"]["R_ALL"] >= .15 and cko[0] > 0:
            res["MENUREAD"] = ("START is protected by its late menu read" if res["drop"]["KO"] <= .15
                               else "the late menu read is needed by START itself (not specific)")
        else:
            res["MENUREAD"] = "not shown"
        res["MENUREAD_ci"] = cko
        res["LATEONLY"] = "the late layers carry most of the break" if res["drop"]["R_LATE63"] >= .5 * res["drop"]["R_ALL63"] else "not shown"
        return res
    out["GAP1"] = gap1(g1); out["GAP2"] = gap2(g2)
    fn = f"{V}/results/r3/s839_analysis.json" if tag == "s839_27b" else f"{V}/results/r3/s839_analysis_{tag}.json"
    json.dump(out, open(fn, "w"), indent=1)
    for cname, c in out["cases"].items():
        print(cname, c["I0"], "\n   P_j", {a: round(x, 3) for a, x in c["P_j"].items()}, "\n   mass", {a: round(x, 3) for a, x in c["mass"].items()})
    print("GAP1", json.dumps(out["GAP1"])[:900]); print("GAP2", json.dumps(out["GAP2"])[:700])


if __name__ == "__main__":
    main(*sys.argv[1:])
