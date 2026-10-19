"""S844 analysis (rules fixed in knowledge/DECISIONS.md before any S844 data).  8 pictures, END(4), stated count N_s in {6, 7, 8, 9} (stated target
t = N_s - 3), early lookup pinned to p in {3, 4, 5, 6}; d = t - p ('left' d < 0, 'same' d = 0, 'right' d > 0).  Paired scene bootstrap (4,000, seed 0).
I0: for every N_s: t1 >= .95; QNOOP_E = END in >= .90 of scenes, 95th percentile of the largest menu-logit change <= .5; START(4) -> 4 >= .90;
donors START(p) -> p >= .90 pooled (>= .80 each); image keys identical (0.0).
Answer shares per condition: P(p - 1), P(p), P(p + 1); margin m = P(p + 1) - P(p - 1).
  BIAS     pooled over 'same' and 'right' conditions: P(p - 1) - P(p + 1) >= .20, interval above 0 -> 'a predecessor bias the goal does not override'.
  COND     OLS slope of m on d over the 16 conditions > 0 with interval above 0 -> 'partial goal conditioning'; OVERRIDE (descriptive): mean m over
           'right' conditions > +.15 with interval above 0 -> 'the goal overrides the bias when it lies to the right'.
  STOP     pooled over the 4 'same' conditions: P(p - 1) + P(p + 1) - P(p) >= .15, interval above 0 -> 'no stopping at the goal'.
  INTERNAL layer-51 answer-row margin z[p + 1] - z[p - 1] (matcher-head mean block scores of E_EARLY_p): mean over 'same' + 'right' < 0 with interval
           below 0 -> 'internal predecessor bias'; slope on d > 0 with interval above 0 -> 'internal goal conditioning'.
  PRED     frozen laws (research/ces/outputs/s844_frozen_laws.json, fitted on S841): (a) M_mix calibration = share of (condition x picture) cells with
           |predicted - observed| <= .15, 'calibrated' if >= .70; (b) mean log-likelihood per answer of M_mix minus M_pred and minus M_goal, both > 0
           with intervals above 0 -> 'the biased, goal-conditioned law predicts the new conditions better than both pure laws'.
Usage: python exp/s844_analyze.py [TAG]  ->  results/r3/s844_analysis.json"""
from __future__ import annotations

import json, sys
import numpy as np

V = "/home/storage2/ai26269345/vlm"
B = 4000
PS = (3, 4, 5, 6); NSS = (6, 7, 8, 9)


def main(tag="s844_27b"):
    d = json.load(open(f"{V}/results/r3/s844_{tag}_anchor.json")); R = d["rows"]
    laws = json.load(open(f"{V}/research/ces/outputs/s844_frozen_laws.json"))["predictions_N8"]
    rng = np.random.default_rng(0)
    by = {ns: {r["si"]: r for r in R if r["Ns"] == ns} for ns in NSS}
    nl = d["n_layer"]; L51 = str(min(l for l in d["look"] if l / (nl - 1) > .75))           # first late look layer (51 in the 27B)
    sis = sorted(set.intersection(*[set(x) for x in by.values()]))
    conds = [(ns, p) for ns in NSS for p in PS]
    dof = lambda ns, p: (ns - 3) - p
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
    P = lambda ns, p, x, ss: float(np.mean([by[ns][s]["land"][f"E_EARLY_p{p}"] == x for s in ss]))
    def zm(ns, p, ss):
        return float(np.mean([by[ns][s]["qk"][f"E_EARLY_p{p}"][L51][9][p] - by[ns][s]["qk"][f"E_EARLY_p{p}"][L51][9][p - 2] for s in ss]))
    def boot(f):
        v = []
        for _ in range(B):
            v.append(f([sis[i] for i in rng.integers(0, len(sis), len(sis))]))
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    out["table"] = {f"Ns{ns}_p{p}": dict(d=dof(ns, p), down=P(ns, p, p - 1, sis), stay=P(ns, p, p, sis), up=P(ns, p, p + 1, sis),
                                         z51_margin=zm(ns, p, sis), counts=np.bincount([by[ns][s]["land"][f"E_EARLY_p{p}"] for s in sis], minlength=9).tolist())
                    for ns, p in conds}
    sr = [(ns, p) for ns, p in conds if dof(ns, p) >= 0]; same = [(ns, p) for ns, p in conds if dof(ns, p) == 0]; right = [(ns, p) for ns, p in conds if dof(ns, p) > 0]
    bias = lambda ss: float(np.mean([P(ns, p, p - 1, ss) - P(ns, p, p + 1, ss) for ns, p in sr]))
    slope = lambda ss, f: float(np.polyfit([dof(ns, p) for ns, p in conds], [f(ns, p, ss) for ns, p in conds], 1)[0])
    mfun = lambda ns, p, ss: P(ns, p, p + 1, ss) - P(ns, p, p - 1, ss)
    stop = lambda ss: float(np.mean([P(ns, p, p - 1, ss) + P(ns, p, p + 1, ss) - P(ns, p, p, ss) for ns, p in same]))
    over = lambda ss: float(np.mean([mfun(ns, p, ss) for ns, p in right]))
    ib = lambda ss: float(np.mean([zm(ns, p, ss) for ns, p in sr]))
    res = dict(bias=(bias(sis), boot(bias)), slope=(slope(sis, mfun), boot(lambda ss: slope(ss, mfun))), stop=(stop(sis), boot(stop)),
               override=(over(sis), boot(over)), ibias=(ib(sis), boot(ib)), islope=(slope(sis, zm), boot(lambda ss: slope(ss, zm))))
    out["stats"] = {k: dict(value=v[0], ci=v[1]) for k, v in res.items()}
    nv = "no valid run"
    out["BIAS"] = nv if not ok_all else ("a predecessor bias the goal does not override" if res["bias"][0] >= .20 and res["bias"][1][0] > 0 else "not shown")
    out["COND"] = nv if not ok_all else ("partial goal conditioning" if res["slope"][0] > 0 and res["slope"][1][0] > 0 else "not shown")
    out["OVERRIDE"] = "the goal overrides the bias when it lies to the right" if res["override"][0] > .15 and res["override"][1][0] > 0 else "not shown"
    out["STOP"] = nv if not ok_all else ("no stopping at the goal" if res["stop"][0] >= .15 and res["stop"][1][0] > 0 else "not shown")
    out["INTERNAL"] = dict(bias=("internal predecessor bias" if res["ibias"][1][1] < 0 else "not shown"),
                           conditioning=("internal goal conditioning" if res["islope"][0] > 0 and res["islope"][1][0] > 0 else "not shown"))
    # frozen laws
    obs = {c: np.bincount([by[c[0]][s]["land"][f"E_EARLY_p{c[1]}"] for s in sis], minlength=9)[1:] / len(sis) for c in conds}
    cal = [abs(laws[f"Ns{ns}_p{p}"]["mix"][x] - obs[(ns, p)][x]) <= .15 for ns, p in conds for x in range(8)]
    def ll(kind, ss):
        v = []
        for ns, p in conds:
            pr = laws[f"Ns{ns}_p{p}"][kind]
            for s in ss:
                a = by[ns][s]["land"][f"E_EARLY_p{p}"]
                if 1 <= a <= 8:
                    v.append(np.log(max(pr[a - 1], 1e-9)))
        return float(np.mean(v))
    gp = lambda ss: ll("mix", ss) - ll("pred", ss); gg = lambda ss: ll("mix", ss) - ll("goal", ss)
    out["PRED"] = dict(calibration=float(np.mean(cal)), calibrated=bool(np.mean(cal) >= .70), ll={k: ll(k, sis) for k in ("mix", "pred", "goal")},
                       gain_vs_pred=(gp(sis), boot(gp)), gain_vs_goal=(gg(sis), boot(gg)))
    out["PRED"]["verdict"] = ("the biased, goal-conditioned law predicts the new conditions better than both pure laws"
                              if out["PRED"]["gain_vs_pred"][1][0] > 0 and out["PRED"]["gain_vs_goal"][1][0] > 0 and gp(sis) > 0 and gg(sis) > 0 else "not shown")
    fn = f"{V}/results/r3/s844_analysis.json" if tag == "s844_27b" else f"{V}/results/r3/s844_analysis_{tag}.json"
    json.dump(out, open(fn, "w"), indent=1)
    print(json.dumps(out["I0"])[:700])
    for c, v in out["table"].items():
        print(c, "d", v["d"], "down/stay/up %.2f/%.2f/%.2f" % (v["down"], v["stay"], v["up"]), "z51 %+.2f" % v["z51_margin"], v["counts"])
    for k in ("BIAS", "COND", "OVERRIDE", "STOP", "INTERNAL"):
        print(k, out[k])
    print("stats", json.dumps(out["stats"])); print("PRED", json.dumps(out["PRED"]))


if __name__ == "__main__":
    main(*sys.argv[1:])
