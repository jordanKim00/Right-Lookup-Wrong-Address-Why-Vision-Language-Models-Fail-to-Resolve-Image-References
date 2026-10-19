"""S849-U analysis (rules fixed in knowledge/DECISIONS.md before any S849 data).  Test A part 2: is the goal-attainment signal found by S848 used to
stop the update?  8 pictures, k = 4, N_s 6-9, early lookup pinned to p 3-6 (match cells: p = N_s - 3); the answer row's residual steered at the
input of layer 46 or 50 along the frozen match direction m (+ALPHA: PLUS, -ALPHA: MINUS) or a random direction r of the same norm in the same
50-PC subspace (RAND).  stay = P(answer = p).  Paired scene bootstrap (4,000, seed 0).
I0 (per N_s; all must pass): t1 >= .95; QNOOP_E = END in >= .90 of scenes, 95th percentile of the largest menu-logit change <= .5; START(4) -> 4
>= .90; donors START(p) -> p >= .90 pooled (>= .80 each); image keys identical (0.0).
LIVE (per site, needed for USE at that site): the steering persists - median over runs of (projection of the answer-row residual on m in PLUS minus
in E_EARLY) / (ALPHA * sigma) >= .50 at the next kept layer (site 46: layer 48; site 50: layer 52).
USE (per site), over the 12 non-match cells: dS = stay(PLUS) - stay(E_EARLY), dR = stay(RAND) - stay(E_EARLY), D = dS - dR.
  dS >= .15 with interval above 0 and D >= .10 with interval above 0 -> 'the goal-attainment signal stops the update when amplified';
  |D| <= .05 with the interval of D inside [-.10, .10] -> 'represented but not used to stop the update'; otherwise 'partial'.
Descriptive: the same contrasts in the four match cells for PLUS, MINUS and RAND; MINUS in non-match cells; the share of answers changed by each arm;
the late hold margin z[p] - max(z[p - 1], z[p + 1]) (layers 51 / 55, boundary B4 and answer rows) per arm; projections at 48 / 50 / 52.
Usage: python exp/s849u_analyze.py [TAG]  ->  results/r3/s849u_analysis.json"""
from __future__ import annotations

import json, sys
import numpy as np

V = "/home/storage2/ai26269345/vlm"
B = 4000
PS = (3, 4, 5, 6); NSS = (6, 7, 8, 9)


def main(tag="s849u_27b"):
    d = json.load(open(f"{V}/results/r3/s849_{tag}_use.json")); R = d["rows"]
    Z = np.load(f"{V}/results/r3/s849_{tag}_use.npz")
    dz = np.load(f"{V}/research/ces/outputs/s849_match_dirs.npz")
    rng = np.random.default_rng(0)
    by = {ns: {r["si"]: r for r in R if r["Ns"] == ns} for ns in NSS}
    sis = sorted(set.intersection(*[set(x) for x in by.values()]))
    RES = d["res_layers"]; alpha = d["alpha"]; SITES = d["sites"]
    out = dict(tag=tag, t1=d["t1_menu"], key_rel_diff_max=d["key_rel_diff_max"], n=len(sis), alpha=alpha, I0={})
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
    match = [(p + 3, p) for p in PS]; nonmatch = [(ns, p) for ns in NSS for p in PS if ns - 3 != p]
    stay = lambda arm, cells, ss: float(np.mean([by[ns][s]["land"][f"{arm}_p{p}"] == p for ns, p in cells for s in ss]))
    def boot(f):
        v = [f([sis[i] for i in rng.integers(0, len(sis), len(sis))]) for _ in range(B)]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    nv = "no valid run"
    res = {}
    for site in SITES:
        sn = 46 if site == SITES[0] else 50
        tagd = "L46_r9" if sn == 46 else "L50_r9"
        m = dz[f"m_{tagd}"].astype(np.float64); sg = float(d["sigma"][str(site)])
        nxt = RES[0] if sn == 46 else RES[-1]
        li = RES.index(nxt)
        ratios = [float((Z[f"Ns{ns}_s{s}_PLUS{sn}_p{p}_res"][li].astype(np.float64) - Z[f"Ns{ns}_s{s}_E_EARLY_p{p}_res"][li].astype(np.float64)) @ m
                        / (alpha * sg)) for ns in NSS for s in sis for p in PS]
        live = float(np.median(ratios))
        dS = lambda ss, sn=sn: stay(f"PLUS{sn}", nonmatch, ss) - stay("E_EARLY", nonmatch, ss)
        dR = lambda ss, sn=sn: stay(f"RAND{sn}", nonmatch, ss) - stay("E_EARLY", nonmatch, ss)
        D = lambda ss, sn=sn: dS(ss, sn) - dR(ss, sn)
        vS, cS, vD, cD = dS(sis), boot(dS), D(sis), boot(D)
        if not ok_all:
            verdict = nv
        elif live < .50:
            verdict = "uninterpretable (the steering does not persist)"
        elif vS >= .15 and cS[0] > 0 and vD >= .10 and cD[0] > 0:
            verdict = "the goal-attainment signal stops the update when amplified"
        elif abs(vD) <= .05 and cD[0] >= -.10 and cD[1] <= .10:
            verdict = "represented but not used to stop the update"
        else:
            verdict = "partial"
        desc = {}
        for arm in ("PLUS", "MINUS", "RAND"):
            for nmc, cells in (("match", match), ("nonmatch", nonmatch)):
                f = lambda ss, arm=arm, cells=cells, sn=sn: stay(f"{arm}{sn}", cells, ss) - stay("E_EARLY", cells, ss)
                desc[f"{arm}_{nmc}"] = [f(sis), boot(f)]
            desc[f"{arm}_changed"] = float(np.mean([by[ns][s]["land"][f"{arm}{sn}_p{p}"] != by[ns][s]["land"][f"E_EARLY_p{p}"]
                                                   for ns in NSS for s in sis for p in PS]))
        res[str(sn)] = dict(live_ratio=live, dS=[vS, cS], dR=[dR(sis), boot(dR)], D=[vD, cD], verdict=verdict, desc=desc)
    out["USE"] = res
    # descriptive: late hold margins per arm
    def hold(arm, p, ss):
        v = []
        for ns in NSS:
            for s in ss:
                for L in ("51", "55"):
                    for row in (5, 9):
                        z = by[ns][s]["qk"][f"{arm}_p{p}"][L][row]
                        v.append(z[p - 1] - max(z[p - 2], z[p] if p < len(z) else -1e9))
        return float(np.mean(v))
    arms = ["E_EARLY"] + [f"{a_}{sn}" for sn in (46, 50) for a_ in ("PLUS", "MINUS", "RAND")]
    out["hold"] = {arm: float(np.mean([hold(arm, p, sis) for p in PS])) for arm in arms}
    out["stay_all"] = {arm: {"match": stay(arm, match, sis), "nonmatch": stay(arm, nonmatch, sis)} for arm in arms}
    fn = f"{V}/results/r3/s849u_analysis.json" if tag == "s849u_27b" else f"{V}/results/r3/s849u_analysis_{tag}.json"
    json.dump(out, open(fn, "w"), indent=1)
    print(json.dumps(out["I0"])[:600])
    for sn, v in res.items():
        print("site", sn, "live %.2f" % v["live_ratio"], "dS %.3f %s" % (v["dS"][0], v["dS"][1]), "dR %.3f" % v["dR"][0], "D %.3f %s" % (v["D"][0], v["D"][1]), "->", v["verdict"])
        print("   desc", json.dumps({k: (round(x[0], 3) if isinstance(x, list) else round(x, 3)) for k, x in v["desc"].items()}))
    print("hold", json.dumps(out["hold"])); print("stay_all", json.dumps(out["stay_all"]))


if __name__ == "__main__":
    main(*sys.argv[1:])
