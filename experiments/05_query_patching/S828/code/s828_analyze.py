"""S828 analysis (rules registered in knowledge/DECISIONS.md with the run, before any S828 data).  Cases (N, k, j): N7 (4, 4), N9 (5, 5) identity,
N8 (5, 4) new condition.  Stage 1 = s828_<tag>_natural.json, stage 2 = s828_<tag>_twoway.json, predictions research/ces/outputs/
s828_predictions_<tag>.json (fixed between the stages).  dP_x(arm) = P(arm -> x) - P(reference -> x); references QNOOP_E (END arms), QNOOP_S
(START arms).  Sides: start side = pictures before j, end side = after j.  Scene bootstrap, 4,000 draws, seed 0.
  I0  t1 >= .95 (both stages); QNOOP_E = END and QNOOP_S = START in >= .90 of scenes per case with the 95th percentile of the largest menu-logit
      difference <= .5; stage-2 END / START land as in stage 1 in >= .90; START(j) -> j >= .90.  Key identity reported (<= .05 'query-only').
  T1  (answer route) identity cases: A4_FULL dP_j >= .15 (interval above 0) -> 'the answer rows' own address is operative'.
  T2  (two routes suffice) identity cases: P(BA_FULL -> j) >= .80 -> 'the two address routes account for the identity failure', else 'incomplete'.
  T3  (dissociation) identity cases: B_FULL lowers the end-side share by >= .15 (interval below 0) while the start-side share changes by > -.10,
      and A4_FULL lowers the start-side share by >= .15 (interval below 0) -> 'the two routes carry different error sides'.
  T4  (sufficiency) R_BA dP_j <= -.30 (interval below 0) -> 'the two-route END address breaks START' (R_B, R_A4 reported).
  T5  (non-address part) BA_PERP dP_j >= .15 (interval above 0) -> 'the non-address query parts carry part of the error', else 'not shown'.
  T6  (answer row alone) |dP_j(A1_FULL) - dP_j(A4_FULL)| <= .10 -> 'the answer row alone carries the answer-route effect'.
  PL1 (interventions predicted) share of (case x arm x {j - 1, j, j + 1}) predictions within .15 of the observed shares >= .70 -> 'the two-route
      law predicts the interventions'.
  PL2 (information) mean log-probability per landing of the predictions over the no-change baseline (recipient's natural shares, +.5 smoothing),
      arms pooled, interval's lower end > 0 -> 'informative'.
  PL3 (new cell) N8 END(5) natural: predictions for {j - 1, j, j + 1} all within .15 -> 'the law transfers to a new cell'.
  Generality: per case.
Descriptive: landings, dP of j - 1 / j / j + 1 and of the sides, menu-logit margins, measured vs replacement-predicted internal scores.
Usage: python s828_analyze.py TAG"""
from __future__ import annotations

import json, sys
import numpy as np

B = 4000
AE = ["B_FULL", "A1_FULL", "A4_FULL", "BA_FULL", "BA_HALF", "BA_PERP"]
AS = ["R_B", "R_A4", "R_BA"]
REF = {**{a: "QNOOP_E" for a in AE}, **{a: "QNOOP_S" for a in AS}}
NAT = {**{a: "END" for a in AE}, **{a: "START" for a in AS}}
CASES = ((7, 4, 4), (9, 5, 5), (8, 5, 4))


def analyze(tag):
    d1 = json.load(open(f"/vlm/results/r3/s828_{tag}_natural.json")); d2 = json.load(open(f"/vlm/results/r3/s828_{tag}_twoway.json"))
    pf = json.load(open(f"/vlm/research/ces/outputs/s828_predictions_{tag}.json"))
    R1 = {(r["N"], r["si"]): r for r in d1["rows"]}; R2 = {(r["N"], r["si"]): r for r in d2["rows"]}
    rng = np.random.default_rng(0)
    kd = max(1.0 if d1.get("key_rel_diff_max") is None else d1["key_rel_diff_max"], 1.0 if d2.get("key_rel_diff_max") is None else d2["key_rel_diff_max"])
    rep = dict(tag=tag, model=d1["model"], t1=[d1["t1_menu"], d2["t1_menu"]], key_rel_diff_max=kd, query_only=bool(kd <= .05), cases={})
    L = str(min(d1["look"], key=lambda l: abs(l - 51)))
    zB = lambda r, run: np.mean([np.array(r["qk"][run][L][2]), np.array(r["qk"][run][L][5])], 0)
    zA = lambda r, run: np.array(r["qk"][run][L][9])
    def boot(rows, f):
        v = []
        for _ in range(B):
            pick = rng.choice(len(rows), len(rows)); v.append(f([rows[i] for i in pick]))
        v = np.array(v, float); v = v[np.isfinite(v)]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else None
    P = lambda rows, run, x: float(np.mean([r["land"][run] == x for r in rows])) if rows else float("nan")
    hits, gains_by_scene = [], {}
    for N, k, j in CASES:
        r1 = [R1[key] for key in sorted(R1) if key[0] == N]; r2 = [R2[key] for key in sorted(R2) if key[0] == N]
        if not r1 or not r2:
            continue
        ident = (k == j); cname = f"N{N}k{k}j{j}"
        side = lambda x: "j" if x == j else ("start" if 0 < x < j else ("end" if x > j else "menu"))
        Ps = lambda rows, run, sd: float(np.mean([side(r["land"][run]) == sd for r in rows]))
        p95 = lambda rows, a1, a2: float(np.percentile([max(abs(x - y) for x, y in zip(r["mlog"][a1], r["mlog"][a2])) for r in rows], 95))
        agree = lambda rows, a1, a2: float(np.mean([r["land"][a1] == r["land"][a2] for r in rows]))
        i0 = {f"{a1}={a2}": dict(agree=agree(r2, a1, a2), p95=p95(r2, a1, a2)) for a1, a2 in (("QNOOP_E", "END"), ("QNOOP_S", "START"))}
        i0["stage_consistency"] = float(np.mean([R2[(N, r["si"])]["land"][x] == r["land"][x] for r in r1 for x in ("END", "START")]))
        i0["start_acc"] = P(r1, "START", j)
        i0["ok"] = bool(min(d1["t1_menu"] or 0, d2["t1_menu"] or 0) >= .95 and all(x["agree"] >= .90 and x["p95"] <= .5 for nm, x in i0.items() if "=" in nm)
                        and i0["stage_consistency"] >= .90 and i0["start_acc"] >= .90)
        c = dict(I0=i0)
        S = {}
        for a in AE + AS:
            f = lambda q, a=a, x=j: P(q, a, x) - P(q, REF[a], x)
            fs = {sd: (lambda q, a=a, sd=sd: Ps(q, a, sd) - Ps(q, REF[a], sd)) for sd in ("start", "end")}
            lm = lambda q, a=a: float(np.mean([r["mlog"][a][j - 1] - max(v_ for i_, v_ in enumerate(r["mlog"][a]) if i_ != j - 1) for r in q]))
            S[a] = dict(dPj=f(r2), ci=boot(r2, f), P_j=P(r2, a, j), d_start=fs["start"](r2), ci_start=boot(r2, fs["start"]),
                        d_end=fs["end"](r2), ci_end=boot(r2, fs["end"]), margin=lm(r2), margin_ref=lm.__call__(r2) if False else None)
        c["arms"] = S
        if ident:
            a4, b, ba = S["A4_FULL"], S["B_FULL"], S["BA_FULL"]
            c["T1"] = "the answer rows' own address is operative" if (a4["dPj"] >= .15 and a4["ci"][0] > 0) else "not shown"
            c["T2"] = "the two address routes account for the identity failure" if ba["P_j"] >= .80 else "incomplete"
            c["T3"] = ("the two routes carry different error sides" if (b["d_end"] <= -.15 and b["ci_end"][1] < 0 and b["d_start"] > -.10
                                                                        and a4["d_start"] <= -.15 and a4["ci_start"][1] < 0) else "not shown")
        rb = S["R_BA"]
        c["T4"] = "the two-route END address breaks START" if (rb["dPj"] <= -.30 and rb["ci"][1] < 0) else "not shown"
        pp = S["BA_PERP"]
        c["T5"] = "the non-address query parts carry part of the error" if (pp["dPj"] >= .15 and pp["ci"][0] > 0) else "not shown"
        c["T6"] = ("the answer row alone carries the answer-route effect" if abs(S["A1_FULL"]["dPj"] - S["A4_FULL"]["dPj"]) <= .10 else "the think-marker rows add")
        # predictions
        pc = pf["cases"][cname]; cand = (j - 1, j, j + 1)
        ph = {}
        for a in AE + AS:
            obs = {x: P(r2, a, x) for x in cand}
            hh = [abs(pc["arms"][a]["shares"][str(x)] - obs[x]) <= .15 for x in cand]; hits.extend(hh); ph[a] = dict(pred={x: pc["arms"][a]["shares"][str(x)] for x in cand}, obs=obs, within=hh)
            base = np.bincount([r["land"][NAT[a]] for r in r2 if r["land"][NAT[a]] >= 1], minlength=N + 1)[1:] + .5
            ps = {s_["si"]: s_["p"] for s_ in pc["arms"][a]["per_scene"]}
            for r in r2:
                x = r["land"][a]
                if x >= 1:
                    gains_by_scene.setdefault((N, r["si"]), []).append(float(np.log(max(ps[r["si"]][x - 1], 1e-9)) - np.log(base[x - 1] / base.sum())))
        c["PRED_arms"] = ph
        nat = pc["natural"]["END"]["shares"]
        c["PL3_natural"] = dict(pred={x: nat[str(x)] for x in cand}, obs={x: P(r1, "END", x) for x in cand},
                                within=[abs(nat[str(x)] - P(r1, "END", x)) <= .15 for x in cand])
        # measured vs replacement-predicted internal scores (descriptive)
        xs, ys = [], []
        for a in AE + AS:
            ps = {s_["si"]: s_ for s_ in pc["arms"][a]["per_scene"]}
            for r in r2:
                for zz, key_ in ((zB, "zB"), (zA, "zA")):
                    pr = np.array(ps[r["si"]][key_]); ob = zz(r, a); pr = pr - pr.mean(); ob = ob - ob.mean()
                    xs.extend(pr.tolist()); ys.extend(ob.tolist())
        c["internal_r"] = float(np.corrcoef(xs, ys)[0, 1])
        c["landing"] = {a: np.bincount([r["land"][a] for r in r2], minlength=N + 1).tolist() for a in r2[0]["land"]}
        c["landing_stage1"] = {a: np.bincount([r["land"][a] for r in r1], minlength=N + 1).tolist() for a in r1[0]["land"]}
        rep["cases"][cname] = c
    gl = [float(np.mean(v_)) for v_ in gains_by_scene.values()]
    glc = boot(gl, lambda q: float(np.mean(q))) if gl else None
    rep["PL1"] = dict(share=float(np.mean(hits)) if hits else None, verdict=("the two-route law predicts the interventions" if hits and np.mean(hits) >= .70 else "not shown"))
    rep["PL2"] = dict(gain=float(np.mean(gl)) if gl else None, ci=glc, verdict=("informative" if glc and glc[0] > 0 else "not shown"))
    n8 = rep["cases"].get("N8k5j4")
    rep["PL3"] = ("the law transfers to a new cell" if (n8 and all(n8["PL3_natural"]["within"])) else "not shown")
    return rep


if __name__ == "__main__":
    tag = sys.argv[1]
    rep = analyze(tag)
    print(f"== {tag} ({rep['model']}) t1 {rep['t1']} key diff {rep['key_rel_diff_max']} query-only {rep['query_only']}")
    for c, v in rep["cases"].items():
        print(f"   {c} I0 {v['I0']}")
        print(f"   {c} " + " | ".join(f"{t}: {v[t]}" for t in ("T1", "T2", "T3", "T4", "T5", "T6") if t in v))
        for a, s_ in v["arms"].items():
            print(f"      {a:8s} P_j {s_['P_j']:.3f} dPj {s_['dPj']:+.3f} {s_['ci']} d_start {s_['d_start']:+.3f} d_end {s_['d_end']:+.3f} margin {s_['margin']:+.2f}")
        print(f"   {c} PL3 natural {v['PL3_natural']}  internal r {v['internal_r']:.3f}")
        print(f"   {c} landing {v['landing']}")
    print(f"   PL1 {rep['PL1']}  PL2 {rep['PL2']}  PL3 {rep['PL3']}")
    json.dump(rep, open("/vlm/results/r3/s828_analysis.json", "w"), indent=1, default=float)
