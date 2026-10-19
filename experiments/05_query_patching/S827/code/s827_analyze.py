"""S827 analysis (rules registered in knowledge/DECISIONS.md with the run, before any S827 data; the unrun first draft is kept as
exp/s827_analyze_draft1.py).  Cases N7 (k = 4) and N9 (k = 5), identity references (j = k).  Stage 1 = natural runs (s827_<tag>_natural.json),
stage 2 = the arms (s827_<tag>_identity.json), predictions fixed between them (research/ces/outputs/s827_predictions_<tag>.json, exp/s827_predict.py).
Scene bootstrap, 4,000 draws, seed 0.
Fixed internal profile for M / C: z(run) = the boundary newline rows' (B1, B4 mean) index-matching-head block scores at L51.  END change
Delta = z(END) - z(START) (same keys: exact); local coefficients relative to the far blocks (outside k - 1, k, k + 1): a_i = Delta_i - mean_far;
D = a_k, F = (a_(k-1) + a_(k+1)) / 2, T = a_(k+1) - a_(k-1).  dP_x(arm) = P(arm -> x) - P(reference -> x); references QNOOP_E (A_*), QNOOP_S (R_*).
  I0  t1 >= .95 (both stages); QNOOP_E = END and QNOOP_S = START in >= .90 of scenes per case with the 95th percentile of the largest menu-logit
      difference <= .5; stage-2 END / START land as in stage 1 in >= .90; START(k) -> k >= .90 (stage 1).  Key identity reported ('query-only
      difference' if the largest relative block-key difference END vs START <= .05).
  M1  (wrong vs weakened; stage 1) END(k) scenes landing on k +- 1: share whose z argmax != k: >= .60 'a wrong address is written', <= .30 'the
      right address is weakened', else 'mixed'.
  M2  (side) in those scenes, agreement of sign(T) with the landing side: >= .70 (interval's lower end > .50) 'the side is written in the boundary
      rows' address', else 'side not readable there'.
  M3  (shape) 'none' if D's interval includes 0; else 'flatten' if F < 0 (interval below 0) and |F| > |D| / 3; 'shift' if F > 0 (interval above
      0) and |F| > |D| / 3; otherwise 'dip'.
  C1  A_FULL dP_k >= .15 (interval above 0) -> 'the boundary rows' address is operative'; else if A_FULL raised the internal margin
      z_k - max(z_(k-1), z_(k+1)) (interval above 0) -> 'address corrected, answer does not follow'; else 'not shown'.
  C2  (H-A vs H-B) 'H-A' if A_DIP dP_k >= .15 (interval above 0), A_PREV dP_k < .10 and R_DIP dP_k <= -.15 (interval below 0); 'H-B' if A_PREV
      dP_k >= .15 (interval above 0) with dP_(k-1) <= -.15 (interval below 0), A_DIP dP_k < .10 and R_PREV dP_(k-1) >= .15 (interval above 0);
      'both' if A_DIP and A_PREV each give dP_k >= .15 (intervals above 0); 'neither' if C1 holds and both give dP_k < .10; otherwise 'mixed'.
      NEXT: A_NEXT dP_(k+1) <= -.15 (interval below 0) -> 'the k + 1 errors come from END's k + 1 part'.
  C3  R_FULL dP_k <= -.15 (interval below 0) -> 'END's address change is sufficient for the error'; side: scenes where R_FULL and END both land on
      k +- 1, agreement >= .70 (interval's lower end > .50) -> 'same side'.
  C4  dose (when the 100 % arm gives dP_k >= .15): g = dP_k(50 %) / dP_k(100 %) for FULL (A_HALF), DIP (A_DIP50), PREV (A_PREV50): 'graded'
      .25-.75, 'saturating' > .75, 'threshold-like' < .25.
  C5  (definition) A_UK vs A_DIP: |dP_k difference| <= .10 -> 'the two copy-direction definitions agree'.
  PRED (the fixed predictions):
      (a) law on END(k): share of {k - 1, k, k + 1} predictions within .15 of stage-1 observed.
      (b) held-out-scene operator: head-level sign agreement >= .75 -> 'scene-independent END operator'; answer gain > 0 (interval above 0) ->
          'the operator predicts scene-level answers'.
      (c) arms, internal: correlation of predicted vs observed change of z_F (observed = arm - recipient natural, stage 2), pooled over arms,
          scenes and blocks, per case: >= .70 -> 'the internal change is predicted' (slope reported: > 1 compounding, < 1 damping); answer:
          share of (arm x {k - 1, k, k + 1}) within .15 >= .70 -> 'calibrated'; mean log-probability gain over the no-change baseline
          (recipient's natural shares, +.5 smoothing) with the interval's lower end > 0 -> 'informative'.
  Generality: per case and both.
Descriptive: every run's landing; dP of k - 1 / k / k + 1; menu-logit margin; internal margins; manipulation checks (observed change of a_k,
a_(k-1), a_(k+1) under each arm); head-level local coefficients with gated output norms.
Usage: python s827_analyze.py TAG [TAG ...]"""
from __future__ import annotations

import json, sys
import numpy as np

B = 4000
AE = ["A_FULL", "A_HALF", "A_DIP", "A_DIP50", "A_PREV", "A_PREV50", "A_NEXT", "A_FLANK", "A_UK"]
AS = ["R_FULL", "R_DIP", "R_PREV", "R_NEXT"]
REF = {**{a: "QNOOP_E" for a in AE}, **{a: "QNOOP_S" for a in AS}}
NAT = {**{a: "END" for a in AE}, **{a: "START" for a in AS}}


def analyze(tag):
    d1 = json.load(open(f"/vlm/results/r3/s827_{tag}_natural.json")); d2 = json.load(open(f"/vlm/results/r3/s827_{tag}_identity.json"))
    pf = json.load(open(f"/vlm/research/ces/outputs/s827_predictions_{tag}.json"))
    R1 = {(r["N"], r["si"]): r for r in d1["rows"]}; R2 = {(r["N"], r["si"]): r for r in d2["rows"]}
    Z2 = np.load(f"/vlm/results/r3/s827_{tag}_identity.npz")
    rng = np.random.default_rng(0)
    L51 = str(min(d1["look"], key=lambda l: abs(l - 51)))
    kd = max(1.0 if d1.get("key_rel_diff_max") is None else d1["key_rel_diff_max"], 1.0 if d2.get("key_rel_diff_max") is None else d2["key_rel_diff_max"])
    rep = dict(tag=tag, model=d1["model"], t1=[d1["t1_menu"], d2["t1_menu"]], n=len(R1), layer=L51, key_rel_diff_max=kd, query_only=bool(kd <= .05))
    z = lambda r, run, li=L51: np.mean([np.array(r["qk"][run][li][2]), np.array(r["qk"][run][li][5])], 0)
    zf = lambda r, run: np.mean([np.array(r["qk"][run][str(pf["layer"])][i]) for i in {"B1B4": [2, 5], "B": [1, 2, 3, 4, 5], "QB": [0, 1, 2, 3, 4, 5]}[pf["rows"]]], 0)
    def boot(rows, f):
        v = []
        for _ in range(B):
            pick = rng.choice(len(rows), len(rows)); v.append(f([rows[i] for i in pick]))
        v = np.array(v, float); v = v[np.isfinite(v)]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else None
    P = lambda rows, run, x: float(np.mean([r["land"][run] == x for r in rows])) if rows else float("nan")
    def loc(r, a1, a0, li=L51):
        k = r["k"]; N = r["N"]; De = z(r, a1, li) - z(r, a0, li)
        far = [i for i in range(N) if i not in (k - 2, k - 1, k)]; m = De[far].mean()
        return {x: float(De[x - 1] - m) for x in (k - 1, k, k + 1)}
    def coefs(r, li=L51):
        a = loc(r, "END", "START", li); k = r["k"]
        return a[k], (a[k - 1] + a[k + 1]) / 2, a[k + 1] - a[k - 1]
    imar = lambda r, run: (lambda zz, k: zz[k - 1] - max(zz[k - 2], zz[k]))(z(r, run), r["k"])
    rep["cases"] = {}
    for N, k in ((7, 4), (9, 5)):
        r1 = [R1[key] for key in sorted(R1) if key[0] == N]; r2 = [R2[key] for key in sorted(R2) if key[0] == N]
        if not r1 or not r2:
            continue
        p95 = lambda rows, a1, a2: float(np.percentile([max(abs(x - y) for x, y in zip(r["mlog"][a1], r["mlog"][a2])) for r in rows], 95))
        agree = lambda rows, a1, a2: float(np.mean([r["land"][a1] == r["land"][a2] for r in rows]))
        i0 = {f"{a1}={a2}": dict(agree=agree(r2, a1, a2), p95=p95(r2, a1, a2)) for a1, a2 in (("QNOOP_E", "END"), ("QNOOP_S", "START"))}
        i0["stage_consistency"] = float(np.mean([R2[(N, r["si"])]["land"][x] == r["land"][x] for r in r1 for x in ("END", "START")]))
        i0["start_acc"] = P(r1, "START", k); i0["start_m1_acc"] = P(r1, "START_m1", k - 1); i0["start_p1_acc"] = P(r1, "START_p1", k + 1)
        i0["ok"] = bool(min(d1["t1_menu"] or 0, d2["t1_menu"] or 0) >= .95 and all(x["agree"] >= .90 and x["p95"] <= .5 for nm, x in i0.items() if "=" in nm)
                        and i0["stage_consistency"] >= .90 and i0["start_acc"] >= .90)
        c = dict(I0=i0)
        # M (stage 1)
        err = [r for r in r1 if r["land"]["END"] in (k - 1, k + 1)]
        mv = lambda q: float(np.mean([int(np.argmax(z(r, "END"))) + 1 != k for r in q])) if q else float("nan")
        m1 = mv(err)
        c["M1"] = dict(n=len(err), moved=m1, ci=boot(err, mv) if err else None,
                       to_answered=float(np.mean([int(np.argmax(z(r, "END"))) + 1 == r["land"]["END"] for r in err])) if err else None,
                       correct_argmax_k=(float(np.mean([int(np.argmax(z(r, "END"))) + 1 == k for r in r1 if r["land"]["END"] == k]))
                                         if any(r["land"]["END"] == k for r in r1) else None),
                       verdict=("a wrong address is written" if m1 >= .60 else "the right address is weakened" if m1 <= .30 else "mixed"))
        sd = lambda q: float(np.mean([(coefs(r)[2] > 0) == (r["land"]["END"] == k + 1) for r in q])) if q else float("nan")
        m2, m2c = sd(err), (boot(err, sd) if err else None)
        c["M2"] = dict(agree=m2, ci=m2c, verdict=("the side is written in the boundary rows' address" if (m2 >= .70 and m2c and m2c[0] > .50)
                                                  else "side not readable there"))
        Dm = lambda q: float(np.mean([coefs(r)[0] for r in q])); Fm = lambda q: float(np.mean([coefs(r)[1] for r in q]))
        Dv, Dc, Fv, Fc = Dm(r1), boot(r1, Dm), Fm(r1), boot(r1, Fm)
        m3 = ("none" if Dc[0] <= 0 <= Dc[1] else "flatten" if (Fv < 0 and Fc[1] < 0 and abs(Fv) > abs(Dv) / 3)
              else "shift" if (Fv > 0 and Fc[0] > 0 and abs(Fv) > abs(Dv) / 3) else "dip")
        c["M3"] = dict(D=Dv, ci_D=Dc, F=Fv, ci_F=Fc, T_mean=float(np.mean([coefs(r)[2] for r in r1])), verdict=m3,
                       per_layer={li: dict(D=float(np.mean([coefs(r, li)[0] for r in r1])), F=float(np.mean([coefs(r, li)[1] for r in r1])))
                                  for li in r1[0]["qk"]["END"]})
        # head level (descriptive): matcher heads at L51, B1 / B4 rows, local coefficients and gated output norms (stage 2 naturals)
        li_ = d2["look"].index(int(L51)); MH = d2["matchers"][L51]; hs = {}
        for h in MH:
            A_ = []
            for r in r2:
                cE = Z2[f"N{N}_s{r['si']}_END_cap"][li_][[2, 5]].astype(np.float32).mean(0)[h]; cS = Z2[f"N{N}_s{r['si']}_START_cap"][li_][[2, 5]].astype(np.float32).mean(0)[h]
                De = cE - cS; far = [i for i in range(N) if i not in (k - 2, k - 1, k)]
                A_.append([De[x - 1] - De[far].mean() for x in (k - 1, k, k + 1)])
            A_ = np.array(A_); ho = np.mean([Z2[f"N{N}_s{r['si']}_END_hout"][li_][0][[2, 5]].astype(np.float32).mean(0)[h] for r in r2])
            hs[str(h)] = dict(mean=A_.mean(0).tolist(), se=(A_.std(0, ddof=1) / np.sqrt(len(A_))).tolist(), out_norm=float(ho))
        c["heads_L51"] = hs
        # C (stage 2)
        def dP(a, x):
            return lambda q: P(q, a, x) - P(q, REF[a], x)
        S = {}
        for a in AE + AS:
            dk = dP(a, k)
            dim = lambda q, a=a: float(np.mean([imar(r, a) - imar(r, REF[a]) for r in q]))
            man = {x: float(np.mean([loc(r, a, "START" if NAT[a] == "END" else "START")[x] - loc(r, REF[a], "START")[x] for r in r2])) for x in (k - 1, k, k + 1)}
            S[a] = dict(dPk=dk(r2), ci=boot(r2, dk), dP_km1=dP(a, k - 1)(r2), ci_km1=boot(r2, dP(a, k - 1)), dP_kp1=dP(a, k + 1)(r2), ci_kp1=boot(r2, dP(a, k + 1)),
                        dmargin=dim(r2), ci_margin=boot(r2, dim), manipulation=man)
        c["arms"] = S
        f = S["A_FULL"]; op = bool(f["dPk"] >= .15 and f["ci"][0] > 0)
        c["C1"] = dict(verdict=("the boundary rows' address is operative" if op else "address corrected, answer does not follow"
                                if (f["ci_margin"] and f["ci_margin"][0] > 0) else "not shown"))
        dp, pv = S["A_DIP"], S["A_PREV"]
        up = lambda s_: s_["dPk"] >= .15 and s_["ci"][0] > 0
        if up(dp) and up(pv):
            c2 = "both"
        elif up(dp) and pv["dPk"] < .10 and S["R_DIP"]["dPk"] <= -.15 and S["R_DIP"]["ci"][1] < 0:
            c2 = "H-A"
        elif (up(pv) and pv["dP_km1"] <= -.15 and pv["ci_km1"][1] < 0 and dp["dPk"] < .10 and S["R_PREV"]["dP_km1"] >= .15 and S["R_PREV"]["ci_km1"][0] > 0):
            c2 = "H-B"
        elif op and dp["dPk"] < .10 and pv["dPk"] < .10:
            c2 = "neither"
        else:
            c2 = "mixed"
        nx = S["A_NEXT"]
        c["C2"] = dict(verdict=c2, next=("the k + 1 errors come from END's k + 1 part" if (nx["dP_kp1"] <= -.15 and nx["ci_kp1"][1] < 0) else "not shown"))
        rf = S["R_FULL"]
        both = [r for r in r2 if r["land"]["R_FULL"] in (k - 1, k + 1) and r["land"]["END"] in (k - 1, k + 1)]
        sg = lambda q: float(np.mean([r["land"]["R_FULL"] == r["land"]["END"] for r in q])) if q else float("nan")
        sv, svc = sg(both), (boot(both, sg) if both else None)
        c["C3"] = dict(sufficient=bool(rf["dPk"] <= -.15 and rf["ci"][1] < 0), side_n=len(both), side_agree=sv, side_ci=svc,
                       same_side=bool(both and sv >= .70 and svc and svc[0] > .50))
        c4 = {}
        for nm, a100, a50 in (("FULL", "A_FULL", "A_HALF"), ("DIP", "A_DIP", "A_DIP50"), ("PREV", "A_PREV", "A_PREV50")):
            if S[a100]["dPk"] >= .15:
                g = S[a50]["dPk"] / S[a100]["dPk"]
                c4[nm] = dict(g=g, verdict=("graded" if .25 <= g <= .75 else "saturating" if g > .75 else "threshold-like"))
            else:
                c4[nm] = dict(g=None, verdict="not tested")
        c["C4"] = c4
        c["C5"] = dict(diff=S["A_UK"]["dPk"] - S["A_DIP"]["dPk"], verdict=("the two copy-direction definitions agree" if abs(S["A_UK"]["dPk"] - S["A_DIP"]["dPk"]) <= .10 else "differ"))
        c["landing"] = {a: np.bincount([r["land"][a] for r in r2], minlength=N + 1).tolist() for a in r2[0]["land"]}
        c["landing_stage1"] = {a: np.bincount([r["land"][a] for r in r1], minlength=N + 1).tolist() for a in r1[0]["land"]}
        # PRED
        cname = f"N{N}k{k}"; pa = pf["a"][cname]
        pa_hits = [abs(pa["pred"][str(x)] - pa["obs"][str(x)]) <= .15 for x in (k - 1, k, k + 1)]
        pb = pf["b"][cname]
        gb = [g for g in pb["answer_gain_per_scene"]]
        gbc = boot(gb, lambda q: float(np.mean(q))) if gb else None
        pc = pf["c"][cname]; xs, ys, hits, gains = [], [], [], []
        base = {nat: np.bincount([r["land"][nat] for r in r2 if r["land"][nat] >= 1], minlength=N + 1)[1:] + .5 for nat in ("END", "START")}
        per_scene_gain = {}
        for a in AE + AS:
            ps = {s_["si"]: s_ for s_ in pc[a]["per_scene"]}
            for r in r2:
                pr = ps[r["si"]]; obs_dz = zf(r, a) - zf(r, NAT[a]); obs_dz = obs_dz - obs_dz.mean(); pdz = np.array(pr["dz"]); pdz = pdz - pdz.mean()
                xs.extend(pdz.tolist()); ys.extend(obs_dz.tolist())
                x = r["land"][a]
                if x >= 1:
                    bb = base[NAT[a]]
                    per_scene_gain.setdefault(r["si"], []).append(float(np.log(max(pr["p"][x - 1], 1e-9)) - np.log(bb[x - 1] / bb.sum())))
            obs = {x: P(r2, a, x) for x in (k - 1, k, k + 1)}
            hits.extend([abs(pc[a]["shares"][str(x)] - obs[x]) <= .15 for x in (k - 1, k, k + 1)])
        xs, ys = np.array(xs), np.array(ys)
        rc = float(np.corrcoef(xs, ys)[0, 1]) if xs.std() > 0 and ys.std() > 0 else float("nan")
        slope = float((xs * ys).sum() / (xs * xs).sum()) if (xs * xs).sum() > 0 else float("nan")
        gl = [np.mean(v_) for v_ in per_scene_gain.values()]
        glc = boot(gl, lambda q: float(np.mean(q))) if gl else None
        c["PRED"] = dict(a=dict(hits=pa_hits, share=float(np.mean(pa_hits))),
                         b=dict(sign_agree=pb["sign_agree"], operator=("scene-independent END operator" if (pb["sign_agree"] or 0) >= .75 else "not shown"),
                                answer_gain=float(np.mean(gb)) if gb else None, ci=gbc,
                                answer=("the operator predicts scene-level answers" if (gbc and gbc[0] > 0) else "not shown")),
                         c=dict(internal_r=rc, slope=slope, internal=("the internal change is predicted" if rc >= .70 else "not predicted"),
                                calibrated_share=float(np.mean(hits)), calibrated=bool(np.mean(hits) >= .70),
                                gain=float(np.mean(gl)) if gl else None, ci_gain=glc, informative=bool(glc and glc[0] > 0)))
        rep["cases"][cname] = c
    cs = rep["cases"]
    rep["generality"] = {key: sorted({cs[c_][key]["verdict"] for c_ in cs}) for key in ("M1", "M2", "M3", "C1", "C2")} if cs else None
    return rep


if __name__ == "__main__":
    allr = {}
    for tag in sys.argv[1:]:
        try:
            rep = analyze(tag)
        except FileNotFoundError as e:
            print(f"[{tag}] missing {e}"); continue
        allr[tag] = rep
        print(f"== {tag} ({rep['model']}) t1 {rep['t1']} n {rep['n']} layer {rep['layer']} key diff {rep['key_rel_diff_max']} query-only {rep['query_only']}")
        for c, v in rep["cases"].items():
            print(f"   {c} I0 {v['I0']}")
            for key in ("M1", "M2", "C1", "C2", "C3", "C4", "C5"):
                print(f"   {c} {key} {v[key]}")
            print(f"   {c} M3 { {x: y for x, y in v['M3'].items() if x != 'per_layer'} }")
            for a, s_ in v["arms"].items():
                print(f"      {a:8s} dPk {s_['dPk']:+.3f} {s_['ci']} dk-1 {s_['dP_km1']:+.3f} dk+1 {s_['dP_kp1']:+.3f} dmargin {s_['dmargin']:+.2f} manip {s_['manipulation']}")
            print(f"   {c} PRED {v['PRED']}")
            print(f"   {c} landing {v['landing']}")
        print(f"   generality {rep['generality']}")
    json.dump(allr, open("/vlm/results/r3/s827_analysis.json", "w"), indent=1, default=float)
