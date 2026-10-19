"""S829 analysis (rules registered in knowledge/DECISIONS.md with the run, before any S829 data).  Cases (N, k, j): N7 (4, 4), N9 (5, 5) identity,
N8 (5, 4).  Stage 1 = s829_<tag>_natural.json, stage 2 = s829_<tag>_mode.json, predictions research/ces/outputs/s829_predictions_<tag>.json.
dP_j(arm) = P(arm -> j) - P(reference -> j); references QNOOP_E (END arms), QNOOP_S (START arms).  Scene bootstrap, 4,000 draws, seed 0.
Late picture mass = index-matching-head picture attention mass, mean of B1, B4 and the answer row, mean of L51 and L55.
  I0  as S828 (t1 >= .95; QNOOP agreement >= .90, 95th percentile <= .5; stage consistency >= .90; START(j) -> j >= .90; key identity reported).
  M0  (the mode variable) END's late picture mass exceeds START's (scene-paired difference, interval above 0) -> 'END re-reads pictures late'.
  T1  (conjunction, reverse) identity cases: R_ALL dP_j <= -.50 (interval below 0) and R_P dP_j > -.10 -> 'END's whole query breaks START, its
      reading mode alone does not' (R_U, R_UM reported).
  T2  (the mode is the picture-reading offset) identity cases: MASS dP_j >= .15 (interval above 0) and >= half of P_FULL's dP_j -> 'the
      picture-vs-text offset carries the reading mode'; and R_UM dP_j <= R_U dP_j - .10 -> 'adding END's offset to its address breaks START more'.
  T3  (where the address matters) identity cases: U_LATE dP_j >= .15 (interval above 0) and >= U_EARLY dP_j + .10 -> 'the late re-lookup's address
      carries the error'; U_EARLY dP_j >= .15 (interval above 0) -> 'the early lookup matters too'.
  T4  identity cases: |dP_j(P_LATE) - dP_j(P_FULL)| <= .10 -> 'the reading mode acts late'.
  PL1 share of (case x arm x {j - 1, j, j + 1}) predictions within .15 >= .70 -> 'the mixture law predicts the arms'.
  PL2 mean log-probability gain over the no-change baseline (recipient's natural shares, +.5 smoothing), interval's lower end > 0 -> 'informative'.
  PLm (the same mixture law from each run's own measured z and late mass, descriptive): calibration share.
Descriptive: landings, dP of j - 1 / j / j + 1, late picture mass per arm, menu-word attention on the target word.
Usage: python s829_analyze.py TAG"""
from __future__ import annotations

import json, sys
import numpy as np

B = 4000
AE = ["U_FULL", "U_EARLY", "U_LATE", "P_FULL", "P_LATE", "MASS"]
AS = ["R_U", "R_P", "R_ALL", "R_UM"]
REF = {**{a: "QNOOP_E" for a in AE}, **{a: "QNOOP_S" for a in AS}}
NAT = {**{a: "END" for a in AE}, **{a: "START" for a in AS}}
CASES = ((7, 4, 4), (9, 5, 5), (8, 5, 4))


def analyze(tag):
    d1 = json.load(open(f"/vlm/results/r3/s829_{tag}_natural.json")); d2 = json.load(open(f"/vlm/results/r3/s829_{tag}_mode.json"))
    pf = json.load(open(f"/vlm/research/ces/outputs/s829_predictions_{tag}.json"))
    R1 = {(r["N"], r["si"]): r for r in d1["rows"]}; R2 = {(r["N"], r["si"]): r for r in d2["rows"]}
    rng = np.random.default_rng(0)
    kd = max(1.0 if d1.get("key_rel_diff_max") is None else d1["key_rel_diff_max"], 1.0 if d2.get("key_rel_diff_max") is None else d2["key_rel_diff_max"])
    rep = dict(tag=tag, model=d1["model"], t1=[d1["t1_menu"], d2["t1_menu"]], key_rel_diff_max=kd, query_only=bool(kd <= .05), cases={})
    look = d1["look"]; L = str(min(look, key=lambda l: abs(l - 51))); L2 = str(min(look, key=lambda l: abs(l - 55)))
    zB = lambda r, run: np.mean([np.array(r["qk"][run][L][2]), np.array(r["qk"][run][L][5])], 0)
    zA = lambda r, run: np.array(r["qk"][run][L][9])
    mass = lambda r, run: float(np.mean([np.mean([r["mass"][run][LL][i] for i in (2, 5, 9)]) for LL in (L, L2)]))
    law = pf["law"]; soft = lambda s: np.exp(s - s.max()) / np.exp(s - s.max()).sum()
    Lz = lambda zb, za: soft(law["bB"] * zb + law["bA"] * za)
    def boot(rows, f):
        v = []
        for _ in range(B):
            pick = rng.choice(len(rows), len(rows)); v.append(f([rows[i] for i in pick]))
        v = np.array(v, float); v = v[np.isfinite(v)]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else None
    P = lambda rows, run, x: float(np.mean([r["land"][run] == x for r in rows])) if rows else float("nan")
    hits, hits_m, gains_by_scene = [], [], {}
    for N, k, j in CASES:
        r1 = [R1[key] for key in sorted(R1) if key[0] == N]; r2 = [R2[key] for key in sorted(R2) if key[0] == N]
        if not r1 or not r2:
            continue
        ident = (k == j); cname = f"N{N}k{k}j{j}"
        p95 = lambda rows, a1, a2: float(np.percentile([max(abs(x - y) for x, y in zip(r["mlog"][a1], r["mlog"][a2])) for r in rows], 95))
        agree = lambda rows, a1, a2: float(np.mean([r["land"][a1] == r["land"][a2] for r in rows]))
        i0 = {f"{a1}={a2}": dict(agree=agree(r2, a1, a2), p95=p95(r2, a1, a2)) for a1, a2 in (("QNOOP_E", "END"), ("QNOOP_S", "START"))}
        i0["stage_consistency"] = float(np.mean([R2[(N, r["si"])]["land"][x] == r["land"][x] for r in r1 for x in ("END", "START")]))
        i0["start_acc"] = P(r1, "START", j)
        i0["ok"] = bool(min(d1["t1_menu"] or 0, d2["t1_menu"] or 0) >= .95 and all(x["agree"] >= .90 and x["p95"] <= .5 for nm, x in i0.items() if "=" in nm)
                        and i0["stage_consistency"] >= .90 and i0["start_acc"] >= .90)
        c = dict(I0=i0)
        dm = lambda q: float(np.mean([mass(r, "END") - mass(r, "START") for r in q]))
        dmc = boot(r1, dm)
        c["M0"] = dict(END=float(np.mean([mass(r, "END") for r in r1])), START=float(np.mean([mass(r, "START") for r in r1])), diff=dm(r1), ci=dmc,
                       verdict=("END re-reads pictures late" if dmc and dmc[0] > 0 else "not shown"))
        S = {}
        for a in AE + AS:
            f = lambda q, a=a: P(q, a, j) - P(q, REF[a], j)
            S[a] = dict(dPj=f(r2), ci=boot(r2, f), P_j=P(r2, a, j), dP_jm1=P(r2, a, j - 1) - P(r2, REF[a], j - 1), dP_jp1=P(r2, a, j + 1) - P(r2, REF[a], j + 1),
                        late_mass=float(np.mean([mass(r, a) for r in r2])))
        c["arms"] = S
        c["late_mass"] = {a: float(np.mean([mass(r, a) for r in r2])) for a in ["QNOOP_E", "QNOOP_S"] + AE + AS}
        if ident:
            ra, rp = S["R_ALL"], S["R_P"]
            c["T1"] = ("END's whole query breaks START, its reading mode alone does not" if (ra["dPj"] <= -.50 and ra["ci"][1] < 0 and rp["dPj"] > -.10) else "not shown")
            ms, pfu = S["MASS"], S["P_FULL"]
            t2a = ms["dPj"] >= .15 and ms["ci"][0] > 0 and ms["dPj"] >= .5 * pfu["dPj"]
            t2b = S["R_UM"]["dPj"] <= S["R_U"]["dPj"] - .10
            c["T2"] = dict(offset=("the picture-vs-text offset carries the reading mode" if t2a else "not shown"),
                           reverse=("adding END's offset to its address breaks START more" if t2b else "not shown"))
            ul, ue = S["U_LATE"], S["U_EARLY"]
            c["T3"] = dict(late=("the late re-lookup's address carries the error" if (ul["dPj"] >= .15 and ul["ci"][0] > 0 and ul["dPj"] >= ue["dPj"] + .10) else "not shown"),
                           early=("the early lookup matters too" if (ue["dPj"] >= .15 and ue["ci"][0] > 0) else "not shown"))
            c["T4"] = "the reading mode acts late" if abs(S["P_LATE"]["dPj"] - S["P_FULL"]["dPj"]) <= .10 else "not shown"
        pc = pf["cases"][cname]; cand = (j - 1, j, j + 1)
        ph = {}
        for a in AE + AS:
            obs = {x: P(r2, a, x) for x in cand}
            hh = [abs(pc["arms"][a]["shares"][str(x)] - obs[x]) <= .15 for x in cand]; hits.extend(hh)
            # measured-variable version of the same law
            pm = []
            for r in r2:
                r1s = R1[(N, r["si"])]; mE, mS = mass(r1s, "END"), mass(r1s, "START")
                w = float(np.clip((mass(r, a) - mS) / (mE - mS), 0, 1)) if abs(mE - mS) > 1e-6 else 0.0
                pm.append(w * Lz(zB(r, a), zA(r, a)) + (1 - w) * Lz(zB(r1s, "START"), zA(r1s, "START")))
            pm = np.mean(pm, 0)
            hits_m.extend([abs(pm[x - 1] - obs[x]) <= .15 for x in cand])
            ph[a] = dict(pred={x: pc["arms"][a]["shares"][str(x)] for x in cand}, pred_measured={x: float(pm[x - 1]) for x in cand}, obs=obs, within=hh)
            base = np.bincount([r["land"][NAT[a]] for r in r2 if r["land"][NAT[a]] >= 1], minlength=N + 1)[1:] + .5
            ps = {s_["si"]: s_["p"] for s_ in pc["arms"][a]["per_scene"]}
            for r in r2:
                x = r["land"][a]
                if x >= 1:
                    gains_by_scene.setdefault((N, r["si"]), []).append(float(np.log(max(ps[r["si"]][x - 1], 1e-9)) - np.log(base[x - 1] / base.sum())))
        c["PRED_arms"] = ph
        c["landing"] = {a: np.bincount([r["land"][a] for r in r2], minlength=N + 1).tolist() for a in r2[0]["land"]}
        rep["cases"][cname] = c
    gl = [float(np.mean(v_)) for v_ in gains_by_scene.values()]
    glc = boot(gl, lambda q: float(np.mean(q))) if gl else None
    rep["PL1"] = dict(share=float(np.mean(hits)) if hits else None, verdict=("the mixture law predicts the arms" if hits and np.mean(hits) >= .70 else "not shown"))
    rep["PL2"] = dict(gain=float(np.mean(gl)) if gl else None, ci=glc, verdict=("informative" if glc and glc[0] > 0 else "not shown"))
    rep["PLm"] = dict(share=float(np.mean(hits_m)) if hits_m else None)
    return rep


if __name__ == "__main__":
    tag = sys.argv[1]
    rep = analyze(tag)
    print(f"== {tag} ({rep['model']}) t1 {rep['t1']} key diff {rep['key_rel_diff_max']} query-only {rep['query_only']}")
    for c, v in rep["cases"].items():
        print(f"   {c} I0 {v['I0']}")
        print(f"   {c} M0 {v['M0']}")
        print(f"   {c} " + " | ".join(f"{t}: {v[t]}" for t in ("T1", "T2", "T3", "T4") if t in v))
        for a, s_ in v["arms"].items():
            pr = v["PRED_arms"][a]
            print(f"      {a:8s} P_j {s_['P_j']:.3f} dPj {s_['dPj']:+.3f} {s_['ci']} late mass {s_['late_mass']:.3f} | pred {pr['pred']} measured-law {pr['pred_measured']} obs {pr['obs']}")
        print(f"   {c} landing {v['landing']}")
    print(f"   PL1 {rep['PL1']}  PL2 {rep['PL2']}  PLm {rep['PLm']}")
    json.dump(rep, open("/vlm/results/r3/s829_analysis.json", "w"), indent=1, default=float)
