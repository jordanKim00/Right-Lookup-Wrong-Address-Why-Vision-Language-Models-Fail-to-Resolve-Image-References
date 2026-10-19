"""S847 analysis (rules fixed in knowledge/DECISIONS.md before any S847 data).  Cases as S845 (identity (7, 4), (9, 5); non-identity (7, 5), (9, 7));
P_copy(arm) = share answering image k (identity: the target; otherwise 'correction removed'); scene bootstrap 4,000 (seed 0), pooled over valid cases.
I0 (per case): t1 >= .95; QNOOP = END_B in >= .90 of scenes (95th percentile of the largest menu-logit change <= .5); START(k) -> k >= .90 (stage A
and the stage-B donor START_B both); image keys identical (0.0); the corrected run moves the answer: P_copy(A_BAND_B) - P_copy(END_B) >= .15.
f(arm) = (P_copy(arm) - P_copy(END_B)) / (P_copy(A_BAND_B) - P_copy(END_B)); g(arm) = (P_copy(A_BAND_B) - P_copy(arm)) / (P_copy(A_BAND_B) - P_copy(END_B));
pooled as sums over valid cases.  (Amendment 10-17 before data: the references END_B / A_BAND_B are re-run in the same pass as the arms, because the
GPU smoke showed cross-batch drift between stages at n = 1; stage-A END / A_BAND stay the query donors; cross-stage agreement is reported.)
  PATH4  f(QALL) >= .80 with lower bound > .60 -> 'the band's correction reaches the answer through the late full-attention queries (51-63)';
         f(QALL) <= .60 -> 'the late queries do not carry all of it'; otherwise 'partial'.
  NEC    g(NEC) >= .70 with lower bound > .50 -> 'the late queries are needed for the band's correction'; g(NEC) <= .40 -> 'other routes suffice';
         otherwise 'partial'.
  WHERE  f(Q5963) - f(Q5155) >= .15 (interval above 0) -> 'the last two layers carry more'; <= -.15 (interval below 0) -> 'layers 51 / 55 carry
         more'; otherwise 'both'.
  ADDR4  f(QALL_U) >= .8 f(QALL) -> 'the address part carries it'; f(QALL_U) <= .5 f(QALL) -> 'the non-address part is needed'; otherwise 'shared'.
  TSUB2  (amendment before data) TRES = END with the START - END text-key part T (S842 settings: rank 16, layers 35-55, ten rows):
         t = (P_copy(TRES) - P_copy(END_B)) / (P_copy(START_B) - P_copy(END_B)) >= .25 with lower bound > 0 -> 'the text-key part restores START's
         answer' (S842 replication); otherwise 'not shown'.
  TREM   TREM = START with the END - START text-key part: r = (P_copy(START_B) - P_copy(TREM)) / (P_copy(START_B) - P_copy(END_B)) >= .25 with lower
         bound > 0 -> 'removing START's text-key part breaks START'; otherwise 'not shown'.  Consistency (reported with both): the late picture mass
         (layers 51 / 55, boundary and answer rows) and the layer 51 / 55 positional margins move in the same direction as the answers.
  QKPRED (amendment before data; user 11:48 'predictions from measured query changes') frozen logistic map from the layer 51 / 55 positional
         margins (B4 and answer rows) to answering k, fitted on S845 only (research/ces/outputs/s847_qk_map.json): share of (case x arm) cells
         with |predicted - observed P_copy| <= .15 >= .70 -> 'the late positional scores predict the answer shares under new interventions'
         (in-sample on S845: .73); per-arm errors are reported (the 59 / 63 edits act outside the map's inputs).
Descriptive: f(Q5155) against S845's BOTH_B (.57); g(NEC5155); per arm the layer 51 / 55 / 59 / 63 answer-row and B4 positional margins
(z[k] - max of the other blocks; 59 / 63 all-head means) and the copy-word logit margin.
Usage: python exp/s847_analyze.py [TAG]  ->  results/r3/s847_analysis.json"""
from __future__ import annotations

import json, sys
import numpy as np

V = "/home/storage2/ai26269345/vlm"
B = 4000
ARMS = ["START", "END", "A_BAND", "START_B", "END_B", "A_BAND_B", "QNOOP", "Q5155", "Q5963", "QALL", "QALL_U", "NEC", "NEC5155", "TRES", "TREM"]


def main(tag="s847_27b"):
    d = json.load(open(f"{V}/results/r3/s847_{tag}_close.json")); R = d["rows"]
    rng = np.random.default_rng(0)
    out = dict(tag=tag, t1=d["t1_menu"], key_rel_diff_max=d["key_rel_diff_max"], cases={})
    valid = {}
    for N, k, j in d["cases"]:
        rows = [r for r in R if r["N"] == N and r["k"] == k]
        if not rows:
            continue
        pc = {a: float(np.mean([r["land"][a] == k for r in rows])) for a in ARMS}
        ag = float(np.mean([r["land"]["QNOOP"] == r["land"]["END_B"] for r in rows]))
        p95 = float(np.percentile([max(abs(x - y) for x, y in zip(r["mlog"]["QNOOP"], r["mlog"]["END_B"])) for r in rows], 95))
        ok = bool(d["t1_menu"] >= .95 and ag >= .90 and p95 <= .5 and pc["START"] >= .90 and pc["START_B"] >= .90 and d["key_rel_diff_max"] == 0.0
                  and pc["A_BAND_B"] - pc["END_B"] >= .15)
        name = f"N{N}k{k}j{j}"
        xs = dict(END=float(np.mean([r["land"]["END_B"] == r["land"]["END"] for r in rows])),
                  A_BAND=float(np.mean([r["land"]["A_BAND_B"] == r["land"]["A_BAND"] for r in rows])))
        out["cases"][name] = dict(I0=dict(qnoop=ag, p95=p95, start=pc["START"], start_b=pc["START_B"], restore=pc["A_BAND_B"] - pc["END_B"], ok=ok),
                                  P_copy=pc, cross_stage_agreement=xs)
        if ok:
            valid[name] = (rows, k)
    out["valid_cases"] = sorted(valid)
    def frac(arm, ss, kind="f"):
        num = den = 0.0
        for nm, (rows, k) in ss.items():
            pa = np.mean([r["land"][arm] == k for r in rows]); pe = np.mean([r["land"]["END_B"] == k for r in rows])
            pb = np.mean([r["land"]["A_BAND_B"] == k for r in rows])
            num += (pa - pe) if kind == "f" else (pb - pa); den += pb - pe
        return float(num / den) if den > 0 else float("nan")
    def boot(f):
        v = []
        for _ in range(B):
            ss = {}
            for nm, (rows, k) in valid.items():
                ss[nm] = ([rows[i] for i in rng.integers(0, len(rows), len(rows))], k)
            v.append(f(ss))
        return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    res = {}
    for arm in ("Q5155", "Q5963", "QALL", "QALL_U"):
        res[arm] = dict(f=frac(arm, valid), ci=boot(lambda ss, a_=arm: frac(a_, ss)))
    for arm in ("NEC", "NEC5155"):
        res[arm] = dict(g=frac(arm, valid, "g"), ci=boot(lambda ss, a_=arm: frac(a_, ss, "g")))
    def tfrac(arm, ss, kind):
        num = den = 0.0
        for nm, (rows, k) in ss.items():
            pa = np.mean([r["land"][arm] == k for r in rows]); pe = np.mean([r["land"]["END_B"] == k for r in rows])
            ps = np.mean([r["land"]["START_B"] == k for r in rows])
            num += (pa - pe) if kind == "t" else (ps - pa); den += ps - pe
        return float(num / den) if den > 0 else float("nan")
    res["TRES"] = dict(t=tfrac("TRES", valid, "t"), ci=boot(lambda ss: tfrac("TRES", ss, "t")))
    res["TREM"] = dict(r=tfrac("TREM", valid, "r"), ci=boot(lambda ss: tfrac("TREM", ss, "r")))
    wd = lambda ss: frac("Q5963", ss) - frac("Q5155", ss)
    res["WHERE_diff"] = dict(value=wd(valid), ci=boot(wd))
    out["f"] = res
    nv = "no valid case"
    if not valid:
        out.update(PATH4=nv, NEC=nv, WHERE=nv, ADDR4=nv, TSUB2=nv, TREM=nv)
    else:
        fq, cq = res["QALL"]["f"], res["QALL"]["ci"]
        out["PATH4"] = ("the band's correction reaches the answer through the late full-attention queries (51-63)" if fq >= .80 and cq[0] > .60
                        else "the late queries do not carry all of it" if fq <= .60 else "partial")
        gn, cn = res["NEC"]["g"], res["NEC"]["ci"]
        out["NEC"] = ("the late queries are needed for the band's correction" if gn >= .70 and cn[0] > .50 else "other routes suffice" if gn <= .40 else "partial")
        w, cw = res["WHERE_diff"]["value"], res["WHERE_diff"]["ci"]
        out["WHERE"] = ("the last two layers carry more" if w >= .15 and cw[0] > 0 else "layers 51 / 55 carry more" if w <= -.15 and cw[1] < 0 else "both")
        fu = res["QALL_U"]["f"]
        out["ADDR4"] = ("the address part carries it" if fu >= .8 * fq else "the non-address part is needed" if fu <= .5 * fq else "shared")
        out["TSUB2"] = ("the text-key part restores START's answer" if res["TRES"]["t"] >= .25 and res["TRES"]["ci"][0] > 0 else "not shown")
        out["TREM"] = ("removing START's text-key part breaks START" if res["TREM"]["r"] >= .25 and res["TREM"]["ci"][0] > 0 else "not shown")
    # descriptive positional margins
    def zm(r, a, L, row):
        z = r["qk"][a][L][row]; k = r["k"]
        return z[k - 1] - max(z[i] for i in range(len(z)) if i != k - 1)
    allrows = [r for rows, _ in valid.values() for r in rows]
    out["margin"] = {f"{L}_{nm}": {a: float(np.mean([zm(r, a, L, row) for r in allrows])) for a in ARMS}
                     for L in [str(x) for x in d["late"]] for row, nm in ((5, "B4"), (9, "answer"))} if allrows else {}
    out["logit_margin"] = {a: float(np.mean([r["mlog"][a][r["k"] - 1] - max(x for i, x in enumerate(r["mlog"][a]) if i != r["k"] - 1) for r in allrows]))
                           for a in ARMS} if allrows else {}
    qm = json.load(open(f"{V}/research/ces/outputs/s847_qk_map.json")); w = np.array(qm["w"])
    def qkp(r, a):
        x = [1.0] + [zm(r, a, L, row) for L, row in (("51", 5), ("51", 9), ("55", 5), ("55", 9))]
        return float(1 / (1 + np.exp(-(np.array(x) @ w))))
    cellp = {}
    for nm, (rows, k) in valid.items():
        for a in ARMS:
            cellp[f"{nm}_{a}"] = dict(pred=float(np.mean([qkp(r, a) for r in rows])), obs=float(np.mean([r["land"][a] == k for r in rows])))
    calq = float(np.mean([abs(c["pred"] - c["obs"]) <= .15 for c in cellp.values()])) if cellp else float("nan")
    out["QKPRED"] = dict(calibration=calq, cells=cellp,
                         verdict=("the late positional scores predict the answer shares under new interventions" if cellp and calq >= .70 else "not shown"))
    def lmass(r, a):
        x = r["mass"].get(a, {})
        v = [np.mean([x[L][i] for i in (5, 9)]) for L in ("51", "55") if L in x]
        return float(np.mean(v)) if v else float("nan")
    out["mass_late"] = {a: float(np.nanmean([lmass(r, a) for r in allrows])) for a in ARMS} if allrows else {}
    fn = f"{V}/results/r3/s847_analysis.json" if tag == "s847_27b" else f"{V}/results/r3/s847_analysis_{tag}.json"
    json.dump(out, open(fn, "w"), indent=1)
    for nm, c in out["cases"].items():
        print(nm, c["I0"], "\n   P_copy", {a: round(x, 3) for a, x in c["P_copy"].items()})
    print("f", json.dumps(res))
    for k_ in ("PATH4", "NEC", "WHERE", "ADDR4", "TSUB2", "TREM"):
        print(k_, out.get(k_))
    for k_, v in out["margin"].items():
        print("margin", k_, {a: round(x, 2) for a, x in v.items()})
    print("mass_late", {a: round(x, 3) for a, x in out["mass_late"].items()})
    print("QKPRED", out["QKPRED"]["calibration"], out["QKPRED"]["verdict"], {c: (round(v["pred"], 2), round(v["obs"], 2)) for c, v in out["QKPRED"]["cells"].items()})


if __name__ == "__main__":
    main(*sys.argv[1:])
