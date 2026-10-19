"""S852 analysis (rules fixed in knowledge/DECISIONS.md before any S852 data; user 10-18 'continue' after the S851 report).  How is the wrong
late query formed?  Eight pictures shown and stated, k in {3, 4, 5} (copy k, target j* = 9 - k), early lookup pinned to p in {3, 4, 5, 6};
30 independent scenes, scene bootstrap 4,000 (seed 0).  M(x) = sign(x - p) for a picture answer x, 0 for a non-picture answer.
I0 (per k; all must pass): t1 >= .95; QNOOP_E = END in >= .90 of scenes and the 95th percentile of the largest menu-logit change <= .5;
  START(k) -> k in >= .90; donors START(p) -> p >= .90 pooled (>= .80 each); image keys identical (0.0); XDSELF (k = 4) gives E_EARLY's answer
  in >= .99 of runs with the 95th percentile of the largest menu-logit difference <= .05.
PC-A (the END write matters here): BANDF changes E_EARLY's answer in >= .25 of runs (pooled over k and p), lower bound > .15.
PC-B (the slope measure sees a step toward the copy when there is one): START's slope b_S (S_EARLY, as DIR) >= .25, lower bound > .10.
DIR  b = mean over scenes of (1/4) sum_p [M_E(p, k=5) - M_E(p, k=3)] / 2 (E_EARLY).  A step toward the copy predicts b > 0, toward the target
  b < 0, a k-independent step b = 0.  b >= .10 with lower bound > 0 -> 'the late step points toward the copied ordinal'; b <= -.10 with upper
  bound < 0 -> 'the late step points toward the end-relative target'; |b| < .10 with the interval inside (-.15, .15) -> 'k does not set the
  direction of the late step in the tested range'; otherwise 'partial'.  Needs I0 and PC-B ('no valid test' otherwise).
XD   on discordant runs (E_EARLY_p and E_EARLY_p' give different answers; p' = PAIR[p]; pooled over k and p), the share of XD_p answers equal
  to E_EARLY_p's (f_rec) and to E_EARLY_p''s (f_don), ratio of sums within each scene resample.  f_rec >= .60 with lower bound > .45 and
  f_don <= .25 -> 'the END write is anchor-independent: the receiving anchor sets the landing'; f_don >= .60 with lower bound > .45 and
  f_rec <= .25 -> 'the END write carries an anchor-specific destination'; otherwise 'the END write is partly anchor-specific'.  Needs I0 and
  PC-A.
XB   the same statistic for XB_p (the whole band of p'): donor -> 'the band's outputs as a whole carry the anchor'; receiver -> 'the anchor that
  sets the landing is carried outside the band'; otherwise 'partly'.
Reading (registered): XD receiver + XB donor -> 'the anchor reaches the late stage through the band's anchor-related output (including the
  layer-47 early read), and the END write is an anchor-independent step'; XD receiver + XB receiver -> 'the END write is an anchor-independent
  step and the anchor reaches the late stage outside the band'; XD donor -> 'the band computes an anchor-specific destination'; otherwise
  'partial'.
Descriptive: per-cell answer distributions; the user's split - identity cells (p = j*: staying, departures by direction, target logit margin)
  and the other cells (target answers, moves toward the target, target logit margin); one-step direction; XD agreement with BANDF; anchor
  dependence of the END write in the stored band outputs (cosine of D(p) and D(p'), |D(p') - D(p)| / |D(p)|, rows B1 / B4 / answer).
Usage: python exp/s852_analyze.py [TAG]  ->  results/r3/s852_analysis.json"""
from __future__ import annotations

import json, os, sys
import numpy as np

V = "/home/storage2/ai26269345/vlm"
B = 4000
KS = (3, 4, 5); PS = (3, 4, 5, 6); PAIR = {3: 5, 5: 3, 4: 6, 6: 4}


def sgn(x, p):
    return 0 if x < 1 else int(np.sign(x - p))


def main(tag="s852_27b"):
    d = json.load(open(f"{V}/results/r3/s852_{tag}_form.json")); R = d["rows"]
    by = {(r["si"], r["k"]): r for r in R}
    sis = sorted(s for s in {r["si"] for r in R} if all((s, k) in by for k in KS))
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(sis), (B, len(sis)))
    def ci(per):                                                            # per-scene values -> mean, 95% interval of the scene mean
        per = np.asarray(per, dtype=float); bs = per[idx].mean(1)
        return [float(per.mean()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]]
    def ci_ratio(num, den):                                                 # per-scene sums -> ratio of sums, 95% interval
        num = np.asarray(num, dtype=float); den = np.asarray(den, dtype=float)
        bs = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-9)
        return [float(num.sum() / max(den.sum(), 1e-9)), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]]
    mlog_diff = lambda r, a1, a2: max(abs(x - y) for x, y in zip(r["mlog"][a1], r["mlog"][a2]))
    out = dict(tag=tag, t1=d["t1_menu"], key_rel_diff_max=d["key_rel_diff_max"], n=len(sis), n_runs=d.get("n_runs"), I0={})
    ok_all = d["t1_menu"] >= .95 and d["key_rel_diff_max"] == 0.0
    for k in KS:
        rows = [by[(s, k)] for s in sis]
        qn = float(np.mean([r["land"]["QNOOP_E"] == r["land"]["END"] for r in rows]))
        p95 = float(np.percentile([mlog_diff(r, "QNOOP_E", "END") for r in rows], 95))
        st = float(np.mean([r["land"]["START"] == k for r in rows]))
        don = {p: float(np.mean([r["land"][f"START_p{p}"] == p for r in rows])) for p in PS}
        ok = bool(qn >= .90 and p95 <= .5 and st >= .90 and np.mean(list(don.values())) >= .90 and min(don.values()) >= .80)
        rec = dict(qnoop=qn, p95=p95, start_acc=st, donors=don)
        if k == 4:
            xs = float(np.mean([r["land"][f"XDSELF_p{p}"] == r["land"][f"E_EARLY_p{p}"] for r in rows for p in PS]))
            xp = float(np.percentile([mlog_diff(r, f"XDSELF_p{p}", f"E_EARLY_p{p}") for r in rows for p in PS], 95))
            xm = float(max(mlog_diff(r, f"XDSELF_p{p}", f"E_EARLY_p{p}") for r in rows for p in PS))
            rec.update(xdself_agree=xs, xdself_p95=xp, xdself_max=xm); ok = ok and xs >= .99 and xp <= .05
        rec["ok"] = bool(ok); out["I0"][str(k)] = rec; ok_all = ok_all and ok
    out["I0"]["all"] = bool(ok_all)
    # PC-A: BANDF changes E_EARLY's answer
    pa = ci([np.mean([by[(s, k)]["land"][f"BANDF_p{p}"] != by[(s, k)]["land"][f"E_EARLY_p{p}"] for k in KS for p in PS]) for s in sis])
    out["PC_A"] = dict(changed=pa, ok=bool(pa[0] >= .25 and pa[1][0] > .15))
    # slope b (DIR) for an arm
    def slope_per_scene(arm):
        return [np.mean([(sgn(by[(s, 5)]["land"][f"{arm}_p{p}"], p) - sgn(by[(s, 3)]["land"][f"{arm}_p{p}"], p)) / 2 for p in PS]) for s in sis]
    bS = ci(slope_per_scene("S_EARLY")); out["PC_B"] = dict(b_S=bS, ok=bool(bS[0] >= .25 and bS[1][0] > .10))
    bE = ci(slope_per_scene("E_EARLY"))
    nvt = "no valid test"
    if not (ok_all and out["PC_B"]["ok"]):
        dv = nvt
    elif bE[0] >= .10 and bE[1][0] > 0:
        dv = "the late step points toward the copied ordinal"
    elif bE[0] <= -.10 and bE[1][1] < 0:
        dv = "the late step points toward the end-relative target"
    elif abs(bE[0]) < .10 and bE[1][0] > -.15 and bE[1][1] < .15:
        dv = "k does not set the direction of the late step in the tested range"
    else:
        dv = "partial"
    out["DIR"] = dict(b=bE, per_p={str(p): ci([(sgn(by[(s, 5)]["land"][f"E_EARLY_p{p}"], p) - sgn(by[(s, 3)]["land"][f"E_EARLY_p{p}"], p)) / 2
                                               for s in sis]) for p in PS}, verdict=dv)
    # XD / XB: which natural answer does the patched run give on discordant runs?
    def follow(arm):
        n_, r_, d_, b_ = [], [], [], []
        for s in sis:
            n = r = dd = 0
            for k in KS:
                L = by[(s, k)]["land"]
                for p in PS:
                    a_rec, a_don = L[f"E_EARLY_p{p}"], L[f"E_EARLY_p{PAIR[p]}"]
                    if a_rec == a_don:
                        continue
                    n += 1; r += int(L[f"{arm}_p{p}"] == a_rec); dd += int(L[f"{arm}_p{p}"] == a_don)
            n_.append(n); r_.append(r); d_.append(dd)
        return dict(n_discordant=int(sum(n_)), f_rec=ci_ratio(r_, n_), f_don=ci_ratio(d_, n_))
    def verdict(f, rec_txt, don_txt, mix_txt):
        if not (ok_all and out["PC_A"]["ok"]):
            return nvt
        if f["f_rec"][0] >= .60 and f["f_rec"][1][0] > .45 and f["f_don"][0] <= .25:
            return rec_txt
        if f["f_don"][0] >= .60 and f["f_don"][1][0] > .45 and f["f_rec"][0] <= .25:
            return don_txt
        return mix_txt
    fx, fb = follow("XD"), follow("XB")
    fx["verdict"] = verdict(fx, "the END write is anchor-independent: the receiving anchor sets the landing",
                            "the END write carries an anchor-specific destination", "the END write is partly anchor-specific")
    fb["verdict"] = verdict(fb, "the anchor that sets the landing is carried outside the band", "the band's outputs as a whole carry the anchor",
                            "the band's outputs partly carry the anchor")
    out["XD"], out["XB"] = fx, fb
    if fx["verdict"].startswith("the END write is anchor-independent") and fb["verdict"].startswith("the band's outputs as a whole"):
        rd = ("the anchor reaches the late stage through the band's anchor-related output (including the layer-47 early read), and the END write "
              "is an anchor-independent step")
    elif fx["verdict"].startswith("the END write is anchor-independent") and fb["verdict"].startswith("the anchor that sets the landing"):
        rd = "the END write is an anchor-independent step and the anchor reaches the late stage outside the band"
    elif fx["verdict"].startswith("the END write carries"):
        rd = "the band computes an anchor-specific destination"
    elif nvt in (fx["verdict"], fb["verdict"]):
        rd = nvt
    else:
        rd = "partial"
    out["READING"] = rd
    # descriptive
    def lmar(r, arm, x):
        lg = np.array(r["mlog"][arm], dtype=float); ix = r["words"].index(r["labs"][x - 1])
        return float(lg[ix] - np.max(np.delete(lg, ix)))
    desc = {"dist": {}, "identity": {}, "nonidentity": {}, "onestep": {}, "xd_vs_bandf": {}}
    arms = ("E_EARLY", "S_EARLY", "BANDF", "XD", "XB")
    for k in KS:
        for p in PS:
            desc["dist"][f"k{k}_p{p}"] = {a: {str(x): float(np.mean([by[(s, k)]["land"][f"{a}_p{p}"] == x for s in sis])) for x in range(0, 9)
                                              if any(by[(s, k)]["land"][f"{a}_p{p}"] == x for s in sis)} for a in arms}
    idc = [(k, 9 - k) for k in KS if 9 - k in PS]; nonc = [(k, p) for k in KS for p in PS if p != 9 - k]
    for a in arms:
        desc["identity"][a] = dict(
            stay=ci([np.mean([by[(s, k)]["land"][f"{a}_p{p}"] == p for k, p in idc]) for s in sis]),
            back=ci([np.mean([0 < by[(s, k)]["land"][f"{a}_p{p}"] < p for k, p in idc]) for s in sis]),
            forward=ci([np.mean([by[(s, k)]["land"][f"{a}_p{p}"] > p for k, p in idc]) for s in sis]),
            lm_target=ci([np.mean([lmar(by[(s, k)], f"{a}_p{p}", p) for k, p in idc]) for s in sis]))
        desc["nonidentity"][a] = dict(
            target=ci([np.mean([by[(s, k)]["land"][f"{a}_p{p}"] == 9 - k for k, p in nonc]) for s in sis]),
            stay=ci([np.mean([by[(s, k)]["land"][f"{a}_p{p}"] == p for k, p in nonc]) for s in sis]),
            toward=ci([np.mean([sgn(by[(s, k)]["land"][f"{a}_p{p}"], p) == int(np.sign(9 - k - p)) and by[(s, k)]["land"][f"{a}_p{p}"] != p
                                for k, p in nonc]) for s in sis]),
            lm_target=ci([np.mean([lmar(by[(s, k)], f"{a}_p{p}", 9 - k) for k, p in nonc]) for s in sis]))
        desc["onestep"][a] = {f"k{k}_p{p}": float(np.mean([(by[(s, k)]["land"][f"{a}_p{p}"] == p + 1) - (by[(s, k)]["land"][f"{a}_p{p}"] == p - 1)
                                                           for s in sis])) for k in KS for p in PS}
    desc["xd_vs_bandf"] = dict(
        xd_eq_bandf=ci([np.mean([by[(s, k)]["land"][f"XD_p{p}"] == by[(s, k)]["land"][f"BANDF_p{p}"] for k in KS for p in PS]) for s in sis]),
        xd_eq_e=ci([np.mean([by[(s, k)]["land"][f"XD_p{p}"] == by[(s, k)]["land"][f"E_EARLY_p{p}"] for k in KS for p in PS]) for s in sis]),
        xb_eq_e=ci([np.mean([by[(s, k)]["land"][f"XB_p{p}"] == by[(s, k)]["land"][f"E_EARLY_p{p}"] for k in KS for p in PS]) for s in sis]),
        bandf_eq_e=ci([np.mean([by[(s, k)]["land"][f"BANDF_p{p}"] == by[(s, k)]["land"][f"E_EARLY_p{p}"] for k in KS for p in PS]) for s in sis]))
    npzf = f"{V}/results/r3/s852_{tag}_form.npz"
    if os.path.exists(npzf):
        Z = np.load(npzf); rowsn = ("B1", "B4", "answer"); an = {}
        for ri, rn in enumerate(rowsn):
            cs, rel = [], []
            for s in sis:
                for k in KS:
                    Dp = {p: (Z[f"k{k}_s{s}_E_EARLY_p{p}_comp"][:, ri].astype(np.float64) - Z[f"k{k}_s{s}_S_EARLY_p{p}_comp"][:, ri].astype(np.float64)).sum(0)
                          for p in PS}
                    for p in PS:
                        a_, b_ = Dp[p], Dp[PAIR[p]]
                        cs.append(float(a_ @ b_ / (np.linalg.norm(a_) * np.linalg.norm(b_) + 1e-12)))
                        rel.append(float(np.linalg.norm(b_ - a_) / (np.linalg.norm(a_) + 1e-12)))
            an[rn] = dict(cos_median=float(np.median(cs)), rel_diff_median=float(np.median(rel)))
        desc["write_anchor_dependence"] = an
    out["desc"] = desc
    fn = f"{V}/results/r3/s852_analysis.json" if tag == "s852_27b" else f"{V}/results/r3/s852_analysis_{tag}.json"
    json.dump(out, open(fn, "w"), indent=1)
    print("I0", json.dumps(out["I0"])); print("PC_A", out["PC_A"]); print("PC_B", out["PC_B"])
    print("DIR", json.dumps(out["DIR"])); print("XD", json.dumps(out["XD"])); print("XB", json.dumps(out["XB"])); print("READING", rd)
    print("identity", json.dumps(desc["identity"])); print("nonidentity", json.dumps(desc["nonidentity"]))
    print("xd_vs_bandf", json.dumps(desc["xd_vs_bandf"])); print("write_anchor_dependence", json.dumps(desc.get("write_anchor_dependence")))


if __name__ == "__main__":
    main(*sys.argv[1:])
