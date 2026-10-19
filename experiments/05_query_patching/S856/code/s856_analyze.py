"""S856 analysis (rules fixed in knowledge/DECISIONS.md before any S856 data; P1, user 10-18).  Does the visual-prefix-conditioned late query carry
the actual-count difference?  6 vs 8 pictures (shared pictures 1-6), k {3, 5}, pinned p {3, 4}; 30 independent scenes; scene bootstrap 4,000
(seed 0); k and p pooled; ratios of sums within each scene resample.  Targets j6 = 7 - k, j8 = 9 - k (both among pictures 1-6).
  g  = l(w_j8) - l(w_j6)   menu-logit difference between the two targets' words (user's measure)
  m  = z_j8 - z_j6         the same two pictures' late block scores (rows B1, B4, answer; layers 51-63 averaged)
  natural count effect G = x(ENc, 8) - x(ENc, 6) (x = g or m; ENc = the same-batch float32 late recompute of the recipient)
  T_6<-8 = sum [x(QX in X6) - x(ENc, 6)] / sum G      (the 8-picture late query moves the 6-picture run toward the 8-picture state)
  T_8<-6 = sum [x(ENc, 8) - x(QX in X8)] / sum G      (the 6-picture late query moves the 8-picture run back)
I0 (all): t1 >= .95; image keys identical within each form and count (0.0); E vs EN (late float32 recompute, same batch) answers >= .90 with the
  95th percentile of the largest menu-logit change <= .5; START(k) -> k >= .90; pin donors START(3), START(4) -> p >= .90; PIN (layers 43 / 47,
  rows B1 / B4 maximum = p) >= .95; SELF = ENc in form A (answers >= .97, 95th percentile of the largest menu-logit difference <= .5 - amended
  before data after the GPU smoke: donor queries are captured in an 8-run batch and injected into a 20-run batch, and with 6 pictures the two
  batch shapes differ numerically (smoke: answers 1.0, p95 .41 for X6, 0.0 for X8); the observed floor is reported).
QX (primary; form A, whole query, all ten read rows, layers 51-63):
  T_6<-8 >= .70 and T_8<-6 >= .70 on g, each with lower bound > .50 -> 'the late query carries the actual-count difference (both directions)';
  else if T on m is >= .70 (lower bound > .50) in both directions while T on g is < .40 in both -> 'the late address moves but the final answer
  does not (readout factors)'; else if T on g and on m are <= .20 (upper bound < .35) in both directions -> 'the late query does not carry it
  (keys / values, gates or other residual paths)'; otherwise 'partial' (both directions reported; the user's table).
The same rule for QX in form NC (no count sentence) and, as registered secondary arms, QXA (rows B1, B4, answer only) and QXU (address part U only).
Descriptive: answers (on runs whose two natural answers differ, the share answering the donor count's answer); per head at the late layers (rows
B1, B4, answer): attention mass on pictures 7-8 (from per-head block scores and picture mass), gated output norm and its picture part.
Usage: python exp/s856_analyze.py [TAG]  ->  results/r3/s856_analysis.json"""
from __future__ import annotations

import json, os, sys
import numpy as np

V = "/home/storage2/ai26269345/vlm"
B = 4000
KS = (3, 5); PS = (3, 4)


def main(tag="s856_27b"):
    d = json.load(open(f"{V}/results/r3/s856_{tag}_qswap.json"))
    Zp = f"{V}/results/r3/s856_{tag}_qswap.npz"; Z = np.load(Zp) if os.path.exists(Zp) else None
    LATE = [str(l) for l in d["late"]]; PINL = [str(l) for l in d["early"][-2:]]
    by = {r["si"]: r for r in d["rows"]}; sis = sorted(by); n = len(sis)
    rng = np.random.default_rng(0); idx = rng.integers(0, n, (B, n))
    mld = lambda r, a1, a2: max(abs(x - y) for x, y in zip(r["mlog"][a1], r["mlog"][a2]))
    def same(pairs):
        ag = float(np.mean([by[s]["land"][a1] == by[s]["land"][a2] for s in sis for a1, a2 in pairs]))
        p95 = float(np.percentile([mld(by[s], a1, a2) for s in sis for a1, a2 in pairs], 95))
        return [ag, p95]
    out = dict(tag=tag, t1=d["t1_menu"], key_rel_diff_max=d["key_rel_diff_max"], n=n, n_runs=d.get("n_runs"))
    i0 = dict(t1=d["t1_menu"] >= .95, keys=d["key_rel_diff_max"] == 0.0)
    for f in d["forms"]:
        for na in (6, 8):
            pre = f"{f}_"; suf = f"_a{na}"
            qn = same([(f"{pre}EN_k{k}p{p}{suf}", f"{pre}E_k{k}p{p}{suf}") for k in KS for p in PS]); i0[f"qnoop_{f}{na}"] = dict(v=qn, ok=bool(qn[0] >= .90 and qn[1] <= .5))
            st = {k: float(np.mean([by[s]["land"][f"{pre}START_q{k}{suf}"] == k for s in sis])) for k in KS}; i0[f"start_{f}{na}"] = dict(v=st, ok=bool(min(st.values()) >= .90))
            dn = {q: float(np.mean([by[s]["land"][f"{pre}START_q{q}{suf}"] == q for s in sis])) for q in PS}; i0[f"donors_{f}{na}"] = dict(v=dn, ok=bool(min(dn.values()) >= .90))
            pin = float(np.mean([int(np.argmax(by[s]["qk"][f"{pre}E_k{k}p{p}{suf}"][L][ri]) + 1 == p) for s in sis for k in KS for p in PS for L in PINL for ri in (2, 5)]))
            i0[f"pin_{f}{na}"] = dict(v=pin, ok=bool(pin >= .95))
            if f == "A":
                sf = same([(f"{pre}SELF_k{k}p{p}{suf}", f"{pre}ENc_k{k}p{p}{suf}") for k in KS for p in PS])
                i0[f"self_{na}"] = dict(v=sf, ok=bool(sf[0] >= .97 and sf[1] <= .5))     # amended before data: cross-batch donor numerics (see docstring)
    ok_all = bool(i0["t1"] and i0["keys"] and all(v["ok"] for k, v in i0.items() if isinstance(v, dict)))
    out["I0"] = dict(i0, all=ok_all); nvt = "no valid test"
    gfun = lambda s, nm, k: float(by[s]["mlog"][nm][9 - k - 1] - by[s]["mlog"][nm][7 - k - 1])
    def mfun(s, nm, k):
        v_ = [by[s]["qk"][nm][L][ri][9 - k - 1] - by[s]["qk"][nm][L][ri][7 - k - 1] for L in LATE for ri in (2, 5, 9)]
        return float(np.mean(v_))
    def T(f, arm, fun):
        n68, n86, den = [], [], []
        for s in sis:
            a_ = b_ = c_ = 0.0
            for k in KS:
                for p in PS:
                    c = f"k{k}p{p}"
                    e6, e8 = fun(s, f"{f}_ENc_{c}_a6", k), fun(s, f"{f}_ENc_{c}_a8", k)
                    a_ += fun(s, f"{f}_{arm}_{c}_a6", k) - e6; b_ += e8 - fun(s, f"{f}_{arm}_{c}_a8", k); c_ += e8 - e6
            n68.append(a_); n86.append(b_); den.append(c_)
        n68, n86, den = map(np.asarray, (n68, n86, den))
        r = lambda nn: [float(nn.sum() / den.sum()), [float(np.percentile(nn[idx].sum(1) / den[idx].sum(1), 2.5)), float(np.percentile(nn[idx].sum(1) / den[idx].sum(1), 97.5))]]
        return dict(T_6from8=r(n68), T_8from6=r(n86), G_mean=float(den.mean() / 4))
    def verdict(tg, tm):
        if not ok_all:
            return nvt
        ok7 = lambda t: t[0] >= .70 and t[1][0] > .50
        lo2 = lambda t: t[0] <= .20 and t[1][1] < .35
        if ok7(tg["T_6from8"]) and ok7(tg["T_8from6"]):
            return "the late query carries the actual-count difference (both directions)"
        if ok7(tm["T_6from8"]) and ok7(tm["T_8from6"]) and tg["T_6from8"][0] < .40 and tg["T_8from6"][0] < .40:
            return "the late address moves but the final answer does not (readout factors)"
        if lo2(tg["T_6from8"]) and lo2(tg["T_8from6"]) and lo2(tm["T_6from8"]) and lo2(tm["T_8from6"]):
            return "the late query does not carry it (keys / values, gates or other residual paths)"
        return "partial"
    res = {}
    for f, arm in (("A", "QX"), ("NC", "QX"), ("A", "QXA"), ("A", "QXU")):
        tg, tm = T(f, arm, gfun), T(f, arm, mfun)
        res[f"{f}_{arm}"] = dict(g=tg, m=tm, verdict=verdict(tg, tm))
    out["RESULTS"] = res; out["QX"] = res["A_QX"]["verdict"]
    ans = {}
    for f, arm in (("A", "QX"), ("NC", "QX"), ("A", "QXA"), ("A", "QXU")):
        for na in (6, 8):
            other = 14 - na; n_ = to_ = st_ = 0
            for s in sis:
                for k in KS:
                    for p in PS:
                        c = f"k{k}p{p}"; L_ = by[s]["land"]; a_own, a_oth = L_[f"{f}_ENc_{c}_a{na}"], L_[f"{f}_ENc_{c}_a{other}"]
                        if a_own == a_oth:
                            continue
                        x = L_[f"{f}_{arm}_{c}_a{na}"]; n_ += 1; to_ += int(x == a_oth); st_ += int(x == a_own)
            ans[f"{f}_{arm}_in{na}"] = dict(n=n_, to_donor_answer=to_ / max(n_, 1), keeps_own=st_ / max(n_, 1))
    out["answers"] = ans
    if Z is not None:
        hd = {}
        for f in ("A",):
            for arm in ("ENc", "QX"):
                for na in (6, 8):
                    m78, tot, pic, gate = [], [], [], []
                    for s in sis:
                        for k in KS:
                            for p in PS:
                                key = f"s{s}_{f}_{arm}_k{k}p{p}_a{na}"
                                cH = Z[key + "_capH"].astype(np.float64); iH = Z[key + "_imgH"].astype(np.float64)    # [L, 3, H, nb], [L, 3, H]
                                sm = np.exp(cH - cH.max(-1, keepdims=True)); sm = sm / sm.sum(-1, keepdims=True)
                                mass_b = sm * iH[..., None]                                                          # attention mass per picture
                                if na == 8:
                                    m78.append(float(mass_b[..., 6:].sum(-1).mean()))
                                ho = Z[key + "_hout"].astype(np.float64)                                             # [L, 3, r, H]
                                tot.append(float(ho[:, 0][:, [2, 5, 9]].mean())); pic.append(float(ho[:, 1][:, [2, 5, 9]].mean())); gate.append(float(ho[:, 2][:, [2, 5, 9]].mean()))
                    hd[f"{arm}_in{na}"] = dict(mass_pics_7_8=(float(np.mean(m78)) if m78 else None), gated_out_norm=float(np.mean(tot)),
                                               gated_out_picture_part=float(np.mean(pic)), mean_gate=float(np.mean(gate)))
        out["heads"] = hd
    fn = f"{V}/results/r3/s856_analysis.json" if tag == "s856_27b" else f"{V}/results/r3/s856_analysis_{tag}.json"
    json.dump(out, open(fn, "w"), indent=1)
    print("I0", json.dumps(out["I0"])[:1600])
    for k_, v_ in res.items():
        print(k_, v_["verdict"], "| g", v_["g"], "| m", v_["m"])
    print("answers", json.dumps(ans)); print("heads", json.dumps(out.get("heads")))


if __name__ == "__main__":
    main(*sys.argv[1:])
