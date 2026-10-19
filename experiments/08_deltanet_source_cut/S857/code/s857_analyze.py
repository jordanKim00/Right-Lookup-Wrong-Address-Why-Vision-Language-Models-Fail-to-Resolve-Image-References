"""S857 analysis (rules fixed in knowledge/DECISIONS.md before any S857 data; P2, user 10-18).  How do pictures 7-8 reach the late query?
30 scenes; scene bootstrap 4,000 (seed 0); k {3, 5} x p {3, 4} pooled; ratios of sums within each scene resample.  Targets j6 = 7 - k (6
pictures) and j8 = 9 - k (8 pictures), both among the shared pictures 1-6.  Measures (as S856):
  g    = l(w_j8) - l(w_j6)                     menu-logit difference of the two targets' words (answer side)
  m51  = z_j8 - z_j6 at the first late layer   block scores of rows B1, B4, answer at layer LATE[0] (51) - pictures 1-6 have the same keys in
                                               every arm, so m51 changes only through the layer-51 query (its QK-effective content)
  mL   = the same averaged over the late layers 51-63
  m51A = m51 of the answer row only (V4: a row that neither restore arm patches)
  f(arm) = sum [x(arm) - x(N6)] / sum [x(N8) - x(N6)]   share of the natural 6 -> 8 picture effect present in the arm.
I0 (all): t1 >= .95; pictures 1-6 block keys identical across the 8-picture configurations (0.0); LEAK: CAsw = CA (END and START runs: answers
  >= .99, 95th percentile of the largest menu-logit difference <= .05); QSELF = N8 (same); WINDOW: CR = N8 in the layer-51 block scores of
  pictures 1-6 (every read row; largest difference <= 1e-6); INSTR: N8 (cut instrument on) vs N8std (natural kernels) answers >= .90 (the
  95th percentile of the largest menu-logit difference is reported - batch shape and kernel differ, S855 / S856);
  donors START(q) -> q >= .90 (N8 and N6); PIN (layers 43 / 47, rows B1 / B4 maximum over the pinned span, pictures 1-6, = p) >= .95 in N8
  (the share whose maximum over all 8 pictures is p is reported); POS: the natural effect on g is
  positive (lower bound of mean G > 0); COMPLETE (positive control of the cut): f_g(CA) and f_m51(CA) both within [-.15, .15].
V1 (needs I0; on g) - does the effect reach the answer through what the text rows carry by layer 51, or through a direct late read of
  pictures 7-8?  QP = f(CR) (late layers cut), DR = f(CQ) (cut before the late layers):  QP >= .70 (lower > .50) and DR <= .30 (upper < .50)
  -> 'through the text-row state at the first late layer'; DR >= .70 (lower > .50) and QP <= .30 (upper < .50) -> 'through a direct late
  read of pictures 7-8'; both >= .70 (lower > .50) -> 'either path suffices'; both <= .30 (upper < .50) -> 'both paths needed'; else 'mixed'.
V2 (needs I0 and f_m51(CQ) <= .15 with upper < .30; on m51) - which route carries pictures 7-8 into the layer-51 query?  FAo = f(CQD) (only
  full attention open before 51), DNo = f(CQF) (only DeltaNet open):  FAo >= .70 (lower > .50) and DNo <= .30 (upper < .50) -> 'full
  attention'; the reverse -> 'DeltaNet'; both >= .70 (lower > .50) -> 'either route suffices'; both <= .30 (upper < .50) -> 'both routes
  needed'; else 'mixed'.
V3 (needs I0; on g) - the late query itself: S = f(RQ) (cut everything, restore only the late queries), N = 1 - f(KQ) (natural, late
  queries from CA):  S >= .70 (lower > .50) and N >= .70 (lower > .50) -> 'the late query is sufficient and necessary'; S >= .70 (lower >
  .50) and N <= .30 (upper < .50) -> 'sufficient, not necessary'; N >= .70 (lower > .50) and S <= .30 (upper < .50) -> 'necessary, not
  sufficient'; both <= .30 (upper < .50) -> 'neither'; else 'partial'.
V4 (needs I0; on m51A) - the writer input rows: RW = f(RW) (B1, B4 at layer WL restored into CA), RT = f(RT) (the question tail restored);
  each 'carries' (>= .50, lower > .30), 'little' (<= .20, upper < .35) or 'partial'; 'H-W' if RW carries and RT little, 'H-Q' the reverse,
  'neither row group at the writer input' if both little, 'H-Mixed' if both >= .20 with lower > .10, else 'unresolved'.
V5 (needs I0; on m51) - when: O1 = f(O1) (only layers 0-34 open), O2 = f(O2) (only 35-50 open):  O1 >= .70 (lower > .50) and O2 <= .30
  (upper < .50) -> 'enters before layer 35'; the reverse -> 'enters in 35-50'; both >= .70 -> 'either window suffices'; both <= .30 ->
  'needs both windows'; else 'mixed'.
V6 (needs I0; on g) - the source content, per arm G78 (grey), D78 (copies of pictures 1 and 5), SW (7 and 8 swapped): >= .70 (lower > .50)
  'keeps the effect'; <= .30 (upper < .50) 'removes it'; else 'partial'.
INT (added 10-18 19:40 during data collection, before any S857 result was seen - user request; descriptive reading, V1-V6 unchanged): the
  2 x 2 of full-attention cut x DeltaNet cut in layers 0-50 (late layers open), Y00 = N8, Y10 = CQF (FA cut), Y01 = CQD (DeltaNet cut),
  Y11 = CQ (both): I = Y11 - Y10 - Y01 + Y00 on g (answer logit gap; also m51), per scene summed over k, p, and as a share of the natural
  effect sum G.  Reading: I / G >= .30 (lower > .10) 'AND-like - each route needed (serial or joint), not independent carriers'; <= -.30
  (upper < -.10) 'OR-like - redundant routes'; |I / G| <= .15 with the interval inside +-.30 'additive (independent partial routes)'; else
  'unclear'.
Descriptive: every f on g, m51 and mL; answers (on runs whose N8 and N6 answers differ, the share answering N8's answer); per head (late
layers, rows B1, B4, answer): attention mass on pictures 7-8, gated output norm and its picture part, mean gate.
Usage: python exp/s857_analyze.py [TAG]  ->  results/r3/s857_analysis.json"""
from __future__ import annotations

import json, os, sys
import numpy as np

V = "/home/storage2/ai26269345/vlm"
B = 4000
KS = (3, 5); PS = (3, 4)


def main(tag="s857_27b"):
    d = json.load(open(f"{V}/results/r3/s857_{tag}_cut.json"))
    Zp = f"{V}/results/r3/s857_{tag}_cut.npz"; Z = np.load(Zp) if os.path.exists(Zp) else None
    LATE = [str(l) for l in d["late"]]; L51 = LATE[0]; PINL = [str(l) for l in d["early"][-2:]]
    by = {r["si"]: r for r in d["rows"]}; sis = sorted(by); n = len(sis)
    rng = np.random.default_rng(0); idx = rng.integers(0, n, (B, n))
    mld = lambda r, a1, a2: max(abs(x - y) for x, y in zip(r["mlog"][a1], r["mlog"][a2]))
    def same(pairs):
        ag = float(np.mean([by[s]["land"][a1] == by[s]["land"][a2] for s in sis for a1, a2 in pairs]))
        p95 = float(np.percentile([mld(by[s], a1, a2) for s in sis for a1, a2 in pairs], 95))
        return [ag, p95]
    C = [(k, p) for k in KS for p in PS]
    out = dict(tag=tag, t1=d["t1_menu"], key_rel_diff_max=d["key_rel_diff_max"], n=n, n_runs=d.get("n_runs"), calls=d.get("calls"))
    i0 = dict(t1=d["t1_menu"] >= .95, keys=d["key_rel_diff_max"] == 0.0)
    lk = same([(f"CAsw_k{k}p{p}", f"CA_k{k}p{p}") for k, p in C] + [(f"CAsw_START_q{q}", f"CA_START_q{q}") for q in PS])
    i0["leak"] = dict(v=lk, ok=bool(lk[0] >= .99 and lk[1] <= .05))
    qs_ = same([(f"QSELF_k{k}p{p}", f"N8_k{k}p{p}") for k, p in C]); i0["qself"] = dict(v=qs_, ok=bool(qs_[0] >= .99 and qs_[1] <= .05))
    wd = max(abs(a_ - b_) for s in sis for k, p in C for ri in range(10)
             for a_, b_ in zip(by[s]["qk"][f"CR_k{k}p{p}"][L51][ri][:6], by[s]["qk"][f"N8_k{k}p{p}"][L51][ri][:6]))
    i0["window"] = dict(v=float(wd), ok=bool(wd <= 1e-6))
    ins = same([(f"N8_k{k}p{p}", f"N8std_k{k}p{p}") for k, p in C]); i0["instr"] = dict(v=ins, ok=bool(ins[0] >= .90))
    dn = {f"{c}_{q}": float(np.mean([by[s]["land"][f"{c}_START_q{q}"] == q for s in sis])) for c in ("N8", "N6") for q in PS}
    i0["donors"] = dict(v=dn, ok=bool(min(dn.values()) >= .90))
    pin = float(np.mean([int(np.argmax(by[s]["qk"][f"N8_k{k}p{p}"][L][ri][:6]) + 1 == p) for s in sis for k, p in C for L in PINL for ri in (2, 5)]))
    pin8 = float(np.mean([int(np.argmax(by[s]["qk"][f"N8_k{k}p{p}"][L][ri]) + 1 == p) for s in sis for k, p in C for L in PINL for ri in (2, 5)]))
    i0["pin"] = dict(v=pin, all8=pin8, ok=bool(pin >= .95))
    gfun = lambda s, nm, k: float(by[s]["mlog"][nm][9 - k - 1] - by[s]["mlog"][nm][7 - k - 1])
    def mfun(layers, rows=(2, 5, 9)):
        def f(s, nm, k):
            return float(np.mean([by[s]["qk"][nm][L][ri][9 - k - 1] - by[s]["qk"][nm][L][ri][7 - k - 1] for L in layers for ri in rows]))
        return f
    FUN = dict(g=gfun, m51=mfun([L51]), mL=mfun(LATE), m51A=mfun([L51], (9,)))
    def share(arm, x):
        fun = FUN[x]; num, den = [], []
        for s in sis:
            a_ = b_ = 0.0
            for k, p in C:
                e6 = fun(s, f"N6_k{k}p{p}", k)
                a_ += fun(s, f"{arm}_k{k}p{p}", k) - e6; b_ += fun(s, f"N8_k{k}p{p}", k) - e6
            num.append(a_); den.append(b_)
        num, den = np.asarray(num), np.asarray(den); bs = num[idx].sum(1) / den[idx].sum(1)
        return [float(num.sum() / den.sum()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]]
    G = np.asarray([np.mean([gfun(s, f"N8_k{k}p{p}", k) - gfun(s, f"N6_k{k}p{p}", k) for k, p in C]) for s in sis])
    gb = G[idx].mean(1)
    i0["pos"] = dict(v=[float(G.mean()), [float(np.percentile(gb, 2.5)), float(np.percentile(gb, 97.5))]], ok=bool(np.percentile(gb, 2.5) > 0))
    arms = [x for x in d["arms"] if x not in ("N8",)]
    F = {arm: {x: share(arm, x) for x in FUN} for arm in arms}
    out["shares"] = F
    i0["complete"] = dict(v=[F["CA"]["g"][0], F["CA"]["m51"][0]], ok=bool(abs(F["CA"]["g"][0]) <= .15 and abs(F["CA"]["m51"][0]) <= .15))
    ok_all = bool(i0["t1"] and i0["keys"] and all(v_["ok"] for v_ in i0.values() if isinstance(v_, dict)))
    out["I0"] = dict(i0, all=ok_all); nvt = "no valid test"
    hi7 = lambda t: t[0] >= .70 and t[1][0] > .50
    lo3 = lambda t: t[0] <= .30 and t[1][1] < .50
    def two(a, b_, la, lb, both_hi, both_lo):
        if hi7(a) and lo3(b_):
            return la
        if hi7(b_) and lo3(a):
            return lb
        if hi7(a) and hi7(b_):
            return both_hi
        if lo3(a) and lo3(b_):
            return both_lo
        return "mixed"
    v1 = nvt if not ok_all else two(F["CR"]["g"], F["CQ"]["g"], "through the text-row state at the first late layer",
                                     "through a direct late read of pictures 7-8", "either path suffices", "both paths needed")
    cq = F["CQ"]["m51"]; pc2 = bool(cq[0] <= .15 and cq[1][1] < .30)
    v2 = nvt if not (ok_all and pc2) else two(F["CQD"]["m51"], F["CQF"]["m51"], "full attention", "DeltaNet", "either route suffices", "both routes needed")
    S_ = F["RQ"]["g"]; f_kq = F["KQ"]["g"]; N_ = [1 - f_kq[0], [1 - f_kq[1][1], 1 - f_kq[1][0]]]
    if not ok_all:
        v3 = nvt
    elif hi7(S_) and hi7(N_):
        v3 = "the late query is sufficient and necessary"
    elif hi7(S_) and lo3(N_):
        v3 = "sufficient, not necessary"
    elif hi7(N_) and lo3(S_):
        v3 = "necessary, not sufficient"
    elif lo3(S_) and lo3(N_):
        v3 = "neither"
    else:
        v3 = "partial"
    def grp(t):
        return "carries" if (t[0] >= .50 and t[1][0] > .30) else "little" if (t[0] <= .20 and t[1][1] < .35) else "partial"
    rw, rt = F["RW"]["m51A"], F["RT"]["m51A"]
    if not ok_all:
        v4 = nvt
    elif grp(rw) == "carries" and grp(rt) == "little":
        v4 = "H-W"
    elif grp(rt) == "carries" and grp(rw) == "little":
        v4 = "H-Q"
    elif grp(rw) == "little" and grp(rt) == "little":
        v4 = "neither row group at the writer input"
    elif rw[0] >= .20 and rw[1][0] > .10 and rt[0] >= .20 and rt[1][0] > .10:
        v4 = "H-Mixed"
    else:
        v4 = "unresolved"
    v5 = nvt if not ok_all else two(F["O1"]["m51"], F["O2"]["m51"], "enters before layer 35", "enters in 35-50", "either window suffices", "needs both windows")
    v6 = {}
    for arm in ("G78", "D78", "SW"):
        t = F[arm]["g"]
        v6[arm] = nvt if not ok_all else "keeps the effect" if hi7(t) else "removes it" if lo3(t) else "partial"
    def inter(x):
        fun = FUN[x]; I_, Gs = [], []
        for s in sis:
            a_ = g_ = 0.0
            for k, p in C:
                y = {arm: fun(s, f"{arm}_k{k}p{p}", k) for arm in ("N8", "CQF", "CQD", "CQ", "N6")}
                a_ += y["CQ"] - y["CQF"] - y["CQD"] + y["N8"]; g_ += y["N8"] - y["N6"]
            I_.append(a_); Gs.append(g_)
        I_, Gs = np.asarray(I_), np.asarray(Gs); bs = I_[idx].sum(1) / Gs[idx].sum(1); bi = I_[idx].mean(1) / len(C)
        sh = [float(I_.sum() / Gs.sum()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]]
        rd = ("AND-like - each route needed (serial or joint), not independent carriers" if sh[0] >= .30 and sh[1][0] > .10 else
              "OR-like - redundant routes" if sh[0] <= -.30 and sh[1][1] < -.10 else
              "additive (independent partial routes)" if abs(sh[0]) <= .15 and sh[1][0] > -.30 and sh[1][1] < .30 else "unclear")
        return dict(I_per_run=[float(I_.mean() / len(C)), [float(np.percentile(bi, 2.5)), float(np.percentile(bi, 97.5))]], I_over_G=sh, reading=rd)
    out["INT"] = {x: inter(x) for x in ("g", "m51")}
    out["V1"] = v1; out["V2"] = v2; out["V2_control"] = dict(f_m51_CQ=cq, ok=pc2); out["V3"] = dict(verdict=v3, S=S_, N=N_)
    out["V4"] = dict(verdict=v4, RW=[rw, grp(rw)], RT=[rt, grp(rt)]); out["V5"] = v5; out["V6"] = v6
    ans = {}
    for arm in d["arms"]:
        n_ = to8 = to6 = 0
        for s in sis:
            for k, p in C:
                L_ = by[s]["land"]; a8, a6 = L_[f"N8_k{k}p{p}"], L_[f"N6_k{k}p{p}"]
                if a8 == a6:
                    continue
                x = L_[f"{arm}_k{k}p{p}"]; n_ += 1; to8 += int(x == a8); to6 += int(x == a6)
        ans[arm] = dict(n=n_, N8_answer=to8 / max(n_, 1), N6_answer=to6 / max(n_, 1))
    out["answers"] = ans
    if Z is not None:
        hd = {}
        for arm in ("N8", "CA", "CQ", "CR", "CQF", "CQD", "RQ", "KQ", "N6"):
            m78, tot, pic, gate = [], [], [], []
            for s in sis:
                for k, p in C:
                    key = f"s{s}_{arm}_k{k}p{p}"
                    if key + "_capH" not in Z:
                        continue
                    cH = Z[key + "_capH"].astype(np.float64); iH = Z[key + "_imgH"].astype(np.float64)
                    mx = np.where(np.isfinite(cH), cH, -np.inf).max(-1, keepdims=True)
                    sm = np.exp(np.where(np.isfinite(cH), cH - mx, -np.inf)); sm = sm / sm.sum(-1, keepdims=True)
                    mass_b = sm * iH[..., None]
                    if mass_b.shape[-1] == 8:
                        m78.append(float(mass_b[..., 6:].sum(-1).mean()))
                    ho = Z[key + "_hout"].astype(np.float64)
                    tot.append(float(ho[:, 0][:, [2, 5, 9]].mean())); pic.append(float(ho[:, 1][:, [2, 5, 9]].mean())); gate.append(float(ho[:, 2][:, [2, 5, 9]].mean()))
            if tot:
                hd[arm] = dict(mass_pics_7_8=(float(np.mean(m78)) if m78 else None), gated_out_norm=float(np.mean(tot)),
                               gated_out_picture_part=float(np.mean(pic)), mean_gate=float(np.mean(gate)))
        out["heads"] = hd
    fn = f"{V}/results/r3/s857_analysis.json" if tag == "s857_27b" else f"{V}/results/r3/s857_analysis_{tag}.json"
    json.dump(out, open(fn, "w"), indent=1)
    print("I0", json.dumps(out["I0"]))
    print("INT", json.dumps(out["INT"]))
    print("V1", v1, "| V2", v2, "| V3", v3, "| V4", v4, "| V5", v5, "| V6", json.dumps(v6))
    for arm, x in F.items():
        print(f"  {arm:6s} " + "  ".join(f"{k_} {v_[0]:+.3f} [{v_[1][0]:+.3f}, {v_[1][1]:+.3f}]" for k_, v_ in x.items()))
    print("answers", json.dumps(ans)); print("heads", json.dumps(out.get("heads")))


if __name__ == "__main__":
    main(*sys.argv[1:])
