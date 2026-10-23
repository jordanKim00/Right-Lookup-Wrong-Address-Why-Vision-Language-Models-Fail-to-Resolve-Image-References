"""S900 analysis (rules fixed before the data; knowledge/DECISIONS.md S900 entry).  Input results/r3/s900_<tag>_h3readout.json.
W_X = T1(X) - T1(E_BANDR) at L51 (T1 = z_(p-1) - z_p of the matcher-head mean block scores, rows B1 / B4 / answer); FA47's END part
F = W_E - W_ALL47R; path share f_X = (W_E - W_X) / F (ratios of means over (scene, pin)); scene bootstrap 4,000 (seed 900).
I0: t1 >= .95; START(p) answers p >= .95; L47 anchor = p for E and R >= .90; QID reproduces E (max late block-score difference <= 1e-3 in
  >= .99); F > 0 with the interval above 0; the h3 path is present (f_QM_h3 > 0 with the interval above 0; S897 .218); the all-head context
  transplant removes part of the step (mean W_E - W_HPV_all > 0 with the interval above 0; S889 K1).  Nothing is read if I0 fails.
R1 (h3: queries or gate): sq = f_QMq_h3 / f_QM_h3, sg = f_QMg_h3 / f_QM_h3: sq >= .70 with the lower bound > .50 and sg <= .30 -> 'h3's MLP46
  path works through its query (what it reads)'; sg >= .70 with the lower bound > .50 and sq <= .30 -> 'through its output gate'; else 'both'.
R2 (h3 and the anchor context): c3 = mean(W_E - W_HPV_h3) / mean(W_E - W_HPV_all): c3 >= .40 with the lower bound > .20 -> 'h3 carries a large
  part of the predecessor-context passing'; c3 <= .10 with the upper bound < .25 -> 'h3 does not carry the context passing'; else 'partial'.
R3 (readout without both late routes): P(answer = p | R_NOBOTH) (all heads' attention to the menu-name tokens and to every picture block set to
  zero at FA51 / 55 / 59 / 63): <= .30 with the upper bound < .50 -> 'the late full-attention reads of pictures or menu words are needed for the
  answer (either route suffices, S893)'; >= .80 with the lower bound > .65 -> 'the answer survives without late full-attention reads of
  pictures and menu words'; else 'partial'.
Descriptive: f of every arm at L51-63; h3's and the self-address heads' L47 anchor-block attention (E, R, arms); dHP_h3 / W and dHP_all / W;
answers per arm (P(p - 1), P(p), P(9th name)); the chosen-name logit margin under NOBOTH; END's p - 1 answers under NOBOTH.
Usage: python3 s900_analyze.py TAG"""
from __future__ import annotations

import collections, json, sys
import numpy as np

TAG = sys.argv[1]
J = json.load(open(f"/vlm/results/r3/s900_{TAG}_h3readout.json"))
OUT = f"/vlm/results/r3/s900_analysis_{TAG}.json"
rng = np.random.default_rng(900); NB = 4000
rows = J["rows"]; S = len(rows); PINS = J["pins"]; lm = J["late_mean"]; LAYS = [str(l) for l in J["lateq"]]
SA = J["selfaddr"]
ARMS = ("ALL47R", "QM_h3", "QMq_h3", "QMg_h3", "HPV_h3", "HPV_all", "NOBOTH", "QID")


def zv(r, key, L):
    return np.asarray(r["zl"][key][L], dtype=float)[:, lm[L]].mean(1).mean(0)


def T1(r, key, p, L="51"):
    z = zv(r, key, L); return z[p - 2] - z[p - 1]


def anchor(q):
    q = np.asarray(q); am = [int(np.argmax(q[i])) + 1 for i in (2, 5, 9)]
    c = collections.Counter(am).most_common(); return c[0][0] if c[0][1] >= 2 else am[2]


def margin(r, key, tgt):
    x = np.asarray(r["mlog"][key], dtype=float); o = np.delete(x, tgt - 1); return float(x[tgt - 1] - o.max())


REC = {}
for si, r in enumerate(rows):
    for p in PINS:
        d = dict(si=si, p=p)
        for L in LAYS:
            tb = T1(r, f"Ep{p}_E_BANDR", p, L); d[f"W_E_{L}"] = T1(r, f"Ep{p}", p, L) - tb
            for X in ARMS:
                d[f"W_{X}_{L}"] = T1(r, f"Ep{p}_{X}", p, L) - tb
        d["vid"] = max(float(np.abs(zv(r, f"Ep{p}_QID", L) - zv(r, f"Ep{p}", L)).max()) for L in LAYS)
        ms = lambda key, hh: float(np.asarray(r["mass47"][key], float)[:, hh, p - 1].mean())
        d["att_h3"] = {X: ms(f"Ep{p}" + ("" if X == "E" else f"_{X}"), [3]) for X in ("E",) + ARMS if X != "NOBOTH"}
        d["att_h3"]["R"] = ms(f"Rp{p}", [3])
        d["att_SA"] = {X: ms(f"Ep{p}" + ("" if X == "E" else f"_{X}"), SA) for X in ("E",) + ARMS if X != "NOBOTH"}
        d["att_SA"]["R"] = ms(f"Rp{p}", SA)
        d["anch"] = (anchor(r["qk47"][f"Ep{p}"]), anchor(r["qk47"][f"Rp{p}"]))
        d["land"] = {k: v for k, v in r["land"].items() if f"p{p}" in k}
        d["mR"] = margin(r, f"Rp{p}", p); d["mR_NOBOTH"] = margin(r, f"Rp{p}_NOBOTH", p)
        REC[(si, p)] = d
SC = sorted({k[0] for k in REC})


def bt(fn):
    n = len(SC); est = fn(np.arange(n)); bs = [fn(rng.integers(0, n, n)) for _ in range(NB)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return dict(est=float(est), lo=float(lo), hi=float(hi))


def M(ix, fn):
    return float(np.mean([fn(REC[(SC[i], p)]) for i in ix for p in PINS]))


F_ = lambda d, L="51": d[f"W_E_{L}"] - d[f"W_ALL47R_{L}"]
fX = lambda ix, X, L="51": M(ix, lambda d: d[f"W_E_{L}"] - d[f"W_{X}_{L}"]) / M(ix, lambda d: F_(d, L))
dX = lambda ix, X, L="51": M(ix, lambda d: d[f"W_E_{L}"] - d[f"W_{X}_{L}"])
res = dict(tag=TAG, n_scenes=S)
st = float(np.mean([r["land"][f"START{p}"] == p for r in rows for p in PINS]))
reach = float(np.mean([a_ == d["p"] for d in REC.values() for a_ in d["anch"]]))
vid_ok = float(np.mean([d["vid"] <= 1e-3 for d in REC.values()]))
bF = bt(lambda ix: M(ix, F_)); bQ = bt(lambda ix: fX(ix, "QM_h3")); bH = bt(lambda ix: dX(ix, "HPV_all"))
I0 = dict(t1=bool((J["t1_menu"] or 0) >= .95), START=bool(st >= .95), pin_reach=bool(reach >= .90), identity_path=bool(vid_ok >= .99),
          fa47_part=bool(bF["lo"] > 0), h3_path=bool(bQ["lo"] > 0), context=bool(bH["lo"] > 0))
res["I0"] = dict(**I0, all=bool(all(I0.values())), t1_value=J["t1_menu"], START_acc=st, reach=reach, vid_ok=vid_ok, F=bF, f_QM_h3=bQ,
                 dHP_all=bH)
ok = res["I0"]["all"]
sq = bt(lambda ix: fX(ix, "QMq_h3") / fX(ix, "QM_h3")); sg = bt(lambda ix: fX(ix, "QMg_h3") / fX(ix, "QM_h3"))
res["R1"] = dict(share_q=sq, share_g=sg, verdict=(("h3's MLP46 path works through its query (what it reads)" if sq["est"] >= .70 and sq["lo"] > .50 and sg["est"] <= .30 else
                                                   "through its output gate" if sg["est"] >= .70 and sg["lo"] > .50 and sq["est"] <= .30 else "both")
                                                  if ok else "not read"))
c3 = bt(lambda ix: dX(ix, "HPV_h3") / dX(ix, "HPV_all"))
res["R2"] = dict(c3=c3, verdict=(("h3 carries a large part of the predecessor-context passing" if c3["est"] >= .40 and c3["lo"] > .20 else
                                  "h3 does not carry the context passing" if c3["est"] <= .10 and c3["hi"] < .25 else "partial") if ok else "not read"))
pnb = bt(lambda ix: M(ix, lambda d: float(d["land"][f"Rp{d['p']}_NOBOTH"] == d["p"])))
res["R3"] = dict(P_p_R_NOBOTH=pnb, verdict=(("the late full-attention reads of pictures or menu words are needed for the answer (either route suffices, S893)"
                                             if pnb["est"] <= .30 and pnb["hi"] < .50 else
                                             "the answer survives without late full-attention reads of pictures and menu words" if pnb["est"] >= .80 and pnb["lo"] > .65
                                             else "partial") if ok else "not read"))
allx = range(len(SC))
ans = lambda key: dict(P_pm1=float(np.mean([d["land"][key.format(p=d["p"])] == d["p"] - 1 for d in REC.values()])),
                       P_p=float(np.mean([d["land"][key.format(p=d["p"])] == d["p"] for d in REC.values()])),
                       P_9=float(np.mean([d["land"][key.format(p=d["p"])] == 9 for d in REC.values()])))
res["descriptive"] = dict(
    f={f"L{L}": {X: fX(allx, X, L) for X in ARMS if X not in ("ALL47R", "QID")} for L in LAYS},
    dHP_over_W={X: dX(allx, X) / M(allx, lambda d: d["W_E_51"]) for X in ("HPV_h3", "HPV_all")},
    h3_anchor_attention={X: M(allx, lambda d, X=X: d["att_h3"][X]) for X in d["att_h3"]},
    selfaddr_anchor_attention={X: M(allx, lambda d, X=X: d["att_SA"][X]) for X in d["att_SA"]},
    mean_W={X: M(allx, lambda d, X=X: d[f"W_{X}_51"]) for X in ARMS} | {"E": M(allx, lambda d: d["W_E_51"])},
    answers={**{X: ans(f"Ep{{p}}_{X}") for X in ARMS}, "E": ans("Ep{p}"), "R": ans("Rp{p}"), "R_NOBOTH": ans("Rp{p}_NOBOTH"), "H_HP": ans("Hp{p}_HP")},
    R_margin=dict(base=M(allx, lambda d: d["mR"]), NOBOTH=M(allx, lambda d: d["mR_NOBOTH"])))
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"]["all"], R1=[res["R1"]["verdict"], sq["est"], sg["est"]], R2=[res["R2"]["verdict"], c3["est"]],
                      R3=[res["R3"]["verdict"], pnb["est"]], f_QM_h3=bQ["est"]), indent=1))
print("->", OUT)
