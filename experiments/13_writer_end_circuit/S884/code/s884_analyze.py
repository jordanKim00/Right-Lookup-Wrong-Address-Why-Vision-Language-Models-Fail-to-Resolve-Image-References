"""S884 analysis (rules fixed before the data; knowledge/DECISIONS.md S884 entry).  Input results/r3/s884_<tag>_valueswap.json.
Late block score z_b(run, L) = mean over rows B1 / B4 / answer and the late_mean matcher heads of the LSE block score of block b.
Writer effect on every block: W(b) = z_b(E) - z_b(E_BANDR); under a value edit X: W_X(b) = z_b(E_X) - z_b(E_X_E_BANDR) (band <- R_X, the same
edit applied to R).  For pin p: predecessor p - 1, anchor p, far f (7 for p 3, 1 for p 4 / 5), second far f2.  Natural bonus to transfer
B = W(p-1) - W(f); natural anchor gap D = mean W(b) over the pictures not adjacent to the anchor (b not in p-1, p, p+1) - W(p) (amended
before data: the smoke's W(f) - W(p) was near 0, so the far block alone cannot serve as the anchor's reference); step W(T1) = W(p-1) - W(p).  Ratios of means over (scene, pin); scene
bootstrap 4,000 (seed 884).  Primary layer L51.
I0: t1 >= .95; START(p) answers p >= .95; L47 anchor = p for E and R >= .90; the identity map (VID) reproduces E, R and E_BANDR (max block-score
  difference <= 1e-3 at every late layer in >= .99 of (scene, pin)); the writer is present (mean W(T1) > 0, interval above 0); B and D > 0 with
  intervals above 0.  Nothing is read if I0 fails.
H1 (VPF: predecessor block's FA47 values <- f's): g = mean[W_VPF(f) - W(f)] / mean B: g >= .50 with the lower bound > .25 -> 'the writer's
  predecessor component follows the content FA47 reads at the predecessor's position (content pointer)'; g <= .15 with the upper bound < .30 ->
  'the predecessor component does not follow FA47's read content'; else 'partial'.
H2 (VAF: anchor block's FA47 values <- f's): s = mean[W_VAF(p) - W(p)] / mean D (the share of the anchor's natural gap released when the
  read rows read another picture's content at the anchor's position): s >= .50 with the lower bound > .25 -> 'the anchor suppression follows the
  content FA47 reads at the anchor's position'; s <= .15 with the upper bound < .30 -> 'not'; else 'partial'.  (Descriptive: the transfer
  [W(f) - W_VAF(f)] / D.)
H3 (VSW: predecessor <-> anchor values): rho = mean W_VSW(T1) / mean W(T1): rho <= -.25 with the upper bound < 0 -> 'exchanging the two read
  contents reverses the writer's step'; rho >= .75 with the lower bound > .50 -> 'the step does not follow the read contents'; else 'reduced'.
H5 (VPA: predecessor block's FA47 values <- the anchor's): ratio = mean W_VPA(T1) / mean W(T1): <= .30 with the upper bound < .50 -> 'the
  writer's step needs the predecessor's own content at its FA47 read (writer-side content coupling)'; >= .70 with the lower bound > .50 -> 'the
  step does not need the predecessor's content at FA47'; else 'mixed'.
H4 (EARLY in R: the anchor block's values <- f's at FA35 / 39 / 43 / 47): P_f = P(answer = f | R_EARLY): >= .50 with the lower bound > .30 ->
  'the late lookup dereferences the content the early lookup copied'; <= .10 -> 'not'; else 'partial'.
Descriptive: the control VFF (W_VFF(T1) / W(T1), [W_VFF(f) - W(f)] / B); the loss at the predecessor under VPF, the anchor's release under VAF;
L55 / L59 / L63; answers per arm (P(p - 1), P(p), P(f)); E_EARLY; total effects z(E_X) - z(E) on p - 1, p, f.
Usage: python3 s884_analyze.py TAG"""
from __future__ import annotations

import collections, json, sys
import numpy as np

TAG = sys.argv[1]
J = json.load(open(f"/vlm/results/r3/s884_{TAG}_valueswap.json"))
OUT = f"/vlm/results/r3/s884_analysis_{TAG}.json"
rng = np.random.default_rng(884); NB = 4000
rows = J["rows"]; S = len(rows); PINS = J["pins"]; lm = J["late_mean"]; LAYS = [str(l) for l in J["lateq"]]
FAR = {int(k): v for k, v in J["far"].items()}; FAR2 = {int(k): v for k, v in J["far2"].items()}
EDITS = ("VID", "VPF", "VAF", "VSW", "VFF", "VPA")


def zv(r, key, L):
    return np.asarray(r["zl"][key][L], dtype=float)[:, lm[L]].mean(1).mean(0)


def anchor(q):
    q = np.asarray(q); am = [int(np.argmax(q[i])) + 1 for i in (2, 5, 9)]
    c = collections.Counter(am).most_common(); return c[0][0] if c[0][1] >= 2 else am[2]


REC = {}
for si, r in enumerate(rows):
    for p in PINS:
        f = FAR[p]; d = dict(si=si, p=p, f=f)
        for L in LAYS:
            W = zv(r, f"Ep{p}", L) - zv(r, f"Ep{p}_E_BANDR", L)
            d[f"W_{L}"] = W
            for X in EDITS:
                d[f"W{X}_{L}"] = zv(r, f"Ep{p}_{X}", L) - zv(r, f"Ep{p}_{X}_E_BANDR", L)
                d[f"TE{X}_{L}"] = zv(r, f"Ep{p}_{X}", L) - zv(r, f"Ep{p}", L)
        vid = max(float(np.abs(zv(r, f"{a_}p{p}_VID{s_}", L) - zv(r, f"{a_}p{p}{s_}", L)).max())
                  for L in LAYS for a_, s_ in (("E", ""), ("R", ""), ("E", "_E_BANDR")))
        d["vid"] = vid
        d["anch"] = (anchor(r["qk47"][f"Ep{p}"]), anchor(r["qk47"][f"Rp{p}"]))
        d["land"] = {k[len(f"Ep{p}"):] if k.startswith(f"Ep{p}") else "R" + k[len(f"Rp{p}"):]: v
                     for k, v in r["land"].items() if k.startswith(f"Ep{p}") or k.startswith(f"Rp{p}")}
        REC[(si, p)] = d
SC = sorted({k[0] for k in REC})


def bt(fn):
    n = len(SC); est = fn(np.arange(n)); bs = [fn(rng.integers(0, n, n)) for _ in range(NB)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return dict(est=float(est), lo=float(lo), hi=float(hi))


def M(ix, fn):
    return float(np.mean([fn(REC[(SC[i], p)]) for i in ix for p in PINS]))


def t1(W, p):
    return W[p - 2] - W[p - 1]


L = "51"
B_ = lambda d, L=L: d[f"W_{L}"][d["p"] - 2] - d[f"W_{L}"][d["f"] - 1]
OTH = lambda p: [b for b in range(1, 9) if b not in (p - 1, p, p + 1)]
D_ = lambda d, L=L: float(np.mean([d[f"W_{L}"][b - 1] for b in OTH(d["p"])])) - d[f"W_{L}"][d["p"] - 1]
res = dict(tag=TAG, n_scenes=S)
st = float(np.mean([r["land"][f"START{p}"] == p for r in rows for p in PINS]))
reach = float(np.mean([a_ == d["p"] for d in REC.values() for a_ in d["anch"]]))
vid_ok = float(np.mean([d["vid"] <= 1e-3 for d in REC.values()]))
wT = bt(lambda ix: M(ix, lambda d: t1(d["W_51"], d["p"]))); bB = bt(lambda ix: M(ix, B_)); bD = bt(lambda ix: M(ix, D_))
I0 = dict(t1=bool((J["t1_menu"] or 0) >= .95), START=bool(st >= .95), pin_reach=bool(reach >= .90), identity_map=bool(vid_ok >= .99),
          writer=bool(wT["lo"] > 0), bonus=bool(bB["lo"] > 0 and bD["lo"] > 0))
res["I0"] = dict(**I0, all=bool(all(I0.values())), t1_value=J["t1_menu"], START_acc=st, reach=reach, vid_ok=vid_ok,
                 vid_max=float(max(d["vid"] for d in REC.values())), mean_W_T1=wT, mean_B=bB, mean_D=bD)
ok = res["I0"]["all"]
g = bt(lambda ix: M(ix, lambda d: d["WVPF_51"][d["f"] - 1] - d["W_51"][d["f"] - 1]) / M(ix, B_))
res["H1"] = dict(g=g, verdict=(("the writer's predecessor component follows the content FA47 reads at the predecessor's position (content pointer)"
                                if g["est"] >= .50 and g["lo"] > .25 else
                                "the predecessor component does not follow FA47's read content" if g["est"] <= .15 and g["hi"] < .30 else "partial")
                               if ok else "not read"))
s_ = bt(lambda ix: M(ix, lambda d: d["WVAF_51"][d["p"] - 1] - d["W_51"][d["p"] - 1]) / M(ix, D_))
res["H2"] = dict(s=s_, verdict=(("the anchor suppression follows the content FA47 reads at the anchor's position" if s_["est"] >= .50 and s_["lo"] > .25 else
                                 "not" if s_["est"] <= .15 and s_["hi"] < .30 else "partial") if ok else "not read"))
rho = bt(lambda ix: M(ix, lambda d: t1(d["WVSW_51"], d["p"])) / M(ix, lambda d: t1(d["W_51"], d["p"])))
res["H3"] = dict(rho=rho, verdict=(("exchanging the two read contents reverses the writer's step" if rho["est"] <= -.25 and rho["hi"] < 0 else
                                    "the step does not follow the read contents" if rho["est"] >= .75 and rho["lo"] > .50 else "reduced") if ok else "not read"))
rpa = bt(lambda ix: M(ix, lambda d: t1(d["WVPA_51"], d["p"])) / M(ix, lambda d: t1(d["W_51"], d["p"])))
res["H5"] = dict(ratio=rpa, verdict=(("the writer's step needs the predecessor's own content at its FA47 read (writer-side content coupling)"
                                      if rpa["est"] <= .30 and rpa["hi"] < .50 else
                                      "the step does not need the predecessor's content at FA47" if rpa["est"] >= .70 and rpa["lo"] > .50 else "mixed")
                                     if ok else "not read"))
pf = bt(lambda ix: M(ix, lambda d: float(d["land"]["R_EARLY"] == d["f"])))
res["H4"] = dict(P_f=pf, verdict=(("the late lookup dereferences the content the early lookup copied" if pf["est"] >= .50 and pf["lo"] > .30 else
                                   "not" if pf["est"] <= .10 else "partial") if ok else "not read"))
desc = {}
for L_ in LAYS:
    BL = lambda d, L_=L_: d[f"W_{L_}"][d["p"] - 2] - d[f"W_{L_}"][d["f"] - 1]
    DL = lambda d, L_=L_: float(np.mean([d[f"W_{L_}"][b - 1] for b in OTH(d["p"])])) - d[f"W_{L_}"][d["p"] - 1]
    allx = range(len(SC))
    desc[f"L{L_}"] = dict(
        B=M(allx, BL), D=M(allx, DL), W_T1=M(allx, lambda d: t1(d[f"W_{L_}"], d["p"])),
        g=M(allx, lambda d: d[f"WVPF_{L_}"][d["f"] - 1] - d[f"W_{L_}"][d["f"] - 1]) / M(allx, BL),
        loss_pred_VPF=M(allx, lambda d: d[f"W_{L_}"][d["p"] - 2] - d[f"WVPF_{L_}"][d["p"] - 2]) / M(allx, BL),
        s=M(allx, lambda d: d[f"WVAF_{L_}"][d["p"] - 1] - d[f"W_{L_}"][d["p"] - 1]) / M(allx, DL),
        transfer_to_f_VAF=M(allx, lambda d: d[f"W_{L_}"][d["f"] - 1] - d[f"WVAF_{L_}"][d["f"] - 1]) / M(allx, DL),
        rho=M(allx, lambda d: t1(d[f"WVSW_{L_}"], d["p"])) / M(allx, lambda d: t1(d[f"W_{L_}"], d["p"])),
        VPA_ratio=M(allx, lambda d: t1(d[f"WVPA_{L_}"], d["p"])) / M(allx, lambda d: t1(d[f"W_{L_}"], d["p"])),
        VFF_ratio=M(allx, lambda d: t1(d[f"WVFF_{L_}"], d["p"])) / M(allx, lambda d: t1(d[f"W_{L_}"], d["p"])),
        VFF_far=M(allx, lambda d: d[f"WVFF_{L_}"][d["f"] - 1] - d[f"W_{L_}"][d["f"] - 1]) / M(allx, BL),
        total={X: dict(pred=M(allx, lambda d, X=X: d[f"TE{X}_{L_}"][d["p"] - 2]), anchor=M(allx, lambda d, X=X: d[f"TE{X}_{L_}"][d["p"] - 1]),
                       far=M(allx, lambda d, X=X: d[f"TE{X}_{L_}"][d["f"] - 1])) for X in EDITS})
arms = sorted({k for d in REC.values() for k in d["land"]})
desc["answers"] = {a_: dict(P_pm1=float(np.mean([d["land"][a_] == d["p"] - 1 for d in REC.values()])), P_p=float(np.mean([d["land"][a_] == d["p"] for d in REC.values()])),
                            P_f=float(np.mean([d["land"][a_] == d["f"] for d in REC.values()]))) for a_ in arms}
res["descriptive"] = desc
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"]["all"], H1=[res["H1"]["verdict"], g["est"]], H2=[res["H2"]["verdict"], s_["est"]], H3=[res["H3"]["verdict"], rho["est"]], H5=[res["H5"]["verdict"], rpa["est"]],
                      H4=[res["H4"]["verdict"], pf["est"]]), indent=1))
print("->", OUT)
