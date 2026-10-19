"""S855 analysis (rules fixed in knowledge/DECISIONS.md before any S855 data; Gap 2 second stage, user 10-18).  Does the actual-count effect of the
late lookup come from positional geometry (multimodal RoPE: the question sits D positions later when 8 pictures are shown) or from pictures 7-8?
30 independent scenes; scene bootstrap 4,000 (seed 0); k {3, 5} x p {3, 4} pooled; ratios of sums within each scene resample.
Readouts: Z6 = late block scores of pictures 1-6 (rows B1, B4, answer; layers 51-63; each 6-vector centred) - pictures 1-6 have the same
positions and keys in every arm, so these change only through the query side; L6 = menu logits of pictures 1-6 (centred).
  G_all  = rho(E6S toward E8 from E6)    the 8-picture geometry applied to the 6-picture content, every layer
  G_late = rho(E6SL toward E8 from E6)   the same shift only in the rotary embedding of the late full-attention layers
  G_back = rho(E8S toward E6 from E8)    the 6-picture geometry applied with pictures 7-8 present (reverse direction)
  X_pic  = rho(E8S toward E8 from E6)    what remains of the 8-picture state at the 6-picture geometry: the part carried by pictures 7-8
  rho(x toward a from b) = sum (x - b).(a - b) / sum |a - b|^2.
I0 (all): t1 >= .95; image keys identical within each picture count (0.0); QNOOP_E = END(k 5) >= .90 with the 95th percentile of the largest
  menu-logit change <= .5 (both counts); START(k) -> k >= .90 (k 3, 5; both counts); pin donors START(3), START(4) -> p >= .90 (natural and
  shifted); PIN (layers 43 / 47, rows B1 / B4 maximum = p) >= .95 in E6 / E8; exactness: E{6,8}_k5p3NX (explicit natural ids, alone) = E{6,8}_k5p3N
  (internal ids, alone), and E{6,8}..Z = E{6,8}.. in the same batch (answers >= .99, 95th percentile of the largest menu-logit difference <= .05).
GEOM (needs I0; on L6, with Z6 reported): G_all >= .70 with lower bound > .50 and G_back >= .70 with lower bound > .50 -> 'the actual-count
  effect is mostly positional geometry'; G_all <= .30 with upper bound < .50 and G_back <= .30 with upper bound < .50 -> 'the actual-count
  effect is not positional geometry'; otherwise 'mixed'.
LATE (needs I0): G_late (L6) >= .50 with lower bound > .30 -> 'the late-layer geometry alone reproduces most of the actual-count effect';
  <= .15 with upper bound < .30 -> 'the late-layer geometry alone does not'; otherwise 'partial'.
PIC (needs I0): X_pic (L6) >= .30 with lower bound > .10 -> 'pictures 7-8 themselves carry part of the effect'; <= .10 with upper bound < .20
  -> 'pictures 7-8 carry little of it'; otherwise 'partial'.
Descriptive: the same shares on Z6, per k, per layer; answers (on runs where E6 and E8 differ: the shares answering E8's / E6's answer);
writer outputs (FA47 / DN49 at B1 / B4) under the all-layer shift (geometry reaching the writer).
Usage: python exp/s855_analyze.py [TAG]  ->  results/r3/s855_analysis.json"""
from __future__ import annotations

import json, sys
import numpy as np

V = "/home/storage2/ai26269345/vlm"
B = 4000
KS = (3, 5); PS = (3, 4)


def main(tag="s855_27b"):
    d = json.load(open(f"{V}/results/r3/s855_{tag}_geom.json")); Z = np.load(f"{V}/results/r3/s855_{tag}_geom.npz")
    comps = [f"{l}{kd}" for l, kd in [tuple(c) for c in d["comps"]]]
    LATE = tuple(str(l) for l in d["late"]); PINL = tuple(str(l) for l in d["early"][-2:])
    FAc, DNc = f"{d['early'][-1]}mix", f"{max(d['band'])}mix"
    by = {r["si"]: r for r in d["rows"]}; sis = sorted(by); n = len(sis)
    rng = np.random.default_rng(0); idx = rng.integers(0, n, (B, n))
    cen = lambda x: np.asarray(x, dtype=float) - np.mean(x)
    mld = lambda r, a1, a2: max(abs(x - y) for x, y in zip(r["mlog"][a1], r["mlog"][a2]))
    def same(a1, a2):
        return [float(np.mean([by[s]["land"][a1] == by[s]["land"][a2] for s in sis])), float(np.percentile([mld(by[s], a1, a2) for s in sis], 95))]
    out = dict(tag=tag, D=d["D"], t1=d["t1_menu"], key_rel_diff_max=d["key_rel_diff_max"], n=n, n_runs=d.get("n_runs"))
    i0 = dict(t1=d["t1_menu"] >= .95, keys=d["key_rel_diff_max"] == 0.0)
    for na in (6, 8):
        qn = same(f"QNOOP_a{na}", f"END_k5_a{na}"); i0[f"qnoop{na}"] = dict(v=qn, ok=bool(qn[0] >= .90 and qn[1] <= .5))
        st = {k: float(np.mean([by[s]["land"][f"START_q{k}_a{na}"] == k for s in sis])) for k in KS}; i0[f"start{na}"] = dict(v=st, ok=bool(min(st.values()) >= .90))
        dn = {f"{q}{sfx}": float(np.mean([by[s]["land"][f"START_q{q}_a{na}{sfx}"] == q for s in sis])) for q in PS for sfx in ("", "_S")}
        i0[f"donors{na}"] = dict(v=dn, ok=bool(min(dn.values()) >= .90))
        pin = float(np.mean([int(np.argmax(by[s]["qk"][f"E{na}_k{k}p{p}"][L][ri]) + 1 == p) for s in sis for k in KS for p in PS for L in PINL for ri in (2, 5)]))
        i0[f"pin{na}"] = dict(v=pin, ok=bool(pin >= .95))
        ex = {"explicit_vs_internal_alone": same(f"E{na}_k5p3NX", f"E{na}_k5p3N"), **{f"Z_k{k}p{p}": same(f"E{na}_k{k}p{p}Z", f"E{na}_k{k}p{p}") for k in KS for p in PS}}
        i0[f"batch_numerics{na}"] = dict(v=same(f"E{na}_k5p3N", f"E{na}_k5p3"), ok=True)   # descriptive: alone vs in the 17-run batch
        i0[f"exact{na}"] = dict(v=ex, ok=bool(all(x[0] >= .99 and x[1] <= .05 for x in ex.values())))
    ok_all = bool(i0["t1"] and i0["keys"] and all(v["ok"] for k, v in i0.items() if isinstance(v, dict)))
    out["I0"] = dict(i0, all=ok_all); nvt = "no valid test"
    def Z6(s, nm):
        return np.concatenate([cen(by[s]["qk"][nm][L][ri][:6]) for L in LATE for ri in (2, 5, 9)])
    def L6(s, nm):
        return cen(by[s]["mlog"][nm][:6])
    def W(s, nm):
        c = Z[f"s{s}_{nm}_comp"].astype(np.float64); return np.concatenate([c[comps.index(cn), r] for cn in (FAc, DNc) for r in (0, 1)])
    def rho(f, x_arm, to_arm, from_arm, cells=None):
        num, den = [], []
        for s in sis:
            a_ = b_ = 0.0
            for k in KS:
                for p in PS:
                    if cells is not None and (k, p) not in cells:
                        continue
                    nm = lambda arm: arm.format(k=k, p=p)
                    x, t, f0 = f(s, nm(x_arm)), f(s, nm(to_arm)), f(s, nm(from_arm))
                    a_ += float((x - f0) @ (t - f0)); b_ += float((t - f0) @ (t - f0))
            num.append(a_); den.append(b_)
        num, den = np.asarray(num), np.asarray(den); bs = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-12)
        return [float(num.sum() / max(den.sum(), 1e-12)), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]]
    E6, E8 = "E6_k{k}p{p}", "E8_k{k}p{p}"
    shares = {}
    for nm_f, f in (("L6", L6), ("Z6", Z6), ("W", W)):
        shares[nm_f] = dict(G_all=rho(f, E6 + "S", E8, E6), G_late=rho(f, E6 + "SL", E8, E6), G_back=rho(f, E8 + "S", E6, E8),
                            G_back_late=rho(f, E8 + "SL", E6, E8), X_pic=rho(f, E8 + "S", E8, E6))
    out["shares"] = shares
    out["shares_by_k"] = {f"k{k}": {nm_f: dict(G_all=rho(f, E6 + "S", E8, E6, [(k, p) for p in PS]), X_pic=rho(f, E8 + "S", E8, E6, [(k, p) for p in PS]))
                                    for nm_f, f in (("L6", L6), ("Z6", Z6))} for k in KS}
    L = shares["L6"]
    g = (nvt if not ok_all else "the actual-count effect is mostly positional geometry"
         if L["G_all"][0] >= .70 and L["G_all"][1][0] > .50 and L["G_back"][0] >= .70 and L["G_back"][1][0] > .50
         else "the actual-count effect is not positional geometry"
         if L["G_all"][0] <= .30 and L["G_all"][1][1] < .50 and L["G_back"][0] <= .30 and L["G_back"][1][1] < .50 else "mixed")
    lt = (nvt if not ok_all else "the late-layer geometry alone reproduces most of the actual-count effect" if L["G_late"][0] >= .50 and L["G_late"][1][0] > .30
          else "the late-layer geometry alone does not" if L["G_late"][0] <= .15 and L["G_late"][1][1] < .30 else "partial")
    pc = (nvt if not ok_all else "pictures 7-8 themselves carry part of the effect" if L["X_pic"][0] >= .30 and L["X_pic"][1][0] > .10
          else "pictures 7-8 carry little of it" if L["X_pic"][0] <= .10 and L["X_pic"][1][1] < .20 else "partial")
    out["GEOM"] = g; out["LATE"] = lt; out["PIC"] = pc
    ans = {}
    for x_arm in ("S", "SL"):
        for src, dst in ((E6, E8), (E8, E6)):
            n_ = a_ = b_ = 0
            for s in sis:
                for k in KS:
                    for p in PS:
                        L_ = by[s]["land"]; a0, a1 = L_[src.format(k=k, p=p)], L_[dst.format(k=k, p=p)]
                        if a0 == a1:
                            continue
                        x = L_[src.format(k=k, p=p) + x_arm]; n_ += 1; a_ += int(x == a1); b_ += int(x == a0)
            ans[f"{src[:2]}{x_arm}"] = dict(n=n_, to_other=a_ / max(n_, 1), stays=b_ / max(n_, 1))
    out["answers"] = ans
    out["dist"] = {nm.format(k=k, p=p): {str(x): float(np.mean([by[s]["land"][nm.format(k=k, p=p)] == x for s in sis])) for x in range(9)
                                         if any(by[s]["land"][nm.format(k=k, p=p)] == x for s in sis)}
                   for nm in (E6, E6 + "S", E6 + "SL", E8, E8 + "S", E8 + "SL") for k in KS for p in PS}
    fn = f"{V}/results/r3/s855_analysis.json" if tag == "s855_27b" else f"{V}/results/r3/s855_analysis_{tag}.json"
    json.dump(out, open(fn, "w"), indent=1)
    print("I0", json.dumps(out["I0"])[:1500])
    print("GEOM", g, "| LATE", lt, "| PIC", pc)
    print("shares", json.dumps(shares))


if __name__ == "__main__":
    main(*sys.argv[1:])
