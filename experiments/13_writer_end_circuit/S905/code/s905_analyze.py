"""S905 analysis (rules fixed before the data; knowledge/DECISIONS.md S905 entry).  Input results/r3/s905_<tag>_nearend.json.
Cells (k, p): END(k) (target t = 9 - k) pinned at p in {t - 1, t, t + 1}; R(p) = START(p) pinned at p; E_BANDR = E with the band 46-49 <- R(p).
At L51 (matcher-head mean block scores, rows B1 / B4 / answer): back-step score T1b = z(p - 1) - z(p), forward score T1f = z(p + 1) - z(p)
(p < 8).  Band effects W_b = T1b(E) - T1b(E_BANDR), W_f = T1f(E) - T1f(E_BANDR).  Ratios of means over (scene, cells); scene bootstrap 4,000
(seed 905).  Near-end targets: k 2, 3 (t 7, 6); middle: k 5 (t 4); k 4 (t 5) is the boundary (descriptive).
I0: t1 >= .95; START(p) answers p >= .95; L47 anchor = p for every E and R >= .90; the natural-cell step is present (W_b(2, 8) > 0 with the
  interval above 0; S902).  Nothing is read if I0 fails.
T1 (hold at a near-end target): ratio_near = mean[W_b(2, 7), W_b(3, 6)] / mean[W_b(2, 8), W_b(3, 7)]: <= .25 with the upper bound < .50 ->
  'near the end, the band holds an anchor that is on the target'; >= .75 with the lower bound > .50 -> 'the band steps back from the target as
  from one after it'; else 'partial hold'.
T1r (reach): ratio_mid = W_b(5, 4) / W_b(5, 5); ratio_mid - ratio_near >= .30 with the lower bound > 0 -> 'the hold at the target weakens from
  near-end to middle targets'; else 'not observed'.
T2 (forward from before a near-end target): mean W_f over (1, 7), (2, 6), (3, 5) > 0 with the interval above 0 -> 'the END band moves an anchor
  before a near-end target forward'; else 'not observed'.
T2r (reach): mean W_f over (1, 7), (2, 6), (3, 5) - W_f(5, 3) > 0 with the interval above 0 -> 'the forward move from before the target is
  stronger near the end than in the middle'; else 'not observed'.
Descriptive: per cell W_b, W_f, answers P(p - 1), P(p), P(p + 1), P(target) for E, E_BANDR and R(p).
Usage: python3 s905_analyze.py TAG"""
from __future__ import annotations

import collections, json, sys
import numpy as np

TAG = sys.argv[1]
J = json.load(open(f"/vlm/results/r3/s905_{TAG}_nearend.json"))
OUT = f"/vlm/results/r3/s905_analysis_{TAG}.json"
rng = np.random.default_rng(905); NB = 4000
rows = J["rows"]; S = len(rows); PINS = J["pins"]; lm = J["late_mean"]; N = J["N"]
CELLS = {int(p): ks for p, ks in J["cells"].items()}


def zv(r, key, L="51"):
    return np.asarray(r["zl"][key][L], dtype=float)[:, lm[L]].mean(1).mean(0)


def anchor(q):
    q = np.asarray(q); am = [int(np.argmax(q[i])) + 1 for i in (2, 5, 9)]
    c = collections.Counter(am).most_common(); return c[0][0] if c[0][1] >= 2 else am[2]


REC = {}
for si, r in enumerate(rows):
    for p, ks in CELLS.items():
        for k in ks:
            E, B = f"E{k}p{p}", f"E{k}p{p}_BANDR"
            zE, zB = zv(r, E), zv(r, B)
            d = dict(si=si, k=k, p=p, t=N + 1 - k, Wb=(zE[p - 2] - zE[p - 1]) - (zB[p - 2] - zB[p - 1]),
                     Wf=((zE[p] - zE[p - 1]) - (zB[p] - zB[p - 1])) if p < N else np.nan,
                     xE=r["land"][E], xB=r["land"][B], xR=r["land"][f"Rp{p}"], aE=anchor(r["qk47"][E]), aR=anchor(r["qk47"][f"Rp{p}"]))
            REC[(si, k, p)] = d
SC = list(range(S))


def bt(fn):
    est = fn(np.arange(S)); bs = [fn(rng.integers(0, S, S)) for _ in range(NB)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return dict(est=float(est), lo=float(lo), hi=float(hi))


def M(ix, cells, key):
    return float(np.mean([REC[(int(i), k, p)][key] for i in ix for (k, p) in cells]))


res = dict(tag=TAG, n_scenes=S)
st = float(np.mean([r["land"][f"START{p}"] == p for r in rows for p in PINS]))
reach = float(np.mean([a_ == d["p"] for d in REC.values() for a_ in (d["aE"], d["aR"])]))
bS = bt(lambda ix: M(ix, [(2, 8)], "Wb"))
I0 = dict(t1=bool((J["t1_menu"] or 0) >= .95), START=bool(st >= .95), pin_reach=bool(reach >= .90), natural_step=bool(bS["lo"] > 0))
res["I0"] = dict(**I0, all=bool(all(I0.values())), t1_value=J["t1_menu"], START_acc=st, reach=reach, Wb_2_8=bS)
ok = res["I0"]["all"]
IDN, ONE, BEF = [(2, 7), (3, 6)], [(2, 8), (3, 7)], [(1, 7), (2, 6), (3, 5)]
b1 = bt(lambda ix: M(ix, IDN, "Wb") / M(ix, ONE, "Wb"))
res["T1"] = dict(ratio=b1, verdict=(("near the end, the band holds an anchor that is on the target" if b1["est"] <= .25 and b1["hi"] < .50 else
                                     "the band steps back from the target as from one after it" if b1["est"] >= .75 and b1["lo"] > .50 else "partial hold")
                                    if ok else "not read"))
b2 = bt(lambda ix: M(ix, BEF, "Wf"))
res["T2"] = dict(Wf=b2, verdict=(("the END band moves an anchor before a near-end target forward" if b2["lo"] > 0 else "not observed") if ok else "not read"))
b1r = bt(lambda ix: M(ix, [(5, 4)], "Wb") / M(ix, [(5, 5)], "Wb") - M(ix, IDN, "Wb") / M(ix, ONE, "Wb"))
res["T1r"] = dict(diff=b1r, ratio_mid=bt(lambda ix: M(ix, [(5, 4)], "Wb") / M(ix, [(5, 5)], "Wb")),
                  verdict=(("the hold at the target weakens from near-end to middle targets" if b1r["est"] >= .30 and b1r["lo"] > 0 else "not observed") if ok else "not read"))
b2r = bt(lambda ix: M(ix, BEF, "Wf") - M(ix, [(5, 3)], "Wf"))
res["T2r"] = dict(diff=b2r, Wf_mid=bt(lambda ix: M(ix, [(5, 3)], "Wf")),
                  verdict=(("the forward move from before the target is stronger near the end than in the middle" if b2r["lo"] > 0 else "not observed") if ok else "not read"))
desc = {}
for (k_, p_) in sorted({(d["k"], d["p"]) for d in REC.values()}):
    ds = [d for d in REC.values() if d["k"] == k_ and d["p"] == p_]
    P = lambda key, off: float(np.mean([d[key] == d["p"] + off for d in ds]))
    desc[f"k{k_}_p{p_}"] = dict(t=N + 1 - k_, Wb=float(np.mean([d["Wb"] for d in ds])), Wf=(float(np.nanmean([d["Wf"] for d in ds])) if p_ < N else None),
                               E=dict(pm1=P("xE", -1), p=P("xE", 0), pp1=P("xE", 1), target=float(np.mean([d["xE"] == d["t"] for d in ds]))),
                               E_BANDR=dict(pm1=P("xB", -1), p=P("xB", 0), pp1=P("xB", 1), target=float(np.mean([d["xB"] == d["t"] for d in ds]))),
                               R=dict(p=P("xR", 0)))
res["descriptive"] = desc
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"]["all"], T1=[res["T1"]["verdict"], b1["est"]], T1r=[res["T1r"]["verdict"], b1r["est"]], T2=[res["T2"]["verdict"], b2["est"]],
                      T2r=[res["T2r"]["verdict"], b2r["est"]],
                      Wb={c: v["Wb"] for c, v in desc.items()}), indent=1))
print("->", OUT)
