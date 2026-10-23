"""S883 analysis (rules fixed before the data; knowledge/DECISIONS.md S883 entry).  Input results/r3/s883_<tag>_readgate.json.
W_X = T1(X) - T1(E_BANDR) at L51 (T1 = z_(p-1) - z_p, matcher-head mean block scores, rows B1 / B4 / answer); the part of the writer's address
effect that depends on a swapped part: C_X = W_E - W_X.  Groups ANCH (self-address + anti-self heads) and H16; ops READ (R's read under E's gate),
GATE (E's read under R's gate), SWAP (the whole output <- R's).  Within-pin Pearson over (scene, pin); scene bootstrap 4,000 (seed 883).
I0: t1 >= .95; START(p) answers p >= .95; L47 anchor = p for E and R >= .90; the writer is present (mean W_E > 0 with the interval above 0);
  the anchor heads matter (mean C_ANCH_SWAP > 0 with the interval above 0).  Nothing is read if I0 fails.
G1 (where the anchor heads' END part lies): share_read = mean C_ANCH_READ / mean C_ANCH_SWAP and share_gate = mean C_ANCH_GATE / mean C_ANCH_SWAP:
  share_read >= .70 with the lower bound > .50 and share_gate <= .30 -> 'the anchor heads' END part lies in what they read (attention pattern and
  values)'; share_gate >= .70 with the lower bound > .50 and share_read <= .30 -> 'it lies in their output gate'; else 'both'.
G2 (which carries the per-scene variation): r_read = r(C_ANCH_READ, W_E), r_gate = r(C_ANCH_GATE, W_E): r_read >= .50 and r_read - r_gate >= .20
  with the interval above 0 -> 'the read carries the per-scene variation'; r_gate >= .50 and r_gate - r_read >= .20 with the interval above 0 ->
  'the gate carries it'; else 'not separable'.
Descriptive: H16 READ / GATE / SWAP, BOTH, REST, ALL47 contributions and per-scene correlations; answers; the anchor heads' L47 attention share
on the anchor block (softmax over the eight block scores, heads ANCH, rows B1 / B4 / answer) in E and R, and its E - R change against C_ANCH_READ.
Usage: python3 s883_analyze.py TAG"""
from __future__ import annotations

import collections, json, sys
import numpy as np

TAG = sys.argv[1]
J = json.load(open(f"/vlm/results/r3/s883_{TAG}_readgate.json"))
OUT = f"/vlm/results/r3/s883_analysis_{TAG}.json"
rng = np.random.default_rng(883); NB = 4000
rows = J["rows"]; S = len(rows); PINS = J["pins"]; lm = J["late_mean"]
ANCH = J["selfaddr"] + J["anti"]
ARMS = ["ANCH_READ", "ANCH_GATE", "ANCH_SWAP", "H16_READ", "H16_GATE", "H16_SWAP", "BOTH_SWAP", "REST_SWAP", "ALL47_SWAP"]


def T1(r, key, p, L="51"):
    z = np.asarray(r["zl"][key][L], dtype=float)[:, lm[L]].mean(1).mean(0); return z[p - 2] - z[p - 1]


def anchor(q):
    q = np.asarray(q); am = [int(np.argmax(q[i])) + 1 for i in (2, 5, 9)]
    c = collections.Counter(am).most_common(); return c[0][0] if c[0][1] >= 2 else am[2]


def att_anchor(z, p):
    z = np.asarray(z, dtype=float)[:, ANCH]                                            # [3, H, 8]
    e = np.exp(z - z.max(-1, keepdims=True)); e /= e.sum(-1, keepdims=True); return float(e[..., p - 1].mean())


REC = {}
for si, r in enumerate(rows):
    for p in PINS:
        tb = T1(r, f"Ep{p}_E_BANDR", p); wE = T1(r, f"Ep{p}", p) - tb
        d = dict(si=si, p=p, W_E=wE, x_E=r["land"][f"Ep{p}"], anch=(anchor(r["qk47"][f"Ep{p}"]), anchor(r["qk47"][f"Rp{p}"])))
        for a_ in ARMS:
            d[f"C_{a_}"] = wE - (T1(r, f"Ep{p}_{a_}", p) - tb); d[f"x_{a_}"] = r["land"][f"Ep{p}_{a_}"]
        d["attE"] = att_anchor(r["z47h"][f"Ep{p}"], p); d["attR"] = att_anchor(r["z47h"][f"Rp{p}"], p)
        REC[(si, p)] = d
SC = sorted({k[0] for k in REC})


def bt(fn):
    n = len(SC); est = fn(np.arange(n)); bs = [fn(rng.integers(0, n, n)) for _ in range(NB)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return dict(est=float(est), lo=float(lo), hi=float(hi))


def m(ix, key):
    return float(np.mean([REC[(SC[i], p)][key] for i in ix for p in PINS]))


def r_(ix, k1, k2):
    x, y = [], []
    for p in PINS:
        u = np.array([REC[(SC[i], p)][k1] for i in ix]); w = np.array([REC[(SC[i], p)][k2] for i in ix])
        x += list(u - u.mean()); y += list(w - w.mean())
    return float(np.corrcoef(x, y)[0, 1])


res = dict(tag=TAG, n_scenes=S)
st = float(np.mean([r["land"][f"START{p}"] == p for r in rows for p in PINS]))
reach = float(np.mean([a_ == d["p"] for d in REC.values() for a_ in d["anch"]]))
wE = bt(lambda ix: m(ix, "W_E")); cs = bt(lambda ix: m(ix, "C_ANCH_SWAP"))
I0 = dict(t1=bool((J["t1_menu"] or 0) >= .95), START=bool(st >= .95), pin_reach=bool(reach >= .90), writer=bool(wE["lo"] > 0), anchor_heads=bool(cs["lo"] > 0))
res["I0"] = dict(**I0, all=bool(all(I0.values())), t1_value=J["t1_menu"], START_acc=st, reach=reach, mean_W_E=wE, mean_C_ANCH_SWAP=cs)
ok = res["I0"]["all"]
sr = bt(lambda ix: m(ix, "C_ANCH_READ") / m(ix, "C_ANCH_SWAP")); sg = bt(lambda ix: m(ix, "C_ANCH_GATE") / m(ix, "C_ANCH_SWAP"))
res["G1"] = dict(share_read=sr, share_gate=sg, verdict=(("the anchor heads' END part lies in what they read (attention pattern and values)"
                                                         if sr["est"] >= .70 and sr["lo"] > .50 and sg["est"] <= .30 else
                                                         "it lies in their output gate" if sg["est"] >= .70 and sg["lo"] > .50 and sr["est"] <= .30 else "both")
                                                        if ok else "not read"))
rr, rg = bt(lambda ix: r_(ix, "C_ANCH_READ", "W_E")), bt(lambda ix: r_(ix, "C_ANCH_GATE", "W_E"))
dd = bt(lambda ix: r_(ix, "C_ANCH_READ", "W_E") - r_(ix, "C_ANCH_GATE", "W_E"))
res["G2"] = dict(r_read=rr, r_gate=rg, diff=dd, verdict=(("the read carries the per-scene variation" if rr["est"] >= .50 and dd["est"] >= .20 and dd["lo"] > 0 else
                                                          "the gate carries it" if rg["est"] >= .50 and dd["est"] <= -.20 and dd["hi"] < 0 else "not separable")
                                                         if ok else "not read"))
desc = {a_: dict(mean_C=m(range(len(SC)), f"C_{a_}"), r_with_W=r_(range(len(SC)), f"C_{a_}", "W_E"),
                 P_pm1=float(np.mean([d[f"x_{a_}"] == d["p"] - 1 for d in REC.values()])), P_p=float(np.mean([d[f"x_{a_}"] == d["p"] for d in REC.values()])))
        for a_ in ARMS}
desc["E"] = dict(mean_W=m(range(len(SC)), "W_E"), P_pm1=float(np.mean([d["x_E"] == d["p"] - 1 for d in REC.values()])), P_p=float(np.mean([d["x_E"] == d["p"] for d in REC.values()])))
desc["anchor_attention"] = dict(E=m(range(len(SC)), "attE"), R=m(range(len(SC)), "attR"))
for d in REC.values():
    d["datt"] = d["attE"] - d["attR"]
desc["anchor_attention"]["r_dE_minus_R_with_C_READ"] = r_(range(len(SC)), "datt", "C_ANCH_READ")
desc["anchor_attention"]["r_dE_minus_R_with_W"] = r_(range(len(SC)), "datt", "W_E")
res["descriptive"] = desc
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"]["all"], G1=[res["G1"]["verdict"], sr["est"], sg["est"]], G2=[res["G2"]["verdict"], rr["est"], rg["est"]]), indent=1))
print("->", OUT)
