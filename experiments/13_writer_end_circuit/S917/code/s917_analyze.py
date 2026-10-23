"""S917 analysis (rules fixed before the data; knowledge/DECISIONS.md S917 entry).  Input results/r3/s917_<tag>_lateroute.json (N8, k 2-6).
Per scene and k at END(k)'s natural L47 anchor a: E, R (START(a)), B (E_BANDR: the writer-free run); late read swaps E_SW (late FA 51-63
gated outputs of the read rows <- B's), B_SW (<- E's), E_SW51 (51 only), E_SWL (55-63), E_SWself (E <- E); late read removals E_NI / B_NI
(picture tokens 51-63), E_NM / B_NM (menu names, descriptive).  dep(arm) = P(answer != a); the writer's effect D = dep(E) - dep(B).
Scene bootstrap 4,000 (seed 917), pooled over k.
I0: t1 >= .95; natural START(q) answers q >= .95 (q < N); the writer effect is present (D >= .30 with the lower bound > .15); the self swap is
  exact (x(E_SWself) = x(E) in >= .99).  Nothing is read if I0 fails.  W4 is read only if the writer-free answer survives the picture-read
  removal (P(x(B_NI) = x(B)) >= .70; otherwise 'not interpretable').
W1 (are the writer's effects on the answer transmitted through the late full-attention reads?): T = [dep(E) - dep(E_SW)] / D >= .75 with the
  lower bound > .50 -> 'the writer's effect on the answer is transmitted through the late full-attention reads'; <= .25 with the upper bound
  < .40 -> 'the writer's effect reaches the answer outside the late full-attention reads'; else 'partial'.
W2 (do the late reads alone recreate it?): U = [dep(B_SW) - dep(B)] / D >= .75 with the lower bound > .50 -> 'the late full-attention reads
  alone recreate the writer's effect in the writer-free run'; <= .25 with the upper bound < .40 -> 'the late reads alone do not recreate it';
  else 'partial'.
W3 (which late reads): T51 = [dep(E) - dep(E_SW51)] / D, TL = [dep(E) - dep(E_SWL)] / D: T51 >= .60 (lower > .40) and TL <= .30 -> 'the first
  late lookup (layer 51) carries most of the transmission'; TL >= .60 (lower > .40) and T51 <= .30 -> 'the later reads (55-63) carry most of
  it'; else 'spread over the late reads'.
W4 (the late picture reads): rho_NI = [dep(E_NI) - dep(B_NI)] / D >= .60 with the lower bound > .40 -> 'the writer's effect reaches the answer
  without the late picture reads'; <= .25 with the upper bound < .40 -> 'the late picture reads carry the writer's effect'; else 'partial'.
Descriptive: per k and arm dep, stay (a), step (a - 1, a - 2), other, answers, the late anchors at L51 and L63, the agreement of the answer with
  the L51 argmax; the NM arms; D per k.  The ratios are 'the share of the writer's effect removed (W1) / recreated (W2) by the swap', not
  mediation shares of the natural error.
Usage: python3 s917_analyze.py TAG [OUT]"""
from __future__ import annotations

import collections, json, sys
import numpy as np

TAG = sys.argv[1]
J = json.load(open(f"/vlm/results/r3/s917_{TAG}_lateroute.json"))
OUT = sys.argv[2] if len(sys.argv) > 2 else f"/vlm/results/r3/s917_analysis_{TAG}.json"
rng = np.random.default_rng(917); NB = 4000
N = J["N"]; KS = J["ks"]; rows = J["rows"]; nS = len(rows)
ARMS = ["E", "R", "B", "E_NI", "B_NI", "E_NM", "B_NM", "E_SW", "B_SW", "E_SW51", "E_SWL", "E_SWself"]
key = lambda arm, k: (f"{arm[0]}{k}_{arm[2:]}" if "_" in arm else f"{arm}{k}")


def anchor(q):
    q = np.asarray(q); am = [int(np.argmax(q[i])) + 1 for i in (2, 5, 9)]
    c = collections.Counter(am).most_common(); return c[0][0] if c[0][1] >= 2 else am[2]


def dep(ix, arm):
    return float(np.mean([float(rows[int(i)]["land"][key(arm, k)] != rows[int(i)]["anchor"][str(k)]) for i in ix for k in KS]))


def _div(a_, b_):
    return a_ / b_ if b_ not in (0, 0.0) and np.isfinite(b_) else float("nan")


def boot(fn):
    est = fn(np.arange(nS)); bs = [fn(rng.integers(0, nS, nS)) for _ in range(NB)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return dict(est=float(est), lo=float(lo), hi=float(hi))


D = lambda ix: dep(ix, "E") - dep(ix, "B")
allx = np.arange(nS)
res = dict(tag=TAG, n_scenes=nS)
st = float(np.mean([r["land"][f"N_S{q}"] == q for r in rows for q in range(1, N)]))
d0 = boot(D)
selfx = float(np.mean([r["land"][key("E_SWself", k)] == r["land"][key("E", k)] for r in rows for k in KS]))
guard = float(np.mean([r["land"][key("B_NI", k)] == r["land"][key("B", k)] for r in rows for k in KS]))
I0 = dict(t1=bool((J["t1_menu"] or 0) >= .95), START=bool(st >= .95), writer_effect=bool(d0["est"] >= .30 and d0["lo"] > .15), self_swap=bool(selfx >= .99))
res["I0"] = dict(**I0, all=bool(all(I0.values())), t1_value=J["t1_menu"], START_acc=st, D=d0, self_swap_same=selfx, B_NI_keeps_B=guard)
ok = res["I0"]["all"]
T = boot(lambda ix: _div(dep(ix, "E") - dep(ix, "E_SW"), D(ix)))
res["W1"] = dict(T=T, verdict=(("the writer's effect on the answer is transmitted through the late full-attention reads" if T["est"] >= .75 and T["lo"] > .50 else
                                "the writer's effect reaches the answer outside the late full-attention reads" if T["est"] <= .25 and T["hi"] < .40 else "partial")
                               if ok else "not read"))
U = boot(lambda ix: _div(dep(ix, "B_SW") - dep(ix, "B"), D(ix)))
res["W2"] = dict(U=U, verdict=(("the late full-attention reads alone recreate the writer's effect in the writer-free run" if U["est"] >= .75 and U["lo"] > .50 else
                                "the late reads alone do not recreate it" if U["est"] <= .25 and U["hi"] < .40 else "partial") if ok else "not read"))
T51 = boot(lambda ix: _div(dep(ix, "E") - dep(ix, "E_SW51"), D(ix))); TL = boot(lambda ix: _div(dep(ix, "E") - dep(ix, "E_SWL"), D(ix)))
if T51["est"] >= .60 and T51["lo"] > .40 and TL["est"] <= .30:
    v3 = "the first late lookup (layer 51) carries most of the transmission"
elif TL["est"] >= .60 and TL["lo"] > .40 and T51["est"] <= .30:
    v3 = "the later reads (55-63) carry most of it"
else:
    v3 = "spread over the late reads"
res["W3"] = dict(T51=T51, TL=TL, verdict=(v3 if ok else "not read"))
rn = boot(lambda ix: _div(dep(ix, "E_NI") - dep(ix, "B_NI"), D(ix)))
res["W4"] = dict(rho_NI=rn, guard=guard, verdict=((("the writer's effect reaches the answer without the late picture reads" if rn["est"] >= .60 and rn["lo"] > .40 else
                                                    "the late picture reads carry the writer's effect" if rn["est"] <= .25 and rn["hi"] < .40 else "partial")
                                                   if guard >= .70 else "not interpretable (the writer-free answer does not survive the removal)") if ok else "not read"))
desc = {}
for k in KS:
    d = {}
    for arm in ARMS:
        xs = [r["land"][key(arm, k)] for r in rows]; a_ = [r["anchor"][str(k)] for r in rows]
        d[arm] = dict(dep=float(np.mean([x != a for x, a in zip(xs, a_)])), stay=float(np.mean([x == a for x, a in zip(xs, a_)])),
                      step=float(np.mean([x in (a - 1, a - 2) for x, a in zip(xs, a_)])), other=float(np.mean([x not in (a, a - 1, a - 2) for x, a in zip(xs, a_)])),
                      answers=dict(collections.Counter(xs)), late51=dict(collections.Counter(anchor(r["mm"][key(arm, k)]["51"]) for r in rows)),
                      late63=dict(collections.Counter(anchor(r["mm"][key(arm, k)]["63"]) for r in rows)),
                      ans_is_L51_argmax=float(np.mean([r["land"][key(arm, k)] == anchor(r["mm"][key(arm, k)]["51"]) for r in rows])))
    d["D"] = d["E"]["dep"] - d["B"]["dep"]; d["anchors"] = dict(collections.Counter(r["anchor"][str(k)] for r in rows))
    desc[f"k{k}"] = d
desc["dep_pooled"] = {arm: dep(allx, arm) for arm in ARMS}
res["descriptive"] = desc
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"]["all"], D=d0["est"], W1=[res["W1"]["verdict"], T["est"]], W2=[res["W2"]["verdict"], U["est"]],
                      W3=[res["W3"]["verdict"], T51["est"], TL["est"]], W4=[res["W4"]["verdict"], rn["est"], guard]), indent=1))
print("->", OUT)
