"""S921 analysis (rules fixed before the data; knowledge/DECISIONS.md S921 entry; advisor review 3).  Input results/r3/s921_<tag>_routesplit.json
(N8, k 2-6).  Per scene and k at END(k)'s natural L47 anchor a: E, R, B (E_BANDR); E_SWX (route X of the late FA read outputs at layers 51-63
<- B's; X = I pictures, M menu names, O the rest, A all), B_SWX (<- E's; X = I, M, A), E_SWself.  dep(arm) = P(answer != a); the writer's
effect D = dep(E) - dep(B); T_X = [dep(E) - dep(E_SWX)] / D (share removed); U_X = [dep(B_SWX) - dep(B)] / D (share recreated).
Scene bootstrap 4,000 (seed 921), pooled over k.
I0: t1 >= .95; natural START(q) answers q >= .95 (q < N); D >= .30 with the lower bound > .15; the self swap is exact (x(E_SWself) = x(E)
  >= .99); the all-route swap reproduces S917 (T_A >= .75 with the lower bound > .50).  Nothing is read if I0 fails.
M1 (which route carries the writer's effect in the natural run): T_M >= .60 (lower > .40) and T_I <= .30 -> 'the late menu-name reads carry
  most of the writer's effect'; T_I >= .60 (lower > .40) and T_M <= .30 -> 'the late picture reads carry most of it'; T_I >= .30 and T_M >= .30
  -> 'both the picture and the menu-name reads carry parts of it'; T_I <= .30 and T_M <= .30 -> 'neither read route alone carries it'; else
  'mixed'.
M2 (which route alone recreates it in the writer-free run): U_M >= .60 (lower > .40) -> 'the menu-name reads alone recreate the writer's
  effect'; else U_I >= .60 (lower > .40) -> 'the picture reads alone recreate it'; else 'no single read route recreates it'.
Descriptive: T_O; U_A; per k every arm's dep and answers; T_I + T_M + T_O against T_A (additivity).
Usage: python3 s921_analyze.py TAG [OUT]"""
from __future__ import annotations

import collections, json, sys
import numpy as np

TAG = sys.argv[1]
J = json.load(open(f"/vlm/results/r3/s921_{TAG}_routesplit.json"))
OUT = sys.argv[2] if len(sys.argv) > 2 else f"/vlm/results/r3/s921_analysis_{TAG}.json"
rng = np.random.default_rng(921); NB = 4000
N = J["N"]; KS = J["ks"]; rows = J["rows"]; nS = len(rows)
ARMS = ["E", "R", "B", "E_SWI", "E_SWM", "E_SWO", "E_SWA", "B_SWI", "B_SWM", "B_SWA", "E_SWself"]
key = lambda arm, k: (f"{arm[0]}{k}_{arm[2:]}" if "_" in arm else f"{arm}{k}")


def dep(ix, arm):
    return float(np.mean([float(rows[int(i)]["land"][key(arm, k)] != rows[int(i)]["anchor"][str(k)]) for i in ix for k in KS]))


def _div(a_, b_):
    return a_ / b_ if b_ not in (0, 0.0) and np.isfinite(b_) else float("nan")


def boot(fn):
    est = fn(np.arange(nS)); bs = [fn(rng.integers(0, nS, nS)) for _ in range(NB)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return dict(est=float(est), lo=float(lo), hi=float(hi))


D = lambda ix: dep(ix, "E") - dep(ix, "B")
T = lambda X: (lambda ix: _div(dep(ix, "E") - dep(ix, f"E_SW{X}"), D(ix)))
U = lambda X: (lambda ix: _div(dep(ix, f"B_SW{X}") - dep(ix, "B"), D(ix)))
allx = np.arange(nS)
res = dict(tag=TAG, n_scenes=nS)
st = float(np.mean([r["land"][f"N_S{q}"] == q for r in rows for q in range(1, N)]))
d0 = boot(D); ta = boot(T("A"))
selfx = float(np.mean([r["land"][key("E_SWself", k)] == r["land"][key("E", k)] for r in rows for k in KS]))
I0 = dict(t1=bool((J["t1_menu"] or 0) >= .95), START=bool(st >= .95), writer_effect=bool(d0["est"] >= .30 and d0["lo"] > .15), self_swap=bool(selfx >= .99),
          all_route=bool(ta["est"] >= .75 and ta["lo"] > .50))
res["I0"] = dict(**I0, all=bool(all(I0.values())), t1_value=J["t1_menu"], START_acc=st, D=d0, T_A=ta, self_swap_same=selfx)
ok = res["I0"]["all"]
ti, tm, to = boot(T("I")), boot(T("M")), boot(T("O"))
if tm["est"] >= .60 and tm["lo"] > .40 and ti["est"] <= .30:
    v1 = "the late menu-name reads carry most of the writer's effect"
elif ti["est"] >= .60 and ti["lo"] > .40 and tm["est"] <= .30:
    v1 = "the late picture reads carry most of it"
elif ti["est"] >= .30 and tm["est"] >= .30:
    v1 = "both the picture and the menu-name reads carry parts of it"
elif ti["est"] <= .30 and tm["est"] <= .30:
    v1 = "neither read route alone carries it"
else:
    v1 = "mixed"
res["M1"] = dict(T_I=ti, T_M=tm, T_O=to, verdict=(v1 if ok else "not read"))
ui, um = boot(U("I")), boot(U("M"))
v2 = ("the menu-name reads alone recreate the writer's effect" if um["est"] >= .60 and um["lo"] > .40 else
      "the picture reads alone recreate it" if ui["est"] >= .60 and ui["lo"] > .40 else "no single read route recreates it")
res["M2"] = dict(U_I=ui, U_M=um, U_A=boot(U("A")), verdict=(v2 if ok else "not read"))
desc = {f"k{k}": {arm: dict(dep=float(np.mean([r["land"][key(arm, k)] != r["anchor"][str(k)] for r in rows])),
                            answers=dict(collections.Counter(r["land"][key(arm, k)] for r in rows))) for arm in ARMS} for k in KS}
desc["dep_pooled"] = {arm: dep(allx, arm) for arm in ARMS}
desc["additivity"] = dict(sum_T=ti["est"] + tm["est"] + to["est"], T_A=ta["est"])
res["descriptive"] = desc
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"]["all"], D=d0["est"], T_A=ta["est"], M1=[res["M1"]["verdict"], ti["est"], tm["est"], to["est"]],
                      M2=[res["M2"]["verdict"], ui["est"], um["est"]]), indent=1))
print("->", OUT)
