"""S868 analysis (rules fixed before the data; knowledge/DECISIONS.md S868 entry).  Input results/r3/s868_<tag>_nearend.json.
Per layout N in {8, 9} and k in {1, 2, 3}: j* = N + 1 - k; answer x; arms compared with the natural END(k) of the same pass and scene;
pooled over the two layouts; scene bootstrap 4,000 (seed 868).
I0: t1 >= .95; natural START(N), START(N - 1), START(N - 2) >= .95; R(q) (START pinned at its own address) answers q >= .90; the pinned arms reach
  their pin (L47 head-mean argmax at rows B1 / B4 / answer = the pinned picture) >= .90.  Nothing is read if I0 fails.
R1 (END(3)'s natural success needs the early end route): acc(PIN_NM1) - acc(E) at k 3 <= -.20 with the interval below 0, and P(x = N - 1 |
  PIN_NM1, k 3) >= .50 -> 'END(3)'s natural success needs the early end route (a sharp N - 1 anchor stays there)'; else 'not'.
R2 (the writer moves an exact target): under PIN_TGT, k 3: P(x = j* - 1) >= .25 with the lower bound > .10 -> "the writer's step moves an exact
  END(3) target off (identity break near the end)"; k 1 and k 2: P(x = j*) >= .70 each -> 'no step at k <= 2: the exact target stays'.
R3 (the break is the writer's): k 3, P(x = j* | TGT_BANDR) - P(x = j* | PIN_TGT) >= .20 with the interval above 0 -> "removing the writer's END
  part keeps the exact target".
R4 (the end is read in the early lookup): EMASK, pooled over k: P(x = j* - 1) - P(x = j* - 1 | E) >= .25 with the interval above 0 -> 'masking the
  last picture in the early lookup moves the end route one picture start-ward'.
R5 (late reads of the end): LMASK, pooled over k: |acc(LMASK) - acc(E)| <= .10 with the interval inside +-.20 -> "the late lookup's direct reads
  of the last picture do not set near-end answers".
Descriptive: per (N, k) answer distributions of every arm; natural accuracy; pin reach per arm.
Usage: python3 s868_analyze.py TAG"""
from __future__ import annotations

import json, sys
import numpy as np

TAG = sys.argv[1]
J = json.load(open(f"/vlm/results/r3/s868_{TAG}_nearend.json"))
OUT = f"/vlm/results/r3/s868_analysis_{TAG}.json"
rng = np.random.default_rng(868)
NB = 4000
rows = J["rows"]; S = len(rows); LAYS = J["lays"]; KS = J["ks"]
E47 = str(J["early"][-1])


def bt(fn):
    est = fn(np.arange(S)); bs = [fn(rng.integers(0, S, S)) for _ in range(NB)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return dict(est=float(est), lo=float(lo), hi=float(hi))


def ans(key):
    return np.array([r["land"][key] for r in rows])


def anchor_r(r, key):
    q = np.asarray(r["qk"][key][E47]); am = [int(np.argmax(q[i])) + 1 for i in (2, 5, 9)]
    c = {v: am.count(v) for v in am}; top = max(c, key=c.get)
    return top if c[top] >= 2 else am[2]


def share(arm, k, target_of):
    pre = [(ans(f"N{N}_END{k}" + ("" if arm == "E" else f"_{arm}")) == target_of(N, k)).astype(float) for N in LAYS]
    return lambda ix: float(np.mean([x[ix].mean() for x in pre]))


J_ = lambda N, k: N + 1 - k
JM1 = lambda N, k: N - k
NM1 = lambda N, k: N - 1
res = dict(tag=TAG, n_scenes=S, n_runs=J["n_runs"])
st = float(np.mean([r["land"][f"N{N}_START{q}"] == q for r in rows for N in LAYS for q in (N, N - 1, N - 2)]))
racc = float(np.mean([r["land"][f"N{N}_R{q}"] == q for r in rows for N in LAYS for q in (N, N - 1, N - 2)]))
reach = {}
for N in LAYS:
    for k in KS:
        for arm, q in (("PIN_NM1", N - 1), ("PIN_TGT", N + 1 - k), ("TGT_BANDR", N + 1 - k)):
            reach[f"N{N}_k{k}_{arm}"] = float(np.mean([anchor_r(r, f"N{N}_END{k}_{arm}") == q for r in rows]))
I0 = dict(t1=bool((J["t1_menu"] or 0) >= .95), START_acc=bool(st >= .95), R_acc=bool(racc >= .90), pin_reach=bool(np.mean(list(reach.values())) >= .90))
res["I0"] = dict(**I0, all=bool(all(I0.values())), t1_value=J["t1_menu"], START_acc_value=st, R_acc_value=racc, pin_reach_value=float(np.mean(list(reach.values()))),
                 pin_reach_by_arm=reach)
ok = res["I0"]["all"]
res["natural_accuracy"] = {f"k{k}": bt(share("E", k, J_)) for k in KS}
r1d = bt(lambda ix: share("PIN_NM1", 3, J_)(ix) - share("E", 3, J_)(ix)); r1s = bt(share("PIN_NM1", 3, NM1))
res["R1"] = dict(dacc=r1d, P_Nm1=r1s, verdict=(("END(3)'s natural success needs the early end route (a sharp N - 1 anchor stays there)"
                                                 if r1d["est"] <= -.20 and r1d["hi"] < 0 and r1s["est"] >= .50 else "not") if ok else "not read"))
r2b = bt(share("PIN_TGT", 3, JM1)); r2s = {f"k{k}": bt(share("PIN_TGT", k, J_)) for k in KS}
res["R2"] = dict(P_jm1_k3=r2b, P_target=r2s,
                 verdict_k3=(("the writer's step moves an exact END(3) target off (identity break near the end)" if r2b["est"] >= .25 and r2b["lo"] > .10 else "not")
                             if ok else "not read"),
                 verdict_k12=(("no step at k <= 2: the exact target stays" if r2s["k1"]["est"] >= .70 and r2s["k2"]["est"] >= .70 else "not") if ok else "not read"))
r3 = bt(lambda ix: share("TGT_BANDR", 3, J_)(ix) - share("PIN_TGT", 3, J_)(ix))
res["R3"] = dict(diff=r3, P_target_BANDR={f"k{k}": bt(share("TGT_BANDR", k, J_)) for k in KS},
                 verdict=(("removing the writer's END part keeps the exact target" if r3["est"] >= .20 and r3["lo"] > 0 else "not") if ok else "not read"))
r4 = bt(lambda ix: float(np.mean([share("EMASK", k, JM1)(ix) - share("E", k, JM1)(ix) for k in KS])))
res["R4"] = dict(dP_jm1=r4, by_k={f"k{k}": bt(lambda ix, k=k: share("EMASK", k, JM1)(ix) - share("E", k, JM1)(ix)) for k in KS},
                 verdict=(("masking the last picture in the early lookup moves the end route one picture start-ward" if r4["est"] >= .25 and r4["lo"] > 0 else "not")
                          if ok else "not read"))
r5 = bt(lambda ix: float(np.mean([share("LMASK", k, J_)(ix) - share("E", k, J_)(ix) for k in KS])))
res["R5"] = dict(dacc=r5, verdict=(("the late lookup's direct reads of the last picture do not set near-end answers"
                                    if abs(r5["est"]) <= .10 and r5["lo"] >= -.20 and r5["hi"] <= .20 else "not") if ok else "not read"))
desc = {}
for N in LAYS:
    for k in KS:
        for arm in ("E", "PIN_NM1", "PIN_TGT", "TGT_BANDR", "EMASK", "LMASK"):
            key = f"N{N}_END{k}" + ("" if arm == "E" else f"_{arm}")
            x = ans(key); desc[key] = {str(v): float(np.mean(x == v)) for v in range(1, 10) if np.any(x == v)}
res["descriptive"] = desc
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"]["all"], R1=res["R1"]["verdict"], R2=[res["R2"]["verdict_k3"], res["R2"]["verdict_k12"]], R3=res["R3"]["verdict"],
                      R4=res["R4"]["verdict"], R5=res["R5"]["verdict"]), indent=1))
print("->", OUT)
