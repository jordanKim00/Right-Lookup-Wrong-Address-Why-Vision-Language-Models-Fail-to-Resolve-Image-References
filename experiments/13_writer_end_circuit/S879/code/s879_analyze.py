"""S879 analysis (rules fixed before the data; knowledge/DECISIONS.md S879 entry).  Input results/r3/s879_<tag>_reinforce.json.
Middle identity cells (N, k) = (5, 3), (7, 4), (9, 5): target j* = N + 1 - k = k.  Arms NAT, EMASK (the end route weakened: the read rows may not
read the last picture at 35-47), BANDS (the band's read-row outputs <- START(k)'s: no END step), EMASK_BANDS; scene bootstrap 4,000 (seed 879).
I0: t1 >= .95; START(k) answers k >= .95 in every layout; NAT accuracy at N 5 END(3) >= .70 (S870 observed .88).  Nothing is read if I0 fails.
B1 (reinforcement at N 5): acc(EMASK) - acc(NAT) at N5 END(3) <= -.20 with the interval below 0 -> 'the end route supports the middle identity at
  N 5 (copy and end routes reinforce)'; else 'not'.
B2 (the writer's step makes the errors once the support is weakened): at N5 END(3), P(x = j* - 1 | EMASK) - P(x = j* - 1 | EMASK_BANDS) >= .15
  with the interval above 0 -> 'with the end route weakened, the writer's END step moves the answer one picture start-ward'; else 'not'.
B3 (small-N specificity): [acc(EMASK) - acc(NAT)] at N5 END(3) minus the same at N7 END(4) <= -.15 with the interval below 0 -> 'the protection is
  specific to small N'; else 'not'.
Descriptive: every cell's answer distribution per arm; BANDS - NAT accuracy per cell (the writer's END step at the middle identity); the L47
anchor (majority of B1 / B4 / answer) and the L47 margin of the target block (head-mean score of j* minus the best other block, answer row).
Usage: python3 s879_analyze.py TAG"""
from __future__ import annotations

import collections, json, sys
import numpy as np

TAG = sys.argv[1]
J = json.load(open(f"/vlm/results/r3/s879_{TAG}_reinforce.json"))
OUT = f"/vlm/results/r3/s879_analysis_{TAG}.json"
rng = np.random.default_rng(879); NB = 4000
rows = J["rows"]; S = len(rows); CELLS = [tuple(c) for c in J["cells"]]; E47 = str(J["early"][-1])
ARMS = ("NAT", "EMASK", "BANDS", "EMASK_BANDS")


def X(N, arm):
    return np.array([r["land"][f"N{N}_{arm}"] for r in rows])


def bt(fn):
    est = fn(np.arange(S)); bs = [fn(rng.integers(0, S, S)) for _ in range(NB)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return dict(est=float(est), lo=float(lo), hi=float(hi))


def anchor(r, key):
    q = np.asarray(r["qk"][key][E47]); am = [int(np.argmax(q[i])) + 1 for i in (2, 5, 9)]
    c = collections.Counter(am).most_common(); return c[0][0] if c[0][1] >= 2 else am[2]


def margin(r, key, j):
    q = np.asarray(r["qk"][key][E47])[9]; o = np.delete(q, j - 1); return float(q[j - 1] - o.max())


res = dict(tag=TAG, n_scenes=S, n_runs=J["n_runs"])
st = float(np.mean([r["land"][f"N{N}_START{k}"] == k for r in rows for N, k in CELLS]))
nat5 = float(np.mean(X(5, "NAT") == 3))
I0 = dict(t1=bool((J["t1_menu"] or 0) >= .95), START=bool(st >= .95), N5_reproduces=bool(nat5 >= .70))
res["I0"] = dict(**I0, all=bool(all(I0.values())), t1_value=J["t1_menu"], START_acc=st, N5_NAT_acc=nat5)
ok = res["I0"]["all"]
acc = lambda N, arm, j: (lambda ix: float(np.mean(X(N, arm)[ix] == j)))
b1 = bt(lambda ix: acc(5, "EMASK", 3)(ix) - acc(5, "NAT", 3)(ix))
b2 = bt(lambda ix: acc(5, "EMASK", 2)(ix) - acc(5, "EMASK_BANDS", 2)(ix))
b3 = bt(lambda ix: (acc(5, "EMASK", 3)(ix) - acc(5, "NAT", 3)(ix)) - (acc(7, "EMASK", 4)(ix) - acc(7, "NAT", 4)(ix)))
res["B1"] = dict(diff=b1, verdict=(("the end route supports the middle identity at N 5 (copy and end routes reinforce)" if b1["est"] <= -.20 and b1["hi"] < 0
                                    else "not") if ok else "not read"))
res["B2"] = dict(diff=b2, verdict=(("with the end route weakened, the writer's END step moves the answer one picture start-ward" if b2["est"] >= .15 and b2["lo"] > 0
                                    else "not") if ok else "not read"))
res["B3"] = dict(diff=b3, verdict=(("the protection is specific to small N" if b3["est"] <= -.15 and b3["hi"] < 0 else "not") if ok else "not read"))
desc = {}
for N, k in CELLS:
    d = {}
    for arm in ARMS:
        x = X(N, arm); key = f"N{N}_{arm}"
        d[arm] = dict(acc=float(np.mean(x == k)), P_jm1=float(np.mean(x == k - 1)), P_jp1=float(np.mean(x == k + 1)),
                      dist={str(v_): float(np.mean(x == v_)) for v_ in range(1, 10) if np.any(x == v_)},
                      anchor={str(v_): c for v_, c in collections.Counter(anchor(r, key) for r in rows).items()},
                      margin47=float(np.mean([margin(r, key, k) for r in rows])))
    d["BANDS_minus_NAT"] = bt(lambda ix, N=N, k=k: acc(N, "BANDS", k)(ix) - acc(N, "NAT", k)(ix))
    d["EMASK_minus_NAT"] = bt(lambda ix, N=N, k=k: acc(N, "EMASK", k)(ix) - acc(N, "NAT", k)(ix))
    desc[f"N{N}_END{k}"] = d
res["descriptive"] = desc
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"]["all"], B1=[res["B1"]["verdict"], b1["est"]], B2=[res["B2"]["verdict"], b2["est"]], B3=[res["B3"]["verdict"], b3["est"]]), indent=1))
print("->", OUT)
