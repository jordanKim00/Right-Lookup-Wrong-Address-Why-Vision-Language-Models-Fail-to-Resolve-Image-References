"""S916 analysis (rules fixed before the data; knowledge/DECISIONS.md S916 entry).  Input results/r3/s916_<tag>_bypass.json (N8, k 1, 2, 3, 5).
Arms per scene and k: E (END(k)), S (START(k)), RS35 / RS47 / RS51 / RS55 (the ten read rows <- START(k)'s at that layer's input), TS35 (the
question rows from the frame word on, outside the read rows, <- START(k)'s at layer 35), AS35 (both; from layer 35 the run is START(k)'s).
end-side = answer >= N - 2 (pictures 6-8 at N8; S911 R3's measure); START-like = answer k.  Scene bootstrap 4,000 (seed 916).
I0: t1 >= .95; natural START(k) answers k >= .95 (pooled over k); AS35 gives START(k)'s own answer in >= .95 (pooled); natural END(2) is
  end-side in >= .80.  Nothing is read if I0 fails.
B1 (are the read rows at layer 35 necessary for END(2)'s end-side answer?): e = P(end-side | RS35, k = 2) >= .50 with the lower bound > .35 ->
  'END(2)'s end-side answer survives replacing the read rows by START(2)'s at layer 35: the end information reaches the answer from the other
  question rows after layer 35'; <= .20 with the upper bound < .35 -> 'the read rows at layer 35 carry END(2)'s end-side answer'; else
  'partial'.
B2 (are the other question rows at layer 35 necessary?): P(x(TS35) = x(E), k = 2) >= .80 with the lower bound > .65 -> 'the other question rows
  are not needed for END(2)'s answer when the read rows keep END's state'; <= .50 with the upper bound < .65 -> 'END(2)'s answer needs the other
  question rows at layer 35'; else 'partial'.
B3 (does END(2) differ from END(3) and END(5)?): d = P(START-like | RS35, k in {3, 5}) - P(START-like | RS35, k = 2) >= .30 with the lower
  bound > .15 -> 'the read-row swap returns END(3) and END(5) to START but not END(2)'; else 'not observed'.
Descriptive: per k and arm the end-side, START-like and keeps-END shares and answer distributions; the late (L51) and early (L47) anchors;
  for k = 2 the end-side share by swap layer (35, 47, 51, 55).
Usage: python3 s916_analyze.py TAG [OUT]"""
from __future__ import annotations

import collections, json, sys
import numpy as np

TAG = sys.argv[1]
J = json.load(open(f"/vlm/results/r3/s916_{TAG}_bypass.json"))
OUT = sys.argv[2] if len(sys.argv) > 2 else f"/vlm/results/r3/s916_analysis_{TAG}.json"
rng = np.random.default_rng(916); NB = 4000
N = J["N"]; KS = J["ks"]; ARMS = ["E", "S"] + J["arms"]; rows = J["rows"]; nS = len(rows)


def key(arm, k):
    return f"{arm}{k}" if arm in ("E", "S") else f"{arm}_{k}"


def x(r, arm, k):
    return int(r["land"][key(arm, k)])


def anchor(q):
    q = np.asarray(q); am = [int(np.argmax(q[i])) + 1 for i in (2, 5, 9)]
    c = collections.Counter(am).most_common(); return c[0][0] if c[0][1] >= 2 else am[2]


def share(ix, arm, ks, fn):
    v = [float(fn(r, k, x(r, arm, k))) for i in ix for r in [rows[int(i)]] for k in ks]
    return float(np.mean(v)) if v else float("nan")


endside = lambda r, k, xx: xx >= N - 2
startlike = lambda r, k, xx: xx == k
keepsE = lambda r, k, xx: xx == x(r, "E", k)


def boot(fn):
    est = fn(np.arange(nS)); bs = [fn(rng.integers(0, nS, nS)) for _ in range(NB)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return dict(est=float(est), lo=float(lo), hi=float(hi))


allx = np.arange(nS)
res = dict(tag=TAG, n_scenes=nS)
st = share(allx, "S", KS, startlike); asr = share(allx, "AS35", KS, lambda r, k, xx: xx == x(r, "S", k)); e2 = share(allx, "E", [2], endside)
I0 = dict(t1=bool((J["t1_menu"] or 0) >= .95), START=bool(st >= .95), AS35=bool(asr >= .95), END2=bool(e2 >= .80))
res["I0"] = dict(**I0, all=bool(all(I0.values())), t1_value=J["t1_menu"], START_acc=st, AS35_reproduces_START=asr, END2_endside=e2)
ok = res["I0"]["all"]
b1 = boot(lambda ix: share(ix, "RS35", [2], endside))
res["B1"] = dict(endside=b1, verdict=(("END(2)'s end-side answer survives replacing the read rows by START(2)'s at layer 35: the end information reaches the answer from the other question rows after layer 35"
                                       if b1["est"] >= .50 and b1["lo"] > .35 else
                                       "the read rows at layer 35 carry END(2)'s end-side answer" if b1["est"] <= .20 and b1["hi"] < .35 else "partial") if ok else "not read"))
b2 = boot(lambda ix: share(ix, "TS35", [2], keepsE))
res["B2"] = dict(keepsE=b2, verdict=(("the other question rows are not needed for END(2)'s answer when the read rows keep END's state" if b2["est"] >= .80 and b2["lo"] > .65 else
                                      "END(2)'s answer needs the other question rows at layer 35" if b2["est"] <= .50 and b2["hi"] < .65 else "partial") if ok else "not read"))
K35 = [k for k in KS if k in (3, 5)]
b3 = boot(lambda ix: share(ix, "RS35", K35, startlike) - share(ix, "RS35", [2], startlike))
res["B3"] = dict(d=b3, verdict=(("the read-row swap returns END(3) and END(5) to START but not END(2)" if b3["est"] >= .30 and b3["lo"] > .15 else "not observed")
                                if ok else "not read"))
desc = {}
for k in KS:
    d = {}
    for arm in ARMS:
        xs = [x(r, arm, k) for r in rows]
        d[arm] = dict(endside=share(allx, arm, [k], endside), startlike=share(allx, arm, [k], startlike), keepsE=share(allx, arm, [k], keepsE),
                      answers=dict(collections.Counter(xs)),
                      late51=dict(collections.Counter(anchor(r["mm"][key(arm, k)]["51"]) for r in rows)),
                      early47=dict(collections.Counter(anchor(r["mm"][key(arm, k)]["47"]) for r in rows)))
    desc[f"k{k}"] = d
desc["k2_by_layer"] = {arm: boot(lambda ix, arm=arm: share(ix, arm, [2], endside)) for arm in ("RS35", "RS47", "RS51", "RS55")}
res["descriptive"] = desc
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"]["all"], B1=[res["B1"]["verdict"], b1["est"]], B2=[res["B2"]["verdict"], b2["est"]], B3=[res["B3"]["verdict"], b3["est"]]), indent=1))
print("->", OUT)
