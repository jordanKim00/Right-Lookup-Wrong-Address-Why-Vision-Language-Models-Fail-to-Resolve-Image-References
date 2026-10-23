"""S893 analysis (rules fixed before the data; knowledge/DECISIONS.md S893 entry).  Input results/r3/s893_<tag>_readroute.json.
Answers by name (1-based slot); scene bootstrap 4,000 (seed 893) over (scene, pin).
I0: t1 >= .95; START(p) answers p >= .95 and R answers p >= .95; L47 anchor = p for R >= .90; the factor-1 instrument R_LID reproduces R (max late
  block-score difference <= 1e-3 in >= .99 of (scene, pin)).  Nothing is read if I0 fails.
M1 (image route alone in START): P(answer = p | R_NOMENU) >= .90 -> 'the image route alone resolves START'; <= .60 -> 'START needs the menu
  route'; else 'partial'.
M2 (does the menu route hold START's answer?): P(answer = f | R_WIDE_NOMENU) >= .50 with the lower bound > .30 -> 'with the menu route removed,
  START's answer follows the read rows' image content'; <= .10 -> 'not'; else 'partial'.
M3 (the writer's route): dI = P(p - 1 | E_NOIMG) - P(p - 1 | E), dM = P(p - 1 | E_NOMENU) - P(p - 1 | E): dI <= -.25 with the upper bound < 0 and
  |dM| < .10 -> 'the writer's step acts through the image route'; dM <= -.25 with the upper bound < 0 and |dI| < .10 -> 'through the menu
  route'; else 'both'.
Descriptive: answers per arm (P(p - 1), P(p), P(f)), late T1 and f - p block-score changes, menu logit margins.
Usage: python3 s893_analyze.py TAG"""
from __future__ import annotations

import collections, json, sys
import numpy as np

TAG = sys.argv[1]
J = json.load(open(f"/vlm/results/r3/s893_{TAG}_readroute.json"))
OUT = f"/vlm/results/r3/s893_analysis_{TAG}.json"
rng = np.random.default_rng(893); NB = 4000
rows = J["rows"]; S = len(rows); PINS = J["pins"]; lm = J["late_mean"]; LAYS = [str(l) for l in J["lateq"]]
FAR = {int(k): v for k, v in J["far"].items()}
ARMS = ("R", "E", "R_NOMENU", "R_NOIMG", "E_NOMENU", "E_NOIMG", "R_WIDE", "R_WIDE_NOMENU", "E_WIDE_NOMENU", "R_WIDE_NOIMG", "R_LID")


def zv(r, key, L):
    return np.asarray(r["zl"][key][L], dtype=float)[:, lm[L]].mean(1).mean(0)


def anchor(q):
    q = np.asarray(q); am = [int(np.argmax(q[i])) + 1 for i in (2, 5, 9)]
    c = collections.Counter(am).most_common(); return c[0][0] if c[0][1] >= 2 else am[2]


def key(arm, p):
    run, *rest = arm.split("_", 1)
    return f"{run}p{p}" + (f"_{rest[0]}" if rest else "")


REC = {}
for si, r in enumerate(rows):
    for p in PINS:
        f = FAR[p]; d = dict(si=si, p=p, f=f)
        d["x"] = {a_: r["land"][key(a_, p)] for a_ in ARMS}
        d["vid"] = max(float(np.abs(zv(r, key("R_LID", p), L) - zv(r, key("R", p), L)).max()) for L in LAYS)
        d["anchR"] = anchor(r["qk47"][key("R", p)])
        for L in LAYS:
            for a_ in ARMS:
                z = zv(r, key(a_, p), L); d[f"t1_{a_}_{L}"] = z[p - 2] - z[p - 1]; d[f"fp_{a_}_{L}"] = z[f - 1] - z[p - 1]
        REC[(si, p)] = d
SC = sorted({k[0] for k in REC})


def bt(fn):
    n = len(SC); est = fn(np.arange(n)); bs = [fn(rng.integers(0, n, n)) for _ in range(NB)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return dict(est=float(est), lo=float(lo), hi=float(hi))


def P(ix, arm, which):
    return float(np.mean([REC[(SC[i], p)]["x"][arm] == (p - 1 if which == "pm1" else p if which == "p" else REC[(SC[i], p)]["f"]) for i in ix for p in PINS]))


res = dict(tag=TAG, n_scenes=S)
st = float(np.mean([r["land"][f"START{p}"] == p for r in rows for p in PINS]))
rp = float(np.mean([d["x"]["R"] == d["p"] for d in REC.values()])); reach = float(np.mean([d["anchR"] == d["p"] for d in REC.values()]))
vid_ok = float(np.mean([d["vid"] <= 1e-3 for d in REC.values()]))
I0 = dict(t1=bool((J["t1_menu"] or 0) >= .95), START=bool(st >= .95 and rp >= .95), pin_reach=bool(reach >= .90), identity=bool(vid_ok >= .99))
res["I0"] = dict(**I0, all=bool(all(I0.values())), t1_value=J["t1_menu"], START_acc=st, R_acc=rp, reach=reach, vid_ok=vid_ok)
ok = res["I0"]["all"]
m1 = bt(lambda ix: P(ix, "R_NOMENU", "p"))
res["M1"] = dict(P_p=m1, verdict=(("the image route alone resolves START" if m1["est"] >= .90 else "START needs the menu route" if m1["est"] <= .60 else "partial") if ok else "not read"))
m2 = bt(lambda ix: P(ix, "R_WIDE_NOMENU", "f"))
res["M2"] = dict(P_f=m2, verdict=(("with the menu route removed, START's answer follows the read rows' image content" if m2["est"] >= .50 and m2["lo"] > .30 else
                                   "not" if m2["est"] <= .10 else "partial") if ok else "not read"))
dI = bt(lambda ix: P(ix, "E_NOIMG", "pm1") - P(ix, "E", "pm1")); dM = bt(lambda ix: P(ix, "E_NOMENU", "pm1") - P(ix, "E", "pm1"))
res["M3"] = dict(dI=dI, dM=dM, verdict=(("the writer's step acts through the image route" if dI["est"] <= -.25 and dI["hi"] < 0 and abs(dM["est"]) < .10 else
                                         "through the menu route" if dM["est"] <= -.25 and dM["hi"] < 0 and abs(dI["est"]) < .10 else "both") if ok else "not read"))
allx = range(len(SC))
res["descriptive"] = dict(answers={a_: {w: P(allx, a_, w) for w in ("pm1", "p", "f")} for a_ in ARMS},
                          late={f"L{L}": {a_: dict(t1=float(np.mean([d[f"t1_{a_}_{L}"] for d in REC.values()])), f_minus_p=float(np.mean([d[f"fp_{a_}_{L}"] for d in REC.values()])))
                                          for a_ in ARMS} for L in LAYS})
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"]["all"], M1=[res["M1"]["verdict"], m1["est"]], M2=[res["M2"]["verdict"], m2["est"]], M3=[res["M3"]["verdict"], dI["est"], dM["est"]]), indent=1))
print("->", OUT)
