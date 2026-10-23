"""S891 analysis (rules fixed before the data; knowledge/DECISIONS.md S891 entry).  Inputs results/r3/s891_<tag>_attnpredict.json and the frozen
predictions research/ces/outputs/writer_circuit_2026-10-20/main/s891_frozen/s891_predictions.json (hash in DECISIONS).
Measured block change at L51: dz_b(X) = z_b(X) - z_b(base run) (base R for R arms, E for E_SA150; matcher-head mean LSE block score, rows
B1 / B4 / answer).  Predicted dzhat_b from the frozen file.  Within-run centring over the eight blocks for profile comparisons.  Scene bootstrap
4,000 (seed 891) over (scene, pin).
I0: t1 >= .95; START(p) answers p >= .95; L47 anchor = p for E and R >= .90; the factor-1 instrument R_ID reproduces R (max late block-score
  difference <= 1e-3 in >= .99 of (scene, pin)).  Nothing is read if I0 fails.
P1 (profiles): per arm, the pooled Pearson over (scene, pin, block) of the centred measured vs predicted changes; an arm is 'predicted' if r >= .80
  with the lower bound > .70.  >= 4 of the 6 arms predicted -> 'the composition map predicts the late-address effects of FA47 attention
  interventions'; <= 1 -> 'not'; else 'partial'.
P2 (step size): for R_SA50, R_SA2P, R_ALL2P, E_SA150 the per-(scene, pin) T1 change (dz_(p-1) - dz_p): Pearson(measured, predicted) >= .60 and
  slope (least squares through the origin, measured on predicted) in [.5, 2] -> arm 'calibrated'; >= 3 of 4 -> 'calibrated'; else 'not calibrated'.
P3 (answers, R_ALL2P): predicted 'answer = p - 1' iff T1(R) + predicted T1 change > 0 at L51; agreement with the actual answer (= p - 1 or not)
  >= .75 with the lower bound > .65 -> 'answers predicted'; else 'not'.
Descriptive: per-arm r and slopes at L51-63, answers per arm, mean measured / predicted T1 changes.
Usage: python3 s891_analyze.py TAG"""
from __future__ import annotations

import collections, json, sys
import numpy as np

TAG = sys.argv[1]
J = json.load(open(f"/vlm/results/r3/s891_{TAG}_attnpredict.json"))
PR = json.load(open("/vlm/research/ces/outputs/writer_circuit_2026-10-20/main/s891_frozen/s891_predictions.json"))
OUT = f"/vlm/results/r3/s891_analysis_{TAG}.json"
rng = np.random.default_rng(891); NB = 4000
rows = J["rows"]; S = len(rows); PINS = J["pins"]; lm = J["late_mean"]; LAYS = [str(l) for l in J["lateq"]]
ARMS = ("R_SA50", "R_ANTI50", "R_H1650", "R_SA2P", "R_ALL2P", "E_SA150")


def zv(r, key, L):
    return np.asarray(r["zl"][key][L], dtype=float)[:, lm[L]].mean(1).mean(0)


def anchor(q):
    q = np.asarray(q); am = [int(np.argmax(q[i])) + 1 for i in (2, 5, 9)]
    c = collections.Counter(am).most_common(); return c[0][0] if c[0][1] >= 2 else am[2]


REC = {}
for si, r in enumerate(rows):
    for p in PINS:
        d = dict(si=si, p=p)
        for L in LAYS:
            for arm in ARMS:
                run, nm = arm.split("_", 1)
                base = f"{run}p{p}"; key = f"{run}p{p}_{nm}"
                d[f"m_{arm}_{L}"] = zv(r, key, L) - zv(r, base, L)
        for arm in ARMS:
            d[f"p_{arm}"] = np.asarray(PR[arm][f"{si},{p}"], float)
        d["T1R"] = (lambda z: z[p - 2] - z[p - 1])(zv(r, f"Rp{p}", "51"))
        d["vid"] = max(float(np.abs(zv(r, f"Rp{p}_ID", L) - zv(r, f"Rp{p}", L)).max()) for L in LAYS)
        d["anch"] = (anchor(r["qk47"][f"Ep{p}"]), anchor(r["qk47"][f"Rp{p}"]))
        d["land"] = {k: v for k, v in r["land"].items() if f"p{p}" in k}
        REC[(si, p)] = d
SC = sorted({k[0] for k in REC})


def bt(fn):
    n = len(SC); est = fn(np.arange(n)); bs = [fn(rng.integers(0, n, n)) for _ in range(NB)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return dict(est=float(est), lo=float(lo), hi=float(hi))


def prof_r(ix, arm, L="51"):
    x, y = [], []
    for i in ix:
        for p in PINS:
            d = REC[(SC[i], p)]; m = d[f"m_{arm}_{L}"]; pr = d[f"p_{arm}"]
            x += list(m - m.mean()); y += list(pr - pr.mean())
    return float(np.corrcoef(x, y)[0, 1])


def t1(v, p):
    return v[p - 2] - v[p - 1]


def step(ix, arm, L="51"):
    x = np.array([t1(REC[(SC[i], p)][f"p_{arm}"], p) for i in ix for p in PINS]); y = np.array([t1(REC[(SC[i], p)][f"m_{arm}_{L}"], p) for i in ix for p in PINS])
    return float(np.corrcoef(x, y)[0, 1]), float((x * y).sum() / (x * x).sum())


res = dict(tag=TAG, n_scenes=S)
st = float(np.mean([r["land"][f"START{p}"] == p for r in rows for p in PINS]))
reach = float(np.mean([a_ == d["p"] for d in REC.values() for a_ in d["anch"]]))
vid_ok = float(np.mean([d["vid"] <= 1e-3 for d in REC.values()]))
I0 = dict(t1=bool((J["t1_menu"] or 0) >= .95), START=bool(st >= .95), pin_reach=bool(reach >= .90), identity=bool(vid_ok >= .99))
res["I0"] = dict(**I0, all=bool(all(I0.values())), t1_value=J["t1_menu"], START_acc=st, reach=reach, vid_ok=vid_ok, vid_max=float(max(d["vid"] for d in REC.values())))
ok = res["I0"]["all"]
p1 = {arm: bt(lambda ix, arm=arm: prof_r(ix, arm)) for arm in ARMS}
npred = sum(int(v["est"] >= .80 and v["lo"] > .70) for v in p1.values())
res["P1"] = dict(r=p1, n_predicted=npred, verdict=(("the composition map predicts the late-address effects of FA47 attention interventions" if npred >= 4 else
                                                    "not" if npred <= 1 else "partial") if ok else "not read"))
p2 = {}
for arm in ("R_SA50", "R_SA2P", "R_ALL2P", "E_SA150"):
    r_, sl = step(range(len(SC)), arm); p2[arm] = dict(r=r_, slope=sl, calibrated=bool(r_ >= .60 and .5 <= sl <= 2))
ncal = sum(int(v["calibrated"]) for v in p2.values())
res["P2"] = dict(arms=p2, n_calibrated=ncal, verdict=(("calibrated" if ncal >= 3 else "not calibrated") if ok else "not read"))
def agree(ix):
    v = []
    for i in ix:
        for p in PINS:
            d = REC[(SC[i], p)]; pred = (d["T1R"] + t1(d["p_R_ALL2P"], p)) > 0; act = d["land"][f"Rp{p}_ALL2P"] == p - 1
            v.append(pred == act)
    return float(np.mean(v))
p3 = bt(agree)
res["P3"] = dict(agreement=p3, verdict=(("answers predicted" if p3["est"] >= .75 and p3["lo"] > .65 else "not") if ok else "not read"))
allx = range(len(SC))
res["descriptive"] = dict(
    profile_r={f"L{L}": {arm: prof_r(allx, arm, L) for arm in ARMS} for L in LAYS},
    step={f"L{L}": {arm: dict(zip(("r", "slope"), step(allx, arm, L))) for arm in ARMS} for L in LAYS},
    mean_T1_change={arm: dict(measured=float(np.mean([t1(d[f"m_{arm}_51"], d["p"]) for d in REC.values()])),
                              predicted=float(np.mean([t1(d[f"p_{arm}"], d["p"]) for d in REC.values()]))) for arm in ARMS},
    answers={arm: dict(P_pm1=float(np.mean([d["land"][f"{arm[0]}p{d['p']}_{arm.split('_', 1)[1]}"] == d["p"] - 1 for d in REC.values()])),
                       P_p=float(np.mean([d["land"][f"{arm[0]}p{d['p']}_{arm.split('_', 1)[1]}"] == d["p"] for d in REC.values()]))) for arm in ARMS})
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"]["all"], P1=[res["P1"]["verdict"], npred], P2=[res["P2"]["verdict"], ncal], P3=[res["P3"]["verdict"], p3["est"]]), indent=1))
print("->", OUT)
