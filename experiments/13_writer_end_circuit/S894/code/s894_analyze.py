"""S894 analysis (rules fixed before the data; knowledge/DECISIONS.md S894 entry).  Inputs results/r3/s894_<tag>_endpull.json and the frozen
predictions research/ces/outputs/writer_circuit_2026-10-20/main/s894_frozen/s894_predictions.json (hash in DECISIONS).
T1 = z(p - 1) - z(p) at L51 (matcher-head mean LSE block scores, rows B1 / B4 / answer); natural writer step W = T1(E) - T1(E_BANDR); arm change
dT1(X) = T1(X) - T1(base run) (base R for R arms, E for E arms); measured block changes dz_b(X) vs the frozen predictions.  Scene bootstrap 4,000
(seed 894) over (scene, pin).
I0: t1 >= .95; START(p) answers p >= .95; L47 anchor = p for E and R >= .90; R_ID reproduces R (max late block-score difference <= 1e-3 in >= .99);
  W > 0 with the interval above 0.  Nothing is read if I0 fails.
U1 (necessity, END): -mean dT1(E_SA_S2A) / mean W >= .20 with the lower bound > .05 -> 'undoing the end-ward pull of the self-address heads
  removes part of the writer's step'; <= .05 with the upper bound < .15 -> 'not observed'; else 'small'.
U2 (sufficiency, START): mean dT1(R_SA2S40) > 0 with the interval above 0 -> 'mimicking the end-ward pull in START moves the late address toward
  the predecessor'; else 'not observed'.
U3 (prediction): for each of the four arms, the pooled Pearson of the within-run centred measured vs predicted block changes; >= 3 of 4 arms with
  r >= .80 (lower bound > .70) -> 'the composition map predicts the end-pull interventions'; else 'partial' (>= 1) or 'not'.
Descriptive: mean measured / predicted T1 changes, the share of W per arm, answers per arm, L55-63.
Usage: python3 s894_analyze.py TAG"""
from __future__ import annotations

import collections, json, sys
import numpy as np

TAG = sys.argv[1]
J = json.load(open(f"/vlm/results/r3/s894_{TAG}_endpull.json"))
PR = json.load(open("/vlm/research/ces/outputs/writer_circuit_2026-10-20/main/s894_frozen/s894_predictions.json"))
OUT = f"/vlm/results/r3/s894_analysis_{TAG}.json"
rng = np.random.default_rng(894); NB = 4000
rows = J["rows"]; S = len(rows); PINS = J["pins"]; lm = J["late_mean"]; LAYS = [str(l) for l in J["lateq"]]
ARMS = ("R_SA2S40", "R_SA2S100", "E_SA_S2A", "E_ALL_S2A")


def zv(r, key, L):
    return np.asarray(r["zl"][key][L], dtype=float)[:, lm[L]].mean(1).mean(0)


def anchor(q):
    q = np.asarray(q); am = [int(np.argmax(q[i])) + 1 for i in (2, 5, 9)]
    c = collections.Counter(am).most_common(); return c[0][0] if c[0][1] >= 2 else am[2]


t1 = lambda v, p: v[p - 2] - v[p - 1]
REC = {}
for si, r in enumerate(rows):
    for p in PINS:
        d = dict(si=si, p=p)
        for L in LAYS:
            d[f"W_{L}"] = t1(zv(r, f"Ep{p}", L) - zv(r, f"Ep{p}_E_BANDR", L), p)
            for arm in ARMS:
                run, nm = arm.split("_", 1)
                d[f"m_{arm}_{L}"] = zv(r, f"{run}p{p}_{nm}", L) - zv(r, f"{run}p{p}", L)
        for arm in ARMS:
            d[f"p_{arm}"] = np.asarray(PR[arm][f"{si},{p}"], float)
        d["vid"] = max(float(np.abs(zv(r, f"Rp{p}_ID", L) - zv(r, f"Rp{p}", L)).max()) for L in LAYS)
        d["anch"] = (anchor(r["qk47"][f"Ep{p}"]), anchor(r["qk47"][f"Rp{p}"]))
        d["land"] = {k: v for k, v in r["land"].items() if f"p{p}" in k}
        REC[(si, p)] = d
SC = sorted({k[0] for k in REC})


def bt(fn):
    n = len(SC); est = fn(np.arange(n)); bs = [fn(rng.integers(0, n, n)) for _ in range(NB)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return dict(est=float(est), lo=float(lo), hi=float(hi))


def M(ix, fn):
    return float(np.mean([fn(REC[(SC[i], p)]) for i in ix for p in PINS]))


def prof_r(ix, arm, L="51"):
    x, y = [], []
    for i in ix:
        for p in PINS:
            d = REC[(SC[i], p)]; m = d[f"m_{arm}_{L}"]; pr = d[f"p_{arm}"]
            x += list(m - m.mean()); y += list(pr - pr.mean())
    return float(np.corrcoef(x, y)[0, 1])


res = dict(tag=TAG, n_scenes=S)
st = float(np.mean([r["land"][f"START{p}"] == p for r in rows for p in PINS]))
reach = float(np.mean([a_ == d["p"] for d in REC.values() for a_ in d["anch"]]))
vid_ok = float(np.mean([d["vid"] <= 1e-3 for d in REC.values()]))
bW = bt(lambda ix: M(ix, lambda d: d["W_51"]))
I0 = dict(t1=bool((J["t1_menu"] or 0) >= .95), START=bool(st >= .95), pin_reach=bool(reach >= .90), identity=bool(vid_ok >= .99), writer=bool(bW["lo"] > 0))
res["I0"] = dict(**I0, all=bool(all(I0.values())), t1_value=J["t1_menu"], START_acc=st, reach=reach, vid_ok=vid_ok, W=bW)
ok = res["I0"]["all"]
u1 = bt(lambda ix: -M(ix, lambda d: t1(d["m_E_SA_S2A_51"], d["p"])) / M(ix, lambda d: d["W_51"]))
res["U1"] = dict(share=u1, verdict=(("undoing the end-ward pull of the self-address heads removes part of the writer's step" if u1["est"] >= .20 and u1["lo"] > .05 else
                                     "not observed" if u1["est"] <= .05 and u1["hi"] < .15 else "small") if ok else "not read"))
u2 = bt(lambda ix: M(ix, lambda d: t1(d["m_R_SA2S40_51"], d["p"])))
res["U2"] = dict(dT1=u2, verdict=(("mimicking the end-ward pull in START moves the late address toward the predecessor" if u2["lo"] > 0 else "not observed") if ok else "not read"))
u3 = {arm: bt(lambda ix, arm=arm: prof_r(ix, arm)) for arm in ARMS}
npred = sum(int(v["est"] >= .80 and v["lo"] > .70) for v in u3.values())
res["U3"] = dict(r=u3, n_predicted=npred, verdict=(("the composition map predicts the end-pull interventions" if npred >= 3 else "partial" if npred >= 1 else "not") if ok else "not read"))
allx = range(len(SC))
res["descriptive"] = dict(
    mean_T1_change={arm: dict(measured=M(allx, lambda d, arm=arm: t1(d[f"m_{arm}_51"], d["p"])), predicted=M(allx, lambda d, arm=arm: t1(d[f"p_{arm}"], d["p"]))) for arm in ARMS},
    share_of_W={arm: M(allx, lambda d, arm=arm: t1(d[f"m_{arm}_51"], d["p"])) / M(allx, lambda d: d["W_51"]) for arm in ARMS},
    profile_r={f"L{L}": {arm: prof_r(allx, arm, L) for arm in ARMS} for L in LAYS},
    answers={arm: dict(P_pm1=float(np.mean([d["land"][f"{arm[0]}p{d['p']}_{arm.split('_', 1)[1]}"] == d["p"] - 1 for d in REC.values()])),
                       P_p=float(np.mean([d["land"][f"{arm[0]}p{d['p']}_{arm.split('_', 1)[1]}"] == d["p"] for d in REC.values()]))) for arm in ARMS})
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"]["all"], U1=[res["U1"]["verdict"], u1["est"]], U2=[res["U2"]["verdict"], u2["est"]], U3=[res["U3"]["verdict"], npred]), indent=1))
print("->", OUT)
