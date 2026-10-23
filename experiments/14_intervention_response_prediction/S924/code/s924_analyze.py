"""S924 analysis (rules fixed before the data; knowledge/DECISIONS.md S924 entry; user 10-23: keep going until intervention prediction clearly
works).  Input results/r3/s924_<tag>_collect.json (exp/s924_collect.py, N8 'from the last', fresh scenes).  Frozen: MC and the anchor-offset
prior OFF (main/s924_frozen_MC.json, exp/s924_fit.py on natural runs and the interventions E_BANDR / PIN / PIN_BANDR of S913 / S914 / S923 only;
hash checked).  Novel interventions (never in training), all at END(k)'s natural L47 anchor a: PINm (address part of the L35-47 read-row
queries <- START(a - 1)'s), PINp (<- START(a + 1)'s), PINt (<- START(t)'s, t = N + 1 - k), PINtB (PINt + the band 46-49 read-row outputs
<- START(t)'s); seen types on fresh scenes: E_BANDR, PIN, PIN_BANDR.  Each prediction unit: the address read (the L47 profile of the run, or
for a band-removed arm of its writer-present twin), its anchor and sharpness, the writer flag, k, N (exp/s924_fit.py).
Rates per arm (pooled; and per k cell): acc = P(answer = t), stay = P(answer = the unit's anchor), step = P(answer = anchor - 1).
Scene bootstrap 4,000 (seed 924).
I0: t1 >= .95; natural START(q) answers q >= .95 (q < N); the novel pins reach their address (L47 anchor = the pinned position in >= .90 of
  PINm, PINp, PINt runs); the frozen hash matches; >= 200 natural END runs.  Nothing is read if I0 fails.
N1 (novel interventions): over the 4 novel arms x 3 rates (12 pooled numbers) the largest |predicted - observed| <= .10 and the correlation over
  (k, arm, rate) cells with >= 10 runs >= .85 -> 'the mechanism-calibrated predictor predicts novel interventions'; >= 9 of 12 within .10 and
  r >= .70 -> 'partial'; else 'not'.
N2 (against the anchor-and-k offset prior on the novel arms): d = NLL(OFF) - NLL(MC) >= .05 with the lower bound > 0 -> 'the graded profile and
  sharpness add to the anchor, k and writer state'; |d| < .05 with the interval inside [-.05, .05] -> 'the anchor, k and writer state carry the
  prediction'; d <= -.05 -> 'the offset prior predicts better'; else 'partial'.
N3 (seen intervention types on fresh scenes): for E -> E_BANDR and E -> PIN, |predicted - observed| change of acc and of stay (4 numbers)
  <= .10 and the (k, rate) cell correlation >= .70 for both -> 'predicts writer removal and sharpening on fresh scenes'; one -> 'partial';
  none -> 'not'.
Descriptive: per arm and k predicted (MC, OFF) and observed rates; NLLs per arm; M3 (S913) on E / E_BANDR / PIN for reference.
Usage: python3 s924_analyze.py TAG [OUT]"""
from __future__ import annotations

import collections, hashlib, importlib.util, json, sys
import numpy as np

sys.dont_write_bytecode = True
TAG = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else f"/vlm/results/r3/s924_analysis_{TAG}.json"
MAIN = "/vlm/research/ces/outputs/writer_circuit_2026-10-20/main"
HASH_MC = "c61393992ed3484a"
rng = np.random.default_rng(924); NB = 4000
spec = importlib.util.spec_from_file_location("s924_fit", "/vlm/exp/s924_fit.py"); FIT = importlib.util.module_from_spec(spec); spec.loader.exec_module(FIT)
PF = json.load(open(f"{MAIN}/s924_frozen_MC.json")); hmc = hashlib.sha256(open(f"{MAIN}/s924_frozen_MC.json", "rb").read()).hexdigest()[:16]
W = np.asarray(PF["MC"]); OFF = PF["OFF"]
NOVEL = dict(PINm=("B{k}_PINm", "B{k}_PINm", 1), PINp=("B{k}_PINp", "B{k}_PINp", 1), PINt=("B{k}_PINt", "B{k}_PINt", 1), PINtB=("B{k}_PINt", "B{k}_PINtB", 0))
SEEN = dict(E=("B{k}_E", "B{k}_E", 1), B=("B{k}_E", "B{k}_E_BANDR", 0), PIN=("B{k}_PIN", "B{k}_PIN", 1), PINB=("B{k}_PIN", "B{k}_PIN_BANDR", 0))
J = json.load(open(f"/vlm/results/r3/s924_{TAG}_collect.json")); N = J["N"]; nS = len(J["rows"])
U = FIT.load(TAG, {**SEEN, **NOVEL})
for u in U:
    u["p_MC"] = FIT.predict(u, W); u["p_OFF"] = FIT.predict_off(u, OFF)
RATES = ("acc", "stay", "step")


def rate_obs(u, r):
    return float(u["x"] == (u["t"] if r == "acc" else u["a"] if r == "stay" else u["a"] - 1))


def rate_pred(u, r, m):
    b = u["t"] if r == "acc" else u["a"] if r == "stay" else u["a"] - 1
    return float(u[f"p_{m}"][b]) if 1 <= b <= u["N"] else 0.0


def sel(ix, arms=None, k=None):
    cnt = collections.Counter(int(i) for i in ix)
    return [u for u in U for _ in range(cnt.get(u["si"], 0)) if (arms is None or u["arm"] in arms) and (k is None or u["k"] == k)]


def mean(v):
    return float(np.mean(v)) if len(v) else float("nan")


def boot(fn):
    est = fn(np.arange(nS)); bs = [fn(rng.integers(0, nS, nS)) for _ in range(NB)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return dict(est=float(est), lo=float(lo), hi=float(hi))


allx = np.arange(nS)
res = dict(tag=TAG, hash_MC=hmc)
st = float(np.mean([r["land"][f"N_S{q}"] == q for r in J["rows"] for q in range(1, N)]))
reach_all = []
for r in J["rows"]:
    for k in J["ks"]:
        if f"B{k}_E" not in r["land"]:
            continue
        a = r["anchor"][str(k)]
        for arm, pos in (("PINm", a - 1), ("PINp", a + 1), ("PINt", N + 1 - k)):
            key = f"B{k}_{arm}"
            if key in r["land"]:
                reach_all.append(float(FIT.anchor_of(r["qk47"][key]) == pos))
nE = len([u for u in U if u["arm"] == "E"])
I0 = dict(t1=bool((J["t1_menu"] or 0) >= .95), START=bool(st >= .95), pins_reach=bool(mean(reach_all) >= .90), frozen=bool(hmc == HASH_MC), size=bool(nE >= 200))
res["I0"] = dict(**I0, all=bool(all(I0.values())), t1_value=J["t1_menu"], START_acc=st, pins_reach_value=mean(reach_all), n_E=nE)
ok = res["I0"]["all"]
# N1
pooled = {}
for arm in NOVEL:
    us = [u for u in U if u["arm"] == arm]
    for r in RATES:
        pooled[f"{arm}_{r}"] = dict(pred=mean([rate_pred(u, r, "MC") for u in us]), obs=mean([rate_obs(u, r) for u in us]), n=len(us))
errs = [abs(v["pred"] - v["obs"]) for v in pooled.values()]
cp, co = [], []
for arm in NOVEL:
    for k in J["ks"]:
        us = [u for u in U if u["arm"] == arm and u["k"] == k]
        if len(us) >= 10:
            for r in RATES:
                cp.append(mean([rate_pred(u, r, "MC") for u in us])); co.append(mean([rate_obs(u, r) for u in us]))
r1 = float(np.corrcoef(cp, co)[0, 1]) if len(cp) >= 3 else float("nan")
within = int(sum(e <= .10 for e in errs))
v1 = ("the mechanism-calibrated predictor predicts novel interventions" if max(errs) <= .10 and r1 >= .85 else
      "partial" if within >= 9 and r1 >= .70 else "not")
res["N1"] = dict(pooled=pooled, max_err=float(max(errs)), within_10=within, cell_r=r1, n_cells=len(cp), verdict=(v1 if ok else "not read"))
# N2
nll = lambda us, m: mean([-np.log(u[f"p_{m}"][FIT.xi(u)]) for u in us])
d2 = boot(lambda ix: nll(sel(ix, NOVEL), "OFF") - nll(sel(ix, NOVEL), "MC"))
v2 = ("the graded profile and sharpness add to the anchor, k and writer state" if d2["est"] >= .05 and d2["lo"] > 0 else
      "the anchor, k and writer state carry the prediction" if abs(d2["est"]) < .05 and d2["lo"] > -.05 and d2["hi"] < .05 else
      "the offset prior predicts better" if d2["est"] <= -.05 else "partial")
res["N2"] = dict(d=d2, NLL=dict(MC=nll(sel(allx, NOVEL), "MC"), OFF=nll(sel(allx, NOVEL), "OFF")), verdict=(v2 if ok else "not read"))
# N3


def change_check(arm):
    out = {}
    for r in ("acc", "stay"):
        e_us = [u for u in U if u["arm"] == "E"]; x_us = [u for u in U if u["arm"] == arm]
        pred = mean([rate_pred(u, r, "MC") for u in x_us]) - mean([rate_pred(u, r, "MC") for u in e_us])
        obs = mean([rate_obs(u, r) for u in x_us]) - mean([rate_obs(u, r) for u in e_us])
        out[r] = dict(pred=pred, obs=obs, err=abs(pred - obs))
    cp_, co_ = [], []
    for k in J["ks"]:
        e_us = [u for u in U if u["arm"] == "E" and u["k"] == k]; x_us = [u for u in U if u["arm"] == arm and u["k"] == k]
        if len(x_us) >= 10:
            for r in ("acc", "stay"):
                cp_.append(mean([rate_pred(u, r, "MC") for u in x_us]) - mean([rate_pred(u, r, "MC") for u in e_us]))
                co_.append(mean([rate_obs(u, r) for u in x_us]) - mean([rate_obs(u, r) for u in e_us]))
    rr = float(np.corrcoef(cp_, co_)[0, 1]) if len(cp_) >= 3 else float("nan")
    return dict(changes=out, cell_r=rr, ok=bool(all(v["err"] <= .10 for v in out.values()) and rr >= .70))


cw, cpn = change_check("B"), change_check("PIN")
nok = int(cw["ok"]) + int(cpn["ok"])
res["N3"] = dict(writer_removal=cw, sharpening=cpn, verdict=(("predicts writer removal and sharpening on fresh scenes" if nok == 2 else "partial" if nok == 1 else "not")
                                                              if ok else "not read"))
desc = {}
for arm in list(SEEN) + list(NOVEL):
    us = [u for u in U if u["arm"] == arm]
    desc[arm] = dict(n=len(us), NLL=dict(MC=nll(us, "MC"), OFF=nll(us, "OFF")),
                     rates={r: dict(obs=mean([rate_obs(u, r) for u in us]), MC=mean([rate_pred(u, r, "MC") for u in us]), OFF=mean([rate_pred(u, r, "OFF") for u in us]))
                            for r in RATES},
                     by_k={str(k): {r: dict(obs=mean([rate_obs(u, r) for u in us if u["k"] == k]), MC=mean([rate_pred(u, r, "MC") for u in us if u["k"] == k]))
                                    for r in RATES} for k in J["ks"]})
res["descriptive"] = desc
json.dump(res, open(OUT, "w"), indent=1, default=float)
print(json.dumps(dict(I0=res["I0"]["all"], N1=[res["N1"]["verdict"], res["N1"]["max_err"], within, r1], N2=[res["N2"]["verdict"], d2["est"]],
                      N3=[res["N3"]["verdict"], cw["changes"], cw["cell_r"], cpn["changes"], cpn["cell_r"]]), indent=1, default=float))
print("->", OUT)
