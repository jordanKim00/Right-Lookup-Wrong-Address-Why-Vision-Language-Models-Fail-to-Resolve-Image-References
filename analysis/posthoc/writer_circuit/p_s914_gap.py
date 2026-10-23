"""Post hoc (not registered; 10-22, after the S914 verdicts) - where does the frozen structured model's intervention miss (S914 P4) come from?
Split the miss into (a) the late state - M3's predicted late scores (B^ + W^, B^ alone, on the natural or the PIN profile) against the observed
L51 late profiles of E, E_BANDR, PIN and PIN_BANDR - and (b) the readout - answers read from the OBSERVED late profiles with a fixed softmax
whose temperature per k group is fitted on the S913 training runs only (natural E; and E + E_BANDR as a check).
Anchor margin m(z) = z(a) - max_{i != a} z(i) at the natural L47 anchor a; departure = P(x != a).
Usage: python3 p_s914_gap.py  (reads results/r3/s913_<tag>_collect.json for s913n7_27b, s913n8_27b, s914n9_27b, s914n6_27b, s914e8_27b)"""
from __future__ import annotations

import collections, importlib.util, json, sys
import numpy as np
from scipy.optimize import minimize

sys.dont_write_bytecode = True
MAIN = "/vlm/research/ces/outputs/writer_circuit_2026-10-20/main"
sys.argv = ["s913_fit", "x", "y"]
spec = importlib.util.spec_from_file_location("s913_fit", "/vlm/exp/s913_fit.py"); FIT = importlib.util.module_from_spec(spec); spec.loader.exec_module(FIT)
P = json.load(open(f"{MAIN}/s913_frozen_predictors.json"))
P3 = {int(g): {kk: np.asarray(v) for kk, v in c.items()} for g, c in P["M3"].items()}
KG = FIT.KG
ARMS = {"E": "E", "BANDR": "E_BANDR", "PIN": "PIN", "PINB": "PIN_BANDR"}


def load(tag):
    J = json.load(open(f"/vlm/results/r3/s913_{tag}_collect.json")); N = J["N"]; lm = J["late_mean"]["51"]; em = J["early_mean"]
    out = []
    for si, r in enumerate(J["rows"]):
        for k in J["ks"]:
            if f"B{k}_E" not in r["zl"]:
                continue
            d = dict(tag=tag, N=N, k=k, si=si, a=r["anchor"][str(k)])
            for nm, key in ARMS.items():
                kk = f"B{k}_{key}"
                if kk in r["zl"]:
                    d[f"z_{nm}"] = FIT.centre(np.asarray(r["zl"][kk]["51"], float)[:, lm].mean(1).mean(0)); d[f"x_{nm}"] = int(r["land"][kk])
            for src, kk in (("nat", f"N_E{k}"), ("pin", f"B{k}_PIN")):
                if kk in r[f"qk47h"]:
                    h = np.asarray(r["qk47h"][kk], float).mean(0); d[f"e47_{src}"] = FIT.centre(h[em["47"]].mean(0))
            out.append(d)
    return out


TRAIN = load("s913n7_27b") + load("s913n8_27b")
TEST = load("s914n9_27b") + load("s914n6_27b") + load("s914e8_27b")


def m3(d, src, useW=True):
    dd = dict(N=d["N"], k=d["k"], a=d["a"], e47=d[f"e47_{src}"])
    return FIT.m3_scores(dd, P3, 47, useW)


def readout(z, T, N):
    v = T * np.asarray(z); v = v - v.max(); p = np.exp(v) / np.exp(v).sum(); return FIT.mix(np.concatenate([[0.0], p]), N)


def fit_T(runs, arms):
    T = {}
    for g in range(4):
        rs = [(d, a) for d in runs if KG(d["k"]) == g for a in arms if f"z_{a}" in d]
        def nll(t):
            return -np.mean([np.log(readout(d[f"z_{a}"], t[0], d["N"])[FIT.xi(d, d[f"x_{a}"])]) for d, a in rs])
        T[g] = float(minimize(nll, np.array([1.0]), method="Nelder-Mead").x[0])
    return T


margin = lambda z, a: float(z[a - 1] - np.max(np.delete(z, a - 1)))
T_E = fit_T(TRAIN, ["E"]); T_EB = fit_T(TRAIN, ["E", "BANDR"])
res = dict(T_trainE=T_E, T_trainE_BANDR=T_EB, T_M3=P["M3_T"]["ans"])
rows = {}
for nm in ARMS:
    ds = [d for d in TEST if f"z_{nm}" in d]
    obs = np.mean([d[f"x_{nm}"] != d["a"] for d in ds])
    dep = lambda T: float(np.mean([1 - readout(d[f"z_{nm}"], T[KG(d["k"])], d["N"])[d["a"]] for d in ds]))
    src = "pin" if nm in ("PIN", "PINB") else "nat"; useW = nm in ("E", "PIN")
    pm = [m3(d, src, useW) for d in ds]
    mo = np.array([margin(d[f"z_{nm}"], d["a"]) for d in ds]); mp = np.array([margin(s, d["a"]) for s, d in zip(pm, ds)])
    tm = {g: float(v) for g, v in P["M3_T"]["ans"].items()}
    rows[nm] = dict(n=len(ds), dep_obs=float(obs), dep_from_observed_late_T_trainE=dep(T_E), dep_from_observed_late_T_trainE_BANDR=dep(T_EB),
                    dep_from_observed_late_T_M3=float(np.mean([1 - readout(d[f"z_{nm}"], tm[str(KG(d["k"]))], d["N"])[d["a"]] for d in ds])),
                    dep_M3=float(np.mean([1 - FIT.mix(np.concatenate([[0.0], np.exp(tm[str(KG(d["k"]))] * s - np.max(tm[str(KG(d["k"]))] * s)) /
                                                                      np.exp(tm[str(KG(d["k"]))] * s - np.max(tm[str(KG(d["k"]))] * s)).sum()]), d["N"])[d["a"]]
                                          for s, d in zip(pm, ds)])),
                    margin_obs=float(mo.mean()), margin_pred=float(mp.mean()), margin_slope_obs_on_pred=float(np.polyfit(mp, mo, 1)[0]),
                    margin_r=float(np.corrcoef(mp, mo)[0, 1]), R2_scores=float(1 - sum(float(np.sum((d[f"z_{nm}"] - s) ** 2)) for s, d in zip(pm, ds)) /
                                                                               sum(float(np.sum(d[f"z_{nm}"] ** 2)) for d in ds)))
res["arms"] = rows
# the interventions' changes: observed margin change vs predicted, per run
chg = {}
for a1, a2 in (("E", "BANDR"), ("E", "PIN"), ("PIN", "PINB")):
    ds = [d for d in TEST if f"z_{a1}" in d and f"z_{a2}" in d]
    s1 = {"E": ("nat", True), "BANDR": ("nat", False), "PIN": ("pin", True), "PINB": ("pin", False)}
    o = np.array([margin(d[f"z_{a2}"], d["a"]) - margin(d[f"z_{a1}"], d["a"]) for d in ds])
    p = np.array([margin(m3(d, *s1[a2]), d["a"]) - margin(m3(d, *s1[a1]), d["a"]) for d in ds])
    chg[f"{a1}->{a2}"] = dict(obs_mean=float(o.mean()), pred_mean=float(p.mean()), r=float(np.corrcoef(p, o)[0, 1]), slope_obs_on_pred=float(np.polyfit(p, o, 1)[0]))
res["margin_changes"] = chg
json.dump(res, open(f"{MAIN}/p_s914_gap.json", "w"), indent=1)
print(json.dumps(res, indent=1))
# ---- (c) readout from later late layers (observed matcher-mean profiles at L55 / L59 / L63; temperature fitted on S913 natural E only)


def load_layer(tag, L):
    J = json.load(open(f"/vlm/results/r3/s913_{tag}_collect.json")); lm = J["late_mean"][str(L)]; out = {}
    for si, r in enumerate(J["rows"]):
        for k in J["ks"]:
            for nm, key in ARMS.items():
                kk = f"B{k}_{key}"
                if kk in r["zl"] and str(L) in r["zl"][kk]:
                    out[(si, k, nm)] = FIT.centre(np.asarray(r["zl"][kk][str(L)], float)[:, lm].mean(1).mean(0))
    return out


late = {}
for L in (51, 55, 59, 63):
    Z = {tag: load_layer(tag, L) for tag in ("s913n7_27b", "s913n8_27b", "s914n9_27b", "s914n6_27b", "s914e8_27b")}
    for d in TRAIN + TEST:
        for nm in ARMS:
            if (d["si"], d["k"], nm) in Z[d["tag"]]:
                d[f"z_{nm}"] = Z[d["tag"]][(d["si"], d["k"], nm)]
    T = fit_T(TRAIN, ["E"])
    late[str(L)] = {nm: dict(dep_obs=float(np.mean([d[f"x_{nm}"] != d["a"] for d in TEST if f"z_{nm}" in d])),
                             dep_readout=float(np.mean([1 - readout(d[f"z_{nm}"], T[KG(d["k"])], d["N"])[d["a"]] for d in TEST if f"z_{nm}" in d])),
                             argmax_is_answer=float(np.mean([int(np.argmax(d[f"z_{nm}"])) + 1 == d[f"x_{nm}"] for d in TEST if f"z_{nm}" in d])))
                    for nm in ARMS}
res["late_layer_readout"] = late
# ---- (d) the early margin m47 at the anchor: natural training range vs the PIN profiles
m47 = lambda e, a: float(e[a - 1] - np.max(np.delete(e, a - 1)))
tr = np.array([m47(d["e47_nat"], d["a"]) for d in TRAIN]); te_n = np.array([m47(d["e47_nat"], d["a"]) for d in TEST]); te_p = np.array([m47(d["e47_pin"], d["a"]) for d in TEST if "e47_pin" in d])
res["m47"] = dict(train_natural=np.percentile(tr, [5, 50, 95]).tolist(), test_natural=np.percentile(te_n, [5, 50, 95]).tolist(), test_pin=np.percentile(te_p, [5, 50, 95]).tolist(),
                  pin_above_train_p95=float(np.mean(te_p > np.percentile(tr, 95))))
json.dump(res, open(f"{MAIN}/p_s914_gap.json", "w"), indent=1)
print(json.dumps(dict(late_layer_readout=late, m47=res["m47"]), indent=1))
