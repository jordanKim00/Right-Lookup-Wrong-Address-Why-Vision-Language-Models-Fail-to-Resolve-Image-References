"""S929 analysis (rules fixed before the S929 runs; knowledge/DECISIONS.md S929 entry; advisor review 5: does the identified circuit carry the
intervention response?).  Inputs: results/r3/s929_<tag>_circ.json (exp/s929_circ.py) and S927's results/r3/s927_<s927tag>_lr.json for the same
scenes; outcome file from the S929 file.  Restrictions of the backward flow (values unchanged): R0 full; R1 read rows only (positions outside the
read rows detached at layers >= 35); R2 = R1 + after the band only the late full-attention reads (DeltaNet mixers and MLPs at 50-63 detached).
Predictions are first order (S927 LR): pins z + T_R[q]; band-replacing arms z + Dp_R[p] + BAND_R[q] (B = BAND[a], PINB = Dp[a] + BAND[a],
PINtB = Dp[t] + BAND[t]).  Rates acc / stay / step with the unit anchor of exp/s924_analyze.py; run agreement A = P(predicted = observed).
C0 (instrument): R0's T reproduces S927's T for the same runs (correlation of all entries >= .99).  Nothing else is read if C0 fails.
C1 (carriage, pins): median over runs of <T_R2, T_R0> / <T_R0, T_R0> (picture logits centred over b, all q) >= .80 -> 'the identified circuit
  carries the first-order address response'; >= .50 -> 'partial'; else 'not'.  (R1's share descriptive.)
C2 (prediction, pins PIN / PINm / PINp / PINt): max over the 12 rates of |R2 - observed| <= the same for R0 + .05, and A_R2 >= A_R0 - .05 ->
  'circuit-restricted prediction matches the full model'; one of the two -> 'partial'; neither -> 'not'.
C3 (band-replacing arms B / PINB / PINtB): carriage share of (Dp + BAND) under R2 vs R0 (median) and the C2 comparison on their 9 rates:
  share >= .80 and both C2 conditions -> 'the band's response reaches the answer through the late FA reads'; share >= .50 or one condition ->
  'partial'; else 'not'.
Usage: python3 s929_analyze.py TAG S927TAG [OUT]"""
from __future__ import annotations

import importlib.util, json, sys
import numpy as np

sys.dont_write_bytecode = True
TAG, S27 = sys.argv[1], sys.argv[2]
OUT = sys.argv[3] if len(sys.argv) > 3 else f"/vlm/results/r3/s929_analysis_{TAG}.json"
spec = importlib.util.spec_from_file_location("s924_fit", "/vlm/exp/s924_fit.py"); FIT = importlib.util.module_from_spec(spec); spec.loader.exec_module(FIT)
L = json.load(open(f"/vlm/results/r3/s929_{TAG}_circ.json")); C = json.load(open(L["collect"])); N = C["N"]
R7 = {rec["si"]: rec for rec in json.load(open(f"/vlm/results/r3/s927_{S27}_lr.json"))["rows"]}
PINS = dict(PIN=("PIN", "PIN", 0), PINm=("PINm", "PINm", -1), PINp=("PINp", "PINp", 1), PINt=("PINt", "PINt", None))
BANDS = dict(B=("E", "E_BANDR"), PINB=("PIN", "PIN_BANDR"), PINtB=("PINt", "PINtB"))
cen = lambda A: A[:, :N] - A[:, :N].mean(1, keepdims=True)
c0a, c0b = [], []; shareT = {"R1": [], "R2": []}; shareB = []; U = []
for rec in L["rows"]:
    r = C["rows"][rec["si"]]; r7 = R7.get(rec["si"])
    for k_, o in rec["k"].items():
        k = int(k_); t = N + 1 - k; an = o["anchor"]; z = np.array(o["z"])
        T = {"R0": np.array(o["T"]), "R1": np.array(o["T_R1"]), "R2": np.array(o["T_R2"])}
        Dp = {"R0": np.array(o["Dp"]), "R2": np.array(o["Dp_R2"])}; BD = {"R0": np.array(o["BAND"]), "R2": np.array(o["BAND_R2"])}
        if r7 is not None and k_ in r7["k"]:
            c0a += np.array(r7["k"][k_]["T"]).ravel().tolist(); c0b += T["R0"].ravel().tolist()
        den = (cen(T["R0"]) ** 2).sum()
        if den > 0:
            for rv in ("R1", "R2"):
                shareT[rv].append(float((cen(T[rv]) * cen(T["R0"])).sum() / den))
        G0 = np.stack([Dp["R0"][q - 1] + BD["R0"][q - 1] for q in {an, t}]); G2 = np.stack([Dp["R2"][q - 1] + BD["R2"][q - 1] for q in {an, t}])
        if (cen(G0) ** 2).sum() > 0:
            shareB.append(float((cen(G2) * cen(G0)).sum() / (cen(G0) ** 2).sum()))
        for arm, (ka, kx, d) in PINS.items():
            q = t if d is None else an + d
            if not 1 <= q <= N or f"B{k}_{kx}" not in r["land"]:
                continue
            U.append(dict(arm=arm, grp="pin", k=k, t=t, a=FIT.anchor_of(r["qk47"][f"B{k}_{ka}"]), x=int(r["land"][f"B{k}_{kx}"]),
                          xp={rv: int(np.argmax(z + T[rv][q - 1])) + 1 for rv in ("R0", "R1", "R2")}))
        pq = dict(B=(None, an), PINB=(an, an), PINtB=(t, t))
        for arm, (ka, kx) in BANDS.items():
            p, q = pq[arm]
            if f"B{k}_{kx}" not in r["land"]:
                continue
            xp = {rv: int(np.argmax(z + (Dp[rv][p - 1] if p is not None else 0.0) + BD[rv][q - 1])) + 1 for rv in ("R0", "R2")}
            U.append(dict(arm=arm, grp="band", k=k, t=t, a=FIT.anchor_of(r["qk47"][f"B{k}_{ka}"]), x=int(r["land"][f"B{k}_{kx}"]), xp=xp))
hit = lambda u, r, x: float(x == (u["t"] if r == "acc" else u["a"] if r == "stay" else u["a"] - 1))


def compare(arms, rv="R2"):
    rates = {}; mx = {"R0": 0.0, rv: 0.0}
    for arm in arms:
        us = [u for u in U if u["arm"] == arm]
        for r in ("acc", "stay", "step"):
            o_ = np.mean([hit(u, r, u["x"]) for u in us])
            for m in ("R0", rv):
                p_ = np.mean([hit(u, r, u["xp"][m]) for u in us]); mx[m] = max(mx[m], abs(p_ - o_)); rates[f"{arm}_{r}_{m}"] = float(p_)
            rates[f"{arm}_{r}_obs"] = float(o_)
    us = [u for u in U if u["arm"] in arms]
    A = {m: float(np.mean([u["xp"][m] == u["x"] for u in us])) for m in ("R0", rv)}
    return dict(max_err=mx, agree=A, rates=rates, cond_err=bool(mx[rv] <= mx["R0"] + .05), cond_agree=bool(A[rv] >= A["R0"] - .05))


res = dict(tag=TAG, s927=S27, n_runs=len(shareT["R2"]))
c0 = float(np.corrcoef(c0a, c0b)[0, 1]) if c0a else float("nan")
res["C0"] = dict(corr=c0, holds=bool(c0 >= .99))
if not res["C0"]["holds"]:
    json.dump(res, open(OUT, "w"), indent=1); print(json.dumps(res, indent=1)); sys.exit(0)
s2 = float(np.median(shareT["R2"])); s1 = float(np.median(shareT["R1"]))
res["C1"] = dict(share_R2=s2, share_R1=s1, verdict=("the identified circuit carries the first-order address response" if s2 >= .80 else "partial" if s2 >= .50 else "not"))
cp = compare(tuple(PINS)); nc = int(cp["cond_err"]) + int(cp["cond_agree"])
res["C2"] = dict(**cp, verdict=("circuit-restricted prediction matches the full model" if nc == 2 else "partial" if nc == 1 else "not"))
cb = compare(tuple(BANDS)); sb = float(np.median(shareB)); nb = int(cb["cond_err"]) + int(cb["cond_agree"])
res["C3"] = dict(share_R2=sb, **cb, verdict=("the band's response reaches the answer through the late FA reads" if sb >= .80 and nb == 2 else
                                             "partial" if sb >= .50 or nb >= 1 else "not"))
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(C0=res["C0"], C1=res["C1"], C2=[res["C2"]["verdict"], cp["max_err"], cp["agree"]], C3=[res["C3"]["verdict"], sb, cb["max_err"], cb["agree"]]), indent=1))
print("->", OUT)
