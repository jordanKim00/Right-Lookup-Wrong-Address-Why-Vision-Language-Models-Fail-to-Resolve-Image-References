"""S927 analysis (rules fixed before the comparison; knowledge/DECISIONS.md S927 entry; user 10-23 '모델 내부 그라디언트 ... 수학적으로 우아하게').
Predictor LR (exp/s927_lr.py): no fitted parameter - the first-order response of the model's own menu logits to the exact local change of each
intervention at the circuit sites; predicted answer = argmax over the menu words of z + dz (z: float32 menu logits from the final hidden state; positions 1..N
pictures, larger = distractor); the no-change baseline uses the natural answer x_nat.
Inputs: results/r3/s927_<lrtag>_lr.json (natural runs reproduced from the outcome file's seed; anchors read from it) and the outcome file
(results/r3/s924_<tag>_collect.json: S924 N8 or S925 N7).  Arms: PIN, PINm, PINp, PINt (pins), B = E_BANDR, PINB = PIN_BANDR, PINtB (band
replaced); unit anchor = the L47 anchor of the arm's address run (exp/s924_analyze.py); acc = P(answer = t), stay = P(answer = the unit's
anchor), step = P(answer = anchor - 1).  Scene bootstrap 4,000 (seed 927).
I0: natural answers reproduced (x_nat, the collector's definition, equals one of the outcome file's two natural END answers N_E / B_E -
  they differ at bf16 ties) in >= .95 of runs; the vectorised pin equals QPath._qsub
  (max abs difference <= 1e-3 where checked).  Nothing is read if I0 fails.
L1 (the S924 N1 test for LR): over PINm, PINp, PINt, PINtB x (acc, stay, step) (12 pooled numbers) the largest |predicted - observed| <= .10
  and the correlation over (k, arm, rate) cells with >= 10 runs >= .85 -> 'the parameter-free linear response predicts the interventions';
  >= 9 of 12 within .10 and r >= .70 -> 'partial'; else 'not'.
L2 (run level): over the 7 arms pooled, agreement A = P(predicted answer = observed answer) against the no-change baseline A0 = P(natural
  answer = observed answer): A - A0 >= .15 with the lower bound > 0 -> 'LR predicts which runs change and where they go'; A - A0 > 0 with the
  lower bound > 0 -> 'partial'; else 'not'.
L3 (the S924 N3 test for LR: writer removal B and sharpening PIN against E): |predicted - observed| change of acc and of stay (4 numbers) <= .10
  and the (k, rate) cell correlation >= .70 for both -> 'predicts writer removal and sharpening'; one -> 'partial'; none -> 'not'.
Descriptive: per arm and k predicted / observed rates and agreement; the logit-change correlation and slope (observed vs linear) per arm.
Usage: python3 s927_analyze.py LRTAG [OUT]"""
from __future__ import annotations

import collections, importlib.util, json, sys
import numpy as np

sys.dont_write_bytecode = True
TAG = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else f"/vlm/results/r3/s927_analysis_{TAG}.json"
spec = importlib.util.spec_from_file_location("s924_fit", "/vlm/exp/s924_fit.py"); FIT = importlib.util.module_from_spec(spec); spec.loader.exec_module(FIT)
rng = np.random.default_rng(927); NB = 4000
L = json.load(open(f"/vlm/results/r3/s927_{TAG}_lr.json")); C = json.load(open(L["collect"])); N = C["N"]
ARMS = dict(PIN=("PIN", "PIN"), PINm=("PINm", "PINm"), PINp=("PINp", "PINp"), PINt=("PINt", "PINt"),
            B=("E", "E_BANDR"), PINB=("PIN", "PIN_BANDR"), PINtB=("PINt", "PINtB"))
NOVEL = ("PINm", "PINp", "PINt", "PINtB"); RATES = ("acc", "stay", "step")
U = []; rep = []; chk = []
for rec in L["rows"]:
    r = C["rows"][rec["si"]]
    for k_, o in rec["k"].items():
        k = int(k_); t = N + 1 - k; z = np.array(o["z"]); xn = int(o["x_nat"]); an = o["anchor"]
        rep.append(xn in (r["land"][f"B{k}_E"], r["land"].get(f"N_E{k}"))); chk.append(o.get("pin_check_maxabs", 0.0))
        Tm, Dp, BD = np.array(o["T"]), np.array(o["Dp"]), np.array(o["BAND"])
        dz = dict(PIN=Tm[an - 1], PINt=Tm[t - 1], B=BD[an - 1], PINB=Dp[an - 1] + BD[an - 1], PINtB=Dp[t - 1] + BD[t - 1])
        if an >= 2:
            dz["PINm"] = Tm[an - 2]
        if an <= N - 1:
            dz["PINp"] = Tm[an]
        U.append(dict(arm="E", k=k, t=t, a=FIT.anchor_of(r["qk47"][f"B{k}_E"]), x=int(r["land"][f"B{k}_E"]), xp=xn, xn=xn, si=rec["si"]))
        for arm, (ka, kx) in ARMS.items():
            if arm not in dz or f"B{k}_{kx}" not in r["land"]:
                continue
            xp = int(np.argmax(z + dz[arm])) + 1
            U.append(dict(arm=arm, k=k, t=t, a=FIT.anchor_of(r["qk47"][f"B{k}_{ka}"]), x=int(r["land"][f"B{k}_{kx}"]), xp=xp, xn=xn, si=rec["si"],
                          dzo=(np.array(r["mlog"][f"B{k}_{kx}"]) - np.array(r["mlog"][f"B{k}_E"])).tolist(), dzl=dz[arm][:N].tolist()))
nS = len(L["rows"]); allx = np.arange(nS)
hit = lambda u, r, x: float(x == (u["t"] if r == "acc" else u["a"] if r == "stay" else u["a"] - 1))
mean = lambda v: float(np.mean(v)) if len(v) else float("nan")


def sel(ix, arms):
    cnt = collections.Counter(int(i) for i in ix)
    return [u for u in U for _ in range(cnt.get(u["si"], 0)) if u["arm"] in arms]


def boot(fn):
    est = fn(allx); bs = [fn(rng.integers(0, nS, nS)) for _ in range(NB)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return dict(est=float(est), lo=float(lo), hi=float(hi))


res = dict(tag=TAG, collect=L["collect"], n_scenes=nS)
I0 = dict(reproduced=bool(mean(rep) >= .95), pin_exact=bool(max(chk or [0.0]) <= 1e-3))
res["I0"] = dict(**I0, all=bool(all(I0.values())), reproduced_value=mean(rep), pin_check=max(chk or [0.0]))
ok = res["I0"]["all"]
# L1
pooled = {f"{arm}_{r}": dict(pred=mean([hit(u, r, u["xp"]) for u in U if u["arm"] == arm]), obs=mean([hit(u, r, u["x"]) for u in U if u["arm"] == arm]))
          for arm in NOVEL for r in RATES}
errs = [abs(v["pred"] - v["obs"]) for v in pooled.values()]
cp, co = [], []
for arm in NOVEL:
    for k in C["ks"]:
        us = [u for u in U if u["arm"] == arm and u["k"] == k]
        if len(us) >= 10:
            for r in RATES:
                cp.append(mean([hit(u, r, u["xp"]) for u in us])); co.append(mean([hit(u, r, u["x"]) for u in us]))
r1 = float(np.corrcoef(cp, co)[0, 1]) if len(cp) >= 3 else float("nan"); within = int(sum(e <= .10 for e in errs))
v1 = ("the parameter-free linear response predicts the interventions" if max(errs) <= .10 and r1 >= .85 else
      "partial" if within >= 9 and r1 >= .70 else "not")
res["L1"] = dict(pooled=pooled, max_err=float(max(errs)), within_10=within, cell_r=r1, n_cells=len(cp), verdict=(v1 if ok else "not read"))
# L2
ARM7 = tuple(ARMS)
dA = boot(lambda ix: mean([float(u["xp"] == u["x"]) for u in sel(ix, ARM7)]) - mean([float(u["xn"] == u["x"]) for u in sel(ix, ARM7)]))
v2 = ("LR predicts which runs change and where they go" if dA["est"] >= .15 and dA["lo"] > 0 else "partial" if dA["est"] > 0 and dA["lo"] > 0 else "not")
res["L2"] = dict(A=mean([float(u["xp"] == u["x"]) for u in U if u["arm"] in ARM7]), A0=mean([float(u["xn"] == u["x"]) for u in U if u["arm"] in ARM7]),
                 diff=dA, verdict=(v2 if ok else "not read"))
# L3


def change_check(arm):
    out = {}
    e_us = [u for u in U if u["arm"] == "E"]; x_us = [u for u in U if u["arm"] == arm]
    for r in ("acc", "stay"):
        pred = mean([hit(u, r, u["xp"]) for u in x_us]) - mean([hit(u, r, u["x"]) for u in e_us])
        obs = mean([hit(u, r, u["x"]) for u in x_us]) - mean([hit(u, r, u["x"]) for u in e_us])
        out[r] = dict(pred=pred, obs=obs, err=abs(pred - obs))
    cp_, co_ = [], []
    for k in C["ks"]:
        e_k = [u for u in e_us if u["k"] == k]; x_k = [u for u in x_us if u["k"] == k]
        if len(x_k) >= 10:
            for r in ("acc", "stay"):
                cp_.append(mean([hit(u, r, u["xp"]) for u in x_k]) - mean([hit(u, r, u["x"]) for u in e_k]))
                co_.append(mean([hit(u, r, u["x"]) for u in x_k]) - mean([hit(u, r, u["x"]) for u in e_k]))
    rr = float(np.corrcoef(cp_, co_)[0, 1]) if len(cp_) >= 3 else float("nan")
    return dict(changes=out, cell_r=rr, ok=bool(all(v["err"] <= .10 for v in out.values()) and rr >= .70))


cw, cs = change_check("B"), change_check("PIN"); nok = int(cw["ok"]) + int(cs["ok"])
res["L3"] = dict(writer_removal=cw, sharpening=cs, verdict=(("predicts writer removal and sharpening" if nok == 2 else "partial" if nok == 1 else "not")
                                                           if ok else "not read"))
desc = {}
for arm in ARMS:
    us = [u for u in U if u["arm"] == arm]
    if not us:
        continue
    dzo = np.array([u["dzo"] for u in us]); dzl = np.array([u["dzl"] for u in us])
    desc[arm] = dict(n=len(us), agree=mean([float(u["xp"] == u["x"]) for u in us]), nochange=mean([float(u["xn"] == u["x"]) for u in us]),
                     dz_corr=float(np.corrcoef(dzo.ravel(), dzl.ravel())[0, 1]), dz_slope=float((dzo * dzl).sum() / (dzl * dzl).sum()),
                     rates={r: dict(pred=mean([hit(u, r, u["xp"]) for u in us]), obs=mean([hit(u, r, u["x"]) for u in us])) for r in RATES},
                     by_k={str(k): {r: [mean([hit(u, r, u["xp"]) for u in us if u["k"] == k]), mean([hit(u, r, u["x"]) for u in us if u["k"] == k])]
                                    for r in RATES} for k in C["ks"]})
res["descriptive"] = desc
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"], L1=[res["L1"]["verdict"], res["L1"]["max_err"], within, r1], L2=[res["L2"]["verdict"], res["L2"]["A"], res["L2"]["A0"], dA],
                      L3=[res["L3"]["verdict"], cw["changes"], cw["cell_r"], cs["changes"], cs["cell_r"]]), indent=1))
print("->", OUT)
