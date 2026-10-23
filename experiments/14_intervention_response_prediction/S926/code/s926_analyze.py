"""S926 analysis (registered in knowledge/DECISIONS.md before the outcome data) - prospective test, on fresh scenes at two N with four
intervention types never run before S926, of the predictor SEP-HERM (no fitted parameter; exp/s928h_quad.py --arms B,Bt,Bm,Bp):
  every intervention's effect = the integral of the model's own gradient along the intervention's path (S927 / S928);
  pin-only arms: first order with the exact local change at the pinned readers (S927 LR: dz = T[q]);
  band-replacing arms: the pin part to first order around the band (Dp[p]) plus the band part by the quadratic along the band's own path
  (band read-row outputs set to natural + e (START(q) - END), held; HERM_q = 4 (z(e = 1/2) - z) - BAND[q]; the full replacement e = 1 is never
  run): dz = Dp[p] + HERM_q;  S927's LR (BAND[q] in place of HERM_q) is reported alongside (descriptive).
Inputs per N: results/r3/s928h_<tag>_quad.json (the predictor on the S926 scenes; anchors only from the outcome file) and
results/r3/s926_<tag>_collect.json.
Arms (unit anchor = the L47 anchor of the arm's address run, exp/s924_analyze.py): new types PINm2 (U <- START(a - 2)), PINp2 (a + 2), PINmB
(a - 1 with the band <- START(a - 1)), PINpB (a + 1, START(a + 1)); S924's set PINm, PINp, PINt, PINtB; writer removal B = E_BANDR, PINB, and
sharpening PIN.  LR's change for each arm is a sum of transfer-matrix rows: PINm2 = T[a - 2], PINp2 = T[a + 2], PINmB = Dp[a - 1] + BAND[a - 1],
PINpB = Dp[a + 1] + BAND[a + 1], PINm = T[a - 1], PINp = T[a + 1], PINt = T[t], PINtB = Dp[t] + BAND[t], B = BAND[a], PINB = Dp[a] + BAND[a],
PIN = T[a] (LR; for SEP-HERM every BAND[q] is replaced by HERM_q); predicted answer = argmax of z + dz over the menu words (z: float32 menu
logits).  Verdicts below are for SEP-HERM; the same numbers for LR are descriptive.  Scene bootstrap 4,000 (seed 926).
Per N:
I0: natural answers reproduced (x_nat equals one of the outcome file's two natural END answers) >= .95; the vectorised pin equals
  QPath._qsub (<= 1e-3); t1 >= .95; START(q) answers q >= .95 (q < N); the new pins reach their address (L47 anchor = the pinned position in
  >= .90 of PINm2 / PINp2 runs); every band-replacing arm has its band-only half node (SEP-HERM).
P1 (new types): over PINm2, PINp2, PINmB, PINpB x (acc, stay, step) (12 pooled numbers) the largest |predicted - observed| <= .10 and the
  (k, arm, rate) cell correlation (cells >= 10 runs) >= .85 -> 'predicted'; >= 9 of 12 within .10 and r >= .70 -> 'partial'; else 'not'.
P2 (S924's set on fresh scenes): the same on PINm, PINp, PINt, PINtB.
P3 (run level, all 11 arms): A - A0 >= .15 with the lower bound > 0 -> 'predicts which runs change and where'; > 0 with lower > 0 -> 'partial'.
P4 (writer removal and sharpening, the S924 N3 test): both within .10 and cell r >= .70 -> 'predicted'; one -> 'partial'; none -> 'not'.
Joint: P1 'predicted' at both N -> 'the parameter-free linear response predicts novel interventions'.
Usage: python3 s926_analyze.py TAG [OUT] [SEP-HERM | LR]"""
from __future__ import annotations

import collections, importlib.util, json, sys
import numpy as np

sys.dont_write_bytecode = True
TAG = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else f"/vlm/results/r3/s926_analysis_{TAG}.json"
spec = importlib.util.spec_from_file_location("s924_fit", "/vlm/exp/s924_fit.py"); FIT = importlib.util.module_from_spec(spec); spec.loader.exec_module(FIT)
rng = np.random.default_rng(926); NB = 4000
L = json.load(open(f"/vlm/results/r3/s928h_{TAG}_quad.json")); C = json.load(open(L["collect"])); N = C["N"]
ARMS = dict(PINm2=("PINm2", "PINm2"), PINp2=("PINp2", "PINp2"), PINmB=("PINm", "PINmB"), PINpB=("PINp", "PINpB"),
            PINm=("PINm", "PINm"), PINp=("PINp", "PINp"), PINt=("PINt", "PINt"), PINtB=("PINt", "PINtB"),
            B=("E", "E_BANDR"), PINB=("PIN", "PIN_BANDR"), PIN=("PIN", "PIN"))
NEW = ("PINm2", "PINp2", "PINmB", "PINpB"); OLD = ("PINm", "PINp", "PINt", "PINtB"); RATES = ("acc", "stay", "step")
U = []; rep = []; chk = []; reach = []; HN = []; MISS = []
MODE = sys.argv[3] if len(sys.argv) > 3 else "SEP-HERM"            # "LR" for the descriptive comparison
for rec in L["rows"]:
    r = C["rows"][rec["si"]]
    for k_, o in rec["k"].items():
        k = int(k_); t = N + 1 - k; z = np.array(o["z"]); xn = int(o["x_nat"]); an = o["anchor"]
        rep.append(xn in (r["land"][f"B{k}_E"], r["land"].get(f"N_E{k}"))); chk.append(o.get("pin_check_maxabs", 0.0))
        Tm, Dp, BD0 = np.array(o["T"]), np.array(o["Dp"]), np.array(o["BAND"]); Qd = o.get("quad", {})
        BD = BD0.copy(); nherm = 0; hq = {}
        for q in range(1, N + 1):                                                     # HERM_q from the band-only half node at q (any name with that q)
            nm = next((n_ for n_, qq in (("B", an), ("Bt", t), ("Bm", an - 1), ("Bp", an + 1)) if qq == q and n_ in Qd), None)
            hq[q] = nm is not None
            if nm is not None and MODE == "SEP-HERM":
                BD[q - 1] = 4 * np.array(Qd[nm]["dz_half"]) - BD0[q - 1]; nherm += 1
        for q_ in {an, t, an - 1, an + 1}:                                            # every band-replacing arm needs its node
            if 1 <= q_ <= N and MODE == "SEP-HERM" and not hq.get(q_, False):
                MISS.append((rec["si"], k, q_))
        HN.append(nherm)
        row = lambda M, p: M[p - 1] if 1 <= p <= N else None
        dz = dict(PINm2=row(Tm, an - 2), PINp2=row(Tm, an + 2), PINm=row(Tm, an - 1), PINp=row(Tm, an + 1), PINt=row(Tm, t), PIN=row(Tm, an),
                  PINmB=(row(Dp, an - 1) + row(BD, an - 1)) if an >= 2 else None, PINpB=(row(Dp, an + 1) + row(BD, an + 1)) if an <= N - 1 else None,
                  PINtB=row(Dp, t) + row(BD, t), B=row(BD, an), PINB=row(Dp, an) + row(BD, an))
        U.append(dict(arm="E", k=k, t=t, a=FIT.anchor_of(r["qk47"][f"B{k}_E"]), x=int(r["land"][f"B{k}_E"]), xp=xn, xn=xn, si=rec["si"]))
        for arm, pos in (("PINm2", an - 2), ("PINp2", an + 2)):
            if f"B{k}_{arm}" in r["qk47"]:
                reach.append(float(FIT.anchor_of(r["qk47"][f"B{k}_{arm}"]) == pos))
        for arm, (ka, kx) in ARMS.items():
            if dz.get(arm) is None or f"B{k}_{kx}" not in r["land"]:
                continue
            U.append(dict(arm=arm, k=k, t=t, a=FIT.anchor_of(r["qk47"][f"B{k}_{ka}"]), x=int(r["land"][f"B{k}_{kx}"]), xp=int(np.argmax(z + dz[arm])) + 1,
                          xn=xn, si=rec["si"]))
nS = len(L["rows"]); allx = np.arange(nS)
hit = lambda u, r, x: float(x == (u["t"] if r == "acc" else u["a"] if r == "stay" else u["a"] - 1))
mean = lambda v: float(np.mean(v)) if len(v) else float("nan")


def sel(ix, arms):
    cnt = collections.Counter(int(i) for i in ix)
    return [u for u in U for _ in range(cnt.get(u["si"], 0)) if u["arm"] in arms]


def boot(fn):
    est = fn(allx); bs = [fn(rng.integers(0, nS, nS)) for _ in range(NB)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return dict(est=float(est), lo=float(lo), hi=float(hi))


def rate_test(arms):
    pooled = {f"{arm}_{r}": dict(pred=mean([hit(u, r, u["xp"]) for u in U if u["arm"] == arm]), obs=mean([hit(u, r, u["x"]) for u in U if u["arm"] == arm]),
                                 n=len([u for u in U if u["arm"] == arm])) for arm in arms for r in RATES}
    errs = [abs(v["pred"] - v["obs"]) for v in pooled.values()]
    cp, co = [], []
    for arm in arms:
        for k in C["ks"]:
            us = [u for u in U if u["arm"] == arm and u["k"] == k]
            if len(us) >= 10:
                for r in RATES:
                    cp.append(mean([hit(u, r, u["xp"]) for u in us])); co.append(mean([hit(u, r, u["x"]) for u in us]))
    rr = float(np.corrcoef(cp, co)[0, 1]) if len(cp) >= 3 else float("nan"); within = int(sum(e <= .10 for e in errs))
    v = "predicted" if max(errs) <= .10 and rr >= .85 else "partial" if within >= 9 and rr >= .70 else "not"
    return dict(pooled=pooled, max_err=float(max(errs)), within_10=within, cell_r=rr, n_cells=len(cp), verdict=v)


res = dict(tag=TAG, collect=L["collect"], N=N, n_scenes=nS, mode=MODE, herm_nodes_per_run=float(np.mean(HN)) if HN else 0.0)
st = float(np.mean([r["land"][f"N_S{q}"] == q for r in C["rows"] for q in range(1, N)]))
I0 = dict(reproduced=bool(mean(rep) >= .95), pin_exact=bool(max(chk or [0.0]) <= 1e-3), t1=bool((C["t1_menu"] or 0) >= .95), START=bool(st >= .95),
          reach=bool(mean(reach) >= .90), herm_nodes=bool(len(MISS) == 0))
res["I0"] = dict(**I0, all=bool(all(I0.values())), reproduced_value=mean(rep), pin_check=max(chk or [0.0]), t1_value=C["t1_menu"], START_acc=st, reach_value=mean(reach), herm_missing=len(MISS))
ok = res["I0"]["all"]
p1, p2 = rate_test(NEW), rate_test(OLD)
res["P1"] = {**p1, "verdict": p1["verdict"] if ok else "not read"}; res["P2"] = {**p2, "verdict": p2["verdict"] if ok else "not read"}
dA = boot(lambda ix: mean([float(u["xp"] == u["x"]) for u in sel(ix, tuple(ARMS))]) - mean([float(u["xn"] == u["x"]) for u in sel(ix, tuple(ARMS))]))
v3 = ("predicts which runs change and where" if dA["est"] >= .15 and dA["lo"] > 0 else "partial" if dA["est"] > 0 and dA["lo"] > 0 else "not")
res["P3"] = dict(A=mean([float(u["xp"] == u["x"]) for u in U if u["arm"] in ARMS]), A0=mean([float(u["xn"] == u["x"]) for u in U if u["arm"] in ARMS]),
                 diff=dA, verdict=(v3 if ok else "not read"))


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
res["P4"] = dict(writer_removal=cw, sharpening=cs, verdict=(("predicted" if nok == 2 else "partial" if nok == 1 else "not") if ok else "not read"))
res["descriptive"] = {arm: dict(n=len([u for u in U if u["arm"] == arm]), agree=mean([float(u["xp"] == u["x"]) for u in U if u["arm"] == arm]),
                                nochange=mean([float(u["xn"] == u["x"]) for u in U if u["arm"] == arm]),
                                rates={r: [mean([hit(u, r, u["xp"]) for u in U if u["arm"] == arm]), mean([hit(u, r, u["x"]) for u in U if u["arm"] == arm])]
                                       for r in RATES}) for arm in ARMS}
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"]["all"], P1=[res["P1"]["verdict"], p1["max_err"], p1["within_10"], p1["cell_r"]],
                      P2=[res["P2"]["verdict"], p2["max_err"], p2["within_10"], p2["cell_r"]], P3=[res["P3"]["verdict"], res["P3"]["A"], res["P3"]["A0"]],
                      P4=[res["P4"]["verdict"], cw["changes"], cs["changes"]]), indent=1, default=float))
print("->", OUT)
