"""Post hoc development check of the S928 path-integral variants against observed outcomes (S924 scenes).  Per arm and variant
(LR = S927 one-node rule at e = 0, MID = midpoint, HERM = quadratic Hermite, CUB = cubic Hermite): predicted answer = argmax of the predicted
menu logits; rates acc / stay / step (unit anchor as exp/s924_analyze.py) predicted vs observed, run-level agreement, and the change of the
anchor-vs-natural-answer margin (predicted vs observed).
Usage: python3 p_s928_eval.py QUADTAG"""
import collections, importlib.util, json, sys
import numpy as np
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("s924_fit", "/vlm/exp/s924_fit.py"); FIT = importlib.util.module_from_spec(spec); spec.loader.exec_module(FIT)
TAG = sys.argv[1]
L = json.load(open(f"/vlm/results/r3/s928_{TAG}_quad.json")); C = json.load(open(L["collect"])); N = C["N"]
ARMS = dict(PIN=("PIN", "PIN"), PINm=("PINm", "PINm"), PINp=("PINp", "PINp"), PINt=("PINt", "PINt"), PINm2=("PINm2", "PINm2"), PINp2=("PINp2", "PINp2"),
            B=("E", "E_BANDR"), PINB=("PIN", "PIN_BANDR"), PINtB=("PINt", "PINtB"), PINmB=("PINm", "PINmB"), PINpB=("PINp", "PINpB"))
VARS = ("LR", "MID", "HERM", "CUB")
U = []
for rec in L["rows"]:
    r = C["rows"][rec["si"]]
    for k_, o in rec["k"].items():
        k = int(k_); t = N + 1 - k; an = o["anchor"]; z = np.array(o["z"]); xn = int(o["x_nat"])
        Tm, Dp, BD = np.array(o["T"]), np.array(o["Dp"]), np.array(o["BAND"])
        row = lambda M, p: M[p - 1] if 1 <= p <= N else None
        g0 = dict(PIN=row(Tm, an), PINm=row(Tm, an - 1), PINp=row(Tm, an + 1), PINt=row(Tm, t), PINm2=row(Tm, an - 2), PINp2=row(Tm, an + 2),
                  B=row(BD, an), PINB=row(Dp, an) + row(BD, an), PINtB=row(Dp, t) + row(BD, t),
                  PINmB=(row(Dp, an - 1) + row(BD, an - 1)) if an >= 2 else None, PINpB=(row(Dp, an + 1) + row(BD, an + 1)) if an <= N - 1 else None)
        for arm, qd in o.get("quad", {}).items():
            ka, kx = ARMS[arm]
            if g0.get(arm) is None or f"B{k}_{kx}" not in r["land"]:
                continue
            gh = np.array(qd["g_half"]); dh = np.array(qd["dz_half"]); g = g0[arm]
            pred = dict(LR=z + g, MID=z + gh, HERM=z + 4 * dh - g, CUB=z + g + 2 * gh - 4 * dh)
            a_u = FIT.anchor_of(r["qk47"][f"B{k}_{ka}"]); x = int(r["land"][f"B{k}_{kx}"])
            obs = np.array(r["mlog"][f"B{k}_{kx}"]) - np.array(r["mlog"][f"B{k}_E"])
            u = dict(arm=arm, k=k, t=t, a=a_u, x=x, xn=xn, xp={v: int(np.argmax(p)) + 1 for v, p in pred.items()})
            if xn <= N and a_u != xn:
                u["marg_obs"] = obs[a_u - 1] - obs[xn - 1]; u["marg"] = {v: (p - z)[a_u - 1] - (p - z)[xn - 1] for v, p in pred.items()}
            U.append(u)
hit = lambda u, r, x: float(x == (u["t"] if r == "acc" else u["a"] if r == "stay" else u["a"] - 1))
res = {}
for arm in ARMS:
    us = [u for u in U if u["arm"] == arm]
    if not us:
        continue
    res[arm] = dict(n=len(us))
    line = [f"{arm:6s} n {len(us):3d}"]
    obs_r = {r: np.mean([hit(u, r, u["x"]) for u in us]) for r in ("acc", "stay", "step")}
    line.append("obs " + " ".join(f"{r}:{v:.2f}" for r, v in obs_r.items()))
    for v in VARS:
        pr = {r: np.mean([hit(u, r, u["xp"][v]) for u in us]) for r in ("acc", "stay", "step")}
        ag = np.mean([u["xp"][v] == u["x"] for u in us]); err = max(abs(pr[r] - obs_r[r]) for r in pr)
        mo = [u for u in us if "marg_obs" in u]
        ratio = np.mean([u["marg_obs"] for u in mo]) / np.mean([u["marg"][v] for u in mo]) if mo else float("nan")
        res[arm][v] = dict(rates={r: float(x) for r, x in pr.items()}, agree=float(ag), max_err=float(err), margin_obs_over_pred=float(ratio))
        line.append(f"| {v} agree {ag:.2f} maxerr {err:.2f} m-ratio {ratio:.2f}")
    res[arm]["obs"] = {r: float(x) for r, x in obs_r.items()}
    res[arm]["nochange"] = float(np.mean([u["xn"] == u["x"] for u in us]))
    print(" ".join(line))
json.dump(res, open(f"/vlm/research/ces/outputs/writer_circuit_2026-10-20/main/p_s928_eval_{TAG}.json", "w"), indent=1)
