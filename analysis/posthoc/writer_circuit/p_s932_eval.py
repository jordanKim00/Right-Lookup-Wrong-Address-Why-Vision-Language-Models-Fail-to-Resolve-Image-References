"""Post hoc development evaluation of S932dev: pins by LR (first order, exact local change), RP-HERM (quadratic along the real pin path: value at
e = 1/2, slope by the finite difference at e = .1) and RP-SEC (straight line through the half point: z + 2 (z(1/2) - z)); band-replacing arms by
JNT-HERM (joint path) - against the observed answers.  Usage: python3 p_s932_eval.py TAG"""
import importlib.util, json, sys
import numpy as np
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("s924_fit", "/vlm/exp/s924_fit.py"); FIT = importlib.util.module_from_spec(spec); spec.loader.exec_module(FIT)
TAG = sys.argv[1]
L = json.load(open(f"/vlm/results/r3/s932_{TAG}_pathq.json")); C = json.load(open(L["collect"])); N = C["N"]
PINS = dict(PIN=("PIN", "PIN", 0), PINm=("PINm", "PINm", -1), PINp=("PINp", "PINp", 1), PINm2=("PINm2", "PINm2", -2), PINp2=("PINp2", "PINp2", 2), PINt=("PINt", "PINt", None))
BANDS = dict(B=("E", "E_BANDR"), PINB=("PIN", "PIN_BANDR"), PINtB=("PINt", "PINtB"), PINmB=("PINm", "PINmB"), PINpB=("PINp", "PINpB"))
U = []
for rec in L["rows"]:
    r = C["rows"][rec["si"]]
    for k_, o in rec["k"].items():
        k = int(k_); t = N + 1 - k; an = o["anchor"]; z = np.array(o["z"]); Tm, Dp, BD = np.array(o["T"]), np.array(o["Dp"]), np.array(o["BAND"])
        PP = o.get("pinpath", {}); Q = o.get("quad", {})
        for arm, (ka, kx, d) in PINS.items():
            q = t if d is None else an + d
            if not 1 <= q <= N or f"B{k}_{kx}" not in r["land"] or str(q) not in PP:
                continue
            d10, d50 = np.array(PP[str(q)]["dz10"]), np.array(PP[str(q)]["dz50"])
            pred = dict(LR=z + Tm[q - 1], RP_HERM=z + 4 * d50 - d10 / .1, RP_SEC=z + 2 * d50)
            U.append(dict(arm=arm, k=k, t=t, a=FIT.anchor_of(r["qk47"][f"B{k}_{ka}"]), x=int(r["land"][f"B{k}_{kx}"]), xp={v: int(np.argmax(p)) + 1 for v, p in pred.items()}))
        pq = dict(B=(None, an), PINB=(an, an), PINtB=(t, t), PINmB=(an - 1, an - 1), PINpB=(an + 1, an + 1))
        for arm, (ka, kx) in BANDS.items():
            p, q = pq[arm]
            if not 1 <= q <= N or f"B{k}_{kx}" not in r["land"] or arm not in Q:
                continue
            g0 = (Dp[p - 1] if p is not None else 0.0) + BD[q - 1]
            pred = dict(LR=z + g0, JNT=z + 4 * np.array(Q[arm]["dz_half"]) - g0)
            U.append(dict(arm=arm, k=k, t=t, a=FIT.anchor_of(r["qk47"][f"B{k}_{ka}"]), x=int(r["land"][f"B{k}_{kx}"]), xp={v: int(np.argmax(pv)) + 1 for v, pv in pred.items()}))
hit = lambda u, r, x: float(x == (u["t"] if r == "acc" else u["a"] if r == "stay" else u["a"] - 1))
for arm in list(PINS) + list(BANDS):
    us = [u for u in U if u["arm"] == arm]
    if not us:
        continue
    obs = {r: np.mean([hit(u, r, u["x"]) for u in us]) for r in ("acc", "stay", "step")}
    line = [f"{arm:6s} n {len(us):3d} obs " + " ".join(f"{r}:{v:.2f}" for r, v in obs.items())]
    for v in us[0]["xp"]:
        pr = {r: np.mean([hit(u, r, u["xp"][v]) for u in us]) for r in obs}
        line.append(f"| {v} agree {np.mean([u['xp'][v] == u['x'] for u in us]):.2f} " + " ".join(f"{r}:{pr[r]:.2f}" for r in pr) + f" max {max(abs(pr[r] - obs[r]) for r in pr):.2f}")
    print(" ".join(line))
