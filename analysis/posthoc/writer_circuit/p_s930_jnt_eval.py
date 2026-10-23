"""Post hoc (S926 N8 is development data for this): band-replacing arms predicted three ways - LR (first order), SEP-HERM (registered S926:
pin part first order + quadratic along the band-only path), JNT-HERM (quadratic along the joint path: pin and band moved together,
HERM = 4 (z(1/2) - z) - (Dp[p] + BAND[q])) - against the observed answers; rates and run agreement per arm.
Usage: python3 p_s930_jnt_eval.py SEPFILE JNTFILE"""
import importlib.util, json, sys
import numpy as np
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("s924_fit", "/vlm/exp/s924_fit.py"); FIT = importlib.util.module_from_spec(spec); spec.loader.exec_module(FIT)
S = json.load(open(sys.argv[1])); J = json.load(open(sys.argv[2])); C = json.load(open(S["collect"])); N = C["N"]
ARMS = dict(B=("E", "E_BANDR"), PINB=("PIN", "PIN_BANDR"), PINtB=("PINt", "PINtB"), PINmB=("PINm", "PINmB"), PINpB=("PINp", "PINpB"))
jrow = {rec["si"]: rec for rec in J["rows"]}; U = []
for rec in S["rows"]:
    r = C["rows"][rec["si"]]; rj = jrow.get(rec["si"])
    if rj is None:
        continue
    for k_, o in rec["k"].items():
        oj = rj["k"].get(k_)
        if oj is None:
            continue
        k = int(k_); t = N + 1 - k; an = o["anchor"]; z = np.array(o["z"])
        Dp, BD = np.array(o["Dp"]), np.array(o["BAND"]); Qs = o.get("quad", {}); Qj = oj.get("quad", {})
        pq = dict(B=(None, an), PINB=(an, an), PINtB=(t, t), PINmB=(an - 1, an - 1), PINpB=(an + 1, an + 1))
        for arm, (ka, kx) in ARMS.items():
            p, q = pq[arm]
            if not 1 <= q <= N or f"B{k}_{kx}" not in r["land"] or arm not in Qj:
                continue
            g0 = (Dp[p - 1] if p is not None else 0.0) + BD[q - 1]
            nm = next((n_ for n_, qq in (("B", an), ("Bt", t), ("Bm", an - 1), ("Bp", an + 1)) if qq == q and n_ in Qs), None)
            sep = (Dp[p - 1] if p is not None else 0.0) + (4 * np.array(Qs[nm]["dz_half"]) - BD[q - 1]) if nm else None
            jnt = 4 * np.array(Qj[arm]["dz_half"]) - g0
            pred = dict(LR=z + g0, SEP=z + sep if sep is not None else None, JNT=z + jnt)
            a_u = FIT.anchor_of(r["qk47"][f"B{k}_{ka}"]); x = int(r["land"][f"B{k}_{kx}"])
            U.append(dict(arm=arm, k=k, t=t, a=a_u, x=x, xp={v: int(np.argmax(pv)) + 1 for v, pv in pred.items() if pv is not None}))
hit = lambda u, r, x: float(x == (u["t"] if r == "acc" else u["a"] if r == "stay" else u["a"] - 1))
for arm in ARMS:
    us = [u for u in U if u["arm"] == arm]
    if not us:
        continue
    obs = {r: np.mean([hit(u, r, u["x"]) for u in us]) for r in ("acc", "stay", "step")}
    line = [f"{arm:6s} n {len(us):3d} obs " + " ".join(f"{r}:{v:.2f}" for r, v in obs.items())]
    for v in ("LR", "SEP", "JNT"):
        uv = [u for u in us if v in u["xp"]]
        pr = {r: np.mean([hit(u, r, u["xp"][v]) for u in uv]) for r in obs}
        line.append(f"| {v} agree {np.mean([u['xp'][v] == u['x'] for u in uv]):.2f} " + " ".join(f"{r}:{pr[r]:.2f}" for r in pr)
                    + f" maxerr {max(abs(pr[r] - obs[r]) for r in pr):.2f}")
    print(" ".join(line))
