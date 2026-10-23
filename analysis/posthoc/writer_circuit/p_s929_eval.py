"""Post hoc development check of the SEPARABLE path-integral predictor: the pin part to first order with the exact local change (S927 LR: T for
pin-only arms, Dp for the pin part of band-replacing arms) and the band part by quadrature along the band's own path (band read-row outputs set
to natural + e (START(q) - END), held - the intervention's exact path): B = BANDx[a], PINB = Dp[a] + BANDx[a], PINtB = Dp[t] + BANDx[t],
PINmB = Dp[a - 1] + BANDx[a - 1], PINpB = Dp[a + 1] + BANDx[a + 1], with BANDx = LR | MID | HERM | CUB of the band-only arms B (q = a), Bt (q = t),
Bm (q = a - 1), Bp (q = a + 1) of exp/s928_quad.py (or s928h for HERM).
Usage: python3 p_s929_eval.py QUADFILE"""
import importlib.util, json, sys
import numpy as np
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("s924_fit", "/vlm/exp/s924_fit.py"); FIT = importlib.util.module_from_spec(spec); spec.loader.exec_module(FIT)
L = json.load(open(sys.argv[1])); C = json.load(open(L["collect"])); N = C["N"]
ARMS = dict(PIN=("PIN", "PIN"), PINm=("PINm", "PINm"), PINp=("PINp", "PINp"), PINt=("PINt", "PINt"), PINm2=("PINm2", "PINm2"), PINp2=("PINp2", "PINp2"),
            B=("E", "E_BANDR"), PINB=("PIN", "PIN_BANDR"), PINtB=("PINt", "PINtB"), PINmB=("PINm", "PINmB"), PINpB=("PINp", "PINpB"))
VARS = ("LR", "MID", "HERM", "CUB")
U = []
for rec in L["rows"]:
    r = C["rows"][rec["si"]]
    for k_, o in rec["k"].items():
        k = int(k_); t = N + 1 - k; an = o["anchor"]; z = np.array(o["z"]); xn = int(o["x_nat"])
        Tm, Dp, BD = np.array(o["T"]), np.array(o["Dp"]), np.array(o["BAND"]); Q = o.get("quad", {})

        def band(q, nm=None):
            g = BD[q - 1]
            nm = next((n_ for n_, qq in (("B", an), ("Bt", t), ("Bm", an - 1), ("Bp", an + 1)) if qq == q and n_ in Q), None)   # the band-only node depends on q only
            if nm is None:
                return {"LR": g}
            dh = np.array(Q[nm]["dz_half"]); out = {"LR": g, "HERM": 4 * dh - g}
            if "g_half" in Q[nm]:
                gh = np.array(Q[nm]["g_half"]); out.update(MID=gh, CUB=g + 2 * gh - 4 * dh)
            return out
        row = lambda M, p: M[p - 1] if 1 <= p <= N else None
        bq = {"B": (an, None), "PINB": (an, an), "PINtB": (t, t), "PINmB": (an - 1, an - 1), "PINpB": (an + 1, an + 1)}
        qn = {an: "B", t: "Bt", an - 1: "Bm", an + 1: "Bp"}
        for arm, (ka, kx) in ARMS.items():
            if f"B{k}_{kx}" not in r["land"]:
                continue
            if arm in bq:
                q, p = bq[arm]
                if not 1 <= q <= N:
                    continue
                bv = band(q); pin = Dp[p - 1] if p is not None else 0.0
                pred = {v: z + pin + d for v, d in bv.items()}
            else:
                p = dict(PIN=an, PINm=an - 1, PINp=an + 1, PINt=t, PINm2=an - 2, PINp2=an + 2)[arm]
                if not 1 <= p <= N:
                    continue
                pred = {v: z + Tm[p - 1] for v in VARS}
            a_u = FIT.anchor_of(r["qk47"][f"B{k}_{ka}"]); x = int(r["land"][f"B{k}_{kx}"])
            U.append(dict(arm=arm, k=k, t=t, a=a_u, x=x, xn=xn, xp={v: int(np.argmax(pv)) + 1 for v, pv in pred.items()}))
hit = lambda u, r, x: float(x == (u["t"] if r == "acc" else u["a"] if r == "stay" else u["a"] - 1))
for arm in ARMS:
    us = [u for u in U if u["arm"] == arm]
    if not us:
        continue
    obs_r = {r: np.mean([hit(u, r, u["x"]) for u in us]) for r in ("acc", "stay", "step")}
    line = [f"{arm:6s} n {len(us):3d} obs " + " ".join(f"{r}:{v:.2f}" for r, v in obs_r.items())]
    for v in VARS:
        uv = [u for u in us if v in u["xp"]]
        if len(uv) < len(us):                                                     # the band-only node is missing for some units
            line.append(f"| {v} n/a ({len(uv)}/{len(us)})"); continue
        pr = {r: np.mean([hit(u, r, u["xp"][v]) for u in uv]) for r in obs_r}
        line.append(f"| {v} agree {np.mean([u['xp'][v] == u['x'] for u in uv]):.2f} maxerr {max(abs(pr[r] - obs_r[r]) for r in pr):.2f}")
    print(" ".join(line))
