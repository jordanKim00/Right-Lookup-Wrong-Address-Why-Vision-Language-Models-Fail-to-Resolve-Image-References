"""Development evaluation of S929 (circuit-restricted linear response).  For the address pins (PIN, PINm, PINp, PINt) and each restriction
R0 (full), R1 (read rows only), R2 (read rows, and after the band only the late FA reads): predicted answer = argmax(z + T_R[q]); rates
acc / stay / step predicted vs observed, run-level agreement, and the share of the full first-order response carried by the restricted paths,
<T_R, T_R0> / <T_R0, T_R0> over the centred picture logits (per run, median), also per k.
Usage: python3 p_s929_circ_eval.py TAG"""
import collections, importlib.util, json, sys
import numpy as np
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("s924_fit", "/vlm/exp/s924_fit.py"); FIT = importlib.util.module_from_spec(spec); spec.loader.exec_module(FIT)
TAG = sys.argv[1]
L = json.load(open(f"/vlm/results/r3/s929_{TAG}_circ.json")); C = json.load(open(L["collect"])); N = C["N"]
ARMS = dict(PIN=("PIN", "PIN"), PINm=("PINm", "PINm"), PINp=("PINp", "PINp"), PINt=("PINt", "PINt"))
RV = ("R0", "R1", "R2"); U = []; share = collections.defaultdict(list); share_k = collections.defaultdict(lambda: collections.defaultdict(list))
for rec in L["rows"]:
    r = C["rows"][rec["si"]]
    for k_, o in rec["k"].items():
        k = int(k_); t = N + 1 - k; an = o["anchor"]; z = np.array(o["z"]); xn = int(o["x_nat"])
        Tm = {"R0": np.array(o["T"]), "R1": np.array(o["T_R1"]), "R2": np.array(o["T_R2"])}
        c = lambda A: (A[:, :N] - A[:, :N].mean(1, keepdims=True))
        for rv in ("R1", "R2"):
            den = (c(Tm["R0"]) ** 2).sum()
            if den > 0:
                sh = float((c(Tm[rv]) * c(Tm["R0"])).sum() / den); share[rv].append(sh); share_k[rv][k].append(sh)
        for arm, (ka, kx) in ARMS.items():
            p = dict(PIN=an, PINm=an - 1, PINp=an + 1, PINt=t)[arm]
            if not 1 <= p <= N or f"B{k}_{kx}" not in r["land"]:
                continue
            a_u = FIT.anchor_of(r["qk47"][f"B{k}_{ka}"]); x = int(r["land"][f"B{k}_{kx}"])
            U.append(dict(arm=arm, k=k, t=t, a=a_u, x=x, xn=xn, xp={rv: int(np.argmax(z + Tm[rv][p - 1])) + 1 for rv in RV}))
hit = lambda u, r, x: float(x == (u["t"] if r == "acc" else u["a"] if r == "stay" else u["a"] - 1))
print("share of the full first-order response carried (median over runs):", {rv: round(float(np.median(v)), 3) for rv, v in share.items()})
for rv in ("R1", "R2"):
    print("  by k", rv, {k: round(float(np.median(v)), 2) for k, v in sorted(share_k[rv].items())})
for arm in ARMS:
    us = [u for u in U if u["arm"] == arm]
    obs = {r: np.mean([hit(u, r, u["x"]) for u in us]) for r in ("acc", "stay", "step")}
    line = [f"{arm:5s} n {len(us):3d} obs " + " ".join(f"{r}:{v:.2f}" for r, v in obs.items()) + f" nochange {np.mean([u['xn'] == u['x'] for u in us]):.2f}"]
    for rv in RV:
        pr = {r: np.mean([hit(u, r, u["xp"][rv]) for u in us]) for r in obs}
        line.append(f"| {rv} agree {np.mean([u['xp'][rv] == u['x'] for u in us]):.2f} maxerr {max(abs(pr[r] - obs[r]) for r in pr):.2f}")
    print(" ".join(line))
