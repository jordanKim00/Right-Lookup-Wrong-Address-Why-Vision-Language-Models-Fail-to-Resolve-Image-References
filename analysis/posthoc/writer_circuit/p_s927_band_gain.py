"""Post hoc (after S927a's registered verdicts): is LR's band-arm miss a gain (magnitude) problem?  For each unit: the margin between the
unit's anchor a and the natural answer x_nat, predicted change (linear) vs observed change (menu logits of the arm minus the natural run).
Also: where the predicted answer goes when the observed answer stays at a."""
import collections, importlib.util, json, sys
import numpy as np
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("s924_fit", "/vlm/exp/s924_fit.py"); FIT = importlib.util.module_from_spec(spec); spec.loader.exec_module(FIT)
TAG = sys.argv[1]
L = json.load(open(f"/vlm/results/r3/s927_{TAG}_lr.json")); C = json.load(open(L["collect"])); N = C["N"]
ARMS = dict(PIN=("PIN", "PIN"), PINm=("PINm", "PINm"), PINp=("PINp", "PINp"), PINt=("PINt", "PINt"), B=("E", "E_BANDR"), PINB=("PIN", "PIN_BANDR"), PINtB=("PINt", "PINtB"))
st = collections.defaultdict(list); conf = collections.defaultdict(collections.Counter)
for rec in L["rows"]:
    r = C["rows"][rec["si"]]
    for k_, o in rec["k"].items():
        k = int(k_); t = N + 1 - k; an = o["anchor"]; z = np.array(o["z"]); xn = int(o["x_nat"])
        Tm, Dp, BD = np.array(o["T"]), np.array(o["Dp"]), np.array(o["BAND"])
        dz = dict(PIN=Tm[an - 1], PINt=Tm[t - 1], B=BD[an - 1], PINB=Dp[an - 1] + BD[an - 1], PINtB=Dp[t - 1] + BD[t - 1])
        if an >= 2: dz["PINm"] = Tm[an - 2]
        if an <= N - 1: dz["PINp"] = Tm[an]
        for arm, (ka, kx) in ARMS.items():
            if arm not in dz or f"B{k}_{kx}" not in r["land"]:
                continue
            a_u = FIT.anchor_of(r["qk47"][f"B{k}_{ka}"]); x = int(r["land"][f"B{k}_{kx}"])
            if xn > N or a_u == xn:
                continue
            obs = np.array(r["mlog"][f"B{k}_{kx}"]) - np.array(r["mlog"][f"B{k}_E"])
            st[arm].append((dz[arm][a_u - 1] - dz[arm][xn - 1], obs[a_u - 1] - obs[xn - 1]))
            xp = int(np.argmax(z + dz[arm])) + 1
            if x == a_u and xp != a_u:
                conf[arm]["pred=x_nat" if xp == xn else "pred=a-1" if xp == a_u - 1 else "pred=t" if xp == t else "pred other"] += 1
for arm, v in st.items():
    v = np.array(v); p, o_ = v[:, 0], v[:, 1]
    print(f"{arm:6s} n {len(v):4d}  margin(a vs x_nat) change: pred mean {p.mean():.2f}  obs mean {o_.mean():.2f}  ratio of means {o_.mean() / p.mean():.2f}"
          f"  corr {np.corrcoef(p, o_)[0, 1]:.3f}  slope(obs~pred) {(p * o_).sum() / (p * p).sum():.2f}   misses where obs stays: {dict(conf[arm])}")
