"""Development evaluation of the S927 linear-response (LR) predictor (no fitted parameter) against observed intervention outcomes.
Inputs: results/r3/s927_<lrtag>_lr.json and the outcome file it was built from (s924_* / s926_* collect).  Per arm (unit anchor as in
exp/s924_analyze.py: the L47 anchor of the arm's address run): predicted answer = argmax(z + dz) over the menu words (positions; > N =
distractor); observed answer = the arm's answer.  Reports: natural reproduction, finite-difference checks, per-arm agreement P(pred = obs) and
the no-change baseline P(natural answer = obs), pooled rates (acc / stay / step) predicted vs observed, and the predicted change of the answer.
Usage: python3 p_s927_eval.py LRTAG [OUT]"""
import collections, importlib.util, json, sys
import numpy as np
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("s924_fit", "/vlm/exp/s924_fit.py"); FIT = importlib.util.module_from_spec(spec); spec.loader.exec_module(FIT)
TAG = sys.argv[1]
L = json.load(open(f"/vlm/results/r3/s927_{TAG}_lr.json")); C = json.load(open(L["collect"].replace("/vlm/", "/vlm/"))); N = C["N"]
ARMS = dict(PIN=("PIN", "PIN"), PINm=("PINm", "PINm"), PINp=("PINp", "PINp"), PINt=("PINt", "PINt"), PINm2=("PINm2", "PINm2"), PINp2=("PINp2", "PINp2"),
            B=("E", "E_BANDR"), PINB=("PIN", "PIN_BANDR"), PINtB=("PINt", "PINtB"), PINmB=("PINm", "PINmB"), PINpB=("PINp", "PINpB"))
res = dict(tag=TAG, collect=L["collect"]); U = []
rep = []; fd = collections.defaultdict(list)
for rec in L["rows"]:
    r = C["rows"][rec["si"]]
    for k_, o in rec["k"].items():
        k = int(k_); t = N + 1 - k; z = np.array(o["z"]); xn = int(o["x_nat"])
        rep.append(xn == r["land"][f"B{k}_E"])
        for key, d in o.get("fd", {}).items():
            fd[key].append(dict(fd=d["fd"], lin=d["lin"]))
        an = o["anchor"]; Tm, Dp, BD = np.array(o["T"]), np.array(o["Dp"]), np.array(o["BAND"])
        comp = dict(PIN=lambda: Tm[an - 1], PINm=lambda: Tm[an - 2], PINp=lambda: Tm[an], PINt=lambda: Tm[t - 1], PINm2=lambda: Tm[an - 3],
                    PINp2=lambda: Tm[an + 1], B=lambda: BD[an - 1], PINB=lambda: Dp[an - 1] + BD[an - 1], PINtB=lambda: Dp[t - 1] + BD[t - 1],
                    PINmB=lambda: Dp[an - 2] + BD[an - 2], PINpB=lambda: Dp[an] + BD[an])
        for arm, (ka, kx) in ARMS.items():
            if f"B{k}_{kx}" not in r["land"]:
                continue
            a_u = FIT.anchor_of(r["qk47"][f"B{k}_{ka}"]); x = int(r["land"][f"B{k}_{kx}"])
            o[arm] = comp[arm]().tolist()
            zp = z + np.array(o[arm]); xp = int(np.argmax(zp)) + 1
            U.append(dict(arm=arm, k=k, t=t, a=a_u, x=x, xp=xp, xn=xn, si=rec["si"], dz_obs=(np.array(r["mlog"][f"B{k}_{kx}"]) - np.array(r["mlog"][f"B{k}_E"])).tolist(),
                          dz_lin=np.array(o[arm])[:N].tolist()))
res["natural_reproduced"] = float(np.mean(rep)); res["n_runs"] = len(rep)
res["pin_check_maxabs"] = max([o.get("pin_check_maxabs", 0.0) for rec in L["rows"] for o in rec["k"].values()] or [0.0])
res["fd"] = {}
for key, lst in fd.items():
    f = np.array([x["fd"] for x in lst]); l = np.array([x["lin"] for x in lst])
    res["fd"][key] = dict(corr=float(np.corrcoef(f.ravel(), l.ravel())[0, 1]), slope=float((f * l).sum() / (l * l).sum()), fd=f.round(3).tolist(), lin=l.round(3).tolist())
rate = lambda u, r, x: float(x == (u["t"] if r == "acc" else u["a"] if r == "stay" else u["a"] - 1))
res["arms"] = {}
for arm in ARMS:
    us = [u for u in U if u["arm"] == arm]
    if not us:
        continue
    dzo = np.array([u["dz_obs"] for u in us]); dzl = np.array([u["dz_lin"] for u in us])
    res["arms"][arm] = dict(n=len(us), agree=float(np.mean([u["xp"] == u["x"] for u in us])), nochange=float(np.mean([u["xn"] == u["x"] for u in us])),
                            changed_obs=float(np.mean([u["x"] != u["xn"] for u in us])), changed_pred=float(np.mean([u["xp"] != u["xn"] for u in us])),
                            dz_corr=float(np.corrcoef(dzo.ravel(), dzl.ravel())[0, 1]), dz_slope=float((dzo * dzl).sum() / (dzl * dzl).sum()),
                            rates={r: [round(float(np.mean([rate(u, r, u["xp"]) for u in us])), 3), round(float(np.mean([rate(u, r, u["x"]) for u in us])), 3)]
                                   for r in ("acc", "stay", "step")})
print(json.dumps({k: v for k, v in res.items() if k != "fd"}, indent=1))
for key, v in res["fd"].items():
    print("FD", key, "corr %.3f slope %.3f" % (v["corr"], v["slope"]))
out = sys.argv[2] if len(sys.argv) > 2 else f"/vlm/research/ces/outputs/writer_circuit_2026-10-20/main/p_s927_eval_{TAG}.json"
json.dump(res, open(out, "w"), indent=1)
print("->", out)
