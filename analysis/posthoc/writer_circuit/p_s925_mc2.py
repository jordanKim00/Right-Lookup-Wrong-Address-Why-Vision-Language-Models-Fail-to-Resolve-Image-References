"""Post hoc (after S925's registered verdicts; NOT a registered test): does a target-attraction term fix the frozen MC's acc deficit?
MC2 = MC's features + 1(b = t) (t = N + 1 - k, a function of the input), per (k group, writer flag) as before; fitted on MC's training sets and
arms only (E, E_BANDR, PIN, PIN_BANDR).  Checks: (1) leave-one-set-out NLL on the training sets (MC vs MC2) - evidence available before S924;
(2) the registered N1 / N3 statistics recomputed for MC2 on S924 and S925 (post hoc, the test arms were seen when the term was chosen)."""
import collections, importlib.util, json, sys
import numpy as np
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("s924_fit", "/vlm/exp/s924_fit.py"); FIT = importlib.util.module_from_spec(spec); spec.loader.exec_module(FIT)
MAIN = "/vlm/research/ces/outputs/writer_circuit_2026-10-20/main"
W1 = np.asarray(json.load(open(f"{MAIN}/s924_frozen_MC.json"))["MC"])
feats1 = FIT.feats


def feats2(u):
    N, a, k, e, m, t = u["N"], u["a"], u["k"], u["e"], u["m"], u["t"]; F = np.zeros((N, 11))
    for b in range(1, N + 1):
        F[b - 1] = [e[b - 1], b == a, b == a - 1, b == a - 2, b == a + 1, b == k, b == N, b == 1, m * (b == a), m * (b == a - 1), b == t]
    return F


def fit2(units):
    FIT.NF = 11; FIT.feats = feats2
    try:
        return FIT.fit(units)
    finally:
        FIT.NF = 10; FIT.feats = feats1


def pred(u, W, which):
    f = feats2 if which == 2 else feats1
    pm, _ = FIT.mixp(f(u) @ np.asarray(W)[u["g"] * 2 + u["w"]], u["N"]); return pm


ARMS_T = FIT.ARMS_TRAIN
NOVEL = dict(PINm=("B{k}_PINm", "B{k}_PINm", 1), PINp=("B{k}_PINp", "B{k}_PINp", 1), PINt=("B{k}_PINt", "B{k}_PINt", 1), PINtB=("B{k}_PINt", "B{k}_PINtB", 0))
data = {tag: FIT.load(tag, ARMS_T) for tag in FIT.TRAIN}
res = dict(note=__doc__)
# (1) leave-one-set-out on the training sets
lo = {}
for tag in FIT.TRAIN:
    tr = [u for t_, us in data.items() if t_ != tag for u in us]; te = data[tag]
    Wa, _ = FIT.fit(tr); Wb, _ = fit2(tr)
    n1 = float(np.mean([-np.log(pred(u, Wa, 1)[FIT.xi(u)]) for u in te])); n2 = float(np.mean([-np.log(pred(u, Wb, 2)[FIT.xi(u)]) for u in te]))
    accE = [(float(u["x"] == u["t"]), float(pred(u, Wa, 1)[u["t"]]), float(pred(u, Wb, 2)[u["t"]])) for u in te if u["arm"] == "E"]
    lo[tag] = dict(NLL_MC=n1, NLL_MC2=n2, E_acc_obs_MC_MC2=np.mean(accE, 0).round(3).tolist())
    print("LOSO", tag, round(n1, 4), round(n2, 4), lo[tag]["E_acc_obs_MC_MC2"])
res["loso"] = lo
# (2) MC2 on all training sets, then the registered statistics on S924 / S925
allu = [u for us in data.values() for u in us]
W2, f2 = fit2(allu); res["train_nll_MC2"] = f2; res["W_t"] = {f"g{g}_w{w}": float(W2[g * 2 + w][10]) for g in range(5) for w in (0, 1)}
print("train nll MC2", round(f2, 4), "theta_t", {k: round(v, 2) for k, v in res["W_t"].items()})
for tag in ("s924_27b", "s925_27b"):
    U = FIT.load(tag, {**ARMS_T, **NOVEL}); J = json.load(open(f"/vlm/results/r3/s924_{tag}_collect.json")); KS = J["ks"]
    for u in U:
        u["p1"] = pred(u, W1, 1); u["p2"] = pred(u, W2, 2)
    def rp(u, r, m):
        b = u["t"] if r == "acc" else u["a"] if r == "stay" else u["a"] - 1
        return float(u[m][b]) if 1 <= b <= u["N"] else 0.0
    def ro(u, r):
        return float(u["x"] == (u["t"] if r == "acc" else u["a"] if r == "stay" else u["a"] - 1))
    out = {}
    for m in ("p1", "p2"):
        pooled = {f"{arm}_{r}": (np.mean([rp(u, r, m) for u in U if u["arm"] == arm]), np.mean([ro(u, r) for u in U if u["arm"] == arm]))
                  for arm in NOVEL for r in ("acc", "stay", "step")}
        errs = [abs(p - o) for p, o in pooled.values()]
        cp, co = [], []
        for arm in NOVEL:
            for k in KS:
                us = [u for u in U if u["arm"] == arm and u["k"] == k]
                if len(us) >= 10:
                    for r in ("acc", "stay", "step"):
                        cp.append(np.mean([rp(u, r, m) for u in us])); co.append(np.mean([ro(u, r) for u in us]))
        r1 = float(np.corrcoef(cp, co)[0, 1])
        n1v = "predicted" if max(errs) <= .10 and r1 >= .85 else "partial" if sum(e <= .10 for e in errs) >= 9 and r1 >= .70 else "not"
        ch = {}
        for arm in ("B", "PIN"):
            for r in ("acc", "stay"):
                e_ = [u for u in U if u["arm"] == "E"]; x_ = [u for u in U if u["arm"] == arm]
                ch[f"{arm}_{r}"] = (np.mean([rp(u, r, m) for u in x_]) - np.mean([rp(u, r, m) for u in e_]), np.mean([ro(u, r) for u in x_]) - np.mean([ro(u, r) for u in e_]))
        nll_nov = float(np.mean([-np.log(u[m][FIT.xi(u)]) for u in U if u["arm"] in NOVEL]))
        nll_E = float(np.mean([-np.log(u[m][FIT.xi(u)]) for u in U if u["arm"] == "E"]))
        out[m] = dict(N1=n1v, max_err=float(max(errs)), within=int(sum(e <= .10 for e in errs)), cell_r=r1,
                      pooled={k: [round(float(p), 3), round(float(o), 3)] for k, (p, o) in pooled.items()},
                      N3_changes={k: [round(float(p), 3), round(float(o), 3), round(abs(float(p - o)), 3)] for k, (p, o) in ch.items()},
                      NLL_novel=nll_nov, NLL_E=nll_E)
        print(tag, m, n1v, "max", round(max(errs), 3), "within", out[m]["within"], "r", round(r1, 3), "NLL novel", round(nll_nov, 4), "E", round(nll_E, 4))
        print("    pooled [pred, obs]", out[m]["pooled"])
        print("    N3 [pred, obs, err]", out[m]["N3_changes"])
    res[tag] = out
json.dump(res, open(f"{MAIN}/p_s925_mc2.json", "w"), indent=1, default=float)
print("->", f"{MAIN}/p_s925_mc2.json")
