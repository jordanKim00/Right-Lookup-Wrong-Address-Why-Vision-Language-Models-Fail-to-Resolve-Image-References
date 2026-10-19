"""S751 analysis, frozen with the pre-registration (knowledge/DECISIONS.md, 2026-10-13 00:3x) before any S751 data.
F = answers on the donor's target, base = the recipient's natural answers on that target, S(arm) = (F(arm) - base) / (F(RES26) - base).
I0             NOOP (BOTH from the recipient's own run, k = 5) equals the natural answer on >= .95 of scenes, and F(5<-2, RES26) >= .5 * acc(2)
OP             SINGLE if some sublayer has S >= .5 in both (5<-2) and (3<-2); else DISTRIBUTED if ATT, MLP or BOTH has S >= .5 in both pairs;
               else NOT_IN_23_26
ERROR_CARRIER  (2<-5), scenes where the donor END(5) answered wrong: the arms whose reproduction of the donor's actual choice C(arm) is
               >= .5 * C(RES26) and >= the baseline (recipient's natural answer = donor's choice) + .10
Descriptive: S for every arm and pair, C for every arm, scene-bootstrap 95% intervals (4,000 draws) for F of the rule arms.
Usage: python s751_analyze.py TAG
"""
import json, sys
import numpy as np

N = 7
SINGLE = ["A23", "A24", "A25", "A26", "M23", "M24", "M25", "M26"]
GROUPS = ["ATT", "MLP", "BOTH"]


def boot(x, B=4000, seed=0):
    x = np.asarray(x, float); r = np.random.default_rng(seed)
    m = [x[r.integers(0, len(x), len(x))].mean() for _ in range(B)]
    return [round(float(x.mean()), 3), round(float(np.percentile(m, 2.5)), 3), round(float(np.percentile(m, 97.5)), 3)]


def analyze(tag):
    D = f"/vlm/results/r3/s751_{tag}_ops"
    meta = json.load(open(D + ".json")); M = meta["meta"]; n = len(M)
    tgt = lambda k: N + 1 - k
    L = lambda key: np.array([m["land"][key] for m in M])
    acc = {k: float(np.mean(L(f"E{k}") == tgt(k))) for k in meta["KS"]}
    arms = SINGLE + GROUPS + ["RES22", "RES26"]
    out = dict(n=n, acc={str(k): round(v, 3) for k, v in acc.items()}, checks={}, S={}, F={}, C={})
    noop = float(np.mean(L("NOOP|5") == L("E5")))
    out["checks"]["I0"] = bool(noop >= .95 - 1e-9 and np.mean(L("RES26|5<2") == tgt(2)) >= .5 * acc[2] - 1e-9)
    out["i0"] = dict(noop=round(noop, 3), F_RES26_5from2=boot(L("RES26|5<2") == tgt(2)), acc2=round(acc[2], 3))
    S = {}
    for (k, kp) in ((5, 2), (3, 2), (2, 5)):
        base = float(np.mean(L(f"E{k}") == tgt(kp))); full = float(np.mean(L(f"RES26|{k}<{kp}") == tgt(kp)))
        for arm in arms:
            f = float(np.mean(L(f"{arm}|{k}<{kp}") == tgt(kp)))
            S[(arm, k, kp)] = (f - base) / (full - base) if full - base > 1e-9 else float("nan")
            out["F"][f"{arm}|{k}<{kp}"] = boot(L(f"{arm}|{k}<{kp}") == tgt(kp)) if arm in GROUPS + ["RES26"] else round(f, 3)
            out["S"][f"{arm}|{k}<{kp}"] = round(S[(arm, k, kp)], 3)
        out["F"][f"base|{k}<{kp}"] = round(base, 3)
    sing = [a for a in SINGLE if S[(a, 5, 2)] >= .5 - 1e-9 and S[(a, 3, 2)] >= .5 - 1e-9]
    grp = [a for a in GROUPS if S[(a, 5, 2)] >= .5 - 1e-9 and S[(a, 3, 2)] >= .5 - 1e-9]
    out["checks"]["OP"] = ("SINGLE", sing) if sing else ("DISTRIBUTED", grp) if grp else ("NOT_IN_23_26", [])
    # error carrier on (2 <- 5)
    dc = L("E5"); rn = L("E2"); wrong = np.where(dc != tgt(5))[0]
    base_c = float(np.mean(rn[wrong] == dc[wrong]))
    C = {arm: float(np.mean(L(f"{arm}|2<5")[wrong] == dc[wrong])) for arm in arms}
    out["C"] = {arm: round(v, 3) for arm, v in C.items()}; out["C"]["baseline"] = round(base_c, 3); out["C"]["n_wrong"] = int(len(wrong))
    out["checks"]["ERROR_CARRIER"] = [a for a in SINGLE + GROUPS if C[a] >= .5 * C["RES26"] - 1e-9 and C[a] >= base_c + .10 - 1e-9]
    print(f"[{tag}] acc {out['acc']} · I0 {out['i0']} -> {out['checks']['I0']}")
    print(f"[{tag}] OP {out['checks']['OP']} · ERROR_CARRIER {out['checks']['ERROR_CARRIER']}")
    for (k, kp) in ((5, 2), (3, 2), (2, 5)):
        print(f"   {k}<-{kp} base {out['F'][f'base|{k}<{kp}']} · S: " + " ".join(f"{a} {out['S'][f'{a}|{k}<{kp}']:+.2f}" for a in arms))
    print(f"   C (2<-5, donor wrong n={len(wrong)}, baseline {base_c:.2f}): " + " ".join(f"{a} {C[a]:.2f}" for a in arms))
    return out


if __name__ == "__main__":
    res = {t: analyze(t) for t in sys.argv[1:]}
    json.dump(res, open("/vlm/results/r3/s751_analysis.json", "w"), indent=1)
