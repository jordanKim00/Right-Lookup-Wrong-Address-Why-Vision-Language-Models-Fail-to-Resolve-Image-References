"""S750 analysis, frozen with the pre-registration (knowledge/DECISIONS.md, 2026-10-12 22:3x) before any S750 data.
F(pair, L, G) = answers on the donor's target; base(pair) = the recipient's natural answers on the donor's target; acc(k') = the donor's
natural accuracy; E = F - base.
I0          NOOP answers equal the natural answers on >= .95 of (scene, k in {2, 5}), and F(5<-2, L34, ALL) >= .5 * acc(2)
BIRTH(pair, G)  the earliest L with F >= base + .5 * (acc - base) and F >= base + .10 (None if no layer qualifies)
CARRIER     at L* = BIRTH(5<-2, ALL): the groups G != ALL with F(5<-2, L*, G) >= base + .5 * (F(5<-2, L*, ALL) - base)
FAR_MERGED  at L34, ALL: E(5<-4) <= .15 and E(3<-2) >= .30
Descriptive: F and E for every (pair, L, G), scene-bootstrap 95% intervals (4,000 draws) for the rule cells.
Usage: python s750_analyze.py TAG
"""
import json, sys
import numpy as np

N = 7


def boot(x, B=4000, seed=0):
    x = np.asarray(x, float); r = np.random.default_rng(seed)
    m = [x[r.integers(0, len(x), len(x))].mean() for _ in range(B)]
    return [round(float(x.mean()), 3), round(float(np.percentile(m, 2.5)), 3), round(float(np.percentile(m, 97.5)), 3)]


def analyze(tag):
    D = f"/vlm/results/r3/s750_{tag}_where"
    meta = json.load(open(D + ".json"))
    M = meta["meta"]; RES = meta["RES"]; G = meta["GROUPS"]; PAIRS = [tuple(p) for p in meta["PAIRS"]]; n = len(M)
    tgt = lambda k: N + 1 - k
    L = lambda key: np.array([m["land"][key] for m in M])
    acc = {k: float(np.mean(L(f"E{k}") == tgt(k))) for k in meta["KS"]}
    base = {(k, kp): float(np.mean(L(f"E{k}") == tgt(kp))) for (k, kp) in PAIRS}
    F = {(k, kp, l, g): float(np.mean(L(f"R{l}|{g}|{k}<{kp}") == tgt(kp))) for (k, kp) in PAIRS for l in RES for g in G}
    out = dict(n=n, acc={str(k): round(v, 3) for k, v in acc.items()}, base={f"{k}<{kp}": round(v, 3) for (k, kp), v in base.items()},
               checks={}, F={}, birth={})
    noop = float(np.mean([M[i]["land"][f"NOOP|{k}"] == M[i]["land"][f"E{k}"] for i in range(n) for k in (2, 5)]))
    top = max(RES)
    f52 = F[(5, 2, top, "ALL")]
    out["checks"]["I0"] = bool(noop >= .95 - 1e-9 and f52 >= .5 * acc[2] - 1e-9)
    out["i0"] = dict(noop=round(noop, 3), F_5from2_top_ALL=boot(L(f"R{top}|ALL|5<2") == tgt(2)), acc2=round(acc[2], 3))
    for (k, kp) in PAIRS:
        b = base[(k, kp)]; a = acc[kp]
        for g in G:
            birth = None
            for l in sorted(RES):
                f = F[(k, kp, l, g)]
                if f >= b + .5 * (a - b) - 1e-9 and f >= b + .10 - 1e-9:
                    birth = l; break
            out["birth"][f"{k}<{kp}|{g}"] = birth
            out["F"][f"{k}<{kp}|{g}"] = {str(l): round(F[(k, kp, l, g)], 3) for l in sorted(RES)}
    ls = out["birth"]["5<2|ALL"]
    if ls is not None:
        fa = F[(5, 2, ls, "ALL")]; b = base[(5, 2)]
        out["checks"]["CARRIER"] = [g for g in G if g != "ALL" and F[(5, 2, ls, g)] >= b + .5 * (fa - b) - 1e-9]
    else:
        out["checks"]["CARRIER"] = None
    out["checks"]["BIRTH_5from2_ALL"] = ls
    e54 = F[(5, 4, top, "ALL")] - base[(5, 4)]; e32 = F[(3, 2, top, "ALL")] - base[(3, 2)]
    out["checks"]["FAR_MERGED"] = bool(e54 <= .15 + 1e-9 and e32 >= .30 - 1e-9)
    out["merge_detail"] = dict(E_5from4=round(e54, 3), E_3from2=round(e32, 3),
                               E_5from4_ci=boot((L(f"R{top}|ALL|5<4") == tgt(4)).astype(float) - (L("E5") == tgt(4)).astype(float)),
                               E_3from2_ci=boot((L(f"R{top}|ALL|3<2") == tgt(2)).astype(float) - (L("E3") == tgt(2)).astype(float)))
    print(f"[{tag}] acc {out['acc']} · base {out['base']} · I0 {out['i0']} -> {out['checks']['I0']}")
    print(f"[{tag}] BIRTH(5<-2, ALL) {ls} · CARRIER {out['checks']['CARRIER']} · FAR_MERGED {out['checks']['FAR_MERGED']} {out['merge_detail']}")
    for (k, kp) in PAIRS:
        for g in G:
            print(f"   {k}<-{kp} {g:4s} birth {str(out['birth'][f'{k}<{kp}|{g}']):4s} F by L {out['F'][f'{k}<{kp}|{g}']}")
    return out


if __name__ == "__main__":
    res = {t: analyze(t) for t in sys.argv[1:]}
    json.dump(res, open("/vlm/results/r3/s750_analysis.json", "w"), indent=1)
