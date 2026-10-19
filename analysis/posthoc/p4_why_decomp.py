"""Why does combining ordinal, anchor and visual-prefix information give a wrong address? (post hoc, NOT registered; user 10-18 after S856b /
S857: 'z_i = f_ordinal(i; k, p) + f_visual(i; X) + f_interaction' is only a functional hypothesis - check each term against the model's own
late lookup.)  No new run.  Stored data:
  S854-1  N_a {6, 8} x N_s {6, 8} x k {3, 5} x p {3, 4}, END pinned at p (E) and START(k) pinned at p (S), plus natural END (unpinned)
  S852    N_a = N_s = 8, k {3, 4, 5} x p {3, 4, 5, 6}, END pinned (E_EARLY) and START(k) pinned (S_EARLY)
  S857    N8 / N6 / cut arms, k {3, 5} x p {3, 4}
Address readouts per run: the answer position (1..N_a; non-picture answers dropped) and the late lookup centre mu_L = sum_i i * pi_i, pi = the
softmax over the picture blocks of the late block scores (rows B1, B4, answer; layer 51, and the late mean 51-63), an analysis readout of where
the late queries point (not the attention mass over all tokens).
Address law: mu = c + b_p p + b_k k + b_N N_a + b_s N_s (+ interactions), least squares over all runs, scene bootstrap 2,000 (seed 3).  The
reverse reference needs (b_p, b_k, b_N, b_s) = (0, -1, +1, 0); the forward reference (START) needs (0, +1, 0, 0).
Per-term causal anchors: the visual term = mu(N8) - mu(CA) per (k, p) in S857 (the complete source cut reproduces the 6-picture state), required
+2 for every (k, p); the ordinal term = mu(k 5) - mu(k 3) at fixed p, N_a (required -2 under END, +2 under START); the anchor term = mu(p 4) -
mu(p 3) at fixed k, N_a (required 0).
Usage: python3 exp/p4_why_decomp.py  ->  research/ces/outputs/why_decomp.json"""
from __future__ import annotations

import json
import numpy as np

V = "/home/storage2/ai26269345/vlm"
NB = 2000


def centre(qk, L, nb, rows=(2, 5, 9)):
    ps = []
    for ri in rows:
        z = np.asarray(qk[L][ri][:nb], dtype=float); z = z[np.isfinite(z)][:nb]
        e = np.exp(z - z.max()); ps.append(e / e.sum())
    p = np.mean(ps, 0)
    return float(np.sum(np.arange(1, len(p) + 1) * p))


def late_centre(qk, LATE, nb):
    return float(np.mean([centre(qk, L, nb) for L in LATE]))


def c6(qk, L):
    """centre over the shared pictures 1-6 only (renormalised): free of the 'more blocks' artefact of a diffuse lookup."""
    return centre(qk, L, 6)


def c6L(qk, LATE):
    return float(np.mean([c6(qk, L) for L in LATE]))


def ols(rows, cols, ycol):
    X = np.array([[1.0] + [r[c] for c in cols] for r in rows]); y = np.array([r[ycol] for r in rows])
    return np.linalg.lstsq(X, y, rcond=None)[0]


def fit(rows, cols, ycol, seed=3):
    rng = np.random.default_rng(seed)
    sis = sorted({r["si"] for r in rows}); bys = {s: [r for r in rows if r["si"] == s] for s in sis}
    b0 = ols(rows, cols, ycol); bs = []
    for _ in range(NB):
        pick = rng.choice(sis, len(sis)); rr = [r for s in pick for r in bys[s]]
        bs.append(ols(rr, cols, ycol))
    bs = np.array(bs)
    return {nm: [float(b0[i]), [float(np.percentile(bs[:, i], 2.5)), float(np.percentile(bs[:, i], 97.5))]] for i, nm in enumerate(["c"] + cols)}


def main():
    out = {}
    # ---- S854-1: full factorial
    d = json.load(open(f"{V}/results/r3/s854a_s854a_27b_count.json")); LATE = [str(l) for l in d["late"]]; L51 = LATE[0]
    rows = []
    for r in d["rows"]:
        for arm in ("E", "S"):
            for na in (6, 8):
                for ns in (6, 8):
                    for k in (3, 5):
                        for p in (3, 4):
                            nm = f"{arm}_s{ns}k{k}p{p}_a{na}"
                            a = r["land"][nm]
                            rows.append(dict(si=r["si"], arm=arm, N=na, Ns=ns, k=k, p=p, kN=k * na, kp=k * p, pN=p * na,
                                             ans=(a if 1 <= a <= na else np.nan), mu51=centre(r["qk"][nm], L51, na), muL=late_centre(r["qk"][nm], LATE, na),
                                             m6L=c6L(r["qk"][nm], LATE), **{f"m6_{L}": c6(r["qk"][nm], L) for L in d["read"] for L in [str(L)]}))
        for na in (6, 8):
            for ns in (6, 8):
                for k in (3, 5):
                    nm = f"END_s{ns}k{k}_a{na}"; a = r["land"][nm]
                    rows.append(dict(si=r["si"], arm="ENDnat", N=na, Ns=ns, k=k, p=np.nan, kN=k * na, ans=(a if 1 <= a <= na else np.nan),
                                     mu51=centre(r["qk"][nm], L51, na), muL=late_centre(r["qk"][nm], LATE, na),
                                     mu39=centre(r["qk"][nm], "39", na), m6L=c6L(r["qk"][nm], LATE),
                                     **{f"m6_{L}": c6(r["qk"][nm], L) for L in [str(x) for x in d["read"]]}))
    res = {}
    for arm in ("E", "S"):
        R = [x for x in rows if x["arm"] == arm]
        res[arm] = {}
        res[arm]["m6_by_layer"] = {L: fit(R, ["p", "k", "N", "Ns"], f"m6_{L}") for L in [str(x) for x in d["read"]]}
        for y in ("mu51", "muL", "m6L", "ans"):
            RR = [x for x in R if np.isfinite(x[y])]
            res[arm][y] = dict(main=fit(RR, ["p", "k", "N", "Ns"], y), inter=fit(RR, ["p", "k", "N", "Ns", "kN", "kp", "pN"], y))
    R = [x for x in rows if x["arm"] == "ENDnat"]
    res["ENDnat"] = {y: fit([x for x in R if np.isfinite(x[y])], ["k", "N", "Ns"], y) for y in ("mu39", "mu51", "muL", "m6L", "ans")}
    res["ENDnat"]["m6_by_layer"] = {L: fit(R, ["k", "N", "Ns"], f"m6_{L}") for L in [str(x) for x in d["read"]]}
    res["ENDnat"]["m6L_inter"] = fit(R, ["k", "N", "Ns", "kN"], "m6L")
    out["S854_1"] = res
    # cell means (E): where the late lookup and the answer sit, against the targets
    cells = {}
    for na in (6, 8):
        for k in (3, 5):
            for p in (3, 4):
                R = [x for x in rows if x["arm"] == "E" and x["N"] == na and x["Ns"] == 8 and x["k"] == k and x["p"] == p]
                cells[f"N{na}k{k}p{p}"] = dict(target=na + 1 - k, mu51=float(np.mean([x["mu51"] for x in R])), muL=float(np.mean([x["muL"] for x in R])),
                                               ans_mean=float(np.nanmean([x["ans"] for x in R])),
                                               ans_dist={str(j): float(np.mean([x["ans"] == j for x in R])) for j in range(1, na + 1)})
    out["S854_1_cells_Ns8"] = cells
    # ---- S852: wider p and k at N = 8
    d2 = json.load(open(f"{V}/results/r3/s852_s852_27b_form.json")); LATE2 = [str(l) for l in d2["late"]]
    rows2 = []
    for r in d2["rows"]:                                                          # one row per (scene, k)
        k = r["k"]
        for arm in ("E_EARLY", "S_EARLY"):
            for p in (3, 4, 5, 6):
                    nm = f"{arm}_p{p}"
                    a = r["land"][nm]
                    rows2.append(dict(si=r["si"], arm=arm, k=k, p=p, kp=k * p, ans=(a if 1 <= a <= 8 else np.nan),
                                      mu51=centre(r["qk"][nm], LATE2[0], 8), muL=late_centre(r["qk"][nm], LATE2, 8)))
    res2 = {}
    for arm in ("E_EARLY", "S_EARLY"):
        R = [x for x in rows2 if x["arm"] == arm]
        res2[arm] = {y: dict(main=fit([x for x in R if np.isfinite(x[y])], ["p", "k"], y), inter=fit([x for x in R if np.isfinite(x[y])], ["p", "k", "kp"], y))
                     for y in ("mu51", "muL", "ans")}
    out["S852"] = res2
    # ---- S857: the visual term per (k, p), its size against the required +2
    d3 = json.load(open(f"{V}/results/r3/s857_s857_27b_cut.json")); LATE3 = [str(l) for l in d3["late"]]
    vis = {}
    rng = np.random.default_rng(5)
    for k in (3, 5):
        for p in (3, 4):
            v6 = np.array([c6L(r["qk"][f"N8_k{k}p{p}"], LATE3) - c6L(r["qk"][f"CA_k{k}p{p}"], LATE3) for r in d3["rows"]])
            bi = rng.integers(0, len(v6), (NB, len(v6)))
            vis[f"m6L_k{k}p{p}"] = [float(v6.mean()), [float(np.percentile(v6[bi].mean(1), 2.5)), float(np.percentile(v6[bi].mean(1), 97.5))]]
            dv, d86, dans, dans86 = [], [], [], []
            for r in d3["rows"]:
                m8 = late_centre(r["qk"][f"N8_k{k}p{p}"], LATE3, 8); mca = late_centre(r["qk"][f"CA_k{k}p{p}"], LATE3, 8)
                m6 = late_centre(r["qk"][f"N6_k{k}p{p}"], LATE3, 6)
                dv.append(m8 - mca); d86.append(m8 - m6)
                a8, aca, a6 = r["land"][f"N8_k{k}p{p}"], r["land"][f"CA_k{k}p{p}"], r["land"][f"N6_k{k}p{p}"]
                if a8 >= 1 and aca >= 1:
                    dans.append(a8 - aca)
                if a8 >= 1 and a6 >= 1:
                    dans86.append(a8 - a6)
            vis[f"k{k}p{p}"] = dict(muL_N8_minus_CA=float(np.mean(dv)), muL_N8_minus_N6=float(np.mean(d86)), ans_N8_minus_CA=float(np.mean(dans)),
                                   ans_N8_minus_N6=float(np.mean(dans86)), required=2)
    out["S857_visual_term"] = vis
    json.dump(out, open(f"{V}/research/ces/outputs/why_decomp.json", "w"), indent=1)
    f = lambda t: f"{t[0]:+.2f} [{t[1][0]:+.2f}, {t[1][1]:+.2f}]"
    print("S854-1 address law (required END: p 0, k -1, N +1, Ns 0; START: p 0, k +1, N 0)")
    for arm in ("E", "S", "ENDnat"):
        print(f"  {arm} m6 (centre over pictures 1-6) by layer: " + "  ".join(f"L{L} N {f(v['N'])} k {f(v['k'])}" for L, v in res[arm]["m6_by_layer"].items()))
    for arm in ("E", "S"):
        m = res[arm]["m6L"]["main"]; print(f"  {arm} m6L  " + "  ".join(f"{nm} {f(m[nm])}" for nm in ("p", "k", "N", "Ns")))
        m = res[arm]["m6L"]["inter"]; print(f"  {arm} m6L inter " + "  ".join(f"{nm} {f(m[nm])}" for nm in ("kN", "kp", "pN")))
    print("  ENDnat m6L", {nm: f(v) for nm, v in res["ENDnat"]["m6L"].items()}, "| inter kN", f(res["ENDnat"]["m6L_inter"]["kN"]))
    for arm in ("E", "S"):
        for y in ("mu51", "muL", "ans"):
            m = res[arm][y]["main"]; print(f"  {arm} {y:5s} " + "  ".join(f"{nm} {f(m[nm])}" for nm in ("p", "k", "N", "Ns")))
        for y in ("muL", "ans"):
            m = res[arm][y]["inter"]; print(f"  {arm} {y:5s} inter " + "  ".join(f"{nm} {f(m[nm])}" for nm in ("kN", "kp", "pN")))
    for y in ("mu39", "mu51", "muL", "ans"):
        m = res["ENDnat"][y]; print(f"  END natural {y:5s} " + "  ".join(f"{nm} {f(m[nm])}" for nm in ("k", "N", "Ns")))
    print("S852 (N 8): ")
    for arm in ("E_EARLY", "S_EARLY"):
        for y in ("muL", "ans"):
            m = res2[arm][y]["main"]; mi = res2[arm][y]["inter"]; print(f"  {arm} {y:4s} p {f(m['p'])}  k {f(m['k'])}  | kp {f(mi['kp'])}")
    print("cells (E, N_s 8):", json.dumps({k_: dict(target=v["target"], muL=round(v["muL"], 2), ans=round(v["ans_mean"], 2)) for k_, v in cells.items()}))
    print("S857 visual term:", json.dumps({k_: (v if isinstance(v, list) else {kk: round(vv, 2) for kk, vv in v.items()}) for k_, v in vis.items()}))
    for arm in ("E",):
        m = res[arm]["ans"]["inter"]; print("  E ans full interaction model:", {nm: f(v) for nm, v in m.items()})


if __name__ == "__main__":
    main()
