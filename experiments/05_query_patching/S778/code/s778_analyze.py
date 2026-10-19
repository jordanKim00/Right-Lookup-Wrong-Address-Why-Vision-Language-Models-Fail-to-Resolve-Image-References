"""S778 analysis, frozen with the pre-registration (knowledge/DECISIONS.md, 2026-10-14 12:2x) before any S778 data.
Per cell: ta = N - 4, tb = N - 1.  fix(G) = P(FIX@G|E5, ta) - P(E5, ta) (E5's rows G use Sa's L35-47 queries); nec_S(G) = P(Sa, ta) - P(KO@G|Sa, ta),
nec_E(G) = P(E2, tb) - P(KO@G|E2, tb) (rows G may not read image tokens at L35-47).  Groups: Q (question end and menu rows), B (turn boundary), R (think
markers and the answer row), BR, ALL.
I0           NOOP agrees with the natural answer on >= .95 of scenes for E5, Sa and E2.
FIX_G        fix(G) >= .20 for G in Q, B, R, ALL.
FIX_LOCUS    among Q, B, R: the group whose fix exceeds both others by >= .10 (if it is >= .20); SHARED if the largest is >= .20 but not .10 ahead; NONE otherwise.
NEC_START_G  nec_S(G) >= .20 for G in B, R, BR, ALL.      NEC_E2_G  nec_E(G) >= .20.
ROLE_START   B_NEEDED / R_NEEDED / BOTH_NEEDED (nec_S(B), nec_S(R) >= .20); REDUNDANT if neither but nec_S(BR) >= .20; DISTRIBUTED if only nec_S(ALL) >= .20; NONE.
ROLE_E2      the same on nec_E.
Descriptive: the answer distributions; KOlate@R (rows R blind to images at L51-59); the boundary row's and the answer row's query-based target-block
share under every arm at L35-47 (block scores from the query; the knockout masks the attention, not these scores).
Usage: python s778_analyze.py TAG [TAG ...]
"""
import json, os, sys
import numpy as np


def sm(sc):
    x = sc - sc.max(-1, keepdims=True); e = np.exp(x); return e / e.sum(-1, keepdims=True)


def role(n):
    b, r = n["B"] >= .20 - 1e-9, n["R"] >= .20 - 1e-9
    if b and r:
        return "BOTH_NEEDED"
    if b:
        return "B_NEEDED"
    if r:
        return "R_NEEDED"
    if n["BR"] >= .20 - 1e-9:
        return "REDUNDANT"
    return "DISTRIBUTED" if n["ALL"] >= .20 - 1e-9 else "NONE"


def cell(tag):
    D = f"/vlm/results/r3/s778_{tag}_readrows"
    d = json.load(open(D + ".json")); M = d["meta"]; z = np.load(D + ".npz"); N, ta, tb = d["N"], d["ta"], d["tb"]
    A_ = lambda m, k: m["ans"][k]["pos"]
    P = lambda k, b: float(np.mean([A_(m, k) == b for m in M]))
    arms = list(M[0]["ans"])
    out = dict(tag=tag, N=N, n=len(M), checks={}, dist={k: {str(b): round(P(k, b), 3) for b in range(0, N + 1) if P(k, b) > 0} for k in arms})
    agree = lambda a, b: float(np.mean([A_(m, a) == A_(m, b) for m in M]))
    i0 = {nm: round(agree(f"NOOP|{nm}", nm), 3) for nm in ("E5", "Sa", "E2")}
    out["i0"] = i0; out["checks"]["I0"] = bool(all(x >= .95 - 1e-9 for x in i0.values()))
    fix = {g: round(P(f"FIX@{g}|E5", ta) - P("E5", ta), 3) for g in ("Q", "B", "R", "ALL")}
    out["fix"] = fix
    for g in fix:
        out["checks"][f"FIX_{g}"] = bool(fix[g] >= .20 - 1e-9)
    tri = {g: fix[g] for g in ("Q", "B", "R")}
    best = max(tri, key=tri.get)
    out["checks"]["FIX_LOCUS"] = (best if tri[best] >= .20 - 1e-9 and all(tri[best] >= tri[g] + .10 - 1e-9 for g in tri if g != best) else
                                  "SHARED" if tri[best] >= .20 - 1e-9 else "NONE")
    nS = {g: round(P("Sa", ta) - P(f"KO@{g}|Sa", ta), 3) for g in ("B", "R", "BR", "ALL")}
    nE = {g: round(P("E2", tb) - P(f"KO@{g}|E2", tb), 3) for g in ("B", "R", "BR", "ALL")}
    out["nec_start"] = nS; out["nec_e2"] = nE
    out["late_ko"] = dict(Sa=round(P("Sa", ta) - P("KOlate@R|Sa", ta), 3), E2=round(P("E2", tb) - P("KOlate@R|E2", tb), 3))
    for g in nS:
        out["checks"][f"NEC_START_{g}"] = bool(nS[g] >= .20 - 1e-9); out["checks"][f"NEC_E2_{g}"] = bool(nE[g] >= .20 - 1e-9)
    out["checks"]["ROLE_START"] = role(nS); out["checks"]["ROLE_E2"] = role(nE)
    look = {}
    for k in arms:
        tgt = ta if k.endswith("E5") or k.endswith("Sa") else tb
        r = {}
        for li in d["RB"]["early"] + d["RB"]["mid"]:
            s = np.mean([sm(z[f"{m['si']}|{k}|{li}"].astype(np.float64)) .mean(1)[:, tgt - 1] for m in M], 0)
            r[str(li)] = [round(float(s[0]), 3), round(float(s[1]), 3)]
        look[k] = r
    out["target_share_rows"] = look
    print(f"[{tag}] N {N} · checks {out['checks']} · I0 {i0}")
    print(f"   fix {fix} · nec START {nS} · nec E2 {nE} · late R ko {out['late_ko']}")
    for k in ("Sa", "KO@B|Sa", "KO@R|Sa", "E2", "KO@B|E2", "E5", "FIX@B|E5", "FIX@R|E5"):
        print(f"   target share (boundary row, answer row) {k}: {look[k]}")
    return out


if __name__ == "__main__":
    json.dump({t: cell(t) for t in sys.argv[1:]}, open(os.environ.get("ANALYSIS_OUT", "/vlm/results/r3/s778_analysis.json"), "w"), indent=1)
