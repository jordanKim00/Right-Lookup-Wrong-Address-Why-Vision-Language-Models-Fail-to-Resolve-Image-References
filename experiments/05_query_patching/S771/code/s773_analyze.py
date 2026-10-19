"""S773 analysis, frozen with the pre-registration (knowledge/DECISIONS.md, 2026-10-14 01:4x) before any S773 data.
Near writers (readers L43-59): Nc (END(2) <- START(2)), d6(x) = P6(E2) - P6(x); B (END(5) <- END(2)), g6(x) = P6(x) - P6(E5).
  gap = effect(DIR_M42) - effect(DIR_M34) (what L35-42 adds; S771 Nc .55, B .15); E(w) = effect(M34 + w) - effect(DIR_M34) for w in FA35, FA39,
  DN35-42, MLP35-42.
NEAR_GAP      Nc gap >= .25 (S771 replicated).
NEAR_WRITER   (Nc, if NEAR_GAP) exactly one w with E(w) >= .5 gap -> w; two or more -> REDUNDANT; none -> JOINT.
NEAR_WRITER_B (B, if its gap >= .15) the same rule on g6.
Far writers (late readers L51-59 only): C (END(2) <- END(5)), rep(x) = on the error scenes (natural END(5) answers neither 3 nor END(2)'s answer)
  the share whose answer equals natural END(5)'s; A (END(5) <- F5P1), r(x) = late-head alignment r_Hlate (S771 head set and definition).
FAR_INBAND    rep(C_DIR_M50) - rep(C_DIR_M42) >= .20 and rep(C_DIR_M50) >= .5 rep(C_QTOT@late): the L43-50 writes carry END(5)'s late error.
FAR_WRITER    (if FAR_INBAND) exactly one w in FA43, FA47, DN43-50, MLP43-50 with rep(C_M42 + w) - rep(C_DIR_M42) >= .5 (rep(M50) - rep(M42));
              two or more -> REDUNDANT; none -> JOINT.
LATE_INBAND_A r(A_DIR_M50) - r(A_DIR_M42) >= .25.   LATE_WRITER_A (if LATE_INBAND_A) the same writer rule on r.
Instrument I0: NOOP and DIR0 for END(5), END(2) agree with the natural answers on >= .95 of scenes.
Descriptive: every arm's answer distribution, all effects with scene-bootstrap 95% intervals for rep, the phi of pair A, the late readers' weight
on END(5)'s answered menu word for pair C.
Usage: python s773_analyze.py TAG
"""
import json, sys
import numpy as np


def boot(x, B=4000, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return None
    r = np.random.default_rng(seed); m = [x[r.integers(0, len(x), len(x))].mean() for _ in range(B)]
    return [round(float(x.mean()), 3), round(float(np.percentile(m, 2.5)), 3), round(float(np.percentile(m, 97.5)), 3)]


def sm(sc):
    x = sc - sc.max(-1, keepdims=True); e = np.exp(x); return e / e.sum(-1, keepdims=True)


def writer(eff, base, gap, names, pre):
    big = [w for w in names if eff[f"{pre}{w}"] - base >= .5 * gap - 1e-9]
    return big[0] if len(big) == 1 else "REDUNDANT" if len(big) >= 2 else "JOINT"


def analyze(tag):
    D = f"/vlm/results/r3/s773_{tag}_writers"
    d = json.load(open(D + ".json")); M = d["meta"]; n = len(M); z = np.load(D + ".npz")
    A_ = lambda m, k: m["ans"][k]["pos"]
    P = lambda k, b: float(np.mean([A_(m, k) == b for m in M]))
    arms = list(M[0]["ans"])
    out = dict(n=n, checks={}, dist={k: {str(b): round(P(k, b), 3) for b in range(0, 8) if P(k, b) > 0} for k in arms})
    agree = lambda a, b: float(np.mean([A_(m, a) == A_(m, b) for m in M]))
    i0 = {f"{k}_{nm}": round(agree(f"{k}_{nm}", nm), 3) for nm in ("E5", "E2") for k in ("NOOP", "DIR0")}
    out["i0"] = i0; out["checks"]["I0"] = bool(all(x >= .95 - 1e-9 for x in i0.values()))
    NW = ("FA35", "FA39", "DN35-42", "MLP35-42"); FW = ("FA43", "FA47", "DN43-50", "MLP43-50")
    q6 = P("E2", 6); d6 = {k: round(q6 - P(k, 6), 3) for k in arms if k.startswith("Nc_")}; out["d6"] = d6
    gN = d6["Nc_DIR_M42"] - d6["Nc_DIR_M34"]; out["near_gap"] = round(gN, 3)
    out["checks"]["NEAR_GAP"] = bool(gN >= .25 - 1e-9)
    out["checks"]["NEAR_WRITER"] = writer(d6, d6["Nc_DIR_M34"], gN, NW, "Nc_M34+") if out["checks"]["NEAR_GAP"] else None
    p6 = P("E5", 6); g6 = {k: round(P(k, 6) - p6, 3) for k in arms if k.startswith("B_")}; out["g6"] = g6
    gB = g6["B_DIR_M42"] - g6["B_DIR_M34"]; out["near_gap_B"] = round(gB, 3)
    out["checks"]["NEAR_WRITER_B"] = writer(g6, g6["B_DIR_M34"], gB, NW, "B_M34+") if gB >= .15 - 1e-9 else None
    err = [m for m in M if A_(m, "E5") != 3 and A_(m, "E5") != A_(m, "E2")]
    out["n_err"] = len(err)
    rep = {k: float(np.mean([A_(m, k) == A_(m, "E5") for m in err])) if err else float("nan") for k in arms if k.startswith("C_")}
    out["rep"] = {k: boot([A_(m, k) == A_(m, "E5") for m in err]) for k in arms if k.startswith("C_")}
    gC = rep["C_DIR_M50"] - rep["C_DIR_M42"]
    out["checks"]["FAR_INBAND"] = bool(gC >= .20 - 1e-9 and rep["C_DIR_M50"] >= .5 * rep["C_QTOT@late"] - 1e-9)
    out["checks"]["FAR_WRITER"] = writer(rep, rep["C_DIR_M42"], gC, FW, "C_M42+") if out["checks"]["FAR_INBAND"] else None
    HL = {51: [2, 3, 23, 4, 22, 12], 55: [13, 14, 2, 15, 0, 16], 59: [21, 3, 20, 1, 5, 8]}
    if d["nl"] == 64:
        sH = lambda k: float(np.mean([np.mean([sm(z[f"{m['si']}|{k}|{li}"].astype(np.float64))[1][h][4] for li, H in HL.items() for h in H]) for m in M]))
        e5, f5 = sH("E5"), sH("F5P1"); gap = f5 - e5
        r = {k: round((sH(k) - e5) / gap, 3) if gap >= .10 - 1e-9 else None for k in arms if k.startswith("A_")}
        out["r_late"] = r; out["late_head_gap"] = round(gap, 3)
        if gap >= .10 - 1e-9:
            gA = r["A_DIR_M50"] - r["A_DIR_M42"]
            out["checks"]["LATE_INBAND_A"] = bool(gA >= .25 - 1e-9)
            out["checks"]["LATE_WRITER_A"] = writer(r, r["A_DIR_M42"], gA, FW, "A_M42+") if out["checks"]["LATE_INBAND_A"] else None
    e5p, f5p = P("E5", 5), P("F5P1", 5)
    out["phi_A"] = {k: round((P(k, 5) - e5p) / (f5p - e5p), 3) if f5p > e5p else None for k in arms if k.startswith("A_")}
    word = lambda m, k: int(np.argmax(m["ans"][k]["ll"]))
    RL = d["RB"]["late"]
    out["C_late_option"] = {k: round(float(np.mean([np.mean([(lambda w: w[word(m, "E5")] / max(w.sum(), 1e-12))(z[f"{m['si']}|{k}|{li}|opt"].astype(np.float64)[1].mean(0))
                                                             for li in RL]) for m in err])), 3) if err else None
                            for k in ["E2", "E5"] + [x for x in arms if x.startswith("C_")]}
    print(f"[{tag}] n {n} · checks {out['checks']} · I0 {i0}")
    print(f"   d6 {d6} · near gap {out['near_gap']}\n   g6 {g6} · gap B {out['near_gap_B']}\n   rep (n_err {len(err)}) {out['rep']}")
    print(f"   r_late {out.get('r_late')} · phi_A {out['phi_A']}\n   C late option share of END(5)'s word {out['C_late_option']}")
    return out


if __name__ == "__main__":
    json.dump({sys.argv[1]: analyze(sys.argv[1])}, open("/vlm/results/r3/s773_analysis.json", "w"), indent=1)
