"""S771 analysis, frozen with the pre-registration (knowledge/DECISIONS.md, 2026-10-14 00:2x) before any S771 data.
Pair A (END(5) <- F5P1): phi(x) = (P5(x) - P5(E5)) / (P5(F5P1) - P5(E5)), P5 = share answering image 5 (the frame-driven literal recovery).
Pair B (END(5) <- END(2)): g6(x) = P6(x) - P6(E5), P6 = share answering image 6 (END(2)'s target).
Pair C (END(2) <- END(5)): on the error scenes (natural END(5) answers neither 3 nor END(2)'s answer), rep(x) = share whose answer equals natural END(5)'s.
Pair Nc (END(2) <- START(2)): d6(x) = P6(E2) - P6(x) (END(2)'s correct answer lost when its readers get START(2)'s message).
I0          NOOP_E5 = E5, NOOP_E2 = E2, DIR0_E5 = E5, DIR0_E2 = E2 in answers on >= .95 of scenes.
REPL_A      P5(F5P1) - P5(E5) >= .30.
A_Q         phi(A_QTOT@ML) >= .50: the frame effect reaches the answer through the L43-59 queries.
A_DIRECT    phi(A_DIR_M34) >= .5 phi(A_QTOT@ML): the readers take the message as it stands after the L27-34 frame read (no L35-42 processing needed).
A_HOP       phi(A_DIR_M42) - phi(A_DIR_M34) >= .25: the early band (L35-42) adds to what the readers need.
SENDER      per pair (A on phi, Nc on d6), read only if the pair's DIR_M34 effect e34 >= .25: DN / FA / MLP if exactly that L27-34 sublayer's direct
            path (DIR_P1X) has effect >= .5 e34; REDUNDANT if two or more; JOINT if none.
A_ROWS      CONV_ROWS if phi(A_DIR_M34c) >= .5 phi(A_DIR_M34) and > phi(A_DIR_M34r); OTHER_ROWS if the reverse; BOTH otherwise (read if phi(A_DIR_M34) >= .25).
B_Q         g6(B_QTOT@ML) >= .30.   B_ORIGIN (if B_Q): the earliest boundary b in M26, M34, M42 with g6(B_DIR_b) >= .5 g6(B_QTOT@ML):
            ORDINAL (M26) / FRAME (M34) / EARLY (M42); READER if none.
C_ERR       rep(C_QTOT@ML) >= .40 and >= rep(E2) + .25: END(5)'s L43-59 queries carry END(5)'s scene-specific errors into END(2).
C_ORIGIN    (if C_ERR) the same boundary rule on rep.
Nc_Q        d6(Nc_QTOT@ML) >= .30: END(2)'s success needs its own L43-59 queries.   Nc_ORIGIN (if Nc_Q): the same boundary rule on d6.
Head sets (fixed from S769's saved block scores before any S771 data: per layer the six heads with the largest START(4) - END(4) gap in the
answer row's block-4 share, the same sets at N = 7 and N = 8): H_mid = L43 {18, 23, 9, 7, 8, 17}, L47 {17, 13, 15, 2, 20, 14};
H_late = L51 {2, 3, 23, 4, 22, 12}, L55 {13, 14, 2, 15, 0, 16}, L59 {21, 3, 20, 1, 5, 8}.  r_H(x) = (s_H(x) - s_H(E5)) / (s_H(F5P1) - s_H(E5)),
s_H = mean over scenes and the set's heads of the answer row's softmax block-5 share; read only if s_H(F5P1) - s_H(E5) >= .10.
HEAD_MID_CONV   r_Hmid(A_DIR_P1DN) >= .50 and r_Hlate(A_DIR_P1DN) <= .25: the DeltaNet senders' direct message aligns the mid readers, not the late ones.
HEAD_LATE_CONJ  r_Hlate(A_DIR_P1DNFA) >= .50 while r_Hlate(A_DIR_P1DN) <= .25 and r_Hlate(A_DIR_P1FA) <= .25: the late readers need both mixer messages.
AMP_TOWARD  P3(AMP_E5_M34x2) >= P3(E5) + .15: amplifying END(5)'s own END-vs-START message at the readers moves the answer to the correct image 3
            (a weak but correctly directed END message); otherwise reported as NOT_TOWARD with the answer distribution and reader block shares (3, 5, 7).
Descriptive: every arm's answer distribution; phi / g6 / rep / d6 tables (scene-bootstrap 95% intervals for rep); mid vs late; from the reader block
scores (row "answer", head-mean softmax over the seven blocks) per reader layer: the share of the pair's target block, and for C the alignment
index ARI = JSD(arm, E2) / (JSD(arm, E2) + JSD(arm, E5)) (1 = END(5)'s alignment, 0 = END(2)'s); the readers' weight on the pair's menu word
(A: image 5's word, B / Nc: image 6's, C: natural END(5)'s answered word) as a share of their weight on all menu words.
Usage: python s771_analyze.py TAG
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


def jsd(a, b):
    m = .5 * (a + b); kl = lambda u, w: float(np.sum(u * (np.log2(u + 1e-12) - np.log2(w + 1e-12))))
    return .5 * (kl(a, m) + kl(b, m))


def boundary(eff, pre, tot):
    hit = [b for b in ("M26", "M34", "M42") if f"{pre}_DIR_{b}" in eff and eff[f"{pre}_DIR_{b}"] >= .5 * tot - 1e-9]
    return {"M26": "ORDINAL", "M34": "FRAME", "M42": "EARLY"}[hit[0]] if hit else "READER"


def sender(eff, pre):
    e34 = eff[f"{pre}_DIR_M34"]
    if e34 < .25 - 1e-9:
        return None
    big = [k for k in ("DN", "FA", "MLP") if eff[f"{pre}_DIR_P1{k}"] >= .5 * e34 - 1e-9]
    return big[0] if len(big) == 1 else "REDUNDANT" if len(big) >= 2 else "JOINT"


def analyze(tag):
    D = f"/vlm/results/r3/s771_{tag}_qpath"
    d = json.load(open(D + ".json")); M = d["meta"]; n = len(M); z = np.load(D + ".npz"); RL = d["RB"]["ML"]
    A_ = lambda m, k: m["ans"][k]["pos"]
    P = lambda k, b: float(np.mean([A_(m, k) == b for m in M]))
    agree = lambda a, b: float(np.mean([A_(m, a) == A_(m, b) for m in M]))
    arms = list(M[0]["ans"])
    out = dict(n=n, checks={}, dist={k: {str(b): round(P(k, b), 3) for b in range(0, 8) if P(k, b) > 0} for k in arms})
    i0 = {f"{k}_{nm}": round(agree(f"{k}_{nm}", nm), 3) for nm in ("E5", "E2") for k in ("NOOP", "DIR0")}
    out["i0"] = i0; out["checks"]["I0"] = bool(all(x >= .95 - 1e-9 for x in i0.values()))
    e5, f5 = P("E5", 5), P("F5P1", 5); R = f5 - e5
    out["A"] = dict(E5=round(e5, 3), F5P1=round(f5, 3), R=round(R, 3))
    out["checks"]["REPL_A"] = bool(R >= .30 - 1e-9)
    phi = {k: round((P(k, 5) - e5) / R, 3) if R > 0 else float("nan") for k in arms if k.startswith("A_")}; out["phi"] = phi
    out["checks"]["A_Q"] = bool(phi["A_QTOT@ML"] >= .50 - 1e-9)
    out["checks"]["A_DIRECT"] = bool(out["checks"]["A_Q"] and phi["A_DIR_M34"] >= .5 * phi["A_QTOT@ML"] - 1e-9) if out["checks"]["A_Q"] else None
    out["checks"]["A_HOP"] = bool(phi["A_DIR_M42"] - phi["A_DIR_M34"] >= .25 - 1e-9) if out["checks"]["A_Q"] else None
    out["checks"]["A_SENDER"] = sender(phi, "A")
    if phi["A_DIR_M34"] >= .25 - 1e-9:
        c, r = phi["A_DIR_M34c"], phi["A_DIR_M34r"]
        out["checks"]["A_ROWS"] = ("CONV_ROWS" if c >= .5 * phi["A_DIR_M34"] - 1e-9 and c > r else
                                   "OTHER_ROWS" if r >= .5 * phi["A_DIR_M34"] - 1e-9 and r > c else "BOTH")
    else:
        out["checks"]["A_ROWS"] = None
    p6 = P("E5", 6)
    g6 = {k: round(P(k, 6) - p6, 3) for k in arms if k.startswith("B_")}; out["g6"] = g6
    out["checks"]["B_Q"] = bool(g6["B_QTOT@ML"] >= .30 - 1e-9)
    out["checks"]["B_ORIGIN"] = boundary(g6, "B", g6["B_QTOT@ML"]) if out["checks"]["B_Q"] else None
    out["B_sender_desc"] = {k: g6[f"B_DIR_P1{k}"] for k in ("DN", "FA", "MLP")}
    err = [m for m in M if A_(m, "E5") != 3 and A_(m, "E5") != A_(m, "E2")]
    out["n_err"] = len(err)
    repv = lambda k: float(np.mean([A_(m, k) == A_(m, "E5") for m in err])) if err else float("nan")
    keysC = ["E2"] + [k for k in arms if k.startswith("C_")]
    out["rep"] = {k: boot([A_(m, k) == A_(m, "E5") for m in err]) for k in keysC}
    rep = {k: repv(k) for k in keysC}
    out["checks"]["C_ERR"] = bool(err and rep["C_QTOT@ML"] >= .40 - 1e-9 and rep["C_QTOT@ML"] >= rep["E2"] + .25 - 1e-9)
    out["checks"]["C_ORIGIN"] = boundary(rep, "C", rep["C_QTOT@ML"]) if out["checks"]["C_ERR"] else None
    q6 = P("E2", 6)
    d6 = {k: round(q6 - P(k, 6), 3) for k in arms if k.startswith("Nc_")}; out["d6"] = d6
    out["checks"]["Nc_Q"] = bool(d6["Nc_QTOT@ML"] >= .30 - 1e-9)
    out["checks"]["Nc_ORIGIN"] = boundary(d6, "Nc", d6["Nc_QTOT@ML"]) if out["checks"]["Nc_Q"] else None
    out["checks"]["Nc_SENDER"] = sender(d6, "Nc")
    out["Nc_to_literal2"] = {k: round(P(k, 2) - P("E2", 2), 3) for k in arms if k.startswith("Nc_")}
    amp = [k for k in arms if k.startswith("AMP_")]
    out["amp"] = {k: dict(P3=round(P(k, 3), 3), P5=round(P(k, 5), 3), P6=round(P(k, 6), 3), P7=round(P(k, 7), 3)) for k in amp + ["E5", "E2"]}
    out["checks"]["AMP_TOWARD"] = bool(P("AMP_E5_M34x2", 3) >= P("E5", 3) + .15 - 1e-9)
    HS = {"mid": {43: [18, 23, 9, 7, 8, 17], 47: [17, 13, 15, 2, 20, 14]},
          "late": {51: [2, 3, 23, 4, 22, 12], 55: [13, 14, 2, 15, 0, 16], 59: [21, 3, 20, 1, 5, 8]}}
    if d["nl"] == 64:
        def sH(k, hs, b=5):
            return float(np.mean([np.mean([sm(z[f"{m['si']}|{k}|{li}"].astype(np.float64))[1][h][b - 1] for li, H in hs.items() for h in H]) for m in M]))
        gapH = {g: sH("F5P1", hs) - sH("E5", hs) for g, hs in HS.items()}
        rH = {g: {k: round((sH(k, hs) - sH("E5", hs)) / gapH[g], 3) for k in arms if k.startswith("A_")} if gapH[g] >= .10 - 1e-9 else None
              for g, hs in HS.items()}
        out["head_gap"] = {g: round(x, 3) for g, x in gapH.items()}; out["r_head"] = rH
        ok = rH["mid"] is not None and rH["late"] is not None
        out["checks"]["HEAD_MID_CONV"] = bool(rH["mid"]["A_DIR_P1DN"] >= .50 - 1e-9 and rH["late"]["A_DIR_P1DN"] <= .25 + 1e-9) if ok else None
        out["checks"]["HEAD_LATE_CONJ"] = (bool(rH["late"]["A_DIR_P1DNFA"] >= .50 - 1e-9 and rH["late"]["A_DIR_P1DN"] <= .25 + 1e-9
                                                and rH["late"]["A_DIR_P1FA"] <= .25 + 1e-9) if rH["late"] is not None else None)
    # reader alignment (row "answer", head-mean block share)
    bs = lambda m, k, li: sm(z[f"{m['si']}|{k}|{li}"].astype(np.float64))[1].mean(0)
    tgt = {"A": lambda m: 5, "B": lambda m: 6, "Nc": lambda m: 6}
    align = {}
    for li in RL:
        row = {}
        for k in arms:
            pre = k.split("_")[0]
            if pre in tgt:
                row[k] = round(float(np.mean([bs(m, k, li)[tgt[pre](m) - 1] for m in M])), 3)
        for k in ("E5", "E2", "F5P1", "S2", "S5") + tuple(amp):
            row[k] = {f"b{b}": round(float(np.mean([bs(m, k, li)[b - 1] for m in M])), 3) for b in (3, 5, 6, 7)}
        if err:
            for k in keysC[1:]:
                ari = []
                for m in err:
                    a, e2_, e5_ = bs(m, k, li), bs(m, "E2", li), bs(m, "E5", li)
                    j2, j5 = jsd(a, e2_), jsd(a, e5_)
                    ari.append(j2 / (j2 + j5) if j2 + j5 > 0 else .5)
                row[f"ARI:{k}"] = round(float(np.mean(ari)), 3)
        align[str(li)] = row
    out["align"] = align
    # reader option reading (row "answer"): head-mean weight on the pair's option word, as a share of the weight on all menu words
    def osh(m, k, li, j):
        w = z[f"{m['si']}|{k}|{li}|opt"].astype(np.float64)[1].mean(0); return w[j] / max(w.sum(), 1e-12)
    word = lambda m, k: (m["labs"] + m["extra"]).index((m["labs"] + m["extra"])[int(np.argmax(m["ans"][k]["ll"]))])
    optr = {}
    for li in RL:
        row = {}
        for k in arms:
            pre = k.split("_")[0]
            if pre == "A":
                row[k] = round(float(np.mean([osh(m, k, li, 4) for m in M])), 3)
            elif pre in ("B", "Nc"):
                row[k] = round(float(np.mean([osh(m, k, li, 5) for m in M])), 3)
            elif pre == "C" and err:
                row[k] = round(float(np.mean([osh(m, k, li, word(m, "E5")) for m in err])), 3)
        for k in ("E5", "E2", "F5P1", "S2"):
            row[k] = dict(o5=round(float(np.mean([osh(m, k, li, 4) for m in M])), 3), o6=round(float(np.mean([osh(m, k, li, 5) for m in M])), 3),
                          own=round(float(np.mean([osh(m, k, li, word(m, k)) for m in M])), 3))
        optr[str(li)] = row
    out["option_reading"] = optr
    print(f"[{tag}] n {n} · checks {out['checks']} · I0 {i0} · A {out['A']} · error scenes {len(err)}")
    if "r_head" in out:
        print(f"   head gap {out['head_gap']} · r_head {out['r_head']}")
    print(f"   phi {phi}\n   g6 {g6}\n   rep {out['rep']}\n   d6 {d6} · Nc to literal 2 {out['Nc_to_literal2']}\n   amp {out['amp']}")
    for li, row in align.items():
        print(f"   L{li} " + " · ".join(f"{k} {v}" for k, v in row.items() if k.startswith("ARI") or k in ("E5", "E2", "F5P1", "S2", "S5") or k.startswith("AMP")))
    return out


if __name__ == "__main__":
    json.dump({sys.argv[1]: analyze(sys.argv[1])}, open("/vlm/results/r3/s771_analysis.json", "w"), indent=1)
