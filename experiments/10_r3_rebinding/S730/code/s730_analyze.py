"""S730 analysis, frozen with the pre-registration (knowledge/DECISIONS.md, 2026-10-10) before any S730 data.
Per model, pooled over scenes and k in {2, 3, 5, 6} (k = 4 is the middle cell, where R³ writes the frame only; reported separately):
    acc[c]  = P(answer lands on image j)        tgt[c] / lit[c] = matcher heads' mean block share at j / at k (answer row)
    cos[c]  = mean cosine of the matcher heads' queries (after q_norm, before rotary) with the same scene's START(j) queries
Rules:
    QUERY_RESTORED[m]:    cos[R3] - cos[END] >= .10  and  tgt[R3] >= .8 tgt[START]
    BEHAVIOR_RESTORED[m]: acc[R3] >= .9 acc[START]
    TWO_WRITES[m]:        acc[R3] >= max(acc[ADDR], acc[FRAME]) + .10
    WRITE_LAYER_OK[m]:    (S731, when R3W exists) acc[R3W] >= acc[R3] - .03 and cos[R3W] >= cos[R3] - .03
Usage: python s730_analyze.py TAG [TAG ...]
"""
import json, sys
import numpy as np


def summarize(tag):
    d = json.load(open(f"/vlm/results/r3/s730_{tag}_stagewise.json")); R = d["rows"]
    conds = [c for c in ("START", "END", "R3", "ADDR", "FRAME", "R3W") if f"land_{c}" in R[0]]
    out = {}
    for sub, rows in (("k!=4", [r for r in R if r["k"] != 4]), ("k=4", [r for r in R if r["k"] == 4])):
        out[sub] = {c: dict(acc=float(np.mean([r[f"land_{c}"] == r["j"] for r in rows])), tgt=float(np.mean([r[f"tgt_{c}"] for r in rows])),
                            lit=float(np.mean([r[f"lit_{c}"] for r in rows])), cos=float(np.mean([r[f"cos_{c}"] for r in rows])),
                            lit_ans=float(np.mean([r[f"land_{c}"] == r["k"] for r in rows]))) for c in conds}
    perk = {int(k): {c: float(np.mean([r[f"land_{c}"] == r["j"] for r in R if r["k"] == k])) for c in conds} for k in sorted({r["k"] for r in R})}
    S = out["k!=4"]
    chk = dict(QUERY_RESTORED=S["R3"]["cos"] - S["END"]["cos"] >= .10 and S["R3"]["tgt"] >= .8 * S["START"]["tgt"],
               BEHAVIOR_RESTORED=S["R3"]["acc"] >= .9 * S["START"]["acc"],
               TWO_WRITES=S["R3"]["acc"] >= max(S["ADDR"]["acc"], S["FRAME"]["acc"]) + .10)
    if "R3W" in S:
        chk["WRITE_LAYER_OK"] = S["R3W"]["acc"] >= S["R3"]["acc"] - .03 and S["R3W"]["cos"] >= S["R3"]["cos"] - .03
    print(f"[{tag}] (k≠4) " + " | ".join(f"{c}: 정답 {S[c]['acc']:.2f} · 문자답 {S[c]['lit_ans']:.2f} · 목표몫 {S[c]['tgt']:.2f} · 문자몫 {S[c]['lit']:.2f} · cos(START) {S[c]['cos']:.3f}" for c in conds))
    K = out["k=4"]
    print(f"[{tag}] (k=4, 틀만 쓰는 칸) " + " · ".join(f"{c} {K[c]['acc']:.2f}" for c in conds))
    print(f"[{tag}] k 별 정답: " + " · ".join(f"k{k}: " + "/".join(f"{perk[k][c]:.2f}" for c in conds) for k in perk) + f"  ({'/'.join(conds)})")
    print(f"[{tag}] 판정 {chk}")
    return dict(summary=out, perk=perk, checks=chk, n=len(R), heads=len(d["heads"]), choice=d.get("choice"))


if __name__ == "__main__":
    import os
    P = "/vlm/results/r3/s730_analysis.json"
    allr = json.load(open(P)) if os.path.exists(P) else {}
    for t in sys.argv[1:]:
        allr[t] = summarize(t)
    json.dump(allr, open(P, "w"), indent=1)
