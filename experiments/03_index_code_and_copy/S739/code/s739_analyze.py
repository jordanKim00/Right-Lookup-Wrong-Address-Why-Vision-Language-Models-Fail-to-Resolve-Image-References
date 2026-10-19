"""S739 analysis, frozen with the pre-registration (knowledge/DECISIONS.md 2026-10-10 20:48) before any S739 data.
Per model and size regime (UNI: all 224 px, MIX: 224 / 448 / 672 px per image): accuracy (answer on the target j) for START, END, R3;
literal answers of END (answer on image k); matcher-head share on the target for START.
    BLOCK_CODE   = acc_START(MIX) >= acc_START(UNI) - .05 and tgt_START(MIX) >= tgt_START(UNI) - .05
    SAME_FAILURE = |acc_END(MIX) - acc_END(UNI)| <= .10 and |lit_END(MIX) - lit_END(UNI)| <= .10
    R3_UNEQUAL   = acc_R3(MIX) >= .90 and acc_R3(MIX) >= acc_R3(UNI) - .05
Reported: the same quantities per regime and the spread of block token counts in MIX.
Usage: python s739_analyze.py TAG [TAG ...]
"""
import json, os, sys
import numpy as np


def analyze(tag):
    d = json.load(open(f"/vlm/results/r3/s739_{tag}_tokens.json")); R = d["rows"]
    S = {}
    for reg in ("UNI", "MIX"):
        rs = [r for r in R if r["regime"] == reg]
        S[reg] = dict(acc_START=float(np.mean([r["land_START"] == r["j"] for r in rs])), acc_END=float(np.mean([r["land_END"] == r["j"] for r in rs])),
                      acc_R3=float(np.mean([r["land_R3"] == r["j"] for r in rs])), lit_END=float(np.mean([r["land_END"] == r["k"] for r in rs])),
                      tgt_START=float(np.mean([r["tgt_START"] for r in rs])), tgt_END=float(np.mean([r["tgt_END"] for r in rs])),
                      lit_share_END=float(np.mean([r["lit_END"] for r in rs])), tgt_R3=float(np.mean([r["tgt_R3"] for r in rs])), n=len(rs))
    toks = [t for r in R if r["regime"] == "MIX" for t in r["block_tokens"]]
    S["MIX_block_tokens"] = dict(min=int(min(toks)), median=float(np.median(toks)), max=int(max(toks))) if toks else None
    U, M = S["UNI"], S["MIX"]
    chk = {"BLOCK_CODE": M["acc_START"] >= U["acc_START"] - .05 and M["tgt_START"] >= U["tgt_START"] - .05,
           "SAME_FAILURE": abs(M["acc_END"] - U["acc_END"]) <= .10 and abs(M["lit_END"] - U["lit_END"]) <= .10,
           "R3_UNEQUAL": M["acc_R3"] >= .90 and M["acc_R3"] >= U["acc_R3"] - .05}
    for reg in ("UNI", "MIX"):
        x = S[reg]
        print(f"[{tag}] {reg}: START {x['acc_START']:.3f} (share {x['tgt_START']:.3f}) · END {x['acc_END']:.3f} (literal answers {x['lit_END']:.3f}) · R3 {x['acc_R3']:.3f}")
    print(f"[{tag}] MIX block tokens {S['MIX_block_tokens']} · checks {chk}")
    return dict(summary=S, checks=chk)


if __name__ == "__main__":
    P = "/vlm/results/r3/s739_analysis.json"
    out = json.load(open(P)) if os.path.exists(P) else {}
    out.update({t: analyze(t) for t in sys.argv[1:]})
    json.dump(out, open(P, "w"), indent=1)
