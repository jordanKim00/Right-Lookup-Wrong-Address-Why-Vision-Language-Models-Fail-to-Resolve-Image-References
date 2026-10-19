"""S766 analysis (replication of the S759 frame-read -> early lookup chain on new scenes), frozen with the pre-registration
(knowledge/DECISIONS.md, 2026-10-13 19:0x) before any S766 data.  Data: exp/s759_frame.py run with --seed 76600 (tag s766_27b).
share_L(arm, b) = mean over scenes of the all-heads mean (rows tail1 / answer) of the softmax block share on image b at layer L.
CHAIN_F5   share_39(F5:P1, 5) >= share_39(E5, 5) + .20 and >= .45;  share_39(F5:P23, 5) <= share_39(E5, 5) + .05;  P(answer 5 | F5:P1) >= P(answer 5 | E5) + .30.
CONV       share_39(F5:P1conv, 5) >= share_39(E5, 5) + .15 and share_39(F5:P1fa, 5) <= share_39(E5, 5) + .10.
CHAIN_F2   share_39(F2:P1, 2) >= share_39(E2, 2) + .15 and share_39(F2:P23, 2) <= share_39(E2, 2) + .05.
Also re-reads the S759 registered verdicts with exp/s759_analyze.py on the new data (reported, not re-judged here).
Usage: python s766_analyze.py TAG
"""
import json, sys
import numpy as np


def sm(sc):
    x = sc - sc.max(-1, keepdims=True); e = np.exp(x); return e / e.sum(-1, keepdims=True)


def analyze(tag):
    D = f"/vlm/results/r3/s759_{tag}_frame"
    d = json.load(open(D + ".json")); M = d["meta"]; z = np.load(D + ".npz")
    sh = lambda arm, b, L=39: float(np.mean([sm(z[f"{m['si']}|{arm}|{L}"].astype(np.float64))[1:3].mean(0).mean(0)[b - 1] for m in M]))
    ans = lambda arm, b: float(np.mean([m["land"][arm] == b for m in M]))
    v = {a: round(sh(a, 5), 3) for a in ("E5", "S5", "F5:P1", "F5:P23", "F5:P1conv", "F5:P1fa", "F5:P", "F5:P2", "F5:P3")}
    v2 = {a: round(sh(a, 2), 3) for a in ("E2", "S2", "F2:P1", "F2:P23", "F2:P12")}
    a5 = {a: round(ans(a, 5), 3) for a in ("E5", "S5", "F5:P1", "F5:P23", "F5:P1conv", "F5:P1fa")}
    out = dict(n=len(M), checks={}, share5_L39=v, share2_L39=v2, answer5=a5)
    out["checks"]["CHAIN_F5"] = bool(v["F5:P1"] >= v["E5"] + .20 - 1e-9 and v["F5:P1"] >= .45 - 1e-9 and v["F5:P23"] <= v["E5"] + .05 + 1e-9
                                     and a5["F5:P1"] >= a5["E5"] + .30 - 1e-9)
    out["checks"]["CONV"] = bool(v["F5:P1conv"] >= v["E5"] + .15 - 1e-9 and v["F5:P1fa"] <= v["E5"] + .10 + 1e-9)
    out["checks"]["CHAIN_F2"] = bool(v2["F2:P1"] >= v2["E2"] + .15 - 1e-9 and v2["F2:P23"] <= v2["E2"] + .05 + 1e-9)
    print(f"[{tag}] n {out['n']} · checks {out['checks']}")
    print(f"   block 5 share L39 {v}\n   answer 5 {a5}\n   block 2 share L39 {v2}")
    return out


if __name__ == "__main__":
    json.dump({sys.argv[1]: analyze(sys.argv[1])}, open("/vlm/results/r3/s766_analysis.json", "w"), indent=1)
