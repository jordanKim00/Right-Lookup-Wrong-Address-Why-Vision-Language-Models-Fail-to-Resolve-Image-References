"""S845 descriptive re-analysis (post hoc, NOT registered).  Per arm, pooled over the four cases: answer = k share, and the positional margin
z[k] - max(other blocks) of the boundary B4 and answer rows at layers 47, 51 and 55 (matcher-head block scores).  BOTH_B carries A_BAND's late
queries at every read row, so its layer 51 / 55 scores equal A_BAND's by construction; a gap between their answers is the part of the band's
correction that does not pass through the layer 51 / 55 query-key scores.  Also: answer = k given a positive vs non-positive margin (all arms).
Usage: python3 exp/p4_s845_scores.py  ->  research/ces/outputs/s845_scores.json"""
from __future__ import annotations

import json
import numpy as np

V = "/home/storage2/ai26269345/vlm"
ARMS = ["START", "END", "QNOOP", "A_BAND", "A_4749", "ADDR_B", "READ_B", "BOTH_B", "READT_B", "ADDR_47", "READ_47", "BOTH_47"]


def main():
    d = json.load(open(f"{V}/results/r3/s845_s845_27b_wr.json")); R = d["rows"]
    def zm(r, a, L, row):
        z = r["qk"][a][L][row]; k = r["k"]
        return z[k - 1] - max(z[i] for i in range(len(z)) if i != k - 1)
    out = dict(note="post hoc, not registered", answer_k={a: float(np.mean([r["land"][a] == r["k"] for r in R])) for a in ARMS}, margin={}, link={})
    for L in ("47", "51", "55"):
        for row, nm in ((5, "B4"), (9, "answer")):
            out["margin"][f"{L}_{nm}"] = {a: float(np.mean([zm(r, a, L, row) for r in R])) for a in ARMS}
    for L in ("51", "55"):
        for row, nm in ((5, "B4"), (9, "answer")):
            x = np.array([zm(r, a, L, row) for r in R for a in ARMS]); y = np.array([r["land"][a] == r["k"] for r in R for a in ARMS])
            out["link"][f"{L}_{nm}"] = dict(pos=float(y[x > 0].mean()), nonpos=float(y[x <= 0].mean()), n_pos=int((x > 0).sum()), r=float(np.corrcoef(x, y)[0, 1]))
    json.dump(out, open(f"{V}/research/ces/outputs/s845_scores.json", "w"), indent=1)
    print("answer=k", {a: round(v, 3) for a, v in out["answer_k"].items()})
    for k_, v in out["margin"].items():
        print("margin", k_, {a: round(x, 2) for a, x in v.items()})
    print("link", out["link"])


if __name__ == "__main__":
    main()
