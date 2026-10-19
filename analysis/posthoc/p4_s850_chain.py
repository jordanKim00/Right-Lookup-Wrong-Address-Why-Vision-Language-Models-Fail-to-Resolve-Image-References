"""S850 descriptive follow-up (post hoc, NOT registered): where does the steering chain stop?  Late hold margin z[p] - max(z[p - 1], z[p + 1])
(boundary B4 and answer rows; matcher heads at 51 / 55, all heads at 59 / 63) change vs NOOP_B per late layer and arm, and P(answer = p) given
that the layer-51 address prefers p (hold > 0) per arm.
Usage: python3 exp/p4_s850_chain.py  ->  research/ces/outputs/s850_chain.json"""
import json
import numpy as np

V = "/home/storage2/ai26269345/vlm"
d = json.load(open(f"{V}/results/r3/s850_s850_27b_bstop.json")); R = d["rows"]; PS = (3, 4, 5)
def hold(r, arm, p, L):
    v = []
    for row in (5, 9):
        z = r["qk"][f"{arm}_p{p}"][L][row]; v.append(z[p - 1] - max(z[p - 2], z[p] if p < len(z) else -1e9))
    return float(np.mean(v))
out = {}
for arm in ("NOOP_B", "PLUS8", "PLUS16", "RAND8", "RAND16", "MINUS8", "FULLB", "S_EARLY"):
    dh = {L: float(np.mean([hold(r, arm, p, L) - hold(r, "NOOP_B", p, L) for r in R for p in PS])) for L in ("51", "55", "59", "63")}
    h = np.array([hold(r, arm, p, "51") for r in R for p in PS]); st = np.array([r["land"][f"{arm}_p{p}"] == p for r in R for p in PS])
    out[arm] = dict(dhold_vs_noop=dh, share_hold51_pos=float(np.mean(h > 0)), stay_given_hold51_pos=float(st[h > 0].mean()) if (h > 0).any() else None,
                    stay_given_hold51_nonpos=float(st[h <= 0].mean()))
json.dump(dict(note="post hoc, not registered", arms=out), open(f"{V}/research/ces/outputs/s850_chain.json", "w"), indent=1)
for a, v in out.items():
    print(a, {k: (round(x, 2) if isinstance(x, float) else {kk: round(xx, 2) for kk, xx in x.items()}) for k, x in v.items()})
