"""S750 exploratory analysis (planned before any S750 data, DECISIONS 2026-10-12 22:5x; not a registered verdict).
For a donor END(k') split by whether its natural answer was right or wrong, how often does each (layer, token group) residual transplant
make the recipient answer the donor's ACTUAL choice (C), against the donor's target (F) and the recipient staying on its own target (R)?
Baseline: the recipient's natural answer equals the donor's choice.  Pairs: (2 <- 5) main, (5 <- 4) and (5 <- 2) for contrast.
Usage: python s750_explore_donorchoice.py TAG"""
import json, sys
import numpy as np
N = 7
tag = sys.argv[1]
d = json.load(open(f"/vlm/results/r3/s750_{tag}_where.json")); M = d["meta"]; RES = sorted(d["RES"]); G = d["GROUPS"]
tgt = lambda k: N + 1 - k
for (k, kp) in ((2, 5), (5, 4), (5, 2)):
    dc = np.array([m["land"][f"E{kp}"] for m in M]); rn = np.array([m["land"][f"E{k}"] for m in M])
    for split, mask in (("donor right", dc == tgt(kp)), ("donor wrong", dc != tgt(kp))):
        idx = np.where(mask)[0]
        if not len(idx):
            continue
        wrong_imgs = {int(b): int((dc[idx] == b).sum()) for b in sorted(set(dc[idx].tolist()))}
        base = float(np.mean(rn[idx] == dc[idx]))
        print(f"== {k}<-{kp} · {split}: {len(idx)} scenes · donor choices {wrong_imgs} · baseline (recipient natural = donor choice) {base:.2f}")
        for g in G:
            row = []
            for L in RES:
                a = np.array([M[i]["land"][f"R{L}|{g}|{k}<{kp}"] for i in idx])
                row.append(f"L{L} C{np.mean(a == dc[idx]):.2f}/F{np.mean(a == tgt(kp)):.2f}/R{np.mean(a == tgt(k)):.2f}")
            print(f"   {g:4s} " + "  ".join(row))
