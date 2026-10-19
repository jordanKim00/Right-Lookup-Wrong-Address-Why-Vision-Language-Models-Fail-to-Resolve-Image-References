"""S782 analysis, frozen with the pre-registration (knowledge/DECISIONS.md) before any S782 data.
AGREE      per cell and question (END(5), START(N - 4), END(2), R3), the share of scenes where the first generated word equals the menu-restricted answer
           >= .90 (and pooled over questions >= .95).
IN_MENU    share of free answers that are menu words >= .95.
Descriptive: disagreement examples; accuracy under each read.
Usage: python s782_analyze.py TAG [TAG ...]
"""
import json, sys
import numpy as np


def cell(tag):
    d = json.load(open(f"/vlm/results/r3/s782_{tag}_freegen.json")); M = d["meta"]; N, ta = d["N"], d["ta"]
    tgt = {"E5": ta, "S": ta, "E2": N - 1, "R3": ta}
    out = dict(tag=tag, N=N, n=len(M), agree={}, in_menu={}, acc={}, checks={}, examples=[])
    for q in ("E5", "S", "E2", "R3"):
        ag = [m["out"][q]["free"] == m["out"][q]["restricted"].lower() for m in M]
        im = [m["out"][q]["free"] in [w.lower() for w in m["menu"]] for m in M]
        out["agree"][q] = round(float(np.mean(ag)), 3); out["in_menu"][q] = round(float(np.mean(im)), 3)
        word = lambda m: m["labs"][tgt[q] - 1].lower()
        out["acc"][q] = dict(restricted=round(float(np.mean([m["out"][q]["restricted"].lower() == word(m) for m in M])), 3),
                             free=round(float(np.mean([m["out"][q]["free"] == word(m) for m in M])), 3))
        out["examples"] += [dict(q=q, restricted=m["out"][q]["restricted"], text=m["out"][q]["text"]) for m, g in zip(M, ag) if not g][:3]
        out["checks"][f"AGREE_{q}"] = bool(out["agree"][q] >= .90 - 1e-9)
    pooled = float(np.mean([m["out"][q]["free"] == m["out"][q]["restricted"].lower() for m in M for q in ("E5", "S", "E2", "R3")]))
    out["agree_pooled"] = round(pooled, 3); out["checks"]["AGREE_POOLED"] = bool(pooled >= .95 - 1e-9)
    out["checks"]["IN_MENU"] = bool(min(out["in_menu"].values()) >= .95 - 1e-9)
    print(f"[{tag}] N {N} · checks {out['checks']} · agree {out['agree']} · pooled {pooled:.3f} · in menu {out['in_menu']} · acc {out['acc']}")
    for e in out["examples"][:6]:
        print("   disagreement:", e)
    return out


if __name__ == "__main__":
    json.dump({t: cell(t) for t in sys.argv[1:]}, open("/vlm/results/r3/s782_analysis.json", "w"), indent=1)
