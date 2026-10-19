"""Descriptive (not registered; 10-16 20:4x, stage-1 data only): which heads carry END's change of the boundary newline rows' (B1, B4) block
scores in the S827 identity cells, and do they write?  Per look layer and head: local coefficients of z_END - z_START relative to the far blocks
(a_(k-1), a_k, a_(k+1)), their scene mean / standard error, the gated output norm of the head at B1 / B4 (END, START), its picture part, and the
mean output gate.  A head is a 'dip head' if a_k < 0 beyond 3 SE with |a_k| > |a_(k-1)|, |a_(k+1)|; a 'prev head' / 'next head' if a_(k-1) /
a_(k+1) > 0 beyond 3 SE and is the largest of the three.
Usage: python exp/s827_heads_desc.py TAG  ->  research/ces/outputs/s827_heads_desc_<TAG>.json"""
from __future__ import annotations

import json, sys
import numpy as np


def main(tag):
    V = "/home/storage2/ai26269345/vlm"
    d = json.load(open(f"{V}/results/r3/s827_{tag}_natural.json")); Z = np.load(f"{V}/results/r3/s827_{tag}_natural.npz")
    look = d["look"]; MH = {int(l): h for l, h in d["matchers"].items()}
    out = {}
    for N, k in ((7, 4), (9, 5)):
        rows = [r for r in d["rows"] if r["N"] == N]
        far = [i for i in range(N) if i not in (k - 2, k - 1, k)]
        res = {}
        for li, L in enumerate(look):
            capE = np.stack([Z[f"N{N}_s{r['si']}_END_cap"][li].astype(np.float32)[[2, 5]].mean(0) for r in rows])     # [S, H, nb]
            capS = np.stack([Z[f"N{N}_s{r['si']}_START_cap"][li].astype(np.float32)[[2, 5]].mean(0) for r in rows])
            hoE = np.stack([Z[f"N{N}_s{r['si']}_END_hout"][li].astype(np.float32)[:, [2, 5]].mean(1) for r in rows])   # [S, 3, H]
            hoS = np.stack([Z[f"N{N}_s{r['si']}_START_hout"][li].astype(np.float32)[:, [2, 5]].mean(1) for r in rows])
            De = capE - capS
            A = np.stack([De[..., x - 1] - De[..., far].mean(-1) for x in (k - 1, k, k + 1)], -1)                    # [S, H, 3]
            mu, se = A.mean(0), A.std(0, ddof=1) / np.sqrt(len(rows))
            H = A.shape[1]; lay = {}
            for h in range(H):
                m, s_ = mu[h], se[h]
                kind = None
                if m[1] < -3 * s_[1] and abs(m[1]) >= max(abs(m[0]), abs(m[2])):
                    kind = "dip"
                elif m[0] > 3 * s_[0] and m[0] >= max(abs(m[1]), m[2]):
                    kind = "prev"
                elif m[2] > 3 * s_[2] and m[2] >= max(abs(m[1]), m[0]):
                    kind = "next"
                lay[str(h)] = dict(a=m.tolist(), se=s_.tolist(), kind=kind, matcher=h in MH.get(L, []),
                                   out_E=float(hoE[:, 0, h].mean()), out_S=float(hoS[:, 0, h].mean()), img_E=float(hoE[:, 1, h].mean()),
                                   img_S=float(hoS[:, 1, h].mean()), gate_E=float(hoE[:, 2, h].mean()), gate_S=float(hoS[:, 2, h].mean()))
            res[str(L)] = lay
        out[f"N{N}k{k}"] = res
        print(f"== N{N} k{k}")
        for L, lay in res.items():
            kinds = {kk: [h for h, x in lay.items() if x["kind"] == kk] for kk in ("dip", "prev", "next")}
            def wsum(hs, i):
                return sum(lay[h]["a"][i] * lay[h]["img_E"] for h in hs)
            tot_img = sum(x["img_E"] for x in lay.values())
            print(f"  L{L}: dip {kinds['dip']} prev {kinds['prev']} next {kinds['next']} | matcher {[h for h, x in lay.items() if x['matcher']]}")
            for kk in ("dip", "prev", "next"):
                for h in kinds[kk]:
                    x = lay[h]
                    print(f"     {kk:4s} h{h:>2s} a(k-1,k,k+1) {np.round(x['a'], 2).tolist()} img out E {x['img_E']:.2f} S {x['img_S']:.2f} "
                          f"(share of layer {x['img_E'] / tot_img:.2f}) gate E {x['gate_E']:.2f} matcher {x['matcher']}")
    json.dump(out, open(f"{V}/research/ces/outputs/s827_heads_desc_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
