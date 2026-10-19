"""Exploratory (saved S705a arrays, no model call): layer by layer, how sharply do the START-only matcher head-rows of each captured layer
select the target for END(2) (image 6) and END(5) (image 3), against START(6) and START(3) asking for the same images?  All 30 scenes."""
import json
import numpy as np
N = 7
def lse(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True); m = np.where(np.isfinite(m), m, 0.)
    return np.squeeze(m, axis) + np.log(np.sum(np.exp(x - m), axis=axis))
D = "/vlm/results/r3/s705a_qa_27b_qkanatomy"
meta = json.load(open(D + ".json")); z = np.load(D + ".npz"); M = meta["meta"]; CAP = meta["CAP"]; ns = len(M)
def share(si, key, li):
    own = np.array(M[si]["owner"]); limg = z[f"{si}|{key}|{li}|limg"].astype(np.float64)
    sc = np.stack([lse(limg[:, :, own == b]) for b in range(N)], -1); return np.exp(sc - lse(sc)[..., None])   # [rows, H, N]
disc = range(ns // 2)
print("layer | START-only head-rows | target share: START(6) END(2) | START(3) END(5) | END(5) on literal 5 / neighbours 2,4")
for li in CAP:
    sd = np.mean([share(si, f"S{t}m0", li)[:, :, t - 1] for si in disc for t in range(1, N + 1)], 0)       # [rows, H] discovery diag
    sel = [(ri, h) for ri in (1, 2) for h in range(sd.shape[1]) if sd[ri, h] >= .5]
    if not sel: print(li, "none"); continue
    g = lambda key, b: np.mean([share(si, key, li)[ri, h, b - 1] for si in range(ns) for (ri, h) in sel])
    print(f"L{li} | {len(sel):3d} | {g('S6m0', 6):.2f} {g('E2m0', 6):.2f} | {g('S3m0', 3):.2f} {g('E5m0', 3):.2f} | {g('E5m0', 5):.2f} / {g('E5m0', 2) + g('E5m0', 4):.2f}")
