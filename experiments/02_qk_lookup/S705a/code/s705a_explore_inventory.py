"""Exploratory: per matcher head-row, how sharply does END(k) select the block d = k-1 steps before the visible end (a - d), pooled over
anchors a = 7, 6, 5 (m = 0, 1, 2), excluding cells where a - d equals the literal block k?  Held-out scenes.  Also the same head's START
share at the same blocks (its ceiling).  No model call."""
import json
import numpy as np
N = 7
def lse(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True); m = np.where(np.isfinite(m), m, 0.)
    return np.squeeze(m, axis) + np.log(np.sum(np.exp(x - m), axis=axis))
D = "/vlm/results/r3/s705a_qa_27b_qkanatomy"
meta = json.load(open(D + ".json")); z = np.load(D + ".npz"); M = meta["meta"]; ns = len(M)
hold = list(range(ns // 2, ns))
A = json.load(open("/vlm/results/r3/s705a_analysis.json"))["qa_27b"]["heads"]["matchers"]
def share(si, key, li, ri, vis):
    own = np.array(M[si]["owner"]); limg = z[f"{si}|{key}|{li}|limg"][ri].astype(np.float64)
    sc = np.stack([lse(limg[:, own == b]) for b in range(N)], -1)[:, :vis]
    return np.exp(sc - lse(sc)[:, None])
rows = []
for (li, h, ri) in A:
    r = {"head": f"L{li}h{h}r{ri}"}
    for d in range(0, 4):
        vals, sv = [], []
        for m in (0, 1, 2):
            a = N - m; k = d + 1; t = a - d
            if t < 1 or t == k: continue
            for si in hold:
                key = f"E{k}m{m}"
                if f"{si}|{key}|{li}|limg" not in z.files: continue
                vals.append(share(si, key, li, ri, a)[h, t - 1])
                if m == 0: sv.append(share(si, f"S{t}m0", li, ri, N)[h, t - 1])
        r[d] = float(np.mean(vals)) if vals else np.nan
        r[f"S{d}"] = float(np.mean(sv)) if sv else np.nan
    rows.append(r)
P = np.array([[r[d] for d in range(4)] for r in rows])
print("matcher head-rows", len(rows), "| chance per cell ~ 1/a = .14-.20")
print("mean share at a-d over heads:", " ".join(f"d{d} {np.nanmean(P[:, d]):.2f}" for d in range(4)))
for thr in (.4, .5, .6):
    print(f"heads with share >= {thr}:", " ".join(f"d{d} {int(np.nansum(P[:, d] >= thr))}" for d in range(4)))
best = np.nanargmax(P[:, 1:], 1) + 1
print("best offset among d>=1 per head:", {int(d): int((best == d).sum()) for d in (1, 2, 3)})
top2 = sorted(rows, key=lambda r: -r[2])[:8]
print("top heads at d=2:", [(r["head"], round(r[2], 2), round(r[1], 2)) for r in top2])
print("corr(d1, d2) over heads:", round(float(np.corrcoef(P[:, 1], P[:, 2])[0, 1]), 2))
