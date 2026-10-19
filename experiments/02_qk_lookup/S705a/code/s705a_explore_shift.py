"""Exploratory (not a registered test): does the matcher selection for END(k), k>=2, follow the moved anchor by m (offset model A)
or stay at its own place unless excluded (prior + exclusion model B)?  Saved S705a arrays, MATCHER heads fixed on discovery scenes,
read on held-out scenes.  No model call."""
import json, sys
import numpy as np
sys.path.insert(0, "/vlm/exp")
N = 7
def lse(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True); m = np.where(np.isfinite(m), m, 0.)
    return np.squeeze(m, axis) + np.log(np.sum(np.exp(x - m), axis=axis))
D = "/vlm/results/r3/s705a_qa_27b_qkanatomy"
meta = json.load(open(D + ".json")); z = np.load(D + ".npz"); M = meta["meta"]; ns = len(M)
hold = list(range(ns // 2, ns))
A = json.load(open("/vlm/results/r3/s705a_analysis.json"))["qa_27b"]["heads"]["matchers"]
def scores(si, key, li, ri):
    own = np.array(M[si]["owner"]); limg = z[f"{si}|{key}|{li}|limg"][ri].astype(np.float64)
    return np.stack([lse(limg[:, own == b]) for b in range(N)], -1)          # [H, N]
print("matcher head-rows:", len(A))
print("cond   | A-pred B-pred | argmax histogram over visible blocks (share of head x scene) | behaviour landing")
for k in range(1, 7):
    for m in (0, 1, 2):
        if k > N - m: continue
        key = f"E{k}m{m}"; hist = np.zeros(N); n = 0; anc = []
        for (li, h, ri) in A:
            for si in hold:
                if f"{si}|{key}|{li}|limg" not in z.files: continue
                sc = scores(si, key, li, ri)[h, :N - m]
                hist[int(np.argmax(sc))] += 1; n += 1
                a = N - m
                if a >= 2: anc.append(sc[a - 1] - sc[a - 2])        # score(anchor) - score(anchor-1)
        if not n: continue
        land = [M[si]["land"].get(key) for si in hold]
        lh = np.bincount([l for l in land if l], minlength=N + 1)[1:] / max(1, len([l for l in land if l]))
        Apred = N - m + 1 - k; Bpred = N + 1 - k if N + 1 - k <= N - m and N + 1 - k != N - m else "excl"
        print(f"E{k}m{m} | A {Apred} B {Bpred} | " + " ".join(f"{x:.2f}" for x in hist[:N - m] / n) +
              f" | anc-(anc-1) {np.mean(anc):+.2f} | land " + " ".join(f"{x:.2f}" for x in lh[:N - m]))
