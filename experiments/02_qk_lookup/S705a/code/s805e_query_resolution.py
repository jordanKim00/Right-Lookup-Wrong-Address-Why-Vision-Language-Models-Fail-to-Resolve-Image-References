"""S805e (descriptive re-analysis of saved S705a captures, no model call, not pre-registered): how much does the answer-row query of each
index-matching head resolve which item is referenced, for start- and end-relative questions?  Qwen3.5-27B, N = 7, 30 scenes, answer row.
Features: the head's query in its non-rotary dimensions (64..255; the rotary part depends on the absolute position, which differs across scenes).
  acc7   leave-one-scene-out nearest-centroid accuracy over the 7 questions of a frame (START(1..7) or END(1..7)), mean over heads
  d'     neighbour discriminability: for START(t) vs START(t+1) and END(k) vs END(k+1) (whose targets also differ by one), the direction of the
         class-mean difference fitted without the held-out scene, the held-out projections pooled, d' = mean difference / pooled s.d.
  END->START  each END(k) query classified by the START centroids (fitted on the other scenes): share assigned to the target t = N + 1 - k,
         to the literal t = k, and the mean |assigned - target|
Usage: python3 exp/s805e_query_resolution.py -> research/ces/outputs/s805e_query_resolution.json"""
import json, collections
import numpy as np

ROOT = "/home/storage2/ai26269345/vlm"
J = json.load(open(f"{ROOT}/results/r3/s705a_qa_27b_qkanatomy.json"))
Z = np.load(f"{ROOT}/results/r3/s705a_qa_27b_qkanatomy.npz")
MH = collections.defaultdict(list)
for l, h in json.load(open(f"{ROOT}/exp/s786_heads/27b.json"))["matchers"]:
    MH[int(l)].append(int(h))
N, S, RD = 7, len(J["meta"]), 64
out = {}
for L in (39, 43, 47, 51, 55):
    Q = {fr: np.stack([np.stack([Z[f"{si}|{fr}{t}m0|{L}|q"][2][MH[L], RD:].astype(np.float64) for t in range(1, N + 1)]) for si in range(S)])
         for fr in ("S", "E")}                                                       # [scene, question, head, dim]
    res = {}
    for fr in ("S", "E"):
        X = Q[fr]; acc = []; dp = {}
        for h in range(X.shape[2]):
            hit = 0
            for s in range(S):
                C = np.delete(X[:, :, h], s, 0).mean(0)                            # [7, dim] centroids without scene s
                for t in range(N):
                    hit += int(np.argmin(((C - X[s, t, h]) ** 2).sum(-1)) == t)
            acc.append(hit / (S * N))
        for t in range(N - 1):
            proj = {0: [], 1: []}
            for h in range(X.shape[2]):
                for s in range(S):
                    tr = np.delete(X[:, :, h], s, 0); w = tr[:, t + 1].mean(0) - tr[:, t].mean(0); w /= np.linalg.norm(w) + 1e-12
                    proj[0].append(X[s, t, h] @ w); proj[1].append(X[s, t + 1, h] @ w)
            a, b = np.array(proj[0]), np.array(proj[1])
            dp[f"{t + 1}-{t + 2}"] = float((b.mean() - a.mean()) / np.sqrt(.5 * (a.var() + b.var())))
        res[fr] = dict(acc7=float(np.mean(acc)), dprime=dp)
    # END queries classified by START centroids
    asg = collections.defaultdict(list)
    for h in range(Q["S"].shape[2]):
        for s in range(S):
            C = np.delete(Q["S"][:, :, h], s, 0).mean(0)
            for k in range(1, N + 1):
                asg[k].append(int(np.argmin(((C - Q["E"][s, k - 1, h]) ** 2).sum(-1))) + 1)
    res["END_by_START"] = {str(k): dict(target=float(np.mean(np.array(v) == N + 1 - k)), literal=float(np.mean(np.array(v) == k)),
                                        dist=float(np.mean(np.abs(np.array(v) - (N + 1 - k))))) for k, v in asg.items()}
    out[f"L{L}"] = res
    print(f"== L{L} ({len(MH[L])} heads): acc7 START {res['S']['acc7']:.2f} END {res['E']['acc7']:.2f}")
    print("   d' START (t, t+1): " + " ".join(f"{p} {v:.1f}" for p, v in res["S"]["dprime"].items()))
    print("   d' END   (k, k+1): " + " ".join(f"{p} {v:.1f}" for p, v in res["E"]["dprime"].items()))
    print("   END(k) by START centroids, on target / literal / mean |assigned - target|: " +
          " | ".join(f"k{k}: {v['target']:.2f}/{v['literal']:.2f}/{v['dist']:.1f}" for k, v in res["END_by_START"].items()))
json.dump(out, open(f"{ROOT}/research/ces/outputs/s805e_query_resolution.json", "w"), indent=1)
