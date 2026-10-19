"""S805g (descriptive re-analysis of saved S705a captures, no model call, not pre-registered): where does the decodable ordinal identity of the
answer-row query live - inside the address subspace U (span of the centred block-mean image keys of the head, per scene) or outside it?
Qwen3.5-27B, N = 7, 30 scenes (every scene has the same token positions, so all 256 post-RoPE dims compare across scenes), index-matching heads.
  acc7(part)  leave-one-scene-out nearest-centroid accuracy over the 7 questions of a frame, from the full query, from P_U q, from (I - P_U) q;
              mean over heads (chance 1/7)
  norm share  |P_U q|^2 / |q|^2 (mean)
  between     share of the across-question variance (centroid spread) that lies in U vs outside it (pooled over heads)
Usage: python3 exp/s805g_decode_split.py -> research/ces/outputs/s805g_decode_split.json"""
import json, collections
import numpy as np

ROOT = "/home/storage2/ai26269345/vlm"
J = json.load(open(f"{ROOT}/results/r3/s705a_qa_27b_qkanatomy.json"))
Z = np.load(f"{ROOT}/results/r3/s705a_qa_27b_qkanatomy.npz")
MH = collections.defaultdict(list)
for l, h in json.load(open(f"{ROOT}/exp/s786_heads/27b.json"))["matchers"]:
    MH[int(l)].append(int(h))
N, S = 7, len(J["meta"])


def ubasis(Kh, own):
    Kb = np.array([Kh[own == b].mean(0) for b in range(N)]); C = Kb - Kb.mean(0)
    u, s, vt = np.linalg.svd(C, full_matrices=False)
    return vt[s > 1e-6 * s.max()]


def loo(X):                                                       # X [scene, question, dim] -> nearest-centroid accuracy
    hit = 0
    for s in range(X.shape[0]):
        C = np.delete(X, s, 0).mean(0)
        for t in range(N):
            hit += int(np.argmin(((C - X[s, t]) ** 2).sum(-1)) == t)
    return hit / (X.shape[0] * N)


out = {}
for L in (39, 43, 47, 51, 55):
    res = {}
    for fr in ("S", "E"):
        acc = collections.defaultdict(list); ns = []; btw = collections.defaultdict(float)
        for h in MH[L]:
            P = {"full": np.zeros((S, N, 256)), "U": np.zeros((S, N, 256)), "perp": np.zeros((S, N, 256))}
            for si in range(S):
                own = np.array(J["meta"][si]["owner"]); Kh = Z[f"{si}|img|{L}|k"].astype(np.float64)[:, h // 6, :]; B = ubasis(Kh, own)
                for t in range(1, N + 1):
                    q = Z[f"{si}|{fr}{t}m0|{L}|q"][2, h].astype(np.float64); qu = B.T @ (B @ q)
                    P["full"][si, t - 1], P["U"][si, t - 1], P["perp"][si, t - 1] = q, qu, q - qu
                    ns.append(float(qu @ qu / (q @ q)))
            for part, X in P.items():
                acc[part].append(loo(X))
            for part in ("U", "perp"):
                m = P[part].mean(0); btw[part] += float(((m - m.mean(0)) ** 2).sum())
        tot = btw["U"] + btw["perp"]
        res[fr] = dict(acc7={p: float(np.mean(v)) for p, v in acc.items()}, norm_share_U=float(np.mean(ns)),
                       between_share_U=btw["U"] / tot)
    out[f"L{L}"] = res
    print(f"== L{L} ({len(MH[L])} heads)")
    for fr in ("S", "E"):
        r = res[fr]
        print(f"   {'START' if fr == 'S' else 'END  '} acc7 full {r['acc7']['full']:.2f} U {r['acc7']['U']:.2f} perp {r['acc7']['perp']:.2f} | "
              f"|P_U q|^2 share {r['norm_share_U']:.2f} | across-question spread in U {r['between_share_U']:.2f}")
json.dump(out, open(f"{ROOT}/research/ces/outputs/s805g_decode_split.json", "w"), indent=1)
