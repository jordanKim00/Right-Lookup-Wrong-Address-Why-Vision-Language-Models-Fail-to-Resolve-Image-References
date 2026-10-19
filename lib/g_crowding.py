"""agent_math_1015 / g_crowding (descriptive CPU re-analysis of S705a 27B QK captures; answer-free).  Second task G: address crowding.
In the image-key-compatible space of each matcher head (U = span of the 7 centred block-mean post-RoPE image keys of its KV head, per scene and
look layer), the START(t) answer-row query's U-part A_t is the address of picture t.  Per t: the minimal address gap d_t = min_{t' != t}
|A_t - A_t'| / mean_t |A_t| (scale-free), and the logit-level gap of START(t) itself, g_t = s_t(t) - max_{b != t} s_t(b) with s(b) = q . kbar_b / 16
(block-mean-key logits).  Per END(k): the logit gap of its own query to the target, s(target) - max_{b != target} s(b), its nearest START address
in U (by distance) and the gap between the nearest and second-nearest address.  Neighbour asymmetry for the copy: |A_k - A_{k-1}| vs
|A_k - A_{k+1}| (does the literal k's address sit closer to k-1?).  Per scene: correlation of the target's crowding d_j with END(k)'s own
lookup target share (answer-free; S705a answers are contaminated and not used).
Output: research/ces/outputs/agent_math_1015/g_crowding.json"""
import json
import numpy as np

O = "/home/storage2/ai26269345/vlm/research/ces/outputs/agent_math_1015"
C = np.load(f"{O}/s705a_cache.npz"); share = C["share"]; q = C["q"]; kbar = C["kbar"]; runs = list(C["runs"]); cap = [int(x) for x in C["cap"]]
mat = json.load(open("/home/storage2/ai26269345/vlm/results/r3/s705a_qa_27b_matchers.json"))["matchers"]
MH = {L: [h for (l, h) in mat if l == L] for L in cap}
NS = q.shape[0]; NB = 7
S_ = lambda t: runs.index(f"S{t}m0"); E_ = lambda k: runs.index(f"E{k}m0")
out = {}
for L in (43, 47, 51, 55):
    li = cap.index(L); H = MH[L]
    d = np.zeros((NS, len(H), NB)); gS = np.zeros((NS, len(H), NB)); asym = {k: [] for k in (2, 3, 4, 5, 6)}
    gE = np.zeros((NS, len(H), NB)); nearestE = np.zeros((NS, len(H), NB), int); sepE = np.zeros((NS, len(H), NB))
    for si in range(NS):
        for hi, h in enumerate(H):
            Kb = kbar[si, li, h // 6]                          # (7, 256) block-mean keys
            Kc = Kb - Kb.mean(0)
            Uq, Us, _ = np.linalg.svd(Kc.T, full_matrices=False)  # columns span U (rank <= 6)
            Ub = Uq[:, Us > 1e-6 * Us.max()]
            A = np.array([Ub @ (Ub.T @ q[si, S_(t), li, h]) for t in range(1, NB + 1)])   # START addresses in U
            sc = np.linalg.norm(A, axis=1).mean()
            D = np.linalg.norm(A[:, None] - A[None], axis=2) / sc
            for t in range(NB):
                d[si, hi, t] = np.min(np.delete(D[t], t))
                s = Kb @ q[si, S_(t + 1), li, h] / 16.0
                gS[si, hi, t] = s[t] - np.max(np.delete(s, t))
            for k in (2, 3, 4, 5, 6):
                asym[k].append((D[k - 1, k - 2], D[k - 1, k]))     # |A_k - A_{k-1}|, |A_k - A_{k+1}|
            for k in range(1, NB + 1):
                tq = q[si, E_(k), li, h]; s = Kb @ tq / 16.0; tgt = 8 - k
                gE[si, hi, k - 1] = s[tgt - 1] - np.max(np.delete(s, tgt - 1))
                aE = Ub @ (Ub.T @ tq); dist = np.linalg.norm(A - aE, axis=1) / sc; o = np.argsort(dist)
                nearestE[si, hi, k - 1] = o[0] + 1; sepE[si, hi, k - 1] = dist[o[1]] - dist[o[0]]
    res = dict(start_gap_d=d.mean((0, 1)).round(3).tolist(), start_logit_gap=gS.mean((0, 1)).round(3).tolist(),
               end_logit_gap_to_target=gE.mean((0, 1)).round(3).tolist(), end_nearest_is_target=[float((nearestE[:, :, k - 1] == 8 - k).mean()) for k in range(1, NB + 1)],
               end_nearest_is_km1=[float((nearestE[:, :, k - 1] == k - 1).mean()) if k > 1 else None for k in range(1, NB + 1)],
               end_nearest_is_literal=[float((nearestE[:, :, k - 1] == k).mean()) for k in range(1, NB + 1)],
               end_nearest_sep=sepE.mean((0, 1)).round(3).tolist(),
               copy_neighbour_asym={k: dict(to_km1=float(np.mean([a for a, b in asym[k]])), to_kp1=float(np.mean([b for a, b in asym[k]])),
                                            frac_km1_closer=float(np.mean([a < b for a, b in asym[k]]))) for k in asym})
    # per-scene: target crowding vs END lookup target share (answer-free)
    pers = {}
    for k in (3, 4, 5, 6):
        tgt = 8 - k
        dj = d[:, :, tgt - 1].mean(1)                       # scene's crowding of the target address (head mean)
        ts = share[:, E_(k), li][:, H, tgt - 1].mean(1)     # scene's END(k) head-mean target share
        km1 = share[:, E_(k), li][:, H, k - 2].mean(1)
        dk = np.array([np.mean([0]) for _ in range(NS)])
        pers[k] = dict(r_crowd_target_share=float(np.corrcoef(dj, ts)[0, 1]), r_crowd_km1_share=float(np.corrcoef(dj, km1)[0, 1]))
    res["per_scene"] = pers
    out[L] = res
    print(f"== L{L} (matcher heads {len(H)})")
    print("   START address gap d_t (t=1..7):", res["start_gap_d"])
    print("   START logit gap g_t (t=1..7):  ", res["start_logit_gap"])
    print("   END(k) logit gap to target (k=1..7):", res["end_logit_gap_to_target"])
    print("   END(k) nearest START address = target:", [round(x, 2) for x in res["end_nearest_is_target"]], " = k-1:", [None if x is None else round(x, 2) for x in res["end_nearest_is_km1"]], " = literal k:", [round(x, 2) for x in res["end_nearest_is_literal"]])
    print("   copy neighbour |A_k-A_{k-1}| vs |A_k-A_{k+1}|:", {k: (round(v['to_km1'], 3), round(v['to_kp1'], 3), round(v['frac_km1_closer'], 2)) for k, v in res["copy_neighbour_asym"].items()})
    print("   per-scene r(crowding of target, END target share):", {k: round(v['r_crowd_target_share'], 2) for k, v in pers.items()})
json.dump(out, open(f"{O}/g_crowding.json", "w"), indent=1)
