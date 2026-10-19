"""S799 (re-analysis of the saved S705a captures, no model call; descriptive): where the end-relative query sits on the start-relative address axis.
For each index-matching head (S720 heads for Qwen3.5-4B / 9B, S705a matchers for 27B) and the held-out half of the scenes, the answer-row queries
(post-rotary; START(t) and END(k) prompts have the same length, so the rotation is shared) are averaged over scenes.  The address axis of a head is
the first principal axis of its seven start-relative mean queries, scaled by least squares so that START(t) maps to t.  Reported per model and k:
the median over heads of the end-relative coordinate u(k) (copy predicts u = k, composition u = N + 1 - k), the fraction of the start-relative
variance the axis explains, and the per-scene spread of u for START(t) and END(k).
Usage: python s799_qaxis.py -> results/r3/s799_qaxis.json"""
import json
import numpy as np

R = "/vlm/results/r3"; N = 7
HEADS = {"qa_4b": f"{R}/s720_4b_heads.json", "qa_9b": f"{R}/s720_9b_heads.json", "qa_27b": f"{R}/s705a_qa_27b_matchers.json"}
ri = 2
out = {}
for tag, hf in HEADS.items():
    heads = [tuple(x) for x in json.load(open(hf))["matchers"]]
    D = f"{R}/s705a_{tag}_qkanatomy"; M = json.load(open(D + ".json"))["meta"]; z = np.load(D + ".npz")
    scenes = list(range(len(M) // 2, len(M)))
    U = {k: [] for k in range(1, N + 1)}; EV = []; SPREAD_S = {t: [] for t in range(1, N + 1)}; SPREAD_E = {k: [] for k in range(1, N + 1)}
    for (L, h) in heads:
        def q(fr, i, si):
            key = f"{si}|{fr}{i}m0|{L}|q"
            return z[key][ri, h].astype(np.float64) if key in z.files else None
        S = {t: [q("S", t, si) for si in scenes] for t in range(1, N + 1)}
        E = {k: [q("E", k, si) for si in scenes] for k in range(1, N + 1)}
        if any(x is None for t in S for x in S[t]) or any(x is None for k in E for x in E[k]):
            continue
        Sm = np.stack([np.mean(S[t], 0) for t in range(1, N + 1)])                # [7, d]
        mu = Sm.mean(0); C = Sm - mu
        _, sv, Vt = np.linalg.svd(C, full_matrices=False)
        pc1 = Vt[0]; EV.append(float(sv[0] ** 2 / (sv ** 2).sum()))
        x = C @ pc1; t = np.arange(1, N + 1)
        a, b = np.polyfit(x, t, 1)
        u = lambda v: float(a * ((v - mu) @ pc1) + b)
        for k in range(1, N + 1):
            U[k].append(u(np.mean(E[k], 0)))
            SPREAD_E[k].append(float(np.std([u(v) for v in E[k]])))
        for t_ in range(1, N + 1):
            SPREAD_S[t_].append(float(np.std([u(v) for v in S[t_]])))
    out[tag] = dict(n_heads=len(U[1]), ev_pc1_median=float(np.median(EV)),
                    u_end_median={str(k): round(float(np.median(U[k])), 2) for k in U},
                    u_end_iqr={str(k): [round(float(np.percentile(U[k], 25)), 2), round(float(np.percentile(U[k], 75)), 2)] for k in U},
                    spread_start={str(t): round(float(np.median(SPREAD_S[t])), 2) for t in SPREAD_S},
                    spread_end={str(k): round(float(np.median(SPREAD_E[k])), 2) for k in SPREAD_E})
    o = out[tag]
    print(f"{tag}: heads {o['n_heads']} PC1 explains {o['ev_pc1_median']:.2f} of START variance (median)")
    print("   u(END k):   " + " ".join(f"k{k}:{o['u_end_median'][str(k)]:.2f}" for k in range(1, N + 1)) + "   (copy u = k, composed u = 8 - k)")
    print("   spread START " + " ".join(f"{o['spread_start'][str(t)]:.2f}" for t in range(1, N + 1)) + " | END " + " ".join(f"{o['spread_end'][str(k)]:.2f}" for k in range(1, N + 1)))
json.dump(out, open(f"{R}/s799_qaxis.json", "w"), indent=1)
