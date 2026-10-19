"""agent_math_1015 / e_first_image (descriptive CPU re-analysis; second task E).  Is the first image (k = N) reached by a separate,
boundary-anchored pattern, and how does it differ from N + 1 - k for middle k?
(1) Behaviour, clean S801 27B: END accuracy at k = 1, 2, N-2, N-1, N for N = 6, 7, 8, and what the k = N errors are.
(2) S705a 27B QK (answer-free), answer row, look layers: per query head (all 24 and matcher heads), the share on block 1 for END(7) vs START(1),
    vs END(k) with k = 3..5 (block 1 is not their target: a 'boundary' preference) and vs START(3..5); the across-head correlation of END(7)'s
    block-1 share with START(1)'s (address pattern) and with the END-boundary preference (boundary pattern); the U-space cosine of END(k)'s query
    with START(N + 1 - k)'s, per k (how START-like the address is).
(3) S807 CNT6 at k = 6 (count 6 with 7 pictures makes k = N_s): the jump to picture 1.
Output: research/ces/outputs/agent_math_1015/e_first_image.json"""
import ast, json
from collections import Counter
import numpy as np

R3 = "/home/storage2/ai26269345/vlm/results/r3"; O = "/home/storage2/ai26269345/vlm/research/ces/outputs/agent_math_1015"
out = {}
# (1) behaviour
beh = {}
for r in json.load(open(f"{R3}/s801_s801_27b_grounding.json"))["rows"]:
    e = r["END"]; e = ast.literal_eval(e) if isinstance(e, str) else e
    beh.setdefault((int(r["N"]), int(r["k"])), []).append(int(e["land"]))
b1 = {}
for (N, k), L in sorted(beh.items()):
    c = Counter(L); n = len(L)
    b1[f"N{N}_k{k}"] = dict(n=n, acc=c[N + 1 - k] / n, literal=c[k] / n if k <= N else None, dist=[c.get(b, 0) for b in range(N + 1)])
out["s801"] = b1
print("S801 END accuracy (k = 1, 2, N-1, N):")
for N in (6, 7, 8):
    print(f"  N{N}: " + " · ".join(f"k{k} {b1[f'N{N}_k{k}']['acc']:.2f}" for k in (1, 2, N - 2, N - 1, N) if f"N{N}_k{k}" in b1) +
          f" | k=N dist {b1[f'N{N}_k{N}']['dist']}")
# (2) S705a
C = np.load(f"{O}/s705a_cache.npz"); share = C["share"]; q = C["q"]; kbar = C["kbar"]; runs = list(C["runs"]); cap = [int(x) for x in C["cap"]]
mat = json.load(open("/home/storage2/ai26269345/vlm/results/r3/s705a_qa_27b_matchers.json"))
MH = {L: [h for (l, h) in mat["matchers"] if l == L] for L in cap}; AN = {L: [h for (l, h) in mat["anchors"] if l == L] for L in cap}
R_ = lambda nm: runs.index(nm)
qa = {}
for L in cap:
    li = cap.index(L)
    b1_E7 = share[:, R_("E7m0"), li, :, 0].mean(0); b1_S1 = share[:, R_("S1m0"), li, :, 0].mean(0)
    b1_Emid = np.mean([share[:, R_(f"E{k}m0"), li, :, 0].mean(0) for k in (3, 4, 5)], 0)
    b1_Smid = np.mean([share[:, R_(f"S{t}m0"), li, :, 0].mean(0) for t in (3, 4, 5)], 0)
    allh = list(range(24)); H = MH[L]
    def cor(a, b, hs):
        return float(np.corrcoef(a[hs], b[hs])[0, 1])
    # U-space cosine END(k) vs START(8-k), matcher heads
    cosk = {}
    for k in range(1, 8):
        v = []
        for si in range(q.shape[0]):
            for h in H:
                Kb = kbar[si, li, h // 6]; Kc = Kb - Kb.mean(0); Uq, Us, _ = np.linalg.svd(Kc.T, full_matrices=False); Ub = Uq[:, Us > 1e-6 * Us.max()]
                a = Ub.T @ q[si, R_(f"E{k}m0"), li, h]; b = Ub.T @ q[si, R_(f"S{8 - k}m0"), li, h]
                v.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)))
        cosk[k] = float(np.mean(v))
    qa[L] = dict(mh_b1_E7=float(b1_E7[H].mean()), mh_b1_S1=float(b1_S1[H].mean()), mh_b1_Emid=float(b1_Emid[H].mean()), mh_b1_Smid=float(b1_Smid[H].mean()),
                 all_b1_E7=float(b1_E7.mean()), all_b1_S1=float(b1_S1.mean()), all_b1_Emid=float(b1_Emid.mean()),
                 r_heads_E7_S1_all=cor(b1_E7, b1_S1, allh), r_heads_E7_Emid_all=cor(b1_E7, b1_Emid, allh),
                 r_heads_E7_S1_mh=cor(b1_E7, b1_S1, H) if len(H) > 2 else None, r_heads_E7_Emid_mh=cor(b1_E7, b1_Emid, H) if len(H) > 2 else None,
                 anchors_b1_E7=float(b1_E7[AN[L]].mean()) if AN[L] else None, anchors_b1_S1=float(b1_S1[AN[L]].mean()) if AN[L] else None,
                 top_E7_heads=[int(h) for h in np.argsort(-b1_E7)[:5]], top_S1_heads=[int(h) for h in np.argsort(-b1_S1)[:5]],
                 U_cos_END_k_vs_START_target=cosk)
    v = qa[L]
    print(f"L{L}: block-1 share E7 {v['mh_b1_E7']:.2f} (all {v['all_b1_E7']:.2f}) · S1 {v['mh_b1_S1']:.2f} (all {v['all_b1_S1']:.2f}) · END mid {v['mh_b1_Emid']:.2f} · START mid {v['mh_b1_Smid']:.2f} | "
          f"across 24 heads r(E7, S1) {v['r_heads_E7_S1_all']:+.2f} · r(E7, END-mid) {v['r_heads_E7_Emid_all']:+.2f} | top E7 {v['top_E7_heads']} top S1 {v['top_S1_heads']} | "
          f"U-cos END(k)~START(8-k) " + " ".join(f"k{k} {c:.2f}" for k, c in cosk.items()))
out["s705a"] = qa
# (3) S807 CNT6 at k = 6
c = Counter(r["land"]["CNT6"] for r in json.load(open(f"{R3}/s807_s807_27b_source.json"))["rows"] if r["k"] == 6)
c0 = Counter(r["land"]["END"] for r in json.load(open(f"{R3}/s807_s807_27b_source.json"))["rows"] if r["k"] == 6)
out["s807_cnt6_k6"] = dict(cnt6=[c.get(b, 0) for b in range(8)], end=[c0.get(b, 0) for b in range(8)])
print("S807 k6: END", out["s807_cnt6_k6"]["end"], " CNT6", out["s807_cnt6_k6"]["cnt6"])
json.dump(out, open(f"{O}/e_first_image.json", "w"), indent=1)
