"""S805i (descriptive re-analysis of saved S705a captures, no model call, not pre-registered; answers not used - S705a landings are pre-S782):
for END(k) at the answer row, each index-matching head's nearest start-relative address at L43 and at L51 (U = span of the centred block-mean
post-RoPE image keys of that head; A_t = START(t)'s query part in U; nearest by Euclidean distance in U), and the transition table
address(L43) -> address(L51) per k.  Qwen3.5-27B, N = 7, 30 scenes.
Usage: python3 exp/s805i_transition.py -> research/ces/outputs/s805i_transition.json"""
import json, collections
import numpy as np
ROOT = "/home/storage2/ai26269345/vlm"
J = json.load(open(f"{ROOT}/results/r3/s705a_qa_27b_qkanatomy.json")); Z = np.load(f"{ROOT}/results/r3/s705a_qa_27b_qkanatomy.npz")
MH = collections.defaultdict(list)
for l, h in json.load(open(f"{ROOT}/exp/s786_heads/27b.json"))["matchers"]:
    MH[int(l)].append(int(h))
N = 7; S = len(J["meta"]); LA, LB = 43, 51
def nearest(si, L, h, fr, k):
    own = np.array(J["meta"][si]["owner"]); Kh = Z[f"{si}|img|{L}|k"].astype(np.float64)[:, h // 6, :]
    Kb = np.array([Kh[own == b].mean(0) for b in range(N)]); C = Kb - Kb.mean(0)
    u, s, vt = np.linalg.svd(C, full_matrices=False); B = vt[s > 1e-6 * s.max()]
    A = np.array([B.T @ (B @ Z[f"{si}|S{t}m0|{L}|q"][2, h].astype(np.float64)) for t in range(1, N + 1)])
    a = B.T @ (B @ Z[f"{si}|{fr}{k}m0|{L}|q"][2, h].astype(np.float64))
    return int(np.argmin(np.linalg.norm(A - a, axis=1))) + 1
out = {}
for k in (2, 3, 4, 5, 6, 7):
    # per scene: majority nearest address over the heads of each layer (heads differ between layers, so the scene is the unit)
    trans = collections.Counter(); addrA = collections.Counter(); addrB = collections.Counter()
    for si in range(S):
        va = collections.Counter(nearest(si, LA, h, "E", k) for h in MH[LA]).most_common(1)[0][0]
        vb = collections.Counter(nearest(si, LB, h, "E", k) for h in MH[LB]).most_common(1)[0][0]
        trans[(va, vb)] += 1; addrA[va] += 1; addrB[vb] += 1
    j = N + 1 - k
    out[f"k{k}"] = dict(j=j, L43=dict(sorted(addrA.items())), L51=dict(sorted(addrB.items())), transitions={f"{a}->{b}": c for (a, b), c in sorted(trans.items())})
    print(f"END({k}) target {j}: L43 majority address {dict(sorted(addrA.items()))} | L51 {dict(sorted(addrB.items()))}")
    print("    transitions L43->L51:", ", ".join(f"{a}->{b}: {c}" for (a, b), c in trans.most_common()))
json.dump(out, open(f"{ROOT}/research/ces/outputs/s805i_transition.json", "w"), indent=1)
