"""S805j (descriptive re-analysis of saved S705a captures, no model call, not pre-registered; answers not used): can a simple mixture of two
start-relative addresses produce an intermediate position?  For each scene and index-matching head at L51 / L55, the query
q(lam) = (1 - lam) q_START(5) + lam q_START(2) (also the U-part-only version: START(5)'s query with its U part replaced by the mixture of the two
U parts) is scored against the scene's image keys (block logsumexp of q.k / 16); mean block shares (softmax over the 7 blocks) of blocks 2-5.
If no lam gives blocks 3 or 4 a share near the END(5) data (L51: 3 ~ .37-weight, 4 ~ .42-weight in the address profile), a two-address mixture
cannot make the intermediate answers.  Usage: python3 exp/s805j_interp.py -> research/ces/outputs/s805j_interp.json"""
import json, collections
import numpy as np
ROOT = "/home/storage2/ai26269345/vlm"
J = json.load(open(f"{ROOT}/results/r3/s705a_qa_27b_qkanatomy.json")); Z = np.load(f"{ROOT}/results/r3/s705a_qa_27b_qkanatomy.npz")
MH = collections.defaultdict(list)
for l, h in json.load(open(f"{ROOT}/exp/s786_heads/27b.json"))["matchers"]:
    MH[int(l)].append(int(h))
N = 7; LAMS = [round(x, 1) for x in np.linspace(0, 1, 11)]
out = {}
for L in (51, 55):
    sh = {mode: {lam: [] for lam in LAMS} for mode in ("full", "U")}
    endsh = []
    for si in range(len(J["meta"])):
        own = np.array(J["meta"][si]["owner"]); K = Z[f"{si}|img|{L}|k"].astype(np.float64)
        q5, q2, qE = (Z[f"{si}|{a}m0|{L}|q"][2].astype(np.float64) for a in ("S5", "S2", "E5"))
        for h in MH[L]:
            Kh = K[:, h // 6, :]
            Kb = np.array([Kh[own == b].mean(0) for b in range(N)]); C = Kb - Kb.mean(0)
            u, s, vt = np.linalg.svd(C, full_matrices=False); B = vt[s > 1e-6 * s.max()]
            def share(q):
                lg = Kh @ q / 16; S_ = np.array([np.logaddexp.reduce(lg[own == b]) for b in range(N)]); p = np.exp(S_ - S_.max()); return p / p.sum()
            endsh.append(share(qE[h]))
            for lam in LAMS:
                sh["full"][lam].append(share((1 - lam) * q5[h] + lam * q2[h]))
                mixU = B.T @ (B @ ((1 - lam) * q5[h] + lam * q2[h])); q = q5[h] - B.T @ (B @ q5[h]) + mixU
                sh["U"][lam].append(share(q))
    res = {mode: {str(lam): np.mean(v, 0).round(3).tolist() for lam, v in d.items()} for mode, d in sh.items()}
    res["END5"] = np.mean(endsh, 0).round(3).tolist(); out[f"L{L}"] = res
    print(f"== L{L}: END(5) natural block shares 1..7: {res['END5']}")
    for mode in ("full", "U"):
        print(f"   {mode}: " + " | ".join(f"lam {lam}: 2 {res[mode][str(lam)][1]:.2f} 3 {res[mode][str(lam)][2]:.2f} 4 {res[mode][str(lam)][3]:.2f} 5 {res[mode][str(lam)][4]:.2f}" for lam in (0.0, 0.3, 0.5, 0.7, 1.0)))
json.dump(out, open(f"{ROOT}/research/ces/outputs/s805j_interp.json", "w"), indent=1)
