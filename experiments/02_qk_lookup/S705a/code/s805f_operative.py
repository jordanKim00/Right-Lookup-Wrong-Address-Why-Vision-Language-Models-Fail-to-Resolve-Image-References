"""S805f (descriptive re-analysis of saved S705a captures, no model call, not pre-registered): which part of the answer-row query of each
index-matching head moves the image lookup, and how much of the END queries' ordinal difference lies in that part?
Qwen3.5-27B, N = 7, 30 scenes, answer row, all 256 dims as stored (q.k / 16 reproduces the stored logits, corr 1.000000).
Block logit S_b(q) = logsumexp over block b's 66 image tokens of q.k_p / 16.  Its gradient in q is kbar_b(q) / 16, with kbar_b(q) the mean key of
block b weighted by the head's own attention inside the block, so the block shares (softmax over b) move only along
W(q) = span{kbar_b(q) - mean_b kbar_b(q)} (rank <= 6): the operative address subspace at q.  U = span{Kb - mean_b Kb} with Kb the plain mean
key of block b (S808's address subspace, FINDINGS §880).
  overlap(U, W)  mean squared cosine of the principal angles between U and W(q) (1 = same subspace; two random 6-dim subspaces of 256 dims ~ .02)
  rho            share of a one-step query change inside W at the first query: START(t) -> START(t+1) and END(k) -> END(k+1)
                 (a random direction gives rank / 256 ~ .02)
  gain           |centered change of the 7 block logits| / |query change| for the same steps (nats per unit query norm)
  repair         END(k) query moved by parts of d = q_START(j) - q_END(k), j = 8 - k: the target block share (softmax of the 7 block logits) with
                 q_E, q_E + P_U d, q_E + (I - P_U) d, q_E + P_W d, q_E + (I - P_W) d (W = W(q_S(j)) + W(q_E) jointly), and q_S(j); one head, one
                 layer at a time (the lookup's immediate response, not a model run)
Groups: near k = 2; middle k = 3, 4, 5; first-image side k = 6, 7 (START steps t = 1..6, END steps k = 1..6).
Usage: python3 exp/s805f_operative.py -> research/ces/outputs/s805f_operative.json"""
import json, collections
import numpy as np

ROOT = "/home/storage2/ai26269345/vlm"
J = json.load(open(f"{ROOT}/results/r3/s705a_qa_27b_qkanatomy.json"))
Z = np.load(f"{ROOT}/results/r3/s705a_qa_27b_qkanatomy.npz")
MH = collections.defaultdict(list)
for l, h in json.load(open(f"{ROOT}/exp/s786_heads/27b.json"))["matchers"]:
    MH[int(l)].append(int(h))
N, S = 7, len(J["meta"])
GROUPS = dict(near=(2,), middle=(3, 4, 5), first_side=(6, 7))


def basis(C):
    u, s, vt = np.linalg.svd(C, full_matrices=False)
    return vt[s > 1e-6 * s.max()]                                                    # rows: orthonormal basis


def blocks(Kh, own, q):
    lg = Kh @ q / 16.0
    S_ = np.array([np.logaddexp.reduce(lg[own == b]) for b in range(N)])
    kb = []
    for b in range(N):
        x = lg[own == b]; w = np.exp(x - x.max()); w /= w.sum(); kb.append(w @ Kh[own == b])
    return S_, np.array(kb)


def share(S_, j):
    p = np.exp(S_ - S_.max()); p /= p.sum()
    return float(p[j - 1])


def proj(B, x):
    return B.T @ (B @ x)


out = {}
for L in (39, 43, 47, 51, 55):
    acc = collections.defaultdict(list)
    for si in range(S):
        K = Z[f"{si}|img|{L}|k"].astype(np.float64); own = np.array(J["meta"][si]["owner"])
        Q = {f"{fr}{t}": Z[f"{si}|{fr}{t}m0|{L}|q"][2].astype(np.float64) for fr in ("S", "E") for t in range(1, N + 1)}
        for h in MH[L]:
            Kh = K[:, h // 6, :]; Qh = {nm: q[h] for nm, q in Q.items()}
            Ub = basis(np.array([Kh[own == b].mean(0) for b in range(N)]) - np.array([Kh[own == b].mean(0) for b in range(N)]).mean(0))
            info = {}
            for nm, q in Qh.items():
                S_, kb = blocks(Kh, own, q)
                info[nm] = (S_, basis(kb - kb.mean(0)))
            for nm, (S_, Wb) in info.items():
                acc[f"overlap_{nm[0]}"].append(float((Ub @ Wb.T).__pow__(2).sum() / min(len(Ub), len(Wb))))
            for fr in ("S", "E"):
                for t in range(1, N):
                    a, b = f"{fr}{t}", f"{fr}{t + 1}"; d = Qh[b] - Qh[a]; nd = np.linalg.norm(d)
                    Wb = info[a][1]
                    dS = (info[b][0] - info[b][0].mean()) - (info[a][0] - info[a][0].mean())
                    acc[f"rho_{fr}{t}"].append(float(np.linalg.norm(proj(Wb, d)) ** 2 / nd ** 2))
                    acc[f"gain_{fr}{t}"].append(float(np.linalg.norm(dS) / nd))
            for k in range(2, N + 1):
                j = N + 1 - k; qE, qS = Qh[f"E{k}"], Qh[f"S{j}"]; d = qS - qE
                WJ = basis(np.vstack([info[f"S{j}"][1], info[f"E{k}"][1]]))
                for nm, q in (("END", qE), ("U", qE + proj(Ub, d)), ("Uperp", qE + d - proj(Ub, d)), ("W", qE + proj(WJ, d)),
                              ("Wperp", qE + d - proj(WJ, d)), ("START", qS)):
                    acc[f"rep_{nm}_k{k}"].append(share(blocks(Kh, own, q)[0], j))
                acc[f"repfrac_U_k{k}"].append(float(np.linalg.norm(proj(Ub, d)) ** 2 / np.linalg.norm(d) ** 2))
                acc[f"repfrac_W_k{k}"].append(float(np.linalg.norm(proj(WJ, d)) ** 2 / np.linalg.norm(d) ** 2))
    res = {k: float(np.mean(v)) for k, v in acc.items()}
    out[f"L{L}"] = res
    print(f"== L{L} ({len(MH[L])} heads)  overlap(U, W): START {res['overlap_S']:.2f} END {res['overlap_E']:.2f}")
    print("   rho  START t->t+1: " + " ".join(f"{res[f'rho_S{t}']:.2f}" for t in range(1, N)) +
          " | END k->k+1: " + " ".join(f"{res[f'rho_E{t}']:.2f}" for t in range(1, N)))
    print("   gain START: " + " ".join(f"{res[f'gain_S{t}']:.3f}" for t in range(1, N)) +
          " | END: " + " ".join(f"{res[f'gain_E{t}']:.3f}" for t in range(1, N)))
    for k in range(2, N + 1):
        print(f"   repair k{k} (target {N + 1 - k}) share: " + " ".join(f"{nm} {res[f'rep_{nm}_k{k}']:.2f}" for nm in ("END", "U", "Uperp", "W", "Wperp", "START")) +
              f" | |P_U d|^2 {res[f'repfrac_U_k{k}']:.2f} |P_W d|^2 {res[f'repfrac_W_k{k}']:.2f}")
json.dump(out, open(f"{ROOT}/research/ces/outputs/s805f_operative.json", "w"), indent=1)
