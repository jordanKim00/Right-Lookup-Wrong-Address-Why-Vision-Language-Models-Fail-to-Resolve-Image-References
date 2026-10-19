"""S805c (descriptive re-analysis of saved S705a captures, no model call, not pre-registered): what kind of address does the far end-relative
query of Qwen3.5-27B carry?  The stored queries and image keys reproduce the stored logits exactly (limg = q . k / 16, query head h uses key
head h // 6).  Per (scene, index-matching head, layer): the code subspace U = span of the block-mean image keys centred across the 7 blocks;
P q = the query's projection on U.  For END(k) against the start-relative queries START(t) of the same scene and head:
  gain       |P q_END| / |P q_START(target)|                       (a pure low-gain address keeps the direction, shrinks the norm)
  cos_t      cosine of P q_END with P q_START(t), t = 1..7         (which start-relative address it points at)
  NNLS       P q_END ~ sum_t w_t P q_START(t), w >= 0:  weight sum, weight centre - target, weight spread (sd over t), R^2
             (interpolation: weight on target and its neighbours; superposition: weight on blocks two or more apart)
Usage: python3 exp/s805c_query_decomp.py -> research/ces/outputs/s805c_query_decomp.json"""
import json, collections
import numpy as np

ROOT = "/home/storage2/ai26269345/vlm"
J = json.load(open(f"{ROOT}/results/r3/s705a_qa_27b_qkanatomy.json"))
Z = np.load(f"{ROOT}/results/r3/s705a_qa_27b_qkanatomy.npz")
MH = collections.defaultdict(list)
for l, h in json.load(open(f"{ROOT}/exp/s786_heads/27b.json"))["matchers"]:
    MH[int(l)].append(int(h))
N, S, ROW = 7, len(J["meta"]), 2


def nnls(A, b, iters=3000):
    """Projected gradient NNLS (A: [d, m], b: [d]); small m, exact enough for a description."""
    x = np.zeros(A.shape[1]); AtA = A.T @ A; Atb = A.T @ b; lr = 1.0 / (np.linalg.eigvalsh(AtA).max() + 1e-12)
    for _ in range(iters):
        x = np.maximum(0, x - lr * (AtA @ x - Atb))
    return x


def basis(si, L, kv):
    K = Z[f"{si}|img|{L}|k"][:, kv].astype(np.float64)                   # [462, 256]
    own = np.array(J["meta"][si]["owner"])
    Kb = np.stack([K[own == b].mean(0) for b in range(N)]); C = Kb - Kb.mean(0)
    U, s, _ = np.linalg.svd(C.T, full_matrices=False)                    # [256, 7]
    return U[:, s > 1e-6 * s.max()]


PAIRS = [("E5m0", 3), ("E4m0", 4), ("E3m0", 5), ("E2m0", 6), ("E6m0", 2)]
out = {}
for L in (47, 51, 55):
    for qe, tgt in PAIRS:
        rec = []
        for si in range(S):
            Qs = {t: Z[f"{si}|S{t}m0|{L}|q"][ROW].astype(np.float64) for t in range(1, N + 1)}
            Qe = Z[f"{si}|{qe}|{L}|q"][ROW].astype(np.float64)
            for h in MH[L]:
                U = basis(si, L, h // 6); P = lambda v: U.T @ v
                pe = P(Qe[h]); ps = {t: P(Qs[t][h]) for t in Qs}
                cos = np.array([pe @ ps[t] / (np.linalg.norm(pe) * np.linalg.norm(ps[t]) + 1e-12) for t in range(1, N + 1)])
                A = np.stack([ps[t] for t in range(1, N + 1)], 1); w = nnls(A, pe)
                fit = A @ w; r2 = 1 - ((pe - fit) ** 2).sum() / ((pe - pe.mean()) ** 2).sum() if pe.size > 1 else np.nan
                ws = w.sum(); tt = np.arange(1, N + 1)
                cen = float((w * tt).sum() / ws) if ws > 0 else np.nan
                spr = float(np.sqrt((w * (tt - cen) ** 2).sum() / ws)) if ws > 0 else np.nan
                far = float(w[np.abs(tt - tgt) >= 2].sum() / ws) if ws > 0 else np.nan
                rec.append(dict(gain=float(np.linalg.norm(pe) / (np.linalg.norm(ps[tgt]) + 1e-12)), cos_tgt=float(cos[tgt - 1]),
                                best_t=int(cos.argmax()) + 1, wsum=float(ws), wcen=cen - tgt, wspread=spr, wfar=far, r2=float(r2)))
        g = lambda key: float(np.nanmedian([r[key] for r in rec]))
        bt = collections.Counter(r["best_t"] for r in rec)
        out[f"L{L}|{qe}"] = dict(target=tgt, n=len(rec), gain=g("gain"), cos_target=g("cos_tgt"), wsum=g("wsum"), wcentre=g("wcen"),
                                 wspread=g("wspread"), wfar=g("wfar"), r2=g("r2"), best_t={str(t): bt.get(t, 0) / len(rec) for t in range(1, N + 1)})
    # reference: START(t) against START(t') of other scenes is not defined here; START's own gain is 1 by construction
json.dump(out, open(f"{ROOT}/research/ces/outputs/s805c_query_decomp.json", "w"), indent=1)
for L in (47, 51, 55):
    print(f"== L{L} ({len(MH[L])} heads; medians over head-scenes)")
    for qe, tgt in PAIRS:
        r = out[f"L{L}|{qe}"]
        print(f"  END({qe[1]}) target {tgt}: gain {r['gain']:.2f} | cos to START(target) {r['cos_target']:.2f} | NNLS weight sum {r['wsum']:.2f}, "
              f"centre {r['wcentre']:+.2f}, spread {r['wspread']:.2f}, share >= 2 away {r['wfar']:.2f}, R2 {r['r2']:.2f} | best-cos START t: "
              + " ".join(f"{t}:{v:.2f}" for t, v in r["best_t"].items()))
