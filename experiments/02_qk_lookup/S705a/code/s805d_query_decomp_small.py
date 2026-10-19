"""S805d (descriptive re-analysis of saved S705a captures, no model call, not pre-registered): the S805c decomposition (FINDINGS §872) for
Qwen3.5-4B and 9B.  Is the far end-relative query of the smaller models a copy of the literal ordinal's start-relative address (START(k)), a
smeared end-relative address (the average of the target's and its neighbours' addresses, as in 27B), or the target's own address?
Per (scene, index-matching head, layer), in the code subspace U of that layer and key head (centred block-mean image keys), cosines of the
END(k) query with START(target), with the normalised average of START(target - 1, target, target + 1), and with START(k) (the copy); the
logits are first checked to be reproduced by q . k * scale.
Usage: python3 exp/s805d_query_decomp_small.py -> research/ces/outputs/s805d_query_decomp_small.json"""
import json, collections
import numpy as np

ROOT = "/home/storage2/ai26269345/vlm"
N = 7
cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
out = {}
for tag, htag in (("4b", "35_4b"), ("9b", "35_9b")):
    J = json.load(open(f"{ROOT}/results/r3/s705a_qa_{tag}_qkanatomy.json")); Z = np.load(f"{ROOT}/results/r3/s705a_qa_{tag}_qkanatomy.npz")
    MH = collections.defaultdict(list)
    for l, h in json.load(open(f"{ROOT}/exp/s786_heads/{htag}.json"))["matchers"]:
        MH[int(l)].append(int(h))
    S = len(J["meta"])
    q0 = Z[f"0|S1m0|{J['CAP'][-1]}|q"]; k0 = Z[f"0|img|{J['CAP'][-1]}|k"]; H, KV, hd = q0.shape[1], k0.shape[1], q0.shape[2]; G = H // KV
    lg0 = Z[f"0|S1m0|{J['CAP'][-1]}|limg"].astype(np.float64)
    rec = q0[2, 0].astype(np.float64) @ k0[:, 0].astype(np.float64).T
    chk = float(np.corrcoef(rec, lg0[2, 0])[0, 1]); scale = float(np.polyfit(rec, lg0[2, 0], 1)[0])
    print(f"== {tag}: heads {H}, key heads {KV}, head dim {hd}; logit reproduction corr {chk:.6f}, scale {scale:.5f} (1/sqrt(hd) = {hd ** -.5:.5f})")
    def basis(si, L, kv):
        K = Z[f"{si}|img|{L}|k"][:, kv].astype(np.float64); own = np.array(J["meta"][si]["owner"])
        Kb = np.stack([K[own == b].mean(0) for b in range(N)]); C = Kb - Kb.mean(0)
        U, s, _ = np.linalg.svd(C.T, full_matrices=False); return U[:, s > 1e-6 * s.max()]
    for L in sorted(MH):
        R = collections.defaultdict(list)
        for si in range(S):
            Qs = {t: Z[f"{si}|S{t}m0|{L}|q"][2].astype(np.float64) for t in range(1, N + 1)}
            Qe = {k: Z[f"{si}|E{k}m0|{L}|q"][2].astype(np.float64) for k in (2, 3, 4, 5, 6)}
            for h in MH[L]:
                U = basis(si, L, h // G); P = lambda v: U.T @ v; ps = {t: P(Qs[t][h]) for t in Qs}
                R["S3~S4"].append(cos(ps[3], ps[4]))
                for k in (2, 3, 4, 5, 6):
                    tgt = N + 1 - k; pe = P(Qe[k][h]); nb = [t for t in (tgt - 1, tgt, tgt + 1) if 1 <= t <= N]
                    avg = np.mean([ps[t] / np.linalg.norm(ps[t]) for t in nb], 0)
                    R[f"E{k}|target"].append(cos(pe, ps[tgt])); R[f"E{k}|smear"].append(cos(pe, avg)); R[f"E{k}|copy"].append(cos(pe, ps[k]))
        med = {key: float(np.median(v)) for key, v in R.items()}
        out[f"{tag}|L{L}"] = med
        print(f"  L{L} ({len(MH[L])} heads): START(3)~START(4) {med['S3~S4']:+.2f} | " +
              " | ".join(f"END({k}) tgt {med[f'E{k}|target']:+.2f} smear {med[f'E{k}|smear']:+.2f} copy {med[f'E{k}|copy']:+.2f}" for k in (2, 3, 5, 6)))
json.dump(out, open(f"{ROOT}/research/ces/outputs/s805d_query_decomp_small.json", "w"), indent=1)
