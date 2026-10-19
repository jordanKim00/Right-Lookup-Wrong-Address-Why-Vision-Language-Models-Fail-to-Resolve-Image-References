"""S705a follow-up (exploratory, registered as such in knowledge/DECISIONS.md): contrastive query-key bases (Lee et al. 2602.04752) for the
MATCHER heads, fitted on START runs of the discovery scenes and read on the held-out END runs.
Per head (layer, head, row): q = post-RoPE query of the row; positive key = the target block's key, negative = mean of the other blocks'
keys, with a block's key = the mean post-RoPE key over its tokens ("blk") or the key of its vision_end token ("ve").
  dC = mean_{scene, t} q (k_pos - k_neg)^T  ->  SVD  ->  rank (99% squared-singular-value energy), U (query side), V (key side).
Held-out readings:
  sub_argmax(END k): argmax_j (q^T U_r U_r^T) (V_r V_r^T k_j) -- the block the relation subspace alone prefers; compared with the
                     target N+1-k, with k (literal), and with the native block-score argmax;
  key manifold: coordinates V_r^T k_j of the seven blocks (per scene), first component vs j (Spearman) -- an index code on the keys?
  query reading: the START query coordinates U_r^T q(t) define an index line; END(k) queries are placed on it (t_hat) by least squares on
                 the discovery START means -- t_hat vs N+1-k per k (spread grows with steps?).
Usage: python s705a_contrast.py TAG
"""
import json, sys
import numpy as np

N = 7


def basis(dC, frac=.99):
    U, s, Vh = np.linalg.svd(dC)
    e = s ** 2; c = np.cumsum(e) / e.sum()
    r = int(np.searchsorted(c, frac) + 1)
    return U[:, :r], Vh[:r].T, r, s


def main(tag):
    D = f"/vlm/results/r3/s705a_{tag}_qkanatomy"
    meta = json.load(open(D + ".json")); z = np.load(D + ".npz"); M = meta["meta"]
    heads = json.load(open(f"/vlm/results/r3/s705a_{tag}_matchers.json"))
    an = json.load(open("/vlm/results/r3/s705a_analysis.json"))[tag]["heads"]["matchers"]
    ns = len(M); disc, hold = list(range(ns // 2)), list(range(ns // 2, ns))
    out = {}
    for (li, h, ri) in an:
        for kind in ("blk", "ve"):
            def keys(si):
                K = z[f"{si}|img|{li}|k"].astype(np.float64)                     # [P, KVH, d]
                nh_q = z[f"{si}|S1m0|{li}|q"].shape[1]; kh = h // (nh_q // K.shape[1])   # GQA: query head -> its KV head
                own = np.array(M[si]["owner"])
                if kind == "blk":
                    return np.stack([K[own == b, kh].mean(0) for b in range(N)])   # [N, d]
                last = [np.where(own == b)[0][-1] for b in range(N)]               # vision_end of each block
                return np.stack([K[p, kh] for p in last])
            Q = lambda si, key: z[f"{si}|{key}|{li}|q"][ri, h].astype(np.float64)
            dC = np.zeros((Q(0, "S1m0").shape[0],) * 2)
            n = 0
            for si in disc:
                Kb = keys(si)
                for t in range(1, N + 1):
                    q = Q(si, f"S{t}m0"); neg = np.delete(Kb, t - 1, 0).mean(0)
                    dC += np.outer(q, Kb[t - 1] - neg); n += 1
            dC /= n
            U, V, r, s = basis(dC)
            # START index line in query coordinates (discovery means)
            cs = np.stack([np.mean([U.T @ Q(si, f"S{t}m0") for si in disc], 0) for t in range(1, N + 1)])   # [N, r]
            A = np.c_[np.arange(1, N + 1), np.ones(N)]
            W, *_ = np.linalg.lstsq(cs, A, rcond=None)                               # coords -> (t, 1)
            res = dict(rank=r, sv=[float(x) for x in s[:6]])
            sub_hit, sub_lit, nat_hit, that = {}, {}, {}, {}
            for k in range(1, N + 1):
                hits, lits, nats, th = [], [], [], []
                for si in hold:
                    key = f"E{k}m0"
                    if f"{si}|{key}|{li}|q" not in z.files:
                        continue
                    Kb = keys(si); q = Q(si, key)
                    sc = (q @ U @ U.T) @ (V @ V.T @ Kb.T)
                    am = int(np.argmax(sc)) + 1
                    hits.append(am == N + 1 - k); lits.append(am == k)
                    own = np.array(M[si]["owner"]); limg = z[f"{si}|{key}|{li}|limg"][ri, h].astype(np.float64)
                    nb = [np.log(np.sum(np.exp(limg[own == b] - limg.max()))) for b in range(N)]
                    nats.append(int(np.argmax(nb)) + 1 == N + 1 - k)
                    th.append(float((U.T @ q) @ W[:, 0]))
                sub_hit[k], sub_lit[k], nat_hit[k] = float(np.mean(hits)), float(np.mean(lits)), float(np.mean(nats))
                that[k] = (float(np.mean(th)), float(np.std(th)))
            # key manifold: first coordinate vs block index (held-out)
            rho = []
            for si in hold:
                c1 = (keys(si) @ V)[:, 0]
                rho.append(np.corrcoef(np.argsort(np.argsort(c1)), np.arange(N))[0, 1])
            res.update(sub_target=sub_hit, sub_literal=sub_lit, native_target=nat_hit, t_hat=that, key_rank_corr=float(np.mean(rho)))
            out[f"L{li}h{h}r{ri}|{kind}"] = res
            print(f"[{tag}] L{li} h{h} row{ri} {kind}: rank {r} | key-index rho {np.mean(rho):+.2f} | subspace->target by k "
                  + " ".join(f"{k}:{sub_hit[k]:.2f}" for k in range(1, N + 1)) + " | native " + " ".join(f"{nat_hit[k]:.2f}" for k in range(1, N + 1))
                  + " | t_hat(k) " + " ".join(f"{that[k][0]:.1f}±{that[k][1]:.1f}" for k in range(1, N + 1)))
    json.dump(out, open(f"/vlm/results/r3/s705a_{tag}_contrast.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
