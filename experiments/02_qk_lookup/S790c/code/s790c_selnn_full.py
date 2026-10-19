"""S790c (saved-capture re-analysis; the full table behind S790b): for each END(k), the distribution over t of the start-relative question
START(t) whose block-share distribution is nearest (Jensen-Shannon), per model -- a 7 x 7 matrix per model.  Same captures, heads, rows and
scenes as S790b.  Usage: python s790c_selnn_full.py -> results/r3/s790c_selnn_full.json

S790b (saved-capture re-analysis, no model call; control for S790): the S790 query cosine can favour the same-ordinal start-relative query
for a lexical reason (END(k) and START(k) share the ordinal word).  Selection space has no such confound: for each (held-out scene, matcher
head), which start-relative question START(t), t = 1..7, has the block-share distribution nearest (Jensen-Shannon) to that of END(k)?
Reported: share of pairs whose nearest t is k (copy), N+1-k (composition) or another t, per model and k.  Same captures, heads and rows as S790.
Usage: python s790b_selnn.py -> results/r3/s790b_selnn.json"""
import json
import numpy as np

R = "/vlm/results/r3"; N = 7
HEADS = {"qa_4b": f"{R}/s720_4b_heads.json", "qa_9b": f"{R}/s720_9b_heads.json", "qa_27b": f"{R}/s705a_qa_27b_matchers.json"}


def lse(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True); m = np.where(np.isfinite(m), m, 0.)
    return np.squeeze(m, axis) + np.log(np.sum(np.exp(x - m), axis=axis))


def jsd(p, q):
    m = .5 * (p + q)
    kl = lambda a, b: float(np.sum(a * (np.log(a + 1e-12) - np.log(b + 1e-12))))
    return .5 * kl(p, m) + .5 * kl(q, m)


out = {}
for tag, hf in HEADS.items():
    heads = [tuple(x) for x in json.load(open(hf))["matchers"]]
    D = f"{R}/s705a_{tag}_qkanatomy"; M = json.load(open(D + ".json"))["meta"]; z = np.load(D + ".npz"); ri = 2
    res = {str(k): dict(copy=0, comp=0, other=0, n=0) for k in range(1, N + 1)}; full = {str(k): [0] * N for k in range(1, N + 1)}
    for si in range(len(M) // 2, len(M)):
        own = np.array(M[si]["owner"])
        for (L, h) in heads:
            def share(fr, q):
                key = f"{si}|{fr}{q}m0|{L}|limg"
                if key not in z.files:
                    return None
                sc = np.stack([lse(z[key][ri][h].astype(np.float64)[own == b]) for b in range(N)]); return np.exp(sc - lse(sc))
            S = {t: share("S", t) for t in range(1, N + 1)}
            if any(v is None for v in S.values()):
                continue
            for k in range(1, N + 1):
                e = share("E", k)
                if e is None:
                    continue
                t = min(S, key=lambda t_: jsd(e, S[t_]))
                r = res[str(k)]; r["n"] += 1
                r["copy" if t == k else "comp" if t == N + 1 - k else "other"] += 1; full[str(k)][t - 1] += 1
    out[tag] = {k: dict(**{x: (v / r["n"] if x != "n" else v) for x, v in r.items()}, nearest_t=[c / max(1, r["n"]) for c in full[k]]) for k, r in res.items()}
    print(tag, " ".join(f"k{k}: copy {r['copy']:.2f} comp {r['comp']:.2f} other {r['other']:.2f}" for k, r in out[tag].items()))
json.dump(out, open(f"{R}/s790c_selnn_full.json", "w"), indent=1)
