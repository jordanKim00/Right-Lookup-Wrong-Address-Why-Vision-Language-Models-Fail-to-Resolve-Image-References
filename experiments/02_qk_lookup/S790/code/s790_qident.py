"""S790 (saved-capture analysis, no model call): is the end-relative query the start-relative query of the same ordinal?
For the index-matching heads of Qwen3.5-4B / 9B / 27B (S705a captures, held-out half of the scenes, answer row), compare each END(k) query with
START(k) (same ordinal word, copy) and START(N+1-k) (same image, composition): mean cosine over heads and scenes, and the share of (scene, head)
pairs whose END(k) query is closer to START(k) than to START(N+1-k).  Also the lookup's own decoded address: the block-share-weighted mean block
index E[b] of each query (no fitted map).  N = 7, k = 1..7 (k = 4: both references coincide).
Usage: python s790_qident.py  -> results/r3/s790_qident.json"""
import json
import numpy as np

R = "/vlm/results/r3"; N = 7
HEADS = {"qa_4b": f"{R}/s720_4b_heads.json", "qa_9b": f"{R}/s720_9b_heads.json", "qa_27b": f"{R}/s705a_qa_27b_matchers.json"}


def lse(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True); m = np.where(np.isfinite(m), m, 0.)
    return np.squeeze(m, axis) + np.log(np.sum(np.exp(x - m), axis=axis))


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


out = {}
for tag, hf in HEADS.items():
    heads = [tuple(x) for x in json.load(open(hf))["matchers"]]
    D = f"{R}/s705a_{tag}_qkanatomy"; M = json.load(open(D + ".json"))["meta"]; z = np.load(D + ".npz")
    ri = 2                                                                          # answer row
    held = range(len(M) // 2, len(M))
    res = {}
    for k in range(1, N + 1):
        c_copy, c_comp, closer, eb_end, eb_s = [], [], [], [], []
        for si in held:
            own = np.array(M[si]["owner"])
            for (L, h) in heads:
                kq = lambda fr, q: f"{si}|{fr}{q}m0|{L}|q"
                kl = lambda fr, q: f"{si}|{fr}{q}m0|{L}|limg"
                if not all(x in z.files for x in (kq("E", k), kq("S", k), kq("S", N + 1 - k))):
                    continue
                qe, qs, qc = z[kq("E", k)][ri][h].astype(np.float64), z[kq("S", k)][ri][h].astype(np.float64), z[kq("S", N + 1 - k)][ri][h].astype(np.float64)
                a, b = cos(qe, qs), cos(qe, qc)
                c_copy.append(a); c_comp.append(b); closer.append(a > b)
                sc = np.stack([lse(z[kl("E", k)][ri][h].astype(np.float64)[own == bb]) for bb in range(N)]); p = np.exp(sc - lse(sc))
                eb_end.append(float((np.arange(1, N + 1) * p).sum()))
        res[str(k)] = dict(n=len(c_copy), cos_copy=float(np.mean(c_copy)), cos_comp=float(np.mean(c_comp)),
                           closer_to_copy=float(np.mean(closer)), addr_end=float(np.mean(eb_end)))
    # start-relative decoded address, for reference
    sres = {}
    for t in range(1, N + 1):
        eb = []
        for si in held:
            own = np.array(M[si]["owner"])
            for (L, h) in heads:
                kl = f"{si}|S{t}m0|{L}|limg"
                if kl not in z.files:
                    continue
                sc = np.stack([lse(z[kl][ri][h].astype(np.float64)[own == bb]) for bb in range(N)]); p = np.exp(sc - lse(sc))
                eb.append(float((np.arange(1, N + 1) * p).sum()))
        sres[str(t)] = float(np.mean(eb))
    out[tag] = dict(heads=len(heads), end=res, start_addr=sres)
    print(tag, len(heads), "heads")
    for k, r in res.items():
        print(f"  k={k}: cos(END,START(k)) {r['cos_copy']:.3f} · cos(END,START(N+1-k)) {r['cos_comp']:.3f} · closer to copy {r['closer_to_copy']:.2f} · "
              f"E[b] END {r['addr_end']:.2f} (START(k) {sres[k]:.2f}, START(N+1-k) {sres[str(N + 1 - int(k))]:.2f})  n={r['n']}")
json.dump(out, open(f"{R}/s790_qident.json", "w"), indent=1)
