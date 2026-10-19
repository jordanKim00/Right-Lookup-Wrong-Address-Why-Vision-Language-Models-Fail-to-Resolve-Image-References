"""S805b (descriptive re-analysis of saved S705a QK captures, no model call, not pre-registered): why is the far end-relative lookup of
Qwen3.5-27B a broad window?  Per (scene, index-matching head, lookup layer) the block log-shares at the answer row, log p_b (softmax over the
462 image-token logits, summed per block), for a start-relative and an end-relative question with the same target block.
  contrast   log p_peak - log mean(p of the other blocks)            (gain: a lower contrast with the same shape = a weaker address)
  halfmax    number of blocks with p >= p_peak / 2                   (resolution: the width of the window)
  modes      local maxima with p >= p_peak / 2                       (superposition: more than one mode inside one head)
  centre     the share-weighted mean block over the half-max window, minus the target (bias), and its spread across heads in a scene
  sharpened  the argmax after multiplying the block log-shares by beta = 4 (where a low-gain query would concentrate)
Usage: python3 exp/s805b_window_shape.py -> research/ces/outputs/s805b_window_shape.json"""
import json, collections
import numpy as np

ROOT = "/home/storage2/ai26269345/vlm"
J = json.load(open(f"{ROOT}/results/r3/s705a_qa_27b_qkanatomy.json"))
Z = np.load(f"{ROOT}/results/r3/s705a_qa_27b_qkanatomy.npz")
MH = collections.defaultdict(list)
for l, h in json.load(open(f"{ROOT}/exp/s786_heads/27b.json"))["matchers"]:
    MH[int(l)].append(int(h))
N, S = 7, len(J["meta"])


def logp(si, q, L):
    lg = Z[f"{si}|{q}|{L}|limg"][2].astype(np.float64)
    own = np.array(J["meta"][si]["owner"])
    w = np.exp(lg - lg.max(-1, keepdims=True)); w /= w.sum(-1, keepdims=True)
    p = np.stack([w[:, own == b].sum(-1) for b in range(N)], -1)
    return np.log(np.clip(p, 1e-12, 1))                                           # [24, 7]


def stats(lp, tgt):
    p = np.exp(lp); pk = int(p.argmax()); others = np.delete(p, pk)
    hm = p >= p[pk] / 2
    modes = sum(1 for b in range(N) if hm[b] and (b == 0 or p[b] >= p[b - 1]) and (b == N - 1 or p[b] >= p[b + 1]))
    idx = np.flatnonzero(hm); c = float((p[idx] * (idx + 1)).sum() / p[idx].sum())
    sh = np.exp(4 * lp); sh /= sh.sum()
    return dict(contrast=float(np.log(p[pk]) - np.log(others.mean())), halfmax=int(hm.sum()), modes=modes, centre=c - tgt,
                sharp_arg=int(sh.argmax()) + 1, peak=pk + 1)


PAIRS = [("S3m0", "E5m0", 3), ("S4m0", "E4m0", 4), ("S5m0", "E3m0", 5), ("S6m0", "E2m0", 6), ("S2m0", "E6m0", 2)]
out = {}
for L in (47, 51, 55):
    for qs, qe, tgt in PAIRS:
        for q in (qs, qe):
            rows = [[stats(lp, tgt) for lp in logp(si, q, L)[MH[L]]] for si in range(S)]
            flat = [x for r in rows for x in r]
            cs = [np.std([x["centre"] for x in r]) for r in rows]
            out[f"L{L}|{q}"] = dict(target=tgt, contrast=float(np.median([x["contrast"] for x in flat])),
                                    halfmax=float(np.mean([x["halfmax"] for x in flat])), multimodal=float(np.mean([x["modes"] >= 2 for x in flat])),
                                    centre_bias=float(np.mean([x["centre"] for x in flat])), centre_spread_heads=float(np.mean(cs)),
                                    sharp_on_target=float(np.mean([x["sharp_arg"] == tgt for x in flat])),
                                    sharp_hist=np.bincount([x["sharp_arg"] for x in flat], minlength=N + 1)[1:].tolist())
json.dump(out, open(f"{ROOT}/research/ces/outputs/s805b_window_shape.json", "w"), indent=1)
for L in (47, 51, 55):
    print(f"== L{L} ({len(MH[L])} heads): median contrast (nats) | mean half-max width (blocks) | multimodal share | centre bias · spread across heads | beta=4 argmax on target")
    for qs, qe, tgt in PAIRS:
        for q in (qs, qe):
            r = out[f"L{L}|{q}"]
            print(f"  {q[:2]:>2} target {tgt}: contrast {r['contrast']:.2f} | width {r['halfmax']:.2f} | multimodal {r['multimodal']:.2f} | centre {r['centre_bias']:+.2f} · {r['centre_spread_heads']:.2f} | sharp on target {r['sharp_on_target']:.2f} {r['sharp_hist']}")
