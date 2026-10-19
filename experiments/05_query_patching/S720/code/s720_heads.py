"""S720 head sets: index-matching heads of Qwen3.5-4B / 9B from the saved S705a arrays (START diagonal share >= .5 at the answer or
tail1 row on the discovery scenes 0-14) -- the small-model analogue of S705a's MATCHER set without the END condition (small models
do not aim END at the target, §740).  Writes /vlm/results/r3/s720_{4b,9b}_heads.json."""
import json
import numpy as np

N = 7


def lse(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True); m = np.where(np.isfinite(m), m, 0.)
    return np.squeeze(m, axis) + np.log(np.sum(np.exp(x - m), axis=axis))


for tag in ("qa_4b", "qa_9b"):
    D = f"/vlm/results/r3/s705a_{tag}_qkanatomy"
    meta = json.load(open(D + ".json")); z = np.load(D + ".npz"); M = meta["meta"]; ns = len(M); CAP = meta["CAP"]
    heads = set()
    for L in CAP:
        for ri in (1, 2):
            vals = []
            for si in range(ns // 2):
                for t in range(1, N + 1):
                    key = f"{si}|S{t}m0|{L}|limg"
                    if key not in z.files:
                        continue
                    own = np.array(M[si]["owner"]); limg = z[key][ri].astype(np.float64)
                    sc = np.stack([lse(limg[:, own == b]) for b in range(N)], -1)
                    vals.append(np.exp(sc - lse(sc)[:, None])[:, t - 1])
            for h in np.nonzero(np.mean(vals, 0) >= .5)[0]:
                heads.add((int(L), int(h)))
    heads = sorted(heads)
    out = f"/vlm/results/r3/s720_{tag.split('_')[1]}_heads.json"
    json.dump(dict(tag=tag, model=meta["model"], matchers=[list(x) for x in heads],
                   rule="START diagonal share >= .5 at ans or tail1, discovery scenes 0-14 (S705a arrays)"), open(out, "w"))
    print(tag, meta["model"], "CAP", CAP, "index heads", len(heads), heads, "->", out)
