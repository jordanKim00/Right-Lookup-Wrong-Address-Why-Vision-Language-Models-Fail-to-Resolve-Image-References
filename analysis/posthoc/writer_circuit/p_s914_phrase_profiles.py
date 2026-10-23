import json, numpy as np, collections
def load(tag):
    J = json.load(open(f"/vlm/results/r3/s913_{tag}_collect.json")); em = J["early_mean"]["47"]; out = collections.defaultdict(list)
    for r in J["rows"]:
        for k in J["ks"]:
            key = f"N_E{k}"
            h = np.asarray(r["qk47h"][key], float).mean(0); e = h[em].mean(0); e = e - e.mean()
            a = r["anchor"][str(k)]; srt = np.sort(e)[::-1]
            out[k].append(dict(sd=e.std(), top2=srt[0] - srt[1], m47=e[a - 1] - np.max(np.delete(e, a - 1)), a=a, x=r["land"][key]))
    return J["N"], out
for tag in ("s913n8_27b", "s914e8_27b"):
    N, o = load(tag)
    print(tag, "N", N)
    for k in sorted(o):
        v = o[k]
        print(f"  k{k}: sd {np.mean([d['sd'] for d in v]):.2f} top1-top2 {np.mean([d['top2'] for d in v]):.2f} m47 {np.mean([d['m47'] for d in v]):.2f} anchors {dict(collections.Counter(d['a'] for d in v).most_common(3))} acc {np.mean([d['x'] == N + 1 - k for d in v]):.2f}")
