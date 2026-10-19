"""S855 stored data: the user's query-content x RoPE-phase cross-evaluation at layer 51 (post hoc, NOT registered; user 10-18).  No new run.
At layer 51 the late-only shift arms keep the query content of their own run and change only the rotary phase, and pictures 1-6 have the same
keys in every arm, so the block scores of pictures 1-6 give an exact 2 x 2: s(Q6, th6) = E6, s(Q6, th8) = E6SL, s(Q8, th6) = E8SL,
s(Q8, th8) = E8 (layers 55-63 of the shift arms also carry the changed layer-51 output, so only layer 51 is exact).  Total change D = E8 - E6;
phase effect at the 6-picture content = E6SL - E6, content effect at the 8-picture phase = E8 - E6SL (and the reverse order: content at the
6-picture phase = E8SL - E6, phase at the 8-picture content = E8 - E8SL).  Shares = projection on D / |D|^2 (sums over scenes, rows B1, B4,
answer, k, p; each 6-vector centred); scene bootstrap 4,000 (seed 11).
Usage: python3 exp/p4_s855_xeval.py  ->  research/ces/outputs/s855_xeval.json"""
import json
import numpy as np

V = "/home/storage2/ai26269345/vlm"


def main():
    d = json.load(open(f"{V}/results/r3/s855_s855_27b_geom.json")); by = {r["si"]: r for r in d["rows"]}; sis = sorted(by); n = len(sis)
    rng = np.random.default_rng(11); idx = rng.integers(0, n, (4000, n))
    cen = lambda x: np.asarray(x, dtype=float) - np.mean(x)
    out = {}
    for L in [str(l) for l in d["late"]]:
        z = lambda s, nm: np.concatenate([cen(by[s]["qk"][nm][L][ri][:6]) for ri in (2, 5, 9)])
        terms = {"phase_at_Q6": ("E6{}SL", "E6{}"), "content_at_th8": ("E8{}", "E6{}SL"), "content_at_th6": ("E8{}SL", "E6{}"), "phase_at_Q8": ("E8{}", "E8{}SL")}
        num = {t: [] for t in terms}; den = []
        for s in sis:
            a = {t: 0.0 for t in terms}; b = 0.0
            for k in (3, 5):
                for p in (3, 4):
                    c = f"_k{k}p{p}"; D = z(s, f"E8{c}") - z(s, f"E6{c}"); b += float(D @ D)
                    for t, (x, y) in terms.items():
                        a[t] += float((z(s, x.format(c)) - z(s, y.format(c))) @ D)
            for t in terms:
                num[t].append(a[t])
            den.append(b)
        den = np.asarray(den)
        out[L] = {t: [float(np.sum(v) / den.sum()), [float(np.percentile(np.asarray(v)[idx].sum(1) / den[idx].sum(1), q)) for q in (2.5, 97.5)]]
                  for t, v in num.items()}
    json.dump(dict(note="post hoc; exact 2 x 2 only at the first late layer", shares=out), open(f"{V}/research/ces/outputs/s855_xeval.json", "w"), indent=1)
    for L, v in out.items():
        print(f"layer {L}{' (exact)' if L == str(d['late'][0]) else ' (shift arms carry layer-51 changes)'}: " +
              " | ".join(f"{t} {x[0]:+.3f} [{x[1][0]:+.3f}, {x[1][1]:+.3f}]" for t, x in v.items()))


if __name__ == "__main__":
    main()
