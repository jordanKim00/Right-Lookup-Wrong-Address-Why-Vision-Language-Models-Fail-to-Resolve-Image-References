"""P0 (user 10-18): where does the actual-count change of the address first appear, by read row and layer? (post hoc, NOT registered; stored S855
data, no new run.)  Block scores are the projection of the query onto the picture keys, so their change is the QK-effective part of the query
change (head-averaged: matcher heads at 35-55, all heads at 59 / 63; per-head keys were not stored).  For every read row r (question mark, B0-B4,
three thinking rows, answer) and full-attention layer l (35-63), with pictures 1-6 (same positions and keys in both counts; each 6-vector centred):
  dz_a = z(E8) - z(E6)    the actual-count change (same scene, k, p)
  align  = sum dz_a . dL / sqrt(sum |dz_a|^2 sum |dL|^2) with dL = the change of the menu logits of pictures 1-6 (the answer-side direction)
  size   = sqrt(sum |dz_a|^2 / sum |dz_k|^2), dz_k = z(k 5) - z(k 3) at the same count and p (the ordinal change at the same row and layer)
  cons   = |mean dz_a|^2 / mean |dz_a|^2 over scenes (shared direction; about 1/30 for scene-specific noise)
  pic    = sum (z(E8S) - z(E6)) . dz_a / sum |dz_a|^2 (the part that stays with pictures 7-8 at the 6-picture geometry)
Early layers 35-47 carry the pinned address part of the read rows, so their scores are dominated by the pin.
Usage: python3 exp/p4_s855_rows.py  ->  research/ces/outputs/s855_rows.json"""
import json
import numpy as np

V = "/home/storage2/ai26269345/vlm"
ROWS = ["qm", "B0", "B1", "B2", "B3", "B4", "think", "nl", "think_end", "answer"]


def main():
    d = json.load(open(f"{V}/results/r3/s855_s855_27b_geom.json")); by = {r["si"]: r for r in d["rows"]}; sis = sorted(by)
    cen = lambda x: np.asarray(x, dtype=float) - np.mean(x)
    LAYERS = [str(l) for l in d["read"]]
    dL = {(s, k, p): cen(by[s]["mlog"][f"E8_k{k}p{p}"][:6]) - cen(by[s]["mlog"][f"E6_k{k}p{p}"][:6]) for s in sis for k in (3, 5) for p in (3, 4)}
    out = {}
    for L in LAYERS:
        for ri, rn in enumerate(ROWS):
            z = lambda s, nm: cen(by[s]["qk"][nm][L][ri][:6])
            num = den_a = den_l = den_k = pic = 0.0; per_scene = []
            for s in sis:
                acc = np.zeros(6 * 4); j = 0
                for k in (3, 5):
                    for p in (3, 4):
                        da = z(s, f"E8_k{k}p{p}") - z(s, f"E6_k{k}p{p}")
                        dk = z(s, f"E8_k5p{p}") - z(s, f"E8_k3p{p}")
                        num += float(da @ dL[(s, k, p)]); den_a += float(da @ da); den_l += float(dL[(s, k, p)] @ dL[(s, k, p)]); den_k += float(dk @ dk)
                        pic += float((z(s, f"E8_k{k}p{p}S") - z(s, f"E6_k{k}p{p}")) @ da)
                        acc[j:j + 6] = da; j += 6
                per_scene.append(acc)
            P = np.array(per_scene)
            out[f"L{L}_{rn}"] = dict(align=num / np.sqrt(den_a * den_l), size=float(np.sqrt(den_a / max(den_k, 1e-12))),
                                     cons=float(np.sum(P.mean(0) ** 2) / np.mean(np.sum(P ** 2, 1))), pic=pic / max(den_a, 1e-12))
    json.dump(dict(note="post hoc P0; head-averaged block scores of pictures 1-6", cells=out), open(f"{V}/research/ces/outputs/s855_rows.json", "w"), indent=1)
    print("align with the answer-side change (rows x layers)")
    print("        " + " ".join(f"{L:>6s}" for L in LAYERS))
    for rn in ROWS:
        print(f"{rn:9s}" + " ".join(f"{out[f'L{L}_{rn}']['align']:+6.2f}" for L in LAYERS))
    print("size relative to the ordinal change at the same row and layer")
    for rn in ROWS:
        print(f"{rn:9s}" + " ".join(f"{out[f'L{L}_{rn}']['size']:6.2f}" for L in LAYERS))
    print("consistency across scenes")
    for rn in ROWS:
        print(f"{rn:9s}" + " ".join(f"{out[f'L{L}_{rn}']['cons']:6.2f}" for L in LAYERS))
    print("share kept by pictures 7-8 at the 6-picture geometry")
    for rn in ROWS:
        print(f"{rn:9s}" + " ".join(f"{out[f'L{L}_{rn}']['pic']:6.2f}" for L in LAYERS))


if __name__ == "__main__":
    main()
