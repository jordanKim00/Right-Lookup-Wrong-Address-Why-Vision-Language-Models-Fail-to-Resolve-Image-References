"""27B head sets on one footing (saved S705a arrays, no model call; reply to the mock review, 2026-10-11).
A = the 73 index-matching heads of S705a (START diagonal share >= .5 and END k = 2 target share >= .4, at the ans or tail1 row, discovery scenes).
B = the rule of the smaller models (exp/s720_heads.py): START diagonal share >= .5 at the ans or tail1 row, discovery scenes -- a superset of A.
Both are evaluated exactly as Figure fig_c1 (exp/figs/make_c_figs.qk_tables): owner-defined blocks, answer row, held-out scenes.
Writes /vlm/research/ces/outputs/r3_lure_latest/s705a_headsets_27b.json."""
import json, sys
import numpy as np
sys.path.insert(0, "/vlm/exp/figs"); sys.path.insert(0, "/vlm/exp")
N = 7
R = "/vlm/results/r3"


def lse(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True); m = np.where(np.isfinite(m), m, 0.)
    return np.squeeze(m, axis) + np.log(np.sum(np.exp(x - m), axis=axis))


D = f"{R}/s705a_qa_27b_qkanatomy"
meta = json.load(open(D + ".json")); z = np.load(D + ".npz"); M = meta["meta"]; ns = len(M); CAP = meta["CAP"]
disc, hold = range(ns // 2), range(ns // 2, ns)
B = set()
for L in CAP:
    for ri in (1, 2):                                                       # tail1, ans (as exp/s720_heads.py)
        vals = []
        for si in disc:
            for t in range(1, N + 1):
                key = f"{si}|S{t}m0|{L}|limg"
                if key not in z.files:
                    continue
                own = np.array(M[si]["owner"]); limg = z[key][ri].astype(np.float64)
                sc = np.stack([lse(limg[:, own == b]) for b in range(N)], -1)
                vals.append(np.exp(sc - lse(sc)[:, None])[:, t - 1])
        for h in np.nonzero(np.mean(vals, 0) >= .5)[0]:
            B.add((L, int(h)))
A = {tuple(x) for x in json.load(open(f"{R}/s705a_qa_27b_matchers.json"))["matchers"]}


def tables(heads, ri=2):
    S = np.zeros((N, N)); E = np.zeros((N, N)); cS = np.zeros(N); cE = np.zeros(N)
    for si in hold:
        own = np.array(M[si]["owner"])
        for fr, mat, cnt in (("S", S, cS), ("E", E, cE)):
            for q in range(1, N + 1):
                for (L, h) in heads:
                    key = f"{si}|{fr}{q}m0|{L}|limg"
                    if key not in z.files:
                        continue
                    sc = np.stack([lse(z[key][ri][h].astype(np.float64)[own == b]) for b in range(N)])
                    mat[q - 1] += np.exp(sc - lse(sc)); cnt[q - 1] += 1
    return S / cS[:, None], E / cE[:, None]


out = {"A_n": len(A), "B_n": len(B), "A_subset_of_B": A <= B, "B_minus_A": sorted(B - A),
       "B_layers": {int(L): sum(1 for l, _ in B if l == L) for L in sorted({l for l, _ in B})}}
for nm, H in (("A", sorted(A)), ("B", sorted(B))):
    S, E = tables(H)
    out[nm] = dict(start_diag=float(np.mean(np.diag(S))),
                   end_target={k: float(E[k - 1, N - k]) for k in range(1, N + 1)},
                   end_literal={k: float(E[k - 1, k - 1]) for k in range(1, N + 1)})
    print(nm, len(H), "START diag %.3f" % out[nm]["start_diag"], "| END target", {k: round(v, 3) for k, v in out[nm]["end_target"].items()},
          "| literal", {k: round(v, 3) for k, v in out[nm]["end_literal"].items()})
print("A subset of B:", out["A_subset_of_B"], "| B layers", out["B_layers"])
json.dump(out, open("/vlm/research/ces/outputs/r3_lure_latest/s705a_headsets_27b.json", "w"), indent=1)
