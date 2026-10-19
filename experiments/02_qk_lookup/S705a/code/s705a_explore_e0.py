"""E0 (exploratory, saved S705a arrays, no model call): is "every matcher is sharpest one step from the end" a product of selecting matchers
with an END k = 2 criterion?  Head-rows are selected on discovery scenes (0-14) by (1) the S705a MATCHER rule (START diagonal >= .5 and
END k = 2 target share >= .4, ans / tail1 rows), (2) START diagonal >= .5 only (ans / tail1), (3) no rule (every head x row at the captured
layers).  On held-out scenes (15-29), per head-row and step d = k - 1 in {0..3}: S(d) = mean block share of the block d steps before the
visible end a (a = 7, 6, 5 for m = 0, 1, 2; cells where that block is the literal block k are excluded) and M(d) = its block-score margin
over the best other visible block."""
import json
import numpy as np
N = 7
def lse(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True); m = np.where(np.isfinite(m), m, 0.)
    return np.squeeze(m, axis) + np.log(np.sum(np.exp(x - m), axis=axis))
D = "/vlm/results/r3/s705a_qa_27b_qkanatomy"
meta = json.load(open(D + ".json")); z = np.load(D + ".npz"); M = meta["meta"]; CAP = meta["CAP"]; ns = len(M)
disc, hold = list(range(ns // 2)), list(range(ns // 2, ns))
cache = {}
def scores(si, key, li):
    k = (si, key, li)
    if k not in cache:
        own = np.array(M[si]["owner"]); limg = z[f"{si}|{key}|{li}|limg"].astype(np.float64)      # [rows, H, P]
        cache[k] = np.stack([lse(limg[:, :, own == b]) for b in range(N)], -1)                    # [rows, H, N]
    return cache[k]
def share(sc, vis):
    s = sc[..., :vis]; return np.exp(s - lse(s)[..., None])
H = 24; ROWS = (0, 1, 2)
# discovery statistics for selection
sdiag, e2 = {}, {}
for li in CAP:
    sd = np.mean([share(scores(si, f"S{t}m0", li), N)[:, :, t - 1] for si in disc for t in range(1, N + 1)], 0)   # [rows, H]
    ee = np.mean([share(scores(si, "E2m0", li), N)[:, :, 5] for si in disc], 0)
    for ri in ROWS:
        for h in range(H):
            sdiag[(li, h, ri)] = sd[ri, h]; e2[(li, h, ri)] = ee[ri, h]
sets = {"END-selected (S705a)": [x for x in sdiag if x[2] in (1, 2) and sdiag[x] >= .5 and e2[x] >= .4],
        "START-only": [x for x in sdiag if x[2] in (1, 2) and sdiag[x] >= .5],
        "all head-rows": list(sdiag)}
def profile(x):
    li, h, ri = x; S, Mg = {}, {}
    for d in range(4):
        vs, ms = [], []
        for m in (0, 1, 2):
            a = N - m; k = d + 1; t = a - d
            if t < 1 or t == k: continue
            for si in hold:
                key = f"E{k}m{m}"
                if f"{si}|{key}|{li}|limg" not in z.files: continue
                sc = scores(si, key, li)[ri, h, :a]
                vs.append(share(sc, a)[t - 1]); ms.append(sc[t - 1] - np.max(np.delete(sc, t - 1)))
        S[d] = float(np.mean(vs)); Mg[d] = float(np.mean(ms))
    return S, Mg
for name, hs in sets.items():
    P = [profile(x) for x in hs]
    S = np.array([[p[0][d] for d in range(4)] for p in P]); Mg = np.array([[p[1][d] for d in range(4)] for p in P])
    uniq = len({(x[0], x[1]) for x in hs})
    best = np.argmax(S[:, 1:], 1) + 1; strong = S[:, 1:].max(1) >= .3
    print(f"== {name}: {len(hs)} head-rows ({uniq} distinct heads)")
    print("   mean S(d):", " ".join(f"d{d} {S[:, d].mean():.2f}" for d in range(4)), "| mean margin:", " ".join(f"d{d} {Mg[:, d].mean():+.2f}" for d in range(4)))
    print("   S(d) >= .5:", " ".join(f"d{d} {int((S[:, d] >= .5).sum())}" for d in range(4)), "| margin > 0:", " ".join(f"d{d} {int((Mg[:, d] > 0).sum())}" for d in range(4)))
    print("   best d>=1 among head-rows with max S >= .3:", {d: int(((best == d) & strong).sum()) for d in (1, 2, 3)}, "of", int(strong.sum()))
    i2 = np.argsort(-(S[:, 2] - S[:, 1]))[:5]
    print("   largest S(2) - S(1):", [(f"L{hs[i][0]}h{hs[i][1]}r{hs[i][2]}", round(S[i, 1], 2), round(S[i, 2], 2), round(S[i, 3], 2)) for i in i2])
    if len(hs) > 2:
        print(f"   corr S(1),S(2) {np.corrcoef(S[:, 1], S[:, 2])[0, 1]:.2f} · corr S(2),S(3) {np.corrcoef(S[:, 2], S[:, 3])[0, 1]:.2f}")
# head-rows whose best step (d >= 1) is d = 3 in the unselected set
hs = sets["all head-rows"]; P = [profile(x) for x in hs]
S = np.array([[p[0][d] for d in range(4)] for p in P])
for i in np.where((np.argmax(S[:, 1:], 1) + 1 == 3) & (S[:, 1:].max(1) >= .3))[0]:
    print("   best at d=3:", f"L{hs[i][0]}h{hs[i][1]}r{hs[i][2]}", "S(d0..d3)", [round(v, 2) for v in S[i]], "START diag", round(float(sdiag[hs[i]]), 2))
