"""Post hoc (10-19, mock-review W1; not registered): does the late update stop when the pinned early position equals the target counted from
the images actually shown, j_a = N_shown + 1 - k?  The registered stop tests (S844 / S846) varied only the stated count.  Saved runs only.
  S854-1 (registered 10-18; 30 scenes; N_shown {6, 8} x N_stated {6, 8} x k {3, 5} x pinned p {3, 4}): at p = 4 the same scenes reach
    j_a = p in (k 3, N_shown 6) and (k 5, N_shown 8) and miss it in the other N_shown.  Contrast: staying at p, reached minus not reached,
    same k and p, pooled over N_stated; difference-in-differences against the same N_shown swap at p = 3, where j_a never equals p.
  S846 (registered 10-17; 30 scenes x 4 stated counts; N_shown 8, k 4, so j_a = 5): staying at p = 5 against p = 4 and p = 6 (descriptive,
    confounded with p).
Scene bootstrap 4,000 (seed 4000).  Output: research/ces/outputs/p6_stop_visual_target.json.  Usage: python3 exp/p6_stop_visual_target.py"""
import ast, json
import numpy as np

V = "/home/storage2/ai26269345/vlm"
land = lambda r: ast.literal_eval(r["land"]) if isinstance(r["land"], str) else r["land"]

L = [land(r) for r in json.load(open(f"{V}/results/r3/s854a_s854a_27b_count.json"))["rows"]]
S = len(L)
stay = lambda k, p, na, ns, idx: np.array([L[i][f"E_s{ns}k{k}p{p}_a{na}"] == p for i in idx], float).mean()
cells = {f"k{k}_p{p}_Na{na}_Ns{ns}": dict(stay=stay(k, p, na, ns, range(S)), j_a=na + 1 - k, reached=(p == na + 1 - k))
         for k in (3, 5) for p in (3, 4) for na in (6, 8) for ns in (6, 8)}


def contrast(idx):
    m = lambda k, p, na: np.mean([stay(k, p, na, ns, idx) for ns in (6, 8)])
    a, a0 = m(3, 4, 6) - m(3, 4, 8), m(3, 3, 6) - m(3, 3, 8)          # k 3: reached at N_shown 6 (j_a 4)
    b, b0 = m(5, 4, 8) - m(5, 4, 6), m(5, 3, 8) - m(5, 3, 6)          # k 5: reached at N_shown 8 (j_a 4)
    return dict(k3=a, k5=b, pooled=(a + b) / 2, did=((a - a0) + (b - b0)) / 2)


pt = contrast(range(S))
rng = np.random.default_rng(4000)
bs = [contrast(rng.integers(0, S, S)) for _ in range(4000)]
ci = {k: [float(np.percentile([b[k] for b in bs], q)) for q in (2.5, 97.5)] for k in pt}
reached = float(np.mean([v["stay"] for v in cells.values() if v["reached"]]))
not_reached_p4 = float(np.mean([v["stay"] for k_, v in cells.items() if "_p4_" in k_ and not v["reached"]]))

R = json.load(open(f"{V}/results/r3/s846_s846_27b_swap.json"))["rows"]
s846 = {p: float(np.mean([land(r)[f"E_EARLY_p{p}"] == p for r in R])) for p in (3, 4, 5, 6)}

out = dict(note="post hoc, not registered; saved runs only", s854_1=dict(cells=cells, contrast=pt, ci=ci, scenes=S,
           p4_reached_mean=reached, p4_not_reached_mean=not_reached_p4),
           s846=dict(stay_by_p=s846, visual_target_p=5, n_runs=len(R)))
json.dump(out, open(f"{V}/research/ces/outputs/p6_stop_visual_target.json", "w"), indent=1, default=float)
print(f"S854-1 p=4 stay: reached {reached:.3f} vs not {not_reached_p4:.3f}; pooled {pt['pooled']:+.3f} {np.round(ci['pooled'], 3)}; "
      f"DiD {pt['did']:+.3f} {np.round(ci['did'], 3)}; k3 {pt['k3']:+.3f} {np.round(ci['k3'], 3)}; k5 {pt['k5']:+.3f} {np.round(ci['k5'], 3)}")
print("S846 stay by p (j_a = 5):", {p: round(v, 3) for p, v in s846.items()})
