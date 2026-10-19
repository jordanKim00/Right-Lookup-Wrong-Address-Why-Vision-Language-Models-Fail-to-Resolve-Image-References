import json, sys, numpy as np
sys.path.insert(0, "/vlm/exp")
import s787_analyze as A
res = A.fit({"27b": ("s787_27b", "s787_27b", "s787_27b")})["27b"]
for i, v in sorted(res.items()):
    print(i, {k: v[k] for k in v})
rows = json.load(open("/vlm/results/s549_endshift_s787_27b.json"))["rows"]
rng = np.random.default_rng(0)
def boot(sel, f, B=4000):
    vals = [np.mean([f(sel[i]) for i in rng.integers(0, len(sel), len(sel))]) for _ in range(B)]
    return [round(float(np.percentile(vals, 2.5)), 2), round(float(np.percentile(vals, 97.5)), 2)]
clean = [r for r in rows if r["j"] != r["k"] and r["j"] - 1 != r["k"] and r["j"] + 1 != r["k"]]
print("clean cells", len(clean), "by (N,k):", sorted({(r["N"], r["k"]) for r in clean}))
for k in sorted({r["k"] for r in clean}):
    s = [r for r in clean if r["k"] == k]
    f = lambda a, off: float(np.mean([r[f"R|{a}"] == r["j"] + off for r in s]))
    print(f" k{k} n{len(s)}: none j-1 {f('none',-1):.2f} j {f('none',0):.2f} j+1 {f('none',1):.2f} | m1 j-1 {f('m1',-1):.2f} j {f('m1',0):.2f} j+1 {f('m1',1):.2f} | p1 j-1 {f('p1',-1):.2f} j {f('p1',0):.2f} j+1 {f('p1',1):.2f}")
for nm, sel in (("near k=2", [r for r in clean if r["k"] == 2]), ("far k>=3", [r for r in clean if r["k"] >= 3])):
    if not sel: continue
    dm = lambda r: (r["R|m1"] == r["j"] - 1) - (r["R|none"] == r["j"] - 1)
    dp = lambda r: (r["R|p1"] == r["j"] + 1) - (r["R|none"] == r["j"] + 1)
    print(f" {nm} (n {len(sel)}): m1 shift to j-1 {np.mean([dm(r) for r in sel]):+.2f} {boot(sel, dm)} · p1 shift to j+1 {np.mean([dp(r) for r in sel]):+.2f} {boot(sel, dp)}")
