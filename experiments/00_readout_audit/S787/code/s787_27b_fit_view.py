"""S787 27B: the registered S652 fit (I0-I4) on the clean re-measure, using exp/s787_analyze.fit unchanged (descriptive print)."""
import sys
sys.path.insert(0, "/vlm/exp")
import s787_analyze as A
res = A.fit({"27b": ("s787_27b", "s787_27b", "s787_27b")})["27b"]
for i, v in sorted(res.items()):
    print(i, " ".join(f"{k} {v[k]}" for k in v))
