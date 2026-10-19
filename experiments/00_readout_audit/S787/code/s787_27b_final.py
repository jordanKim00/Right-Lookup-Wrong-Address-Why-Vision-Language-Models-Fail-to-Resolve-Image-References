"""S787 27B final (registered analysis rules unchanged, exp/s787_analyze.py functions): S652 fit (I0-I4), S658 f rule, S736 checks
(FRAME_NEEDED_NEAR, FRAME_SEMANTIC, FRAME_NEEDED_ALL for 27B) and the re-reflection = composition law on the 27B cells."""
import json, sys
sys.path.insert(0, "/vlm/exp")
import s787_analyze as A
import s736_analyze as A736
rep = dict(fit=A.fit({"27b": ("s787_27b", "s787_27b", "s787_27b")})["27b"], s658=A.s658("s787_27b"), s736=A736.analyze("s787_27b"))
cells = A.reflect_cells(json.load(open("/vlm/results/r3/s736_s787_27b_frame.json"))["rows"], "S787/S736", "Qwen3.5-27B", by_n=True)
inner = [c for c in cells if c["t"] not in (1, c["N"])]
rep["reflect"] = dict(cells=cells, all=A.law(cells), inner=A.law(inner))
for i, v in sorted(rep["fit"].items()):
    print("fit", i, " ".join(f"{k} {v[k]}" for k in v))
o = rep["s658"]; print(f"S658 f1 {o['f_QRP1']['f']:.2f} {o['f_QRP1']['ci']} · f2 {o['f_QRP2']['f']:.2f} {o['f_QRP2']['ci']} -> {o['verdict']}")
print(f"S736 checks {rep['s736']['checks']}")
print(f"reflect all {rep['reflect']['all']} · inner {rep['reflect']['inner']}")
json.dump(rep, open("/vlm/results/r3/s787_27b_final.json", "w"), indent=1, default=float)
