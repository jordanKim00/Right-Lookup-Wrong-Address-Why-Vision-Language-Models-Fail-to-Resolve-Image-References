"""S787 27B: the registered S658 f rule on the clean re-measure (exp/s787_analyze.s658 unchanged)."""
import sys
sys.path.insert(0, "/vlm/exp")
import s787_analyze as A
o = A.s658("s787_27b")
print(f"n {o['n']} acc " + " ".join(f"{k} {v:.2f}" for k, v in o["acc"].items()) + " | surf " + " ".join(f"{k} {v:.2f}" for k, v in o["surf"].items()))
print(f"f1 {o['f_QRP1']} · f2 {o['f_QRP2']} -> verdict {o['verdict']} | QFP hold {o['hold']}")
