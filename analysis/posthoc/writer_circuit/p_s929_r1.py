"""Descriptive (S929): the read-rows-only restriction R1 for the pins, with the same comparison as C2 (max rate error and run agreement)."""
import importlib.util, json, sys
sys.argv = [sys.argv[0], sys.argv[1], sys.argv[2], "/tmp/_s929_r1.json"]
spec = importlib.util.spec_from_file_location("s929a", "/vlm/exp/s929_analyze.py"); A = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(A)
except SystemExit:
    pass
cp = A.compare(tuple(A.PINS), rv="R1")
print(sys.argv[1], "pins R1: max err", round(cp["max_err"]["R1"], 3), "vs R0", round(cp["max_err"]["R0"], 3), "| agree", round(cp["agree"]["R1"], 3), "vs", round(cp["agree"]["R0"], 3))
