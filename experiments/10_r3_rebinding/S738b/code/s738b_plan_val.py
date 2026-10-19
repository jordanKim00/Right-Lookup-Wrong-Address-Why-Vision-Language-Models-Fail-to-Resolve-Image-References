"""S738b: the registered S724 plan analysis (exp/s724_plan.py main) on the selection split and the de-duplicated confirmation split,
written to its own file so that the registered record results/r3/s724_plan.json stays as it was.
Usage: python s738b_plan_val.py SEL:VAL [...]"""
import json, os, sys
sys.path.insert(0, "/vlm/exp")
import s724_plan as P

OUT = "/vlm/research/ces/outputs/r3_lure_latest/s738b_plan_val.json"
out = json.load(open(OUT)) if os.path.exists(OUT) else {}
for pair in sys.argv[1:]:
    sel, val = pair.split(":")
    out[sel] = P.main(sel, val)
json.dump(out, open(OUT, "w"), indent=1, default=float)
print("written", OUT)
