"""S851 descriptive follow-up (post hoc, NOT registered): late picture mass (rows B4 + answer) per arm and layer, and the margin of p per read row at
layers 51 / 55 (rows 0-9 = question mark, B0-B4, think, newline, think-end, answer), to see which part of the late queries the steering left
uncorrected.  Usage: python3 exp/p4_s851_desc.py  ->  research/ces/outputs/s851_desc.json"""
import json
import numpy as np
V = "/home/storage2/ai26269345/vlm"
d = json.load(open(f"{V}/results/r3/s851_s851_27b_path.json")); R = d["rows"]; PS = (3, 4, 5)
ARMS = ("E_EARLY", "S_EARLY", "FULLB", "PLUS16", "PLUSQ", "FULLBQE")
out = dict(note="post hoc, not registered", mass={}, row_margin={})
for a in ARMS:
    out["mass"][a] = {L: float(np.mean([np.mean([r["mass"][f"{a}_p{p}"][L][row] for row in (5, 9)]) for r in R for p in PS])) for L in ("47", "51", "55", "59", "63")}
    out["row_margin"][a] = {L: [float(np.mean([(lambda z: z[p - 1] - max(z[i] for i in range(len(z)) if i != p - 1))(r["qk"][f"{a}_p{p}"][L][row])
                                                for r in R for p in PS])) for row in range(10)] for L in ("51", "55")}
json.dump(out, open(f"{V}/research/ces/outputs/s851_desc.json", "w"), indent=1)
print(json.dumps(out["mass"])); print(json.dumps({a: {L: [round(x, 2) for x in v] for L, v in out["row_margin"][a].items()} for a in ("E_EARLY", "FULLB", "PLUS16")}))
