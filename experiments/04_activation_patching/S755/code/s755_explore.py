"""S755 exploratory (after the registered analysis; not a verdict): answer distributions per form, and whether cross-form transplants make the
recipient follow the donor's actual natural answer in the same scene (the registered ABSTRACT_CODE scored only image 6)."""
import json, sys
import numpy as np
d = json.load(open(f"/vlm/results/r3/s755_{sys.argv[1]}_forms.json")); M = d["meta"]; LAY = d["LAY"]
L = lambda k: np.array([m["land"][k] for m in M])
for k in ("F1|1", "F2|1", "F3|1", "F4|1", "F5|1", "F1|2", "F2|2", "F3|2", "F4|2", "F1|3", "F2|3", "F3|3", "F1|4", "F2|4", "F3|4"):
    l = L(k); print(f"{k:5s} " + " ".join(f"{b}:{np.mean(l == b):.2f}" for b in range(1, 8)))
rec = L("F1|4")
for src in ("F1|1", "F2|1", "F3|1", "F4|1"):
    don = L(src)
    for l in LAY:
        x = L(f"X{l}|{src}>F1|4")
        print(f"X{l} {src}->F1|4: follow donor's answer {np.mean(x == don):.2f} (baseline recipient natural = donor answer {np.mean(rec == don):.2f}) · on 6 {np.mean(x == 6):.2f} · on 5 {np.mean(x == 5):.2f}")
