import json, numpy as np, collections
for tag in ("s913n7_27b", "s913n8_27b", "s914n9_27b", "s914n6_27b", "s914e8_27b"):
    J = json.load(open(f"/vlm/results/r3/s913_{tag}_collect.json")); N = J["N"]; k = N; lm = J["late_mean"]["51"]
    an = [r["anchor"][str(k)] for r in J["rows"]]; xs = [r["land"][f"N_E{k}"] for r in J["rows"]]
    xb = [r["land"].get(f"B{k}_E_BANDR") for r in J["rows"]]
    l51 = [int(np.argmax(np.asarray(r["zl"][f"B{k}_E"]["51"], float)[:, lm].mean(1).mean(0))) + 1 for r in J["rows"] if f"B{k}_E" in r["zl"]]
    print(f"{tag} {J.get('phrase')} N{N} k{k}: anchors {dict(collections.Counter(an).most_common(4))} answers {dict(collections.Counter(xs).most_common(4))} "
          f"BANDR {dict(collections.Counter(xb).most_common(3))} L51 {dict(collections.Counter(l51).most_common(3))}")
