"""Post hoc (10-22, not registered): is S920 / S922's address-only R³ run (R3A: the ordinal rewritten k -> t, the END frame kept) the same
computation as the natural END(t) question?  Per scene and k with END(t) in the run's natural set: P(x(R3A_k) = x(N_E t)) against
P(x(R3A_k) = x(N_E k)), P(x(R3A_k) = x(St)), the chance level of two independent draws from the cell's answer distributions, and the
agreement of the L47 anchors."""
import json, collections
import numpy as np


def anchor(q):
    q = np.asarray(q); am = [int(np.argmax(q[i])) + 1 for i in (2, 5, 9)]
    c = collections.Counter(am).most_common(); return c[0][0] if c[0][1] >= 2 else am[2]


out = {}
for tag in ("s920A_27b", "s920B_27b", "s922_27b"):
    J = json.load(open(f"/vlm/results/r3/s920_{tag}_r3circuit.json")); N = J["N"]; rows = J["rows"]
    for k in J["ks"]:
        t = N + 1 - k
        if f"N_E{t}" not in rows[0]["land"]:
            continue
        a = [r["land"][f"R3A_{k}"] for r in rows]; e_t = [r["land"][f"N_E{t}"] for r in rows]; e_k = [r["land"][f"N_E{k}"] for r in rows]
        st = [r["land"][f"St_{k}"] for r in rows]
        ca, ce = collections.Counter(a), collections.Counter(e_t); n = len(rows)
        chance = sum((ca[v] / n) * (ce[v] / n) for v in ca)
        anc_a = [anchor(r["mm"][f"R3A_{k}"]["47"]) for r in rows]; anc_t = [anchor(r["mm"][f"N_E{t}"]["47"]) for r in rows]
        out[f"{tag}_k{k}"] = dict(N=N, k=k, t=t, same_as_END_t=float(np.mean([x == y for x, y in zip(a, e_t)])),
                                  same_as_END_k=float(np.mean([x == y for x, y in zip(a, e_k)])), same_as_START_t=float(np.mean([x == y for x, y in zip(a, st)])),
                                  chance_END_t=float(chance), L47_same_as_END_t=float(np.mean([x == y for x, y in zip(anc_a, anc_t)])))
agg = {m: float(np.mean([v[m] for v in out.values()])) for m in ("same_as_END_t", "same_as_END_k", "same_as_START_t", "chance_END_t", "L47_same_as_END_t")}
json.dump(dict(cells=out, mean=agg), open("/vlm/research/ces/outputs/writer_circuit_2026-10-20/main/p_s920_addronly_is_endt.json", "w"), indent=1)
print(json.dumps(agg, indent=1))
for c, v in out.items():
    print(c, {m: round(x, 2) for m, x in v.items() if m not in ("N", "k", "t")})
