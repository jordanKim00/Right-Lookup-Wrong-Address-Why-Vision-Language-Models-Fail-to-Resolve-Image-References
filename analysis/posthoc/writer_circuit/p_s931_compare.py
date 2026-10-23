"""Descriptive (S931; no verdict): the END circuit instruments side by side for pictures (S926 N8) and word lists (S931 N8), same N, arms,
predictor and readout.  Per k: natural END accuracy, the answer's offset from the target (x - t) and from the L47 anchor (x - a), the anchor's
offset (a - t); writer removal (E_BANDR: stay at a, accuracy); and from the gradients (no fitted parameter) the address-to-answer transfer:
mean centred T[q][q] (copy), T[q][q - 1] (one step back), and the band share 1 - <Dp, T> / <T, T>.
Usage: python3 p_s931_compare.py"""
import collections, importlib.util, json, sys
import numpy as np
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("s924_fit", "/vlm/exp/s924_fit.py"); FIT = importlib.util.module_from_spec(spec); spec.loader.exec_module(FIT)
SETS = dict(image=("/vlm/results/r3/s926_s926n8_27b_collect.json", "/vlm/results/r3/s928h_s926n8_27b_quad.json"),
            text=("/vlm/results/r3/s931_s931n8_27b_collect.json", "/vlm/results/r3/s931q_s931n8_27b_quad.json"))
out = {}
for name, (cf, qf) in SETS.items():
    C = json.load(open(cf)); L = json.load(open(qf)); N = C["N"]; KS = C["ks"]
    st = float(np.mean([r["land"][f"N_S{q}"] == q for r in C["rows"] for q in range(1, N + 1)]))
    res = dict(START_acc=st, k={})
    for k in KS:
        t = N + 1 - k; E = []; B = []
        for r in C["rows"]:
            if f"B{k}_E" not in r["land"]:
                continue
            a = FIT.anchor_of(r["qk47"][f"B{k}_E"]); x = r["land"][f"B{k}_E"]; xb = r["land"][f"B{k}_E_BANDR"]
            E.append((x, a)); B.append((xb, a))
        res["k"][k] = dict(n=len(E), acc=float(np.mean([x == t for x, a in E])), anchor_at_t=float(np.mean([a == t for x, a in E])),
                           x_minus_t=dict(collections.Counter(int(x - t) if x <= N else 99 for x, a in E).most_common(4)),
                           x_minus_a=dict(collections.Counter(int(x - a) if x <= N else 99 for x, a in E).most_common(4)),
                           bandr_stay=float(np.mean([xb == a for xb, a in B])), bandr_acc=float(np.mean([xb == t for xb, a in B])))
    tr = collections.defaultdict(lambda: dict(copy=[], step=[], share=[]))
    for rec in L["rows"]:
        for k_, o in rec["k"].items():
            T = np.array(o["T"])[:, :N]; Dp = np.array(o["Dp"])[:, :N]; Tc = T - T.mean(1, keepdims=True)
            tr[int(k_)]["copy"] += [Tc[q, q] for q in range(N)]; tr[int(k_)]["step"] += [Tc[q, q - 1] for q in range(1, N)]
            if (T * T).sum() > 0:
                tr[int(k_)]["share"].append(1 - (Dp * T).sum() / (T * T).sum())
    for k in KS:
        res["k"][k].update(T_copy=float(np.mean(tr[k]["copy"])), T_step=float(np.mean(tr[k]["step"])), band_share=float(np.median(tr[k]["share"])))
    out[name] = res
    print(f"== {name}  START {st:.3f}")
    for k in KS:
        d = res["k"][k]
        print(f"  k {k}: acc {d['acc']:.2f}  anchor=t {d['anchor_at_t']:.2f}  x-t {d['x_minus_t']}  x-a {d['x_minus_a']}  | BANDR stay {d['bandr_stay']:.2f} acc {d['bandr_acc']:.2f}"
              f"  | T copy {d['T_copy']:+.2f} step {d['T_step']:+.2f} band {d['band_share']:.2f}")
json.dump(out, open("/vlm/research/ces/outputs/writer_circuit_2026-10-20/main/p_s931_compare.json", "w"), indent=1)
