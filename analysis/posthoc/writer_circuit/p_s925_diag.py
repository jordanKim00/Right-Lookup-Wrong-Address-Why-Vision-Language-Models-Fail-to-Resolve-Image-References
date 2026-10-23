"""Post hoc (after S925's registered verdicts): where did the frozen MC miss at N7?  Per set (training and test) and arm: observed vs
predicted acc / stay / step, the anchor position relative to t (a - t), and the answer offset from the anchor (x - a) - pooled and per k."""
import collections, importlib.util, json, sys
import numpy as np
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("s924_fit", "/vlm/exp/s924_fit.py"); FIT = importlib.util.module_from_spec(spec); spec.loader.exec_module(FIT)
MAIN = "/vlm/research/ces/outputs/writer_circuit_2026-10-20/main"
PF = json.load(open(f"{MAIN}/s924_frozen_MC.json")); W = np.asarray(PF["MC"])
ARMS = dict(E=("B{k}_E", "B{k}_E", 1), B=("B{k}_E", "B{k}_E_BANDR", 0), PIN=("B{k}_PIN", "B{k}_PIN", 1), PINB=("B{k}_PIN", "B{k}_PIN_BANDR", 0),
            PINm=("B{k}_PINm", "B{k}_PINm", 1), PINp=("B{k}_PINp", "B{k}_PINp", 1), PINt=("B{k}_PINt", "B{k}_PINt", 1), PINtB=("B{k}_PINt", "B{k}_PINtB", 0))
out = {}
for tag in ["s913n7_27b", "s913n8_27b", "s914n9_27b", "s914n6_27b", "s914e8_27b", "s923A_27b", "s924_27b", "s925_27b"]:
    U = FIT.load(tag, ARMS)
    for u in U:
        u["p"] = FIT.predict(u, W)
    res = {}
    for arm in ARMS:
        us = [u for u in U if u["arm"] == arm]
        if not us:
            continue
        def rates(us):
            o = dict(acc=np.mean([u["x"] == u["t"] for u in us]), stay=np.mean([u["x"] == u["a"] for u in us]), step=np.mean([u["x"] == u["a"] - 1 for u in us]))
            p = dict(acc=np.mean([u["p"][u["t"]] for u in us]), stay=np.mean([u["p"][u["a"]] for u in us]),
                     step=np.mean([u["p"][u["a"] - 1] if u["a"] >= 2 else 0 for u in us]))
            return {r: [round(float(o[r]), 3), round(float(p[r]), 3)] for r in o}
        res[arm] = dict(n=len(us), pooled=rates(us),
                        a_minus_t=dict(sorted(collections.Counter(int(u["a"] - u["t"]) for u in us).items())),
                        x_minus_a=dict(sorted(collections.Counter(int(u["x"] - u["a"]) if 1 <= u["x"] <= u["N"] else 99 for u in us).items())),
                        sharp=round(float(np.mean([u["m"] for u in us])), 4),
                        by_k={k: rates([u for u in us if u["k"] == k]) for k in sorted(set(u["k"] for u in us))},
                        a_minus_t_by_k={k: dict(sorted(collections.Counter(int(u["a"] - u["t"]) for u in us if u["k"] == k).items())) for k in sorted(set(u["k"] for u in us))})
    out[tag] = res
    e = res["E"]
    print(tag, "E n", e["n"], "pooled [obs, MC]", e["pooled"], "a-t", e["a_minus_t"], "sharp", e["sharp"])
json.dump(out, open(f"{MAIN}/p_s925_diag.json", "w"), indent=1, default=float)
print("->", f"{MAIN}/p_s925_diag.json")
