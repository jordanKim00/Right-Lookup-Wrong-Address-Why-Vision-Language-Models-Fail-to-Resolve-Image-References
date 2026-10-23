"""Post hoc (after S925's registered verdicts): the joint of the anchor's distance from the target (a - t) and the answer's move (x - a), per
writer flag, observed vs the frozen MC - training sets pooled, S924, S925.  Where does MC's acc deficit live?"""
import collections, importlib.util, json, sys
import numpy as np
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("s924_fit", "/vlm/exp/s924_fit.py"); FIT = importlib.util.module_from_spec(spec); spec.loader.exec_module(FIT)
MAIN = "/vlm/research/ces/outputs/writer_circuit_2026-10-20/main"
W = np.asarray(json.load(open(f"{MAIN}/s924_frozen_MC.json"))["MC"])
ARMS = dict(E=("B{k}_E", "B{k}_E", 1), B=("B{k}_E", "B{k}_E_BANDR", 0), PIN=("B{k}_PIN", "B{k}_PIN", 1), PINB=("B{k}_PIN", "B{k}_PIN_BANDR", 0),
            PINm=("B{k}_PINm", "B{k}_PINm", 1), PINp=("B{k}_PINp", "B{k}_PINp", 1), PINt=("B{k}_PINt", "B{k}_PINt", 1), PINtB=("B{k}_PINt", "B{k}_PINtB", 0))
SETS = dict(train=["s913n7_27b", "s913n8_27b", "s914n9_27b", "s914n6_27b", "s914e8_27b", "s923A_27b"], s924=["s924_27b"], s925=["s925_27b"])
out = {}
for name, tags in SETS.items():
    U = []
    for tag in tags:
        U += FIT.load(tag, ARMS)
    for u in U:
        u["p"] = FIT.predict(u, W)
    out[name] = {}
    for arm in ARMS:
        us = [u for u in U if u["arm"] == arm]
        if not us:
            continue
        tab = {}
        for d in sorted(set(int(u["a"] - u["t"]) for u in us)):
            g = [u for u in us if u["a"] - u["t"] == d]
            if len(g) < 8:
                continue
            row = dict(n=len(g), acc_obs=float(np.mean([u["x"] == u["t"] for u in g])), acc_MC=float(np.mean([u["p"][u["t"]] for u in g])))
            mv = collections.Counter(int(u["x"] - u["a"]) if 1 <= u["x"] <= u["N"] else 99 for u in g)
            row["move_obs"] = {str(o): round(c / len(g), 3) for o, c in sorted(mv.items())}
            pm = collections.defaultdict(float)
            for u in g:
                for b in range(1, u["N"] + 1):
                    pm[b - u["a"]] += u["p"][b] / len(g)
            row["move_MC"] = {str(o): round(v, 3) for o, v in sorted(pm.items()) if v >= .02}
            tab[str(d)] = row
        out[name][arm] = tab
for name in out:
    for arm in ("E", "B", "PIN", "PINm", "PINp", "PINt"):
        if arm not in out[name]:
            continue
        print(f"== {name} {arm}")
        for d, r in out[name][arm].items():
            print(f"  a-t={d:>3s} n={r['n']:4d} acc {r['acc_obs']:.2f}/{r['acc_MC']:.2f}  obs {r['move_obs']}  MC {r['move_MC']}")
json.dump(out, open(f"{MAIN}/p_s925_joint.json", "w"), indent=1)
