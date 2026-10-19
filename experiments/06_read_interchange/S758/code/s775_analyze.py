"""S775 analysis, frozen with the pre-registration (knowledge/DECISIONS.md, 2026-10-14 06:0x) before any S775 data.
r(x) = late-head alignment (S771 / S773 / S774 head set H_late, answer row, block 5): (s(x) - s(E5)) / (s(F5P1) - s(E5)).  e(V) = r(PP_V) - r(PP_M42).
I0         NOOP_E5 agrees with END(5) on >= .95 of scenes, and G = e(NAT) >= .50 (S774 LAYERS = G2 replicated: F5P1's own L48-50 DeltaNet differences).
REPL_TAIL  e(FULL) >= .5 G (the tail rows as the source, S774 SOURCE = TAIL, replicated with L48-50 only).
ROUTE      (if REPL_TAIL) F = e(FULL): CONV if e(CONV) >= .5 F and e(MEM) < .5 F; MEM if e(MEM) >= .5 F and e(CONV) < .5 F; BOTH if both; JOINT if neither.
WINDOW     (if REPL_TAIL) LAST3 if e(LAST3) >= .5 F and e(EARLY) < .5 F; EARLY if the reverse; BOTH if both; JOINT if neither.
Descriptive: every arm's answer distribution, r and phi (answer-5 recovery) for every arm.
Usage: python s775_analyze.py TAG
"""
import json, sys
import numpy as np

HL = {51: [2, 3, 23, 4, 22, 12], 55: [13, 14, 2, 15, 0, 16], 59: [21, 3, 20, 1, 5, 8]}


def sm(sc):
    x = sc - sc.max(-1, keepdims=True); e = np.exp(x); return e / e.sum(-1, keepdims=True)


def verdict(a, b, F):
    ka, kb = a[1] >= .5 * F - 1e-9, b[1] >= .5 * F - 1e-9
    return a[0] if ka and not kb else b[0] if kb and not ka else "BOTH" if ka and kb else "JOINT"


def analyze(tag):
    D = f"/vlm/results/r3/s775_{tag}_dnroute"
    d = json.load(open(D + ".json")); M = d["meta"]; n = len(M); z = np.load(D + ".npz")
    A_ = lambda m, k: m["ans"][k]["pos"]
    P = lambda k, b: float(np.mean([A_(m, k) == b for m in M]))
    arms = list(M[0]["ans"])
    out = dict(n=n, checks={}, dist={k: {str(b): round(P(k, b), 3) for b in range(0, 8) if P(k, b) > 0} for k in arms})
    nh = z[f"{M[0]['si']}|E5|{d['RB']['late'][0]}"].shape[1]
    HS = HL if d["nl"] == 64 else {li: list(range(nh)) for li in d["RB"]["late"]}
    sH = lambda k: float(np.mean([np.mean([sm(z[f"{m['si']}|{k}|{li}"].astype(np.float64))[1][h][4] for li, H in HS.items() for h in H]) for m in M]))
    e5, f5 = sH("E5"), sH("F5P1")
    r = {k: round((sH(k) - e5) / (f5 - e5), 3) if abs(f5 - e5) > 1e-9 else None for k in arms}
    out["r_late"] = r
    p5e, p5f = P("E5", 5), P("F5P1", 5)
    out["phi"] = {k: round((P(k, 5) - p5e) / (p5f - p5e), 3) if p5f > p5e else None for k in arms}
    e = {v: (r[f"PP_{v}"] or 0) - (r["PP_M42"] or 0) for v in ("NAT", "FULL", "CONV", "MEM", "LAST3", "EARLY")}
    out["e"] = {k: round(x, 3) for k, x in e.items()}
    noop = float(np.mean([A_(m, "NOOP_E5") == A_(m, "E5") for m in M]))
    out["i0"] = dict(noop=round(noop, 3), G=round(e["NAT"], 3))
    out["checks"]["I0"] = bool(noop >= .95 - 1e-9 and e["NAT"] >= .50 - 1e-9)
    out["checks"]["REPL_TAIL"] = bool(e["FULL"] >= .5 * e["NAT"] - 1e-9)
    if out["checks"]["REPL_TAIL"]:
        F = e["FULL"]
        out["checks"]["ROUTE"] = verdict(("CONV", e["CONV"]), ("MEM", e["MEM"]), F)
        out["checks"]["WINDOW"] = verdict(("LAST3", e["LAST3"]), ("EARLY", e["EARLY"]), F)
    else:
        out["checks"]["ROUTE"] = None; out["checks"]["WINDOW"] = None
    print(f"[{tag}] n {n} · checks {out['checks']} · I0 {out['i0']} · e {out['e']}")
    print(f"   r_late {r}\n   phi {out['phi']}")
    return out


if __name__ == "__main__":
    json.dump({sys.argv[1]: analyze(sys.argv[1])}, open("/vlm/results/r3/s775_analysis.json", "w"), indent=1)
