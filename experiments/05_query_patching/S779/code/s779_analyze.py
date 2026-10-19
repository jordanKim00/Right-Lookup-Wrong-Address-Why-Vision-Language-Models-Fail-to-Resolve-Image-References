"""S779 analysis, frozen with the pre-registration (knowledge/DECISIONS.md, 2026-10-14 14:3x) before any S779 data.
ta = N - 4.  P(x, b) = share answering image position b in that run's own scene; Pw(x, w) = share answering word w.
In B the address word is labsB[ta] (= A's image-N word) and the content word is A's image-ta word (now at B's image N); in B2 the address word is the
first extra word (B2's image ta) and the content word is again A's image-ta word.
I0         NOOP|E5_A = E5_A on >= .95 of scenes, and R3_OK: G = P(R3_A, ta) - P(E5_A, ta) >= .30 (the repair works on these prompts).
QADDR_V    (V = B47, ALL47, ALL59; read only if fixA(V) = P(E5_A<S:V, ta) - P(E5_A, ta) >= .15)  addrB = gain of B's address word, contB = gain of the
           content word in E5_B<S:V over E5_B; addrB2, contB2 the same in B2.  ADDRESS if addrB >= .15, addrB >= contB + .10, addrB2 >= .15 and
           addrB2 >= contB2 + .10;  CONTENT the mirror;  BOTH if not these and an address and a content gain are both >= .15 in B or in B2;  NEITHER if
           every gain < .15;  MIXED otherwise.
R3_NEC_V   (V = ALL47, ALL59) P(R3_A, ta) - P(R3_A<E5:V, ta) >= .5 G (reverting R3's queries to END(5)'s removes the repair).
R3_SUF59   P(E5_A<R3:ALL59, ta) - P(E5_A, ta) >= .5 G (R3's queries alone repair END(5)).
R3_ADDR    the QADDR rule on E5_B<R3:ALL59 and E5_B2<R3:ALL59 (read if R3_SUF59).
R3_PART    removed by reverting the image part (D, M) vs the text part (T) of the L35-59 attention output to END(5)'s: IMAGE if image >= .5 G and text
           < .5 G; TEXT the mirror; BOTH if both; NEITHER otherwise.
VISUAL_END on the scenes where E5_A answers an image other than ta (a = that image): follow_i = Pw(E5_rep_i, second extra word); VISUAL if follow_a >= .30,
           follow_a >= follow_j + .20 and follow_a >= follow_r + .20.
VISUAL_R3  on the scenes where R3_A answers ta: follow_i = Pw(R3_rep_i, second extra word); VISUAL if follow_j >= .30, follow_j >= follow_a + .20 and
           follow_j >= follow_r + .20.
Descriptive: answer distributions; the logit drop of the answered word when each image is replaced; answer changes under the reversed menu for E5 and
R3; R3's plan statistics (share of scenes it wrote; t, k_hat, N_hat); late-head block gains for the dissociation arms.
Usage: python s779_analyze.py TAG [TAG ...]
"""
import json, sys
import numpy as np


def rule(aB, cB, aB2, cB2):
    if aB >= .15 - 1e-9 and aB >= cB + .10 - 1e-9 and aB2 >= .15 - 1e-9 and aB2 >= cB2 + .10 - 1e-9:
        return "ADDRESS"
    if cB >= .15 - 1e-9 and cB >= aB + .10 - 1e-9 and cB2 >= .15 - 1e-9 and cB2 >= aB2 + .10 - 1e-9:
        return "CONTENT"
    if (aB >= .15 - 1e-9 and cB >= .15 - 1e-9) or (aB2 >= .15 - 1e-9 and cB2 >= .15 - 1e-9):
        return "BOTH"
    if max(aB, cB, aB2, cB2) < .15 - 1e-9:
        return "NEITHER"
    return "MIXED"


def cell(tag):
    D = f"/vlm/results/r3/s779_{tag}_handoff"
    d = json.load(open(D + ".json")); M = d["meta"]; N, ta = d["N"], d["ta"]; n = len(M)
    A_ = lambda m, k: m["ans"][k]["pos"]
    W_ = lambda m, k: m["ans"][k]["word"]
    P = lambda k, b: float(np.mean([A_(m, k) == b for m in M]))
    Pw = lambda k, f: float(np.mean([W_(m, k) == f(m) for m in M]))
    arms = list(M[0]["ans"])
    out = dict(tag=tag, N=N, n=n, checks={}, dist={k: {str(b): round(P(k, b), 3) for b in range(0, N + 1) if P(k, b) > 0} for k in arms})
    noop = float(np.mean([A_(m, "NOOP|E5_A") == A_(m, "E5_A") for m in M]))
    G = P("R3_A", ta) - P("E5_A", ta)
    wrote = float(np.mean([any(x["act"] or x["fonly"] for x in m["r3"]["A"]) for m in M]))
    out["i0"] = dict(noop=round(noop, 3), G=round(G, 3), r3_wrote=round(wrote, 3), E5=round(P("E5_A", ta), 3), R3=round(P("R3_A", ta), 3))
    out["checks"]["I0"] = bool(noop >= .95 - 1e-9 and G >= .30 - 1e-9)
    addrB = lambda m: m["labsB"][ta - 1]; cont = lambda m: m["labs"][ta - 1]; addrB2 = lambda m: m["extra"][0]
    gain = lambda arm, base, f: Pw(arm, f) - Pw(base, f)
    out["qaddr"] = {}
    for V in ("B47", "ALL47", "ALL59"):
        fx = P(f"E5_A<S:{V}", ta) - P("E5_A", ta)
        g = dict(fixA=round(fx, 3), addrB=round(gain(f"E5_B<S:{V}", "E5_B", addrB), 3), contB=round(gain(f"E5_B<S:{V}", "E5_B", cont), 3),
                 addrB2=round(gain(f"E5_B2<S:{V}", "E5_B2", addrB2), 3), contB2=round(gain(f"E5_B2<S:{V}", "E5_B2", cont), 3))
        out["qaddr"][V] = g
        out["checks"][f"QADDR_{V}"] = rule(g["addrB"], g["contB"], g["addrB2"], g["contB2"]) if fx >= .15 - 1e-9 else None
    for V in ("ALL47", "ALL59"):
        rm = P("R3_A", ta) - P(f"R3_A<E5:{V}", ta)
        out[f"r3_removed_{V}"] = round(rm, 3)
        out["checks"][f"R3_NEC_{V}"] = bool(G > 0 and rm >= .5 * G - 1e-9)
    suf = P("E5_A<R3:ALL59", ta) - P("E5_A", ta)
    out["r3_suf"] = round(suf, 3); out["checks"]["R3_SUF59"] = bool(G > 0 and suf >= .5 * G - 1e-9)
    g = dict(addrB=round(gain("E5_B<R3:ALL59", "E5_B", addrB), 3), contB=round(gain("E5_B<R3:ALL59", "E5_B", cont), 3),
             addrB2=round(gain("E5_B2<R3:ALL59", "E5_B2", addrB2), 3), contB2=round(gain("E5_B2<R3:ALL59", "E5_B2", cont), 3))
    out["r3_addr"] = g
    out["checks"]["R3_ADDR"] = rule(g["addrB"], g["contB"], g["addrB2"], g["contB2"]) if out["checks"]["R3_SUF59"] else None
    ri = P("R3_A", ta) - P("R3_A<E5:IMG", ta); rt = P("R3_A", ta) - P("R3_A<E5:TXT", ta)
    out["r3_part"] = dict(image=round(ri, 3), text=round(rt, 3))
    hi, ht = ri >= .5 * G - 1e-9, rt >= .5 * G - 1e-9
    out["checks"]["R3_PART"] = ("BOTH" if hi and ht else "IMAGE" if hi else "TEXT" if ht else "NEITHER") if G > 0 else None
    x1 = lambda m: m["extra"][1]
    errs = [m for m in M if m["p3"]["a_is_answer"]]
    fol = lambda pre, sub, i: float(np.mean([W_(m, f"{pre}_rep{i}") == x1(m) for m in sub])) if sub else float("nan")
    fe = {i: round(fol("E5", errs, i), 3) for i in "jar"}
    out["visual_end"] = dict(n=len(errs), **fe)
    out["checks"]["VISUAL_END"] = bool(errs and fe["a"] >= .30 - 1e-9 and fe["a"] >= fe["j"] + .20 - 1e-9 and fe["a"] >= fe["r"] + .20 - 1e-9)
    oks = [m for m in M if A_(m, "R3_A") == ta]
    fr = {i: round(fol("R3", oks, i), 3) for i in "jar"}
    out["visual_r3"] = dict(n=len(oks), **fr)
    out["checks"]["VISUAL_R3"] = bool(oks and fr["j"] >= .30 - 1e-9 and fr["j"] >= fr["a"] + .20 - 1e-9 and fr["j"] >= fr["r"] + .20 - 1e-9)
    ws = lambda m: m["labs"] + m["extra"]
    def drop(pre, sub, i, base):
        v_ = []
        for m in sub:
            w = W_(m, base); j_ = ws(m).index(w)
            v_.append(m["ans"][base]["ll"][j_] - m["ans"][f"{pre}_rep{i}"]["ll"][j_])
        return round(float(np.mean(v_)), 3) if v_ else None
    out["logit_drop"] = dict(END={i: drop("E5", errs, i, "E5_A") for i in "jar"}, R3={i: drop("R3", oks, i, "R3_A") for i in "jar"})
    out["menu_reversal_change"] = dict(END=round(float(np.mean([W_(m, "E5_rev") != W_(m, "E5_A") for m in M])), 3),
                                       R3=round(float(np.mean([W_(m, "R3_rev") != W_(m, "R3_A") for m in M])), 3))
    print(f"[{tag}] N {N} · n {n} · checks {out['checks']} · I0 {out['i0']}")
    print(f"   query address vs content {out['qaddr']}\n   R3 removed {out.get('r3_removed_ALL47')} / {out.get('r3_removed_ALL59')} · R3 sufficiency {out['r3_suf']} · R3 address {g} · part {out['r3_part']}")
    print(f"   visual END {out['visual_end']} · visual R3 {out['visual_r3']} · logit drop {out['logit_drop']} · menu reversal {out['menu_reversal_change']}")
    return out


if __name__ == "__main__":
    json.dump({t: cell(t) for t in sys.argv[1:]}, open("/vlm/results/r3/s779_analysis.json", "w"), indent=1)
