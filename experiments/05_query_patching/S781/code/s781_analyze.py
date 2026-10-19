"""S781 analysis, frozen with the pre-registration (knowledge/DECISIONS.md, 2026-10-14 15:1x) before any S781 data.
ta = N - 4.  In B the address word is labsB[ta] (= A's image-N word) and the content word is A's image-ta word (now at B's image N); in B2 the address
word is the first extra word and the content word is again A's image-ta word.  gain(x, w) = share of x answering w minus that of the natural END(5)
of the same scene.
I0            NOOP|E5_B = E5_B on >= .95 of scenes, and R3 run afresh answers the address word in B and in B2 on >= .90 of scenes.
POS_R3_V      (V = ALL39, ALL47, ALL59) gain(E5_B<R3_B:V, addressB) >= .30 and gain(E5_B2<R3_B2:V, addressB2) >= .30: queries formed by R3 in the same
              scene point that scene's END(5) to its own image ta.
POS_S_V       the same for START(ta)'s queries from the same scene (E5_B<S_B:V, E5_B2<S_B2:V).
DONOR_CONTENT (ALL59) in E5_B<R3_A:ALL59 and E5_B2<R3_A:ALL59 the content-word gain exceeds the address-word gain by >= .10 (S779 replicated).
FIXED_PLAN    A's R3 write plan applied unchanged to B and B2 answers the address word within .05 of R3 run afresh there, and on >= .90.
STABLE_MAP    POS_R3_ALL59 and DONOR_CONTENT and FIXED_PLAN: R3 maps the resolved position to that scene's content; queries carried to another scene
              carry the donor scene's content.
Descriptive: every arm's word classes (address / content / extra / other), the depth profile of the gains, R3 plans per scene.
Usage: python s781_analyze.py TAG [TAG ...]
"""
import json, os, sys
import numpy as np


def cell(tag):
    d = json.load(open(f"/vlm/results/r3/s781_{tag}_posctl.json")); M = d["meta"]; N, ta = d["N"], d["ta"]; n = len(M)
    W_ = lambda m, k: m["ans"][k]["word"]
    A_ = lambda m, k: m["ans"][k]["pos"]
    Pw = lambda k, f: float(np.mean([W_(m, k) == f(m) for m in M]))
    adB = lambda m: m["labsB"][ta - 1]; adB2 = lambda m: m["extra"][0]; cont = lambda m: m["labs"][ta - 1]
    g = lambda arm, base, f: Pw(arm, f) - Pw(base, f)
    out = dict(tag=tag, N=N, n=n, checks={})
    arms = list(M[0]["ans"])
    out["words"] = {k: dict(addrB=round(Pw(k, adB), 3), addrB2=round(Pw(k, adB2), 3), cont=round(Pw(k, cont), 3)) for k in arms}
    noop = float(np.mean([A_(m, "NOOP|E5_B") == A_(m, "E5_B") for m in M]))
    r3b, r3b2 = Pw("R3_B", adB), Pw("R3_B2", adB2)
    out["i0"] = dict(noop=round(noop, 3), R3_B_addr=round(r3b, 3), R3_B2_addr=round(r3b2, 3))
    out["checks"]["I0"] = bool(noop >= .95 - 1e-9 and r3b >= .90 - 1e-9 and r3b2 >= .90 - 1e-9)
    gains = {}
    for V in ("ALL39", "ALL47", "ALL59"):
        x = dict(R3same_B=g(f"E5_B<R3_B:{V}", "E5_B", adB), R3same_B2=g(f"E5_B2<R3_B2:{V}", "E5_B2", adB2),
                 Ssame_B=g(f"E5_B<S_B:{V}", "E5_B", adB), Ssame_B2=g(f"E5_B2<S_B2:{V}", "E5_B2", adB2),
                 R3A_B_addr=g(f"E5_B<R3_A:{V}", "E5_B", adB), R3A_B_cont=g(f"E5_B<R3_A:{V}", "E5_B", cont),
                 R3A_B2_addr=g(f"E5_B2<R3_A:{V}", "E5_B2", adB2), R3A_B2_cont=g(f"E5_B2<R3_A:{V}", "E5_B2", cont))
        gains[V] = {k: round(v, 3) for k, v in x.items()}
        out["checks"][f"POS_R3_{V}"] = bool(x["R3same_B"] >= .30 - 1e-9 and x["R3same_B2"] >= .30 - 1e-9)
        out["checks"][f"POS_S_{V}"] = bool(x["Ssame_B"] >= .30 - 1e-9 and x["Ssame_B2"] >= .30 - 1e-9)
    out["gains"] = gains
    x = gains["ALL59"]
    out["checks"]["DONOR_CONTENT"] = bool(x["R3A_B_cont"] >= x["R3A_B_addr"] + .10 - 1e-9 and x["R3A_B2_cont"] >= x["R3A_B2_addr"] + .10 - 1e-9)
    fb, fb2 = Pw("R3fixA_B", adB), Pw("R3fixA_B2", adB2)
    out["fixed_plan"] = dict(B=round(fb, 3), B2=round(fb2, 3))
    out["checks"]["FIXED_PLAN"] = bool(abs(fb - r3b) <= .05 + 1e-9 and abs(fb2 - r3b2) <= .05 + 1e-9 and min(fb, fb2) >= .90 - 1e-9)
    out["checks"]["STABLE_MAP"] = bool(out["checks"]["POS_R3_ALL59"] and out["checks"]["DONOR_CONTENT"] and out["checks"]["FIXED_PLAN"])
    same_plan = float(np.mean([[(x["kh"], x["t"], x["N_hat"], x["act"]) for x in m["r3"]["B"]] == [(x["kh"], x["t"], x["N_hat"], x["act"]) for x in m["r3"]["A"]]
                               and [(x["kh"], x["t"], x["N_hat"], x["act"]) for x in m["r3"]["B2"]] == [(x["kh"], x["t"], x["N_hat"], x["act"]) for x in m["r3"]["A"]] for m in M]))
    out["r3_plans_identical_A_B_B2"] = round(same_plan, 3)
    print(f"[{tag}] N {N} · n {n} · checks {out['checks']} · I0 {out['i0']} · fixed plan {out['fixed_plan']} · plans identical {same_plan:.3f}")
    for V, x in gains.items():
        print(f"   {V}: {x}")
    return out


if __name__ == "__main__":
    json.dump({t: cell(t) for t in sys.argv[1:]}, open(os.environ.get("ANALYSIS_OUT", "/vlm/results/r3/s781_analysis.json"), "w"), indent=1)
