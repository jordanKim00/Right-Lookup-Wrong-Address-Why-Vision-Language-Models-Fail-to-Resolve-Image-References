"""S844 / S841 descriptive re-analysis (post hoc, NOT registered; written after the registered S844 verdicts were read).
The registered COND / INTERNAL slopes were taken on d = t - p over all conditions, so they mix the stated goal (N_s at fixed p) with the early
position (p at fixed N_s).  This script separates the two and asks whether the late update stops when the pinned early position equals the stated
target.  Scene bootstrap (seed 0).
  within / across   OLS slope of the answer margin P(p + 1) - P(p - 1) and of the internal margin z[p + 1] - z[p - 1] (layers 51, 55; boundary B4
                    and answer rows) on N_s at fixed p (averaged over p), and on -p at fixed N_s (averaged over N_s)
  goal_match        within p: P(stay) when t = p minus the mean P(stay) under the other stated counts; the same for the internal hold margin
                    z[p] - max(z[p - 1], z[p + 1]) of the pinned run (START(p) given for reference)
  mediation         mean natural END answer and mean pinned answer by N_s (how much of the stated-count effect survives the pin)
  scene             agreement of pinned answers across different p within a scene vs shuffled scenes; the same for the layer-55 answer-row argmax;
                    agreement of the pinned answer with the scene's natural END answer under the same N_s, other N_s, and another scene
Usage: python3 exp/p4_s844_within.py  ->  research/ces/outputs/s844_within_p.json"""
from __future__ import annotations

import json
import numpy as np

V = "/home/storage2/ai26269345/vlm"
B = 2000


def load(fn):
    R = json.load(open(f"{V}/results/r3/{fn}"))["rows"]
    by = {(r["si"], r["Ns"]): r for r in R}
    return by, sorted({r["si"] for r in R})


def analyse(fn, PS, NSS, rng):
    by, sis = load(fn)
    P = lambda ns, p, x, ss: float(np.mean([by[(s, ns)]["land"][f"E_EARLY_p{p}"] == x for s in ss]))
    m = lambda ns, p, ss: P(ns, p, p + 1, ss) - P(ns, p, p - 1, ss)
    def zdiff(L, row):
        return lambda ns, p, ss: float(np.mean([by[(s, ns)]["qk"][f"E_EARLY_p{p}"][L][row][p] - by[(s, ns)]["qk"][f"E_EARLY_p{p}"][L][row][p - 2]
                                                for s in ss]))
    def hold(arm, L, row):
        def f(ns, p, ss):
            v = []
            for s in ss:
                z = by[(s, ns)]["qk"][arm.format(p=p)][L][row]
                v.append(z[p - 1] - max(z[p - 2], z[p] if p < len(z) else -1e9))
            return float(np.mean(v))
        return f
    within = lambda f: (lambda ss: float(np.mean([np.polyfit(NSS, [f(ns, p, ss) for ns in NSS], 1)[0] for p in PS])))
    across = lambda f: (lambda ss: float(np.mean([np.polyfit([-p for p in PS], [f(ns, p, ss) for p in PS], 1)[0] for ns in NSS])))
    def goal_match(f):
        def g(ss):
            v = []
            for p in PS:
                ns0 = p + 3
                if ns0 in NSS:
                    v.append(f(ns0, p, ss) - np.mean([f(ns, p, ss) for ns in NSS if ns != ns0]))
            return float(np.mean(v))
        return g
    def boot(g):
        v = [g([sis[i] for i in rng.integers(0, len(sis), len(sis))]) for _ in range(B)]
        return dict(value=g(sis), ci=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))])
    late = sorted({L for r in by.values() for L in r["qk"]["END"].keys() if int(L) >= 51}, key=int)
    out = dict(source=fn, n=len(sis), within={}, across={}, goal_match={}, table={})
    out["within"]["answer_margin"] = boot(within(m)); out["across"]["answer_margin"] = boot(across(m))
    for L in late:
        for row, nm in ((5, "B4"), (9, "answer")):
            f = zdiff(L, row)
            out["within"][f"z{L}_{nm}"] = boot(within(f)); out["across"][f"z{L}_{nm}"] = boot(across(f))
            out["goal_match"][f"hold{L}_{nm}"] = boot(goal_match(hold("E_EARLY_p{p}", L, row)))
            out["table"][f"hold{L}_{nm}"] = {str(p): [hold("E_EARLY_p{p}", L, row)(ns, p, sis) for ns in NSS] for p in PS}
            out["table"][f"hold{L}_{nm}_START"] = {str(p): float(np.mean([hold("START_p{p}", L, row)(ns, p, sis) for ns in NSS])) for p in PS}
    stay = lambda ns, p, ss: P(ns, p, p, ss)
    out["goal_match"]["stay"] = boot(goal_match(stay))
    out["table"]["stay"] = {str(p): [stay(ns, p, sis) for ns in NSS] for p in PS}
    out["mediation"] = dict(natural_mean={str(ns): float(np.mean([by[(s, ns)]["land"]["END"] for s in sis])) for ns in NSS},
                            pinned_mean={str(ns): float(np.mean([by[(s, ns)]["land"][f"E_EARLY_p{p}"] for s in sis for p in PS])) for ns in NSS})
    # scene attractors
    def agree(key, perm):
        v = []
        for s in sis:
            for ns in NSS:
                for i, p in enumerate(PS):
                    for q in PS[i + 1:]:
                        v.append(key(by[(s, ns)], p) == key(by[(perm[s], ns)], q))
        return float(np.mean(v))
    ans = lambda r, p: r["land"][f"E_EARLY_p{p}"]
    L55 = late[-1]
    zarg = lambda r, p: int(np.argmax(r["qk"][f"E_EARLY_p{p}"][L55][9])) + 1
    ident = {s: s for s in sis}
    shuf = [dict(zip(sis, rng.permutation(sis))) for _ in range(100)]
    out["scene"] = dict(answers_same=agree(ans, ident), answers_shuffled=float(np.mean([agree(ans, pm) for pm in shuf])),
                        argmax_same=agree(zarg, ident), argmax_shuffled=float(np.mean([agree(zarg, pm) for pm in shuf])),
                        argmax_is_answer=float(np.mean([zarg(by[(s, ns)], p) == ans(by[(s, ns)], p) for s in sis for ns in NSS for p in PS])))
    def nat(src):
        return float(np.mean([by[(s, ns)]["land"][f"E_EARLY_p{p}"] == src(s, ns) for s in sis for ns in NSS for p in PS]))
    oth = float(np.mean([by[(s, ns)]["land"][f"E_EARLY_p{p}"] == by[(s, y)]["land"]["END"]
                         for s in sis for ns in NSS for y in NSS if y != ns for p in PS]))
    out["scene"].update(natural_same_Ns=nat(lambda s, ns: by[(s, ns)]["land"]["END"]), natural_other_Ns=oth,
                        natural_other_scene=float(np.mean([nat(lambda s, ns, pm=pm: by[(pm[s], ns)]["land"]["END"]) for pm in shuf[:50]])))
    return out


def main():
    rng = np.random.default_rng(0)
    res = dict(note="post hoc, not registered (S844 rules mixed the stated goal with the early position)",
               S844=analyse("s844_s844_27b_anchor.json", (3, 4, 5, 6), (6, 7, 8, 9), rng),
               S841=analyse("s841_s841_27b_anchor.json", (2, 3, 4, 5, 6), (6, 7, 8), rng))
    json.dump(res, open(f"{V}/research/ces/outputs/s844_within_p.json", "w"), indent=1)
    for k in ("S844", "S841"):
        o = res[k]
        print(f"== {k}")
        for t in ("within", "across"):
            print(" ", t, {a: "%+.3f [%+.3f, %+.3f]" % (v["value"], *v["ci"]) for a, v in o[t].items()})
        print("  goal_match", {a: "%+.3f [%+.3f, %+.3f]" % (v["value"], *v["ci"]) for a, v in o["goal_match"].items()})
        print("  stay", {p: [round(x, 2) for x in v] for p, v in o["table"]["stay"].items()})
        print("  mediation", o["mediation"]); print("  scene", {a: round(v, 3) for a, v in o["scene"].items()})


if __name__ == "__main__":
    main()
