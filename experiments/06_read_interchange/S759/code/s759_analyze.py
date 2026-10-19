"""S759 analysis, frozen with the pre-registration (knowledge/DECISIONS.md, 2026-10-13 13:2x) before any S759 data.
F2 (END(2) <- START(2), frame word): t = 2, base END(2), top F2:P.  F5 (END(5) <- START(5)): t = 5, base END(5), top F5:P.
S(arm) = (F_t(arm) - F_t(base)) / (F_t(top) - F_t(base)).  K (END(5) <- END(2), ordinal): t = 6, lo K:EL, hi K:ELP.
I0       NOOP = END(5) and NOOPF = END(2) in answers on >= .975 of scenes; F2:P and F5:P reproduce S758 (F_t >= .90).
BAND     per family: SINGLE(Px) if exactly one of P1, P2, P3 has S >= .50 alone and the two others together (its leave-one-out cell) < .50;
         REDUNDANT if two or more single sub-bands have S >= .50; JOINT if no single one does but a pair does; SPREAD otherwise.
ROUTE    per family, over all of P: FA / CONV / MEM if exactly that route-only arm has S >= .50; SHARED if two or more; DISTRIBUTED if none.
LATE_ORD descriptive: share of the K:EL -> K:ELP gap that the ordinal reads of each sub-band close.
QK       for F2 arms, among (arm, scene) with the answer on image 2, the matcher pointer (S705a head-rows L35-47, tail1 / answer) is on 2 in >= .70.
Descriptive: every arm's answer distribution, sub-band route arms (P1 / P2 fa, conv, mem), pointer shares, scene-bootstrap 95% intervals.
Usage: python s759_analyze.py TAG [PART_TAG ...]
"""
import json, sys
import numpy as np

N = 7
FAMS = {"F2": ("E2", "F2:P", 2), "F5": ("E5", "F5:P", 5)}


def boot(x, B=4000, seed=0):
    x = np.asarray(x, float); r = np.random.default_rng(seed)
    m = [x[r.integers(0, len(x), len(x))].mean() for _ in range(B)]
    return [round(float(x.mean()), 3), round(float(np.percentile(m, 2.5)), 3), round(float(np.percentile(m, 97.5)), 3)]


def lse(x):
    m = x.max(); return m + np.log(np.exp(x - m).sum())


def analyze(tag, parts):
    M, Z = [], {}
    for p in parts:
        D = f"/vlm/results/r3/s759_{p}_frame"
        M += json.load(open(D + ".json"))["meta"]; z = np.load(D + ".npz"); Z.update({k: z[k] for k in z.files})
    M.sort(key=lambda m: m["si"]); n = len(M)
    assert len({m["si"] for m in M}) == n
    arms = list(M[0]["land"])
    L = lambda a: np.array([m["land"][a] for m in M])
    F = lambda a, t: float(np.mean(L(a) == t))
    out = dict(n=n, checks={}, F={}, S={}, dist={a: {str(b): round(float(np.mean(L(a) == b)), 3) for b in range(0, N + 1)} for a in arms})
    i0 = dict(noop=round(float(np.mean(L("NOOP") == L("E5"))), 3), noopf=round(float(np.mean(L("NOOPF") == L("E2"))), 3),
              F2P=round(F("F2:P", 2), 3), F5P=round(F("F5:P", 5), 3))
    out["checks"]["I0"] = bool(i0["noop"] >= .975 - 1e-9 and i0["noopf"] >= .975 - 1e-9 and i0["F2P"] >= .90 - 1e-9 and i0["F5P"] >= .90 - 1e-9)
    out["i0"] = i0
    for fam, (base, top, t) in FAMS.items():
        lo, hi = F(base, t), F(top, t)
        S = lambda a: (F(a, t) - lo) / (hi - lo) if hi != lo else float("nan")
        for a in [x for x in arms if x.startswith(fam + ":")]:
            out["F"][a] = boot(L(a) == t); out["S"][a] = round(S(a), 3)
        single = {c: S(f"{fam}:P{c}") for c in "123"}
        loo = {"1": S(f"{fam}:P23"), "2": S(f"{fam}:P13"), "3": S(f"{fam}:P12")}
        big = [c for c, x in single.items() if x >= .50 - 1e-9]
        if len(big) == 1 and loo[big[0]] < .50 - 1e-9:
            band = f"SINGLE(P{big[0]})"
        elif len(big) >= 2:
            band = "REDUNDANT"
        elif any(S(f"{fam}:P{pp}") >= .50 - 1e-9 for pp in ("12", "13", "23")):
            band = "JOINT"
        else:
            band = "SPREAD" if not big else f"SINGLE_NOT_NECESSARY(P{big[0]})"
        out["checks"][f"BAND_{fam}"] = band
        rs = {"FA": S(f"{fam}:Pfa"), "CONV": S(f"{fam}:Pconv"), "MEM": S(f"{fam}:Pmem")}
        rb = [k for k, x in rs.items() if x >= .50 - 1e-9]
        out["checks"][f"ROUTE_{fam}"] = rb[0] if len(rb) == 1 else ("SHARED" if rb else "DISTRIBUTED")
    lo6, hi6 = F("K:EL", 6), F("K:ELP", 6)
    out["late_ord"] = {f"P{c}": round((F(f"K:EL+P{c}", 6) - lo6) / (hi6 - lo6), 3) if hi6 != lo6 else None for c in "123"}
    out["late_ord_F6"] = {a: round(F(a, 6), 3) for a in ("K:EL", "K:EL+P1", "K:EL+P2", "K:EL+P3", "K:ELP")}
    mt = [tuple(x) for x in json.load(open("/vlm/results/r3/s705a_analysis.json"))["qa_27b"]["heads"]["matchers"]]
    dec = [x for x in mt if x[0] <= 47 and x[2] in (1, 2)]

    def pointer(si, a):
        sh = [np.exp(Z[f"{si}|{a}|{li}"][ri, h].astype(np.float64) - lse(Z[f"{si}|{a}|{li}"][ri, h].astype(np.float64))) for (li, h, ri) in dec]
        return int(np.argmax(np.mean(sh, 0))) + 1
    try:
        f2 = [a for a in arms if a.startswith("F2:")]
        hit = [pointer(M[i]["si"], a) == 2 for a in f2 for i in range(n) if M[i]["land"][a] == 2]
        out["checks"]["QK"] = bool(hit and np.mean(hit) >= .70 - 1e-9); out["qk_pointer_on2_given_answer2"] = round(float(np.mean(hit)), 3) if hit else None
    except KeyError:
        out["checks"]["QK"] = None
    print(f"[{tag}] n {n} · checks {out['checks']} · I0 {i0}")
    for fam in FAMS:
        print(f"   {fam}: " + " · ".join(f"{a.split(':')[1]} {out['S'][a]:+.2f}" for a in arms if a.startswith(fam + ":")))
    print(f"[{tag}] late ordinal {out['late_ord']} · F6 {out['late_ord_F6']} · QK {out.get('qk_pointer_on2_given_answer2')}")
    return out


if __name__ == "__main__":
    tag, parts = sys.argv[1], (sys.argv[2:] or [sys.argv[1]])
    json.dump({tag: analyze(tag, parts)}, open("/vlm/results/r3/s759_analysis.json", "w"), indent=1)
