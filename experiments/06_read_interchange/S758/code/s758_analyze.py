"""S758 analysis, frozen with the pre-registration (knowledge/DECISIONS.md, 2026-10-13 12:3x) before any S758 data.
F_t(arm) = share of scenes whose answer is image t.  Families (recipient <- donor):  K (E5 <- E2) t = 6, base E5, top K:ELP;  K3 (E3 <- E2) t = 6,
base E3, top E2;  F2 (E2 <- S2) t = 2, base E2, top F2:ELP;  F5 (E5 <- S5) t = 5, base E5, top F5:ELP.
Normalised effect S(arm) = (F_t(arm) - F_t(base)) / (F_t(top) - F_t(base)).
I0     NOOP = E5, K:ELP = E2, F2:ELP = S2, F5:ELP = S5, K:E = T22, K:EL = T26, K:E+Lsplit = K:EL in answers on >= .975 of scenes each
       (instrument; nothing else is read if I0 fails).
KREAD  LT_SUFFICES  S(K:L) >= .50;  E_SUFFICES  S(K:E) >= .50 and not LT_SUFFICES;  E_LT_JOINT  S(K:L) < .50, S(K:E) < .50, S(K:EL) >= .50;
       LATE_NEEDED  S(K:EL) < .50.
KNEC   LT_SUFFICES: S(K:EP) <= .50 (Lt is also necessary when every other read is the donor's);  E_LT_JOINT: S(K:LP) <= .50 and S(K:EP) <= .50.
ROUTE  in the family of the KREAD outcome: LT_SUFFICES -> K:L23 .. K:Lmem with lo = E5, hi = K:L;  E_LT_JOINT -> K:E+L23 .. K:E+Lmem with lo = K:E,
       hi = K:EL.  Share s_R = (F6(R) - F6(lo)) / (F6(hi) - F6(lo)) for FA (= L23), Lconv, Lmem: CONV / MEM / FA = that route >= .50 and the other two
       < .50;  SHARED = two or more >= .50;  DISTRIBUTED = none.  SINGLE_LAYER(l) if exactly one of L23..L26 has s >= .50.
QK_FOLLOWS  pointer = image with the largest mean block share over the S705a matcher head-rows at L35-47 (rows tail1 / answer).  Over all K-family arms
       and scenes with the answer on 6, the pointer is on 6 in >= .70, and across K-family arms corr(F6, share of pointer on 6) >= .90.
FRAME  the KREAD classification applied to F2 and F5.
FAR    scenes whose natural END(5) answer a5 is neither 3 nor 6.  In the cell where the "fifth" reads are confined to the bands the KREAD outcome names
       (LT_SUFFICES -> K:EP, E_LT_JOINT -> K:P), the answer equals a5 in >= .50 of these scenes.
Descriptive: answer distributions, F3 / F6 / P(answer = a5), pointer shares, scene-bootstrap 95% intervals (4,000 draws), menu-logit differences
of the I0 pairs.
Usage: python s758_analyze.py TAG [PART_TAG ...]   (parts are concatenated in order)
"""
import json, os, sys
import numpy as np

N = 7
FAMS = {"K": ("E5", "K:ELP", 6), "K3": ("E3", "E2", 6), "F2": ("E2", "F2:ELP", 2), "F5": ("E5", "F5:ELP", 5)}


def boot(x, B=4000, seed=0):
    x = np.asarray(x, float); r = np.random.default_rng(seed)
    m = [x[r.integers(0, len(x), len(x))].mean() for _ in range(B)]
    return [round(float(x.mean()), 3), round(float(np.percentile(m, 2.5)), 3), round(float(np.percentile(m, 97.5)), 3)]


def lse(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(m, axis) + np.log(np.sum(np.exp(x - m), axis=axis))


def analyze(tag, parts):
    M, Z = [], {}
    for p in parts:
        D = f"/vlm/results/r3/s758_{p}_paths"
        M += json.load(open(D + ".json"))["meta"]; z = np.load(D + ".npz"); Z.update({k: z[k] for k in z.files})
    M.sort(key=lambda m: m["si"]); n = len(M)
    assert len({m["si"] for m in M}) == n
    arms = list(M[0]["land"])
    L = lambda a: np.array([m["land"][a] for m in M])
    F = lambda a, t: float(np.mean(L(a) == t))
    out = dict(n=n, checks={}, F={}, S={}, dist={})
    for a in arms:
        out["dist"][a] = {str(b): round(float(np.mean(L(a) == b)), 3) for b in range(0, N + 1)}

    # I0
    pairs = [("NOOP", "E5"), ("K:ELP", "E2"), ("F2:ELP", "S2"), ("F5:ELP", "S5"), ("T22", "K:E"), ("T26", "K:EL"), ("K:E+Lsplit", "K:EL")]
    i0 = {f"{x}={y}": round(float(np.mean(L(x) == L(y))), 3) for x, y in pairs}
    dl = {f"{x}={y}": round(float(max(np.max(np.abs(np.array(m["ll"][x]) - np.array(m["ll"][y]))) for m in M)), 4) for x, y in pairs}
    out["checks"]["I0"] = bool(all(v >= .975 - 1e-9 for v in i0.values())); out["i0_agree"] = i0; out["i0_max_logit_diff"] = dl

    def S(fam, a):
        base, top, t = FAMS[fam]
        lo, hi = F(base, t), F(top, t)
        return (F(a, t) - lo) / (hi - lo) if hi != lo else float("nan")

    for fam, (base, top, t) in FAMS.items():
        for a in [x for x in arms if x.startswith(fam + ":")] + [base, top]:
            out["F"][a if a.startswith(fam) else f"{fam}|{a}"] = boot(L(a) == t)
            if a.startswith(fam + ":"):
                out["S"][a] = round(S(fam, a), 3)

    def kread(fam):
        sL, sE, sEL = S(fam, f"{fam}:L"), S(fam, f"{fam}:E"), S(fam, f"{fam}:EL")
        if sL >= .50 - 1e-9:
            return "LT_SUFFICES"
        if sE >= .50 - 1e-9:
            return "E_SUFFICES"
        if sEL >= .50 - 1e-9:
            return "E_LT_JOINT"
        return "LATE_NEEDED"
    kr = kread("K"); out["checks"]["KREAD"] = kr
    if kr == "LT_SUFFICES":
        out["checks"]["KNEC"] = bool(S("K", "K:EP") <= .50 + 1e-9)
    elif kr == "E_LT_JOINT":
        out["checks"]["KNEC"] = bool(S("K", "K:LP") <= .50 + 1e-9 and S("K", "K:EP") <= .50 + 1e-9)
    else:
        out["checks"]["KNEC"] = None

    LT = [int(x[3:]) for x in arms if x.startswith("K:L") and x[3:].isdigit()]
    fam_r = {"LT_SUFFICES": ("", "E5", "K:L"), "E_LT_JOINT": ("E+", "K:E", "K:EL")}.get(kr)
    if fam_r:
        pre, lo, hi = fam_r
        flo, fhi = F(lo, 6), F(hi, 6)
        share = lambda a: (F(a, 6) - flo) / (fhi - flo) if fhi != flo else float("nan")
        sr = {"FA": share(f"K:{pre}L{LT[0]}"), "Lconv": share(f"K:{pre}Lconv"), "Lmem": share(f"K:{pre}Lmem")}
        big = [k for k, x in sr.items() if x >= .50 - 1e-9]
        out["checks"]["ROUTE"] = ({"FA": "FA", "Lconv": "CONV", "Lmem": "MEM"}[big[0]] if len(big) == 1 else "SHARED" if len(big) > 1 else "DISTRIBUTED")
        sl = {str(l): share(f"K:{pre}L{l}") for l in LT}
        bl = [l for l, x in sl.items() if x >= .50 - 1e-9]
        out["checks"]["LAYER"] = f"SINGLE_LAYER({bl[0]})" if len(bl) == 1 else ("SEVERAL" if bl else "NONE")
        out["route_share"] = {k: round(x, 3) for k, x in sr.items()}; out["layer_share"] = {k: round(x, 3) for k, x in sl.items()}
    else:
        out["checks"]["ROUTE"] = None; out["checks"]["LAYER"] = None

    # QK
    mt = [tuple(x) for x in json.load(open("/vlm/results/r3/s705a_analysis.json"))["qa_27b"]["heads"]["matchers"]]
    dec = [x for x in mt if x[0] <= 47 and x[2] in (1, 2)]

    def pointer(si, a):
        sh = []
        for (li, h, ri) in dec:
            sc = Z[f"{si}|{a}|{li}"][ri, h].astype(np.float64)
            sh.append(np.exp(sc - lse(sc)))
        return int(np.argmax(np.mean(sh, 0))) + 1
    kf = [a for a in arms if a.startswith("K:")]
    P = {a: np.array([pointer(m["si"], a) for m in M]) for a in kf + ["E5", "E2", "E3"] + [a for a in arms if a.startswith(("K3:", "F2:", "F5:"))] + ["S2", "S5"]}
    on6 = [(P[a][i] == 6) for a in kf for i in range(n) if M[i]["land"][a] == 6]
    f6 = np.array([F(a, 6) for a in kf]); p6 = np.array([float(np.mean(P[a] == 6)) for a in kf])
    cor = float(np.corrcoef(f6, p6)[0, 1]) if f6.std() > 0 and p6.std() > 0 else float("nan")
    out["checks"]["QK_FOLLOWS"] = bool(len(on6) and np.mean(on6) >= .70 - 1e-9 and cor >= .90 - 1e-9)
    out["qk"] = dict(pointer_on6_given_answer6=round(float(np.mean(on6)), 3) if on6 else None, corr_F6_P6=round(cor, 3),
                     pointer_share={a: {str(b): round(float(np.mean(P[a] == b)), 3) for b in range(1, N + 1)} for a in P})

    out["checks"]["FRAME"] = {fam: kread(fam) for fam in ("F2", "F5")}

    a5 = L("E5"); far = [i for i in range(n) if a5[i] not in (3, 6)]
    cell = {"LT_SUFFICES": "K:EP", "E_LT_JOINT": "K:P"}.get(kr)
    rep = {a: round(float(np.mean([M[i]["land"][a] == a5[i] for i in far])), 3) for a in kf} if far else {}
    out["checks"]["FAR"] = bool(rep[cell] >= .50 - 1e-9) if (cell and far) else None
    out["far"] = dict(n=len(far), cell=cell, reproduce_a5=rep, base_E2_on_a5=round(float(np.mean([M[i]["land"]["E2"] == a5[i] for i in far])), 3) if far else None)

    print(f"[{tag}] n {n} · checks {out['checks']}")
    print(f"[{tag}] I0 agree {i0} · max menu-logit diff {dl}")
    for fam in FAMS:
        print(f"   {fam}: " + " · ".join(f"{a.split(':')[1]} {out['S'][a]:+.2f}" for a in arms if a.startswith(fam + ":")))
    print(f"[{tag}] route {out.get('route_share')} · layer {out.get('layer_share')} · qk {out['qk']['pointer_on6_given_answer6']} corr {out['qk']['corr_F6_P6']}")
    print(f"[{tag}] far {out['far']}")
    return out


if __name__ == "__main__":
    tag, parts = sys.argv[1], (sys.argv[2:] or [sys.argv[1]])
    json.dump({tag: analyze(tag, parts)}, open(os.environ.get("ANALYSIS_OUT", "/vlm/results/r3/s758_analysis.json"), "w"), indent=1)
