"""S769 analysis, frozen with the pre-registration (knowledge/DECISIONS.md, 2026-10-13 23:1x) before any S769 data.
Data: exp/s769_end4.py at N = 7 (seed 76900) and N = 8 (seed 76901), 40 scenes each.
acc_b(arm) = share of scenes answering image b.  share_L(arm, b) = mean over scenes of the all-heads mean (rows tail1 / answer) of the softmax
block share on image b at layer L (as S766).  S(arm) = (acc_4(arm) - acc_4(E4)) / (acc_4(F4:P) - acc_4(E4)) at N = 7 (descriptive).
I0          per N: NOOPF = E4 in answers on >= .975 of scenes and acc_4(F4:P) >= .90.
PARADOX     (N = 7, target = literal = 4)  FIX if acc_4(F4:P1) >= .80 and >= acc_4(E4) + .40;  PARTIAL if acc_4(F4:P1) >= acc_4(E4) + .20 (not FIX);
            NONE otherwise.  LOOKUP: share_39(F4:P1, 4) >= share_39(E4, 4) + .20.
REVERSE     (N = 7)  BREAK if acc_4(R4:P1) <= acc_4(S4) - .40;  WEAK if acc_4(R4:P1) <= acc_4(S4) - .15 (not BREAK);  NONE otherwise.
            LOOKUP_R: share_39(R4:P1, 4) <= share_39(S4, 4) - .20.
N8_LITERAL  (N = 8)  acc_4(F4:P1) >= .60 and share_39(F4:P1, 4) >= share_39(E4, 4) + .20.
CONV        (N = 7)  acc_4(F4:P1conv) >= acc_4(E4) + .25 and share_39(F4:P1conv, 4) >= share_39(E4, 4) + .15 and share_39(F4:P1fa, 4) <= share_39(E4, 4) + .10.
Descriptive: every arm's answer distribution with scene-bootstrap 95% intervals; per full-attention layer L35-59 the block-4 share of every arm;
at N = 8 acc_5 (correct END(4)), acc_4 (literal), acc_6 (N - 2); the menu-logit margin of image 4 over the best other menu word; Jensen-Shannon
distance of each arm's menu distribution to E4 and S4.
Usage: python s769_analyze.py TAG7 TAG8
"""
import json, os, sys
import numpy as np


def sm(sc):
    x = sc - sc.max(-1, keepdims=True); e = np.exp(x); return e / e.sum(-1, keepdims=True)


def p(ll):
    x = np.array(ll, float); x = np.exp(x - x.max()); return x / x.sum()


def jsd(a, b):
    m = .5 * (a + b); kl = lambda u, w: float(np.sum(u * (np.log2(u + 1e-12) - np.log2(w + 1e-12))))
    return .5 * (kl(a, m) + kl(b, m))


def boot(x, B=4000, seed=0):
    x = np.asarray(x, float); r = np.random.default_rng(seed)
    m = [x[r.integers(0, len(x), len(x))].mean() for _ in range(B)]
    return [round(float(x.mean()), 3), round(float(np.percentile(m, 2.5)), 3), round(float(np.percentile(m, 97.5)), 3)]


def load(tag):
    D = f"/vlm/results/r3/s769_{tag}_end4"
    d = json.load(open(D + ".json")); z = np.load(D + ".npz")
    M = sorted(d["meta"], key=lambda m: m["si"]); assert len({m["si"] for m in M}) == len(M)
    nl = d["nl"]; CAP = d["CAP"]
    near = lambda x: min(CAP, key=lambda l: abs(l - int(round(x / 63 * (nl - 1)))))
    return d, M, z, near


def describe(d, M, z, near):
    N = d["N"]; arms = list(M[0]["land"])
    L = lambda a: np.array([m["land"][a] for m in M])
    acc = lambda a, b: float(np.mean(L(a) == b))
    sh = lambda a, b, Lr: float(np.mean([sm(z[f"{m['si']}|{a}|{Lr}"].astype(np.float64))[1:3].mean(0).mean(0)[b - 1] for m in M]))
    out = dict(n=len(M), N=N, dist={}, acc={}, share4={}, margin4={}, jsd={})
    for a in arms:
        out["dist"][a] = {str(b): round(float(np.mean(L(a) == b)), 3) for b in range(0, N + 1) if np.mean(L(a) == b) > 0}
        out["acc"][a] = {str(b): boot(L(a) == b) for b in ((4, 5, 6) if N == 8 else (4,))}
        out["share4"][a] = {str(Lr): round(sh(a, 4, Lr), 3) for Lr in d["CAP"]}
        mg = []
        for m in M:
            ll = np.array(m["ll"][a]); mg.append(ll[3] - np.max(np.delete(ll, 3)))
        out["margin4"][a] = round(float(np.mean(mg)), 3)
        out["jsd"][a] = dict(E4=round(float(np.mean([jsd(p(m["ll"][a]), p(m["ll"]["E4"])) for m in M])), 3),
                             S4=round(float(np.mean([jsd(p(m["ll"][a]), p(m["ll"]["S4"])) for m in M])), 3))
    return out, acc, sh


def analyze(tag7, tag8):
    res = dict(checks={})
    for tag, key in ((tag7, "N7"), (tag8, "N8")):
        if tag is None:
            continue
        d, M, z, near = load(tag)
        out, acc, sh = describe(d, M, z, near)
        L39 = near(39)
        noopf = float(np.mean([m["land"]["NOOPF"] == m["land"]["E4"] for m in M]))
        out["i0"] = dict(noopf=round(noopf, 3), F4P_acc4=round(acc("F4:P", 4), 3))
        res["checks"][f"I0_{key}"] = bool(noopf >= .975 - 1e-9 and acc("F4:P", 4) >= .90 - 1e-9)
        s39 = lambda a: sh(a, 4, L39)
        if key == "N7":
            e4, s4 = acc("E4", 4), acc("S4", 4)
            f1, r1 = acc("F4:P1", 4), acc("R4:P1", 4)
            res["checks"]["PARADOX"] = "FIX" if f1 >= .80 - 1e-9 and f1 >= e4 + .40 - 1e-9 else "PARTIAL" if f1 >= e4 + .20 - 1e-9 else "NONE"
            res["checks"]["LOOKUP"] = bool(s39("F4:P1") >= s39("E4") + .20 - 1e-9)
            res["checks"]["REVERSE"] = "BREAK" if r1 <= s4 - .40 + 1e-9 else "WEAK" if r1 <= s4 - .15 + 1e-9 else "NONE"
            res["checks"]["LOOKUP_R"] = bool(s39("R4:P1") <= s39("S4") - .20 + 1e-9)
            res["checks"]["CONV"] = bool(acc("F4:P1conv", 4) >= e4 + .25 - 1e-9 and s39("F4:P1conv") >= s39("E4") + .15 - 1e-9
                                         and s39("F4:P1fa") <= s39("E4") + .10 + 1e-9)
            top = acc("F4:P", 4)
            out["S"] = {a: (round((acc(a, 4) - e4) / (top - e4), 3) if top != e4 else None) for a in out["acc"] if a.startswith("F4:")}
        else:
            res["checks"]["N8_LITERAL"] = bool(acc("F4:P1", 4) >= .60 - 1e-9 and s39("F4:P1") >= s39("E4") + .20 - 1e-9)
        out["L39"] = L39; out["share4_L39"] = {a: round(s39(a), 3) for a in out["acc"]}
        res[key] = out
        print(f"[{tag}] N {out['N']} · n {out['n']} · I0 {out['i0']} · L39 = {L39}")
        for a in out["acc"]:
            print(f"   {a:10s} acc {out['acc'][a]} · share4 L39 {out['share4_L39'][a]} · margin4 {out['margin4'][a]} · jsd {out['jsd'][a]} · dist {out['dist'][a]}")
        if "S" in out:
            print(f"   S {out['S']}")
    print(f"checks {res['checks']}")
    return res


if __name__ == "__main__":
    t7 = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "-" else None
    t8 = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None
    json.dump(analyze(t7, t8), open(os.environ.get("ANALYSIS_OUT", "/vlm/results/r3/s769_analysis.json"), "w"), indent=1)
