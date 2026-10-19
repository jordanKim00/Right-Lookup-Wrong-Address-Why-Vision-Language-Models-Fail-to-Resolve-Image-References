"""S764 analysis, frozen with the pre-registration (knowledge/DECISIONS.md, 2026-10-13 18:3x) before any S764 data.
Shares: softmax of the fixed-bank block scores over the N blocks, per head and row; pointer = image with the largest mean share over the S705a
matcher head-rows at L35-47 (rows tail1 / answer), as S757.
LITERAL_N8  N = 8, END(4) and END(5) pooled: pointer on the literal image (k) in >= .50 and on two-before-the-last (6) in <= .20 -> LITERAL;
            on 6 in >= .50 -> ANCHOR2; on the correct image (N + 1 - k) in >= .50 -> CORRECT; otherwise MIXED.
LATE_SHIFT  END(5) at N = 7 and END(4), END(5) at N = 8, scenes answering an image other than the literal: mean share over all heads (rows tail1 /
            answer) on the answered image at L51-59 >= 1.5 x the same at L35-47 (replicates the S763 description).
Descriptive (figure data): per N, question and layer, the mean share on the target, the literal and the answered image, the matcher pointer
distribution, answer distribution, and the mean output gate of the matcher heads.
Usage: python s764_analyze.py TAG7 TAG8
"""
import json, sys
import numpy as np


def sm(sc):
    x = sc - sc.max(-1, keepdims=True); e = np.exp(x); return e / e.sum(-1, keepdims=True)


def load(tag):
    D = f"/vlm/results/r3/s764_{tag}_pairs"
    return json.load(open(D + ".json")), np.load(D + ".npz")


def analyze(tags):
    mt = [tuple(x) for x in json.load(open("/vlm/results/r3/s705a_analysis.json"))["qa_27b"]["heads"]["matchers"]]
    dec = [x for x in mt if x[0] <= 47 and x[2] in (1, 2)]
    out = dict(checks={}, desc={})
    ptr = {}
    late = []
    for tag in tags:
        d, z = load(tag); N = d["N"]; M = d["meta"]; CAPL = d["CAPL"]
        tgt = lambda q: (int(q[1:]) if q[0] == "S" else N + 1 - int(q[1:]))
        lit = lambda q: int(q[1:])
        desc = {}
        for q in d["QS"]:
            P, rows = [], []
            for m in M:
                si = m["si"]
                sh = np.mean([sm(z[f"{si}|{q}|{li}|s"][ri, h].astype(np.float64)) for (li, h, ri) in dec if li in CAPL], 0)
                P.append(int(np.argmax(sh)) + 1)
            P = np.array(P); A = np.array([m["ans"][q]["pos"] for m in M])
            lay = {}
            for li in CAPL:
                S = np.stack([sm(z[f"{m['si']}|{q}|{li}|s"].astype(np.float64))[1:3].mean(0) for m in M])      # [scenes, heads, N]
                G = np.stack([z[f"{m['si']}|{q}|{li}|g"][1:3].mean(0) for m in M])                             # [scenes, heads]
                hm = S.mean(1)                                                                                    # [scenes, N]
                ans_sh = [hm[i, A[i] - 1] for i in range(len(M)) if A[i] >= 1]
                lay[str(li)] = dict(target=round(float(hm[:, tgt(q) - 1].mean()), 3), literal=round(float(hm[:, lit(q) - 1].mean()), 3),
                                    answered=round(float(np.mean(ans_sh)), 3) if ans_sh else None, gate=round(float(G.mean()), 3))
            desc[q] = dict(target=tgt(q), pointer={str(b): round(float(np.mean(P == b)), 3) for b in range(1, N + 1)},
                           answer={str(b): round(float(np.mean(A == b)), 3) for b in range(0, N + 1)}, acc=round(float(np.mean(A == tgt(q))), 3), layers=lay)
            ptr[(N, q)] = P
            if q[0] == "E" and ((N == 7 and q == "E5") or (N == 8 and q in ("E4", "E5"))):
                for i, m in enumerate(M):
                    if A[i] >= 1 and A[i] != lit(q):
                        si = m["si"]
                        f = lambda Ls: np.mean([sm(z[f"{si}|{q}|{li}|s"].astype(np.float64))[1:3].mean(0).mean(0)[A[i] - 1] for li in Ls])
                        late.append((f([l for l in CAPL if 51 <= l <= 59]), f([l for l in CAPL if 35 <= l <= 47])))
        out["desc"][str(N)] = desc
    p8 = np.concatenate([ptr[(8, "E4")], ptr[(8, "E5")]]); lits = np.concatenate([np.full(len(ptr[(8, "E4")]), 4), np.full(len(ptr[(8, "E5")]), 5)])
    cors = np.concatenate([np.full(len(ptr[(8, "E4")]), 5), np.full(len(ptr[(8, "E5")]), 4)])
    fl, fa, fc = float(np.mean(p8 == lits)), float(np.mean(p8 == 6)), float(np.mean(p8 == cors))
    out["checks"]["LITERAL_N8"] = ("LITERAL" if fl >= .50 - 1e-9 and fa <= .20 + 1e-9 else "ANCHOR2" if fa >= .50 - 1e-9 else "CORRECT" if fc >= .50 - 1e-9 else "MIXED")
    out["n8_pointer"] = dict(literal=round(fl, 3), two_before_last=round(fa, 3), correct=round(fc, 3), n=len(p8))
    lt = np.array(late)
    out["checks"]["LATE_SHIFT"] = bool(len(lt) and lt[:, 0].mean() >= 1.5 * lt[:, 1].mean() - 1e-9)
    out["late_shift"] = dict(n=len(lt), L51_59=round(float(lt[:, 0].mean()), 3) if len(lt) else None, L35_47=round(float(lt[:, 1].mean()), 3) if len(lt) else None)
    print(f"checks {out['checks']} · N8 pointer {out['n8_pointer']} · late shift {out['late_shift']}")
    for N_, desc in out["desc"].items():
        for q, x in desc.items():
            print(f"   N{N_} {q} (target {x['target']}) acc {x['acc']} · pointer {x['pointer']}")
    return out


if __name__ == "__main__":
    json.dump(analyze(sys.argv[1:]), open("/vlm/results/r3/s764_analysis.json", "w"), indent=1)
