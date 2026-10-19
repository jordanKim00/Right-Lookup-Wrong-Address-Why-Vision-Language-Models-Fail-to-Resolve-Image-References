"""S795 analysis (descriptive; metrics fixed at registration, knowledge/DECISIONS.md 22:3x): the address lens over depth.
For each lookup layer L* of the index-matching heads and each earlier layer l, the block shares that L*'s heads would give if their query were
computed from the stream after layer l (mean over the heads of L*, then over scenes).  Reported per question:
  target share (block j; START(t): t, END(k): N + 1 - k) and literal share (block k, END only), per l;
  formation depth: the first l at which the target share exceeds .5 (START) and the first l at which END's share on the literal block exceeds .4;
  divergence: Jensen-Shannon distance between the END(k) lens and the START(k) lens (same ordinal) and the START(N + 1 - k) lens (same target).
Registered predictions: small models END(k >= 3) toward the literal block from early layers and to the end; 27B: END and START(same ordinal)
part after L27-34 (END(4): image 4's share falls after that band; END(2): the target's share rises); START reaches its target before L*.
Usage: python s795_analyze.py TAG [TAG ...] -> /vlm/results/r3/s795_analysis.json"""
from __future__ import annotations

import json, sys
import numpy as np

N = 7


def jsd(p, q):
    p, q = np.asarray(p, float) + 1e-12, np.asarray(q, float) + 1e-12
    p /= p.sum(); q /= q.sum(); m = .5 * (p + q)
    return float(np.sqrt(.5 * np.sum(p * np.log(p / m)) + .5 * np.sum(q * np.log(q / m))))


def analyze(tag):
    d = json.load(open(f"/vlm/results/r3/s795_{tag}_lens.json"))
    rows, LS = d["rows"], [int(x) for x in d["lookup_layers"]]
    out = dict(tag=tag, model=d["model"], lookup_layers=LS, per_Ls={})
    for Ls in LS:
        A = {fr: {q: np.array([r[fr][str(q)][str(Ls)] for r in rows]) for q in range(1, N + 1)} for fr in ("START", "END")}   # [scene, l, N]
        res = {}
        for q in range(1, N + 1):
            S = A["START"][q].mean(0); E = A["END"][q].mean(0)                                # [l, N]
            j = N + 1 - q
            res[str(q)] = dict(
                start_target=S[:, q - 1].round(3).tolist(),
                end_target=E[:, j - 1].round(3).tolist(),
                end_literal=E[:, q - 1].round(3).tolist(),
                start_form=next((l for l in range(Ls) if S[l, q - 1] > .5), None),
                end_literal_form=(next((l for l in range(Ls) if E[l, q - 1] > .4), None) if q != j else None),
                jsd_same_ordinal=[round(np.mean([jsd(A["END"][q][s, l], A["START"][q][s, l]) for s in range(len(rows))]), 3) for l in range(Ls)],
                jsd_same_target=[round(np.mean([jsd(A["END"][q][s, l], A["START"][j][s, l]) for s in range(len(rows))]), 3) for l in range(Ls)])
        out["per_Ls"][str(Ls)] = res
    return out


if __name__ == "__main__":
    R = {}
    for tag in sys.argv[1:]:
        try:
            R[tag] = a = analyze(tag)
        except FileNotFoundError:
            print(f"[{tag}] missing"); continue
        print(f"== {tag} ({a['model']}) lookup layers {a['lookup_layers']}")
        for Ls in a["lookup_layers"]:
            res = a["per_Ls"][str(Ls)]
            pick = lambda v, ls: " ".join(f"L{l}:{v[l]:.2f}" for l in ls if l < len(v))
            ls = sorted({0, 4, 8, 12, 16, 20, 24, 26, 28, 30, 32, 34, Ls - 1} & set(range(Ls)))
            print(f" L*={Ls}: START formation depth " + " ".join(f"t{q}:{res[str(q)]['start_form']}" for q in range(1, N + 1)))
            for q in (2, 4, 5, 6):
                r = res[str(q)]
                print(f"   END({q}) target  {pick(r['end_target'], ls)}")
                print(f"   END({q}) literal {pick(r['end_literal'], ls)}")
                print(f"   START({q}) own   {pick(r['start_target'], ls)}")
                print(f"   JSD(END{q}, START{q}) {pick(r['jsd_same_ordinal'], ls)} | JSD(END{q}, START{N+1-q}) {pick(r['jsd_same_target'], ls)}")
    json.dump(R, open("/vlm/results/r3/s795_analysis.json", "w"))
