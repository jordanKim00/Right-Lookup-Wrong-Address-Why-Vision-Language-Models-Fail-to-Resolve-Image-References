"""S824 analysis (rules registered in knowledge/DECISIONS.md with the run, before any S824 data).  Conditions a{N_a}_s{N_s}_k{k}_{E|S}: shown
N_a, stated N_s, ordinal k, END (E) or START (S).  P(cond -> x) = share of scenes landing on picture x.  Scene bootstrap, 4,000 draws, seed 0.
  I0  t1 >= .95; START(k) lands on k in >= .90 of scenes in each matched condition (a7_s7, a8_s8).
  DECOMP (a8_s8): 'no absolute picture-2 bias' if END(5) -> 2 <= .05; 'one past the target present' if END(5) -> 3 >= .10 with the interval's
      lower end > .03; 'one step from the literal present' if END(7) -> 6 >= .25 with the interval's lower end > .10.
  COUNT  in the mismatched conditions (a8_s7 and a7_s8; k in {3, ..., 7}): D = mean over cells of P(stated target N_s + 1 - k) - P(shown target
      N_a + 1 - k): 'the stated count drives the target' if D >= .15 (interval above 0), 'the shown pictures drive it' if D <= -.15 (interval
      below 0), otherwise 'mixed'; START control: START(k) -> k >= .90 in both mismatched conditions.
Descriptive: every condition's landing distribution; END(k) at a7_s7 vs a8_s8 (target-, literal- and absolute-relative errors).
Usage: python s824_analyze.py TAG [TAG ...]"""
from __future__ import annotations

import json, sys
import numpy as np

B = 4000


def analyze(tag):
    d = json.load(open(f"/vlm/results/r3/s824_{tag}_count.json")); R = d["rows"]; rng = np.random.default_rng(0); ks = d["ks"]
    rep = dict(tag=tag, model=d["model"], t1=d["t1_menu"], n=len(R))
    P = lambda rows, c, x: float(np.mean([r["land"][c] == x for r in rows])) if rows else float("nan")
    def boot(f):
        v = []
        for _ in range(B):
            pick = rng.choice(len(R), len(R)); v.append(f([R[i] for i in pick]))
        v = np.array(v, float); v = v[np.isfinite(v)]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else None
    start_ok = {c: min(P(R, f"{c}_k{k}_S", k) for k in ks) for c in ("a7_s7", "a8_s8")}
    rep["I0"] = dict(start_matched=start_ok, ok=bool((d["t1_menu"] or 0) >= .95 and min(start_ok.values()) >= .90))
    p2, p3, p6 = P(R, "a8_s8_k5_E", 2), P(R, "a8_s8_k5_E", 3), P(R, "a8_s8_k7_E", 6)
    c3, c6 = boot(lambda q: P(q, "a8_s8_k5_E", 3)), boot(lambda q: P(q, "a8_s8_k7_E", 6))
    rep["DECOMP"] = dict(END5_to2=p2, END5_to3=p3, ci3=c3, END7_to6=p6, ci6=c6, no_absolute_2=bool(p2 <= .05),
                         one_past_present=bool(p3 >= .10 and c3 and c3[0] > .03), one_step_present=bool(p6 >= .25 and c6 and c6[0] > .10),
                         END5_a7s7={x: P(R, "a7_s7_k5_E", x) for x in range(0, 8)}, END5_a8s8={x: P(R, "a8_s8_k5_E", x) for x in range(0, 9)})
    def D(rows):
        v = []
        for Na, Ns in ((8, 7), (7, 8)):
            for k in ks:
                ts, ta = Ns + 1 - k, Na + 1 - k
                if 1 <= ts <= Na:
                    v.append(P(rows, f"a{Na}_s{Ns}_k{k}_E", ts) - P(rows, f"a{Na}_s{Ns}_k{k}_E", ta))
        return float(np.mean(v)) if v else float("nan")
    dv, cd = D(R), boot(D)
    sc_ok = {c: min(P(R, f"{c}_k{k}_S", k) for k in ks) for c in ("a8_s7", "a7_s8")}
    rep["COUNT"] = dict(D=dv, ci=cd, start_mismatched=sc_ok,
                        verdict=("the stated count drives the target" if (dv >= .15 and cd and cd[0] > 0) else "the shown pictures drive it" if (dv <= -.15 and cd and cd[1] < 0)
                                 else "mixed"),
                        per_cell={f"a{Na}_s{Ns}_k{k}": dict(stated=P(R, f"a{Na}_s{Ns}_k{k}_E", Ns + 1 - k), shown=P(R, f"a{Na}_s{Ns}_k{k}_E", Na + 1 - k),
                                                             literal=P(R, f"a{Na}_s{Ns}_k{k}_E", k)) for Na, Ns in ((8, 7), (7, 8)) for k in ks})
    rep["landing"] = {c: np.bincount([r["land"][c] for r in R], minlength=9).tolist() for c in R[0]["land"]}
    return rep


if __name__ == "__main__":
    allr = {}
    for tag in sys.argv[1:]:
        try:
            rep = analyze(tag)
        except FileNotFoundError:
            print(f"[{tag}] missing"); continue
        allr[tag] = rep
        print(f"== {tag} ({rep['model']}) t1 {rep['t1']} n {rep['n']} I0 {rep['I0']}")
        for key in ("DECOMP", "COUNT"):
            print(f"   {key} {rep[key]}")
    json.dump(allr, open("/vlm/results/r3/s824_analysis.json", "w"), indent=1, default=float)
