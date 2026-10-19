"""S831 analysis (rules registered in knowledge/DECISIONS.md with the run, before any S831 data).  Cases (N, k, j): N7 (5, 3), N9 (7, 3); probes
p (N7 {2, 3, 4, 6, 7}, N9 {2, 3, 4, 5, 6, 8, 9}).  For a probe p, 'toward the target' = p + sign(j - p), 'toward the ordinal' = p + sign(k - p),
'one back' = p - 1.  Shares pooled over scenes and the listed probes; scene bootstrap, 4,000 draws, seed 0.
  I0  t1 >= .95; QNOOP_E = END and QNOOP_S = START in >= .90 of scenes with the 95th percentile of the largest menu-logit difference <= .5;
      START(k) -> k >= .90; donors START(p) -> p pooled >= .90 and each >= .80.
  G1  (target vs ordinal) probes strictly between j and k: share of E_EARLY_p at p + sign(j - p) vs at p + sign(k - p): 'toward the target' if the
      first >= .30 and exceeds the second by >= .15 (interval of the difference above 0); 'toward the ordinal' if the reverse; otherwise 'mixed'.
  G2  (toward the target vs one back) probes below j: share at p + 1 vs at p - 1: 'toward the target' if the first exceeds the second by >= .15
      (interval above 0); 'one back' if the reverse; otherwise 'mixed'.
  G3  (no zero step) probe p = j: P(j - 1) + P(j + 1) >= P(j) + .10 (interval above 0) -> 'moves even at the target'.
  G4  S_EARLY_p -> k pooled >= .50 -> 'START keeps its ordinal'.
  G5  E_LATE_p -> p pooled >= .60 (interval's lower end > .45) -> 'the late address sets the answer'.
  RULE (all probes but p = j): the share at p + sign(j - p) is the largest of the offsets -2..+2 and >= .35 -> 'one step toward the target'.
Descriptive: per-probe landing tables, offset shares by side, mean landing vs p.
Usage: python s831_analyze.py TAG"""
from __future__ import annotations

import json, sys
import numpy as np

B = 4000
PSET = {7: [2, 3, 4, 6, 7], 9: [2, 3, 4, 5, 6, 8, 9]}
CASES = ((7, 5, 3), (9, 7, 3))
sgn = lambda x: (x > 0) - (x < 0)


def analyze(tag):
    d = json.load(open(f"/vlm/results/r3/s831_{tag}_anchor.json")); R = d["rows"]; rng = np.random.default_rng(0)
    rep = dict(tag=tag, model=d["model"], t1=d["t1_menu"], n=len(R), key_rel_diff_max=d.get("key_rel_diff_max"), cases={})
    def boot(rows, f):
        v = []
        for _ in range(B):
            pick = rng.choice(len(rows), len(rows)); v.append(f([rows[i] for i in pick]))
        v = np.array(v, float); v = v[np.isfinite(v)]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else None
    for N, k, j in CASES:
        rows = [r for r in R if r["N"] == N]
        if not rows:
            continue
        ps = PSET[N]
        p95 = lambda a1, a2: float(np.percentile([max(abs(x - y) for x, y in zip(r["mlog"][a1], r["mlog"][a2])) for r in rows], 95))
        agree = lambda a1, a2: float(np.mean([r["land"][a1] == r["land"][a2] for r in rows]))
        i0 = {f"{a1}={a2}": dict(agree=agree(a1, a2), p95=p95(a1, a2)) for a1, a2 in (("QNOOP_E", "END"), ("QNOOP_S", "START"))}
        i0["start_acc"] = float(np.mean([r["land"]["START"] == k for r in rows]))
        dacc = {p: float(np.mean([r["land"][f"START_p{p}"] == p for r in rows])) for p in ps}
        i0["donor_acc"] = dacc; i0["donor_pooled"] = float(np.mean(list(dacc.values())))
        i0["ok"] = bool((d["t1_menu"] or 0) >= .95 and all(x["agree"] >= .90 and x["p95"] <= .5 for nm, x in i0.items() if "=" in nm)
                        and i0["start_acc"] >= .90 and i0["donor_pooled"] >= .90 and min(dacc.values()) >= .80)
        c = dict(I0=i0)
        share = lambda q, pp, tgt: float(np.mean([r["land"][f"E_EARLY_p{p}"] == tgt(p) for r in q for p in pp])) if pp else float("nan")
        mid = [p for p in ps if min(j, k) < p < max(j, k)]
        tj = lambda p: p + sgn(j - p); tk = lambda p: p + sgn(k - p)
        a1, a2 = share(rows, mid, tj), share(rows, mid, tk); ci = boot(rows, lambda q: share(q, mid, tj) - share(q, mid, tk))
        g1 = ("toward the target" if (a1 >= .30 and a1 - a2 >= .15 and ci and ci[0] > 0) else "toward the ordinal" if (a2 >= .30 and a2 - a1 >= .15 and ci and ci[1] < 0) else "mixed")
        c["G1"] = dict(probes=mid, toward_target=a1, toward_ordinal=a2, ci=ci, verdict=g1)
        low = [p for p in ps if p < j]
        b1, b2 = share(rows, low, lambda p: p + 1), share(rows, low, lambda p: p - 1); ci2 = boot(rows, lambda q: share(q, low, lambda p: p + 1) - share(q, low, lambda p: p - 1))
        c["G2"] = dict(probes=low, plus1=b1, minus1=b2, ci=ci2, verdict=("toward the target" if (b1 - b2 >= .15 and ci2 and ci2[0] > 0) else "one back" if (b2 - b1 >= .15 and ci2 and ci2[1] < 0) else "mixed"))
        if j in ps:
            f3 = lambda q: float(np.mean([r["land"][f"E_EARLY_p{j}"] in (j - 1, j + 1) for r in q]) - np.mean([r["land"][f"E_EARLY_p{j}"] == j for r in q]))
            v3, c3 = f3(rows), boot(rows, f3)
            c["G3"] = dict(diff=v3, ci=c3, verdict=("moves even at the target" if (v3 >= .10 and c3 and c3[0] > 0) else "not shown"),
                           land=np.bincount([r["land"][f"E_EARLY_p{j}"] for r in rows], minlength=N + 1).tolist())
        g4 = float(np.mean([r["land"][f"S_EARLY_p{p}"] == k for r in rows for p in ps]))
        c["G4"] = dict(share=g4, verdict=("START keeps its ordinal" if g4 >= .50 else "not shown"))
        f5 = lambda q: float(np.mean([r["land"][f"E_LATE_p{p}"] == p for r in q for p in ps]))
        v5, c5 = f5(rows), boot(rows, f5)
        c["G5"] = dict(share=v5, ci=c5, verdict=("the late address sets the answer" if (v5 >= .60 and c5 and c5[0] > .45) else "not shown"))
        nz = [p for p in ps if p != j]
        offs = {o: float(np.mean([r["land"][f"E_EARLY_p{p}"] - p == o for r in rows for p in nz if r["land"][f"E_EARLY_p{p}"] >= 1])) for o in (-2, -1, 0, 1, 2)}
        tshare = share(rows, nz, tj)
        # the offset toward the target, pooled: offset sign relative to the target direction
        rel = [(r["land"][f"E_EARLY_p{p}"] - p) * sgn(j - p) for r in rows for p in nz if r["land"][f"E_EARLY_p{p}"] >= 1]
        relh = {o: float(np.mean([x == o for x in rel])) for o in (-2, -1, 0, 1, 2)}
        c["RULE"] = dict(toward_target_share=tshare, offsets_toward_target=relh,
                         verdict=("one step toward the target" if (tshare >= .35 and relh[1] == max(relh.values())) else "not shown"))
        c["per_probe"] = {pref: {p: np.bincount([r["land"][f"{pref}_p{p}"] for r in rows], minlength=N + 1).tolist() for p in ps} for pref in ("E_EARLY", "S_EARLY", "E_LATE")}
        c["landing_nat"] = {nm: np.bincount([r["land"][nm] for r in rows], minlength=N + 1).tolist() for nm in ("END", "START", "QNOOP_E", "QNOOP_S")}
        rep["cases"][f"N{N}k{k}j{j}"] = c
    return rep


if __name__ == "__main__":
    tag = sys.argv[1]
    rep = analyze(tag)
    print(f"== {tag} ({rep['model']}) t1 {rep['t1']} n {rep['n']} key diff {rep['key_rel_diff_max']}")
    for c, v in rep["cases"].items():
        print(f"   {c} I0 {v['I0']}")
        for key in ("G1", "G2", "G3", "G4", "G5", "RULE"):
            if key in v:
                print(f"   {c} {key} {v[key]}")
        for pref, x in v["per_probe"].items():
            print(f"      {pref} {x}")
        print(f"   {c} natural {v['landing_nat']}")
    json.dump(rep, open("/vlm/results/r3/s831_analysis.json" if tag == "s831_27b" else f"/vlm/results/r3/s831_analysis_{tag}.json", "w"), indent=1, default=float)
