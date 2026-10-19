"""S830 analysis (rules registered in knowledge/DECISIONS.md with the run, before any S830 data).  Cases N7 END(4), N9 END(5); probe positions
p (N7 {2, 3, 5, 6, 7}, N9 {2, 3, 4, 6, 7, 8, 9}).  Shares are over scenes and probe positions pooled (scene bootstrap, 4,000 draws, seed 0).
  I0  t1 >= .95; QNOOP_E = END and QNOOP_S = START in >= .90 of scenes with the 95th percentile of the largest menu-logit difference <= .5;
      START(k) -> k >= .90; every donor START(p) -> p >= .90 (pooled over p, and >= .80 for each p).
  F1  (END's answer after the early lookup is moved to p) pooled shares of E_EARLY_p landing on p - 1 ('one before'), on p ('copy'), on END(k)'s
      own natural answer of that scene ('unmoved'): 'END answers one before the early lookup' if one-before >= .40 and exceeds copy and unmoved by
      >= .15 (interval of the difference above 0); 'END follows the early lookup' if copy >= .40 and exceeds the other two by >= .15; 'END ignores
      the early lookup' if unmoved >= .50 and exceeds the other two by >= .15; otherwise 'mixed'.
  F2  (control) the same three shares for S_EARLY_p with START(k)'s own answer (k) as 'unmoved': 'START follows the early lookup' / 'START also
      steps back' / 'START keeps its ordinal' / 'mixed' by the same rule.
  F3  E_LATE_p -> p pooled >= .60 (interval's lower end > .45) -> 'the late address sets the answer'.
  F4  (internal) E_EARLY_p: the late (L55) address of the boundary newline rows and the answer row (index-matching-head mean block scores, B1, B4,
      answer row mean) has its maximum on p - 1 in >= .50 of (scene, p) -> 'the late address is written one before the early lookup'.
Descriptive: per-p landing shares, the mean landing against p (slope, intercept) for E_EARLY / S_EARLY / E_LATE, the late address argmax table.
Usage: python s830_analyze.py TAG"""
from __future__ import annotations

import json, sys
import numpy as np

B = 4000
PSET = {7: [2, 3, 5, 6, 7], 9: [2, 3, 4, 6, 7, 8, 9]}


def analyze(tag):
    d = json.load(open(f"/vlm/results/r3/s830_{tag}_anchor.json")); R = d["rows"]; rng = np.random.default_rng(0)
    look = d["look"]; L55 = str(min(look, key=lambda l: abs(l - 55)))
    rep = dict(tag=tag, model=d["model"], t1=d["t1_menu"], n=len(R), key_rel_diff_max=d.get("key_rel_diff_max"), cases={})
    def boot(rows, f):
        v = []
        for _ in range(B):
            pick = rng.choice(len(rows), len(rows)); v.append(f([rows[i] for i in pick]))
        v = np.array(v, float); v = v[np.isfinite(v)]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else None
    late = lambda r, run: np.mean([np.array(r["qk"][run][L55][i]) for i in (2, 5, 9)], 0)
    for N, k in ((7, 4), (9, 5)):
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
        def shares(q, pref, own):
            one = np.mean([r["land"][f"{pref}_p{p}"] == p - 1 for r in q for p in ps])
            cp = np.mean([r["land"][f"{pref}_p{p}"] == p for r in q for p in ps])
            un = np.mean([r["land"][f"{pref}_p{p}"] == own(r) for r in q for p in ps])
            return float(one), float(cp), float(un)
        def verdict(q, pref, own, names):
            one, cp, un = shares(q, pref, own)
            vals = dict(one=one, copy=cp, unmoved=un)
            diff = lambda a_, b_: boot(q, lambda qq: shares(qq, pref, own)[a_] - shares(qq, pref, own)[b_])
            out = dict(shares=vals)
            for i_, (nm, thr) in enumerate((("one", .40), ("copy", .40), ("unmoved", .50))):
                others = [j_ for j_ in range(3) if j_ != i_]
                v_ = [one, cp, un][i_]
                if v_ >= thr and all(v_ >= [one, cp, un][j_] + .15 for j_ in others):
                    cis = [diff(i_, j_) for j_ in others]
                    if all(ci and ci[0] > 0 for ci in cis):
                        out["verdict"] = names[i_]; out["ci"] = cis
                        return out
            out["verdict"] = "mixed"
            return out
        c["F1"] = verdict(rows, "E_EARLY", lambda r: r["land"]["END"], ("END answers one before the early lookup", "END follows the early lookup", "END ignores the early lookup"))
        c["F2"] = verdict(rows, "S_EARLY", lambda r: k, ("START also steps back", "START follows the early lookup", "START keeps its ordinal"))
        fl = lambda q: float(np.mean([r["land"][f"E_LATE_p{p}"] == p for r in q for p in ps]))
        flv, flc = fl(rows), boot(rows, fl)
        c["F3"] = dict(share=flv, ci=flc, verdict=("the late address sets the answer" if (flv >= .60 and flc and flc[0] > .45) else "not shown"))
        am = [int(np.argmax(late(r, f"E_EARLY_p{p}"))) + 1 == p - 1 for r in rows for p in ps]
        c["F4"] = dict(share=float(np.mean(am)), verdict=("the late address is written one before the early lookup" if np.mean(am) >= .50 else "not shown"),
                       argmax_table={p: np.bincount([int(np.argmax(late(r, f"E_EARLY_p{p}"))) + 1 for r in rows], minlength=N + 1).tolist() for p in ps})
        desc = {}
        for pref in ("E_EARLY", "S_EARLY", "E_LATE"):
            per = {p: np.bincount([r["land"][f"{pref}_p{p}"] for r in rows], minlength=N + 1).tolist() for p in ps}
            xs = [p for r in rows for p in ps if r["land"][f"{pref}_p{p}"] >= 1]; ys = [r["land"][f"{pref}_p{p}"] for r in rows for p in ps if r["land"][f"{pref}_p{p}"] >= 1]
            sl, ic = np.polyfit(xs, ys, 1) if len(set(xs)) > 1 else (float("nan"), float("nan"))
            desc[pref] = dict(per_p=per, slope=float(sl), intercept=float(ic))
        c["desc"] = desc
        c["landing_nat"] = {nm: np.bincount([r["land"][nm] for r in rows], minlength=N + 1).tolist() for nm in ("END", "START", "QNOOP_E", "QNOOP_S")}
        rep["cases"][f"N{N}k{k}"] = c
    return rep


if __name__ == "__main__":
    tag = sys.argv[1]
    rep = analyze(tag)
    print(f"== {tag} ({rep['model']}) t1 {rep['t1']} n {rep['n']} key diff {rep['key_rel_diff_max']}")
    for c, v in rep["cases"].items():
        print(f"   {c} I0 {v['I0']}")
        for key in ("F1", "F2", "F3"):
            print(f"   {c} {key} {v[key]}")
        print(f"   {c} F4 share {v['F4']['share']:.3f} {v['F4']['verdict']} argmax {v['F4']['argmax_table']}")
        for pref, x in v["desc"].items():
            print(f"      {pref}: slope {x['slope']:.2f} intercept {x['intercept']:.2f} per p {x['per_p']}")
        print(f"   {c} natural {v['landing_nat']}")
    # 10-17 08:5x (S838): output path only - other tags write their own file instead of overwriting the S830 analysis (rules unchanged)
    json.dump(rep, open("/vlm/results/r3/s830_analysis.json" if tag == "s830_27b" else f"/vlm/results/r3/s830_analysis_{tag}.json", "w"), indent=1, default=float)
