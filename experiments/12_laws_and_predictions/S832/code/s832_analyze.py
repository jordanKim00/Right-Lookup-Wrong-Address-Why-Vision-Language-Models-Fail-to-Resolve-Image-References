"""S832 analysis (rules registered in knowledge/DECISIONS.md with the run, before any S832 data).  Frozen law research/ces/outputs/
s832_frozen_law.json (sha256 21ae68b80e03fde1...).  Cells (N, k, j): N8 END(6) (j 3), N8 END(3) (j 6); probes p as in exp/s832_newcells.py.
  I0  t1 >= .95; QNOOP_E = END in >= .90 of scenes with the 95th percentile of the largest menu-logit difference <= .5; START(k) -> k >= .90;
      donors START(p) -> p pooled >= .90 and each >= .80.
  P1  (calibration) share of (cell x probe x picture 1..N) whose predicted share is within .15 of the observed share >= .80 -> 'the frozen law
      predicts the anchor shifts in new cells'.
  P2  (modes) the predicted most likely picture equals the observed one in >= .50 of (cell x probe).
  P3  (information) mean log-probability per landing of the law (anchor = p) minus that of the cell's own natural END distribution (+.5
      smoothing), scene bootstrap interval above 0 -> 'informative'; minus that of the law with the anchor left at the copy k, interval above 0
      -> 'the forced anchor adds information'.
  P4  (natural cells) the law with a = k predicts every picture's natural END share within .15 -> 'natural cells predicted'.
Scene bootstrap, 4,000 draws, seed 0.  Usage: python s832_analyze.py TAG"""
from __future__ import annotations

import hashlib, json, sys
import numpy as np

B = 4000
CASES = ((8, 6, 3), (8, 3, 6))
PSET = {(8, 6): [2, 3, 4, 5, 7, 8], (8, 3): [2, 4, 5, 6, 7, 8]}


def law_dist(L, N, a, j):
    s, e, w = np.array(L["step"]), np.array(L["end"]), np.array(L["weights"])
    p = np.zeros(N + 1)
    dirs = [int(np.sign(j - a))] if j != a else [1, -1]
    for dr in dirs:
        for o, so in zip(L["off_step"], s):
            x = a + o * dr
            if 1 <= x <= N:
                p[x] += w[0] * so / len(dirs)
    for dd, ed in zip(L["off_end"], e):
        x = j + dd
        if 1 <= x <= N:
            p[x] += w[1] * ed
    p[1:] += w[2] / N
    p[1:] = p[1:] / p[1:].sum()
    return p


def analyze(tag):
    raw = open("/vlm/research/ces/outputs/s832_frozen_law.json").read(); L = json.loads(raw)
    rep = dict(tag=tag, law_sha256=hashlib.sha256(raw.encode()).hexdigest())
    d = json.load(open(f"/vlm/results/r3/s832_{tag}_anchor.json")); R = d["rows"]; rng = np.random.default_rng(0)
    rep.update(model=d["model"], t1=d["t1_menu"], n=len(R), key_rel_diff_max=d.get("key_rel_diff_max"), cases={})
    def boot(rows, f):
        v = []
        for _ in range(B):
            pick = rng.choice(len(rows), len(rows)); v.append(f([rows[i] for i in pick]))
        v = np.array(v, float); v = v[np.isfinite(v)]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else None
    hits, modes, allrows = [], [], []
    nat_ok = []
    for N, k, j in CASES:
        rows = [r for r in R if r["N"] == N and r["k"] == k]
        if not rows:
            continue
        ps = PSET[(N, k)]
        p95 = float(np.percentile([max(abs(x - y) for x, y in zip(r["mlog"]["QNOOP_E"], r["mlog"]["END"])) for r in rows], 95))
        agree = float(np.mean([r["land"]["QNOOP_E"] == r["land"]["END"] for r in rows]))
        dacc = {p: float(np.mean([r["land"][f"START_p{p}"] == p for r in rows])) for p in ps}
        i0 = dict(qnoop_agree=agree, qnoop_p95=p95, start_acc=float(np.mean([r["land"]["START"] == k for r in rows])), donor_acc=dacc,
                  donor_pooled=float(np.mean(list(dacc.values()))))
        i0["ok"] = bool((d["t1_menu"] or 0) >= .95 and agree >= .90 and p95 <= .5 and i0["start_acc"] >= .90 and i0["donor_pooled"] >= .90 and min(dacc.values()) >= .80)
        c = dict(I0=i0, probes={})
        for p in ps:
            pr = law_dist(L, N, p, j)
            ob = np.bincount([r["land"][f"E_EARLY_p{p}"] for r in rows], minlength=N + 1) / len(rows)
            hh = [abs(pr[x] - ob[x]) <= .15 for x in range(1, N + 1)]; hits.extend(hh)
            pm, om = int(np.argmax(pr[1:])) + 1, int(np.argmax(ob[1:])) + 1; modes.append(pm == om)
            c["probes"][p] = dict(pred=np.round(pr[1:], 3).tolist(), obs=np.round(ob[1:], 3).tolist(), within=float(np.mean(hh)), pred_mode=pm, obs_mode=om)
        prn = law_dist(L, N, k, j); obn = np.bincount([r["land"]["END"] for r in rows], minlength=N + 1) / len(rows)
        okn = all(abs(prn[x] - obn[x]) <= .15 for x in range(1, N + 1)); nat_ok.append(okn)
        c["natural"] = dict(pred=np.round(prn[1:], 3).tolist(), obs=np.round(obn[1:], 3).tolist(), all_within=okn)
        allrows.extend([(N, k, j, r) for r in rows])
        rep["cases"][f"N{N}k{k}j{j}"] = c
    def gains(q):
        g1, g2 = [], []
        for N, k, j, r in q:
            natd = np.bincount([rr["land"]["END"] for (n_, k_, j_, rr) in allrows if n_ == N and k_ == k and rr["land"]["END"] >= 1], minlength=N + 1)[1:] + .5
            natd = np.concatenate([[0], natd / natd.sum()]); pc = law_dist(L, N, k, j)
            for p in PSET[(N, k)]:
                x = r["land"][f"E_EARLY_p{p}"]
                if x >= 1:
                    lp = np.log(max(law_dist(L, N, p, j)[x], 1e-9))
                    g1.append(lp - np.log(max(natd[x], 1e-9))); g2.append(lp - np.log(max(pc[x], 1e-9)))
        return float(np.mean(g1)), float(np.mean(g2))
    gv = gains(allrows)
    ci1 = boot(allrows, lambda q: gains(q)[0]); ci2 = boot(allrows, lambda q: gains(q)[1])
    rep["P1"] = dict(share=float(np.mean(hits)), verdict=("the frozen law predicts the anchor shifts in new cells" if np.mean(hits) >= .80 else "not shown"))
    rep["P2"] = dict(share=float(np.mean(modes)), verdict=("modes predicted" if np.mean(modes) >= .50 else "not shown"))
    rep["P3"] = dict(gain_vs_natural=gv[0], ci_natural=ci1, gain_vs_copy_anchor=gv[1], ci_copy=ci2,
                     informative=bool(ci1 and ci1[0] > 0), anchor_adds=bool(ci2 and ci2[0] > 0))
    rep["P4"] = dict(cells_ok=nat_ok, verdict=("natural cells predicted" if nat_ok and all(nat_ok) else "not shown"))
    return rep


if __name__ == "__main__":
    tag = sys.argv[1]
    rep = analyze(tag)
    print(f"== {tag} ({rep['model']}) t1 {rep['t1']} n {rep['n']} key diff {rep['key_rel_diff_max']} law sha {rep['law_sha256'][:16]}")
    for c, v in rep["cases"].items():
        print(f"   {c} I0 {v['I0']}")
        for p, x in v["probes"].items():
            print(f"      p {p}: pred mode {x['pred_mode']} obs mode {x['obs_mode']} within {x['within']:.2f}  pred {x['pred']}  obs {x['obs']}")
        print(f"   {c} natural {v['natural']}")
    print(f"   P1 {rep['P1']}  P2 {rep['P2']}")
    print(f"   P3 {rep['P3']}")
    print(f"   P4 {rep['P4']}")
    json.dump(rep, open("/vlm/results/r3/s832_analysis.json", "w"), indent=1, default=float)
