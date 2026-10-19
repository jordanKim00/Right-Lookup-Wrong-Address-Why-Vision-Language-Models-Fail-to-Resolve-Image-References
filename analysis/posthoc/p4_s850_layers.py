"""S850 stored-data analysis (post hoc, NOT registered; user 10-17 late evening): where does the steering effect lose its selective power?
No new run.  Per arm and late layer (51, 55, 59, 63), rows B4 and answer (head-averaged block scores: matcher heads at 51 / 55, all heads at
59 / 63 - per-head scores, gate outputs and attention outputs were not stored):
  M_l(x) = z_l[x] - max_{b != x} z_l[b] for x = p (the pinned early position = staying) and x = j* (the stated target N_s - 3; = p in match
           cells); level, change vs NOOP_B (scene bootstrap 95%), share of runs with M > 0
  share    relative attention on image x among the images, softmax over z_l (an approximation for head-averaged scores)
  mass     total picture mass of the row (share of attention on all image tokens)
  logit    menu-word margin of x: l(x) - max of the other menu words (the answer readout); share > 0 = answers x
  cond     P(answer = p | M_51(p) > 0) and P(answer = p | M_63(p) > 0) with their denominators (n runs), per arm
Cells: all 9, match 3 (p = j*), non-match 6.  30 independent scenes; bootstrap over scenes.
Usage: python3 exp/p4_s850_layers.py  ->  research/ces/outputs/s850_layers.json"""
from __future__ import annotations

import json
import numpy as np

V = "/home/storage2/ai26269345/vlm"
PS = (3, 4, 5); NSS = (6, 7, 8); LAYERS = ("51", "55", "59", "63"); ROWS = (5, 9)
ARMS = ("NOOP_B", "PLUS8", "PLUS16", "RAND8", "RAND16", "MINUS8", "FULLB", "S_EARLY")


def main():
    d = json.load(open(f"{V}/results/r3/s850_s850_27b_bstop.json")); R = d["rows"]
    by = {(r["si"], r["Ns"]): r for r in R}; sis = sorted({r["si"] for r in R})
    rng = np.random.default_rng(2)
    cells = {"all": [(ns, p) for ns in NSS for p in PS], "match": [(p + 3, p) for p in PS],
             "nonmatch": [(ns, p) for ns in NSS for p in PS if ns - 3 != p]}
    def zrow(r, arm, p, L, row):
        return np.array(r["qk"][f"{arm}_p{p}"][L][row], dtype=float)
    def margin(z, x):
        return float(z[x - 1] - max(z[i] for i in range(len(z)) if i != x - 1))
    def share(z, x):
        e = np.exp(z - z.max()); return float(e[x - 1] / e.sum())
    def lmargin(r, arm, p, x):
        lab = r["labs"]; words = r["words"]; lg = np.array(r["mlog"][f"{arm}_p{p}"])
        ix = words.index(lab[x - 1])
        return float(lg[ix] - max(lg[i] for i in range(len(lg)) if i != ix))
    def stat(arm, cs, f):
        """per-scene mean of f over cells and rows, then mean over scenes; f(r, arm, p, ns) -> float"""
        per = {s: np.mean([f(by[(s, ns)], arm, p, ns) for ns, p in cs]) for s in sis}
        return per
    def boot_diff(pa, pb):
        diff = np.array([pa[s] - pb[s] for s in sis])
        bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
        return [float(diff.mean()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]]
    out = dict(note="post hoc, not registered; head-averaged scores; gate / value outputs not stored", layers={}, logit={}, cond={})
    for cname, cs in cells.items():
        for L in LAYERS:
            for xname in ("p", "jstar"):
                xs = (lambda p, ns: p) if xname == "p" else (lambda p, ns: ns - 3)
                fM = lambda r, arm, p, ns, L=L, xs=xs: np.mean([margin(zrow(r, arm, p, L, row), xs(p, ns)) for row in ROWS])
                fS = lambda r, arm, p, ns, L=L, xs=xs: np.mean([share(zrow(r, arm, p, L, row), xs(p, ns)) for row in ROWS])
                fP = lambda r, arm, p, ns, L=L, xs=xs: float(np.mean([margin(zrow(r, arm, p, L, row), xs(p, ns)) > 0 for row in ROWS]))
                base = {k: stat("NOOP_B", cs, f) for k, f in (("M", fM), ("S", fS))}
                for arm in ARMS:
                    pm, ps_, pp = stat(arm, cs, fM), stat(arm, cs, fS), stat(arm, cs, fP)
                    out["layers"][f"{cname}|L{L}|{xname}|{arm}"] = dict(M=float(np.mean(list(pm.values()))), dM=boot_diff(pm, base["M"]),
                                                                       share=float(np.mean(list(ps_.values()))), dshare=boot_diff(ps_, base["S"]),
                                                                       pos=float(np.mean(list(pp.values()))))
            fmass = lambda r, arm, p, ns, L=L: np.mean([r["mass"][f"{arm}_p{p}"][L][row] for row in ROWS])
            mb = stat("NOOP_B", cs, fmass)
            for arm in ARMS:
                pmass = stat(arm, cs, fmass)
                out["layers"][f"{cname}|L{L}|mass|{arm}"] = dict(mass=float(np.mean(list(pmass.values()))), dmass=boot_diff(pmass, mb))
        for xname in ("p", "jstar"):
            xs = (lambda p, ns: p) if xname == "p" else (lambda p, ns: ns - 3)
            fL = lambda r, arm, p, ns, xs=xs: lmargin(r, arm, p, xs(p, ns))
            fA = lambda r, arm, p, ns, xs=xs: float(r["land"][f"{arm}_p{p}"] == xs(p, ns))
            lb, ab = stat("NOOP_B", cs, fL), stat("NOOP_B", cs, fA)
            for arm in ARMS:
                pl, pa = stat(arm, cs, fL), stat(arm, cs, fA)
                out["logit"][f"{cname}|{xname}|{arm}"] = dict(margin=float(np.mean(list(pl.values()))), dmargin=boot_diff(pl, lb),
                                                              answers=float(np.mean(list(pa.values()))), danswers=boot_diff(pa, ab))
    for arm in ARMS:
        rows = [(by[(s, ns)], p) for s in sis for ns, p in cells["all"]]
        m51 = np.array([np.mean([margin(zrow(r, arm, p, "51", row), p) for row in ROWS]) for r, p in rows])
        m63 = np.array([np.mean([margin(zrow(r, arm, p, "63", row), p) for row in ROWS]) for r, p in rows])
        st = np.array([r["land"][f"{arm}_p{p}"] == p for r, p in rows])
        out["cond"][arm] = dict(n=len(rows), stay=float(st.mean()), n_M51pos=int((m51 > 0).sum()), stay_given_M51pos=float(st[m51 > 0].mean()) if (m51 > 0).any() else None,
                                n_M63pos=int((m63 > 0).sum()), stay_given_M63pos=float(st[m63 > 0].mean()) if (m63 > 0).any() else None,
                                n_both=int(((m51 > 0) & (m63 > 0)).sum()), stay_given_both=float(st[(m51 > 0) & (m63 > 0)].mean()) if ((m51 > 0) & (m63 > 0)).any() else None)
    json.dump(out, open(f"{V}/research/ces/outputs/s850_layers.json", "w"), indent=1)
    for cname in ("all", "match"):
        print(f"== {cname} cells: margin for p (B4 + answer rows), level / change vs NOOP [95%] / share M>0 / image share")
        for arm in ("NOOP_B", "PLUS16", "RAND16", "FULLB", "S_EARLY"):
            line = []
            for L in LAYERS:
                v = out["layers"][f"{cname}|L{L}|p|{arm}"]
                line.append(f"L{L} {v['M']:+.2f} ({v['dM'][0]:+.2f} [{v['dM'][1][0]:+.2f},{v['dM'][1][1]:+.2f}]) pos {v['pos']:.2f} sh {v['share']:.2f}")
            print(f"  {arm:8s} " + " | ".join(line))
        for arm in ("NOOP_B", "PLUS16", "RAND16", "FULLB"):
            v = out["logit"][f"{cname}|p|{arm}"]
            print(f"  logit {arm:8s} margin {v['margin']:+.2f} (d {v['dmargin'][0]:+.2f} [{v['dmargin'][1][0]:+.2f},{v['dmargin'][1][1]:+.2f}]) answers p {v['answers']:.3f} (d {v['danswers'][0]:+.3f} [{v['danswers'][1][0]:+.3f},{v['danswers'][1][1]:+.3f}])")
        for arm in ("NOOP_B", "PLUS16", "FULLB"):
            line = [f"L{L} {out['layers'][f'{cname}|L{L}|mass|{arm}']['mass']:.3f}" for L in LAYERS]
            print(f"  mass {arm:8s} " + " ".join(line))
    print("== conditional stay (all cells)")
    for arm, v in out["cond"].items():
        print(f"  {arm:8s}", {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items()})
    print("== non-match cells: margin for the stated target j*")
    for arm in ("NOOP_B", "PLUS16", "FULLB"):
        line = [f"L{L} {out['layers'][f'nonmatch|L{L}|jstar|{arm}']['M']:+.2f}" for L in LAYERS]
        v = out["logit"][f"nonmatch|jstar|{arm}"]
        print(f"  {arm:8s} " + " ".join(line) + f" | answers j* {v['answers']:.3f}")


if __name__ == "__main__":
    main()
