"""S850 exploratory analyses (NOT registered: written at the user's request after S850 started and before any S850 data was read; the registered
verdicts stay those of exp/s850_analyze.py).  Scene bootstrap (4,000, seed 1; 30 independent scenes, not 270 runs).  k = 4; match m = 1[p = N_s - 3].
  SELECT   selectivity of strengthening for the goal-matched cells (user):
           Delta_match = [P(p | +u, m=1) - P(p | 0, m=1)] - [P(p | +u, m=0) - P(p | 0, m=0)], with 0 = NOOP_B (zero steering through the hook),
           for PLUS8 / PLUS16, and the same for RAND8 / RAND16 as controls.  A goal-attainment stop signal predicts Delta_match > 0; raising p
           everywhere (Delta_match ~ 0 with both terms > 0) reads as stronger positional anchoring, not goal detection.
  WITHIN_P the steering effect on staying per early position p across the stated counts (match cell vs the two non-match cells of the same p), so
           that a count-specific or position-specific sensitivity is not read as a match signal; and the frozen B1 direction's projection per cell
           (does the match cell stand out within its row and within its column?).
  WRITER_C per band component (46-49 mixer / MLP; FA47 = mixer 47, DN49 = mixer 49) and row (B1, B4, answer): the change of its output under
           PLUS16 and RAND16 relative to E_EARLY - size |dC| / |C_S - C_E| and projection onto the start-relative difference (C_S - C_E).
  CHAIN    per arm: writer response (projection), late hold margin change at 51 / 55, answer change - placed in the user's table (writer unchanged /
           writer changed but no late address change / late address changed but no answer change / all changed selectively).  Co-occurrence is
           not mediation; a mediation claim would need a block / restore intervention (as S847).
Usage: python3 exp/p4_s850_explore.py [TAG]  ->  research/ces/outputs/s850_explore.json"""
from __future__ import annotations

import json, sys
import numpy as np

V = "/home/storage2/ai26269345/vlm"
B = 4000
PS = (3, 4, 5); NSS = (6, 7, 8)
COMPN = ["46mix", "46mlp", "47mix(FA47)", "47mlp", "48mix", "48mlp", "49mix(DN49)", "49mlp"]


def main(tag="s850_27b"):
    d = json.load(open(f"{V}/results/r3/s850_{tag}_bstop.json")); R = d["rows"]
    Z = np.load(f"{V}/results/r3/s850_{tag}_bstop.npz")
    dz = np.load(f"{V}/research/ces/outputs/s850_boundary_dirs.npz")
    rng = np.random.default_rng(1)
    by = {ns: {r["si"]: r for r in R if r["Ns"] == ns} for ns in NSS}
    sis = sorted(set.intersection(*[set(x) for x in by.values()]))
    cells = [(ns, p) for ns in NSS for p in PS]; M1 = [(p + 3, p) for p in PS]; M0 = [c for c in cells if c not in M1]
    stay = lambda arm, cs, ss: float(np.mean([by[ns][s]["land"][f"{arm}_p{p}"] == p for ns, p in cs for s in ss]))
    def boot(f):
        v = [f([sis[i] for i in rng.integers(0, len(sis), len(sis))]) for _ in range(B)]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    out = dict(tag=tag, note="exploratory, not registered (user request after S850 started, before data)", n_scenes=len(sis))
    sel = {}
    for arm in ("PLUS8", "PLUS16", "RAND8", "RAND16", "MINUS8", "FULLB"):
        f = lambda ss, arm=arm: (stay(arm, M1, ss) - stay("NOOP_B", M1, ss)) - (stay(arm, M0, ss) - stay("NOOP_B", M0, ss))
        g1 = lambda ss, arm=arm: stay(arm, M1, ss) - stay("NOOP_B", M1, ss)
        g0 = lambda ss, arm=arm: stay(arm, M0, ss) - stay("NOOP_B", M0, ss)
        sel[arm] = dict(Delta_match=[f(sis), boot(f)], effect_match=[g1(sis), boot(g1)], effect_nonmatch=[g0(sis), boot(g0)])
    out["SELECT"] = sel
    wp = {}
    for p in PS:
        row = {}
        for arm in ("PLUS8", "PLUS16", "RAND16"):
            row[arm] = {str(ns): stay(arm, [(ns, p)], sis) - stay("NOOP_B", [(ns, p)], sis) for ns in NSS}
        row["NOOP_B"] = {str(ns): stay("NOOP_B", [(ns, p)], sis) for ns in NSS}
        row["match_Ns"] = p + 3
        wp[str(p)] = row
    m = dz["m_L46_r2"].astype(np.float64)
    proj = np.array([[np.mean([float(Z[f"Ns{ns}_s{s}_E_EARLY_p{p}_res"][d["res_layers"].index(46), 0].astype(np.float64) @ m) for s in sis])
                      for ns in NSS] for p in PS])
    pc = proj - proj.mean(1, keepdims=True) - proj.mean(0, keepdims=True) + proj.mean()
    out["WITHIN_P"] = dict(steering_effect=wp, B1_projection_cell_means=proj.tolist(), B1_projection_double_centred=pc.tolist())
    wc = {}
    for ci, cn in enumerate(COMPN):
        for ri, rn in enumerate(("B1", "B4", "answer")):
            size = {a: [] for a in ("PLUS16", "RAND16")}; prj = {a: [] for a in ("PLUS16", "RAND16")}
            for s in sis:
                for ns, p in cells:
                    CE = Z[f"Ns{ns}_s{s}_E_EARLY_p{p}_comp"][ci, ri].astype(np.float64); CS = Z[f"Ns{ns}_s{s}_S_EARLY_p{p}_comp"][ci, ri].astype(np.float64)
                    D = CS - CE; dn = float(D @ D) + 1e-12
                    for a in ("PLUS16", "RAND16"):
                        dC = Z[f"Ns{ns}_s{s}_{a}_p{p}_comp"][ci, ri].astype(np.float64) - CE
                        size[a].append(float(np.linalg.norm(dC) / np.sqrt(dn))); prj[a].append(float(dC @ D / dn))
            wc[f"{cn}_{rn}"] = {a: dict(rel_size=float(np.median(size[a])), proj_stop=float(np.mean(prj[a]))) for a in size}
    out["WRITER_C"] = wc
    def hold(arm, s, ns, p, L):
        v = []
        for row in (5, 9):
            z = by[ns][s]["qk"][f"{arm}_p{p}"][L][row]
            v.append(z[p - 1] - max(z[p - 2], z[p] if p < len(z) else -1e9))
        return float(np.mean(v))
    chain = {}
    for arm in ("PLUS8", "PLUS16", "RAND8", "RAND16", "MINUS8", "FULLB"):
        dh = {L: float(np.mean([hold(arm, s, ns, p, L) - hold("NOOP_B", s, ns, p, L) for s in sis for ns, p in cells])) for L in ("51", "55")}
        dh_m = {L: float(np.mean([hold(arm, s, ns, p, L) - hold("NOOP_B", s, ns, p, L) for s in sis for ns, p in M1])) for L in ("51", "55")}
        chg = float(np.mean([by[ns][s]["land"][f"{arm}_p{p}"] != by[ns][s]["land"][f"NOOP_B_p{p}"] for s in sis for ns, p in cells]))
        chain[arm] = dict(late_hold_change=dh, late_hold_change_match=dh_m, answers_changed=chg, stay_change_all=stay(arm, cells, sis) - stay("NOOP_B", cells, sis),
                          stay_change_match=stay(arm, M1, sis) - stay("NOOP_B", M1, sis))
    out["CHAIN"] = chain
    json.dump(out, open(f"{V}/research/ces/outputs/s850_explore{'' if tag == 's850_27b' else '_' + tag}.json", "w"), indent=1)
    print(json.dumps(out["SELECT"])); print(json.dumps(out["WITHIN_P"])); print(json.dumps(out["CHAIN"]))
    print(json.dumps({k: v for k, v in out["WRITER_C"].items() if "FA47" in k or "DN49" in k}))


if __name__ == "__main__":
    main(*sys.argv[1:])
