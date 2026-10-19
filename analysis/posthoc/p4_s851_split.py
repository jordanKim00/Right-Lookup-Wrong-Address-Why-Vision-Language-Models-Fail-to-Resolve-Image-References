"""S850 / S851 stored data split by cell type (post hoc, NOT registered; user 10-17 late evening, received after the S851 data were analysed).
No new run.  The registered verdicts stay those of exp/s850_analyze.py and exp/s851_analyze.py, which pooled 'stay = P(answer = p)' over all
nine cells.  The user's point: staying is correct only where the pinned early position equals the stated target (p = j*, j* = N_s - 3), and
an error elsewhere, so the two cell types need their own measures.
  identity cells (p = j*; 3 per scene)    stay = P(answer = p); departures split into p - 1 / p + 1 / other; target logit margin
  non-identity cells (p != j*; 6/scene)   acc = P(answer = j*); stay = P(answer = p) (an error here); toward = P(the answer moved from p
                                          in the direction of j*); target logit margin; logit margin of p
  late address (both types)               margin of j* (and of p) in the block scores of rows B1, B4 and answer at layers 51 / 55
  logit margin of x = l(x) - max of the other menu words.  j* follows the stated count (as registered in S846 / S850 / S851); with
  N_s = 8 it is also the target for the eight shown images.
Every measure is averaged over the cells of a type within a scene; contrasts are scene-paired differences; 95% intervals from 4,000
bootstrap resamples of the 30 independent scenes (not of the 90 / 180 runs).  Ratios (S851): s = (PLUSQ - PLUS16) / (FULLB - E_EARLY),
n = (FULLB - FULLBQE) / (FULLB - E_EARLY), ratio of scene means within each resample.
S851 answer agreement: the share of runs in which two arms give the same answer (does a query swap make the answer the donor's?).
Usage: python3 exp/p4_s851_split.py  ->  research/ces/outputs/s851_split.json"""
from __future__ import annotations

import json
import numpy as np

V = "/home/storage2/ai26269345/vlm"
B = 4000
PS = (3, 4, 5); NSS = (6, 7, 8); ROWS = {"B1": 2, "B4": 5, "answer": 9}
IDC = [(p + 3, p) for p in PS]
NONC = [(ns, p) for ns in NSS for p in PS if ns - 3 != p]


def load(fn):
    d = json.load(open(fn)); by = {(r["si"], r["Ns"]): r for r in d["rows"]}
    sis = sorted(s for s in {r["si"] for r in d["rows"]} if all((s, ns) in by for ns in NSS))
    return by, sis


def ans(r, arm, p):
    return r["land"][f"{arm}_p{p}"]


def lmar(r, arm, p, x):
    lg = np.array(r["mlog"][f"{arm}_p{p}"], dtype=float); ix = r["words"].index(r["labs"][x - 1])
    return float(lg[ix] - np.max(np.delete(lg, ix)))


def qmar(r, arm, p, L, row, x):
    z = np.array(r["qk"][f"{arm}_p{p}"][L][row], dtype=float)
    return float(z[x - 1] - np.max(np.delete(z, x - 1)))


MEAS = {
    "id": {
        "stay": lambda r, a, ns, p: float(ans(r, a, p) == p),
        "to_prev": lambda r, a, ns, p: float(ans(r, a, p) == p - 1),
        "to_next": lambda r, a, ns, p: float(ans(r, a, p) == p + 1),
        "lm_target": lambda r, a, ns, p: lmar(r, a, p, p),
    },
    "non": {
        "acc": lambda r, a, ns, p: float(ans(r, a, p) == ns - 3),
        "stay": lambda r, a, ns, p: float(ans(r, a, p) == p),
        "toward": lambda r, a, ns, p: float((ans(r, a, p) - p) * (ns - 3 - p) > 0),
        "lm_target": lambda r, a, ns, p: lmar(r, a, p, ns - 3),
        "lm_p": lambda r, a, ns, p: lmar(r, a, p, p),
    },
}
for _L in ("51", "55"):
    for _rn, _ri in ROWS.items():
        MEAS["id"][f"M{_L}_{_rn}_target"] = (lambda r, a, ns, p, L=_L, ri=_ri: qmar(r, a, p, L, ri, p))
        MEAS["non"][f"M{_L}_{_rn}_target"] = (lambda r, a, ns, p, L=_L, ri=_ri: qmar(r, a, p, L, ri, ns - 3))
        MEAS["non"][f"M{_L}_{_rn}_p"] = (lambda r, a, ns, p, L=_L, ri=_ri: qmar(r, a, p, L, ri, p))


def per_scene(by, sis, arm, cells, f):
    return np.array([np.mean([f(by[(s, ns)], arm, ns, p) for ns, p in cells]) for s in sis])


def summarize(by, sis, arms, contrasts, ratios, rng, has_qk):
    out = {}
    for typ, cells in (("id", IDC), ("non", NONC)):
        o = {"n_scenes": len(sis), "n_runs_per_arm": len(sis) * len(cells), "level": {}, "contrast": {}, "ratio": {}}
        for mn, f in MEAS[typ].items():
            if mn.startswith("M") and not has_qk:
                continue
            ps = {a: per_scene(by, sis, a, cells, f) for a in arms}
            o["level"][mn] = {a: float(ps[a].mean()) for a in arms}
            idx = rng.integers(0, len(sis), (B, len(sis)))
            for a1, a0 in contrasts:
                dif = ps[a1] - ps[a0]; bs = dif[idx].mean(1)
                o["contrast"][f"{mn}|{a1}-{a0}"] = [float(dif.mean()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]]
            for rn, (num1, num0, den1, den0) in ratios.items():
                num = ps[num1] - ps[num0]; den = ps[den1] - ps[den0]
                g = den[idx].mean(1); v = num[idx].mean(1) / np.where(np.abs(g) > 1e-9, g, np.nan)
                o["ratio"][f"{mn}|{rn}"] = [float(num.mean() / den.mean()) if abs(den.mean()) > 1e-9 else None,
                                            [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))],
                                            dict(den=float(den.mean()), den_ci=[float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))])]
        out[typ] = o
    return out


def agreement(by, sis, pairs, rng):
    out = {}
    for typ, cells in (("id", IDC), ("non", NONC), ("all", IDC + NONC)):
        for a1, a2 in pairs:
            ps = np.array([np.mean([ans(by[(s, ns)], a1, p) == ans(by[(s, ns)], a2, p) for ns, p in cells]) for s in sis])
            bs = ps[rng.integers(0, len(sis), (B, len(sis)))].mean(1)
            out[f"{typ}|{a1}={a2}"] = [float(ps.mean()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]]
    return out


def main():
    rng = np.random.default_rng(11)
    res = dict(note="post hoc, not registered (user request received after the S851 data were analysed); scene-paired, 30-scene bootstrap")
    by, sis = load(f"{V}/results/r3/s850_s850_27b_bstop.json")
    a850 = ("NOOP_B", "PLUS8", "PLUS16", "RAND8", "RAND16", "MINUS8", "FULLB", "S_EARLY")
    res["S850"] = summarize(by, sis, a850, [(a, "NOOP_B") for a in a850 if a != "NOOP_B"] + [("PLUS16", "RAND16"), ("PLUS8", "RAND8")], {}, rng,
                            has_qk=True)
    by, sis = load(f"{V}/results/r3/s851_s851_27b_path.json")
    a851 = ("E_EARLY", "S_EARLY", "FULLB", "PLUS16", "PLUSQ", "FULLBQE")
    con = [("FULLB", "E_EARLY"), ("PLUS16", "E_EARLY"), ("PLUSQ", "PLUS16"), ("FULLB", "FULLBQE"), ("PLUSQ", "FULLB"), ("FULLBQE", "E_EARLY"),
           ("S_EARLY", "E_EARLY")]
    rat = {"s": ("PLUSQ", "PLUS16", "FULLB", "E_EARLY"), "n": ("FULLB", "FULLBQE", "FULLB", "E_EARLY")}
    res["S851"] = summarize(by, sis, a851, con, rat, rng, has_qk=True)
    res["S851"]["agree"] = agreement(by, sis, [("PLUSQ", "FULLB"), ("FULLBQE", "E_EARLY"), ("PLUS16", "FULLB"), ("E_EARLY", "FULLB"),
                                               ("PLUSQ", "PLUS16"), ("FULLBQE", "FULLB"), ("PLUS16", "E_EARLY")], rng)
    json.dump(res, open(f"{V}/research/ces/outputs/s851_split.json", "w"), indent=1)
    f2 = lambda v: f"{v[0]:+.3f} [{v[1][0]:+.3f}, {v[1][1]:+.3f}]"
    for ex in ("S850", "S851"):
        for typ in ("id", "non"):
            o = res[ex][typ]; print(f"== {ex} {typ} cells ({o['n_runs_per_arm']} runs per arm, {o['n_scenes']} scenes)")
            for mn, lv in o["level"].items():
                if mn.startswith("M") and not any(t in mn for t in ("M51_B1", "M51_B4", "M51_answer")):
                    continue
                print(f"  {mn:22s} " + " ".join(f"{a}={v:+.3f}" for a, v in lv.items()))
            for k, v in o["contrast"].items():
                if k.split("|")[0] in ("stay", "acc", "toward", "lm_target", "to_prev", "to_next", "M51_B1_target"):
                    print(f"  d {k:34s} {f2(v)}")
            for k, v in o["ratio"].items():
                if k.split("|")[0] in ("stay", "acc", "toward", "lm_target"):
                    r0 = 'nan' if v[0] is None else f"{v[0]:+.2f}"
                    print(f"  ratio {k:24s} {r0} [{v[1][0]:+.2f}, {v[1][1]:+.2f}]  (den {v[2]['den']:+.3f} [{v[2]['den_ci'][0]:+.3f}, {v[2]['den_ci'][1]:+.3f}])")
    print("== S851 answer agreement (share of runs with the same answer)")
    for k, v in res["S851"]["agree"].items():
        print(f"  {k:26s} {v[0]:.3f} [{v[1][0]:.3f}, {v[1][1]:.3f}]")


if __name__ == "__main__":
    main()
