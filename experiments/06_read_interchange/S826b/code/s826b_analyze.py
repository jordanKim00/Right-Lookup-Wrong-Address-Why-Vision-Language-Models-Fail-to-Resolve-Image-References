"""S826b analysis (S826 re-run, seed 82610; exactness gate set before the S826b data to landing agreement >= .90 (S826 failed at .917 from
near-tie noise among scattered N = 8 answers) with the 95th percentile <= .5; otherwise the S826 rules registered in knowledge/DECISIONS.md with the run, before any S826 data; the S815 rules with the count token in place of
the frame word).  Recipient END(k) told 8 ('END' in the file), donor END(k) told 7 ('START' in the file), 8 pictures, k in {5, 6, 7}.  Follow index
phi = (x - o).(p - o) / |p - o|^2 over the eight picture words' centred menu logits (o = told 8, p = told 7), cells with |p - o|^2 >= 1.
Scene bootstrap, 4,000 draws, seed 0.
  I0  t1 >= .95; NOOP lands as the recipient in >= .95 of cells, SRC_0 as the donor in >= .95; the 95th percentile over cells of the largest
      menu-logit differences <= .5.
  W   count-reading window, E(l) = phi(SRC_l): L_end = the deepest swap layer such that every swap layer up to it has E >= .25 E(0) with the
      interval's lower end > .10; 'late' if L_end >= 43, 'early' if L_end <= 31, 'intermediate' otherwise; 'non-monotone' / 'graded' flags as S815.
  H   who holds the count at layer 35: a = phi(ANS_35), q = phi(QMB_35): 'answer side' a >= .5 and q < .25; 'question side' q >= .5 and a < .25;
      'both' both >= .5; 'neither' both < .25; 'mixed' otherwise.  Parts (descriptive): QT_l (count token to '?') and BD_l (boundary rows).
  D   direct reading by full attention: KO_l (the answer side cannot see the count token at full-attention layers >= l, l in {27, 35}) flips the
      answer in >= .20 of cells (interval's lower end > .10) -> 'needed after l'.
Usage: python s826_analyze.py TAG [TAG ...]"""
from __future__ import annotations

import json, sys
import numpy as np

B = 4000


def analyze(tag):
    d = json.load(open(f"/vlm/results/r3/s826b_{tag}_count.json")); R = d["rows"]; rng = np.random.default_rng(0)
    L51 = str(min(d["look"], key=lambda l: abs(l - 51)))
    nl = d.get("n_layer") or 64
    mp = lambda l: l if nl == 64 else int(round(l / 63 * (nl - 1)))
    rep = dict(tag=tag, model=d["model"], t1=d["t1_menu"], n=len(R), layer=L51)
    P = lambda rows, f: float(np.mean([f(r) for r in rows])) if rows else float("nan")
    def boot(rows, f):
        si = np.array([r["si"] for r in rows]); us = np.unique(si); v = []
        for _ in range(B):
            pick = rng.choice(us, len(us)); v.append(f([rows[i] for s in pick for i in np.flatnonzero(si == s)]))
        v = np.array(v, float); v = v[np.isfinite(v)]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else None
    cen = lambda x: np.array(x[:8], float) - np.mean(x[:8])
    def phi_cell(r, arm):
        o, p, x = cen(r["mlog"]["END"]), cen(r["mlog"]["START"]), cen(r["mlog"][arm])
        dn = float((p - o) @ (p - o))
        return float((x - o) @ (p - o) / dn) if dn >= 1.0 else float("nan")
    def phi(rows, arm):
        v = [phi_cell(r, arm) for r in rows]; v = [x for x in v if np.isfinite(x)]
        return float(np.mean(v)) if v else float("nan")
    arms = [a for a in R[0]["land"] if a not in ("END", "START")]
    # I0
    noop = P(R, lambda r: r["land"]["NOOP"] == r["land"]["END"]); s0 = P(R, lambda r: r["land"]["SRC_0"] == r["land"]["START"])
    dmax = lambda a1, a2: np.array([max(abs(x - y) for x, y in zip(r["mlog"][a1], r["mlog"][a2])) for r in R])
    dn_, d0_ = dmax("NOOP", "END"), dmax("SRC_0", "START")
    md, md0 = float(np.percentile(dn_, 95)), float(np.percentile(d0_, 95))
    rep["I0"] = dict(noop=noop, noop_menu_diff_p95=md, noop_menu_diff_max=float(dn_.max()), src0_is_start=s0, src0_menu_diff_p95=md0,
                     src0_menu_diff_max=float(d0_.max()), eligible=int(sum(np.isfinite(phi_cell(r, "SRC_0")) for r in R)),
                     ok=bool((d["t1_menu"] or 0) >= .95 and noop >= .90 and md <= .5 and s0 >= .90 and md0 <= .5))
    # W
    SL = d["src_layers"]; E = {l: phi(R, f"SRC_{l}") for l in SL}; CI = {l: boot(R, lambda rr, l=l: phi(rr, f"SRC_{l}")) for l in SL}
    meets = {l: bool(E[l] >= .25 * E[0] and CI[l] and CI[l][0] > .10) for l in SL}
    L_end, L75 = None, None
    for l in SL:
        if not meets[l]:
            break
        L_end = l
    for l in SL:
        if E[l] < .75 * E[0]:
            break
        L75 = l
    wv = None if L_end is None else ("late" if L_end >= mp(43) else "early" if L_end <= mp(31) else "intermediate")
    rep["W"] = dict(E=E, ci=CI, L_end=L_end, L75=L75, verdict=wv,
                    non_monotone=bool(L_end is not None and any(meets[l] for l in SL if l > L_end)),
                    graded=bool(L_end is not None and L75 is not None and L_end - L75 >= 12))
    # H
    A35, Q35, S35 = f"ANS_{mp(35)}", f"QMB_{mp(35)}", f"SRC_{mp(35)}"
    a, q = phi(R, A35), phi(R, Q35)
    hv = ("answer side" if (a >= .5 and q < .25) else "question side" if (q >= .5 and a < .25) else "both" if (a >= .5 and q >= .5)
          else "neither" if (a < .25 and q < .25) else "mixed")
    rep["H"] = dict(ANS_35=a, QMB_35=q, ci_a_minus_q=boot(R, lambda rr: phi(rr, A35) - phi(rr, Q35)), verdict=hv,
                    later={x: phi(R, x) for x in (f"ANS_{mp(43)}", f"ANS_{mp(51)}", f"QMB_{mp(43)}", f"QMB_{mp(51)}")})
    # D
    dk = {}
    for l in d["ko_layers"]:
        arm = f"KO_{l}"; fl = lambda rr, arm=arm: P(rr, lambda r: r["land"][arm] != r["land"]["END"])
        c = boot(R, fl)
        dk[arm] = dict(flip=fl(R), ci=c, phi=phi(R, arm), needed=bool(fl(R) >= .20 and c and c[0] > .10))
    rep["D"] = dk
    rep["hypothesis"] = ("source re-read late" if rep["W"]["verdict"] == "late" else {"answer side": "answer-side state", "question side": "question-side state",
                         "neither": "no single holder", "both": "both sides", "mixed": "mixed"}[hv])
    # descriptive
    ks = sorted({r["k"] for r in R})
    rep["phi"] = {a_: dict(all=phi(R, a_), **{f"k{k}": phi([r for r in R if r["k"] == k], a_) for k in ks}) for a_ in arms}
    rep["acc"] = {f"k{k}": {a_: P([r for r in R if r["k"] == k], lambda r, a_=a_: r["land"][a_] == r["j"]) for a_ in arms + ["END", "START"]} for k in ks}
    rep["literal_k"] = {f"k{k}": {a_: P([r for r in R if r["k"] == k], lambda r, a_=a_: r["land"][a_] == r["k"]) for a_ in arms + ["END", "START"]} for k in ks}
    rep["frame_index"] = {a_: {li: [P(R, lambda r, a_=a_, li=li, i=i: r["fidx"][a_][li][i]) for i in (0, 1)] for li in R[0]["fidx"]["END"]}
                          for a_ in arms + ["END", "START"]}
    return rep


if __name__ == "__main__":
    allr = {}
    for tag in sys.argv[1:]:
        try:
            rep = analyze(tag)
        except FileNotFoundError:
            print(f"[{tag}] missing"); continue
        allr[tag] = rep
        print(f"== {tag} ({rep['model']}) t1 {rep['t1']} n {rep['n']} layer {rep['layer']} I0 {rep['I0']}")
        for key in ("W", "H", "D", "hypothesis"):
            print(f"   {key} {rep[key]}")
        print("   phi " + " | ".join(f"{a} {v['all']:.2f}" for a, v in rep["phi"].items()))
    json.dump(allr, open("/vlm/results/r3/s826b_analysis.json", "w"), indent=1, default=float)
