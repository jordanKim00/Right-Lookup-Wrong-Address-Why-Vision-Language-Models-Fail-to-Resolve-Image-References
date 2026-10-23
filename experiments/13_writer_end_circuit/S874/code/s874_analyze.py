"""S874 analysis (exploratory and descriptive; no registered verdicts).  Inputs results/r3/s874_<tag>_vocablens.json (runs, answers),
s874_<tag>_lens.npz (jl = J-lens, ll = logit-lens scores [scene, form, candidate, point, row]) and the Part A rows JSON it names.
Rank of a candidate at (point, row) = 1 + the number of reference candidates with a strictly larger score (1 = top).  Reference sets: the nine
digit tokens '1'..'9' ('dig') and all candidates ('all').  Targets: j* (the correct address: 9 - k for END(k), q for START(q)) and the
answered address ('addr'; AB: the picture whose name was chosen, AQ: the digit said) among digits and j* among all; k (the ordinal's value;
= j* for START) among digits and among all; the frame word (' last' / ' first'), the ordinal word and j*'s number word among all candidates.  Row groups:
the question tail (ordinal .. frame word; best (min) and mean rank over its five rows), '?', B1, B4 and the answer row (A3).  Per family (END
k 2..6 pooled; START q 3..5 pooled) and answer form (AB, AQ): mean rank and top-1 rate at each of the 17 points (L20 .. L63, fin); AB - AQ
paired by (scene, ordinal), averaged within scene, scene bootstrap (--nb 4,000, seed 874) 95% interval (all pairs; and pairs whose AQ answer
is correct).  First point at which j* is top-1 among digits at the answer row ('first hit') and the first point from which it stays top-1
through fin ('stable').  Instrument block, read first: t1 (AB menu-word / AQ digit top-1 shares), the fin check (lens of the final residual
vs the model's logits), the Part A finite-difference check and split-half stability, and a positive control (the said digit's top-1 rate
among digits at the AQ answer row from L55 on).  Both lenses are analysed (jl primary, ll = J = I reference).  A low rank bounds what the
average Jacobian carries to the output vocabulary at that row and depth; it does not show that no address is computed.
Output results/r3/s874_analysis_<tag>.json and a short printed summary.
Usage: python3 s874_analyze.py TAG [--nb 4000] [--root /vlm/results/r3]"""
from __future__ import annotations

import argparse, json, os
import numpy as np

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth"]
NUMW = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
GROUPS = ("tail_min", "tail_mean", "qm", "B1", "B4", "ans")
KEYS = (("jstar", "dig"), ("jstar", "all"), ("addr", "dig"), ("k", "dig"), ("k", "all"), ("frame", "all"), ("ordw", "all"), ("numw", "all"))


def main(a):
    J = json.load(open(f"{a.root}/s874_{a.tag}_vocablens.json"))
    Z = np.load(f"{a.root}/s874_{a.tag}_lens.npz")
    rj = J["rows_json"] if os.path.exists(J["rows_json"]) else os.path.join(a.root, os.path.basename(J["rows_json"]))
    RJ = json.load(open(rj)) if os.path.exists(rj) else None
    names = [c["str"] for c in J["cands"]]
    C = len(names); DIGI = np.array([names.index(str(d)) for d in range(1, 10)]); ALL = np.arange(C)
    PTS, ROWN, KS, QS = J["pts"], J["rown"], J["ks"], J["qs"]
    P = len(PTS)
    TAIL = [ROWN.index(x) for x in ("ord", "img", "from", "the", "frame")]
    GI = dict(qm=ROWN.index("qm"), B1=ROWN.index("B1"), B4=ROWN.index("B4"), ans=ROWN.index("A3"))
    runs = J["runs"]; S = len(J["scenes"])
    idx = {(r["si"], r["frame"], r["q"], r["ans"]): i for i, r in enumerate(runs)}
    fam_ = np.array([r["frame"] for r in runs]); ans_ = np.array([r["ans"] for r in runs])
    rng = np.random.default_rng(874)

    def rank(sc, t, ref):                                                     # sc [C, P, R] -> [P, R]
        return 1 + (sc[ref] > sc[t][None]).sum(0)

    def grp(rk):                                                              # [P, R] -> {group: [P]}
        out = dict(tail_min=rk[:, TAIL].min(1), tail_mean=rk[:, TAIL].mean(1))
        out.update({g: rk[:, i] for g, i in GI.items()})
        return out

    def lens_stats(X):
        RK = {key: {g: np.zeros((len(runs), P)) for g in GROUPS} for key in KEYS}
        for i, r in enumerate(runs):
            sc = X[r["si"], r["fi"]].astype(np.float64)
            q, tgt, addr = r["q"], r["target"], r["addr"]
            tg = dict(jstar=DIGI[tgt - 1], addr=DIGI[addr - 1], k=DIGI[q - 1], frame=names.index(" last" if r["frame"] == "END" else " first"),
                      ordw=names.index(" " + ORDW[q - 1]), numw=names.index(" " + NUMW[tgt - 1]))
            for key in KEYS:
                gg = grp(rank(sc, tg[key[0]], DIGI if key[1] == "dig" else ALL))
                for g in GROUPS:
                    RK[key][g][i] = gg[g]
        res = {}
        for fam in ("END", "START"):
            for ans in ("AB", "AQ"):
                m = (fam_ == fam) & (ans_ == ans)
                res[f"{ans}_{fam}"] = {f"{k0}.{k1}": {g: dict(mean=RK[(k0, k1)][g][m].mean(0).tolist(), top1=(RK[(k0, k1)][g][m] == 1).mean(0).tolist())
                                                      for g in GROUPS} for k0, k1 in KEYS}
                res[f"{ans}_{fam}"]["n"] = int(m.sum())
                hit = RK[("jstar", "dig")]["ans"][m] == 1
                fh = [int(np.argmax(h)) if h.any() else None for h in hit]
                st = []
                for h in hit:
                    s_ = None
                    for p in range(P - 1, -1, -1):
                        if not h[p]:
                            break
                        s_ = p
                    st.append(s_)
                med = lambda xs: PTS[sorted(xs)[len(xs) // 2]] if xs else None
                res[f"{ans}_{fam}"]["first_top1_ans"] = dict(
                    any=float(np.mean([f is not None for f in fh])) if fh else None, median_first=med([f for f in fh if f is not None]),
                    median_stable=med([s for s in st if s is not None]), stable_any=float(np.mean([s is not None for s in st])) if st else None,
                    first_hist={PTS[p]: sum(f == p for f in fh) for p in range(P)})
            diff = {}
            for k0, k1 in KEYS:
                for g in GROUPS:
                    for sub in ("all", "aq_correct"):
                        per = []
                        for si in range(S):
                            d = []
                            for q in (KS if fam == "END" else QS):
                                iA, iQ = idx.get((si, fam, q, "AB")), idx.get((si, fam, q, "AQ"))
                                if iA is None or iQ is None or (sub == "aq_correct" and not runs[iQ]["correct"]):
                                    continue
                                d.append(RK[(k0, k1)][g][iA] - RK[(k0, k1)][g][iQ])
                            if d:
                                per.append(np.mean(d, 0))
                        if not per:
                            continue
                        D_ = np.array(per)
                        bs = D_[rng.integers(0, len(D_), (a.nb, len(D_)))].mean(1)
                        diff[f"{k0}.{k1}|{g}|{sub}"] = dict(mean=D_.mean(0).tolist(), lo=np.percentile(bs, 2.5, 0).tolist(),
                                                            hi=np.percentile(bs, 97.5, 0).tolist(), n_scenes=len(D_))
            res[f"diff_{fam}"] = diff
        maq = ans_ == "AQ"
        res["pos_control_said_digit_top1_ans_AQ"] = (RK[("addr", "dig")]["ans"][maq] == 1).mean(0).tolist()
        return res

    out = dict(tag=a.tag, n_scenes=S, n_runs=len(runs), pts=PTS, groups=list(GROUPS), keys=[f"{k0}.{k1}" for k0, k1 in KEYS], nb=a.nb)
    beh = {}
    for fam in ("END", "START"):
        for ans in ("AB", "AQ"):
            rs = [r for r in runs if r["frame"] == fam and r["ans"] == ans]
            beh[f"{ans}_{fam}"] = dict(n=len(rs), acc=float(np.mean([r["correct"] for r in rs])), t1=float(np.mean([r["t1"] for r in rs])),
                                       per_q={str(q): float(np.mean([r["correct"] for r in rs if r["q"] == q])) for q in sorted({r["q"] for r in rs})})
    out["behaviour"] = beh
    inst = dict(t1_ab=J["t1_ab"], t1_aq=J["t1_aq"], fin_err_max=J["fin_err_max"], fin_err_median=float(np.median([r["fin_err"] for r in runs])))
    if RJ:
        cos = np.array(RJ["split_half_cos"])
        inst.update(n_fit=RJ["n_fit"], fd_summary=RJ.get("fd_summary"),
                    split_half_cos_median={f"L{l}": float(np.median(cos[:, i])) for i, l in enumerate(RJ["src"])},
                    split_half_cos_min={f"L{l}": float(cos[:, i].min()) for i, l in enumerate(RJ["src"])})
    out["instrument"] = inst
    for lens in ("jl", "ll"):
        out[lens] = lens_stats(Z[lens])
    json.dump(out, open(f"{a.root}/s874_analysis_{a.tag}.json", "w"))

    # ---------------- printed summary (jl primary)
    L = out["jl"]
    col = lambda xs, f="{:6.1f}": "".join(f.format(x) for x in xs)
    print(f"[s874 {a.tag}] scenes {S} · runs {len(runs)} · points {' '.join(PTS)}")
    s_ = f"I0  t1 AB {inst['t1_ab']:.3f} · t1 AQ {inst['t1_aq']:.3f} · fin check |dlogit| max {inst['fin_err_max']:.3f} (median {inst['fin_err_median']:.3f})"
    if RJ:
        s_ += f" · fit prompts {inst['n_fit']}"
        print(s_)
        print("    split-half cos (median over candidates): " + " ".join(f"{k} {x:.3f}" for k, x in inst["split_half_cos_median"].items()))
        for key, fs in (inst["fd_summary"] or {}).items():
            c_ = lambda x: "-" if x is None else f"{x:.3f}"
            print(f"    FD {key}: averaged rows corr {c_(fs['avg']['corr'])} slope {c_(fs['avg']['slope'])} rel RMSE {c_(fs['avg']['rel_rmse'])}"
                  f" · local rows corr {c_(fs['local']['corr'])} slope {c_(fs['local']['slope'])} · cos(local, avg) {c_(fs['cos_local_avg'])}")
    else:
        print(s_ + " · rows JSON not found")
    pc = L["pos_control_said_digit_top1_ans_AQ"]
    print("    positive control (AQ answer row, said digit top-1 among digits): " + " ".join(f"{PTS[p]} {pc[p]:.2f}" for p in range(P - 4, P)))
    print("behaviour  " + " · ".join(f"{k} acc {b['acc']:.2f} t1 {b['t1']:.2f} [" + " ".join(f"{q}:{x:.2f}" for q, x in b["per_q"].items()) + "]"
                                     for k, b in beh.items()))
    hdr = "              " + "".join(f"{p:>6}" for p in PTS)
    for fam, lab in (("END", "j* = 9 - k"), ("START", "j* = q")):
        print(f"{fam}: rank of {lab} among the 9 digits (mean; 1 = top) - AB / AQ / AB-AQ (* = 95% scene-bootstrap interval excludes 0)")
        print(hdr)
        for g in (("tail_min", "qm", "B1", "B4", "ans") if fam == "END" else ("tail_min", "ans")):
            d = L[f"diff_{fam}"].get(f"jstar.dig|{g}|all")
            print(f"{g:>9}  AB " + col(L[f"AB_{fam}"]["jstar.dig"][g]["mean"]))
            print(f"{'':>9}  AQ " + col(L[f"AQ_{fam}"]["jstar.dig"][g]["mean"]))
            if d:
                print(f"{'':>9}  d  " + "".join(f"{m:+5.1f}" + ("*" if (lo > 0 or hi < 0) else " ") for m, lo, hi in zip(d["mean"], d["lo"], d["hi"])))
        for ans in ("AB", "AQ"):
            f_ = L[f"{ans}_{fam}"]["first_top1_ans"]
            print(f"  {ans} j* top-1 among digits at the answer row: rate " + col(L[f"{ans}_{fam}"]["jstar.dig"]["ans"]["top1"], "{:6.2f}")
                  + f" · any point {f_['any']:.2f} first median {f_['median_first']} · stable through fin {f_['stable_any']:.2f} from median "
                  f"{f_['median_stable']}")
    print("END: k (ordinal value) among digits, tail best row - AB " + col(L["AB_END"]["k.dig"]["tail_min"]["mean"]) + " | AQ "
          + col(L["AQ_END"]["k.dig"]["tail_min"]["mean"]))
    print(f"END: j* among all {len(names)} candidates, answer row - AB " + col(L["AB_END"]["jstar.all"]["ans"]["mean"]) + " | AQ "
          + col(L["AQ_END"]["jstar.all"]["ans"]["mean"]))
    print(f"END: ' last' among all {len(names)} candidates, tail best row - AB " + col(L["AB_END"]["frame.all"]["tail_min"]["mean"]) + " | AQ "
          + col(L["AQ_END"]["frame.all"]["tail_min"]["mean"]))
    LL = out["ll"]
    print("logit-lens reference, END j* among digits at the answer row - AB " + col(LL["AB_END"]["jstar.dig"]["ans"]["mean"]) + " | AQ "
          + col(LL["AQ_END"]["jstar.dig"]["ans"]["mean"]))
    print(f"-> {a.root}/s874_analysis_{a.tag}.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tag"); ap.add_argument("--nb", type=int, default=4000); ap.add_argument("--root", default="/vlm/results/r3")
    main(ap.parse_args())
