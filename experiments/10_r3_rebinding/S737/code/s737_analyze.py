"""S737 analysis, frozen with the pre-registration (knowledge/DECISIONS.md 2026-10-10) before any S737 data.
Rows = option-line ordinal tokens; option L names forward image k_L; the middle option (literal = named) is left out of the END / R³ rules.
    I0_NAT            mean START share on the named block >= mean 2/N (the option row performs the lookup)
    OPT_LITERAL_NAT   mean END (literal - named) >= .05
    R3_TARGET_NAT     mean END + R³ (named - literal) >= .05
    Q_ROUTE_NAT       (hybrids, END items answered wrongly) acc(Q_ONLY) - acc(G_ONLY) >= .20, read only when I0_NAT holds
Reported: accuracy of START, END, END + R³, and of the causal arms on the wrongly answered END items.
Usage: python s737_analyze.py TAG [TAG ...]
"""
import json, os, sys
import numpy as np


def analyze(tag):
    d = json.load(open(f"/vlm/results/r3/s737_{tag}_natqk.json")); R = d["rows"]
    acc = {f: float(np.mean([r[f"pick_{f}"] == r["gold"] for r in R])) for f in ("START", "END", "R3")}
    diag, base = [], []
    lit_e, lit_r = [], []
    for r in R:
        N = r["N"]
        for (L, k), s in zip(r["opts"], r["START"] or []):
            diag.append(s["named"]); base.append(2 / N)
        for (L, k), e, x in zip(r["opts"], r["END"] or [], r["R3"] or []):
            if 2 * k == N + 1:
                continue
            lit_e.append(e["literal"] - e["named"]); lit_r.append(x["named"] - x["literal"])
    wrong = [r for r in R if "pick_Q_ONLY" in r]
    causal = dict(n=len(wrong))
    if wrong:
        causal.update({arm: float(np.mean([r[f"pick_{arm}"] == r["gold"] for r in wrong])) for arm in ("Q_ONLY", "G_ONLY")})
    chk = {"I0_NAT": float(np.mean(diag)) >= float(np.mean(base)), "OPT_LITERAL_NAT": float(np.mean(lit_e)) >= .05,
           "R3_TARGET_NAT": float(np.mean(lit_r)) >= .05}
    if wrong:
        chk["Q_ROUTE_NAT"] = chk["I0_NAT"] and (causal["Q_ONLY"] - causal["G_ONLY"] >= .20)
    S = dict(acc=acc, start_named=float(np.mean(diag)), chance2=float(np.mean(base)), end_lit_minus_named=float(np.mean(lit_e)),
             r3_named_minus_lit=float(np.mean(lit_r)), causal=causal, n_items=len(R))
    print(f"[{tag}] acc START {acc['START']:.3f} · END {acc['END']:.3f} · R3 {acc['R3']:.3f}; START named {S['start_named']:.3f} (2/N {S['chance2']:.3f}); "
          f"END lit-named {S['end_lit_minus_named']:+.3f}; R3 named-lit {S['r3_named_minus_lit']:+.3f}; causal {causal}; checks {chk}")
    return dict(summary=S, checks=chk)


if __name__ == "__main__":
    P = "/vlm/results/r3/s737_analysis.json"
    out = json.load(open(P)) if os.path.exists(P) else {}
    out.update({t: analyze(t) for t in sys.argv[1:]})
    json.dump(out, open(P, "w"), indent=1)
