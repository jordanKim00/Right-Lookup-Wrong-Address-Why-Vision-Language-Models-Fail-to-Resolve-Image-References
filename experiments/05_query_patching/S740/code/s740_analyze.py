"""S740 analysis, frozen with the pre-registration (knowledge/DECISIONS.md 2026-10-11) before any S740 data.
On repaired items (R³ correct, unedited wrong):
    NOOP_OK               P(pick_NOOP = pick_R3) >= .95                      (the recomputation reproduces the repaired run)
    NAT_QUERY_SUFFICIENT  P(gold | END_QALL) >= .60                           (the repaired full-attention queries carry the repair)
    NAT_QUERY_NECESSARY   P(gold | R3_QBACK_ALL) <= .40                       (putting them back removes most of the repair)
Reported: the matcher arms (R3_QBACK_M, END_QM), the share of R3_QBACK_ALL failures that return to the unedited answer, repaired-item count.
The two NAT_ rules are read only when NOOP_OK holds.
Usage: python s740_analyze.py TAG [TAG ...]
"""
import json, os, sys
import numpy as np


def analyze(tag):
    d = json.load(open(f"/vlm/results/r3/s740_{tag}_natmed.json")); R = d["rows"]
    rep = [r for r in R if "pick_NOOP" in r]
    acc = lambda arm: float(np.mean([r[f"pick_{arm}"] == r["gold"] for r in rep])) if rep else float("nan")
    noop = float(np.mean([r["pick_NOOP"] == r["pick_R3"] for r in rep])) if rep else float("nan")
    fails = [r for r in rep if r["pick_R3_QBACK_ALL"] != r["gold"]]
    back = float(np.mean([r["pick_R3_QBACK_ALL"] == r["pick_END"] for r in fails])) if fails else float("nan")
    S = dict(n_items=len(R), n_written=sum(r["n_writes"] > 0 for r in R), n_repaired=len(rep), noop=noop,
             acc={arm: acc(arm) for arm in ("NOOP", "R3_QBACK_ALL", "END_QALL", "R3_QBACK_M", "END_QM")}, back_to_original=back)
    chk = {"NOOP_OK": noop >= .95}
    if chk["NOOP_OK"]:
        chk["NAT_QUERY_SUFFICIENT"] = S["acc"]["END_QALL"] >= .60
        chk["NAT_QUERY_NECESSARY"] = S["acc"]["R3_QBACK_ALL"] <= .40
    print(f"[{tag}] repaired {len(rep)} of {S['n_written']} written · NOOP {noop:.3f} · " + " · ".join(f"{k} {v:.3f}" for k, v in S["acc"].items())
          + f" · failures back to original {back:.3f} · checks {chk}")
    return dict(summary=S, checks=chk)


if __name__ == "__main__":
    P = "/vlm/results/r3/s740_analysis.json"
    out = json.load(open(P)) if os.path.exists(P) else {}
    out.update({t: analyze(t) for t in sys.argv[1:]})
    json.dump(out, open(P, "w"), indent=1)
