"""S835 analysis (rules fixed in knowledge/DECISIONS.md before any S835 data).  Descriptive re-measure of §928 under a clean readout.
I0: t1 >= .95 for F and R; every item equals S489's (task, index, N, option positions, gold).
Signatures (§928 definitions, exp/p2_w3_mmiu_signature.py): among wrong R answers, 'literal' = the chosen option's end-relative ordinal kr
read from the start is the gold picture's position; 'step' = the copy kr moved one step toward the option's own target lands on the gold
picture (identity options: either neighbour).  Chance = mean over the three wrong options of the same item.
  LIT: literal - chance >= .15 with the item-bootstrap 95% interval (4,000 draws) of the difference > 0 -> 'wrong answers read the option
       ordinal from the start'; otherwise 'no literal signature'.
  STEP: step - chance >= .05 with interval > 0 -> 'step signature above chance'; interval < 0 -> 'step signature below chance'; else 'at chance'.
Both on all items and on items whose R pass has t1 = 1.  Also: agreement of R answers with S489's saved ones; accuracy F / R; literal share per
task.
Amendment (DECISIONS 10-17 02:52, before any S835 outcome was seen): S489's saved run has 50 items per task while S835 runs 100; the primary set
is the items S489 ran ((task, index) among S489's rows - the first 50 per task), on which I0's item identity is judged (no S489 row may be
missing); the 100-per-task set is reported with the same measures as a descriptive extension.
Usage: python exp/s835_analyze.py [TAG]  ->  results/r3/s835_analysis.json"""
from __future__ import annotations

import collections, json, sys
import numpy as np

V = "/home/storage2/ai26269345/vlm"
L = "ABCD"


def sig(r, x):
    N, ks = r["N"], r["ks"]; gpos = ks[L.index(r["gold"])]; pos = ks[L.index(x)]; kr = N + 1 - pos
    step = gpos in (pos - 1, pos + 1) if kr == pos else (kr + (1 if pos > kr else -1)) == gpos
    return float(kr == gpos), float(step)


def signature(rows, seed=0):
    E = [r for r in rows if r["R"] != r["gold"]]
    if not E:
        return None
    lit = np.array([sig(r, r["R"])[0] for r in E]); stp = np.array([sig(r, r["R"])[1] for r in E])
    ch = np.array([[np.mean([sig(r, x)[i] for x in L if x != r["gold"]]) for i in (0, 1)] for r in E])
    rng = np.random.default_rng(seed); B = []
    for _ in range(4000):
        ix = rng.integers(0, len(E), len(E))
        B.append((lit[ix].mean() - ch[ix, 0].mean(), stp[ix].mean() - ch[ix, 1].mean()))
    B = np.array(B); ci = lambda c: [float(np.percentile(B[:, c], 2.5)), float(np.percentile(B[:, c], 97.5))]
    dl, ds = float(lit.mean() - ch[:, 0].mean()), float(stp.mean() - ch[:, 1].mean())
    out = dict(errors=len(E), literal=float(lit.mean()), literal_chance=float(ch[:, 0].mean()), d_literal=dl, ci_literal=ci(0),
               step=float(stp.mean()), step_chance=float(ch[:, 1].mean()), d_step=ds, ci_step=ci(1))
    out["LIT"] = "wrong answers read the option ordinal from the start" if dl >= .15 and out["ci_literal"][0] > 0 else "no literal signature"
    out["STEP"] = ("step signature above chance" if ds >= .05 and out["ci_step"][0] > 0 else
                   "step signature below chance" if out["ci_step"][1] < 0 else "step signature at chance")
    return out


def main(tag="s835_27b"):
    d = json.load(open(f"{V}/results/s835_{tag}_mmiusig.json"))
    old = {(r["task"], r["i"]): r for r in json.load(open(f"{V}/results/s489_mmiu_fix_489a27b.json"))["rows"]}
    rows_all = d["rows"]; rows_p = [r for r in rows_all if (r["task"], r["i"]) in old]          # primary set: S489's items (amendment 02:52)
    same = [all(old[(r["task"], r["i"])][k] == r[k] for k in ("N", "ks", "gold")) for r in rows_p]
    sig_rows = lambda rows: [r for r in rows if r["gold"] in L and len(r["ks"]) == 4]
    R = sig_rows(rows_p)
    t1 = {f: float(np.mean([r[f + "|t1"] for r in rows_p])) for f in ("F", "R")}
    acc = {f: float(np.mean([r[f] == r["gold"] for r in rows_p])) for f in ("F", "R")}
    out = dict(tag=tag, n=len(rows_p), n_sig=len(R), acc=acc, t1=t1, n_s489=len(old))
    out["I0"] = dict(t1_F=t1["F"], t1_R=t1["R"], items_same_as_s489=float(np.mean(same)) if same else 0.0, s489_rows_found=len(rows_p) / len(old))
    out["I0"]["holds"] = bool(t1["F"] >= .95 and t1["R"] >= .95 and out["I0"]["items_same_as_s489"] == 1.0 and out["I0"]["s489_rows_found"] == 1.0)
    out["agree_R_with_s489"] = float(np.mean([old[(r["task"], r["i"])]["R"] == r["R"] for r in rows_p]))
    out["all"] = signature(R)
    out["t1_only"] = signature([r for r in R if r["R|t1"]])
    Rx = sig_rows(rows_all)
    out["extension_100"] = dict(n=len(rows_all), n_sig=len(Rx), acc={f: float(np.mean([r[f] == r["gold"] for r in rows_all])) for f in ("F", "R")},
                                t1={f: float(np.mean([r[f + "|t1"] for r in rows_all])) for f in ("F", "R")}, all=signature(Rx),
                                t1_only=signature([r for r in Rx if r["R|t1"]]))
    bt = collections.defaultdict(list)
    for r in R:
        if r["R"] != r["gold"]:
            bt[r["task"]].append(sig(r, r["R"])[0])
    out["literal_by_task"] = {t: dict(errors=len(x), literal=float(np.mean(x))) for t, x in sorted(bt.items())}
    out["acc_by_task"] = {t: dict(n=len(x), F=float(np.mean([r["F"] == r["gold"] for r in x])), R=float(np.mean([r["R"] == r["gold"] for r in x])))
                          for t, x in sorted(collections.defaultdict(list, {t: [r for r in R if r["task"] == t] for t in {r["task"] for r in R}}).items())}
    fn = f"{V}/results/r3/s835_analysis.json" if tag == "s835_27b" else f"{V}/results/r3/s835_analysis_{tag}.json"
    json.dump(out, open(fn, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("n", "n_s489", "acc", "t1", "I0", "agree_R_with_s489")}, indent=1))
    print("extension_100", json.dumps({k: out["extension_100"][k] for k in ("n", "acc", "t1")}))
    for k in ("all", "t1_only"):
        print(k, json.dumps(out[k], indent=1) if out[k] else None)


if __name__ == "__main__":
    main(*sys.argv[1:])
