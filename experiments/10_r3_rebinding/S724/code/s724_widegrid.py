"""Benchmark-independent Read selection check (saved S724 calibration measurements only, no model call, no benchmark item).
The S724 rule unchanged (detector false detections <= .005 on non-ordinal tokens, binder false binding <= .02, final = admissible combination
with the highest PlanExact on the sel split, confirmation on the val split), applied to a dense generic grid instead of the registered one:
    detector: abs and margin m in {.30, .35, ..., .70};  binder: clause-anchor radius x {1.0, 1.1, ..., 1.6} (+ learned gate where present);
    count: last / max.
Question: does calibration data alone pick the deployed Read (or one with identical plans) out of the wide grid?
Usage: python3 exp/s724_widegrid.py SEL:VAL [...] --out OUT.json   (run inside the container: paths under /vlm)
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, "/vlm/exp")
import s724_plan as P
from s724_select import detect, window, refs, is_end, ORD, FRAMEW

DETS = ["abs"] + [f"margin:{m:.2f}" for m in np.arange(.30, .7001, .05)]
BINDS = [f"proto:{s:.1f}" for s in np.arange(1.0, 1.6001, .1)]
CNTS = ["last", "max"]


def comp_metrics(R):
    S1 = [r for r in R if r.get("kind") != "mixed"]
    det = {}
    for rule in DETS:
        hit = n = fp = no = 0
        for r in S1:
            m, kh = detect(r, rule); rf = dict(refs(r))
            for i, w in enumerate(r["toks"]):
                if i in rf:
                    n += 1; hit += bool(m[i]) and kh[i] == rf[i]
                elif w.strip().lower() not in ORD and w.strip().lower() not in FRAMEW:
                    no += 1; fp += bool(m[i])
        det[rule] = dict(recall=hit / max(n, 1), fp=fp / max(no, 1))
    bind = {}
    for b in BINDS:
        s = float(b.split(":")[1]); er = en = nf = nn = 0
        for r in S1:
            for i, k in refs(r):
                anch = any(r["d_end"][q] <= s * r["thr"]["end"] or r["d_last"][q] <= s * r["thr"]["last"] for q in window(r["toks"], i))
                if is_end(r, i):
                    en += 1; er += anch
                else:
                    nn += 1; nf += anch
        bind[b] = dict(end_recall=er / max(en, 1), false_bind=nf / max(nn, 1))
    if S1 and all("v1" in r for r in S1):
        er = en = nf = nn = 0
        for r in S1:
            g = {x["pos"]: x["g"] for x in r["v1"]}
            for i, k in refs(r):
                b_ = g.get(i) is not None and g[i] > .5
                if is_end(r, i):
                    en += 1; er += b_
                else:
                    nn += 1; nf += b_
        bind["v1_gate"] = dict(end_recall=er / max(en, 1), false_bind=nf / max(nn, 1))
    return det, bind


def plans(R, keys):
    out = []
    for r in R:
        row = {}
        for k in keys:
            if k[0] == "min":
                src, nh = P.plan_min(r, k[1], k[2], k[3])
            else:
                src, nh = P.plan_v1c(r, k[3])
            row[k] = (P.score(r, src, nh)[0], tuple(sorted((p, x[1], x[2]) for p, x in src.items() if x[1] is not None or x[2] is not None)))
        out.append(row)
    return out


def main(sel, val):
    Rs = json.load(open(f"/vlm/results/r3/s724_{sel}_readcal.json"))["rec"]
    Rv = json.load(open(f"/vlm/results/r3/s724_{val}_readcal.json"))["rec"]
    det, bind = comp_metrics(Rs)
    ok_d = [d for d in DETS if det[d]["fp"] <= .005 + 1e-12] or [min(DETS, key=lambda d: det[d]["fp"])]
    ok_b = [b for b in BINDS if bind[b]["false_bind"] <= .02 + 1e-12] or [min(BINDS, key=lambda b: bind[b]["false_bind"])]
    cands = [("min", d, b, c) for d in ok_d for b in ok_b for c in CNTS]
    if "v1_gate" in bind and bind["v1_gate"]["false_bind"] <= .02 + 1e-12 and det["abs"]["fp"] <= .005 + 1e-12:
        cands += [("v1c", "v1", "v1_gate", c) for c in CNTS]
    deployed = tuple(json.load(open("/vlm/results/r3/s724_plan.json"))[sel]["final"])
    keys = list(dict.fromkeys(cands + [deployed]))
    ps, pv = plans(Rs, keys), plans(Rv, keys)
    pe = {k: float(np.mean([row[k][0] for row in ps])) for k in keys}
    order = lambda k: (-round(pe[k], 9), DETS.index(k[1]) if k[1] in DETS else -1, BINDS.index(k[2]) if k[2] in BINDS else 99, CNTS.index(k[3]))
    final = sorted(cands, key=order)[0]
    same_plans_sel = float(np.mean([row[final][1] == row[deployed][1] for row in ps]))
    same_plans_val = float(np.mean([row[final][1] == row[deployed][1] for row in pv]))
    top = sorted(cands, key=order)[:5]
    res = dict(sel=sel, val=val, grid=dict(detectors=DETS, binders=BINDS, counts=CNTS), admissible_detectors=ok_d, admissible_binders=ok_b,
               final=list(final), deployed=list(deployed), planexact_sel=dict(final=pe[final], deployed=pe[deployed]),
               planexact_val=dict(final=float(np.mean([row[final][0] for row in pv])), deployed=float(np.mean([row[deployed][0] for row in pv]))),
               identical_plans=dict(sel=same_plans_sel, val=same_plans_val), top5=[[list(k), pe[k]] for k in top])
    print(f"[{sel}] admissible det {ok_d} · bind {ok_b}")
    print(f"[{sel}] wide-grid choice {final} (sel {pe[final]:.3f}) vs deployed {deployed} (sel {pe[deployed]:.3f}); identical plans sel {same_plans_sel:.3f} · val {same_plans_val:.3f}; "
          f"val PlanExact {res['planexact_val']['final']:.3f} vs {res['planexact_val']['deployed']:.3f}")
    print(f"[{sel}]   top: " + " · ".join(f"{k[1]}+{k[2]}+{k[3]} {v:.3f}" for k, v in res["top5"]))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("pairs", nargs="+"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = [main(*p.split(":")) for p in a.pairs]
    json.dump(out, open(a.out, "w"), indent=1)
