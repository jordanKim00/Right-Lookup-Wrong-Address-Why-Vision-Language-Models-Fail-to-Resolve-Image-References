"""S880 analysis (rules fixed before the data; knowledge/DECISIONS.md S880 entry).  Input results/r3/s880_<tag>_slotswap.json (and S873's transplant
JSON for the reproducibility check).
Writer effect of a variant (base scene, or the hybrid with picture slot b swapped): W = T(E) - T(E_BANDR) for T1 at L51 (address level, primary:
T1 = z_(p-1) - z_p of the matcher-head mean block scores, rows B1 / B4 / answer) and T3 = L_(p-1) - L_p (logit level).  dW_b = W(hybrid b) - W(base)
per (scene, pin).  Slot sensitivity S_b = mean |dW_b| over (scene, pin) for the slot roles predecessor (p - 1), anchor (p), successor (p + 1),
far; scene bootstrap 4,000 (seed 880).
I0: t1 >= .95; START(p) answers p >= .95 in base and hybrid scenes; base W(T1, L51) reproduces S873's natural W of the same scenes and pins
  (Pearson >= .90); L47 anchor = p for base and hybrid E >= .90.  Nothing is read if I0 fails.
D1 (local content): R_loc = mean(S_pred, S_anchor) / S_far (T1, L51): R_loc >= 2 with the lower bound > 1 -> 'the writer magnitude is set
  locally by the anchor / predecessor pictures'; R_loc <= 1.25 -> 'non-local: the whole scene sets it'; else 'graded'.
D2 (anchor or predecessor): (S_anchor - S_pred) / mean(S_pred, S_anchor) >= .25 with the interval above 0 -> 'the anchor picture dominates';
  <= -.25 with the interval below 0 -> 'the predecessor picture dominates'; else 'both'.
D3 (content-coupled step; S877 post hoc): ratio_dup = mean W(DUP) / mean W(base) (T1, L51; DUP = the predecessor slot holds an exact copy of the
  anchor picture): <= .30 with the upper bound < .50 -> 'the writer's step is content-coupled (no step between identical neighbours)';
  >= .70 with the lower bound > .50 -> 'the step is positional (it survives identical neighbours)'; else 'mixed'.
D4 (similarity): (mean W(base) - mean W(SAME)) / mean W(base) >= .20 with the interval above 0 (SAME = another photo of the anchor's name in the
  predecessor slot) -> 'a similar predecessor shrinks the step'; else 'not'.
Descriptive: the same at L55 / L59 / L63 and T3; S_successor; mean signed dW per slot; answer shifts (P(answer = p - 1), P(= p)) per slot
(DUP / SAME answers by name: the anchor's name is ambiguous between p - 1 and p and is reported as such).
Usage: python3 s880_analyze.py TAG [--s873 s873_27b]"""
from __future__ import annotations

import argparse, collections, json
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("tag"); ap.add_argument("--s873", default="s873_27b"); A = ap.parse_args()
R3 = "/vlm/results/r3"
J = json.load(open(f"{R3}/s880_{A.tag}_slotswap.json"))
OUT = f"{R3}/s880_analysis_{A.tag}.json"
rng = np.random.default_rng(880); NB = 4000
rows = J["rows"]; S = len(rows); PINS = J["pins"]; lm = J["late_mean"]; SL = {int(k): v for k, v in J["slots"].items()}
ROLE = ("pred", "anchor", "succ", "far")
NEIGH = ("dup", "same")
LAYS = ("51", "55", "59", "63")


def T1(r, key, p, L):
    z = np.asarray(r["zl"][key][L], dtype=float)[:, lm[L]].mean(1).mean(0); return z[p - 2] - z[p - 1]


def T3(r, key, p):
    x = np.asarray(r["mlog"][key]); return x[p - 2] - x[p - 1]


def anchor(q):
    q = np.asarray(q); am = [int(np.argmax(q[i])) + 1 for i in (2, 5, 9)]
    c = collections.Counter(am).most_common(); return c[0][0] if c[0][1] >= 2 else am[2]


REC = {}
for si, r in enumerate(rows):
    for p in PINS:
        d = dict(si=si, p=p)
        vars_ = [("base", "base")] + [(role, f"s{b}") for role, b in zip(ROLE, SL[p])] + [(k_, k_) for k_ in NEIGH]
        for role, nm in vars_:
            E, B = f"p{p}_{nm}_E", f"p{p}_{nm}_E_BANDR"
            for L in LAYS:
                d[f"W_{role}_{L}"] = T1(r, E, p, L) - T1(r, B, p, L)
            d[f"W3_{role}"] = T3(r, E, p) - T3(r, B, p)
            d[f"x_{role}"] = r["land"][E]; d[f"anc_{role}"] = anchor(r["qk47"][E]) if r["qk47"].get(E) is not None else None
            d[f"st_{role}"] = r["land"][f"p{p}_{nm}_START"]
        REC[(si, p)] = d
SC = sorted({k[0] for k in REC})


def bt(fn):
    n = len(SC); est = fn(np.arange(n)); bs = [fn(rng.integers(0, n, n)) for _ in range(NB)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return dict(est=float(est), lo=float(lo), hi=float(hi))


def Sb(ix, role, key="51"):
    vals = [abs(REC[(SC[i], p)][f"W_{role}_{key}"] - REC[(SC[i], p)][f"W_base_{key}"]) if key in LAYS
            else abs(REC[(SC[i], p)][f"W3_{role}"] - REC[(SC[i], p)]["W3_base"]) for i in ix for p in PINS]
    return float(np.mean(vals))


res = dict(tag=A.tag, n_scenes=S)
st = float(np.mean([d[f"st_{role}"] == d["p"] for d in REC.values() for role in ("base",) + ROLE]))                  # DUP / SAME: ambiguous names
reach = float(np.mean([d[f"anc_{role}"] == d["p"] for d in REC.values() for role in ("base",) + ROLE if d[f"anc_{role}"] is not None]))
try:
    J73 = json.load(open(f"{R3}/s873_{A.s873}_transplant.json")); r73 = J73["rows"]; lm73 = J73["late_mean"]
    def c73(r, key, p):
        z = np.asarray(r["zl"][key]["51"], dtype=float)[:, lm73["51"]].mean(1).mean(0); return z[p - 2] - z[p - 1]
    common = [s for s in SC if s < len(r73)]
    a = [c73(r73[s], f"Ep{p}", p) - c73(r73[s], f"Ep{p}_E_BANDR", p) for s in common for p in PINS]
    b = [REC[(s, p)]["W_base_51"] for s in common for p in PINS]
    repro = float(np.corrcoef(a, b)[0, 1])
except FileNotFoundError:
    repro = float("nan")
I0 = dict(t1=bool((J["t1_menu"] or 0) >= .95), START=bool(st >= .95), reproducible=bool(repro >= .90), pin_reach=bool(reach >= .90))
res["I0"] = dict(**I0, all=bool(all(I0.values())), t1_value=J["t1_menu"], START_acc=st, repro_r=repro, reach=reach)
ok = res["I0"]["all"]
def mW(ix, role):
    return float(np.mean([REC[(SC[i], p)][f"W_{role}_51"] for i in ix for p in PINS]))
d1 = bt(lambda ix: (Sb(ix, "pred") + Sb(ix, "anchor")) / 2 / Sb(ix, "far"))
d3 = bt(lambda ix: mW(ix, "dup") / mW(ix, "base")); d4 = bt(lambda ix: (mW(ix, "base") - mW(ix, "same")) / mW(ix, "base"))
d2 = bt(lambda ix: (Sb(ix, "anchor") - Sb(ix, "pred")) / ((Sb(ix, "anchor") + Sb(ix, "pred")) / 2))
res["D1"] = dict(R_loc=d1, verdict=(("the writer magnitude is set locally by the anchor / predecessor pictures" if d1["est"] >= 2 and d1["lo"] > 1 else
                                     "non-local: the whole scene sets it" if d1["est"] <= 1.25 else "graded") if ok else "not read"))
res["D2"] = dict(rel=d2, verdict=(("the anchor picture dominates" if d2["est"] >= .25 and d2["lo"] > 0 else
                                   "the predecessor picture dominates" if d2["est"] <= -.25 and d2["hi"] < 0 else "both") if ok else "not read"))
res["D3"] = dict(ratio_dup=d3, verdict=(("the writer's step is content-coupled (no step between identical neighbours)" if d3["est"] <= .30 and d3["hi"] < .50 else
                                        "the step is positional (it survives identical neighbours)" if d3["est"] >= .70 and d3["lo"] > .50 else "mixed") if ok else "not read"))
res["D4"] = dict(rel_drop=d4, verdict=(("a similar predecessor shrinks the step" if d4["est"] >= .20 and d4["lo"] > 0 else "not") if ok else "not read"))
desc = {}
for key in LAYS + ("T3",):
    desc[key] = {role: Sb(range(len(SC)), role, key) for role in ROLE}
desc["signed_dW_L51"] = {role: float(np.mean([d[f"W_{role}_51"] - d["W_base_51"] for d in REC.values()])) for role in ROLE + NEIGH}
desc["mean_W_L51"] = {role: float(np.mean([d[f"W_{role}_51"] for d in REC.values()])) for role in ("base",) + ROLE + NEIGH}
desc["mean_W_L63"] = {role: float(np.mean([d[f"W_{role}_63"] for d in REC.values()])) for role in ("base",) + ROLE + NEIGH}
desc["answers"] = {role: dict(P_pm1=float(np.mean([d[f"x_{role}"] == d["p"] - 1 for d in REC.values()])), P_p=float(np.mean([d[f"x_{role}"] == d["p"] for d in REC.values()])))
                   for role in ("base",) + ROLE}
desc["mean_W_base_L51"] = float(np.mean([d["W_base_51"] for d in REC.values()]))
res["descriptive"] = desc
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"]["all"], D1=[res["D1"]["verdict"], d1["est"]], D2=[res["D2"]["verdict"], d2["est"]], D3=[res["D3"]["verdict"], d3["est"]],
                      D4=[res["D4"]["verdict"], d4["est"]], S_L51=desc["51"], mean_W_L51=desc["mean_W_L51"]), indent=1))
print("->", OUT)
