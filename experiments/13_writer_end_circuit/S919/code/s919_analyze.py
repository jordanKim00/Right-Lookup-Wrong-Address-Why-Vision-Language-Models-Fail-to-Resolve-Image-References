"""S919 analysis (rules fixed before the data; knowledge/DECISIONS.md S919 entry).  Input results/r3/s919_<tag>_bandstate.json (N8, k 2-6).
Per scene and k at END(k)'s natural L47 anchor a_k: E, S (natural), EEE (END with its own query at 35-47), SEE (START(a_k) with END's query,
picture mass and output gate at 35-47), and SEE + END's band outputs at the read rows: SEEB7 (DN46, MLP46, MLP47, DN48, MLP48, DN49, MLP49),
SEEMLP, SEEDN, SEEM46, SEEM47, SEEM49, SEED46, SEED49.  dep(cell) = P(answer != a_k); the remaining END state G = dep(EEE) - dep(SEE);
rho_X = [dep(SEE_X) - dep(SEE)] / G (share of it restored by END's band outputs X).  Scene bootstrap 4,000 (seed 919), pooled over k.
I0: t1 >= .95; natural START(q) answers q >= .95 (q < N); the pins reproduce END's natural answer (x(EEE) = x(E) >= .95); S918's remaining END
  state is present (G >= .15 with the lower bound > .05).  Nothing is read if I0 fails.
B1 (is the remaining END state in the band's MLP / DeltaNet outputs?): rho_B7 >= .75 with the lower bound > .50 -> 'the remaining END state is
  carried by the band's MLP and DeltaNet outputs (layers 46-49)'; <= .25 with the upper bound < .40 -> 'the remaining END state is not in the
  band's MLP and DeltaNet outputs'; else 'partial'.
B2 (which family): rho_MLP >= .60 (lower > .35) and rho_DN <= .30 -> 'the band's MLPs carry it'; rho_DN >= .60 (lower > .35) and rho_MLP <= .30
  -> 'the band's DeltaNets carry it'; both >= .40 -> 'both families carry it'; else 'mixed'.
Descriptive: the single components' rho (M46, M47, M49, D46, D49), dep of every cell per k, identity runs (a_k = N + 1 - k) accuracy per cell,
  the two halves of END(k)'s natural L47 margin.  rho is the share of an intervention effect (counterfactual), not a mediation share.
Usage: python3 s919_analyze.py TAG [OUT]"""
from __future__ import annotations

import collections, json, sys
import numpy as np

TAG = sys.argv[1]
J = json.load(open(f"/vlm/results/r3/s919_{TAG}_bandstate.json"))
OUT = sys.argv[2] if len(sys.argv) > 2 else f"/vlm/results/r3/s919_analysis_{TAG}.json"
rng = np.random.default_rng(919); NB = 4000
N = J["N"]; KS = J["ks"]; rows = J["rows"]; nS = len(rows); em = J["early_mean"]["47"]
BANDS = ["B7", "MLP", "DN", "M46", "M47", "M49", "D46", "D49"]
CELLS = ["E", "S", "EEE", "SEE"] + [f"SEE{b}" for b in BANDS]


def margin(r, k):
    h = np.asarray(r["qk47h"][f"N_E{k}"], float).mean(0); e = h[em].mean(0); e = e - e.mean(); s = np.sort(e)[::-1]; return float(s[0] - s[1])


MG = {(si, k): margin(r, k) for si, r in enumerate(rows) for k in KS}
MED = float(np.median(list(MG.values())))


def dep(ix, cell, half=None):
    v = []
    for i in ix:
        r = rows[int(i)]
        for k in KS:
            if half is not None and (MG[(int(i), k)] < MED) != (half == "small"):
                continue
            v.append(float(r["land"][f"{cell}_{k}"] != r["anchor"][str(k)]))
    return float(np.mean(v)) if v else float("nan")


def _div(a_, b_):
    return a_ / b_ if b_ not in (0, 0.0) and np.isfinite(b_) else float("nan")


def boot(fn):
    est = fn(np.arange(nS)); bs = [fn(rng.integers(0, nS, nS)) for _ in range(NB)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return dict(est=float(est), lo=float(lo), hi=float(hi))


allx = np.arange(nS)
G = lambda ix: dep(ix, "EEE") - dep(ix, "SEE")
rho = lambda b: (lambda ix: _div(dep(ix, f"SEE{b}") - dep(ix, "SEE"), G(ix)))
res = dict(tag=TAG, n_scenes=nS, margin_median=MED)
st = float(np.mean([r["land"][f"N_S{q}"] == q for r in rows for q in range(1, N)]))
fid = float(np.mean([r["land"][f"EEE_{k}"] == r["land"][f"E_{k}"] for r in rows for k in KS]))
g0 = boot(G)
I0 = dict(t1=bool((J["t1_menu"] or 0) >= .95), START=bool(st >= .95), pins=bool(fid >= .95), remaining=bool(g0["est"] >= .15 and g0["lo"] > .05))
res["I0"] = dict(**I0, all=bool(all(I0.values())), t1_value=J["t1_menu"], START_acc=st, pin_fidelity=fid, G=g0)
ok = res["I0"]["all"]
R = {b: boot(rho(b)) for b in BANDS}
r7 = R["B7"]
res["B1"] = dict(rho_B7=r7, verdict=(("the remaining END state is carried by the band's MLP and DeltaNet outputs (layers 46-49)" if r7["est"] >= .75 and r7["lo"] > .50 else
                                      "the remaining END state is not in the band's MLP and DeltaNet outputs" if r7["est"] <= .25 and r7["hi"] < .40 else "partial")
                                     if ok else "not read"))
rm, rd = R["MLP"], R["DN"]
if rm["est"] >= .60 and rm["lo"] > .35 and rd["est"] <= .30:
    v2 = "the band's MLPs carry it"
elif rd["est"] >= .60 and rd["lo"] > .35 and rm["est"] <= .30:
    v2 = "the band's DeltaNets carry it"
elif rm["est"] >= .40 and rd["est"] >= .40:
    v2 = "both families carry it"
else:
    v2 = "mixed"
res["B2"] = dict(rho_MLP=rm, rho_DN=rd, verdict=(v2 if ok else "not read"))
desc = dict(rho=R, dep_pooled={c: dep(allx, c) for c in CELLS},
            dep={c: {str(k): float(np.mean([r["land"][f"{c}_{k}"] != r["anchor"][str(k)] for r in rows])) for k in KS} for c in CELLS},
            halves={h: dict(G=dep(allx, "EEE", h) - dep(allx, "SEE", h), rho_B7=_div(dep(allx, "SEEB7", h) - dep(allx, "SEE", h), dep(allx, "EEE", h) - dep(allx, "SEE", h)))
                    for h in ("small", "large")})
idn = [(r, k) for r in rows for k in KS if r["anchor"][str(k)] == N + 1 - k]
desc["identity"] = dict(n=len(idn), acc={c: (float(np.mean([r["land"][f"{c}_{k}"] == N + 1 - k for r, k in idn])) if idn else None) for c in CELLS})
desc["answers"] = {c: {str(k): dict(collections.Counter(r["land"][f"{c}_{k}"] for r in rows)) for k in KS} for c in CELLS}
res["descriptive"] = desc
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"]["all"], G=g0["est"], B1=[res["B1"]["verdict"], r7["est"]], B2=[res["B2"]["verdict"], rm["est"], rd["est"]],
                      singles={b: R[b]["est"] for b in ("M46", "M47", "M49", "D46", "D49")}), indent=1))
print("->", OUT)
