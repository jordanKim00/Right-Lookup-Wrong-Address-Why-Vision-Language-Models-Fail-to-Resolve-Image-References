"""S918 analysis (rules fixed before the data; knowledge/DECISIONS.md S918 entry; advisor review 2, 10-22: address vs reading strength).
Input results/r3/s918_<tag>_reading.json (N8, k 2-6).  Per scene and k, cells state-address-reading over {E = END(k), S = START(a_k)} with
a_k = END(k)'s natural L47 anchor: every run has its read-row query at layers 35 / 39 / 43 / 47 from the address donor (all heads' picture scores
= the donor's) and its picture mass per row and head plus its output gate from the reading donor (or its own state's).  dep(cell) = P(answer
!= a_k).  Scene bootstrap 4,000 (seed 918), pooled over k; every run is used; each statistic is also reported in the two halves of END(k)'s
natural L47 margin (top-1 minus top-2, set median).
I0: t1 >= .95; natural START(q) answers q >= .95 (q < N); the pins reproduce the natural answers (x(EEE) = x(N_E k) and x(SSS) = x(N_S a_k),
  pooled >= .95); the mass edit reaches its target (max |achieved - target| picture mass <= 1e-3 over rows, heads and layers in every cell
  with a mass transplant); END departs (dep(EEE) >= .40).  Nothing is read if I0 fails.
R1 (START's reading at END's address, END state): d1 = dep(EEE) - dep(EES) >= .20 with the lower bound > .10 -> 'at a fixed END address,
  START's reading strength reduces END's departure: the departure depends on how the address is read, not only on where it points';
  <= .05 with the upper bound < .10 -> 'no effective reading-strength effect observed at a fixed address (END state)'; else 'partial'.
R2 (END's reading at END's address, START state): d2 = dep(SEE) - dep(SES) >= .20 with the lower bound > .10 -> 'at a fixed END address,
  END's reading strength makes the START state depart'; <= .05 with the upper bound < .10 -> 'no effective reading-strength effect observed
  at a fixed address (START state)'; else 'partial'.
R3 (address and reading held, state swapped): d3 = dep(EEE) - dep(SEE): |d3| <= .10 with the interval inside [-.15, .15] -> 'with the
  address and the reading strength held, the remaining state does not change the departure'; d3 >= .25 with the lower bound > .15 -> 'the
  END departure needs more than the early address and its reading strength (the remaining END state)'; else 'partial'.
Descriptive: dep of every cell per k; the mass-only and gate-only parts of d1 (dep(EEE) - dep(EESm), dep(EEE) - dep(EESg)) and of d2
  (dep(SEEm) - dep(SES), dep(SEEg) - dep(SES)); the address effects dep(EEE) - dep(ESE), dep(SES) - dep(SSS); the reading effects at START's
  address dep(ESE) - dep(ESS), dep(SSE) - dep(SSS); identity runs (a_k = N + 1 - k) accuracy per cell; margin halves; achieved picture masses.
Usage: python3 s918_analyze.py TAG [OUT]"""
from __future__ import annotations

import collections, json, sys
import numpy as np

TAG = sys.argv[1]
J = json.load(open(f"/vlm/results/r3/s918_{TAG}_reading.json"))
OUT = sys.argv[2] if len(sys.argv) > 2 else f"/vlm/results/r3/s918_analysis_{TAG}.json"
rng = np.random.default_rng(918); NB = 4000
N = J["N"]; KS = J["ks"]; rows = J["rows"]; nS = len(rows); em = J["early_mean"]["47"]
CELLS = ["EEE", "EES", "EESm", "EESg", "ESE", "ESS", "SEE", "SEEm", "SEEg", "SES", "SSE", "SSS"]


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


def boot(fn):
    est = fn(np.arange(nS)); bs = [fn(rng.integers(0, nS, nS)) for _ in range(NB)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return dict(est=float(est), lo=float(lo), hi=float(hi))


allx = np.arange(nS)
res = dict(tag=TAG, n_scenes=nS, margin_median=MED)
st = float(np.mean([r["land"][f"N_S{q}"] == q for r in rows for q in range(1, N)]))
fid = float(np.mean([r["land"][f"EEE_{k}"] == r["land"][f"N_E{k}"] for r in rows for k in KS] +
                    [r["land"][f"SSS_{k}"] == r["land"][f"N_S{r['anchor'][str(k)]}"] for r in rows for k in KS]))
mx = 0.0
for r in rows:                                                                       # the mass edit vs its target (the reading donor's natural picture mass)
    for k in KS:
        for cell in ("EES", "EESm", "SEE", "SEEm", "SSE", "ESS"):
            rd = r["meta"][f"{cell}_{k}"]["reading"]
            for li, ach in r["img2"][f"{cell}_{k}"].items():
                mx = max(mx, float(np.max(np.abs(np.asarray(ach, float) - np.asarray(r["img_nat"][rd][li], float)))))
d0 = dep(allx, "EEE")
I0 = dict(t1=bool((J["t1_menu"] or 0) >= .95), START=bool(st >= .95), pins=bool(fid >= .95), mass_set=bool(mx <= 1e-3), END_departs=bool(d0 >= .40))
res["I0"] = dict(**I0, all=bool(all(I0.values())), t1_value=J["t1_menu"], START_acc=st, pin_fidelity=fid, mass_max_err=mx, dep_EEE=d0)
ok = res["I0"]["all"]
b1 = boot(lambda ix: dep(ix, "EEE") - dep(ix, "EES"))
res["R1"] = dict(d1=b1, verdict=(("at a fixed END address, START's reading strength reduces END's departure: the departure depends on how the address is read, not only on where it points"
                                  if b1["est"] >= .20 and b1["lo"] > .10 else
                                  "no effective reading-strength effect observed at a fixed address (END state)" if b1["est"] <= .05 and b1["hi"] < .10 else "partial")
                                 if ok else "not read"))
b2 = boot(lambda ix: dep(ix, "SEE") - dep(ix, "SES"))
res["R2"] = dict(d2=b2, verdict=(("at a fixed END address, END's reading strength makes the START state depart" if b2["est"] >= .20 and b2["lo"] > .10 else
                                  "no effective reading-strength effect observed at a fixed address (START state)" if b2["est"] <= .05 and b2["hi"] < .10 else "partial")
                                 if ok else "not read"))
b3 = boot(lambda ix: dep(ix, "EEE") - dep(ix, "SEE"))
res["R3"] = dict(d3=b3, verdict=(("with the address and the reading strength held, the remaining state does not change the departure"
                                  if abs(b3["est"]) <= .10 and b3["lo"] > -.15 and b3["hi"] < .15 else
                                  "the END departure needs more than the early address and its reading strength (the remaining END state)"
                                  if b3["est"] >= .25 and b3["lo"] > .15 else "partial") if ok else "not read"))
desc = dict(dep={c: {str(k): float(np.mean([r["land"][f"{c}_{k}"] != r["anchor"][str(k)] for r in rows])) for k in KS} for c in CELLS},
            dep_pooled={c: dep(allx, c) for c in CELLS})
P = desc["dep_pooled"]
desc["parts"] = dict(d1_mass=P["EEE"] - P["EESm"], d1_gate=P["EEE"] - P["EESg"], d2_mass=P["SEEm"] - P["SES"], d2_gate=P["SEEg"] - P["SES"],
                     address_Estate=P["EEE"] - P["ESE"], address_Sstate=P["SES"] - P["SSS"], reading_at_Saddr_Estate=P["ESE"] - P["ESS"],
                     reading_at_Saddr_Sstate=P["SSE"] - P["SSS"])
desc["halves"] = {h: dict(d1=dep(allx, "EEE", h) - dep(allx, "EES", h), d2=dep(allx, "SEE", h) - dep(allx, "SES", h), d3=dep(allx, "EEE", h) - dep(allx, "SEE", h),
                          dep_EEE=dep(allx, "EEE", h)) for h in ("small", "large")}
idn = [(r, k) for r in rows for k in KS if r["anchor"][str(k)] == N + 1 - k]
desc["identity"] = dict(n=len(idn), acc={c: (float(np.mean([r["land"][f"{c}_{k}"] == N + 1 - k for r, k in idn])) if idn else None) for c in CELLS})
desc["answers"] = {c: {str(k): dict(collections.Counter(r["land"][f"{c}_{k}"] for r in rows)) for k in KS} for c in CELLS}
res["descriptive"] = desc
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"]["all"], R1=[res["R1"]["verdict"], b1["est"]], R2=[res["R2"]["verdict"], b2["est"]], R3=[res["R3"]["verdict"], b3["est"]]), indent=1))
print("->", OUT)
