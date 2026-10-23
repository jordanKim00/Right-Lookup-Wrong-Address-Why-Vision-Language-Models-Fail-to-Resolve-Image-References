"""S867 analysis (rules fixed before the data; knowledge/DECISIONS.md S867 entry).  Input results/r3/s867_<tag>_xn.json.
Per run at pin p (END(5) pinned at p in {3, 6} in layouts N 8 and 9): landing x (the answered picture; 0 = other, excluded from the mean),
one-back b = [x = p - 1]; arms are compared with the pinned reference of their own pass (same scene); scene bootstrap 4,000 (seed 867).
  N-shift  Delta_N = mean over p of [mean landing(N9 E5p) - mean landing(N8 E5p)]   (the adversary's S862 post hoc: +.44)
  rho(X)   = mean over p of {[landing(N8 X) - landing(N8 E5p)] + [landing(N9 E5p) - landing(N9 X)]} / (2 Delta_N)  for X in {BAND_XN, LQ_XN,
             BOTH_XN}: the share of the N-shift that the transplant moves, both directions pooled.
I0: t1 >= .95; natural START(3) / START(6) >= .95; pin reach (pinned references: L43 / L47 argmax at rows B1 / B4 / answer = p) >= .95; the
  cross-pass controls BAND_SELF and LQ_SELF close to their reference: mean |landing change| <= .10 picture and answer agreement >= .90.
  Nothing is read if I0 fails.
N0: Delta_N > 0 with the interval above 0 'the pinned landing depends on N' (else X1-X2 not read - no N-shift to locate).
X1: rho(BAND_XN) >= .5 with the lower bound > .2 'the band carries the N-dependent landing (N-aware band)'; rho(BAND_XN) <= .2 with the upper bound
  <= .35 'the band does not carry it (N-blind band)'; else 'partial'.
X2: rho(LQ_XN) >= .6 with the lower bound > .3 'the late queries carry the N-dependent landing (late visible-end term)'; <= .2 with the upper bound
  <= .35 'not carried by the late queries'; else 'partial'.
Descriptive: rho(BOTH_XN); CUT_LAST (late reads of the last picture removed) landing change per layout; one-back changes; per-pin values;
  answer distributions.
Usage: python3 s867_analyze.py TAG"""
from __future__ import annotations

import json, sys
import numpy as np

TAG = sys.argv[1]
J = json.load(open(f"/vlm/results/r3/s867_{TAG}_xn.json"))
OUT = f"/vlm/results/r3/s867_analysis_{TAG}.json"
rng = np.random.default_rng(867)
NB = 4000
rows = J["rows"]; S = len(rows); PINS = J["pins"]
E47, E43 = str(J["early"][-1]), str(J["early"][-2])


def bt(fn):
    est = fn(np.arange(S)); bs = [fn(rng.integers(0, S, S)) for _ in range(NB)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return dict(est=float(est), lo=float(lo), hi=float(hi))


def ans(key):
    return np.array([r["land"][key] for r in rows], dtype=float)


def mland(x, ix):
    v = x[ix]; v = v[v > 0]
    return float(v.mean()) if len(v) else np.nan


def key(n, nm, p):
    return f"X{n}_E5p{p}" if nm == "E" else f"X{n}_{nm}_p{p}"


res = dict(tag=TAG, n_scenes=S, n_runs=J["n_runs"])
st = float(np.mean([r["land"][f"S{n}_START{p}"] == p for r in rows for n in J["lays"] for p in PINS]))
reach = []
for n in J["lays"]:
    for p in PINS:
        for r in rows:
            for L in (E43, E47):
                q = np.asarray(r["qk"][key(n, "E", p)][L]); reach += [int(np.argmax(q[i])) + 1 == p for i in (2, 5, 9)]
ctrl = {}
for nm in ("BAND_SELF", "LQ_SELF"):
    ch = [abs(mland(ans(key(n, nm, p)), np.arange(S)) - mland(ans(key(n, "E", p)), np.arange(S))) for n in J["lays"] for p in PINS]
    ag = [float(np.mean(ans(key(n, nm, p)) == ans(key(n, "E", p)))) for n in J["lays"] for p in PINS]
    ctrl[nm] = dict(mean_abs_landing_change=float(np.mean(ch)), agreement=float(np.mean(ag)), ok=bool(np.mean(ch) <= .10 and np.mean(ag) >= .90))
I0 = dict(t1=bool((J["t1_menu"] or 0) >= .95), START_acc=bool(st >= .95), pin_reach=bool(np.mean(reach) >= .95),
          BAND_SELF=ctrl["BAND_SELF"]["ok"], LQ_SELF=ctrl["LQ_SELF"]["ok"])
res["I0"] = dict(**I0, all=bool(all(I0.values())), t1_value=J["t1_menu"], START_acc_value=st, pin_reach_value=float(np.mean(reach)), controls=ctrl)
ok = res["I0"]["all"]
A = {(n, nm, p): ans(key(n, nm, p)) for n in J["lays"] for p in PINS for nm in ("E", "BAND_SELF", "BAND_XN", "LQ_SELF", "LQ_XN", "BOTH_XN", "CUT_LAST")}
def dN(ix):
    return float(np.mean([mland(A[(9, "E", p)], ix) - mland(A[(8, "E", p)], ix) for p in PINS]))
def rho(nm):
    def f(ix):
        num = np.mean([(mland(A[(8, nm, p)], ix) - mland(A[(8, "E", p)], ix)) + (mland(A[(9, "E", p)], ix) - mland(A[(9, nm, p)], ix)) for p in PINS])
        return float(num / (2 * dN(ix) + 1e-12))
    return f
n0 = bt(dN)
res["N0"] = dict(Delta_N=n0, by_pin={str(p): mland(A[(9, "E", p)], np.arange(S)) - mland(A[(8, "E", p)], np.arange(S)) for p in PINS},
                 verdict=(("the pinned landing depends on N" if n0["lo"] > 0 else "not") if ok else "not read"))
read = ok and res["N0"]["verdict"] != "not"
rb, rl, rbo = bt(rho("BAND_XN")), bt(rho("LQ_XN")), bt(rho("BOTH_XN"))
res["X1"] = dict(rho=rb, verdict=(("the band carries the N-dependent landing (N-aware band)" if rb["est"] >= .5 and rb["lo"] > .2 else
                                   "the band does not carry it (N-blind band)" if rb["est"] <= .2 and rb["hi"] <= .35 else "partial") if read else "not read"))
res["X2"] = dict(rho=rl, verdict=(("the late queries carry the N-dependent landing (late visible-end term)" if rl["est"] >= .6 and rl["lo"] > .3 else
                                   "not carried by the late queries" if rl["est"] <= .2 and rl["hi"] <= .35 else "partial") if read else "not read"))
res["descriptive"] = dict(rho_BOTH=rbo,
                          CUT_LAST={str(n): {str(p): mland(A[(n, "CUT_LAST", p)], np.arange(S)) - mland(A[(n, "E", p)], np.arange(S)) for p in PINS} for n in J["lays"]},
                          one_back={f"N{n}_p{p}_{nm}": float(np.mean(A[(n, nm, p)] == p - 1)) for n in J["lays"] for p in PINS
                                    for nm in ("E", "BAND_XN", "LQ_XN", "BOTH_XN", "CUT_LAST")},
                          landing={f"N{n}_p{p}_{nm}": mland(A[(n, nm, p)], np.arange(S)) for n in J["lays"] for p in PINS
                                   for nm in ("E", "BAND_SELF", "BAND_XN", "LQ_SELF", "LQ_XN", "BOTH_XN", "CUT_LAST")},
                          answers={f"N{n}_p{p}_{nm}": {str(int(v)): float(np.mean(A[(n, nm, p)] == v)) for v in range(0, 10) if np.any(A[(n, nm, p)] == v)}
                                   for n in J["lays"] for p in PINS for nm in ("E", "BAND_XN", "LQ_XN", "BOTH_XN", "CUT_LAST")})
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"]["all"], N0=res["N0"]["verdict"], X1=res["X1"]["verdict"], X2=res["X2"]["verdict"]), indent=1))
print("->", OUT)
