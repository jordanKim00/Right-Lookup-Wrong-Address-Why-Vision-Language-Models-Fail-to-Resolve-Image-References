"""S865 analysis (rules fixed before the data; knowledge/DECISIONS.md S865 entry).  Input results/r3/s865_<tag>_xlayout.json.
Per recipient layout N_r in {7, 8, 9} and k in {4, 5}: E = natural END(k) of N_r (same pass as its arms); answer distributions over
{other, 1..9} pooled over scenes; the donor layout's natural distribution D = END(k) of N_d from the recipient pass of N_d.
  transfer fraction tau(arm) = <P(arm) - P(E), P(D) - P(E)> / |P(D) - P(E)|^2 (least-squares weight of the donor layout's distribution),
  pooled over the ordered pairs of a k by averaging numerator and denominator; scene bootstrap 4,000 (seed 865).
I0: t1 >= .95; natural START(4) / START(5) >= .95 (donor passes); U_SELF close to E: mean TV(U_SELF, E) <= .10 over the six cells (cross-pass
  control); U_START moves the early anchor to k (L47 head-mean argmax at rows B1 / B4 / answer, majority = k in >= .90 of runs pooled; the address
  swap acts - its answers are reported as the copy family P(x in {k - 1, k}), since an END run pinned at k takes the writer's step).  Nothing is
  read if I0 fails.
X1 (k = 4, the cells whose natural answers track N): tau >= .5 with the lower bound > .2 'the early address carries the layout-specific
  (N-dependent) part of the answer'; tau <= .15 with the upper bound <= .35 'the N-dependent part is formed after the early lookup'; else
  'partial'.  Valid only if the layouts differ: mean TV(E_r, E_d) over the pairs >= .25.
X1q (k = 4, the whole early query <- the donor layout's, Q_FROM): the same tau and labels with 'the early query' in place of 'the early
  address'; read with X1 (U-only partial or null with X1q >= .5 places the N-dependent part in the early query's non-address parts).
X2 (k = 5, control: natural answers mostly 4 in every layout): the same tau for U and Q, descriptive; the layouts' TV reported.
rho (co-readout, descriptive): per pair, (mean answer position of the arm - of E_r) / (of E_d - of E_r) and the same for the start-ward
  share P(x < k), pooled over pairs with a scene bootstrap.
X3 (run level, k = 4, descriptive): share of runs whose arm answer equals the same scene's donor-layout natural answer, minus the share for E.
Descriptive: per-pair distributions, tau per pair, P(x = j*_r) and P(x = j*_d) changes, late-score profiles.
Usage: python3 s865_analyze.py TAG"""
from __future__ import annotations

import json, sys
import numpy as np

TAG = sys.argv[1]
J = json.load(open(f"/vlm/results/r3/s865_{TAG}_xlayout.json"))
OUT = f"/vlm/results/r3/s865_analysis_{TAG}.json"
rng = np.random.default_rng(865)
NB = 4000
rows = J["rows"]; S = len(rows); LAYS = J["lays"]; KS = J["ks"]


def bt(fn):
    est = fn(np.arange(S)); bs = [fn(rng.integers(0, S, S)) for _ in range(NB)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return dict(est=float(est), lo=float(lo), hi=float(hi))


def ans(key):
    return np.array([r["land"][key] for r in rows])


def dist(x, ix):
    c = np.bincount(np.clip(x[ix], 0, 9), minlength=10)[:10].astype(float)
    return c / max(c.sum(), 1)


PAIRS = [(nr, nd) for nr in LAYS for nd in LAYS if nd != nr]


def tau_fn(k, arm_of, pairs=PAIRS):
    pre = [(ans(f"X{nr}_{arm_of(nd)}_{k}" if arm_of(nd) != "U_SELF" else f"X{nr}_U_SELF{k}"), ans(f"X{nr}_END{k}"), ans(f"X{nd}_END{k}")) for nr, nd in pairs]
    def f(ix):
        num = den = 0.0
        for a, e, dd in pre:
            pa, pe, pd = dist(a, ix), dist(e, ix), dist(dd, ix)
            num += float(np.dot(pa - pe, pd - pe)); den += float(np.dot(pd - pe, pd - pe))
        return num / (den + 1e-12)
    return f


def tv(a, b, ix):
    return .5 * np.abs(dist(a, ix) - dist(b, ix)).sum()


res = dict(tag=TAG, n_scenes=S, n_runs=J["n_runs"])
ix0 = np.arange(S)
st = float(np.mean([r["land"][f"D{n}_START{k}"] == k for r in rows for n in LAYS for k in KS]))
tv_self = float(np.mean([tv(ans(f"X{n}_U_SELF{k}"), ans(f"X{n}_END{k}"), ix0) for n in LAYS for k in KS]))
E47_ = str(J["early"][-1])
def anchor0(r, key):
    q = np.asarray(r["qk"][key][E47_]); am = [int(np.argmax(q[i])) + 1 for i in (2, 5, 9)]
    c = {v: am.count(v) for v in am}; top = max(c, key=c.get)
    return top if c[top] >= 2 else am[2]
u_start = float(np.mean([anchor0(r, f"X{n}_U_START{k}") == k for r in rows for n in LAYS for k in KS]))
u_start_fam = float(np.mean([np.mean(np.isin(ans(f"X{n}_U_START{k}"), [k - 1, k])) for n in LAYS for k in KS]))
I0 = dict(t1=bool((J["t1_menu"] or 0) >= .95), START_acc=bool(st >= .95), U_SELF_close=bool(tv_self <= .10), U_START_acts=bool(u_start >= .90))
res["I0"] = dict(**I0, all=bool(all(I0.values())), t1_value=J["t1_menu"], START_acc_value=st, TV_U_SELF_E=tv_self, U_START_anchor_k=u_start,
                 U_START_copy_family=u_start_fam,
                 TV_U_SELF_by_cell={f"N{n}_k{k}": float(tv(ans(f"X{n}_U_SELF{k}"), ans(f"X{n}_END{k}"), ix0)) for n in LAYS for k in KS})
ok = res["I0"]["all"]
FROM = lambda nd: f"U_FROM{nd}"
QFROM = lambda nd: f"Q_FROM{nd}"
def rho_fn(k, arm_of, stat):
    pre = [(ans(f"X{nr}_{arm_of(nd)}_{k}"), ans(f"X{nr}_END{k}"), ans(f"X{nd}_END{k}")) for nr, nd in PAIRS]
    g = (lambda x: x.astype(float)) if stat == "pos" else (lambda x: ((x < k) & (x >= 1)).astype(float))
    def f(ix):
        num = den = 0.0
        for a, e, dd in pre:
            num += g(a[ix]).mean() - g(e[ix]).mean(); den += g(dd[ix]).mean() - g(e[ix]).mean()
        return num / (den + 1e-12)
    return f
for name, k in (("X1", 4), ("X2", 5)):
    t = bt(tau_fn(k, FROM)); ts = bt(tau_fn(k, lambda nd: "U_SELF")); tq = bt(tau_fn(k, QFROM))
    lay_tv = float(np.mean([tv(ans(f"X{nr}_END{k}"), ans(f"X{nd}_END{k}"), ix0) for nr, nd in PAIRS]))
    per = {f"{nr}<-{nd}": dict(tau=float(tau_fn(k, FROM, [(nr, nd)])(ix0)), TV_layouts=float(tv(ans(f"X{nr}_END{k}"), ans(f"X{nd}_END{k}"), ix0)),
                               E={str(v): float(np.mean(ans(f"X{nr}_END{k}") == v)) for v in range(0, 10) if np.any(ans(f"X{nr}_END{k}") == v)},
                               arm={str(v): float(np.mean(ans(f"X{nr}_U_FROM{nd}_{k}") == v)) for v in range(0, 10) if np.any(ans(f"X{nr}_U_FROM{nd}_{k}") == v)},
                               donor={str(v): float(np.mean(ans(f"X{nd}_END{k}") == v)) for v in range(0, 10) if np.any(ans(f"X{nd}_END{k}") == v)},
                               dP_jr=float(np.mean(ans(f"X{nr}_U_FROM{nd}_{k}") == nr + 1 - k) - np.mean(ans(f"X{nr}_END{k}") == nr + 1 - k)),
                               dP_jd=float(np.mean(ans(f"X{nr}_U_FROM{nd}_{k}") == nd + 1 - k) - np.mean(ans(f"X{nr}_END{k}") == nd + 1 - k)))
           for nr, nd in PAIRS}
    entry = dict(tau=t, tau_U_SELF=ts, tau_Q=tq, TV_layouts=lay_tv, pairs=per,
                 rho=dict(U_pos=bt(rho_fn(k, FROM, "pos")), U_startward=bt(rho_fn(k, FROM, "sw")), Q_pos=bt(rho_fn(k, QFROM, "pos")),
                          Q_startward=bt(rho_fn(k, QFROM, "sw"))))
    for nr, nd in PAIRS:
        per[f"{nr}<-{nd}"]["tau_Q"] = float(tau_fn(k, QFROM, [(nr, nd)])(ix0))
        per[f"{nr}<-{nd}"]["arm_Q"] = {str(v): float(np.mean(ans(f"X{nr}_Q_FROM{nd}_{k}") == v)) for v in range(0, 10) if np.any(ans(f"X{nr}_Q_FROM{nd}_{k}") == v)}
    if name == "X1":
        if not ok:
            v = "not read"
        elif lay_tv < .25:
            v = "invalid (the layouts' natural answers do not differ enough)"
        elif t["est"] >= .5 and t["lo"] > .2:
            v = "the early address carries the layout-specific (N-dependent) part of the answer"
        elif t["est"] <= .15 and t["hi"] <= .35:
            v = "the N-dependent part is formed after the early lookup"
        else:
            v = "partial"
        entry["verdict"] = v
        if not ok or lay_tv < .25:
            vq = v
        elif tq["est"] >= .5 and tq["lo"] > .2:
            vq = "the early query carries the layout-specific (N-dependent) part of the answer"
        elif tq["est"] <= .15 and tq["hi"] <= .35:
            vq = "the N-dependent part is formed after the early lookup"
        else:
            vq = "partial"
        entry["verdict_Q"] = vq
    res[name] = entry
# X3 run level (k = 4)
def x3(ix):
    vals = []
    for nr, nd in PAIRS:
        a, e, dd = ans(f"X{nr}_U_FROM{nd}_4")[ix], ans(f"X{nr}_END4")[ix], ans(f"X{nd}_END4")[ix]
        vals.append(np.mean(a == dd) - np.mean(e == dd))
    return float(np.mean(vals))
res["X3"] = dict(match_donor_change=bt(x3))
# anchor transplant (descriptive; co-analysis 10-21): the recipient's L47 anchor under U_FROM vs the donor run's natural anchor (same scene)
E47 = str(J["early"][-1])
def anchor_r(r, key):
    q = np.asarray(r["qk"][key][E47]); am = [int(np.argmax(q[i])) + 1 for i in (2, 5, 9)]
    c = {v: am.count(v) for v in am}; top = max(c, key=c.get)
    return top if c[top] >= 2 else am[2]
res["anchor_transplant"] = {f"{nr}<-{nd}_k{k}": dict(
    agree_U=float(np.mean([anchor_r(r, f"X{nr}_U_FROM{nd}_{k}") == anchor_r(r, f"X{nd}_END{k}") for r in rows])),
    agree_E=float(np.mean([anchor_r(r, f"X{nr}_END{k}") == anchor_r(r, f"X{nd}_END{k}") for r in rows])))
    for nr, nd in PAIRS for k in KS}
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"]["all"], X1=res["X1"]["verdict"], X1q=res["X1"]["verdict_Q"], tau4=res["X1"]["tau"], tau4Q=res["X1"]["tau_Q"], tau5=res["X2"]["tau"], TV4=res["X1"]["TV_layouts"],
                      TV5=res["X2"]["TV_layouts"]), indent=1))
print("->", OUT)
