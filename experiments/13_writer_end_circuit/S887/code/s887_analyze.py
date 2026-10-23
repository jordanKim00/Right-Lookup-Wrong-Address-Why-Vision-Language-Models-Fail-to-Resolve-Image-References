"""S887 analysis (rules fixed before the data; knowledge/DECISIONS.md S887 entry).  Input results/r3/s887_<tag>_qpath.json.
Writer effect W_X = T1(X) - T1(E_BANDR) at L51 (T1 = z_(p-1) - z_p, matcher-head mean LSE block scores, rows B1 / B4 / answer).  FA47's END part
F = W_E - W_ALL47R (FA47's read-row output <- R's).  Path shares (ratios of means over (scene, pin)): f_X = (W_E - W_X) / F for X in QM (layer-46
MLP output <- R's in FA47's read-row queries and gates only), QD (DeltaNet 46), QB (both).  Self-address heads' attention mass on the anchor block
a(run) = mean over rows and the heads 1, 4, 12, 17, 20 of FA47's attention mass on block p (as the run's own queries computed it).  Scene
bootstrap 4,000 (seed 887).
I0: t1 >= .95; START(p) answers p >= .95; L47 anchor = p for E and R >= .90; the identity path QID reproduces E (max late block-score
  difference <= 1e-3 in >= .99 of (scene, pin)); FA47's END part F > 0 with the interval above 0; the self-address heads read the anchor less
  under END (a(R) - a(E) > 0 with the interval above 0).  Nothing is read if I0 fails.
Q1: f_QM >= .50 with the lower bound > .25 -> 'layer 46's MLP output, read through FA47's queries, carries most of FA47's END part';
  f_QM <= .15 with the upper bound < .30 -> 'not'; else 'partial'.
Q2: restoration of the self-address heads' anchor attention under QM: rho = mean[a(E_QM) - a(E)] / mean[a(R) - a(E)] >= .50 with the lower
  bound > .25 -> 'layer 46's MLP END write moves the self-address heads off the anchor'; <= .15 with the upper bound < .30 -> 'not'; else 'partial'.
Q3: sufficiency in START: delta = mean[a(R) - a(R_QME)] / mean[a(R) - a(E)] >= .50 with the lower bound > .25 -> 'layer 46's MLP END write alone
  moves START's self-address heads off the anchor'; <= .15 with the upper bound < .30 -> 'not'; else 'partial'.
Descriptive: f_QD, f_QB, rho for QD / QB, the answers per arm (P(p - 1), P(p)), W_X at L55-63, R_QME's late T1 change.
Usage: python3 s887_analyze.py TAG"""
from __future__ import annotations

import collections, json, sys
import numpy as np

TAG = sys.argv[1]
J = json.load(open(f"/vlm/results/r3/s887_{TAG}_qpath.json"))
OUT = f"/vlm/results/r3/s887_analysis_{TAG}.json"
rng = np.random.default_rng(887); NB = 4000
rows = J["rows"]; S = len(rows); PINS = J["pins"]; lm = J["late_mean"]; LAYS = [str(l) for l in J["lateq"]]
SA = J["selfaddr"]
ARMS = ("ALL47R", "QM", "QD", "QB", "QID")


def zv(r, key, L):
    return np.asarray(r["zl"][key][L], dtype=float)[:, lm[L]].mean(1).mean(0)


def T1(r, key, p, L="51"):
    z = zv(r, key, L); return z[p - 2] - z[p - 1]


def anchor(q):
    q = np.asarray(q); am = [int(np.argmax(q[i])) + 1 for i in (2, 5, 9)]
    c = collections.Counter(am).most_common(); return c[0][0] if c[0][1] >= 2 else am[2]


def amass(r, key, p):
    m = np.asarray(r["mass47"][key], dtype=float)                                        # [3, 24, 8]
    return float(m[:, SA, p - 1].mean())


REC = {}
for si, r in enumerate(rows):
    for p in PINS:
        d = dict(si=si, p=p)
        for L in LAYS:
            tb = T1(r, f"Ep{p}_E_BANDR", p, L); d[f"W_E_{L}"] = T1(r, f"Ep{p}", p, L) - tb
            for X in ARMS:
                d[f"W_{X}_{L}"] = T1(r, f"Ep{p}_{X}", p, L) - tb
        d["T1_R"] = T1(r, f"Rp{p}", p); d["T1_RQME"] = T1(r, f"Rp{p}_QME", p)
        d["vid"] = max(float(np.abs(zv(r, f"Ep{p}_QID", L) - zv(r, f"Ep{p}", L)).max()) for L in LAYS)
        for k_, key in (("aE", f"Ep{p}"), ("aR", f"Rp{p}"), ("aQM", f"Ep{p}_QM"), ("aQD", f"Ep{p}_QD"), ("aQB", f"Ep{p}_QB"), ("aRQME", f"Rp{p}_QME")):
            d[k_] = amass(r, key, p)
        d["anch"] = (anchor(r["qk47"][f"Ep{p}"]), anchor(r["qk47"][f"Rp{p}"]))
        d["land"] = {k[len(f"Ep{p}"):] if k.startswith(f"Ep{p}") else "R" + k[len(f"Rp{p}"):]: v
                     for k, v in r["land"].items() if k.startswith(f"Ep{p}") or k.startswith(f"Rp{p}")}
        REC[(si, p)] = d
SC = sorted({k[0] for k in REC})


def bt(fn):
    n = len(SC); est = fn(np.arange(n)); bs = [fn(rng.integers(0, n, n)) for _ in range(NB)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return dict(est=float(est), lo=float(lo), hi=float(hi))


def M(ix, fn):
    return float(np.mean([fn(REC[(SC[i], p)]) for i in ix for p in PINS]))


F_ = lambda d, L="51": d[f"W_E_{L}"] - d[f"W_ALL47R_{L}"]
res = dict(tag=TAG, n_scenes=S)
st = float(np.mean([r["land"][f"START{p}"] == p for r in rows for p in PINS]))
reach = float(np.mean([a_ == d["p"] for d in REC.values() for a_ in d["anch"]]))
vid_ok = float(np.mean([d["vid"] <= 1e-3 for d in REC.values()]))
bF = bt(lambda ix: M(ix, F_)); bA = bt(lambda ix: M(ix, lambda d: d["aR"] - d["aE"]))
I0 = dict(t1=bool((J["t1_menu"] or 0) >= .95), START=bool(st >= .95), pin_reach=bool(reach >= .90), identity_path=bool(vid_ok >= .99),
          fa47_part=bool(bF["lo"] > 0), attention_drop=bool(bA["lo"] > 0))
res["I0"] = dict(**I0, all=bool(all(I0.values())), t1_value=J["t1_menu"], START_acc=st, reach=reach, vid_ok=vid_ok,
                 vid_max=float(max(d["vid"] for d in REC.values())), F=bF, attention_drop_value=bA)
ok = res["I0"]["all"]
fQM = bt(lambda ix: M(ix, lambda d: d["W_E_51"] - d["W_QM_51"]) / M(ix, F_))
res["Q1"] = dict(f_QM=fQM, verdict=(("layer 46's MLP output, read through FA47's queries, carries most of FA47's END part" if fQM["est"] >= .50 and fQM["lo"] > .25 else
                                     "not" if fQM["est"] <= .15 and fQM["hi"] < .30 else "partial") if ok else "not read"))
rho = bt(lambda ix: M(ix, lambda d: d["aQM"] - d["aE"]) / M(ix, lambda d: d["aR"] - d["aE"]))
res["Q2"] = dict(rho=rho, verdict=(("layer 46's MLP END write moves the self-address heads off the anchor" if rho["est"] >= .50 and rho["lo"] > .25 else
                                    "not" if rho["est"] <= .15 and rho["hi"] < .30 else "partial") if ok else "not read"))
dl = bt(lambda ix: M(ix, lambda d: d["aR"] - d["aRQME"]) / M(ix, lambda d: d["aR"] - d["aE"]))
res["Q3"] = dict(delta=dl, verdict=(("layer 46's MLP END write alone moves START's self-address heads off the anchor" if dl["est"] >= .50 and dl["lo"] > .25 else
                                     "not" if dl["est"] <= .15 and dl["hi"] < .30 else "partial") if ok else "not read"))
allx = range(len(SC))
desc = {f"L{L}": {X: M(allx, lambda d, X=X, L=L: d[f"W_E_{L}"] - d[f"W_{X}_{L}"]) / M(allx, lambda d, L=L: F_(d, L)) for X in ("QM", "QD", "QB")} for L in LAYS}
desc["attention"] = {k_: M(allx, lambda d, k_=k_: d[k_]) for k_ in ("aE", "aR", "aQM", "aQD", "aQB", "aRQME")}
desc["rho_QD"] = M(allx, lambda d: d["aQD"] - d["aE"]) / M(allx, lambda d: d["aR"] - d["aE"])
desc["rho_QB"] = M(allx, lambda d: d["aQB"] - d["aE"]) / M(allx, lambda d: d["aR"] - d["aE"])
desc["mean_W"] = {X: M(allx, lambda d, X=X: d[f"W_{X}_51"]) for X in ("E",) + ARMS}
desc["R_QME_T1_change"] = M(allx, lambda d: d["T1_RQME"] - d["T1_R"])
arms_ = sorted({k for d in REC.values() for k in d["land"]})
desc["answers"] = {a_: dict(P_pm1=float(np.mean([d["land"][a_] == d["p"] - 1 for d in REC.values()])), P_p=float(np.mean([d["land"][a_] == d["p"] for d in REC.values()])))
                   for a_ in arms_}
res["descriptive"] = desc
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"]["all"], Q1=[res["Q1"]["verdict"], fQM["est"]], Q2=[res["Q2"]["verdict"], rho["est"]], Q3=[res["Q3"]["verdict"], dl["est"]]), indent=1))
print("->", OUT)
