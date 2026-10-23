"""S869 analysis (rules fixed before the data; knowledge/DECISIONS.md S869 entry).  Input results/r3/s869_<tag>_dose.json.
Per run at pin p (N 8, END(5) pinned; R = START(p) pinned): flip = [x = p - 1].  Two scores per picture b: the answer logit l(b) (menu word) and
the late matcher score z(b) (L51-63 head-mean, rows B1 / B4 / answer averaged).
  Direction E (writer removed and rescaled in END):  base = E_L0 (lambda 0), unit = E (lambda 1); unseen doses lambda .5, 1.5, 2.
  Direction R (END's writer transplanted into START): base = R (lambda 0), unit = R_L1 (lambda 1); unseen doses lambda 2, 3.
  m_base = s_base(p) - s_base(p - 1);  d1 = [s_unit(p - 1) - s_base(p - 1)] - [s_unit(p) - s_base(p)];  pair rule: flip at lambda iff lambda d1 > m_base;
  full rule: x_hat(lambda) = argmax_b [ s_base(b) + lambda (s_unit(b) - s_base(b)) ].
Scenes 0-23 = calibration half (only for the baselines and W_theta), scenes 24-47 = test half; everything registered below is scored on the
test half; scene bootstrap 4,000 (seed 869) over test scenes.
I0: E_L0 bit-identical to E_BANDR (answers and late scores; the lin path is exact); pin reach (E, R: L43 / L47 argmax at rows B1 / B4 / answer = p)
  >= .95; R answers p >= .90; t1 >= .95; natural START(3..5) >= .95.  Nothing is read if I0 fails.
F1 (answer logits, pair rule, unseen doses pooled over both directions): AUC of (lambda d1 - m_base) for flips >= .80 with the lower bound > .70,
  and accuracy - majority baseline >= .10 with the interval above 0 -> 'the answer flips when the writer's dose-scaled contribution exceeds the
  base margin (answer level)'; else 'not'.
F2 (late matcher scores, same rule): AUC >= .75 with the lower bound > .65 -> 'the late matcher scores carry the flip decision'; else 'not'.
F3 (full rule, exact answer, unseen doses pooled): accuracy(x_hat on logits) - accuracy(baseline) >= .10 with the interval above 0, the baseline
  being the majority answer of the same (direction, pin, dose) cell in the calibration half -> 'the per-run computation predicts unseen-dose
  answers better than cell-level behaviour'; else 'not'.
F4 (W_theta: the writer's effect from its input state): ridge regression (standardised features, unpenalised intercept, penalty 1000) fitted
  on the calibration half from E's L47 per-head block scores (3 rows x 24 heads x offsets -2..+2 around p, centred per head) to d1 (logit level,
  direction E); on the test half, AUC of (d_hat - m_base) for E's own flip at lambda 1 (x_E = p - 1 vs not) >= .70 with the lower bound > .60
  and minus the AUC of (-m_base) (the cell-constant-d rule) >= .05 with the interval above 0 -> "the writer's effect is predictable from its
  input state"; else 'not'.
Descriptive: per-dose flip rates; the same rules on the calibration half; held-out R^2 of d_hat; per-pin values.
Usage: python3 s869_analyze.py TAG"""
from __future__ import annotations

import json, sys
import numpy as np

TAG = sys.argv[1]
J = json.load(open(f"/vlm/results/r3/s869_{TAG}_dose.json"))
OUT = f"/vlm/results/r3/s869_analysis_{TAG}.json"
rng = np.random.default_rng(869)
NB = 4000
rows = J["rows"]; S = len(rows); PINS = J["pins"]
SP = rows[0]["subpass"]
LATE = [str(l) for l in J["late"]]
E47, E43 = str(J["early"][-1]), str(J["early"][-2])
CAL = list(range(S // 2)); TEST = list(range(S // 2, S))


def ref_of(armkey):
    return f"{armkey.split('_')[0]}@{SP[armkey]}"


def zl(r, key):
    return np.mean([np.asarray(r["zl"][key][L], dtype=float)[:, J["late_mean"][L]].mean(1).mean(0) for L in LATE], 0)


def score(r, key, level):
    return np.array(r["mlog"][key], dtype=float) if level == "logit" else zl(r, key)


def auc(sc, y):
    sc, y = np.asarray(sc), np.asarray(y)
    pos, neg = sc[y == 1], sc[y == 0]
    if not len(pos) or not len(neg):
        return np.nan
    return float((np.sum(pos[:, None] > neg[None, :]) + .5 * np.sum(pos[:, None] == neg[None, :])) / (len(pos) * len(neg)))


DIRS = {"E": dict(base="E_L0", unit=None, doses={.5: "E_L05", 1.5: "E_L15", 2.0: "E_L2"}),
        "R": dict(base=None, unit="R_L1", doses={2.0: "R_L2", 3.0: "R_L3"})}


def records(level):
    out = []
    for p in PINS:
        for si, r in enumerate(rows):
            for dname, dd in DIRS.items():
                if dname == "E":
                    kb = f"E5p{p}_E_L0"; ku = ref_of(kb)                                   # unit = E itself (same sub-pass)
                else:
                    ku = f"Rp{p}_R_L1"; kb = ref_of(ku)                                    # base = R itself
                sb, su = score(r, kb, level), score(r, ku, level)
                mb = sb[p - 1] - sb[p - 2]; d1 = (su[p - 2] - sb[p - 2]) - (su[p - 1] - sb[p - 1])
                for lam, arm in dd["doses"].items():
                    key = f"{'E5' if dname == 'E' else 'R'}p{p}_{arm}"
                    x = r["land"][key]
                    xhat = int(np.argmax(sb + lam * (su - sb))) + 1
                    out.append(dict(si=si, p=p, dir=dname, lam=lam, y=int(x == p - 1), x=x, s=lam * d1 - mb, pair=int(lam * d1 > mb), xhat=xhat))
    return out


def bt_scenes(fn, scenes):
    scenes = np.array(scenes)
    est = fn(scenes); bs = [fn(rng.choice(scenes, len(scenes))) for _ in range(NB)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return dict(est=float(est), lo=float(lo), hi=float(hi))


res = dict(tag=TAG, n_scenes=S, n_runs=J["n_runs"], calibration_scenes=len(CAL), test_scenes=len(TEST))
# ---------------------------------------------------------------- I0
self_ok = True
for p in PINS:
    for r in rows:
        a, b = f"E5p{p}_E_L0", f"E5p{p}_E_BANDR"
        if r["land"][a] != r["land"][b] or not all(np.array_equal(np.asarray(r["zl"][a][L]), np.asarray(r["zl"][b][L])) for L in r["zl"][b]):
            self_ok = False
reach = []
for p in PINS:
    for rk in ("E5", "R"):
        for r in rows:
            for L in (E43, E47):
                q = np.asarray(r["qk"][f"{rk}p{p}"][L]); reach += [int(np.argmax(q[i])) + 1 == p for i in (2, 5, 9)]
racc = float(np.mean([r["land"][f"Rp{p}"] == p for r in rows for p in PINS]))
st = float(np.mean([r["land"][f"START{p}"] == p for r in rows for p in PINS]))
I0 = dict(lin_bit_identical=bool(self_ok), pin_reach=bool(np.mean(reach) >= .95), R_acc=bool(racc >= .90), t1=bool((J["t1_menu"] or 0) >= .95),
          START_acc=bool(st >= .95))
res["I0"] = dict(**I0, all=bool(all(I0.values())), pin_reach_value=float(np.mean(reach)), R_acc_value=racc, t1_value=J["t1_menu"], START_acc_value=st)
ok = res["I0"]["all"]
# ---------------------------------------------------------------- F1, F2
REC = {lv: records(lv) for lv in ("logit", "late")}
BYS = {lv: {} for lv in REC}
for lv in REC:
    for q in REC[lv]:
        BYS[lv].setdefault(q["si"], []).append(q)
def pick(lv, scenes):
    """records of the scenes in the (re)sample, repeated scenes repeated."""
    return [q for sc in scenes for q in BYS[lv].get(int(sc), [])]
def pair_stats(lv, scenes):
    R = pick(lv, scenes)
    y = np.array([q["y"] for q in R]); s = np.array([q["s"] for q in R]); pr = np.array([q["pair"] for q in R])
    return y, s, pr
def f_auc(lv):
    return lambda sc: auc(*[pair_stats(lv, list(sc))[i] for i in (1, 0)])
def f_accgain(lv):
    def f(sc):
        y, s, pr = pair_stats(lv, list(sc))
        return float(np.mean(pr == y) - max(y.mean(), 1 - y.mean()))
    return f
for name, lv, thr, lob in (("F1", "logit", .80, .70), ("F2", "late", .75, .65)):
    a_ = bt_scenes(f_auc(lv), TEST); g_ = bt_scenes(f_accgain(lv), TEST)
    y, s, pr = pair_stats(lv, TEST)
    if name == "F1":
        v = ("the answer flips when the writer's dose-scaled contribution exceeds the base margin (answer level)"
             if a_["est"] >= thr and a_["lo"] > lob and g_["est"] >= .10 and g_["lo"] > 0 else "not")
    else:
        v = "the late matcher scores carry the flip decision" if a_["est"] >= thr and a_["lo"] > lob else "not"
    res[name] = dict(AUC=a_, acc_minus_majority=g_, flip_rate=float(y.mean()), n=int(len(y)),
                     calibration_half=dict(AUC=f_auc(lv)(CAL), acc_minus_majority=f_accgain(lv)(CAL)), verdict=(v if ok else "not read"))
# ---------------------------------------------------------------- F3
cal_mode = {}
for q in REC["logit"]:
    if q["si"] in set(CAL):
        cal_mode.setdefault((q["dir"], q["p"], q["lam"]), []).append(q["x"])
cal_mode = {c: max(set(v), key=v.count) for c, v in cal_mode.items()}
def f3(sc):
    R = pick("logit", sc)
    return float(np.mean([q["xhat"] == q["x"] for q in R]) - np.mean([cal_mode[(q["dir"], q["p"], q["lam"])] == q["x"] for q in R]))
f3v = bt_scenes(f3, TEST)
Rt = [q for q in REC["logit"] if q["si"] in set(TEST)]
res["F3"] = dict(gain=f3v, acc_rule=float(np.mean([q["xhat"] == q["x"] for q in Rt])), acc_baseline=float(np.mean([cal_mode[(q["dir"], q["p"], q["lam"])] == q["x"] for q in Rt])),
                 verdict=(("the per-run computation predicts unseen-dose answers better than cell-level behaviour" if f3v["est"] >= .10 and f3v["lo"] > 0 else "not")
                          if ok else "not read"))
# ---------------------------------------------------------------- F4
def feats(r, key, p):
    z = np.asarray(r["z47"][key], dtype=float); z = z - z.mean(-1, keepdims=True)
    return np.stack([z[:, :, p - 1 + o] if 0 <= p - 1 + o < z.shape[-1] else np.zeros(z.shape[:2]) for o in (-2, -1, 0, 1, 2)], -1).ravel()
F = []
for p in PINS:
    for si, r in enumerate(rows):
        kb = f"E5p{p}_E_L0"; ku = ref_of(kb)
        sb, su = score(r, kb, "logit"), score(r, ku, "logit")
        F.append(dict(si=si, p=p, x=feats(r, ku, p), d=(su[p - 2] - sb[p - 2]) - (su[p - 1] - sb[p - 1]), mb=sb[p - 1] - sb[p - 2],
                      y=int(r["land"][ku] == p - 1)))
Xc = np.array([f["x"] for f in F if f["si"] in set(CAL)]); Yc = np.array([f["d"] for f in F if f["si"] in set(CAL)])
mu, sd = Xc.mean(0), Xc.std(0) + 1e-6; ym = Yc.mean()
A = (Xc - mu) / sd
w = np.linalg.solve(A.T @ A + 1000.0 * np.eye(A.shape[1]), A.T @ (Yc - ym))
for f in F:
    f["dhat"] = float(((f["x"] - mu) / sd) @ w + ym)
FT = [f for f in F if f["si"] in set(TEST)]
BYF = {}
for f in FT:
    BYF.setdefault(f["si"], []).append(f)
pickf = lambda sc: [f for x in sc for f in BYF.get(int(x), [])]
yt = np.array([f["d"] for f in FT]); pt = np.array([f["dhat"] for f in FT])
r2 = float(1 - np.sum((yt - pt) ** 2) / np.sum((yt - ym) ** 2))
def f4a(sc):
    R = pickf(sc)
    return auc([f["dhat"] - f["mb"] for f in R], [f["y"] for f in R])
def f4d(sc):
    R = pickf(sc)
    return auc([f["dhat"] - f["mb"] for f in R], [f["y"] for f in R]) - auc([-f["mb"] for f in R], [f["y"] for f in R])
a4, d4 = bt_scenes(f4a, TEST), bt_scenes(f4d, TEST)
res["F4"] = dict(AUC=a4, AUC_minus_constant_d=d4, heldout_R2_d=r2, flip_rate_E=float(np.mean([f["y"] for f in FT])),
                 verdict=(("the writer's effect is predictable from its input state" if a4["est"] >= .70 and a4["lo"] > .60 and d4["est"] >= .05 and d4["lo"] > 0
                           else "not") if ok else "not read"))
# ---------------------------------------------------------------- descriptive
res["descriptive"] = dict(flip_rate_by_dose={f"{d}_{lam}": float(np.mean([q["y"] for q in REC["logit"] if q["dir"] == d and q["lam"] == lam]))
                                             for d in DIRS for lam in DIRS[d]["doses"]},
                          E_flip_at_1=float(np.mean([r["land"][ref_of(f"E5p{p}_E_L0")] == p - 1 for r in rows for p in PINS])),
                          R_flip_at_0=float(np.mean([r["land"][f"Rp{p}"] == p - 1 for r in rows for p in PINS])),
                          by_pin={str(p): {f"{q}": float(np.mean([x["y"] for x in REC["logit"] if x["p"] == p and x["dir"] == q])) for q in DIRS} for p in PINS})
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"]["all"], F1=res["F1"]["verdict"], F2=res["F2"]["verdict"], F3=res["F3"]["verdict"], F4=res["F4"]["verdict"]), indent=1))
print("->", OUT)
