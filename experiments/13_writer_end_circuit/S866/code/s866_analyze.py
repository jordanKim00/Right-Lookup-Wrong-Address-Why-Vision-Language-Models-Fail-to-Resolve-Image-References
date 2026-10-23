"""S866 analysis (rules fixed before the data; knowledge/DECISIONS.md S866 entry).  Input results/r3/s866_<tag>_suff.json.
Per run at pin p (N = 8, p in {3, 4, 5}): back b = [x = p - 1], stay s = [x = p]; late profiles z_L = head-mean (late_mean) block scores, rows
B1 / B4 / answer averaged; T' = z[p-1] - z[p] averaged over the four late layers.  Arms on R = START(p) pinned at p are compared with the R of
their own sub-pass, scene by scene, pooled over the three pins (each pin also reported); scene bootstrap 4,000 (seed 866).
  D(X) = R'(R <- X) - R'(R)  for R' = b (primary) or T' (co-primary);  gap = b(E5) - b(R).
I0: SELF_FA and SELF_TAIL bit-identical to their sub-pass R (answers and late scores); pin reach (E5, E2, R: L43 / L47 argmax at rows B1 / B4 /
  answer = p) >= .95; R(p) answers p >= .90; t1 >= .95; natural START(3..5) >= .95.  Nothing is read if I0 fails.
S0: b(E5) - b(E2) >= .20 with the interval above 0 'the END step is on at k 5 and off at k 2 at the same anchors'; else S2 is not read.
S1 (sufficiency): D_b(BAND5) >= .20 with the interval above 0 "the band's END output is sufficient to make a pointer step back from its anchor";
  phi = D_b(BAND5) / gap reported (the share of END's own one-back reproduced in a START run).
  S1-id (headline, p 4 = END5's identity pin, R(4) a correct pointer): D_b(BAND5) at p 4 >= .20 with the interval above 0 "the band's END output
  transplants the identity break into a correct pointer".
S2 (the switch is in the band's output): D_b(BAND2) <= .05 with the upper bound <= .10, and D_b(BAND5) - D_b(BAND2) >= .15 with the interval
  above 0 "k 2's band output does not make the pointer step - the k-switch is carried by the band's output"; BAND3 reported (the k ladder).
S3 (which part suffices; f(X) = D(X) / D(BAND5) per readout): X in {FA47_5, H16_5, ANTI_5, SELFADDR_5, REST47_5, MLPS5}: 'X is sufficient'
  (f >= .5 with the lower bound > 0, b), 'X contributes' (lower bound > 0, b or T'), else 'not observed'; the five-head sets (ANTI_5, SELFADDR_5)
  must also exceed the count-matched RAND5a_5 (difference's lower bound > 0, b or T'); RAND1a_5 / RAND1b_5 / RAND5a_5 |D_b| <= .05 'random heads
  do not make the pointer step'.
S5 (dose, descriptive with a registered monotonicity label): D_b at lambda .5 / 1 / 2 (BAND5_L05, BAND5, BAND5_L2): 'graded' if the point
  estimates are ordered and D_b(L2) - D_b(L05) >= .10 with the interval above 0, else 'not graded'.
S6 (cross-anchor, pins 3 and 5): R(p') <- E5(p)'s band: change of P(x = p - 1) (the donor's predecessor) and of P(x = p' - 1) (the recipient's),
  pooled over (3 -> 5) and (5 -> 3): 'anchor-keyed (the band carries an address)' if the donor's predecessor rises >= .20 with the interval above
  0; 'relative operation' if the recipient's predecessor rises >= .20 (interval above 0) and the donor's does not; else 'mixed'.
S4 (everything before the late lookup): D_b(L50RES5) >= .7 gap with the lower bound > .4 gap "the read rows' state before the late lookup
  carries END's whole step"; D_b(L50RES2) <= .05 'k 2's pre-late state does not'.
Descriptive: TAIL5 vs BAND5; per-pin answer distributions (p 4: does the pointer at END5's identity pin move to 3); U changes.
Usage: python3 s866_analyze.py TAG"""
from __future__ import annotations

import json, sys
import numpy as np

TAG = sys.argv[1]
J = json.load(open(f"/vlm/results/r3/s866_{TAG}_suff.json"))
OUT = f"/vlm/results/r3/s866_analysis_{TAG}.json"
rng = np.random.default_rng(866)
NB = 4000
rows = J["rows"]; S = len(rows); N = J["N"]; PINS = J["pins"]
SP = rows[0]["subpass"]
LATE = [str(l) for l in J["late"]]
E47, E43 = str(J["early"][-1]), str(J["early"][-2])


def bt(fn):
    est = fn(np.arange(S)); bs = [fn(rng.integers(0, S, S)) for _ in range(NB)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return dict(est=float(est), lo=float(lo), hi=float(hi))


def ref_of(armkey):
    return f"{armkey.split('_')[0]}@{SP[armkey]}"


def zl(r, key, L):
    return np.asarray(r["zl"][key][L], dtype=float)[:, J["late_mean"][L]].mean(1).mean(0)


CACHE = {}
def meas(key, p):
    if key not in CACHE:
        x = np.array([r["land"][key] for r in rows])
        Z = {L: np.array([zl(r, key, L) for r in rows]) for L in LATE}
        Tp = np.mean([Z[L][:, p - 2] - Z[L][:, p - 1] for L in LATE], 0)
        z0 = Z[LATE[0]]
        U = (z0[:, p - 2] + z0[:, p]) / 2 - z0[:, p - 1]
        CACHE[key] = dict(x=x, b=(x == p - 1).astype(float), s=(x == p).astype(float), Tp=Tp, U=U)
    return CACHE[key]


def D(arm, R, pins=None):
    pre = []
    for p in (pins or PINS):
        key = f"Rp{p}_{arm}"; pre.append(meas(key, p)[R] - meas(ref_of(key), p)[R])
    return lambda ix: float(np.mean([x[ix].mean() for x in pre]))


def lev(rk, R, pins=None):
    pre = [meas(f"{rk}p{p}", p)[R] for p in (pins or PINS)]
    return lambda ix: float(np.mean([x[ix].mean() for x in pre]))


res = dict(tag=TAG, n_scenes=S, n_runs=J["n_runs"])
# ---------------------------------------------------------------- I0
self_ok = True
for p in PINS:
    for arm in ("SELF_FA", "SELF_TAIL"):
        key = f"Rp{p}_{arm}"; rk = ref_of(key)
        for r in rows:
            if r["land"][key] != r["land"][rk] or not all(np.array_equal(np.asarray(r["zl"][key][L]), np.asarray(r["zl"][rk][L])) for L in r["zl"][rk]):
                self_ok = False
reach = []
for p in PINS:
    for rk in ("E5", "E2", "R"):
        for r in rows:
            for L in (E43, E47):
                q = np.asarray(r["qk"][f"{rk}p{p}"][L]); reach += [int(np.argmax(q[i])) + 1 == p for i in (2, 5, 9)]
racc = float(np.mean([r["land"][f"Rp{p}"] == p for r in rows for p in PINS]))
st = float(np.mean([r["land"][f"START{p}"] == p for r in rows for p in PINS]))
I0 = dict(self_bit_identical=bool(self_ok), pin_reach=bool(np.mean(reach) >= .95), R_acc=bool(racc >= .90), t1=bool((J["t1_menu"] or 0) >= .95),
          START_acc=bool(st >= .95))
res["I0"] = dict(**I0, all=bool(all(I0.values())), pin_reach_value=float(np.mean(reach)), R_acc_value=racc, t1_value=J["t1_menu"], START_acc_value=st)
ok = res["I0"]["all"]
# ---------------------------------------------------------------- S0
s0 = bt(lambda ix: lev("E5", "b")(ix) - lev("E2", "b")(ix))
gap = bt(lambda ix: lev("E5", "b")(ix) - lev("R", "b")(ix))
res["S0"] = dict(E5_minus_E2=s0, gap_E5_minus_R=gap, levels={rk: {str(p): dict(b=float(meas(f"{rk}p{p}", p)["b"].mean()), s=float(meas(f"{rk}p{p}", p)["s"].mean()))
                                                                     for p in PINS} for rk in ("E5", "E2", "R")},
                 verdict=(("the END step is on at k 5 and off at k 2 at the same anchors" if s0["est"] >= .20 and s0["lo"] > 0 else "not") if ok else "not read"))
# ---------------------------------------------------------------- S1, S2
s1 = bt(D("BAND5", "b")); phi = bt(lambda ix: D("BAND5", "b")(ix) / (lev("E5", "b")(ix) - lev("R", "b")(ix) + 1e-12))
res["S1"] = dict(D_b=s1, phi=phi, D_Tp=bt(D("BAND5", "Tp")), by_pin={str(p): D("BAND5", "b", [p])(np.arange(S)) for p in PINS},
                 verdict=(("the band's END output is sufficient to make a pointer step back from its anchor" if s1["est"] >= .20 and s1["lo"] > 0 else "not")
                          if ok else "not read"))
s1id = bt(D("BAND5", "b", [4]))
res["S1"]["identity_p4"] = s1id
res["S1"]["verdict_id"] = (("the band's END output transplants the identity break into a correct pointer" if s1id["est"] >= .20 and s1id["lo"] > 0 else "not")
                           if ok else "not read")
s2 = bt(D("BAND2", "b")); s2d = bt(lambda ix: D("BAND5", "b")(ix) - D("BAND2", "b")(ix))
res["S2"] = dict(D_b=s2, diff=s2d, D_Tp=bt(D("BAND2", "Tp")), D_b_BAND3=bt(D("BAND3", "b")), D_Tp_BAND3=bt(D("BAND3", "Tp")),
                 verdict=(("k 2's band output does not make the pointer step - the k-switch is carried by the band's output"
                           if s2["est"] <= .05 and s2["hi"] <= .10 and s2d["est"] >= .15 and s2d["lo"] > 0 else "not")
                          if (ok and res["S0"]["verdict"] != "not") else "not read"))
# ---------------------------------------------------------------- S3
S3 = {}
for X in ("FA47_5", "H16_5", "ANTI_5", "SELFADDR_5", "REST47_5", "MLPS5"):
    fb = bt(lambda ix, X=X: D(X, "b")(ix) / (D("BAND5", "b")(ix) + 1e-12))
    ft = bt(lambda ix, X=X: D(X, "Tp")(ix) / (D("BAND5", "Tp")(ix) + 1e-12))
    ok5 = True
    if X in ("ANTI_5", "SELFADDR_5"):
        db5 = bt(lambda ix, X=X: D(X, "b")(ix) - D("RAND5a_5", "b")(ix)); dt5 = bt(lambda ix, X=X: D(X, "Tp")(ix) - D("RAND5a_5", "Tp")(ix))
        ok5 = bool(db5["lo"] > 0 or dt5["lo"] > 0)
    v = ("X is sufficient" if (fb["est"] >= .5 and fb["lo"] > 0 and ok5) else "X contributes" if ((fb["lo"] > 0 or ft["lo"] > 0) and ok5)
         else "not observed").replace("X", X)
    S3[X] = dict(f_b=fb, f_Tp=ft, D_b=bt(D(X, "b")), D_Tp=bt(D(X, "Tp")), verdict=(v if (ok and res["S1"]["verdict"] != "not") else "not read"))
rnd = {X: bt(D(X, "b")) for X in ("RAND1a_5", "RAND1b_5", "RAND5a_5")}
S3["random"] = dict(D_b=rnd, verdict=(("random heads do not make the pointer step" if all(abs(v["est"]) <= .05 for v in rnd.values()) else "random heads move it")
                                      if ok else "not read"))
res["S3"] = S3
# ---------------------------------------------------------------- S4
s4 = bt(D("L50RES5", "b")); s4b = bt(D("L50RES2", "b"))
g = gap["est"]
res["S4"] = dict(D_b=s4, D_b_k2=s4b, gap=g,
                 verdict=(("the read rows' state before the late lookup carries END's whole step" if s4["est"] >= .7 * g and s4["lo"] > .4 * g else "not")
                          if ok else "not read"),
                 verdict_k2=(("k 2's pre-late state does not" if s4b["est"] <= .05 else "not") if ok else "not read"))
# ---------------------------------------------------------------- S5 dose, S6 cross-anchor
d05, d1, d2 = bt(D("BAND5_L05", "b")), s1, bt(D("BAND5_L2", "b")); dd = bt(lambda ix: D("BAND5_L2", "b")(ix) - D("BAND5_L05", "b")(ix))
res["S5"] = dict(D_b={"0.5": d05, "1": d1, "2": d2}, D_Tp={"0.5": bt(D("BAND5_L05", "Tp")), "1": bt(D("BAND5", "Tp")), "2": bt(D("BAND5_L2", "Tp"))},
                 diff_2_minus_05=dd, verdict=(("graded" if d05["est"] <= d1["est"] <= d2["est"] and dd["est"] >= .10 and dd["lo"] > 0 else "not graded")
                                              if ok else "not read"))
XP = J["xpins"]; p_, q_ = XP
def xshare(target_of):
    pre = []
    for pr, pd in ((p_, q_), (q_, p_)):
        key = f"XRp{pr}_BAND5X{pd}"; rk = ref_of(key); t = target_of(pr, pd)
        pre.append((np.array([r["land"][key] for r in rows]) == t).astype(float) - (np.array([r["land"][rk] for r in rows]) == t).astype(float))
    return lambda ix: float(np.mean([x[ix].mean() for x in pre]))
xd = bt(xshare(lambda pr, pd: pd - 1)); xr = bt(xshare(lambda pr, pd: pr - 1))
xs_d = bt(lambda ix: float(np.mean([((np.array([r["land"][f"XRp{pr}_BAND5"] for r in rows]) == pr - 1).astype(float)
                                     - (np.array([r["land"][ref_of(f"XRp{pr}_BAND5")] for r in rows]) == pr - 1).astype(float))[ix].mean() for pr in (p_, q_)])))
res["S6"] = dict(donor_pred=xd, recipient_pred=xr, same_pin_reference=xs_d,
                 verdict=(("anchor-keyed (the band carries an address)" if xd["est"] >= .20 and xd["lo"] > 0 else
                           "relative operation" if (xr["est"] >= .20 and xr["lo"] > 0 and not xd["lo"] > 0) else "mixed") if ok else "not read"))
# ---------------------------------------------------------------- descriptive
desc = {}
for p in PINS:
    for key in [f"{rk}p{p}" for rk in ("E5", "E2", "R")] + [f"Rp{p}_{arm}" for arm, _, _, _ in J["specs"]]:
        if key in rows[0]["land"]:
            x = np.array([r["land"][key] for r in rows]); desc[key] = {str(v): float(np.mean(x == v)) for v in range(1, 10) if np.any(x == v)}
for key in [k_ for k_ in rows[0]["land"] if k_.startswith("XR") and "@" not in k_]:
    x = np.array([r["land"][key] for r in rows]); desc[key] = {str(v): float(np.mean(x == v)) for v in range(1, 10) if np.any(x == v)}
res["descriptive"] = dict(answers=desc, TAIL5_minus_BAND5=bt(lambda ix: D("TAIL5", "b")(ix) - D("BAND5", "b")(ix)), D_b_TAIL5=bt(D("TAIL5", "b")),
                          L50RES5_minus_BAND5=bt(lambda ix: D("L50RES5", "b")(ix) - D("BAND5", "b")(ix)),
                          U={X: float(D(X, "U")(np.arange(S))) for X in ("BAND5", "BAND2", "TAIL5", "FA47_5", "H16_5", "ANTI_5", "SELFADDR_5", "MLPS5", "L50RES5")})
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"]["all"], S0=res["S0"]["verdict"], S1=[res["S1"]["verdict"], res["S1"]["verdict_id"]], S2=res["S2"]["verdict"],
                      S5=res["S5"]["verdict"], S6=res["S6"]["verdict"],
                      S3={k: v["verdict"] for k, v in res["S3"].items()}, S4=[res["S4"]["verdict"], res["S4"]["verdict_k2"]]), indent=1))
print("->", OUT)
