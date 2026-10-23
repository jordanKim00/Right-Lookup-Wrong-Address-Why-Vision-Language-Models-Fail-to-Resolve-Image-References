"""S864 analysis (rules fixed before the data; knowledge/DECISIONS.md S864 entry; revised with the design after the four agents' critiques,
10-21 00:2x, before any data).  Input results/r3/s864_<tag>_opident.json.
Per run at pin p (N = 8, p in {3, 4, 5}): back b = [x = p - 1], stay s = [x = p], leave l = [x != p], target hits t4 = [x = 4] (BEF5 / END5)
and t6 = [x = 6] (AFT5); late profiles z_L = head-mean (late_mean) block scores, rows B1 / B4 / answer averaged; T' = z[p-1] - z[p] averaged
over the four late layers.  Arms are compared with their recipient of the same sub-pass, scene by scene; scene bootstrap 4,000 (seed 864).
I0: every SELF_FA bit-identical to its recipient; pin reach (all pinned refs: L43 / L47 argmax at rows B1 / B4 / answer = p) >= .95; R(p) answers
  p >= .90; t1 >= .95; natural START(3..5) >= .95.  Nothing is read if I0 fails.
O0 (how explicit 'before' behaves under the pin; the target of BEF5 is 4 at every pin):
  PC (positive control, p 5): P(x = 4 | BEF5) >= .50 'the pinned BEF form is read'; else O0-O3 are not read (instrument).
  ID (BEF's identity pin, p 4): P(x = 3) >= .30 with the lower bound > .15 'BEF breaks at its identity pin like END (blind step)';
     P(x = 4) >= .60 and P(x = 3) <= .15 'BEF stays at its identity pin (target-aware)'; else 'partial'.
  FW (p 3, target one ahead): P(x = 2) >= .30 with the lower bound > .15 'BEF steps back blindly'; P(x = 4) >= .30 with the lower bound > .15
     'BEF moves to its target'; P(x = 3) >= .60 'no post-lookup change'; else 'partial'.
  Summary: 'blind like END' (ID blind and FW back), 'target-aware' (ID stays and FW to target), 'no post-lookup step' (ID stays and FW none),
     else 'mixed'.  The same quantities for END5 are reported (its known behaviour: a blind one-back).
  O0b (AFT5, target 6): 'after is resolved in the early address' if P(x = p) >= .60 at p 3 and p 4; 'target-aware after' if P(x = 6) >= .30
     at p 3 and p 4; 'blind forward step' if P(x = p + 1) >= .30 at p 3 and p 4; else 'mixed'.
O1: l(BEF5) - l(BEF5 BAND_R) pooled over the three pins >= .15 with the interval above 0 'the band carries BEF's post-lookup change'.
O2: h16 in BEF: (a) answer: l(BEF5) - l(BEF5 H16_R) >= .10 with the interval above 0 and above the mean of RAND1a_R / RAND1b_R by >= .08 (interval
  above 0); (b) score: Delta_T'(H16_R) / Delta_T'(BAND_R) >= .25 with the lower bound > 0.  'h16 carries part of explicit before' if (a), 'h16
  carries part of BEF's predecessor selectivity' if (b).
O3 (one operator; the adversary's stricter rule): read only if END's own h16 effect is large enough, Delta_b(E5 H16_R) >= .15 with the lower
  bound > .05 (pooled), else 'not informative (h16's END effect too small at these pins)'.  Swaps H16_XE (BEF5 <- END5's h16) and H16_XB (END5
  <- BEF5's h16): 'interchangeable' if both swaps' 90% intervals of Delta_b lie within +-.5 Delta_b(E5 H16_R) and of Delta_T' within +-.25
  |Delta_T'(E5 H16_R)|; 'frame-specific' if either |Delta_b| >= .5 Delta_b(E5 H16_R) with its 95% interval excluding 0; else 'unresolved'.
  The band: BAND_XE / BAND_XB by the T' criterion only (the band also carries the frame's gain): 'the band's output is interchangeable' if both
  90% intervals of Delta_T' lie within +-.25 |Delta_T'(E5 BAND_R)|, 'frame-specific band outputs' if either |Delta_T'| >= .5 |Delta_T'(E5
  BAND_R)| with its interval excluding 0, else 'unresolved'.
O3b (anchor-keyed in BEF, cross-pin group 3 <-> 5): shift S = [z[p'-1] - z[p-1]](X_PX) - (recipient), late mean, pooled over both directions;
  r = S(BEF H16_PX) / S(BEF BAND_PX): BAND valid if S(BEF BAND_PX) has its interval above 0; r >= .5 with the lower bound > 0 'h16 carries an
  anchor-keyed predecessor address in BEF too'; r <= .2 with the upper bound <= .35 'anchor-independent in BEF'; else 'mixed'.  END's H16_PX shift
  is reported beside it (S860 V5 replication).
O4: s(BEF5 ANTI_Z) - s(BEF5) pooled >= .10 with the interval above 0 'the anti-self heads make explicit before leave the anchor'.
O5 (after): l(AFT5) - l(AFT5 BAND_R) >= .15 with the interval above 0 'the band carries a post-lookup change for after'; |l(AFT5 H16_R) - l(AFT5)|
  <= .05 'h16 is not used for after'; b(AFT5 H16_XE) - b(AFT5) >= .10 with the interval above 0 "END's h16 output imposes a predecessor step under
  after (frame-blind, anchor-keyed)", else 'not observed'.
O6 (END at the same pins): b(E5) - b(E5 BAND_R) >= .20 and b(E5) - b(E5 H16_R) >= .10, intervals above 0 -> 'replicated'.
Descriptive: per-pin answer distributions of every run, SELFADDR_R, the L47 h16 profiles.
Usage: python3 s864_analyze.py TAG"""
from __future__ import annotations

import json, sys
import numpy as np

TAG = sys.argv[1]
J = json.load(open(f"/vlm/results/r3/s864_{TAG}_opident.json"))
OUT = f"/vlm/results/r3/s864_analysis_{TAG}.json"
rng = np.random.default_rng(864)
NB = 4000
rows = J["rows"]; S = len(rows); N = J["N"]; PINS = J["pins"]; XP = J["xpins"]
SP = rows[0]["subpass"]
LATE = [str(l) for l in J["late"]]
E47, E43 = str(J["early"][-1]), str(J["early"][-2])


def bt(fn, q=(2.5, 97.5)):
    est = fn(np.arange(S)); bs = [fn(rng.integers(0, S, S)) for _ in range(NB)]
    lo, hi = np.nanpercentile(bs, list(q))
    return dict(est=float(est), lo=float(lo), hi=float(hi))


def bt2(fn):
    """95% and 90% intervals from the same resamples."""
    est = fn(np.arange(S)); bs = [fn(rng.integers(0, S, S)) for _ in range(NB)]
    l95, h95, l90, h90 = np.nanpercentile(bs, [2.5, 97.5, 5, 95])
    return dict(est=float(est), lo=float(l95), hi=float(h95), lo90=float(l90), hi90=float(h90))


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
        CACHE[key] = dict(x=x, b=(x == p - 1).astype(float), s=(x == p).astype(float), l=(x != p).astype(float), f=(x == p + 1).astype(float),
                          t4=(x == 4).astype(float), t6=(x == 6).astype(float), Tp=Tp, Zm=np.mean([Z[L] for L in LATE], 0))
    return CACHE[key]


def pin_of(key):
    return int(key.split("_")[0].split("p")[-1])


def lev(rk, R, pins):
    pre = [meas(f"{rk}p{p}", p)[R] for p in pins]
    return lambda ix: float(np.mean([x[ix].mean() for x in pre]))


def d(rk, arm, R, pins=None, sign=1):
    """pooled sign x (recipient - arm) over pins (sign 1: the arm removes R)."""
    pre = []
    for p in (pins or PINS):
        key = f"{rk}p{p}_{arm}"; pre.append(sign * (meas(ref_of(key), p)[R] - meas(key, p)[R]))
    return lambda ix: float(np.mean([x[ix].mean() for x in pre]))


res = dict(tag=TAG, n_scenes=S, n_runs=J["n_runs"])
# ---------------------------------------------------------------- I0
self_ok = True
for p in PINS:
    for rk in ("B5", "E5", "A5"):
        key = f"{rk}p{p}_SELF_FA"; rk0 = ref_of(key)
        for r in rows:
            if r["land"][key] != r["land"][rk0] or not all(np.array_equal(np.asarray(r["zl"][key][L]), np.asarray(r["zl"][rk0][L])) for L in r["zl"][rk0]):
                self_ok = False
reach = []
for key in [f"{rk}p{p}" for p in PINS for rk in ("E5", "B5", "A5", "R")] + [f"X{rk}p{p}" for p in XP for rk in ("B5", "E5")]:
    p = pin_of(key)
    for r in rows:
        for L in (E43, E47):
            q = np.asarray(r["qk"][key][L]); reach += [int(np.argmax(q[i])) + 1 == p for i in (2, 5, 9)]
racc = float(np.mean([r["land"][f"Rp{p}"] == p for r in rows for p in PINS]))
st_acc = float(np.mean([r["land"][f"START{p}"] == p for r in rows for p in PINS]))
I0 = dict(self_bit_identical=bool(self_ok), pin_reach=bool(np.mean(reach) >= .95), R_acc=bool(racc >= .90), t1=bool((J["t1_menu"] or 0) >= .95),
          START_acc=bool(st_acc >= .95))
res["I0"] = dict(**I0, all=bool(all(I0.values())), pin_reach_value=float(np.mean(reach)), R_acc_value=racc, t1_value=J["t1_menu"], START_acc_value=st_acc)
ok = res["I0"]["all"]
# ---------------------------------------------------------------- O0
def share(rk, p, val):
    return bt(lambda ix: float(np.mean(meas(f"{rk}p{p}", p)["x"][ix] == val)))
tab = {rk: {str(p): {str(v): float(np.mean(meas(f"{rk}p{p}", p)["x"] == v)) for v in range(1, 9)} for p in PINS} for rk in ("B5", "E5", "A5", "R")}
pc = share("B5", 5, 4)
id3, id4 = share("B5", 4, 3), share("B5", 4, 4)
fw2, fw4, fw3 = share("B5", 3, 2), share("B5", 3, 4), share("B5", 3, 3)
v_pc = "the pinned BEF form is read" if pc["est"] >= .50 else "instrument: the pinned BEF form is not read"
v_id = ("BEF breaks at its identity pin like END (blind step)" if id3["est"] >= .30 and id3["lo"] > .15 else
        "BEF stays at its identity pin (target-aware)" if id4["est"] >= .60 and id3["est"] <= .15 else "partial")
v_fw = ("BEF steps back blindly" if fw2["est"] >= .30 and fw2["lo"] > .15 else
        "BEF moves to its target" if fw4["est"] >= .30 and fw4["lo"] > .15 else
        "no post-lookup change" if fw3["est"] >= .60 else "partial")
summ = ("blind like END" if v_id.startswith("BEF breaks") and v_fw == "BEF steps back blindly" else
        "target-aware" if v_id.startswith("BEF stays") and v_fw == "BEF moves to its target" else
        "no post-lookup step" if v_id.startswith("BEF stays") and v_fw == "no post-lookup change" else "mixed")
read_bef = ok and v_pc == "the pinned BEF form is read"
a_stay = [float(np.mean(meas(f"A5p{p}", p)["s"])) for p in (3, 4)]; a_t6 = [float(np.mean(meas(f"A5p{p}", p)["t6"])) for p in (3, 4)]
a_f = [float(np.mean(meas(f"A5p{p}", p)["f"])) for p in (3, 4)]
v_aft = ("after is resolved in the early address" if min(a_stay) >= .60 else "target-aware after" if min(a_t6) >= .30 else
         "blind forward step" if min(a_f) >= .30 else "mixed")
res["O0"] = dict(table=tab, PC_P4_at_p5=pc, ID_P3_at_p4=id3, ID_P4_at_p4=id4, FW_P2_at_p3=fw2, FW_P4_at_p3=fw4, FW_P3_at_p3=fw3,
                 END=dict(P3_at_p4=share("E5", 4, 3), P2_at_p3=share("E5", 3, 2), P4_at_p5=share("E5", 5, 4)),
                 verdict_PC=(v_pc if ok else "not read"), verdict_ID=(v_id if read_bef else "not read"), verdict_FW=(v_fw if read_bef else "not read"),
                 verdict=(summ if read_bef else "not read"),
                 AFT=dict(stay_p3_p4=a_stay, target6_p3_p4=a_t6, forward_p3_p4=a_f, verdict=(v_aft if ok else "not read")))
# ---------------------------------------------------------------- O1, O2
o1 = bt(d("B5", "BAND_R", "l"))
res["O1"] = dict(Delta_leave=o1, by_pin={str(p): d("B5", "BAND_R", "l", [p])(np.arange(S)) for p in PINS},
                 verdict=(("the band carries BEF's post-lookup change" if o1["est"] >= .15 and o1["lo"] > 0 else "not") if read_bef else "not read"))
o2a = bt(d("B5", "H16_R", "l")); o2r = bt(lambda ix: d("B5", "H16_R", "l")(ix) - .5 * (d("B5", "RAND1a_R", "l")(ix) + d("B5", "RAND1b_R", "l")(ix)))
o2b = bt(lambda ix: d("B5", "H16_R", "Tp")(ix) / (d("B5", "BAND_R", "Tp")(ix) + 1e-12))
res["O2"] = dict(Delta_leave=o2a, vs_random=o2r, score_ratio=o2b, Delta_Tp_H16=bt(d("B5", "H16_R", "Tp")), Delta_Tp_BAND=bt(d("B5", "BAND_R", "Tp")),
                 RAND=dict(a=bt(d("B5", "RAND1a_R", "l")), b=bt(d("B5", "RAND1b_R", "l"))),
                 verdict_answer=(("h16 carries part of explicit before" if o2a["est"] >= .10 and o2a["lo"] > 0 and o2r["est"] >= .08 and o2r["lo"] > 0
                                  else "not") if read_bef else "not read"),
                 verdict_score=(("h16 carries part of BEF's predecessor selectivity" if o2b["est"] >= .25 and o2b["lo"] > 0 else "not")
                                if read_bef else "not read"))
# ---------------------------------------------------------------- O3
e_h = bt(d("E5", "H16_R", "b")); e_hT = bt(d("E5", "H16_R", "Tp")); e_bT = bt(d("E5", "BAND_R", "Tp"))
sx = {nm: bt2(d(rk, arm, R)) for nm, rk, arm, R in (("H16_XE_b", "B5", "H16_XE", "b"), ("H16_XB_b", "E5", "H16_XB", "b"),
                                                       ("H16_XE_Tp", "B5", "H16_XE", "Tp"), ("H16_XB_Tp", "E5", "H16_XB", "Tp"),
                                                       ("BAND_XE_Tp", "B5", "BAND_XE", "Tp"), ("BAND_XB_Tp", "E5", "BAND_XB", "Tp"))}
if not read_bef:
    v3 = v3b = "not read"
elif not (e_h["est"] >= .15 and e_h["lo"] > .05):
    v3 = "not informative (h16's END effect too small at these pins)"
else:
    mb, mt = .5 * e_h["est"], .25 * abs(e_hT["est"])
    inside = all(-mb <= sx[k]["lo90"] and sx[k]["hi90"] <= mb for k in ("H16_XE_b", "H16_XB_b")) and \
             all(-mt <= sx[k]["lo90"] and sx[k]["hi90"] <= mt for k in ("H16_XE_Tp", "H16_XB_Tp"))
    spec = any(abs(sx[k]["est"]) >= mb and (sx[k]["lo"] > 0 or sx[k]["hi"] < 0) for k in ("H16_XE_b", "H16_XB_b"))
    v3 = "frame-specific h16 outputs" if spec else "h16's output is interchangeable between END and BEF (one predecessor operator)" if inside else "unresolved"
if read_bef:
    mt2 = .25 * abs(e_bT["est"])
    inside_b = all(-mt2 <= sx[k]["lo90"] and sx[k]["hi90"] <= mt2 for k in ("BAND_XE_Tp", "BAND_XB_Tp"))
    spec_b = any(abs(sx[k]["est"]) >= .5 * abs(e_bT["est"]) and (sx[k]["lo"] > 0 or sx[k]["hi"] < 0) for k in ("BAND_XE_Tp", "BAND_XB_Tp"))
    v3b = "frame-specific band outputs" if spec_b else "the band's output is interchangeable" if inside_b else "unresolved"
res["O3"] = dict(E5_H16_R_b=e_h, E5_H16_R_Tp=e_hT, E5_BAND_R_Tp=e_bT, swaps=sx, verdict=v3, verdict_band=v3b)
# ---------------------------------------------------------------- O3b (cross-pin)
p_, q_ = XP
def xshift(fm, X):
    pre = []
    for p, pp in ((p_, q_), (q_, p_)):
        key = f"X{fm}p{p}_{X}_PX{pp}"; rk = ref_of(key)
        za, ze = meas(key, p)["Zm"], meas(rk, p)["Zm"]
        pre.append((za[:, pp - 2] - za[:, p - 2]) - (ze[:, pp - 2] - ze[:, p - 2]))
    return lambda ix: float(np.mean([x[ix].mean() for x in pre]))
sb = bt(xshift("B5", "BAND")); sh = bt(xshift("B5", "H16")); se = bt(xshift("E5", "H16"))
rr = bt(lambda ix: xshift("B5", "H16")(ix) / (xshift("B5", "BAND")(ix) + 1e-12))
v3x = ("h16 carries an anchor-keyed predecessor address in BEF too" if rr["est"] >= .5 and rr["lo"] > 0 else
       "anchor-independent in BEF" if rr["est"] <= .2 and rr["hi"] <= .35 else "mixed")
res["O3b"] = dict(S_BAND_BEF=sb, S_H16_BEF=sh, S_H16_END=se, ratio=rr, band_valid=bool(sb["lo"] > 0),
                  verdict=((v3x if sb["lo"] > 0 else "transfer test invalid") if ok else "not read"))
# ---------------------------------------------------------------- O4, O5, O6
o4 = bt(d("B5", "ANTI_Z", "s", sign=-1))
res["O4"] = dict(Delta_stay=o4, by_pin={str(p): d("B5", "ANTI_Z", "s", [p], sign=-1)(np.arange(S)) for p in PINS},
                 verdict=(("the anti-self heads make explicit before leave the anchor" if o4["est"] >= .10 and o4["lo"] > 0 else "not") if ok else "not read"))
o5a = bt(d("A5", "BAND_R", "l")); o5b = bt(d("A5", "H16_R", "l")); o5c = bt(d("A5", "H16_XE", "b", sign=-1))
res["O5"] = dict(Delta_leave_BAND=o5a, Delta_leave_H16=o5b, Delta_b_H16_XE=o5c,
                 verdict_band=(("the band carries a post-lookup change for after" if o5a["est"] >= .15 and o5a["lo"] > 0 else "not") if ok else "not read"),
                 verdict_h16=(("h16 is not used for after" if abs(o5b["est"]) <= .05 else "not") if ok else "not read"),
                 verdict_xe=(("END's h16 output imposes a predecessor step under after (frame-blind, anchor-keyed)" if o5c["est"] >= .10 and o5c["lo"] > 0
                              else "not observed") if ok else "not read"))
o6a = bt(d("E5", "BAND_R", "b")); o6b = e_h
res["O6"] = dict(Delta_b_BAND=o6a, Delta_b_H16=o6b, by_pin={str(p): dict(BAND=d("E5", "BAND_R", "b", [p])(np.arange(S)), H16=d("E5", "H16_R", "b", [p])(np.arange(S)))
                                                           for p in PINS},
                 verdict=(("replicated" if o6a["est"] >= .20 and o6a["lo"] > 0 and o6b["est"] >= .10 and o6b["lo"] > 0 else "not") if ok else "not read"))
# ---------------------------------------------------------------- descriptive
desc = {}
for key in rows[0]["land"]:
    if key.startswith("PAD") or "@" in key or key.startswith("START"):
        continue
    x = np.array([r["land"][key] for r in rows]); desc[key] = {str(v): float(np.mean(x == v)) for v in range(1, 10) if np.any(x == v)}
res["descriptive"] = dict(answers=desc, SELFADDR_R_leave=bt(d("B5", "SELFADDR_R", "l")), SELFADDR_R_stay=bt(d("B5", "SELFADDR_R", "s", sign=-1)))
def prof(key):
    vals = [np.asarray(r["z47"][key], dtype=float) for r in rows if r.get("z47", {}).get(key) is not None]
    if not vals:
        return None
    z = np.stack(vals)[:, :, J["h16"], :]
    sm = np.exp(z - z.max(-1, keepdims=True)); sm /= sm.sum(-1, keepdims=True)
    return sm.mean((0, 1, 2)).tolist()
res["descriptive"]["h16_L47_profile"] = {f"{rk}p{p}": prof(f"{rk}p{p}") for p in PINS for rk in ("E5", "B5", "A5", "R")}
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"]["all"], O0=[res["O0"]["verdict_PC"], res["O0"]["verdict_ID"], res["O0"]["verdict_FW"], res["O0"]["verdict"]],
                      O0b=res["O0"]["AFT"]["verdict"], O1=res["O1"]["verdict"], O2=[res["O2"]["verdict_answer"], res["O2"]["verdict_score"]],
                      O3=[res["O3"]["verdict"], res["O3"]["verdict_band"]], O3b=res["O3b"]["verdict"], O4=res["O4"]["verdict"],
                      O5=[res["O5"]["verdict_band"], res["O5"]["verdict_h16"], res["O5"]["verdict_xe"]], O6=res["O6"]["verdict"]), indent=1))
print("->", OUT)
