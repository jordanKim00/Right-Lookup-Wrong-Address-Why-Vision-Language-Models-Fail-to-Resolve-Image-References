"""S861 analysis (rules fixed before the data; knowledge/DECISIONS.md S861 entry).  Input results/r3/s861_<tag>_verify.json.  Three parts, judged
separately (user review 10-20).  Every arm is compared with its recipient run of the same sub-pass; effects are scene means, scene bootstrap 4,000.
Targets: N7 END4 -> 4, N9 END5 -> 5 (identity), N8 END5 -> 4, N8 END3 -> 6, N8 END2 -> 7, START4 -> 4, START5 -> 5, BEF5 -> 4, AFT5 -> 6.
I0: every self arm (natural SELF, pinned SELF_FA) bit-identical to its recipient (answers and late scores); t1 >= .95; natural START(4), START(5)
  accuracy >= .95; part C pins reach (head-mean block-score argmax at L43 / L47 = p, rows B1 / B4 / answer) >= .95.
A (natural):
  A1 identity repair, X in {BAND_S, ANTI_Z, SELFADDR_S}: accuracy change (arm - E) pooled over N7 END4 and N9 END5 >= +.15 with the interval above 0.
  A2 the beneficial step removed: accuracy change in N8 END5 <= -.15 with the interval below 0.
  A3 natural double dissociation for X: A1 and A2 both hold.
  A4 h16 direction: H16_S in N8 END5 - accuracy change <= -.10 with the interval below 0 'h16 carries the step's direction in natural runs'.
  A5 near END: N8 END2 accuracy change within +-.10 with the interval inside +-.20 for each of BAND_S, ANTI_Z, SELFADDR_S - 'the near-END address
     is formed before the band'; else 'the band acts at END(2)'.
  A6 END(3): BAND_S - share at 6 falls by >= .15 and share at 7 rises by >= .10 (intervals excluding 0) 'the band adds the last step from the end
     anchor'; else 'not'.
  A7 START protection: each of ANTI_Z, H16_Z, SELFADDR_Z on START4 and START5 - |accuracy change| <= .05 with the interval inside +-.10.
  Descriptive: natural early anchor (L47 head-mean argmax, majority over B1 / B4 / answer) per cell; random arms; ALL47_S; ANTI43_Z.
B (operator identity, N8):
  B0 valid if BEF5's natural early anchor equals END5's in >= .70 of scenes.
  B1 H16_S5 in BEF5: accuracy change <= -.10 with the interval below 0 'h16 takes part in explicit before'.
  B2 BAND_S5 in BEF5: accuracy change <= -.20 with the interval below 0 'the writer carries explicit before'.
  B3 (read with B1) H16_X in BEF5 and in END5: |accuracy change| <= .05 with the interval inside +-.10 in both 'the two predecessor signals are
     exchangeable'; if B1 holds and either H16_X moves accuracy by >= .10 'not exchangeable'.
  B4 H16_S5 in AFT5: |accuracy change| <= .05 with the interval inside +-.10 'h16 is not used for after'.
C (pinned N8, E5 at p in {4, 5} with E3's gate or read): for X in {SELFADDR, H16, ANTI}, one-back share change E5 - arm (pooled p 4, 5):
  'k acts through X's gate' if GATE >= .05 with the interval above 0; 'through X's read' likewise for READ; both / neither.  Late T and U changes
  reported.
Usage: python3 s861_analyze.py TAG"""
from __future__ import annotations

import json, sys
import numpy as np

TAG = sys.argv[1]
J = json.load(open(f"/vlm/results/r3/s861_{TAG}_verify.json"))
OUT = f"/vlm/results/r3/s861_analysis_{TAG}.json"
rng = np.random.default_rng(861)
NB = 4000
rows = J["rows"]; S = len(rows)
SP = rows[0]["subpass"]
TGT = {"N7_END4": 4, "N9_END5": 5, "N8_END5": 4, "N8_END3": 6, "N8_END2": 7, "N8_START4": 4, "N8_START5": 5, "N8_BEF5": 4, "N8_AFT5": 6,
       "C8_E5p4": 4, "C8_E5p5": 4}
L0 = str(J["late"][0]); LM = J["late_mean"][L0]


def ref_of(armkey):
    tagp, recp = armkey.split("_")[0], armkey.split("_")[1]
    return f"{tagp}_{recp}@{SP[armkey]}"


def land(key):
    return np.array([r["land"][key] for r in rows])


def bt(fn):
    est = fn(np.arange(S)); bs = [fn(rng.integers(0, S, S)) for _ in range(NB)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(est=float(est), lo=float(lo), hi=float(hi))


def dshare(armkey, pos):
    """change of the share of answers at pos (arm - its recipient of the same sub-pass)."""
    a, e = land(armkey), land(ref_of(armkey))
    return lambda ix: float(np.mean(a[ix] == pos) - np.mean(e[ix] == pos))


def dacc(armkey):
    base = "_".join(armkey.split("_")[:2])
    return dshare(armkey, TGT[base])


def zmean(key):
    return np.array([np.asarray(r["zl"][key][L0], dtype=float)[:, LM].mean(1).mean(0) for r in rows])     # [S, N]


def early_anchor(key):
    out = []
    for r in rows:
        q = np.asarray(r["qk"][key][str(J["early"][-1])])
        am = [int(np.argmax(q[i])) + 1 for i in (2, 5, 9)]
        out.append(max(set(am), key=am.count))
    return np.array(out)


res = dict(tag=TAG, n_scenes=S)
# ---------------------------------------------------------------- I0
self_ok = True
for key in [k for k in rows[0]["land"] if k.endswith("_SELF") or k.endswith("_SELF_FA")]:
    rk = ref_of(key)
    for r in rows:
        if r["land"][key] != r["land"][rk] or not all(np.array_equal(np.asarray(r["zl"][key][L]), np.asarray(r["zl"][rk][L])) for L in r["zl"][rk]):
            self_ok = False
st_acc = float(np.mean([np.mean(land("N8_START4") == 4), np.mean(land("N8_START5") == 5)]))
reach = []
for key, p in (("C8_E5p4", 4), ("C8_E5p5", 5), ("C8_E3p4", 4), ("C8_E3p5", 5), ("C8_Rp4", 4), ("C8_Rp5", 5)):
    for r in rows:
        for L in (str(J["early"][-2]), str(J["early"][-1])):
            q = np.asarray(r["qk"][key][L]); reach += [int(np.argmax(q[i])) + 1 == p for i in (2, 5, 9)]
I0 = dict(self_bit_identical=bool(self_ok), t1=bool((J["t1_menu"] or 0) >= .95), START_acc=bool(st_acc >= .95), pin_reach=bool(np.mean(reach) >= .95))
res["I0"] = dict(**I0, all=bool(all(I0.values())), t1_value=J["t1_menu"], START_acc_value=st_acc, pin_reach_value=float(np.mean(reach)))
# ---------------------------------------------------------------- A
ID = ["N7_END4", "N9_END5"]
res["A"] = {"natural_accuracy": {k: float(np.mean(land(k) == TGT[k])) for k in ("N7_END4", "N9_END5", "N8_END5", "N8_END3", "N8_END2", "N8_START4", "N8_START5", "N8_BEF5", "N8_AFT5")},
            "early_anchor": {k: {str(v): float(np.mean(early_anchor(k) == v)) for v in sorted(set(early_anchor(k)))} for k in ("N7_END4", "N9_END5", "N8_END5", "N8_END3", "N8_END2", "N8_BEF5", "N8_AFT5")}}
for X in ("BAND_S", "ANTI_Z", "SELFADDR_S"):
    a1 = bt(lambda ix, X=X: float(np.mean([dacc(f"{c}_{X}")(ix) for c in ID])))
    a2 = bt(dacc(f"N8_END5_{X}"))
    v1 = a1["est"] >= .15 and a1["lo"] > 0; v2 = a2["est"] <= -.15 and a2["hi"] < 0
    res["A"][X] = dict(identity_change=a1, helped_change=a2, A1=("repairs identity" if v1 else "not"), A2=("removes the beneficial step" if v2 else "not"),
                       A3=("natural double dissociation" if v1 and v2 else "not"))
a4 = bt(dacc("N8_END5_H16_S"))
res["A"]["A4"] = dict(change=a4, identity_change=bt(lambda ix: float(np.mean([dacc(f"{c}_H16_S")(ix) for c in ID]))),
                      verdict=("h16 carries the step's direction in natural runs" if a4["est"] <= -.10 and a4["hi"] < 0 else "not"))
a5 = {X: bt(dacc(f"N8_END2_{X}")) for X in ("BAND_S", "ANTI_Z", "SELFADDR_S")}
ok5 = all(abs(v["est"]) <= .10 and v["lo"] >= -.20 and v["hi"] <= .20 for v in a5.values())
res["A"]["A5"] = dict(changes=a5, verdict=("the near-END address is formed before the band" if ok5 else "the band acts at END(2)"))
s6, s7 = bt(dshare("N8_END3_BAND_S", 6)), bt(dshare("N8_END3_BAND_S", 7))
res["A"]["A6"] = dict(share6=s6, share7=s7, verdict=("the band adds the last step from the end anchor" if s6["est"] <= -.15 and s6["hi"] < 0 and s7["est"] >= .10 and s7["lo"] > 0 else "not"))
a7 = {}
for st_ in ("N8_START4", "N8_START5"):
    for X in ("ANTI_Z", "H16_Z", "SELFADDR_Z"):
        v = bt(dacc(f"{st_}_{X}"))
        a7[f"{st_}_{X}"] = dict(change=v, verdict=("START unaffected" if abs(v["est"]) <= .05 and v["lo"] >= -.10 and v["hi"] <= .10 else "START affected"))
res["A"]["A7"] = a7
res["A"]["descriptive"] = {f"{c}_{X}": bt(dacc(f"{c}_{X}")) for c in ("N7_END4", "N9_END5", "N8_END5", "N8_END3", "N8_END2")
                           for X in ("ALL47_S", "ANTI43_Z", "H16_S", "RAND5a_S", "RAND5b_S", "RAND1a_S", "RAND1b_S")}
# ---------------------------------------------------------------- B
same = float(np.mean(early_anchor("N8_BEF5") == early_anchor("N8_END5")))
b1, b2 = bt(dacc("N8_BEF5_H16_S5")), bt(dacc("N8_BEF5_BAND_S5"))
bx1, bx2 = bt(dacc("N8_BEF5_H16_X")), bt(dacc("N8_END5_H16_X"))
b4 = bt(dacc("N8_AFT5_H16_S5"))
B1 = b1["est"] <= -.10 and b1["hi"] < 0
exch = all(abs(v["est"]) <= .05 and v["lo"] >= -.10 and v["hi"] <= .10 for v in (bx1, bx2))
res["B"] = dict(anchor_same=same, valid=bool(same >= .70), H16_S5=b1, BAND_S5=b2, H16_X_BEF=bx1, H16_X_END=bx2, H16_S5_AFT=b4,
                B1=("h16 takes part in explicit before" if B1 else "not"),
                B2=("the writer carries explicit before" if b2["est"] <= -.20 and b2["hi"] < 0 else "not"),
                B3=(("the two predecessor signals are exchangeable" if exch else "not exchangeable" if max(abs(bx1["est"]), abs(bx2["est"])) >= .10 else "undecided")
                    if B1 else "not read (B1)"),
                B4=("h16 is not used for after" if abs(b4["est"]) <= .05 and b4["lo"] >= -.10 and b4["hi"] <= .10 else "h16 affects after"),
                descriptive={k: bt(dacc(f"N8_BEF5_{k}")) for k in ("ALL47_S5", "SELFADDR_S5", "ANTI_Z", "RAND1a_S5")} | {"AFT5_BAND_S5": bt(dacc("N8_AFT5_BAND_S5"))})
# ---------------------------------------------------------------- C
def back_change(armsuffix):
    def f(ix):
        vals = []
        for p in (4, 5):
            a, e = land(f"C8_E5p{p}_{armsuffix}"), land(ref_of(f"C8_E5p{p}_{armsuffix}"))
            vals.append(float(np.mean(e[ix] == p - 1) - np.mean(a[ix] == p - 1)))
        return float(np.mean(vals))
    return f
def tchange(armsuffix):
    out = {}
    for p in (4, 5):
        za, ze = zmean(f"C8_E5p{p}_{armsuffix}"), zmean(ref_of(f"C8_E5p{p}_{armsuffix}"))
        T = lambda z: z[:, p - 2] - z[:, p]; U = lambda z: (z[:, p - 2] + z[:, p]) / 2 - z[:, p - 1]
        out[f"p{p}"] = dict(dT=float(np.mean(T(ze) - T(za))), dU=float(np.mean(U(ze) - U(za))))
    return out
res["C"] = {}
for X in ("SELFADDR", "H16", "ANTI"):
    g, rd = bt(back_change(f"{X}_G")), bt(back_change(f"{X}_RD"))
    vg = g["est"] >= .05 and g["lo"] > 0; vr = rd["est"] >= .05 and rd["lo"] > 0
    res["C"][X] = dict(gate=g, read=rd, late_gate=tchange(f"{X}_G"), late_read=tchange(f"{X}_RD"),
                       verdict=("both" if vg and vr else f"k acts through {X}'s gate" if vg else f"k acts through {X}'s read" if vr else "neither"))
res["C"]["E3_one_back"] = {f"p{p}": float(np.mean(land(f"C8_E3p{p}") == p - 1)) for p in (4, 5)}
res["C"]["E5_one_back"] = {f"p{p}": float(np.mean(land(f"C8_E5p{p}") == p - 1)) for p in (4, 5)}
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"]["all"], A={X: [res["A"][X]["A1"], res["A"][X]["A2"], res["A"][X]["A3"]] for X in ("BAND_S", "ANTI_Z", "SELFADDR_S")},
                      A4=res["A"]["A4"]["verdict"], A5=res["A"]["A5"]["verdict"], A6=res["A"]["A6"]["verdict"],
                      A7={k: v["verdict"] for k, v in res["A"]["A7"].items()}, B=[res["B"]["valid"], res["B"]["B1"], res["B"]["B2"], res["B"]["B3"], res["B"]["B4"]],
                      C={X: res["C"][X]["verdict"] for X in ("SELFADDR", "H16", "ANTI")}), indent=1))
print("->", OUT)
