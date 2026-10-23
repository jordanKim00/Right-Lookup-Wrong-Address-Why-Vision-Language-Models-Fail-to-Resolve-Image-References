"""S863 analysis (rules fixed before the data; knowledge/DECISIONS.md S863 entry; revised with the design after the agents' critiques, before
any data).  Input results/r3/s863_<tag>_klocus.json.
Per run with pin p (N = 8, p in {4, 5}): one-back b = [x = p - 1]; leave = [x != p]; late profiles z_L = head-mean (late_mean) block scores at
late layer L, rows B1 / B4 / answer averaged; T' = z[p-1] - z[p] averaged over the four late layers (co-primary: immune to the end-route mass
at p + 1); T = z[p-1] - z[p+1] and U = (z[p-1] + z[p+1]) / 2 - z[p] at L51 (descriptive).  Arms are compared with the E of their own
sub-pass, scene by scene, pooled over the two pins (each pin also reported); scene bootstrap 4,000 (seed 863).
  Delta_off(X) = R(E5) - R(E5 <- X of E2)   (positive: the k-2 version of X switches the step off)
  Delta_on(X)  = R(E2 <- X of E5) - R(E2)   (positive: the k-5 version of X switches the step on),  R = b or T' (co-primary).
I0: SELF_FA, SELF_TAIL and SELF_IN46 bit-identical to their sub-pass E5 (answers and late scores); pin reach (E refs, L43 / L47 argmax at rows B1 / B4 / answer = p)
  >= .95; t1 >= .95; natural START(4) / START(5) accuracy >= .95.  Nothing is read if I0 fails.
K0a: b(E5) - b(E2) >= .20 with the interval above 0 'the step is switched by k at a fixed anchor'; else no k-locus rule is read.
K0b (where k acts, coarse; per R): phi_dir(Y) = Delta_dir(Y) / [R(E5) - R(E2)] for Y in {BAND_TAIL, BAND, L50RES, IN46}.  Verdict on
  BAND_TAIL: both directions >= .6 with lower bounds > .3 'the k-switch is carried by the band's outputs'; both lower bounds > 0 'the band
  carries part of the k-switch, the rest is read by the late layers directly'; else 'the k-switch acts mostly after the band (the late layers
  read k directly)'.  L50RES and IN46 are reported with the same three labels ('the read rows' state before the late lookup' / 'the read rows'
  state entering the band').
Shares (row-level components, normalised by BAND; f_off read when Delta_off(BAND) has its interval above 0, f_on when Delta_on(BAND) >= .15
  (b) or > 0 (T') with the interval above 0): f_dir(X) = Delta_dir(X) / Delta_dir(BAND) for X in {FA47, MLPS, DNS} (K1), {H16, SELFADDR,
  ANTI, REST47} (K2), {MLP47, MLP49, DN49} (K3), {FA43, ANTI43} (K4, before the band).  Labels: 'X carries the k-switch' (f >= .5 with the
  lower bound > 0 in both directions); 'X carries part' (both lower bounds > 0); 'reversed' (both upper bounds < 0); 'one direction'
  (exactly one lower bound > 0 - registered as not a positive finding); else 'not observed'.  With one direction unreadable, the label is
  given for the readable one with '(off only)' / '(on only)'.
  K3b PATH47: f(PATH47) >= .5 f(MLP47) with its lower bound > 0 in both directions 'MLP47's k-dependence comes through FA47', else 'not'.
  K5 complements: f(BAND-X) for X in {FA47, MLP47, MLP49}: <= .7 with the upper bound < 1 in both directions 'X is necessary in context';
  else 'not necessary'.  K5b (descriptive): sum of single shares (FA47, MLP47, MLP49, DN49 and the rest via MLPS / DNS) vs 1.
K6: b(E3) - b(E3 <- BAND of E2) >= .10 with the interval above 0 'k3's step is switched on in the band as well'.
K7: b(E2 <- BAND of E3) - b(E2) >= .10 with the interval above 0 'k3's band switches the step on'; H16_K3ON descriptive.
Descriptive: per-pin values; U, T, leave; h16 / self-address / anti-self block profiles at L47 (softmax of the per-head block scores).
Usage: python3 s863_analyze.py TAG"""
from __future__ import annotations

import json, sys
import numpy as np

TAG = sys.argv[1]
J = json.load(open(f"/vlm/results/r3/s863_{TAG}_klocus.json"))
OUT = f"/vlm/results/r3/s863_analysis_{TAG}.json"
rng = np.random.default_rng(863)
NB = 4000
rows = J["rows"]; S = len(rows); N = J["N"]; PINS = J["pins"]
SP = rows[0]["subpass"]
LATE = [str(l) for l in J["late"]]; L0 = LATE[0]
E47, E43 = str(J["early"][-1]), str(J["early"][-2])
XS = J["xs"]


def bt(fn):
    est = fn(np.arange(S)); bs = [fn(rng.integers(0, S, S)) for _ in range(NB)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return dict(est=float(est), lo=float(lo), hi=float(hi))


def ref_of(armkey):
    recp = armkey.split("_")[0]
    return f"{recp}@{SP[armkey]}"


def zl(r, key, L):
    return np.asarray(r["zl"][key][L], dtype=float)[:, J["late_mean"][L]].mean(1).mean(0)


CACHE = {}
def meas(key, p):
    if key not in CACHE:
        x = np.array([r["land"][key] for r in rows])
        zs = {L: np.array([zl(r, key, L) for r in rows]) for L in LATE}
        z0 = zs[L0]
        Tp = np.mean([zs[L][:, p - 2] - zs[L][:, p - 1] for L in LATE], 0)
        CACHE[key] = dict(b=(x == p - 1).astype(float), leave=(x != p).astype(float), Tp=Tp, T=z0[:, p - 2] - z0[:, p],
                          U=(z0[:, p - 2] + z0[:, p]) / 2 - z0[:, p - 1])
    return CACHE[key]


def pre_delta(X, d, R, pins=None):
    out = []
    for p in (pins or PINS):
        if d == "off":
            key = f"E5p{p}_{X}_OFF"; out.append(meas(ref_of(key), p)[R] - meas(key, p)[R])
        else:
            key = f"E2p{p}_{X}_ON"; out.append(meas(key, p)[R] - meas(ref_of(key), p)[R])
    return out


def delta(X, d, R, pins=None):
    pre = pre_delta(X, d, R, pins)
    return lambda ix: float(np.mean([x[ix].mean() for x in pre]))


def kgap(R, pins=None):
    pre = [meas(f"E5p{p}", p)[R] - meas(f"E2p{p}", p)[R] for p in (pins or PINS)]
    return lambda ix: float(np.mean([x[ix].mean() for x in pre]))


res = dict(tag=TAG, n_scenes=S, n_runs=J["n_runs"])
# ---------------------------------------------------------------- I0
self_ok = True
for p in PINS:
  for key in (f"E5p{p}_SELF_FA", f"E5p{p}_SELF_TAIL", f"E5p{p}_SELF_IN46"):
    rk = ref_of(key)
    for r in rows:
        if r["land"][key] != r["land"][rk] or not all(np.array_equal(np.asarray(r["zl"][key][L]), np.asarray(r["zl"][rk][L])) for L in r["zl"][rk]):
            self_ok = False
reach = []
for p in PINS:
    for k in J["ks"]:
        for r in rows:
            for L in (E43, E47):
                q = np.asarray(r["qk"][f"E{k}p{p}"][L]); reach += [int(np.argmax(q[i])) + 1 == p for i in (2, 5, 9)]
st_acc = float(np.mean([r["land"][f"START{p}"] == p for r in rows for p in PINS]))
I0 = dict(self_bit_identical=bool(self_ok), pin_reach=bool(np.mean(reach) >= .95), t1=bool((J["t1_menu"] or 0) >= .95), START_acc=bool(st_acc >= .95))
res["I0"] = dict(**I0, all=bool(all(I0.values())), pin_reach_value=float(np.mean(reach)), t1_value=J["t1_menu"], START_acc_value=st_acc)
# ---------------------------------------------------------------- K0
k0a = bt(kgap("b"))
res["K0a"] = dict(gap_b=k0a, gap_Tp=bt(kgap("Tp")), gap_U=bt(kgap("U")),
                  E={f"k{k}": {str(p): dict(b=float(meas(f"E{k}p{p}", p)["b"].mean()), leave=float(meas(f"E{k}p{p}", p)["leave"].mean()))
                               for p in PINS} for k in J["ks"]},
                  verdict=("the step is switched by k at a fixed anchor" if k0a["est"] >= .20 and k0a["lo"] > 0 else "not"))
read_ok = res["I0"]["all"] and res["K0a"]["verdict"] != "not"
LAB = {"BAND_TAIL": ("the k-switch is carried by the band's outputs", "the band carries part of the k-switch, the rest is read by the late layers directly",
                     "the k-switch acts mostly after the band (the late layers read k directly)"),
       "BAND": ("the band's read-row outputs carry the k-switch", "the band's read-row outputs carry part", "mostly elsewhere"),
       "L50RES": ("the read rows' state before the late lookup carries the k-switch", "it carries part", "mostly elsewhere"),
       "IN46": ("the read rows' state entering the band carries the k-switch", "it carries part", "mostly elsewhere")}
res["K0b"] = {}
for R in ("b", "Tp"):
    res["K0b"][R] = {}
    for Y in ("BAND_TAIL", "BAND", "L50RES", "IN46"):
        po = bt(lambda ix, Y=Y, R=R: delta(Y, "off", R)(ix) / (kgap(R)(ix) + 1e-12))
        pn = bt(lambda ix, Y=Y, R=R: delta(Y, "on", R)(ix) / (kgap(R)(ix) + 1e-12))
        if po["est"] >= .6 and pn["est"] >= .6 and po["lo"] > .3 and pn["lo"] > .3:
            v = LAB[Y][0]
        elif po["lo"] > 0 and pn["lo"] > 0:
            v = LAB[Y][1]
        else:
            v = LAB[Y][2]
        res["K0b"][R][Y] = dict(phi_off=po, phi_on=pn, D_off=bt(delta(Y, "off", R)), D_on=bt(delta(Y, "on", R)),
                                by_pin={str(p): dict(off=delta(Y, "off", R, [p])(np.arange(S)), on=delta(Y, "on", R, [p])(np.arange(S)),
                                                     gap=kgap(R, [p])(np.arange(S))) for p in PINS},
                                verdict=(v if read_ok else "not read"))
# ---------------------------------------------------------------- shares
def share(X, d, R):
    return bt(lambda ix: delta(X, d, R)(ix) / (delta("BAND", d, R)(ix) + 1e-12))


def label(fo, fn, ro, rn):
    if ro and rn:
        if fo["est"] >= .5 and fn["est"] >= .5 and fo["lo"] > 0 and fn["lo"] > 0:
            return "X carries the k-switch"
        if fo["lo"] > 0 and fn["lo"] > 0:
            return "X carries part"
        if fo["hi"] < 0 and fn["hi"] < 0:
            return "reversed"
        if (fo["lo"] > 0) != (fn["lo"] > 0):
            return "one direction"
        return "not observed"
    f, tag = (fo, "(off only)") if ro else (fn, "(on only)")
    if not (ro or rn):
        return "not read (the band's own effect is not above 0)"
    if f["est"] >= .5 and f["lo"] > 0:
        return f"X carries the k-switch {tag}"
    if f["lo"] > 0:
        return f"X carries part {tag}"
    if f["hi"] < 0:
        return f"reversed {tag}"
    return f"not observed {tag}"


for R in ("b", "Tp"):
    dbo, dbn = bt(delta("BAND", "off", R)), bt(delta("BAND", "on", R))
    ro = bool(dbo["lo"] > 0); rn = bool(dbn["lo"] > 0 and (dbn["est"] >= .15 if R == "b" else True))
    out = dict(read_off=ro, read_on=rn)
    for grp, xs in (("K1", ["FA47", "MLPS", "DNS"]), ("K2", ["H16", "SELFADDR", "ANTI", "REST47"]), ("K3", ["MLP47", "MLP49", "DN49"]),
                    ("K4", ["FA43", "ANTI43"])):
        out[grp] = {}
        for X in xs:
            fo, fn = share(X, "off", R), share(X, "on", R)
            out[grp][X] = dict(f_off=fo, f_on=fn, D_off=bt(delta(X, "off", R)), D_on=bt(delta(X, "on", R)),
                               by_pin={str(p): dict(off=delta(X, "off", R, [p])(np.arange(S)), on=delta(X, "on", R, [p])(np.arange(S))) for p in PINS},
                               verdict=(label(fo, fn, ro, rn).replace("X", X) if read_ok else "not read"))
    po, pn = share("PATH47", "off", R), share("PATH47", "on", R)
    mo, mn = out["K3"]["MLP47"]["f_off"], out["K3"]["MLP47"]["f_on"]
    ok3b = po["est"] >= .5 * mo["est"] and pn["est"] >= .5 * mn["est"] and po["lo"] > 0 and pn["lo"] > 0
    out["K3b"] = dict(f_off=po, f_on=pn, verdict=(("MLP47's k-dependence comes through FA47" if ok3b else "not") if (read_ok and ro and rn) else "not read"))
    out["K5"] = {}
    for X in ("FA47", "MLP47", "MLP49"):
        co, cn = share(f"BAND-{X}", "off", R), share(f"BAND-{X}", "on", R)
        nec = co["est"] <= .7 and cn["est"] <= .7 and co["hi"] < 1 and cn["hi"] < 1
        out["K5"][X] = dict(f_off=co, f_on=cn, verdict=(("X is necessary in context".replace("X", X) if nec else "not necessary") if (read_ok and ro and rn) else "not read"))
    singles = ["FA47", "MLPS", "DNS"]
    out["K5b"] = {d: float(sum(delta(X, d, R)(np.arange(S)) for X in singles) / (delta("BAND", d, R)(np.arange(S)) + 1e-12)) for d in ("off", "on")}
    res[f"shares_{R}"] = out
res["descriptive"] = {"U": {X: dict(D_off=float(delta(X, "off", "U")(np.arange(S))), D_on=float(delta(X, "on", "U")(np.arange(S)))) for X in XS},
                      "T": {X: dict(D_off=float(delta(X, "off", "T")(np.arange(S))), D_on=float(delta(X, "on", "T")(np.arange(S)))) for X in XS},
                      "leave": {X: dict(D_off=float(delta(X, "off", "leave")(np.arange(S))), D_on=float(delta(X, "on", "leave")(np.arange(S)))) for X in XS}}
# ---------------------------------------------------------------- K6, K7
def k67(arm, rk_k, sign):
    def f(ix):
        vals = []
        for p in PINS:
            key = f"E{rk_k}p{p}_{arm}"; a, e = meas(key, p)["b"], meas(ref_of(key), p)["b"]
            vals.append(sign * (a - e)[ix].mean())
        return float(np.mean(vals))
    return f
k6v = bt(k67("BAND_K3OFF", 3, -1)); k7v = bt(k67("BAND_K3ON", 2, 1)); k7h = bt(k67("H16_K3ON", 2, 1))
res["K6"] = dict(Delta=k6v, verdict=("k3's step is switched on in the band as well" if k6v["est"] >= .10 and k6v["lo"] > 0 else "not"))
res["K7"] = dict(Delta=k7v, Delta_H16=k7h, verdict=("k3's band switches the step on" if k7v["est"] >= .10 and k7v["lo"] > 0 else "not"))
# ---------------------------------------------------------------- descriptive: answers, L47 profiles of the writer heads
def prof(key, hs):
    vals = [np.asarray(r["z47"][key], dtype=float) for r in rows if r.get("z47", {}).get(key) is not None]
    if not vals:
        return None
    z = np.stack(vals)[:, :, hs, :]                                                      # [S, 3, heads, N]
    sm = np.exp(z - z.max(-1, keepdims=True)); sm /= sm.sum(-1, keepdims=True)
    return sm.mean((0, 1, 2)).tolist()
res["descriptive"]["answers"] = {}; res["descriptive"]["L47_profiles"] = {}
for p in PINS:
    keys = [f"E{k}p{p}" for k in J["ks"]] + [f"E5p{p}_{X}_OFF" for X in XS] + [f"E2p{p}_{X}_ON" for X in XS] + \
           [f"E3p{p}_BAND_K3OFF", f"E2p{p}_BAND_K3ON", f"E2p{p}_H16_K3ON"]
    for key in keys:
        x = np.array([r["land"][key] for r in rows])
        res["descriptive"]["answers"][key] = {str(v): float(np.mean(x == v)) for v in range(1, 10) if np.any(x == v)}
    for key in [f"E{k}p{p}" for k in J["ks"]] + [f"E5p{p}_{X}_OFF" for X in ("BAND", "IN46", "BAND_TAIL")] + [f"E2p{p}_{X}_ON" for X in ("BAND", "IN46", "BAND_TAIL")]:
        res["descriptive"]["L47_profiles"][key] = {nm: prof(key, hs) for nm, hs in (("h16", J["h16"]), ("selfaddr", J["selfaddr"]), ("anti", J["anti"]))}
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"]["all"], K0a=res["K0a"]["verdict"], K0b={R: {Y: v["verdict"] for Y, v in res["K0b"][R].items()} for R in ("b", "Tp")},
                      shares={R: {g: {X: v["verdict"] for X, v in res[f"shares_{R}"][g].items()} for g in ("K1", "K2", "K3", "K4")}
                              | {"K3b": res[f"shares_{R}"]["K3b"]["verdict"], "K5": {X: v["verdict"] for X, v in res[f"shares_{R}"]["K5"].items()}}
                              for R in ("b", "Tp")}, K6=res["K6"]["verdict"], K7=res["K7"]["verdict"]), indent=1))
print("->", OUT)
