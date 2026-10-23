"""S910 analysis (rules fixed before the data; knowledge/DECISIONS.md S910 entry).  Input results/r3/s910_<tag>_pinnat.json.
Per (scene, k) with a valid natural anchor a: dep(X) = 1(answer of X != a); T1_a(X) = z(a - 1) - z(a) at L51 (matcher-head mean block scores,
rows B1 / B4 / answer).  Writer-free margins B_nat = T1_a(E_BANDR), B_pin = T1_a(PIN_BANDR); band effects W_nat = T1_a(E) - B_nat,
W_pin = T1_a(PIN) - B_pin.  Key cell: k = 2 with the natural anchor on the last picture (a = 8).
tau = [dep(PIN_EB) - dep(PIN)] / [dep(E) - dep(PIN)] (the natural band write in the pinned context); sigma = [dep(E) - dep(E_PB)] /
[dep(E) - dep(PIN)] (the pinned band write in the natural context); means over the key-cell runs; scene bootstrap 4,000 (seed 910).
I0: t1 >= .95; natural START(q) answers q >= .95; pin reach (PIN's L47 anchor = a in >= .90 of phase-B runs); identity (E_EB reproduces E:
  max late block-score difference <= 1e-3 in >= .99); the key cell has >= 20 runs and dep(E) - dep(PIN) >= .25 there.  Nothing is read if I0
  fails.
P1: tau >= .60 with the lower bound > .40 -> 'the band's write depends on the form of the early address: the natural write makes the pinned
  END(2) leave the last picture'; tau <= .20 with the upper bound < .40 and mean(B_pin - B_nat) <= -1 with the upper bound < 0 -> 'pinning deepens
  the writer-free margin; the band's write is not what differs'; else 'mixed'.
P2: sigma >= .60 with the lower bound > .40 -> 'the pinned band write stops the natural departure'; sigma <= .20 with the upper bound < .40 ->
  'not'; else 'partial'.
Descriptive: per k (2, 3, 5, 6) dep and answers for every arm, W_nat, W_pin, B_nat, B_pin; the key cell's B_pin - B_nat and W_pin - W_nat.
Added 10-22 before the main output was read (rules unchanged; user / advisor request): the full competitive margin of the anchor at L51
m = z_a - max_{i != a} z_i for every arm with the winning competitor's offset, and the L47 support of the initial address
m47 = z47_a - max_{i != a} z47_i (rows B1 / B4 / answer averaged) for E and PIN.
Usage: python3 s910_analyze.py TAG"""
from __future__ import annotations

import collections, json, sys
import numpy as np

TAG = sys.argv[1]
J = json.load(open(f"/vlm/results/r3/s910_{TAG}_pinnat.json"))
OUT = f"/vlm/results/r3/s910_analysis_{TAG}.json"
rng = np.random.default_rng(910); NB = 4000
rows = J["rows"]; S = len(rows); lm = J["late_mean"]; KS = J["ks"]; N = J["N"]
ARMS = ("E", "R", "PIN", "E_BANDR", "PIN_BANDR", "PIN_EB", "E_PB", "E_EB")


def zv(r, key, L="51"):
    return np.asarray(r["zl"][key][L], dtype=float)[:, lm[L]].mean(1).mean(0)


def anchor(q):
    q = np.asarray(q); am = [int(np.argmax(q[i])) + 1 for i in (2, 5, 9)]
    c = collections.Counter(am).most_common(); return c[0][0] if c[0][1] >= 2 else am[2]


REC = []
for si, r in enumerate(rows):
    for k in KS:
        a_ = r["anchor"][str(k)]
        if f"B{k}_E" not in r["land"]:
            continue
        d = dict(si=si, k=k, a=a_)
        for X in ARMS:
            key = f"B{k}_{X}"
            d[f"x_{X}"] = r["land"][key]; d[f"dep_{X}"] = float(r["land"][key] != a_)
            z = zv(r, key); d[f"T_{X}"] = z[a_ - 2] - z[a_ - 1]
        for X in ARMS:                                                                              # (10-22 11:4x, before the main output was read)
            z = zv(r, f"B{k}_{X}"); oth = np.delete(z, a_ - 1)
            d[f"m_{X}"] = float(z[a_ - 1] - oth.max()); d[f"win_{X}"] = int(np.argmax(np.where(np.arange(len(z)) == a_ - 1, -np.inf, z))) + 1
        for X in ("E", "PIN"):                                                                      # L47 support of the initial address
            q47 = np.asarray(r["qk47"][f"B{k}_{X}"])[[2, 5, 9]].mean(0); d[f"m47_{X}"] = float(q47[a_ - 1] - np.delete(q47, a_ - 1).max())
        d["vid"] = max(float(np.abs(zv(r, f"B{k}_E_EB", L) - zv(r, f"B{k}_E", L)).max()) for L in ("51", "55", "59", "63"))
        d["pin_anchor"] = anchor(r["qk47"][f"B{k}_PIN"])
        d["B_nat"] = d["T_E_BANDR"]; d["B_pin"] = d["T_PIN_BANDR"]; d["W_nat"] = d["T_E"] - d["B_nat"]; d["W_pin"] = d["T_PIN"] - d["B_pin"]
        REC.append(d)
KEY = lambda d: d["k"] == 2 and d["a"] == N


def bt(fn):
    est = fn(np.arange(S)); bs = [fn(rng.integers(0, S, S)) for _ in range(NB)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return dict(est=float(est), lo=float(lo), hi=float(hi))


def sel(ix, cond):
    cnt = collections.Counter(int(i) for i in ix)
    return [d for d in REC if cond(d) for _ in range(cnt.get(d["si"], 0))]


def M(ix, cond, key):
    ds = sel(ix, cond); return float(np.mean([d[key] for d in ds])) if ds else float("nan")


res = dict(tag=TAG, n_scenes=S)
st = float(np.mean([r["land"][f"N_S{q}"] == q for r in rows for q in range(1, N + 1)]))
reach = float(np.mean([d["pin_anchor"] == d["a"] for d in REC]))
vid_ok = float(np.mean([d["vid"] <= 1e-3 for d in REC]))
allx = np.arange(S)
nkey = len([d for d in REC if KEY(d)]); gap = M(allx, KEY, "dep_E") - M(allx, KEY, "dep_PIN")
I0 = dict(t1=bool((J["t1_menu"] or 0) >= .95), START=bool(st >= .95), pin_reach=bool(reach >= .90), identity=bool(vid_ok >= .99),
          key_cell=bool(nkey >= 20 and gap >= .25))
res["I0"] = dict(**I0, all=bool(all(I0.values())), t1_value=J["t1_menu"], START_acc=st, reach=reach, vid_ok=vid_ok, n_key=nkey, dep_gap=gap)
ok = res["I0"]["all"]
def _div(a_, b_):
    return a_ / b_ if b_ not in (0, 0.0) and np.isfinite(b_) else float("nan")
tau = lambda ix: _div(M(ix, KEY, "dep_PIN_EB") - M(ix, KEY, "dep_PIN"), M(ix, KEY, "dep_E") - M(ix, KEY, "dep_PIN"))
sig = lambda ix: _div(M(ix, KEY, "dep_E") - M(ix, KEY, "dep_E_PB"), M(ix, KEY, "dep_E") - M(ix, KEY, "dep_PIN"))
bT = bt(tau); bS = bt(sig); bM = bt(lambda ix: M(ix, KEY, "B_pin") - M(ix, KEY, "B_nat"))
res["P1"] = dict(tau=bT, margin_diff=bM,
                 verdict=(("the band's write depends on the form of the early address: the natural write makes the pinned END(2) leave the last picture"
                           if bT["est"] >= .60 and bT["lo"] > .40 else
                           "pinning deepens the writer-free margin; the band's write is not what differs"
                           if bT["est"] <= .20 and bT["hi"] < .40 and bM["est"] <= -1 and bM["hi"] < 0 else "mixed") if ok else "not read"))
res["P2"] = dict(sigma=bS, verdict=(("the pinned band write stops the natural departure" if bS["est"] >= .60 and bS["lo"] > .40 else
                                     "not" if bS["est"] <= .20 and bS["hi"] < .40 else "partial") if ok else "not read"))
desc = {}
for k in KS:
    ds = [d for d in REC if d["k"] == k]
    for nm, cond in ((f"k{k}", lambda d, k=k: d["k"] == k), (f"k{k}_a{N}", lambda d, k=k: d["k"] == k and d["a"] == N)):
        dd = [d for d in REC if cond(d)]
        if not dd:
            continue
        desc[nm] = dict(n=len(dd), anchors=dict(collections.Counter(d["a"] for d in dd)),
                        dep={X: float(np.mean([d[f"dep_{X}"] for d in dd])) for X in ARMS},
                        step={X: float(np.mean([d[f"x_{X}"] == d["a"] - 1 for d in dd])) for X in ARMS},
                        W_nat=float(np.mean([d["W_nat"] for d in dd])), W_pin=float(np.mean([d["W_pin"] for d in dd])),
                        B_nat=float(np.mean([d["B_nat"] for d in dd])), B_pin=float(np.mean([d["B_pin"] for d in dd])),
                        m_L51={X: float(np.mean([d[f"m_{X}"] for d in dd])) for X in ARMS},
                        competitor_L51={X: dict(collections.Counter(d[f"win_{X}"] - d["a"] for d in dd)) for X in ("E", "E_BANDR", "PIN", "PIN_BANDR")},
                        m47={X: float(np.mean([d[f"m47_{X}"] for d in dd])) for X in ("E", "PIN")})
res["descriptive"] = desc
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"]["all"], P1=[res["P1"]["verdict"], bT["est"], bM["est"]], P2=[res["P2"]["verdict"], bS["est"]]), indent=1))
print("->", OUT)
