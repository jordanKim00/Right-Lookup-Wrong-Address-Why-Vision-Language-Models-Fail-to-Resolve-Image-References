"""S862 analysis (rules fixed before the data; knowledge/DECISIONS.md S862 entry).  Input results/r3/s862_<tag>_predict.json.
Definitions (fixed before the data):
  anchor a (per run): L47 head-mean block-score argmax over blocks 1..N at rows B1 / B4 / answer (read-row indices 2, 5, 9); the majority of
    the three, the answer row's when all differ.  An arm's a = the anchor of its recipient E in the same sub-pass.
  answer x: the menu word chosen; x > N (an unshown picture's name) = 'other'.  j* = N + 1 - k (END), q (START).
  late profile z: L51 head-mean (late_mean) block scores, rows B1 / B4 / answer averaged.
  Arms are compared with the E of their own sub-pass, scene by scene; scene bootstrap 4,000 (seed 862).
I0 (validity): SELF / SELF_FA bit-identical to their sub-pass E (answers and late scores); t1 >= .95; START accuracy (G, all N and q) >= .95;
  pin reach (P: E and R references, L43 / L47 argmax at rows B1 / B4 / answer = p) >= .95; R(p) answers p >= .90.  A part whose check
  fails is not read (self / t1 / START -> nothing is read; pin reach or R(p) -> D is not read).
A (per-run prediction of natural END answers, G; the frozen predictors are the registered files with their sha256):
  A1 mean per-run log-likelihood, every model's distribution over {other, 1..N} mixed with 1% uniform: CIRC (anchor-conditioned) minus the
     best of the anchor-free R7, MU, POINTER, COPY, UNIFORM > 0 with the interval above 0 'the measured anchor plus the frozen circuit rule
     predicts natural answers better than every behavioural comparator'; else 'not better than <best>'.
  A2 (the writer's direction, model-free): END runs with k >= 3 and a >= 2: P(x < a) >= .60 with the lower bound > .50 'END moves the answer
     start-ward of the early anchor'; START runs: P(x = a) >= .90 'START keeps the anchor'.
  A3 (the switch): P(x < a) at k >= 3 minus at k <= 2 (END, a >= 2) >= .10 with the interval above 0 'the start-ward move is switched on by k'.
  A4 (calibration to N + 1 - k; the advisor's open gap): copy-anchor runs (a = k) in the cells k in {4, 5, 6} with N - k >= 3 (no start route):
     needed move m = a - j* = 2k - N - 1 (-2 .. +2).  Slope of [x < a] on m within k (k fixed effects, least squares, scene bootstrap):
     |slope| <= .05 with the interval inside +-.10 'the start-ward step is not calibrated to the needed distance (N-blind) in the tested
     conditions'; slope >= .10 with the interval above 0 'the start-ward move grows with the needed distance (behavioural calibration; its
     locus is read with D4)'; else 'partial'.  Each k needs >= 2 N values.
B (cell-level prediction, G END cells N 6-9 x k = 1..N): mean total-variation distance between predicted and observed answer distributions over
  {other, 1..N}.  Ranked for CIRC (anchor=None), TH2 (theorist), R7, MU, POINTER, COPY, UNIFORM; B1: 'TH2 / CIRC predicts the grid at least as
  well as the best anchor-free behavioural fit' if its mean TV <= min(R7, MU) + .01, else 'not'.  No claim beyond the ranking.
C (natural interventions, the six I cells; donors = the same-ordinal START; deltas pooled over cells):
  C1 BAND_S: dP(x = k) >= +.30 with the interval above 0 'the band writes the START address k (an absolute address)';
  C2 SELFADDR_S: dP(x = k) >= +.20 with the interval above 0 'the self-address heads write the START address k';
  C3 ANTI_Z: dP(x = a) >= +.15 with the interval above 0, and dP(x = a)(ANTI_Z) - dP(x = a)(RAND5a_Z) >= .10 with the interval above 0
     'the anti-self heads make the answer leave the anchor (new cells, natural runs)';
  C4 H16_S in the cells with k >= 4: dP(x = a - 1) <= -.10 with the interval below 0, and below the mean of RAND1a_S / RAND1b_S by >= .08 with
     the interval of that difference below 0 'h16 carries the predecessor direction';
  C5 (same cells): dP(x = a)(ANTI_Z) - dP(x = a)(H16_S) >= .10 with the interval above 0 'h16 carries the direction, not the leaving';
  C6 RAND1a_S and RAND1b_S: |dP(x = k)|, |dP(x = a)|, |dP(x = a - 1)| <= .08 each (pooled) 'no effect from random single heads'; else 'random
     heads move answers' (C4's random margin is then the only control read);
  C7 (run-level double dissociation, ANTI_Z): runs whose E anchor a = j*: dP(x = j*) >= +.15 with the interval above 0; runs with a - 1 = j*:
     dP(x = j*) <= -.15 with the interval below 0 -> 'the per-run anchor predicts whether removing the leaving helps or hurts'.  Each side needs
     >= 15 runs from >= 8 scenes, else 'not testable (too few runs)'.
D (pinned transfer at new anchors 2, 3, 6; N9 END5 and N8 END5 - the same ordinal and pins in two layouts):
  shift S_X(p <- p') = [z[p'-1] - z[p-1]](X_PX{p'}) - [z[p'-1] - z[p-1]](E); ratio r = S_H16 / S_BAND, pooled within a pair class.
  D1 interior pairs 3 <-> 6 (both layouts, both directions): BAND valid if S_BAND > 0 with the interval above 0; r >= .5 with the lower bound
     > 0 'h16 carries an anchor-keyed address at new anchors'; r <= .2 with the upper bound <= .35 'anchor-independent'; else 'mixed'.
  D2 donor pin 2 (p' - 1 = 1), recipients 3 and 6: P(x = 1) under BAND_PX2 and under H16_PX2 (each pooled): >= .30 'transfer to picture 1';
     <= .10 'no effective transfer to picture 1 observed'; else 'partial'.  The pinned E at 2: P(x = 1) <= .10 'picture-1 exclusion at the
     pinned anchor 2', >= .30 'no exclusion', else 'partial'.
  D3 (descriptive): pairs 2 <-> 3 and 2 <-> 6 shifts and ratios; score changes at blocks 1, 2, 3; H16_R / BAND_R removal of T at the new pins.
  D4 (N-invariance of the band's step at a fixed ordinal and anchor; the advisor's calibration gap): k = 5, p in {3, 6}; the band's END part
     B_X(N, p) = X(E) - X(BAND_R) for X = T (z[p-1] - z[p+1]) and U ((z[p-1] + z[p+1]) / 2 - z[p]); delta_X = mean_p [B_X(9, p) - B_X(8, p)],
     m_X = mean_p [B_X(9, p) + B_X(8, p)] / 2.  Valid if m_X's interval is above 0.  'N-invariant' if |delta_X| <= .25 m_X with the interval
     of delta_X inside +-.5 m_X; 'N-dependent' if the interval of delta_X excludes 0 and |delta_X| >= .25 m_X; else 'unresolved'.  The needed
     move differs (p = 6: two steps in N8, one in N9; p = 3: one and two steps end-ward), so a calibrated band would differ between layouts.
     Behaviour: the one-back share of E at p, N9 - N8, |.| <= .10 with the interval inside +-.20 'the pinned step is N-blind in behaviour'.
E (descriptive): per (N, k) accuracy, anchor distribution, x - a distribution, P(x = a - 1) by (N, k, a); I-cell per-cell shares; P-cell
  answer distributions; the agents' frozen arm and cell predictions scored where given.
Usage: python3 s862_analyze.py TAG"""
from __future__ import annotations

import collections, hashlib, importlib.util, json, sys
import numpy as np

sys.dont_write_bytecode = True
TAG = sys.argv[1]
J = json.load(open(f"/vlm/results/r3/s862_{TAG}_predict.json"))
OUT = f"/vlm/results/r3/s862_analysis_{TAG}.json"
rng = np.random.default_rng(862)
NB = 4000
rows = J["rows"]; S = len(rows)
SP = rows[0]["subpass"]
E47, E43 = str(J["early"][-1]), str(J["early"][-2])
L0 = str(J["late"][0]); LM = J["late_mean"][L0]
AG = "/vlm/research/ces/outputs/writer_circuit_2026-10-20"
FROZEN = {"adversary": (f"{AG}/a co-analysis/predictions/s862_frozen.py", None),          # sha256 filled in at registration
          "theorist": (f"{AG}/a co-analysis/t5_s862_predictions_TH2a.json", None)}
REG = json.load(open(f"{AG}/main/s862_frozen_hashes.json")) if __import__("os").path.exists(f"{AG}/main/s862_frozen_hashes.json") else {}
ICELLS = [(9, 4), (9, 6), (9, 7), (7, 5), (6, 4), (6, 3)]
PCELLS = [tuple(c) for c in J["p_cells"]]                                             # [(9, 5), (8, 5)] in the run
PINS = (2, 3, 6)


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def bt(fn):
    est = fn(np.arange(S)); bs = [fn(rng.integers(0, S, S)) for _ in range(NB)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return dict(est=float(est), lo=float(lo), hi=float(hi))


def ref_of(armkey):
    tagp, recp = armkey.split("_")[0], armkey.split("_")[1]
    return f"{tagp}_{recp}@{SP[armkey]}"


def anchor_r(r, key, L=E47):
    q = np.asarray(r["qk"][key][L])
    am = [int(np.argmax(q[i])) + 1 for i in (2, 5, 9)]
    c = collections.Counter(am).most_common()
    return c[0][0] if c[0][1] >= 2 else am[2]


def land(key, N):
    x = np.array([r["land"][key] for r in rows])
    return np.where((x >= 1) & (x <= N), x, 0)                                           # 0 = other


def anchors(key):
    return np.array([anchor_r(r, key) for r in rows])


def zlate(r, key):
    return np.asarray(r["zl"][key][L0], dtype=float)[:, LM].mean(1).mean(0)


res = dict(tag=TAG, n_scenes=S, n_runs=J["n_runs"])
# ---------------------------------------------------------------- I0
self_ok, bad = True, []
for key in [k for k in rows[0]["land"] if k.endswith("_SELF") or k.endswith("_SELF_FA")]:
    rk = ref_of(key)
    for r in rows:
        if r["land"][key] != r["land"][rk] or not all(np.array_equal(np.asarray(r["zl"][key][L]), np.asarray(r["zl"][rk][L])) for L in r["zl"][rk]):
            self_ok = False; bad.append(key)
st = [float(np.mean(land(f"G{n}_START{q}", n) == q)) for n in (6, 7, 8, 9) for q in range(1, n + 1)]
st_acc = float(np.mean(st))
reach, racc = [], []
for n, k in PCELLS:
    for p in PINS:
        for key in (f"P{n}_E{k}p{p}", f"P{n}_Rp{p}"):
            for r in rows:
                for L in (E43, E47):
                    q = np.asarray(r["qk"][key][L]); reach += [int(np.argmax(q[i])) + 1 == p for i in (2, 5, 9)]
        racc.append(float(np.mean(land(f"P{n}_Rp{p}", n) == p)))
I0 = dict(self_bit_identical=bool(self_ok), t1=bool((J["t1_menu"] or 0) >= .95), START_acc=bool(st_acc >= .95), pin_reach=bool(np.mean(reach) >= .95),
          R_acc=bool(np.mean(racc) >= .90))
res["I0"] = dict(**I0, core=bool(I0["self_bit_identical"] and I0["t1"] and I0["START_acc"]), D_valid=bool(I0["pin_reach"] and I0["R_acc"]),
                 t1_value=J["t1_menu"], START_acc_value=st_acc, START_by_cell=st, pin_reach_value=float(np.mean(reach)), R_acc_value=float(np.mean(racc)),
                 self_bad=sorted(set(bad))[:20])
# ---------------------------------------------------------------- frozen predictors
hashes = {nm: sha(pth) for nm, (pth, _) in FROZEN.items() if __import__("os").path.exists(pth)}
_pp = f"{AG}/a co-analysis/predictions/s862_params.json"
if __import__("os").path.exists(_pp):
    hashes["adversary_params"] = sha(_pp)
res["frozen"] = dict(hashes_now=hashes, hashes_registered=REG, match={nm: (REG.get(nm) == h) for nm, h in hashes.items()})
ADV = None
if "adversary" in hashes and res["frozen"]["match"].get("adversary") and res["frozen"]["match"].get("adversary_params"):
    spec = importlib.util.spec_from_file_location("s862_frozen", FROZEN["adversary"][0]); ADV = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ADV)
TH2 = json.load(open(FROZEN["theorist"][0]))["cells"] if ("theorist" in hashes and res["frozen"]["match"].get("theorist")) else None


def mix(p, N):
    p = np.asarray(p, dtype=float); assert len(p) == N + 1, (len(p), N)
    p = np.clip(p, 0, None); p = p / p.sum()
    return .99 * p + .01 / (N + 1)


def ll_runs(model, N, k, form, an, x):
    """per-run log-likelihood of answers x (0 = other) under ADV.predict."""
    out = np.zeros(len(x))
    for i in range(len(x)):
        p = mix(ADV.predict(model, N, k, form, "E", (int(an[i]) if model == "CIRC" else None), None), N)
        out[i] = np.log(p[int(x[i])])
    return out


# ---------------------------------------------------------------- A (per-run, G END)
GEND = [(n, k) for n in (6, 7, 8, 9) for k in range(1, n + 1)]
X = {c: land(f"G{c[0]}_END{c[1]}", c[0]) for c in GEND}
A_ = {c: anchors(f"G{c[0]}_END{c[1]}") for c in GEND}
XS = {(n, q): land(f"G{n}_START{q}", n) for n in (6, 7, 8, 9) for q in range(1, n + 1)}
AS = {(n, q): anchors(f"G{n}_START{q}") for n in (6, 7, 8, 9) for q in range(1, n + 1)}
res["A"] = {}
if ADV is not None:
    MODELS = ["CIRC", "R7", "MU", "POINTER", "COPY", "UNIFORM"]
    LL = {m: np.stack([ll_runs(m, n, k, "END", A_[(n, k)], X[(n, k)]) for (n, k) in GEND], 1) for m in MODELS}     # [S, cells]
    mean_ll = {m: bt(lambda ix, m=m: float(LL[m][ix].mean())) for m in MODELS}
    comp = [m for m in MODELS if m != "CIRC"]
    best = max(comp, key=lambda m: mean_ll[m]["est"])
    d1 = bt(lambda ix: float(LL["CIRC"][ix].mean() - LL[best][ix].mean()))
    res["A"]["A1"] = dict(mean_ll=mean_ll, best_comparator=best, diff=d1,
                          verdict=("the measured anchor plus the frozen circuit rule predicts natural answers better than every behavioural comparator"
                                   if d1["est"] > 0 and d1["lo"] > 0 else f"not better than {best}"),
                          by_N={str(n): {m: float(np.mean([LL[m][:, i].mean() for i, c in enumerate(GEND) if c[0] == n])) for m in MODELS} for n in (6, 7, 8, 9)})
else:
    res["A"]["A1"] = dict(verdict="not available (frozen adversary file missing or hash mismatch)")
def frac_lt(cells):
    def f(ix):
        num = den = 0
        for c in cells:
            x, a = X[c][ix], A_[c][ix]; m = a >= 2
            num += np.sum((x < a) & (x >= 1) & m); den += np.sum(m)
        return num / max(den, 1)
    return f
K3 = [c for c in GEND if c[1] >= 3]; K12 = [c for c in GEND if c[1] <= 2]
a2 = bt(frac_lt(K3)); a2s = bt(lambda ix: float(np.mean(np.concatenate([XS[c][ix] == AS[c][ix] for c in XS]))))
res["A"]["A2"] = dict(P_lt_anchor_k3=a2, P_stay_START=a2s,
                      verdict=("END moves the answer start-ward of the early anchor" if a2["est"] >= .60 and a2["lo"] > .50 else "not"),
                      verdict_START=("START keeps the anchor" if a2s["est"] >= .90 else "not"))
a3 = bt(lambda ix: frac_lt(K3)(ix) - frac_lt(K12)(ix))
res["A"]["A3"] = dict(diff=a3, P_lt_k12=bt(frac_lt(K12)),
                      verdict=("the start-ward move is switched on by k" if a3["est"] >= .10 and a3["lo"] > 0 else "not"))
A4C = [(n, k) for k in (4, 5, 6) for n in (6, 7, 8, 9) if n - k >= 3]
def slope4(ix):
    ys, ms, ks = [], [], []
    for (n, k) in A4C:
        x, a = X[(n, k)][ix], A_[(n, k)][ix]; sel = a == k
        ys += list(((x < a) & (x >= 1))[sel].astype(float)); ms += [2 * k - n - 1] * int(sel.sum()); ks += [k] * int(sel.sum())
    ys, ms, ks = np.array(ys), np.array(ms, dtype=float), np.array(ks)
    if len(ys) < 10:
        return np.nan
    Xd = np.stack([ms] + [(ks == kk).astype(float) for kk in sorted(set(ks.tolist()))], 1)
    return float(np.linalg.lstsq(Xd, ys, rcond=None)[0][0])
a4 = bt(slope4)
nvals = {k: [n for (n, kk) in A4C if kk == k] for k in (4, 5, 6)}
a4cells = {f"N{n}_END{k}": dict(m=2 * k - n - 1, n_copy=int(np.sum(A_[(n, k)] == k)),
                                P_lt_a_copy=float(np.mean(((X[(n, k)] < k) & (X[(n, k)] >= 1))[A_[(n, k)] == k])) if np.any(A_[(n, k)] == k) else None)
           for (n, k) in A4C}
res["A"]["A4"] = dict(slope=a4, cells=a4cells, n_values=nvals,
                      verdict=("the start-ward step is not calibrated to the needed distance (N-blind) in the tested conditions"
                               if abs(a4["est"]) <= .05 and a4["lo"] >= -.10 and a4["hi"] <= .10 else
                               "the start-ward move grows with the needed distance (behavioural calibration; its locus is read with D4)"
                               if a4["est"] >= .10 and a4["lo"] > 0 else "partial"))
# ---------------------------------------------------------------- B (cell-level)
def obs_dist(c):
    n = c[0]; cnt = np.bincount(X[c], minlength=n + 1)[:n + 1]
    return cnt / cnt.sum()
TV = {}
cellpred = {}
if ADV is not None:
    for m in ("CIRC", "R7", "MU", "POINTER", "COPY", "UNIFORM"):
        cellpred[m] = {c: mix(ADV.predict(m, c[0], c[1], "END", "E", None, None), c[0]) for c in GEND}
if TH2 is not None:
    cellpred["TH2"] = {c: mix(TH2[f"N{c[0]}_END{c[1]}"]["prob_other_then_pictures_1toN"], c[0]) for c in GEND if f"N{c[0]}_END{c[1]}" in TH2}
for m, P in cellpred.items():
    TV[m] = float(np.mean([.5 * np.abs(P[c] - obs_dist(c)).sum() for c in P]))
res["B"] = dict(mean_TV=TV, ranking=sorted(TV, key=TV.get))
if TV:
    ref_ = min([TV[m] for m in ("R7", "MU") if m in TV], default=None)
    res["B"]["B1"] = {m: ("predicts the grid at least as well as the best anchor-free behavioural fit" if (ref_ is not None and TV[m] <= ref_ + .01) else "not")
                      for m in ("TH2", "CIRC") if m in TV}
    res["B"]["TV_by_cell"] = {f"N{c[0]}_END{c[1]}": {m: float(.5 * np.abs(P[c] - obs_dist(c)).sum()) for m, P in cellpred.items() if c in P} for c in GEND}
# ---------------------------------------------------------------- C (I cells)
def ishare(arm, cells, target):
    """pooled change of P(x = target) (arm - its sub-pass E); target(N, k, a) -> array of target positions per scene."""
    pre = {}
    for (n, k) in cells:
        key = f"I{n}_END{k}_{arm}"; rk = ref_of(key)
        xa, xe, a = land(key, n), land(rk, n), anchors(rk)
        pre[(n, k)] = (xa, xe, target(n, k, a))
    def f(ix):
        return float(np.mean([np.mean(xa[ix] == t[ix]) - np.mean(xe[ix] == t[ix]) for xa, xe, t in pre.values()]))
    return f
TK = lambda n, k, a: np.full(len(a), k)
TA = lambda n, k, a: a
TAM1 = lambda n, k, a: a - 1
K4 = [c for c in ICELLS if c[1] >= 4]
C = {}
c1 = bt(ishare("BAND_S", ICELLS, TK))
C["C1"] = dict(dP_k=c1, verdict=("the band writes the START address k (an absolute address)" if c1["est"] >= .30 and c1["lo"] > 0 else "not"))
c2 = bt(ishare("SELFADDR_S", ICELLS, TK))
C["C2"] = dict(dP_k=c2, verdict=("the self-address heads write the START address k" if c2["est"] >= .20 and c2["lo"] > 0 else "not"))
c3 = bt(ishare("ANTI_Z", ICELLS, TA)); c3r = bt(ishare("RAND5a_Z", ICELLS, TA))
c3d = bt(lambda ix: ishare("ANTI_Z", ICELLS, TA)(ix) - ishare("RAND5a_Z", ICELLS, TA)(ix))
C["C3"] = dict(dP_a=c3, dP_a_RAND5a_Z=c3r, diff=c3d,
               verdict=("the anti-self heads make the answer leave the anchor (new cells, natural runs)"
                        if c3["est"] >= .15 and c3["lo"] > 0 and c3d["est"] >= .10 and c3d["lo"] > 0 else "not"))
c4 = bt(ishare("H16_S", K4, TAM1))
c4r = bt(lambda ix: .5 * (ishare("RAND1a_S", K4, TAM1)(ix) + ishare("RAND1b_S", K4, TAM1)(ix)))
c4d = bt(lambda ix: ishare("H16_S", K4, TAM1)(ix) - .5 * (ishare("RAND1a_S", K4, TAM1)(ix) + ishare("RAND1b_S", K4, TAM1)(ix)))
C["C4"] = dict(dP_am1=c4, dP_am1_RAND1=c4r, diff=c4d,
               verdict=("h16 carries the predecessor direction" if c4["est"] <= -.10 and c4["hi"] < 0 and c4d["est"] <= -.08 and c4d["hi"] < 0 else "not"))
c5 = bt(lambda ix: ishare("ANTI_Z", K4, TA)(ix) - ishare("H16_S", K4, TA)(ix))
C["C5"] = dict(diff=c5, dP_a_H16=bt(ishare("H16_S", K4, TA)), dP_a_ANTI=bt(ishare("ANTI_Z", K4, TA)),
               verdict=("h16 carries the direction, not the leaving" if c5["est"] >= .10 and c5["lo"] > 0 else "not"))
c6 = {arm: {t: float(ishare(arm, ICELLS, fn)(np.arange(S))) for t, fn in (("k", TK), ("a", TA), ("am1", TAM1))} for arm in ("RAND1a_S", "RAND1b_S")}
C["C6"] = dict(changes=c6, verdict=("no effect from random single heads" if all(abs(v) <= .08 for d in c6.values() for v in d.values()) else "random heads move answers"))
# C7: run-level double dissociation
def c7(side):
    sel = []
    for (n, k) in ICELLS:
        key = f"I{n}_END{k}_ANTI_Z"; rk = ref_of(key); js = n + 1 - k
        xa, xe, a = land(key, n), land(rk, n), anchors(rk)
        m = (a == js) if side == "id" else (a - 1 == js)
        sel += [(s, float(xa[s] == js) - float(xe[s] == js)) for s in range(S) if m[s]]
    return sel
C["C7"] = {}
ok7 = True; v7 = {}
for side in ("id", "helped"):
    sel = c7(side); scenes = sorted({s for s, _ in sel})
    by = collections.defaultdict(list)
    for s, d in sel:
        by[s].append(d)
    def f(ix, by=by):
        vals = [v for s in ix for v in by.get(int(s), [])]
        return float(np.mean(vals)) if vals else np.nan
    est = bt(f) if sel else dict(est=np.nan, lo=np.nan, hi=np.nan)
    v7[side] = dict(n_runs=len(sel), n_scenes=len(scenes), dP_correct=est)
    if len(sel) < 15 or len(scenes) < 8:
        ok7 = False
C["C7"] = dict(**v7, verdict=("not testable (too few runs)" if not ok7 else
                              "the per-run anchor predicts whether removing the leaving helps or hurts"
                              if (v7["id"]["dP_correct"]["est"] >= .15 and v7["id"]["dP_correct"]["lo"] > 0 and
                                  v7["helped"]["dP_correct"]["est"] <= -.15 and v7["helped"]["dP_correct"]["hi"] < 0) else "not"))
desc = {}
for (n, k) in ICELLS:
    rk0 = f"I{n}_END{k}"
    a0 = anchors(rk0); x0 = land(rk0, n)
    d = dict(j=n + 1 - k, acc_E=float(np.mean(x0 == n + 1 - k)), anchor={str(v): float(np.mean(a0 == v)) for v in sorted(set(a0.tolist()))},
             x_minus_a={str(v): float(np.mean(x0 - a0 == v)) for v in sorted(set((x0 - a0).tolist()))})
    for arm in ("BAND_S", "ANTI_Z", "SELFADDR_S", "H16_S", "RAND1a_S", "RAND1b_S", "RAND5a_Z"):
        key = f"I{n}_END{k}_{arm}"; rk = ref_of(key)
        xa, xe, a = land(key, n), land(rk, n), anchors(rk)
        d[arm] = dict(dacc=float(np.mean(xa == n + 1 - k) - np.mean(xe == n + 1 - k)), dk=float(np.mean(xa == k) - np.mean(xe == k)),
                      da=float(np.mean(xa == a) - np.mean(xe == a)), dam1=float(np.mean(xa == a - 1) - np.mean(xe == a - 1)),
                      dist={str(v): float(np.mean(xa == v)) for v in range(0, n + 1)})
    desc[f"N{n}_END{k}"] = d
C["descriptive"] = desc
res["C"] = C
# ---------------------------------------------------------------- D (P cells)
D = {}
if res["I0"]["D_valid"]:
    ZC = {}
    for n, k in PCELLS:
        for p in PINS:
            for pp in PINS:
                if pp == p:
                    continue
                for X_ in ("H16", "BAND"):
                    key = f"P{n}_E{k}p{p}_{X_}_PX{pp}"; rk = ref_of(key)
                    ZC[(n, p, pp, X_)] = (np.array([zlate(r, key) for r in rows]), np.array([zlate(r, rk) for r in rows]))
    def shift(pairs, X_):
        def f(ix):
            vals = []
            for n, k in PCELLS:
                for p, pp in pairs:
                    za, ze = ZC[(n, p, pp, X_)]; za, ze = za[ix], ze[ix]
                    vals.append(float(np.mean((za[:, pp - 2] - za[:, p - 2]) - (ze[:, pp - 2] - ze[:, p - 2]))))
            return float(np.mean(vals))
        return f
    for cls, pairs in (("3_6", [(3, 6), (6, 3)]), ("2_3", [(2, 3), (3, 2)]), ("2_6", [(2, 6), (6, 2)])):
        sb = bt(shift(pairs, "BAND")); sh = bt(shift(pairs, "H16"))
        rr = bt(lambda ix, pairs=pairs: shift(pairs, "H16")(ix) / (shift(pairs, "BAND")(ix) + 1e-12))
        v = ("h16 carries an anchor-keyed address at new anchors" if rr["est"] >= .5 and rr["lo"] > 0 else
             "anchor-independent" if rr["est"] <= .2 and rr["hi"] <= .35 else "mixed")
        D[cls] = dict(S_BAND=sb, S_H16=sh, ratio=rr, band_valid=bool(sb["lo"] > 0), verdict=(v if sb["lo"] > 0 else "transfer test invalid"))
    D["D1"] = D["3_6"]["verdict"]
    def p1(arm):
        def f(ix):
            vals = []
            for n, k in PCELLS:
                for p in (3, 6):
                    key = f"P{n}_E{k}p{p}_{arm}"
                    vals.append(float(np.mean(land(key, n)[ix] == 1)))
            return float(np.mean(vals))
        return f
    d2 = {arm: bt(p1(arm)) for arm in ("BAND_PX2", "H16_PX2")}
    d2e = bt(lambda ix: float(np.mean([np.mean(land(f"P{n}_E{k}p2", n)[ix] == 1) for n, k in PCELLS])))
    lab = lambda e: "transfer to picture 1" if e >= .30 else "no effective transfer to picture 1 observed" if e <= .10 else "partial"
    D["D2"] = dict(P1=d2, P1_E_pin2=d2e, verdict={arm: lab(v["est"]) for arm, v in d2.items()},
                   verdict_E=("picture-1 exclusion at the pinned anchor 2" if d2e["est"] <= .10 else "no exclusion" if d2e["est"] >= .30 else "partial"))
    pdesc = {}
    for n, k in PCELLS:
        for p in PINS:
            rk0 = f"P{n}_E{k}p{p}"; x0 = land(rk0, n)
            dd = dict(dist_E={str(v): float(np.mean(x0 == v)) for v in range(0, n + 1)}, R_acc=float(np.mean(land(f"P{n}_Rp{p}", n) == p)))
            for arm in [f"{X_}_PX{pp}" for pp in PINS if pp != p for X_ in ("H16", "BAND")] + ["H16_R", "BAND_R"]:
                key = f"{rk0}_{arm}"; rk = ref_of(key)
                xa = land(key, n); za = np.array([zlate(r, key) for r in rows]); ze = np.array([zlate(r, rk) for r in rows])
                T_ = lambda z: z[:, p - 2] - z[:, p] if p < n else z[:, p - 2]
                dd[arm] = dict(dist={str(v): float(np.mean(xa == v)) for v in range(0, n + 1)}, dz=(za - ze).mean(0).tolist(),
                               dT=float(np.mean(T_(za) - T_(ze))))
            pdesc[f"N{n}_E{k}p{p}"] = dd
    D["descriptive"] = pdesc
    if sorted(PCELLS) == [(8, 5), (9, 5)]:
        def Tz(z, p):
            return z[:, p - 2] - z[:, p]
        def Uz(z, p):
            return (z[:, p - 2] + z[:, p]) / 2 - z[:, p - 1]
        BZ = {}
        for n in (8, 9):
            for p in (3, 6):
                key = f"P{n}_E5p{p}_BAND_R"; rk = ref_of(key)
                BZ[(n, p)] = (np.array([zlate(r, rk) for r in rows]), np.array([zlate(r, key) for r in rows]),
                              land(rk, n), land(key, n))
        def bx(fn, n, p, ix):
            ze, za, _, _ = BZ[(n, p)]
            return float(np.mean(fn(ze[ix], p) - fn(za[ix], p)))
        d4 = {}
        for nm, fn in (("T", Tz), ("U", Uz)):
            dl = bt(lambda ix, fn=fn: float(np.mean([bx(fn, 9, p, ix) - bx(fn, 8, p, ix) for p in (3, 6)])))
            mm = bt(lambda ix, fn=fn: float(np.mean([(bx(fn, 9, p, ix) + bx(fn, 8, p, ix)) / 2 for p in (3, 6)])))
            m_ = mm["est"]
            if not mm["lo"] > 0:
                v = "invalid (the band's END part is not above 0)"
            elif abs(dl["est"]) <= .25 * m_ and dl["lo"] >= -.5 * m_ and dl["hi"] <= .5 * m_:
                v = "N-invariant"
            elif (dl["lo"] > 0 or dl["hi"] < 0) and abs(dl["est"]) >= .25 * m_:
                v = "N-dependent"
            else:
                v = "unresolved"
            d4[nm] = dict(delta=dl, mean=mm, by_pin={str(p): {str(n): bx(fn, n, p, np.arange(S)) for n in (8, 9)} for p in (3, 6)}, verdict=v)
        ob = bt(lambda ix: float(np.mean([np.mean(BZ[(9, p)][2][ix] == p - 1) - np.mean(BZ[(8, p)][2][ix] == p - 1) for p in (3, 6)])))
        d4["one_back_E_N9_minus_N8"] = ob
        d4["verdict_behaviour"] = ("the pinned step is N-blind in behaviour" if abs(ob["est"]) <= .10 and ob["lo"] >= -.20 and ob["hi"] <= .20 else "not")
        D["D4"] = d4
else:
    D["verdict"] = "not read (pin reach or R(p) failed)"
res["D"] = D
# ---------------------------------------------------------------- E (descriptive grid)
grid = {}
for (n, k) in GEND:
    x, a = X[(n, k)], A_[(n, k)]; js = n + 1 - k
    grid[f"N{n}_END{k}"] = dict(j=js, acc=float(np.mean(x == js)), anchor={str(v): float(np.mean(a == v)) for v in sorted(set(a.tolist()))},
                                x_minus_a={str(v): float(np.mean(x - a == v)) for v in sorted(set((x - a).tolist()))},
                                P_lt_a=float(np.mean((x < a) & (x >= 1))), P_am1=float(np.mean(x == a - 1)), P_stay=float(np.mean(x == a)),
                                dist={str(v): float(np.mean(x == v)) for v in range(0, n + 1)})
res["E"] = dict(grid=grid, START={f"N{n}_START{q}": float(np.mean(XS[(n, q)] == q)) for (n, q) in XS})
# the agents' arm predictions (descriptive): mean log-likelihood of the arm answers
if ADV is not None:
    armll = {}
    for (n, k) in ICELLS:
        for arm in ("BAND_S", "ANTI_Z", "SELFADDR_S", "H16_S", "RAND1a_S", "RAND1b_S", "RAND5a_Z"):
            key = f"I{n}_END{k}_{arm}"; rk = ref_of(key)
            xa, a = land(key, n), anchors(rk)
            try:
                armll[key] = float(np.mean([np.log(mix(ADV.predict("CIRC", n, k, "END", arm, int(a[s]), None), n)[int(xa[s])]) for s in range(S)]))
            except Exception as ex:                                                       # an arm the frozen file does not cover
                armll[key] = f"not covered ({type(ex).__name__})"
    for n, k in PCELLS:
        for p in PINS:
            rk0 = f"P{n}_E{k}p{p}"
            for arm in ["E"] + [f"{X_}_PX{pp}" for pp in PINS if pp != p for X_ in ("H16", "BAND")] + ["H16_R", "BAND_R"]:
                key = rk0 if arm == "E" else f"{rk0}_{arm}"; xa = land(key, n)
                try:
                    armll[key] = float(np.mean([np.log(mix(ADV.predict("CIRC", n, k, "END", arm, None, p), n)[int(xa[s])]) for s in range(S)]))
                except Exception as ex:
                    armll[key] = f"not covered ({type(ex).__name__})"
    res["E"]["CIRC_arm_ll"] = armll
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"]["core"], D_valid=res["I0"]["D_valid"], A1=res["A"]["A1"]["verdict"], A2=[res["A"]["A2"]["verdict"], res["A"]["A2"]["verdict_START"]],
                      A3=res["A"]["A3"]["verdict"], A4=res["A"]["A4"]["verdict"], B=res["B"].get("B1"), B_rank=res["B"]["ranking"],
                      C={c: res["C"][c]["verdict"] for c in ("C1", "C2", "C3", "C4", "C5", "C6", "C7")},
                      D=({k_: (v["verdict"] if isinstance(v, dict) and "verdict" in v else
                               {kk: vv["verdict"] for kk, vv in v.items() if isinstance(vv, dict) and "verdict" in vv} if k_ == "D4" else v)
                          for k_, v in res["D"].items() if k_ != "descriptive"})), indent=1))
print("->", OUT)
