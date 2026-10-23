"""S923 analysis (rules fixed before the data; knowledge/DECISIONS.md S923 entry; user 'continue' toward prediction by the identified algorithm).
Set A - results/r3/s913_<tagA>_collect.json (exp/s913_collect.py --pin 1, N8 'from the last', fresh scenes): per natural END(k) run at its
natural L47 anchor a, the arms E, E_BANDR (writer removed), PIN (initial address sharpened to START(a)'s), PIN_BANDR.  Frozen predictors: AP
(main/s923_frozen_AP.json, exp/s923_fit.py on the S913 training runs only; hash checked) and the S913 model M3 (main/s913_frozen_predictors.json
db299c27045230be; S914's writer-removal readout with M3's own temperatures).  dep = P(answer != a); an intervention's predicted change = mean
predicted dep(E) - dep(arm) against the observed.  Scene bootstrap 4,000 (seed 923).
Set B - results/r3/s920_<tagB>_r3circuit.json (exp/s920_r3circuit.py, N8, k 1-8, fresh scenes): R³'s algebra predicted by the identified
algorithm - address write only = END(t), frame write only = START(k), both = START(t) (t = N + 1 - k).
I0: set A - t1 >= .95, natural START(q) answers q >= .95 (q < N), PIN reaches its anchor (PIN's L47 anchor = a in >= .90), both frozen hashes
  match, >= 200 natural END runs; set B - t1 >= .95, START >= .95, R³'s READ right in >= .80 of the R3 runs.  Nothing is read if I0 fails.
AP1 (writer removal): |predicted - observed| change of dep from E to E_BANDR (pooled) <= .10 and the correlation over k cells (>= 10 runs) >= .70
  -> 'the circuit-structured predictor predicts the effect of removing the writer'; else 'not'.
AP2 (sharpening): the same for E -> PIN -> 'the circuit-structured predictor predicts the effect of sharpening the initial address'; else 'not'.
AP3 (against M3): e_X = |pred - obs| summed over the two interventions; d = e_M3 - e_AP >= .05 with the lower bound > 0 -> 'the circuit-structured
  revision predicts interventions better than the S913 model'; else 'not'.
AP4 (natural answers): NLL(AP) - NLL(M3) <= .02 with the upper bound < .05 -> 'no loss on natural answers'; else 'loses on natural answers'.
RB1: P(x(R3A_k) = x(N_E t)) >= .85 with the lower bound > .75 -> 'R³'s address write alone turns the question into END(t)'; else 'not'.
RB2: P(x(R3F_k) = x(N_S k)) >= .85 with the lower bound > .75 -> 'R³'s frame write alone turns it into START(k)'; else 'not'.
RB3: P(x(R3_k) = x(N_S t)) >= .90 with the lower bound > .80 -> 'both writes make it START(t)'; else 'not'.
Descriptive: per k predicted and observed dep for every arm and model; NLLs; the R³ agreements per k and their chance levels.
Usage: python3 s923_analyze.py TAG_A TAG_B [OUT]"""
from __future__ import annotations

import collections, hashlib, importlib.util, json, sys
import numpy as np

sys.dont_write_bytecode = True
TA, TB = sys.argv[1:3]
OUT = sys.argv[3] if len(sys.argv) > 3 else f"/vlm/results/r3/s923_analysis_{TA}.json"
MAIN = "/vlm/research/ces/outputs/writer_circuit_2026-10-20/main"
HASH_AP, HASH_M3 = "0b2abbda088edd72", "db299c27045230be"
rng = np.random.default_rng(923); NB = 4000
_argv = sys.argv; sys.argv = ["s923_fit", "x", "y"]
spec = importlib.util.spec_from_file_location("s923_fit", "/vlm/exp/s923_fit.py"); FIT = importlib.util.module_from_spec(spec); spec.loader.exec_module(FIT)
sys.argv = _argv
PA = json.load(open(f"{MAIN}/s923_frozen_AP.json")); PM = json.load(open(f"{MAIN}/s913_frozen_predictors.json"))
ha = hashlib.sha256(open(f"{MAIN}/s923_frozen_AP.json", "rb").read()).hexdigest()[:16]
hm = hashlib.sha256(open(f"{MAIN}/s913_frozen_predictors.json", "rb").read()).hexdigest()[:16]
KG = FIT.KG
AP = {int(g): {kk: np.asarray(v) for kk, v in c.items()} for g, c in PA["AP"].items()}
M3 = {int(g): {kk: np.asarray(v) for kk, v in c.items()} for g, c in PM["M3"].items()}


def softmax(v):
    v = np.asarray(v, float); v = v - v.max(); e = np.exp(v); return e / e.sum()


def p_ap(d, writer=True):
    b, w = FIT.ap_scores(d, AP, 47); T = PA["AP_T"][str(KG(d["k"]))]
    return FIT.mix(np.concatenate([[0.0], softmax(T["B"] * b + (T["W"] * w if writer else 0.0))]), d["N"])


def p_m3(d, writer=True):
    sc = FIT.m3_scores(d, M3, 47, writer); t = PM["M3_T"]["ans"][str(KG(d["k"]))]
    return FIT.mix(np.concatenate([[0.0], softmax(t * sc)]), d["N"])


# ---------------- set A
J = json.load(open(f"/vlm/results/r3/s913_{TA}_collect.json")); N = J["N"]; em = J["early_mean"]
REC = []
for si, r in enumerate(J["rows"]):
    for k in J["ks"]:
        if f"B{k}_E" not in r["land"] or f"B{k}_PIN" not in r["land"]:
            continue
        a = r["anchor"][str(k)]
        def prof(key):
            h = np.asarray(r["qk47h"][key], float).mean(0); return FIT.centre(h[em["47"]].mean(0))
        d = dict(N=N, k=k, si=si, a=a, e47=prof(f"N_E{k}"), x=int(r["land"][f"N_E{k}"]))
        dp = dict(d); dp["e47"] = prof(f"B{k}_PIN")
        q47 = np.asarray(r["qk47"][f"B{k}_PIN"]); am = [int(np.argmax(q47[i])) + 1 for i in (2, 5, 9)]
        c_ = collections.Counter(am).most_common(1)[0]; pa = c_[0] if c_[1] >= 2 else am[2]
        dep = lambda p: float(1 - p[a])
        rec = dict(si=si, k=k, a=a, pin_anchor=pa,
                   obs=dict(E=float(int(r["land"][f"B{k}_E"]) != a), B=float(int(r["land"][f"B{k}_E_BANDR"]) != a),
                            PIN=float(int(r["land"][f"B{k}_PIN"]) != a), PINB=float(int(r["land"][f"B{k}_PIN_BANDR"]) != a)),
                   AP=dict(E=dep(p_ap(d)), B=dep(p_ap(d, False)), PIN=dep(p_ap(dp)), PINB=dep(p_ap(dp, False))),
                   M3=dict(E=dep(p_m3(d)), B=dep(p_m3(d, False)), PIN=dep(p_m3(dp)), PINB=dep(p_m3(dp, False))),
                   nll=dict(AP=float(-np.log(p_ap(d)[FIT.xi(d)])), M3=float(-np.log(p_m3(d)[FIT.xi(d)]))))
        REC.append(rec)
nS = len(J["rows"])


def sel(ix):
    cnt = collections.Counter(int(i) for i in ix)
    return [d for d in REC for _ in range(cnt.get(d["si"], 0))]


def boot(fn):
    est = fn(np.arange(nS)); bs = [fn(rng.integers(0, nS, nS)) for _ in range(NB)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return dict(est=float(est), lo=float(lo), hi=float(hi))


def change(ds, src, arm):
    return float(np.mean([d[src]["E"] - d[src][arm] for d in ds]))


def interv(arm, model):
    pred = change(REC, model, arm); obs = change(REC, "obs", arm)
    cells = collections.defaultdict(list)
    for d in REC:
        cells[d["k"]].append(d)
    cp, co = [], []
    for k, ds in cells.items():
        if len(ds) >= 10:
            cp.append(change(ds, model, arm)); co.append(change(ds, "obs", arm))
    r_ = float(np.corrcoef(cp, co)[0, 1]) if len(cp) >= 3 else float("nan")
    return dict(pred=pred, obs=obs, err=abs(pred - obs), cell_r=r_, n_cells=len(cp), ok=bool(abs(pred - obs) <= .10 and r_ >= .70))


res = dict(tags=[TA, TB], hash_AP=ha, hash_M3=hm)
st = float(np.mean([r["land"][f"N_S{q}"] == q for r in J["rows"] for q in range(1, N)]))
reach = float(np.mean([d["pin_anchor"] == d["a"] for d in REC]))
JB = json.load(open(f"/vlm/results/r3/s920_{TB}_r3circuit.json")); NBn = JB["N"]; rowsB = JB["rows"]; nSB = len(rowsB)
stB = float(np.mean([r["land"][f"N_S{q}"] == q for r in rowsB for q in range(1, NBn)]))
readB = float(np.mean([float(r["plan"][f"R3_{k}"]["n_writes"] > 0 and any(d_["kh"] == k and d_["t"] == NBn + 1 - k and d_["N_hat"] == NBn
                                                                        for d_ in r["plan"][f"R3_{k}"]["detail"])) for r in rowsB for k in JB["ks"]]))
I0 = dict(A_t1=bool((J["t1_menu"] or 0) >= .95), A_START=bool(st >= .95), A_pin_reach=bool(reach >= .90), A_hashes=bool(ha == HASH_AP and hm == HASH_M3),
          A_size=bool(len(REC) >= 200), B_t1=bool((JB["t1_menu"] or 0) >= .95), B_START=bool(stB >= .95), B_READ=bool(readB >= .80))
res["I0"] = dict(**I0, all=bool(all(I0.values())), values=dict(A_t1=J["t1_menu"], A_START=st, pin_reach=reach, n_runs=len(REC), B_t1=JB["t1_menu"], B_START=stB, B_READ=readB))
ok = res["I0"]["all"]
iw = {m: interv("B", m) for m in ("AP", "M3")}; ip = {m: interv("PIN", m) for m in ("AP", "M3")}
res["AP1"] = dict(**iw, verdict=(("the circuit-structured predictor predicts the effect of removing the writer" if iw["AP"]["ok"] else "not") if ok else "not read"))
res["AP2"] = dict(**ip, verdict=(("the circuit-structured predictor predicts the effect of sharpening the initial address" if ip["AP"]["ok"] else "not") if ok else "not read"))
d3 = boot(lambda ix: (lambda ds: (abs(change(ds, "M3", "B") - change(ds, "obs", "B")) + abs(change(ds, "M3", "PIN") - change(ds, "obs", "PIN")))
                      - (abs(change(ds, "AP", "B") - change(ds, "obs", "B")) + abs(change(ds, "AP", "PIN") - change(ds, "obs", "PIN"))))(sel(ix)))
res["AP3"] = dict(d=d3, verdict=(("the circuit-structured revision predicts interventions better than the S913 model" if d3["est"] >= .05 and d3["lo"] > 0 else "not") if ok else "not read"))
d4 = boot(lambda ix: (lambda ds: float(np.mean([d["nll"]["AP"] - d["nll"]["M3"] for d in ds])))(sel(ix)))
res["AP4"] = dict(d=d4, verdict=(("no loss on natural answers" if d4["est"] <= .02 and d4["hi"] < .05 else "loses on natural answers") if ok else "not read"))
# ---------------- set B


def bootB(fn):
    est = fn(np.arange(nSB)); bs = [fn(rng.integers(0, nSB, nSB)) for _ in range(NB)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return dict(est=float(est), lo=float(lo), hi=float(hi))


def agree(ix, arm, ref):
    v = []
    for i in ix:
        r = rowsB[int(i)]
        for k in JB["ks"]:
            t = NBn + 1 - k; rk = ref(k, t)
            if rk in r["land"]:
                v.append(float(r["land"][f"{arm}_{k}"] == r["land"][rk]))
    return float(np.mean(v)) if v else float("nan")


b1 = bootB(lambda ix: agree(ix, "R3A", lambda k, t: f"N_E{t}")); b2 = bootB(lambda ix: agree(ix, "R3F", lambda k, t: f"N_S{k}"))
b3 = bootB(lambda ix: agree(ix, "R3", lambda k, t: f"N_S{t}"))
res["RB1"] = dict(p=b1, verdict=(("R³'s address write alone turns the question into END(t)" if b1["est"] >= .85 and b1["lo"] > .75 else "not") if ok else "not read"))
res["RB2"] = dict(p=b2, verdict=(("R³'s frame write alone turns it into START(k)" if b2["est"] >= .85 and b2["lo"] > .75 else "not") if ok else "not read"))
res["RB3"] = dict(p=b3, verdict=(("both writes make it START(t)" if b3["est"] >= .90 and b3["lo"] > .80 else "not") if ok else "not read"))
desc = dict(A={str(k): dict(n=len([d for d in REC if d["k"] == k]), **{src: {arm: float(np.mean([d[src][arm] for d in REC if d["k"] == k])) for arm in ("E", "B", "PIN", "PINB")}
                                                                       for src in ("obs", "AP", "M3")}) for k in J["ks"]},
            nll=dict(AP=float(np.mean([d["nll"]["AP"] for d in REC])), M3=float(np.mean([d["nll"]["M3"] for d in REC]))),
            B={str(k): dict(R3A_END_t=agree(np.arange(nSB), "R3A", lambda kk, t, k=k: f"N_E{t}" if kk == k else "x"),
                            R3F_START_k=agree(np.arange(nSB), "R3F", lambda kk, t, k=k: f"N_S{kk}" if kk == k else "x"),
                            R3_START_t=agree(np.arange(nSB), "R3", lambda kk, t, k=k: f"N_S{t}" if kk == k else "x")) for k in JB["ks"]})
res["descriptive"] = desc
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"]["all"], AP1=[res["AP1"]["verdict"], iw["AP"], iw["M3"]], AP2=[res["AP2"]["verdict"], ip["AP"], ip["M3"]],
                      AP3=[res["AP3"]["verdict"], d3["est"]], AP4=[res["AP4"]["verdict"], d4["est"]],
                      RB1=[res["RB1"]["verdict"], b1["est"]], RB2=[res["RB2"]["verdict"], b2["est"]], RB3=[res["RB3"]["verdict"], b3["est"]]), indent=1))
print("->", OUT)
