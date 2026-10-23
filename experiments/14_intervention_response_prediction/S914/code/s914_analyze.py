"""S914 analysis (rules fixed before the data; knowledge/DECISIONS.md S914 entry; the advisor's design, 10-22).  Frozen predictors
main/s913_frozen_predictors.json (fitted by exp/s913_fit.py on the S913 training runs only, N 7 and 8; hash checked).  Test runs
results/r3/s913_<tag>_collect.json for three sets never used in fitting: new N9 scenes ('from the last'), N6 (an image count the predictors
never saw), N8 with the new question form 'from the end'; each with --pin 1 (E, R, E_BANDR, PIN, PIN_BANDR at the natural L47 anchor a).
Measurement boundary: the predictors read the L47 (or L43) early block scores of the read rows - for L47 this already includes MLP46's
influence on FA47's queries; the claim is 'late-lookup prediction from the address-related state at FA47's input', not an upstream writer input.
Per natural END run: answer NLL of every model (M0, M1, M2a, M2b, M3, M3noW, M2a_L43, M3_L43, TH2a) over {other, 1..N} with 1% uniform; M3's
late-address NLL (the natural late argmax at L51); R^2 of M3's B^ and W^ against the run's B and W (per-run centred, pooled).  Departure from
the anchor dep = P(x != a).  Scene bootstrap 4,000 within each set (seed 914); pooled statistics resample all three sets.
I0: t1 >= .95 and natural START(q) answers q >= .95 (q < N) in every set; the frozen hash matches; PIN reaches its anchor (PIN's L47 anchor = a in
  >= .90); every set has >= 200 natural END runs.  Nothing is read if I0 fails.
P1 (profile beyond argmax): NLL(M1) - NLL(M2a), pooled, >= .05 with the lower bound > 0 -> 'the full early-address profile predicts answers
  beyond its argmax'; else 'not'.
P2 (structured writer model): d = NLL(M2a) - NLL(M3), pooled: d >= .02 with the lower bound > 0 -> 'the structured B^ + W^ model predicts
  answers better than the free profile model'; d >= -.02 with the lower bound > -.05 -> 'as well as the free profile model'; else 'worse'.
P3 (the two components): pooled R^2(B^) >= .30 and R^2(W^) >= .30 -> 'the early profile predicts both the writer-free support and the writer's
  update' (else 'not'); and NLL(M3noW) - NLL(M3) >= .05 with the lower bound > 0 -> 'the predicted update adds to the answer prediction' (else
  'not').
P4 (interventions): writer removal - predicted change P_M3(x != a | E) - P_M3(x != a | E, W^ = 0; M3's own temperatures) against the observed
  dep(E) - dep(E_BANDR);
  initial-address sharpening - P_M3(x != a | E) - P_M3(x != a | PIN's own profile) against dep(E) - dep(PIN).  An intervention is 'predicted'
  if the pooled |predicted - observed| <= .10 and the correlation over (set, k) cells with >= 10 runs is >= .70.  Both -> 'the frozen structured
  model predicts the effect of removing the writer and of sharpening the initial address'; one -> 'partial'; none -> 'not'.
P5 (heads): NLL(M2a) - NLL(M2b), pooled, > 0 with the lower bound > 0 -> 'head-level early profiles add information beyond the head mean'; else
  'not observed'.
P6 (behaviour priors): min(NLL(M0), NLL(TH2a)) - NLL(M2a), pooled, > 0 with the lower bound > 0 -> 'beats the behaviour priors'; else 'not'.
Descriptive: every quantity per set and per (set, k); the L43 variants; PIN_BANDR (M3noW on PIN's profile vs dep(PIN_BANDR)).
Usage: python3 s914_analyze.py TAG_N9 TAG_N6 TAG_E8"""
from __future__ import annotations

import collections, hashlib, importlib.util, json, sys
import numpy as np

sys.dont_write_bytecode = True
TAGS = sys.argv[1:4]
FP = "/vlm/research/ces/outputs/writer_circuit_2026-10-20/main/s913_frozen_predictors.json"
HASH = "db299c27045230be"                                                            # the frozen predictors (DECISIONS, 10-22 13:29)
AG = "/vlm/research/ces/outputs/writer_circuit_2026-10-20"
OUT = f"/vlm/results/r3/s914_analysis_{TAGS[0]}.json"
rng = np.random.default_rng(914); NB = 4000
_argv = sys.argv; sys.argv = ["s913_fit", "x", "y"]
spec = importlib.util.spec_from_file_location("s913_fit", "/vlm/exp/s913_fit.py"); FIT = importlib.util.module_from_spec(spec); spec.loader.exec_module(FIT)
sys.argv = _argv
spec = importlib.util.spec_from_file_location("t5", f"{AG}/a co-analysis/t5_s862_predict.py"); T5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(T5)
THP = np.asarray(json.load(open(f"{AG}/main/s870_frozen_TH2a_N5N7.json"))["params_raw"])
PO = json.load(open(f"{AG}/a co-analysis/t5_s862_predictions_TH2a.json"))["p_other_by_N"]
P = json.load(open(FP)); fhash = hashlib.sha256(open(FP, "rb").read()).hexdigest()[:16]
KG = FIT.KG


def softmax(v):
    v = np.asarray(v, float); v = v - v.max(); e = np.exp(v); return e / e.sum()


def logit_p(d, W, fk):
    return FIT.mix(np.concatenate([[0.0], softmax(fk(d) @ np.asarray(W)[KG(d["k"])])]), d["N"])


def m3_p(d, name, L, T, useW=True, sameT=False):
    """sameT: the writer-removal prediction - B^ alone read out with M3's own temperatures (the readout is not changed by removing the writer);
    otherwise M3noW uses its separately fitted temperatures (the P2 / P3 ablation)."""
    P3 = {int(g): {kk: np.asarray(v) for kk, v in c.items()} for g, c in P[name].items()}
    sc = FIT.m3_scores(d, P3, L, useW); t = P[(name if (useW or sameT) else name.replace("M3", "M3noW")) + "_T"]
    return FIT.mix(np.concatenate([[0.0], softmax(t["ans"][str(KG(d["k"]))] * sc)]), d["N"]), softmax(t["late"] * sc), sc


def offset_p(d, M, ref):
    c = M[str(KG(d["k"]))]; r0 = ref(d); p = np.array([c.get(str(b - r0), 0) + .5 for b in range(1, d["N"] + 1)], float)
    return FIT.mix(np.concatenate([[0.0], p / p.sum()]), d["N"])


def th_p(d):
    N = d["N"]; po = float(PO.get(str(N), np.mean([float(v) for v in PO.values()])))
    th = np.asarray(T5.th2(THP, N, d["k"]), float); th = th / th.sum(); return FIT.mix(np.concatenate([[po], (1 - po) * th]), N)


def load_test(tag):
    J = json.load(open(f"/vlm/results/r3/s913_{tag}_collect.json")); N = J["N"]; lm = J["late_mean"]; em = J["early_mean"]
    out = []
    for si, r in enumerate(J["rows"]):
        for k in J["ks"]:
            if f"B{k}_E" not in r["zl"] or f"B{k}_PIN" not in r["zl"]:
                continue
            a = r["anchor"][str(k)]
            def prof(key, L):
                h = np.asarray(r[f"qk{L}h"][key], float).mean(0); return FIT.centre(h[em[str(L)]].mean(0)), np.stack([FIT.centre(x) for x in h])
            zl = lambda key_: np.asarray(r["zl"][key_]["51"], float)[:, lm["51"]].mean(1).mean(0)
            d = dict(tag=tag, N=N, k=k, si=si, a=a, x=int(r["land"][f"N_E{k}"]))
            for L in (47, 43):
                d[f"e{L}"], d[f"eh{L}"] = prof(f"N_E{k}", L)
            zE, zB = zl(f"B{k}_E"), zl(f"B{k}_E_BANDR"); d["B"] = FIT.centre(zB); d["W"] = FIT.centre(zE - zB); d["late"] = int(np.argmax(zE)) + 1
            pin = dict(d); pin["e47"], pin["eh47"] = prof(f"B{k}_PIN", 47); pin["e43"], pin["eh43"] = prof(f"B{k}_PIN", 43)
            d["pin"] = pin
            d["x_BANDR"] = int(r["land"][f"B{k}_E_BANDR"]); d["x_PIN"] = int(r["land"][f"B{k}_PIN"]); d["x_PINB"] = int(r["land"][f"B{k}_PIN_BANDR"])
            q47 = np.asarray(r["qk47"][f"B{k}_PIN"]); am = [int(np.argmax(q47[i])) + 1 for i in (2, 5, 9)]
            d["pin_anchor"] = collections.Counter(am).most_common(1)[0][0] if collections.Counter(am).most_common(1)[0][1] >= 2 else am[2]
            out.append(d)
    return J, out


FILES, REC = {}, []
for tag in TAGS:
    J, rs = load_test(tag); FILES[tag] = J; REC += rs
MODELS = ("M0", "M1", "M2a", "M2b", "M3", "M3noW", "M2a_L43", "M3_L43", "TH2a")
for d in REC:
    p = {}
    p["M0"] = offset_p(d, P["M0"], lambda d_: d_["N"] + 1 - d_["k"]); p["M1"] = offset_p(d, P["M1"], lambda d_: d_["a"])
    p["M2a"] = logit_p(d, P["M2a"], lambda d_: FIT.fmean(d_, 47)); p["M2b"] = logit_p(d, P["M2b"], lambda d_: FIT.fhead(d_, 47))
    p["M2a_L43"] = logit_p(d, P["M2a_L43"], lambda d_: FIT.fmean(d_, 43))
    p["M3"], late, sc = m3_p(d, "M3", 47, None); p["M3noW"], lateB, scB = m3_p(d, "M3", 47, None, useW=False)
    p["M3_L43"], _, _ = m3_p(d, "M3_L43", 43, None); p["TH2a"] = th_p(d)
    j = FIT.xi(d); d["nll"] = {m: float(-np.log(p[m][j])) for m in MODELS}
    d["nll_late"] = float(-np.log(.99 * late[d["late"] - 1] + .01 / d["N"])); d["nll_lateB"] = float(-np.log(.99 * lateB[d["late"] - 1] + .01 / d["N"]))
    P3 = {int(g): {kk: np.asarray(v) for kk, v in c.items()} for g, c in P["M3"].items()}
    X = np.concatenate([FIT.f3(d, 47), np.ones((d["N"], 1))], 1); c = P3[KG(d["k"])]
    d["Bhat"] = X @ c["B"]; d["What"] = X @ c["W"]
    a = d["a"]; dep = lambda pp: float(1 - pp[a])                                    # pp over {other, 1..N}: P(x != a)
    pB_same, _, _ = m3_p(d, "M3", 47, None, useW=False, sameT=True)
    d["pdep"] = dict(E=dep(p["M3"]), BANDR=dep(pB_same))
    pp_pin, _, _ = m3_p(d["pin"], "M3", 47, None); pp_pinB, _, _ = m3_p(d["pin"], "M3", 47, None, useW=False, sameT=True)
    d["pdep"]["PIN"] = dep(pp_pin); d["pdep"]["PINB"] = dep(pp_pinB)
    d["odep"] = dict(E=float(d["x"] != a), BANDR=float(d["x_BANDR"] != a), PIN=float(d["x_PIN"] != a), PINB=float(d["x_PINB"] != a))


def boot(fn, tags):
    est = fn({tg: np.arange(len(FILES[tg]["rows"])) for tg in tags}); bs = []
    for _ in range(NB):
        bs.append(fn({tg: rng.integers(0, len(FILES[tg]["rows"]), len(FILES[tg]["rows"])) for tg in tags}))
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return dict(est=float(est), lo=float(lo), hi=float(hi))


def sel(ixs):
    out = []
    for tg, ix in ixs.items():
        cnt = collections.Counter(int(i) for i in ix)
        out += [d for d in REC if d["tag"] == tg for _ in range(cnt.get(d["si"], 0))]
    return out


NLL = lambda ixs, m: float(np.mean([d["nll"][m] for d in sel(ixs)]))
res = dict(tags=TAGS, frozen_hash=fhash)
t1 = {tg: FILES[tg]["t1_menu"] for tg in TAGS}
st = {tg: float(np.mean([r["land"][f"N_S{q}"] == q for r in FILES[tg]["rows"] for q in range(1, FILES[tg]["N"])])) for tg in TAGS}
reach = float(np.mean([d["pin_anchor"] == d["a"] for d in REC])); nper = {tg: len([d for d in REC if d["tag"] == tg]) for tg in TAGS}
I0 = dict(t1=bool(all((v or 0) >= .95 for v in t1.values())), START=bool(all(v >= .95 for v in st.values())),
          frozen=bool(HASH is None or fhash == HASH), pin_reach=bool(reach >= .90), size=bool(all(v >= 200 for v in nper.values())))
res["I0"] = dict(**I0, all=bool(all(I0.values())), t1_values=t1, START_acc=st, pin_reach_value=reach, n_runs=nper)
ok = res["I0"]["all"]
b1 = boot(lambda ixs: NLL(ixs, "M1") - NLL(ixs, "M2a"), TAGS)
res["P1"] = dict(d=b1, verdict=(("the full early-address profile predicts answers beyond its argmax" if b1["est"] >= .05 and b1["lo"] > 0 else "not") if ok else "not read"))
b2 = boot(lambda ixs: NLL(ixs, "M2a") - NLL(ixs, "M3"), TAGS)
res["P2"] = dict(d=b2, verdict=(("the structured B^ + W^ model predicts answers better than the free profile model" if b2["est"] >= .02 and b2["lo"] > 0 else
                                 "as well as the free profile model" if b2["est"] >= -.02 and b2["lo"] > -.05 else "worse") if ok else "not read"))


def r2(key, tkey):
    yt = np.concatenate([d[tkey] for d in REC]); yp = np.concatenate([d[key] for d in REC]); return float(1 - np.sum((yt - yp) ** 2) / np.sum(yt ** 2))


rB, rW = r2("Bhat", "B"), r2("What", "W")
b3 = boot(lambda ixs: NLL(ixs, "M3noW") - NLL(ixs, "M3"), TAGS)
res["P3"] = dict(R2_B=rB, R2_W=rW, d_noW=b3,
                 verdict_components=(("the early profile predicts both the writer-free support and the writer's update" if rB >= .30 and rW >= .30 else "not") if ok else "not read"),
                 verdict_update=(("the predicted update adds to the answer prediction" if b3["est"] >= .05 and b3["lo"] > 0 else "not") if ok else "not read"))
cells = collections.defaultdict(list)
for d in REC:
    cells[(d["tag"], d["k"])].append(d)
def interv(obs_key, pred_a, pred_b):
    pred = np.mean([d["pdep"][pred_a] - d["pdep"][pred_b] for d in REC]); obs = np.mean([d["odep"]["E"] - d["odep"][obs_key] for d in REC])
    cp, co = [], []
    for c, ds in cells.items():
        if len(ds) >= 10:
            cp.append(np.mean([d["pdep"][pred_a] - d["pdep"][pred_b] for d in ds])); co.append(np.mean([d["odep"]["E"] - d["odep"][obs_key] for d in ds]))
    r = float(np.corrcoef(cp, co)[0, 1]) if len(cp) >= 3 else float("nan")
    return dict(pred=float(pred), obs=float(obs), cell_r=r, n_cells=len(cp), ok=bool(abs(pred - obs) <= .10 and r >= .70))
iw = interv("BANDR", "E", "BANDR"); ip = interv("PIN", "E", "PIN")
nok = int(iw["ok"]) + int(ip["ok"])
res["P4"] = dict(writer_removal=iw, sharpening=ip, verdict=(("the frozen structured model predicts the effect of removing the writer and of sharpening the initial address" if nok == 2 else
                                                              "partial" if nok == 1 else "not") if ok else "not read"))
b5 = boot(lambda ixs: NLL(ixs, "M2a") - NLL(ixs, "M2b"), TAGS)
res["P5"] = dict(d=b5, verdict=(("head-level early profiles add information beyond the head mean" if b5["est"] > 0 and b5["lo"] > 0 else "not observed") if ok else "not read"))
b6 = boot(lambda ixs: min(NLL(ixs, "M0"), NLL(ixs, "TH2a")) - NLL(ixs, "M2a"), TAGS)
res["P6"] = dict(d=b6, verdict=(("beats the behaviour priors" if b6["est"] > 0 and b6["lo"] > 0 else "not") if ok else "not read"))
desc = {}
for tg in TAGS:
    ds = [d for d in REC if d["tag"] == tg]
    desc[tg] = dict(N=FILES[tg]["N"], phrase=FILES[tg].get("phrase"), n=len(ds), nll={m: float(np.mean([d["nll"][m] for d in ds])) for m in MODELS},
                    nll_late=dict(M3=float(np.mean([d["nll_late"] for d in ds])), M3noW=float(np.mean([d["nll_lateB"] for d in ds]))),
                    dep_obs={kk: float(np.mean([d["odep"][kk] for d in ds])) for kk in ("E", "BANDR", "PIN", "PINB")},
                    dep_pred={kk: float(np.mean([d["pdep"][kk] for d in ds])) for kk in ("E", "BANDR", "PIN", "PINB")},
                    by_k={str(k): dict(n=len([d for d in ds if d["k"] == k]), acc=float(np.mean([d["x"] == d["N"] + 1 - k for d in ds if d["k"] == k])),
                                       nll={m: float(np.mean([d["nll"][m] for d in ds if d["k"] == k])) for m in MODELS},
                                       dep_obs={kk: float(np.mean([d["odep"][kk] for d in ds if d["k"] == k])) for kk in ("E", "BANDR", "PIN")},
                                       dep_pred={kk: float(np.mean([d["pdep"][kk] for d in ds if d["k"] == k])) for kk in ("E", "BANDR", "PIN")})
                          for k in FILES[tg]["ks"] if any(d["k"] == k for d in ds)})
res["descriptive"] = desc
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"]["all"], P1=[res["P1"]["verdict"], b1["est"]], P2=[res["P2"]["verdict"], b2["est"]],
                      P3=[res["P3"]["verdict_components"], rB, rW, res["P3"]["verdict_update"], b3["est"]], P4=[res["P4"]["verdict"], iw, ip],
                      P5=[res["P5"]["verdict"], b5["est"]], P6=[res["P6"]["verdict"], b6["est"]]), indent=1))
print("->", OUT)
