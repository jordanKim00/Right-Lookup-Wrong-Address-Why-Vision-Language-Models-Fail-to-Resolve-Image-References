"""S913 analysis (rules fixed before the data; knowledge/DECISIONS.md S913 entry).  Inputs results/r3/s913_<tag>_grid.json for N 5, 6 (unseen
by the frozen models) and N 8 (fresh scenes).  Frozen predictors (main/s913_frozen_models.json, fitted 10-22 on the 1,360 registered natural END
runs of S892 / S899 / S902 / S906 / S909 / S910 at N 7-9; hash checked):
  PROFILE - conditional logit over positions b with features z(b-1), z(b), z(b+1), z(b+2), z(b+3) of the run's own L47 matcher-mean early-
            address profile (rows B1 / B4 / answer, standardised within the run; -3 outside 1..N), last-picture and copy (b = k) indicators;
            weights per k group (1, 2, 3, >= 4);
  NOSHIFT - the same without z(b+1..3) (no one-back support);
  OFFSET  - behaviour only: P(x - t | k group) from the training runs (add .5);
  TH2a    - the theorist's route model with the 10-20 frozen parameters at this N (P(other) = TH2a's p_other_by_N at N, else its mean);
  ARGMAX  - the profile's argmax with a .02 floor.
Every distribution is over {other, 1..N} mixed with 1% uniform; LL = mean per-run log-likelihood of the natural END answer.  Scene bootstrap
4,000 within each N (seed 913).
I0: t1 >= .95 and natural START(q) answers q >= .95 (q < N) in every run file; the frozen models' hash matches.  Nothing is read if I0 fails.
P1 (unseen N, pooled N 5 + 6): d1 = LL(PROFILE) - max(LL(TH2a), LL(OFFSET)) >= .05 with the lower bound > 0 -> 'the early-address-profile
  model predicts single natural END answers at unseen N better than the route model and the behaviour-only model'; d1 <= 0 with the upper bound
  < .05 -> 'not'; else 'partial'.
P2 (the one-back support term, pooled N 5 + 6 + 8): LL(PROFILE) - LL(NOSHIFT) >= .05 with the lower bound > 0 -> 'the one-back support term
  carries predictive information on new data'; else 'not'.
P3 (fresh scenes at a seen N, N 8): LL(PROFILE) - max(LL(TH2a), LL(OFFSET)) > 0 with the lower bound > 0 -> 'holds on fresh N8 scenes'; else
  'not'.
Descriptive: per (N, k) every model's LL, the observed accuracy and each model's mean P(target), and the cell-level TV of each model's averaged
distribution against the observed one.
Usage: python3 s913_analyze.py TAG_N5 TAG_N6 TAG_N8"""
from __future__ import annotations

import collections, hashlib, importlib.util, json, sys
import numpy as np

sys.dont_write_bytecode = True
TAGS = sys.argv[1:4]
AG = "/vlm/research/ces/outputs/writer_circuit_2026-10-20"
FM = f"{AG}/main/s913_frozen_models.json"; HASH = "04ba944bcb6acfd6"
OUT = f"/vlm/results/r3/s913_analysis_{TAGS[0]}.json"
rng = np.random.default_rng(913); NB = 4000
M = json.load(open(FM)); match = hashlib.sha256(open(FM, "rb").read()).hexdigest()[:16] == HASH
spec = importlib.util.spec_from_file_location("t5", f"{AG}/a co-analysis/t5_s862_predict.py"); T5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(T5)
THP = np.asarray(json.load(open(f"{AG}/main/s870_frozen_TH2a_N5N7.json"))["params_raw"])
PO = json.load(open(f"{AG}/a co-analysis/t5_s862_predictions_TH2a.json"))["p_other_by_N"]
KG = lambda k: 0 if k == 1 else 1 if k == 2 else 2 if k == 3 else 3
JS = (-1, 0, 1, 2, 3)


def feats(z, N, k):
    zn = (z - z.mean()) / (z.std() + 1e-6); F = np.zeros((N, len(JS) + 2))
    for b in range(1, N + 1):
        for i, j in enumerate(JS):
            bb = b + j; F[b - 1, i] = zn[bb - 1] if 1 <= bb <= N else -3.0
        F[b - 1, len(JS)] = float(b == N); F[b - 1, len(JS) + 1] = float(b == k)
    return F


def mix(p, N):                                                                       # p over {other, 1..N}
    p = np.clip(np.asarray(p, float), 0, None); p = p / p.sum(); return .99 * p + .01 / (N + 1)


def preds(z, N, k):
    F = feats(z, N, k); out = {}
    for nm in ("PROFILE", "NOSHIFT"):
        W = np.asarray(M[nm]["W"])[KG(k)]; lg = F @ W; lg = lg - lg.max(); p = np.exp(lg) / np.exp(lg).sum()
        out[nm] = mix(np.concatenate([[0.0], p]), N)
    c = M["OFFSET"][str(KG(k))]; t = N + 1 - k
    p = np.array([c.get(str(b - t), 0) + .5 for b in range(1, N + 1)], float); out["OFFSET"] = mix(np.concatenate([[0.0], p / p.sum()]), N)
    po = float(PO.get(str(N), np.mean([float(v) for v in PO.values()])))
    th = np.asarray(T5.th2(THP, N, k), float); th = th / th.sum(); out["TH2a"] = mix(np.concatenate([[po], (1 - po) * th]), N)
    a = int(np.argmax(z)) + 1; p = np.full(N, .02); p[a - 1] = 1.0; out["ARGMAX"] = mix(np.concatenate([[0.0], p / p.sum()]), N)
    return out


MODELS = ("PROFILE", "NOSHIFT", "OFFSET", "TH2a", "ARGMAX")
REC = []; FILES = {}
for tag in TAGS:
    J = json.load(open(f"/vlm/results/r3/s913_{tag}_grid.json")); FILES[tag] = J; N = J["N"]
    for si, r in enumerate(J["rows"]):
        for k in J["ks"]:
            key = f"N_E{k}"
            z = np.asarray(r["qk47"][key])[[2, 5, 9]].mean(0); x = int(r["land"][key]); xi = x if 1 <= x <= N else 0
            P = preds(z, N, k)
            REC.append(dict(tag=tag, N=N, k=k, si=si, x=x, P={m: P[m] for m in MODELS}, ll={m: float(np.log(P[m][xi])) for m in MODELS}))


def boot(fn, tags):
    est = fn({tg: np.arange(len(FILES[tg]["rows"])) for tg in tags}); bs = []
    for _ in range(NB):
        bs.append(fn({tg: rng.integers(0, len(FILES[tg]["rows"]), len(FILES[tg]["rows"])) for tg in tags}))
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return dict(est=float(est), lo=float(lo), hi=float(hi))


def LL(ixs, m):
    v = []
    for tg, ix in ixs.items():
        cnt = collections.Counter(int(i) for i in ix)
        v += [d["ll"][m] for d in REC if d["tag"] == tg for _ in range(cnt.get(d["si"], 0))]
    return float(np.mean(v))


res = dict(tags=TAGS, frozen_match=match)
t1 = {tg: FILES[tg]["t1_menu"] for tg in TAGS}
st = {tg: float(np.mean([r["land"][f"N_S{q}"] == q for r in FILES[tg]["rows"] for q in range(1, FILES[tg]["N"])])) for tg in TAGS}
I0 = dict(t1=bool(all((v or 0) >= .95 for v in t1.values())), START=bool(all(v >= .95 for v in st.values())), frozen=bool(match))
res["I0"] = dict(**I0, all=bool(all(I0.values())), t1_values=t1, START_acc=st)
ok = res["I0"]["all"]
U = TAGS[:2]; ALL = TAGS
d1 = boot(lambda ixs: LL(ixs, "PROFILE") - max(LL(ixs, "TH2a"), LL(ixs, "OFFSET")), U)
res["P1"] = dict(d=d1, LL={m: LL({tg: np.arange(len(FILES[tg]["rows"])) for tg in U}, m) for m in MODELS},
                 verdict=(("the early-address-profile model predicts single natural END answers at unseen N better than the route model and the behaviour-only model"
                           if d1["est"] >= .05 and d1["lo"] > 0 else "not" if d1["est"] <= 0 and d1["hi"] < .05 else "partial") if ok else "not read"))
d2 = boot(lambda ixs: LL(ixs, "PROFILE") - LL(ixs, "NOSHIFT"), ALL)
res["P2"] = dict(d=d2, verdict=(("the one-back support term carries predictive information on new data" if d2["est"] >= .05 and d2["lo"] > 0 else "not") if ok else "not read"))
d3 = boot(lambda ixs: LL(ixs, "PROFILE") - max(LL(ixs, "TH2a"), LL(ixs, "OFFSET")), [TAGS[2]])
res["P3"] = dict(d=d3, LL={m: LL({TAGS[2]: np.arange(len(FILES[TAGS[2]]["rows"]))}, m) for m in MODELS},
                 verdict=(("holds on fresh N8 scenes" if d3["est"] > 0 and d3["lo"] > 0 else "not") if ok else "not read"))
desc = {}
for tg in TAGS:
    N = FILES[tg]["N"]
    for k in FILES[tg]["ks"]:
        ds = [d for d in REC if d["tag"] == tg and d["k"] == k]; t = N + 1 - k
        obs = np.zeros(N + 1)
        for d in ds:
            obs[d["x"] if 1 <= d["x"] <= N else 0] += 1
        obs /= obs.sum()
        desc[f"N{N}_k{k}"] = dict(n=len(ds), acc=float(obs[t]), LL={m: float(np.mean([d["ll"][m] for d in ds])) for m in MODELS},
                                  P_target={m: float(np.mean([d["P"][m][t] for d in ds])) for m in MODELS},
                                  TV={m: float(.5 * np.abs(np.mean([d["P"][m] for d in ds], 0) - obs).sum()) for m in MODELS})
res["descriptive"] = desc
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"]["all"], P1=[res["P1"]["verdict"], d1["est"]], P2=[res["P2"]["verdict"], d2["est"]], P3=[res["P3"]["verdict"], d3["est"]],
                      LL_unseen=res["P1"]["LL"]), indent=1))
print("->", OUT)
