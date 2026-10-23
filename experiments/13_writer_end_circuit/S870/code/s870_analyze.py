"""S870 analysis (rules fixed before the data; knowledge/DECISIONS.md S870 entry).  Input results/r3/s870_<tag>_grid.json.
Frozen files (hashes checked against main/s870_frozen_hashes.json):
  TH2a at N 5 / 7 (main/s870_frozen_TH2a_N5N7.json: the theorist's route model - end route with bounded reach, copy route with one band step
  at k >= 3, start route - with the TH2a parameters frozen on 10-20);  the adversary's s862_frozen.py (R7, MU, POINTER, COPY, UNIFORM, CIRC with
  anchor = None);  F_early (main/s870_frozen_Fearly.json: per-k weights of END(k)'s early profile on [START(k), START(N), START(N-1),
  START(N-2), START(N-3)], fitted on S862, k <= N - 1).
Answers x over {other, 1..N}; every model's distribution mixed with 1% uniform; scene bootstrap 4,000 (seed 870).
I0: t1 >= .95; START accuracy (all q, N 5 and 7) >= .95; frozen hashes match.  Nothing is read if I0 fails.
B (behaviour, every END(k), k = 1..N, at N 5 and 7 - the user's question):
  B1 mean TV over the 12 cells: TH2a vs the best of R7 / MU: 'the circuit-structured route model predicts every END cell at N 5 and 7 better
     than the best behaviour-only fit' if TV(TH2a) - min(TV(R7), TV(MU)) <= -.03 with the interval below 0; 'at least as well' if <= +.01;
     else 'not'.
  B2 mean per-run log-likelihood: TH2a - max(R7, MU) > 0 with the interval above 0 'better per run', else 'not'.
  B3 (descriptive): per cell observed accuracy and TH2a's P(target); predicted vs observed mode; N 5 (unseen N) reported separately.
F (internal, the early address law; cells k <= N - 1):
  F1 pooled R^2 of END(k)'s L47 early profile (rows B1 / B4 / answer averaged, centred) predicted from the same scene's START profiles with the
     frozen weights >= .50, and R^2(F_early) - R^2(copy-only frozen weight) >= .15 with the interval above 0 -> 'the early address is a lawful
     mixture of copy and end-offset routes (frozen F_early generalises to new scenes and an unseen N)'; else 'not'.
  F2 argmax of the predicted profile = the measured L47 anchor (majority of B1 / B4 / answer, answer row if all differ) in >= .60 of runs, and
     >= the copy-only anchor (k) agreement + .10 with the interval above 0 -> 'F_early predicts the early anchor'; else 'not'.
  N 5 and the k = N cells reported separately (descriptive).
Usage: python3 s870_analyze.py TAG"""
from __future__ import annotations

import collections, hashlib, importlib.util, json, sys
import numpy as np

sys.dont_write_bytecode = True
TAG = sys.argv[1]
J = json.load(open(f"/vlm/results/r3/s870_{TAG}_grid.json"))
OUT = f"/vlm/results/r3/s870_analysis_{TAG}.json"
rng = np.random.default_rng(870)
NB = 4000
AG = "/vlm/research/ces/outputs/writer_circuit_2026-10-20"
rows = J["rows"]; S = len(rows); LAYS = J["lays"]
E47 = str(J["early"][-1])
FILES = {"TH2a_N5N7": f"{AG}/main/s870_frozen_TH2a_N5N7.json", "Fearly": f"{AG}/main/s870_frozen_Fearly.json",
         "adversary": f"{AG}/a co-analysis/predictions/s862_frozen.py", "adversary_params": f"{AG}/a co-analysis/predictions/s862_params.json"}
REG = json.load(open(f"{AG}/main/s870_frozen_hashes.json"))
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
match = {k: sha(p) == REG.get(k) for k, p in FILES.items()}


def bt(fn):
    est = fn(np.arange(S)); bs = [fn(rng.integers(0, S, S)) for _ in range(NB)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return dict(est=float(est), lo=float(lo), hi=float(hi))


def land(key, N):
    x = np.array([r["land"][key] for r in rows]); return np.where((x >= 1) & (x <= N), x, 0)


res = dict(tag=TAG, n_scenes=S, n_runs=J["n_runs"], frozen_match=match)
st = float(np.mean([np.mean(land(f"G{N}_START{q}", N) == q) for N in LAYS for q in range(1, N + 1)]))
I0 = dict(t1=bool((J["t1_menu"] or 0) >= .95), START_acc=bool(st >= .95), frozen=bool(all(match.values())))
res["I0"] = dict(**I0, all=bool(all(I0.values())), t1_value=J["t1_menu"], START_acc_value=st)
ok = res["I0"]["all"]
# ---------------------------------------------------------------- B
TH = json.load(open(FILES["TH2a_N5N7"]))["cells"]
spec = importlib.util.spec_from_file_location("s862_frozen", FILES["adversary"]); ADV = importlib.util.module_from_spec(spec); spec.loader.exec_module(ADV)
def mix(p, N):
    p = np.clip(np.asarray(p, dtype=float), 0, None); p = p / p.sum(); return .99 * p + .01 / (N + 1)
CELLS = [(N, k) for N in LAYS for k in range(1, N + 1)]
PRED = {"TH2a": {c: mix(TH[f"N{c[0]}_END{c[1]}"]["prob_other_then_pictures_1toN"], c[0]) for c in CELLS}}
for m in ("R7", "MU", "POINTER", "COPY", "UNIFORM", "CIRC"):
    PRED[m] = {c: mix(ADV.predict(m, c[0], c[1], "END", "E", None, None), c[0]) for c in CELLS}
X = {c: land(f"G{c[0]}_END{c[1]}", c[0]) for c in CELLS}
def obs(c, ix):
    cnt = np.bincount(X[c][ix], minlength=c[0] + 1)[:c[0] + 1].astype(float); return cnt / cnt.sum()
def tv(m, ix, cells=CELLS):
    return float(np.mean([.5 * np.abs(PRED[m][c] - obs(c, ix)).sum() for c in cells]))
def ll(m, ix, cells=CELLS):
    return float(np.mean(np.concatenate([np.log(PRED[m][c][X[c][ix]]) for c in cells])))
ix0 = np.arange(S)
res["B"] = dict(TV={m: tv(m, ix0) for m in PRED}, LL={m: ll(m, ix0) for m in PRED},
                TV_N5={m: tv(m, ix0, [c for c in CELLS if c[0] == 5]) for m in PRED}, TV_N7={m: tv(m, ix0, [c for c in CELLS if c[0] == 7]) for m in PRED})
b1 = bt(lambda ix: tv("TH2a", ix) - min(tv("R7", ix), tv("MU", ix)))
b2 = bt(lambda ix: ll("TH2a", ix) - max(ll("R7", ix), ll("MU", ix)))
res["B"]["B1"] = dict(diff=b1, verdict=(("the circuit-structured route model predicts every END cell at N 5 and 7 better than the best behaviour-only fit"
                                         if b1["est"] <= -.03 and b1["hi"] < 0 else "at least as well" if b1["est"] <= .01 else "not") if ok else "not read"))
res["B"]["B2"] = dict(diff=b2, verdict=(("better per run" if b2["est"] > 0 and b2["lo"] > 0 else "not") if ok else "not read"))
res["B"]["B3"] = {f"N{N}_END{k}": dict(target=N + 1 - k, observed_acc=float(np.mean(X[(N, k)] == N + 1 - k)),
                                       TH2a_P_target=float(PRED["TH2a"][(N, k)][N + 1 - k]),
                                       observed_mode=int(np.argmax(obs((N, k), ix0)[1:])) + 1, TH2a_mode=int(np.argmax(PRED["TH2a"][(N, k)][1:])) + 1,
                                       observed={str(v): float(x) for v, x in enumerate(obs((N, k), ix0)) if x >= .04})
                     for (N, k) in CELLS}
# ---------------------------------------------------------------- F
FE = json.load(open(FILES["Fearly"]))["layers"]["47"]
def prof(r, key):
    q = np.asarray(r["qk"][key][E47])[[2, 5, 9]].mean(0); return q - q.mean()
def anchor(r, key):
    q = np.asarray(r["qk"][key][E47]); am = [int(np.argmax(q[i])) + 1 for i in (2, 5, 9)]
    c = collections.Counter(am).most_common(); return c[0][0] if c[0][1] >= 2 else am[2]
RUNS = []
for N in LAYS:
    for k in range(1, N + 1):
        for si, r in enumerate(rows):
            P = prof(r, f"G{N}_END{k}")
            regs = [prof(r, f"G{N}_START{k}")] + [prof(r, f"G{N}_START{N - o}") if N - o >= 1 else np.zeros(N) for o in range(4)]
            w = np.asarray(FE["W"].get(str(k), FE["W"][max(FE["W"], key=int)]))
            pred = np.stack(regs, 1) @ w; pc = FE["W_copy"].get(str(k), 0.0) * regs[0]
            RUNS.append(dict(N=N, k=k, si=si, P=P, pred=pred, pc=pc, a=anchor(r, f"G{N}_END{k}"), ap=int(np.argmax(pred)) + 1))
BY = collections.defaultdict(list)
for q in RUNS:
    BY[q["si"]].append(q)
def sel(ix, keep):
    return [q for s in ix for q in BY[int(s)] if keep(q)]
def r2(ix, key="pred", keep=lambda q: q["k"] <= q["N"] - 1):
    R = sel(ix, keep); P = np.concatenate([q["P"] for q in R]); Q = np.concatenate([q[key] for q in R])
    return float(1 - np.sum((P - Q) ** 2) / np.sum(P ** 2))
f1 = bt(lambda ix: r2(ix)); f1d = bt(lambda ix: r2(ix) - r2(ix, "pc"))
res["F"] = dict(F1=dict(R2=f1, R2_minus_copy=f1d, R2_copy=bt(lambda ix: r2(ix, "pc")),
                        R2_N5=r2(ix0, keep=lambda q: q["N"] == 5 and q["k"] <= 4), R2_N7=r2(ix0, keep=lambda q: q["N"] == 7 and q["k"] <= 6),
                        R2_kN=r2(ix0, keep=lambda q: q["k"] == q["N"]),
                        R2_by_k={f"N{N}_k{k}": r2(ix0, keep=lambda q, N=N, k=k: q["N"] == N and q["k"] == k) for N in LAYS for k in range(1, N + 1)},
                        verdict=(("the early address is a lawful mixture of copy and end-offset routes (frozen F_early generalises to new scenes and an unseen N)"
                                  if f1["est"] >= .50 and f1d["est"] >= .15 and f1d["lo"] > 0 else "not") if ok else "not read")))
def agree(ix, which):
    R = sel(ix, lambda q: q["k"] <= q["N"] - 1)
    return float(np.mean([(q["ap"] if which == "F" else q["k"]) == q["a"] for q in R]))
f2 = bt(lambda ix: agree(ix, "F")); f2d = bt(lambda ix: agree(ix, "F") - agree(ix, "copy"))
res["F"]["F2"] = dict(agree=f2, minus_copy=f2d, agree_copy=agree(ix0, "copy"),
                      by_k={f"N{N}_k{k}": float(np.mean([q["ap"] == q["a"] for q in RUNS if q["N"] == N and q["k"] == k])) for N in LAYS for k in range(1, N + 1)},
                      verdict=(("F_early predicts the early anchor" if f2["est"] >= .60 and f2d["est"] >= .10 and f2d["lo"] > 0 else "not") if ok else "not read"))
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"]["all"], B1=res["B"]["B1"]["verdict"], B2=res["B"]["B2"]["verdict"], TV=res["B"]["TV"],
                      F1=res["F"]["F1"]["verdict"], F2=res["F"]["F2"]["verdict"]), indent=1))
print("->", OUT)
