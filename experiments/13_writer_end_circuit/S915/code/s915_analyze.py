"""S915 analysis (rules fixed before the data; knowledge/DECISIONS.md S915 entry; the advisor's sufficiency demand, 10-22).  Inputs
results/r3/s915_<tag>_cross.json: set A (N8, 'from the last' and 'from the end', every k crossed, P43 arms) and set B (N9, 'from the last',
neighbouring k crossed).  Frozen predictors main/s913_frozen_predictors.json (S913 training runs only; hash checked).  Pins replace every
head's read-row query at L35, 39, 43, 47 (P47) or L35, 39, 43 (P43) by a donor's (exp/s915_cross.py, part='all': the early lookup is held).
Per run: answer x over {other, 1..N}; the achieved early profile e_L (L = 47, 43: matcher-head mean of rows B1 / B4 / answer, centred over
blocks, magnitudes kept) and its anchor a (majority of the three rows' argmax, as in S913); M3 predictions use the run's achieved profile and
anchor with a stated k.  Scene bootstrap 4,000 within each set (seed 915); pooled statistics resample both sets.
I0: t1 >= .95 and natural START(q) answers q >= .95 (q < N) in each set; the frozen hash matches; self-pins reproduce the natural answer
  (P(x(SP) = x(N)) >= .95 in each set); the P47 pins hold the donor's early lookup within the instrument's batch noise (amended 10-22 after the
  advisor's review, before the data): at every pinned layer (35, 39, 43, 47) the median over P47 crossed runs of the run's max |difference| of
  the centred per-head block scores of the query actually used (rows B1 / B4 / answer) from its donor's is <= 1.5 x the same median for the
  natural re-runs NR against their first runs, in all P47 crossed runs and in the small-margin half (donor's L47 top-1 minus top-2 below the
  set's median); the anchor agreement (all runs, decisive donors, each margin half) is reported, not used.  Nothing is read if I0 fails.
All verdicts use every run; each statistic is also reported in the two halves of the donor's initial-address margin.
What the P47 pin holds (smoke 416a8bb / bc): the read rows' query at L35-47, hence every head's scores over every picture token (block
scores equal the donor's within the re-run noise); the scores over the question text are the donor's query against the state's own text
keys, so the share of attention going to pictures vs text follows the state (max picture-mass difference ~.2-.36 vs ~.015 for re-runs;
reported per layer).  Verdicts are worded 'early lookup' for this held picture-score profile.
S1 (phrase, set A): with the early address held, does the phrasing's remaining state change the answer?  delta_S = mean over (scene, k) of
  1/2 [1(x(XP_X k) != x(SP_E k)) + 1(x(XP_E k) != x(SP_X k))]; delta_S <= .05 with the upper bound < .10 -> 'with the early lookup held, the
  phrasing's remaining state leaves the answer unchanged'; delta_S >= .15 with the lower bound > .10 -> 'the phrasing's remaining state changes
  the answer at a fixed early lookup'; else 'partial'.  Reported with it: the profile effect at a fixed state delta_P (1/2 [1(x(XP_E k) !=
  x(SP_E k)) + 1(x(XP_X k) != x(SP_X k))]) and the natural difference delta_nat = P(x(N_X k) != x(N_E k)).
K0 (k crossing, set A, model-free): END(k2)'s state with END(k)'s early address.  Three outcomes - the donor's own (x(SP_E k)), the state's own
  (x(SP_E k2)), and the composition (the donor's anchor a_k minus the state's own step a_k2 - x(SP_E k2)); over runs where the three are
  pairwise distinct, F_comp, F_donor, F_state = shares of crossed answers equal to each.  The largest share with its difference to the second
  >= .10 and the lower bound > 0 names the verdict: 'the crossed answer composes the early address with the state's step' / 'follows the early
  address' / 'follows the remaining state'; else 'mixed'.
K1 (the state's k, set A KC): NLL(M3 with the donor's k) - NLL(M3 with the state's k) >= .10 with the lower bound > 0 -> 'the state's k acts at
  a fixed early address as the frozen structured model predicts'; <= 0 with the upper bound < .05 -> 'no effective action of the state's k
  beyond the early address observed'; else 'partial'.
K2 (the held address, set A KC): NLL(M3 on the state's own natural profile) - NLL(M3 on the achieved profile), both with the state's k, >= .10
  with the lower bound > 0 -> 'the held early address acts as the frozen model predicts'; else 'not'.
K3 (direction and size, set A KC): per (k2, k) cell, the observed change of the mean answer position (in-range answers; KC run minus the
  state's self-pin, same scene) against M3's predicted change (expected position over 1..N on the achieved KC profile minus on the self-pin's
  profile, state's k); over cells with >= 10 runs: Pearson r >= .80 and the slope of observed on predicted in [.67, 1.5] -> 'the frozen model
  predicts the direction and size of the change from swapping the early address'; r >= .60 -> 'partial'; else 'not'.
C (frame, sets A + B pooled, k >= 2): dep_E = P(x(SP_E k) != a_k), dep_S = P(x(FC_S k) != a_k) (a_k = END(k)'s natural anchor; FC_S k =
  START(a_k) with END(k)'s early lookup); dep_E - dep_S >= .30 with the lower bound > .15 -> 'at the same early lookup the END state departs
  and the START state stays: the departure is END-conditioned'; <= .10 with the upper bound < .20 -> 'no effective END-conditioning observed';
  else 'partial'.
U1 (address part vs full early lookup, both sets): agreement of the answers under the address-part pin (P47U) and the full-query pin (P47) for
  the same state and donor - crossed pairs XPU / XP, KCU / KC (|k - k2| = 1), FCU / FC; AG >= .90 with the lower bound > .85 -> 'holding only
  the address part reproduces the outcome of holding the full early lookup'; AG <= .75 with the upper bound < .85 -> 'the early query's
  non-address part changes the outcome'; else 'partial'; the self-pin agreement SPU / SP is reported as the baseline.
Descriptive: per k every rate; identity runs (a_k = N + 1 - k); FC_E (END(k) with START(a_k)'s address) against SP_S; set B KC; P43 arms
  (self-pin fidelity, delta_S at P43, KC43 shares, M3_L43 K1 / K2 analogues, how far the free L47 profile moves from the donor's).
Usage: python3 s915_analyze.py TAG_A TAG_B"""
from __future__ import annotations

import collections, hashlib, importlib.util, json, sys
import numpy as np

sys.dont_write_bytecode = True
TAGS = sys.argv[1:3]
FP = "/vlm/research/ces/outputs/writer_circuit_2026-10-20/main/s913_frozen_predictors.json"
HASH = "db299c27045230be"
OUT = f"/vlm/results/r3/s915_analysis_{TAGS[0]}.json" if len(sys.argv) < 4 else sys.argv[3]
rng = np.random.default_rng(915); NB = 4000
_argv = sys.argv; sys.argv = ["s913_fit", "x", "y"]
spec = importlib.util.spec_from_file_location("s913_fit", "/vlm/exp/s913_fit.py"); FIT = importlib.util.module_from_spec(spec); spec.loader.exec_module(FIT)
sys.argv = _argv
P = json.load(open(FP)); fhash = hashlib.sha256(open(FP, "rb").read()).hexdigest()[:16]
KG = FIT.KG


def softmax(v):
    v = np.asarray(v, float); v = v - v.max(); e = np.exp(v); return e / e.sum()


P3C = {}


def m3_p(d, name="M3", L=47):
    if name not in P3C:
        P3C[name] = {int(g): {kk: np.asarray(v) for kk, v in c.items()} for g, c in P[name].items()}
    sc = FIT.m3_scores(d, P3C[name], L, True); t = P[name + "_T"]
    return FIT.mix(np.concatenate([[0.0], softmax(t["ans"][str(KG(d["k"]))] * sc)]), d["N"])


def anchor_of(q):
    q = np.asarray(q); am = [int(np.argmax(q[i])) + 1 for i in (2, 5, 9)]
    c = collections.Counter(am).most_common(); return c[0][0] if c[0][1] >= 2 else am[2]


SETS = {}
for tag in TAGS:
    J = json.load(open(f"/vlm/results/r3/s915_{tag}_cross.json")); N = J["N"]; em = J["early_mean"]
    S = []
    for si, r in enumerate(J["rows"]):
        R = {}
        for key, x in r["land"].items():
            q47 = np.asarray(r["mm"][key]["47"]); am = {int(np.argmax(q47[i])) for i in (2, 5, 9)}
            d = dict(key=key, x=int(x), N=N, a=anchor_of(q47), decisive=len(am) == 1, meta=r["meta"].get(key, {}))
            for L in (47, 43):
                h = np.asarray(r[f"qk{L}h"][key], float).mean(0); d[f"e{L}"] = FIT.centre(h[em[str(L)]].mean(0))
            srt = np.sort(d["e47"])[::-1]; d["margin"] = float(srt[0] - srt[1]); d["inst"] = r.get("inst", {}).get(key)
            R[key] = d
        S.append(R)
    SETS[tag] = dict(J=J, N=N, S=S, nS=len(S))
A, B = TAGS


def xi(x, N):
    return x if 1 <= x <= N else 0


def runs(tag, ix):
    cnt = collections.Counter(int(i) for i in ix)
    return [SETS[tag]["S"][si] for si in sorted(cnt) for _ in range(cnt[si])]


def boot(fn, tags):
    est = fn({tg: np.arange(SETS[tg]["nS"]) for tg in tags}); bs = []
    for _ in range(NB):
        bs.append(fn({tg: rng.integers(0, SETS[tg]["nS"], SETS[tg]["nS"]) for tg in tags}))
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return dict(est=float(est), lo=float(lo), hi=float(hi))


def nanmean(v):
    v = [x for x in v if x is not None and np.isfinite(x)]; return float(np.mean(v)) if v else float("nan")


res = dict(tags=TAGS, frozen_hash=fhash)
# ---------------- I0
I0 = dict(frozen=bool(fhash == HASH)); vals = {}; MEDM = {}
for tg in TAGS:
    st = SETS[tg]; N = st["N"]; J = st["J"]
    sq = nanmean([float(R[f"N_S{q}"]["x"] == q) for R in st["S"] for q in range(1, N)])
    sp = nanmean([float(R[k]["x"] == R["N_" + k[3:]]["x"]) for R in st["S"] for k in R if k.startswith("SP_")])
    hold, holdD, cor, mg = [], [], [], []
    for R in st["S"]:
        for k, d in R.items():
            if d["meta"].get("pins") == "P47" and not k.startswith("SP_"):
                dn = R[d["meta"]["donor"]]; hold.append(float(d["a"] == dn["a"])); cor.append(float(np.corrcoef(d["e47"], dn["e47"])[0, 1]))
                mg.append(dn["margin"])
                if dn["decisive"]:
                    holdD.append(float(d["a"] == dn["a"]))
    med = float(np.median(mg)) if mg else float("nan"); MEDM[tg] = med
    ins = {}
    P47X = [(R, d) for R in st["S"] for k, d in R.items() if d["meta"].get("pins") == "P47" and not k.startswith("SP_")]
    NRs = [d for R in st["S"] for k, d in R.items() if k.startswith("NR_")]
    for li in ("35", "39", "43", "47"):
        cr = [d["inst"][li][0] for R, d in P47X]; crs = [d["inst"][li][0] for R, d in P47X if R[d["meta"]["donor"]]["margin"] < med]
        nr = [d["inst"][li][0] for d in NRs]
        ins[li] = dict(cross_med=float(np.median(cr)), cross_small_med=float(np.median(crs)) if crs else float("nan"), nr_med=float(np.median(nr)),
                       cross_p90=float(np.percentile(cr, 90)), nr_p90=float(np.percentile(nr, 90)),
                       key_cross_med=float(np.median([d["inst"][li][5] for R, d in P47X])), key_nr_med=float(np.median([d["inst"][li][5] for d in NRs])),
                       corr_cross_mean=nanmean([d["inst"][li][3] for R, d in P47X]), corr_nr_mean=nanmean([d["inst"][li][3] for d in NRs]),
                       mass_cross_med=float(np.median([d["inst"][li][4] for R, d in P47X])), mass_nr_med=float(np.median([d["inst"][li][4] for d in NRs])))
    nrf = nanmean([float(R[k]["x"] == R["N_" + k[3:]]["x"]) for R in st["S"] for k in R if k.startswith("NR_")])
    vals[tg] = dict(t1=J["t1_menu"], START=sq, selfpin=sp, natural_rerun_same_answer=nrf, hold_all=nanmean(hold), hold_decisive=nanmean(holdD),
                    n_decisive=len(holdD), n_crossed=len(hold), corr=nanmean(cor), margin_median=med,
                    hold_small=nanmean([h for h, m in zip(hold, mg) if m < med]), hold_large=nanmean([h for h, m in zip(hold, mg) if m >= med]),
                    instrument=ins, n_runs=J["n_runs"])
    I0[f"{tg}_t1"] = bool((J["t1_menu"] or 0) >= .95); I0[f"{tg}_START"] = bool(sq >= .95); I0[f"{tg}_selfpin"] = bool(sp >= .95)
    I0[f"{tg}_lookup_held"] = bool(all(v["cross_med"] <= 1.5 * v["nr_med"] and v["cross_small_med"] <= 1.5 * v["nr_med"] for v in ins.values()))
res["I0"] = dict(**I0, all=bool(all(I0.values())), values=vals)
ok = res["I0"]["all"]
NA = SETS[A]["N"]; KA = range(1, NA + 1)


def per_scene(tag, ix, fn):
    v = []
    for R in runs(tag, ix):
        v += fn(R)
    return nanmean(v)


# ---------------- S1 phrase
dS = lambda ixs: per_scene(A, ixs[A], lambda R: [.5 * (float(R[f"XP_X{k}"]["x"] != R[f"SP_E{k}"]["x"]) + float(R[f"XP_E{k}"]["x"] != R[f"SP_X{k}"]["x"])) for k in KA])
dP = lambda ixs: per_scene(A, ixs[A], lambda R: [.5 * (float(R[f"XP_E{k}"]["x"] != R[f"SP_E{k}"]["x"]) + float(R[f"XP_X{k}"]["x"] != R[f"SP_X{k}"]["x"])) for k in KA])
dN = lambda ixs: per_scene(A, ixs[A], lambda R: [float(R[f"N_X{k}"]["x"] != R[f"N_E{k}"]["x"]) for k in KA])
b1 = boot(dS, [A])
res["S1"] = dict(delta_S=b1, delta_P=boot(dP, [A]), delta_nat=boot(dN, [A]),
                 verdict=(("with the early lookup held, the phrasing's remaining state leaves the answer unchanged" if b1["est"] <= .05 and b1["hi"] < .10 else
                           "the phrasing's remaining state changes the answer at a fixed early lookup" if b1["est"] >= .15 and b1["lo"] > .10 else "partial")
                          if ok else "not read"))
# ---------------- K: k crossing (set A)


def kc_rows(R, pins="P47", pre="KC_E", sp="SP_E"):
    out = []
    for key, d in R.items():
        if not key.startswith(pre) or d["meta"].get("pins") != pins:
            continue
        k2, k = (int(z) for z in key[len(pre):].split("_"))
        ak, ak2 = R[f"N_E{k}"]["a"], R[f"N_E{k2}"]["a"]; xs = R[f"{sp}{k2}"]["x"]
        out.append(dict(d=d, k=k, k2=k2, xd=R[f"{sp}{k}"]["x"], xs=xs, xc=ak - (ak2 - xs), nat=R[f"N_E{k2}"], spd=R[f"{sp}{k2}"]))
    return out


def kd(o, k, prof=None):
    src = prof or o["d"]; return dict(N=o["d"]["N"], k=k, a=src["a"], e47=src["e47"], e43=src["e43"])


def build_kc(tag, pins="P47", pre="KC_E", sp="SP_E", name="M3", L=47):
    """per scene: the crossed runs with their three reference answers and the frozen model's NLL under the state's k, the donor's k, and the
    state's own natural profile (computed once; the bootstrap only resamples scenes)."""
    N = SETS[tag]["N"]; out = []
    for R in SETS[tag]["S"]:
        rows = []
        for o in kc_rows(R, pins, pre, sp):
            j = xi(o["d"]["x"], N)
            nll = dict(state=-np.log(m3_p(kd(o, o["k2"]), name, L)[j]), donor=-np.log(m3_p(kd(o, o["k"]), name, L)[j]),
                       nat=-np.log(m3_p(kd(o, o["k2"], prof=o["nat"]), name, L)[j]))
            rows.append(dict(k=o["k"], k2=o["k2"], x=o["d"]["x"], xd=o["xd"], xs=o["xs"], xc=o["xc"], nll=nll, o=o))
        out.append(rows)
    return out


def k0(KCs, ix):
    c = collections.Counter()
    for si in ix:
        for o in KCs[int(si)]:
            if len({o["xd"], o["xs"], o["xc"]}) < 3:
                continue
            x = o["x"]; c["n"] += 1; c["comp"] += x == o["xc"]; c["donor"] += x == o["xd"]; c["state"] += x == o["xs"]
    return {m: (c[m] / c["n"] if c["n"] else float("nan")) for m in ("comp", "donor", "state")} | dict(n=c["n"])


def kn(KCs, ix, which):
    return nanmean([o["nll"][which] for si in ix for o in KCs[int(si)]])


KCA = build_kc(A)
K0v = k0(KCA, np.arange(SETS[A]["nS"]))
order = sorted(("comp", "donor", "state"), key=lambda m: -(K0v[m] if np.isfinite(K0v[m]) else -1)); top, sec = order[0], order[1]
gap = boot(lambda ixs: (lambda r_: r_[top] - r_[sec])(k0(KCA, ixs[A])), [A])
names0 = dict(comp="the crossed answer composes the early lookup's address with the state's step", donor="follows the early lookup", state="follows the remaining state")
res["K0"] = dict(shares=K0v, top=top, gap=gap, verdict=((names0[top] if gap["est"] >= .10 and gap["lo"] > 0 else "mixed") if ok else "not read"))
k1 = boot(lambda ixs: kn(KCA, ixs[A], "donor") - kn(KCA, ixs[A], "state"), [A])
res["K1"] = dict(d=k1, verdict=(("the state's k acts at a fixed early lookup as the frozen structured model predicts" if k1["est"] >= .10 and k1["lo"] > 0 else
                                 "no effective action of the state's k beyond the early lookup observed" if k1["est"] <= 0 and k1["hi"] < .05 else "partial")
                                if ok else "not read"))
k2_ = boot(lambda ixs: kn(KCA, ixs[A], "nat") - kn(KCA, ixs[A], "state"), [A])
res["K2"] = dict(d=k2_, verdict=(("the held early lookup acts as the frozen model predicts" if k2_["est"] >= .10 and k2_["lo"] > 0 else "not") if ok else "not read"))


def epos(p, N):
    q = np.asarray(p[1:], float); q = q / q.sum(); return float(q @ np.arange(1, N + 1))


cells = collections.defaultdict(lambda: dict(o=[], p=[]))
for rows_ in KCA:
    for o in (q_["o"] for q_ in rows_):
        x, xs = o["d"]["x"], o["spd"]["x"]
        if not (1 <= x <= NA and 1 <= xs <= NA):
            continue
        cells[(o["k2"], o["k"])]["o"].append(x - xs)
        cells[(o["k2"], o["k"])]["p"].append(epos(m3_p(kd(o, o["k2"])), NA) - epos(m3_p(kd(o, o["k2"], prof=o["spd"])), NA))
cc = [(np.mean(c["p"]), np.mean(c["o"])) for c in cells.values() if len(c["o"]) >= 10]
if len(cc) >= 3:
    pp_, oo_ = np.array(cc).T; r3 = float(np.corrcoef(pp_, oo_)[0, 1]); sl = float(np.polyfit(pp_, oo_, 1)[0])
else:
    r3 = sl = float("nan")
res["K3"] = dict(r=r3, slope=sl, n_cells=len(cc), cells={f"{k2}_{k}": dict(n=len(c["o"]), obs=float(np.mean(c["o"])) if c["o"] else None,
                                                                           pred=float(np.mean(c["p"])) if c["p"] else None) for (k2, k), c in sorted(cells.items())},
                 verdict=(("the frozen model predicts the direction and size of the change from swapping the early lookup" if r3 >= .80 and .67 <= sl <= 1.5 else
                           "partial" if r3 >= .60 else "not") if ok else "not read"))
# ---------------- C: frame


def depC(ixs, arm, idn=False):
    v = []
    for tg, ix in ixs.items():
        N = SETS[tg]["N"]
        for R in runs(tg, ix):
            for k in range(2, N + 1):
                ak = R[f"N_E{k}"]["a"]
                if idn and ak != N + 1 - k:
                    continue
                v.append(float(R[f"{arm}{k}"]["x"] != ak))
    return nanmean(v)


bC = boot(lambda ixs: depC(ixs, "SP_E") - depC(ixs, "FC_S"), TAGS)
res["C"] = dict(d=bC, dep_E=depC({tg: np.arange(SETS[tg]["nS"]) for tg in TAGS}, "SP_E"), dep_S=depC({tg: np.arange(SETS[tg]["nS"]) for tg in TAGS}, "FC_S"),
                verdict=(("at the same early lookup the END state departs and the START state stays: the departure is END-conditioned" if bC["est"] >= .30 and bC["lo"] > .15 else
                          "no effective END-conditioning observed" if bC["est"] <= .10 and bC["hi"] < .20 else "partial") if ok else "not read"))
# ---------------- U1: address part vs full early lookup


def upairs(R, N):
    out = []
    for key in R:
        if key.startswith(("XPU_", "KCU_", "FCU_")):
            other = key.replace("XPU_", "XP_").replace("KCU_", "KC_").replace("FCU_", "FC_")
            if other in R:
                out.append((R[key], R[other]))
    return out


def ag_u(ixs, pairs=upairs, small=None):
    v = []
    for tg, ix in ixs.items():
        for R in runs(tg, ix):
            for du, da in pairs(R, SETS[tg]["N"]):
                if small is not None and (R[da["meta"]["donor"]]["margin"] < MEDM[tg]) != small:
                    continue
                v.append(float(du["x"] == da["x"]))
    return nanmean(v)


selfp = lambda R, N: [(R[f"SPU_E{k}"], R[f"SP_E{k}"]) for k in range(1, N + 1) if f"SPU_E{k}" in R]
bU = boot(lambda ixs: ag_u(ixs), TAGS)
res["U1"] = dict(AG=bU, AG_self=boot(lambda ixs: ag_u(ixs, selfp), TAGS),
                 AG_small=ag_u({tg: np.arange(SETS[tg]["nS"]) for tg in TAGS}, small=True), AG_large=ag_u({tg: np.arange(SETS[tg]["nS"]) for tg in TAGS}, small=False),
                 verdict=(("holding only the address part reproduces the outcome of holding the full early lookup" if bU["est"] >= .90 and bU["lo"] > .85 else
                           "the early query's non-address part changes the outcome" if bU["est"] <= .75 and bU["hi"] < .85 else "partial") if ok else "not read"))
# ---------------- the two halves of the donor's initial-address margin (every statistic, descriptive)


def halves():
    out = {}
    for sm, nm in ((True, "small_margin"), (False, "large_margin")):
        h = {}
        Rs = SETS[A]["S"]; m = MEDM[A]
        v = []
        for R in Rs:
            for k in KA:
                if (R[f"N_E{k}"]["margin"] < m) == sm:
                    v.append(float(R[f"XP_X{k}"]["x"] != R[f"SP_E{k}"]["x"]))
                if (R[f"N_X{k}"]["margin"] < m) == sm:
                    v.append(float(R[f"XP_E{k}"]["x"] != R[f"SP_X{k}"]["x"]))
        h["S1_delta_S"] = dict(est=nanmean(v), n=len(v))
        rows_ = [o for si, rws in enumerate(KCA) for o in rws if (Rs[si][f"N_E{o['k']}"]["margin"] < m) == sm]
        c = collections.Counter()
        for o in rows_:
            if len({o["xd"], o["xs"], o["xc"]}) == 3:
                c["n"] += 1; c["comp"] += o["x"] == o["xc"]; c["donor"] += o["x"] == o["xd"]; c["state"] += o["x"] == o["xs"]
        h["K0"] = {q: (c[q] / c["n"] if c["n"] else float("nan")) for q in ("comp", "donor", "state")} | dict(n=c["n"])
        h["K1"] = nanmean([o["nll"]["donor"] - o["nll"]["state"] for o in rows_]); h["K2"] = nanmean([o["nll"]["nat"] - o["nll"]["state"] for o in rows_])
        h["n_KC"] = len(rows_)
        vE, vS = [], []
        for tg in TAGS:
            for R in SETS[tg]["S"]:
                for k in range(2, SETS[tg]["N"] + 1):
                    if (R[f"N_E{k}"]["margin"] < MEDM[tg]) == sm:
                        ak = R[f"N_E{k}"]["a"]; vE.append(float(R[f"SP_E{k}"]["x"] != ak)); vS.append(float(R[f"FC_S{k}"]["x"] != ak))
        h["C"] = dict(dep_E=nanmean(vE), dep_S=nanmean(vS), d=nanmean(vE) - nanmean(vS), n=len(vE))
        out[nm] = h
    return out


res["margin_halves"] = halves()
# ---------------- descriptive
full = {tg: np.arange(SETS[tg]["nS"]) for tg in TAGS}
desc = {}
for tg in TAGS:
    st = SETS[tg]; N = st["N"]; D = {}
    for k in range(1, N + 1):
        t = N + 1 - k; Rs = st["S"]; row = {}
        for arm in ("N_E", "SP_E", "FC_E", "FC_S", "N_X", "SP_X", "XP_X", "XP_E", "SP43_E", "XP43_X"):
            pr = [(R[f"{arm}{k}"], R) for R in Rs if f"{arm}{k}" in R]
            if pr:
                row[arm] = dict(acc=float(np.mean([d["x"] == t for d, _ in pr])), dep=float(np.mean([d["x"] != R[f"N_E{k}"]["a"] for d, R in pr])),
                                answers=dict(collections.Counter(d["x"] for d, _ in pr)))
        row["anchors"] = dict(collections.Counter(R[f"N_E{k}"]["a"] for R in Rs))
        idr = [R for R in Rs if R[f"N_E{k}"]["a"] == t]
        row["identity"] = dict(n=len(idr), acc_SP_E=nanmean([float(R[f"SP_E{k}"]["x"] == t) for R in idr]), acc_FC_S=nanmean([float(R[f"FC_S{k}"]["x"] == t) for R in idr]))
        D[f"k{k}"] = row
    desc[tg] = D
desc["identity_pooled"] = dict(n=sum(len([R for R in SETS[tg]["S"] for k in range(2, SETS[tg]["N"] + 1) if R[f"N_E{k}"]["a"] == SETS[tg]["N"] + 1 - k]) for tg in TAGS),
                               dep_E=depC(full, "SP_E", True), dep_S=depC(full, "FC_S", True))
desc["FC_E_vs_SP_S"] = {tg: nanmean([float(R[f"FC_E{k}"]["x"] != R[f"N_E{k}"]["a"]) for R in SETS[tg]["S"] for k in range(2, SETS[tg]["N"] + 1)]) for tg in TAGS}
KCB = build_kc(B)
desc["setB_K0"] = k0(KCB, full[B]); desc["setB_K1"] = kn(KCB, full[B], "donor") - kn(KCB, full[B], "state"); desc["setB_K2"] = kn(KCB, full[B], "nat") - kn(KCB, full[B], "state")
if SETS[A]["J"].get("p43"):
    sp43 = nanmean([float(R[f"SP43_E{k}"]["x"] == R[f"N_E{k}"]["x"]) for R in SETS[A]["S"] for k in KA])
    dS43 = nanmean([float(R[f"XP43_X{k}"]["x"] != R[f"SP43_E{k}"]["x"]) for R in SETS[A]["S"] for k in KA])
    drift = [dict(a=float(d["a"] == R[d["meta"]["donor"]]["a"]), r=float(np.corrcoef(d["e47"], R[d["meta"]["donor"]]["e47"])[0, 1]))
             for R in SETS[A]["S"] for key, d in R.items() if d["meta"].get("pins") == "P43" and not key.startswith("SP43")]
    K43 = build_kc(A, "P43", "KC43_E", "SP43_E", "M3_L43", 43)
    desc["P43"] = dict(selfpin=sp43, delta_S43=dS43, K0=k0(K43, full[A]), K1_L43=kn(K43, full[A], "donor") - kn(K43, full[A], "state"),
                       K2_L43=kn(K43, full[A], "nat") - kn(K43, full[A], "state"),
                       L47_anchor_kept=nanmean([x["a"] for x in drift]), L47_corr=nanmean([x["r"] for x in drift]))
res["descriptive"] = desc
json.dump(res, open(OUT, "w"), indent=1, default=float)
print(json.dumps(dict(I0=res["I0"]["all"], I0v=vals, S1=[res["S1"]["verdict"], b1["est"], res["S1"]["delta_nat"]["est"]], K0=[res["K0"]["verdict"], K0v],
                      K1=[res["K1"]["verdict"], k1["est"]], K2=[res["K2"]["verdict"], k2_["est"]], K3=[res["K3"]["verdict"], r3, sl],
                      C=[res["C"]["verdict"], bC["est"]], U1=[res["U1"]["verdict"], bU["est"]]), indent=1, default=float))
print("->", OUT)
