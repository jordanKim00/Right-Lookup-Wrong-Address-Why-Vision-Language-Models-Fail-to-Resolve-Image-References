"""S922 analysis = S920's rules on one replication set (registered in knowledge/DECISIONS.md S922 entry): results/r3/s920_<tag>_r3circuit.json
at N7, k 1-7 (the end boundaries k = 1, 7 included; k = 4 is the identity cell where R³ writes the frame only).  Below, S920's text with
'both sets' read as 'the set', R5 on k = (N + 1) / 2, and the boundary cells also reported apart.  Per scene and k (target t = N + 1 - k,
a = END(k)'s natural L47 anchor): E, Sa (START(a)), E_B (E + Sa's band outputs), St (START(t)), R3, R3_B (+ St's band), R3A (address write
only), R3A_B, R3F (frame write only), Sk (START(k)).  acc = P(answer = t).  Band effects: W_E = P(x(E) != a) - P(x(E_B) != a) (the natural
END-specific rewrite), W_R3 = P(x(R3) != t) - P(x(R3_B) != t), W_R3A = the same for R3A.  Late index I(arm) = mean over layers 55 / 59 / 63
and rows B1 / B4 / answer of [cos(arm, St) - cos(E, St)] / [1 - cos(E, St)] (late full-attention read outputs; 1 = START(t)'s, 0 = END's).
Scene bootstrap 4,000 within each set (seed 920); pooled statistics resample both sets.
I0: t1 >= .95 and natural START(q) answers q >= .95 (q < N) in each set; R³'s READ is right (k_hat = k, t, N_hat = N, >= 1 write) in >= .80 of
  the R3 runs of each set; natural END(k) accuracy <= .70 (pooled); the natural band effect is present (W_E >= .30 with the lower bound > .15,
  pooled).  Nothing is read if I0 fails.  Every statistic is also reported on the runs whose READ is right.
R1 (repair): acc(R3) - acc(E) >= .30 with the lower bound > .20 -> 'R³ repairs END(k) on these scenes'; else 'not'.
R2 (the band after R³; central): W_E - W_R3 >= .25 with the lower bound > .15 and W_R3 <= .10 -> 'after R³ the band's END-specific rewriting
  is gone'; W_R3 >= .25 with the lower bound > .15 -> 'the band still rewrites after R³'; else 'partial'.
R3 (the late reads; central): I(R3) >= .50 with the lower bound > .30 -> 'R³ moves the late read outputs at least halfway to direct START(t)'s';
  <= .15 with the upper bound < .30 -> 'R³ leaves the late read outputs END-like'; else 'partial'.
R4 (the two writes): a = acc(R3), aA = acc(R3A), aF = acc(R3F): a - max(aA, aF) >= .15 with the lower bound > .05 -> 'both writes are needed';
  else a - aA <= .05 with the upper bound < .10 -> 'the address write suffices'; else a - aF <= .05 with the upper bound < .10 -> 'the frame
  write suffices'; else 'partial'.
R5 (identity, set B k = 5, R³ writes the frame only): acc(R3) - acc(E) >= .40 with the lower bound > .25 -> 'R³'s frame-only write repairs the
  identity collapse'; else 'partial' if the lower bound > 0, else 'not'.
R6 (an address-only repair as an induced identity cell): among R3A runs whose L47 anchor = t, P(x(R3A) != t) - P(x(R3A_B) != t) >= .20 with
  the lower bound > .10 -> 'with the address set to the target by R³ but the END frame kept, the END band moves the answer off it'; <= .05
  with the upper bound < .10 -> 'no effective rewrite of the corrected address observed'; else 'partial'.
Descriptive: per set and k every arm's accuracy, answers, L47 / L51 / L63 anchors, late indices per layer (every arm), the READ ledger; R2 / R3
  on repaired runs (x(R3) = t) and on READ-right runs.
Usage: python3 s922_analyze.py TAG [OUT]  (rules R1-R4, R6 and I0's END / band conditions on the middle k = 2 .. N - 1, as S920; boundaries reported apart)"""
from __future__ import annotations

import collections, json, sys
import numpy as np

TAGS = sys.argv[1:2]
OUT = sys.argv[2] if len(sys.argv) > 2 else f"/vlm/results/r3/s922_analysis_{TAGS[0]}.json"
rng = np.random.default_rng(922); NB = 4000
ARMS = ["E", "Sa", "E_B", "St", "R3", "R3_B", "R3A", "R3A_B", "R3F", "Sk"]
SETS = {}
for tg in TAGS:
    J = json.load(open(f"/vlm/results/r3/s920_{tg}_r3circuit.json")); SETS[tg] = dict(J=J, N=J["N"], KS=J["ks"], rows=J["rows"], nS=len(J["rows"]))


def anchor(q):
    q = np.asarray(q); am = [int(np.argmax(q[i])) + 1 for i in (2, 5, 9)]
    c = collections.Counter(am).most_common(); return c[0][0] if c[0][1] >= 2 else am[2]


def read_ok(r, k, N):
    p = r["plan"][f"R3_{k}"]
    return bool(p["n_writes"] > 0 and any(d["kh"] == k and d["t"] == N + 1 - k and d["N_hat"] == N for d in p["detail"]))


def units(ixs, ks=None, cond=None):
    """(row, k, N) for the resampled scenes of every set"""
    out = []
    for tg, ix in ixs.items():
        st = SETS[tg]
        for i in ix:
            r = st["rows"][int(i)]
            for k in st["KS"]:
                if ks is not None and k not in ks.get(tg, st["KS"]):
                    continue
                if cond is not None and not cond(r, k, st["N"]):
                    continue
                out.append((r, k, st["N"]))
    return out


def mean(v):
    v = [x for x in v if x is not None and np.isfinite(x)]; return float(np.mean(v)) if v else float("nan")


x = lambda r, arm, k: r["land"][f"{arm}_{k}"]
acc = lambda U, arm: mean([float(x(r, arm, k) == N + 1 - k) for r, k, N in U])
W_E = lambda U: mean([float(x(r, "E", k) != r["anchor"][str(k)]) for r, k, N in U]) - mean([float(x(r, "E_B", k) != r["anchor"][str(k)]) for r, k, N in U])
Wt = lambda U, arm: mean([float(x(r, arm, k) != N + 1 - k) for r, k, N in U]) - mean([float(x(r, f"{arm}_B", k) != N + 1 - k) for r, k, N in U])


def late_index(U, arm, layers=("55", "59", "63")):
    v = []
    for r, k, N in U:
        lo = r["lateout"][str(k)]
        for l in layers:
            for a_, e_ in zip(lo[arm][l]["St"], lo["E"][l]["St"]):
                v.append((a_ - e_) / max(1e-6, 1 - e_))
    return mean(v)


def boot(fn, tags=None, **kw):
    tags = tags or TAGS
    est = fn(units({tg: np.arange(SETS[tg]["nS"]) for tg in tags}, **kw)); bs = []
    for _ in range(NB):
        bs.append(fn(units({tg: rng.integers(0, SETS[tg]["nS"], SETS[tg]["nS"]) for tg in tags}, **kw)))
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return dict(est=float(est), lo=float(lo), hi=float(hi))


MIDK = {TAGS[0]: list(range(2, SETS[TAGS[0]]["N"]))}
ALL = units({tg: np.arange(SETS[tg]["nS"]) for tg in TAGS}, ks=MIDK)
res = dict(tags=TAGS)
I0 = {}; vals = {}
for tg in TAGS:
    st = SETS[tg]; N = st["N"]; U = units({tg: np.arange(st["nS"])})
    sq = mean([float(r["land"][f"N_S{q}"] == q) for r in st["rows"] for q in range(1, N)])
    rd = mean([float(read_ok(r, k, N)) for r, k, _ in U])
    vals[tg] = dict(t1=st["J"]["t1_menu"], START=sq, READ_right=rd)
    I0[f"{tg}_t1"] = bool((st["J"]["t1_menu"] or 0) >= .95); I0[f"{tg}_START"] = bool(sq >= .95); I0[f"{tg}_READ"] = bool(rd >= .80)
ae = acc(ALL, "E"); we = boot(W_E, ks=MIDK)
I0["END_errs"] = bool(ae <= .70); I0["band_effect"] = bool(we["est"] >= .30 and we["lo"] > .15)
res["I0"] = dict(**I0, all=bool(all(I0.values())), values=vals, acc_END=ae, W_E=we)
ok = res["I0"]["all"]
b1 = boot(lambda U: acc(U, "R3") - acc(U, "E"), ks=MIDK)
res["R1"] = dict(d=b1, verdict=(("R³ repairs END(k) on these scenes" if b1["est"] >= .30 and b1["lo"] > .20 else "not") if ok else "not read"))
d2 = boot(lambda U: W_E(U) - Wt(U, "R3"), ks=MIDK); w2 = boot(lambda U: Wt(U, "R3"), ks=MIDK)
res["R2"] = dict(W_E_minus_W_R3=d2, W_R3=w2,
                 verdict=(("after R³ the band's END-specific rewriting is gone" if d2["est"] >= .25 and d2["lo"] > .15 and w2["est"] <= .10 else
                           "the band still rewrites after R³" if w2["est"] >= .25 and w2["lo"] > .15 else "partial") if ok else "not read"))
b3 = boot(lambda U: late_index(U, "R3"), ks=MIDK)
res["R3"] = dict(I=b3, verdict=(("R³ moves the late read outputs at least halfway to direct START(t)'s" if b3["est"] >= .50 and b3["lo"] > .30 else
                                 "R³ leaves the late read outputs END-like" if b3["est"] <= .15 and b3["hi"] < .30 else "partial") if ok else "not read"))
both = boot(lambda U: acc(U, "R3") - max(acc(U, "R3A"), acc(U, "R3F")), ks=MIDK)
dA = boot(lambda U: acc(U, "R3") - acc(U, "R3A"), ks=MIDK); dF = boot(lambda U: acc(U, "R3") - acc(U, "R3F"), ks=MIDK)
if both["est"] >= .15 and both["lo"] > .05:
    v4 = "both writes are needed"
elif dA["est"] <= .05 and dA["hi"] < .10:
    v4 = "the address write suffices"
elif dF["est"] <= .05 and dF["hi"] < .10:
    v4 = "the frame write suffices"
else:
    v4 = "partial"
res["R4"] = dict(both=both, d_addr_only=dA, d_frame_only=dF, verdict=(v4 if ok else "not read"))
B = TAGS[0]; idk = {B: [(SETS[B]["N"] + 1) // 2]}
b5 = boot(lambda U: acc(U, "R3") - acc(U, "E"), tags=[B], ks=idk)
res["R5"] = dict(d=b5, n=len(units({B: np.arange(SETS[B]["nS"])}, ks=idk)),
                 verdict=(("R³'s frame-only write repairs the identity collapse" if b5["est"] >= .40 and b5["lo"] > .25 else "partial" if b5["lo"] > 0 else "not")
                          if ok else "not read"))
anc_t = lambda r, k, N: anchor(r["mm"][f"R3A_{k}"]["47"]) == N + 1 - k
b6 = boot(lambda U: Wt(U, "R3A"), cond=anc_t, ks=MIDK)
res["R6"] = dict(d=b6, n=len(units({tg: np.arange(SETS[tg]["nS"]) for tg in TAGS}, cond=anc_t, ks=MIDK)),
                 verdict=(("with the address set to the target by R³ but the END frame kept, the END band moves the answer off it" if b6["est"] >= .20 and b6["lo"] > .10 else
                           "no effective rewrite of the corrected address observed" if b6["est"] <= .05 and b6["hi"] < .10 else "partial") if ok else "not read"))
desc = {}
for tg in TAGS:
    st = SETS[tg]; N = st["N"]; D = {}
    for k in st["KS"]:
        U = units({tg: np.arange(st["nS"])}, ks={tg: [k]}); t = N + 1 - k
        D[f"k{k}"] = {arm: dict(acc=acc(U, arm), answers=dict(collections.Counter(x(r, arm, k) for r, k_, _ in U)),
                                L47=dict(collections.Counter(anchor(r["mm"][f"{arm}_{k}"]["47"]) for r, k_, _ in U)),
                                L51=dict(collections.Counter(anchor(r["mm"][f"{arm}_{k}"]["51"]) for r, k_, _ in U)),
                                L63=dict(collections.Counter(anchor(r["mm"][f"{arm}_{k}"]["63"]) for r, k_, _ in U)),
                                late_index={l: late_index(U, arm, (l,)) for l in ("51", "55", "59", "63")}) for arm in ARMS}
        D[f"k{k}"]["W"] = dict(W_E=W_E(U), W_R3=Wt(U, "R3"), W_R3A=Wt(U, "R3A")); D[f"k{k}"]["READ_right"] = mean([float(read_ok(r, k, N)) for r, _, _ in U])
        D[f"k{k}"]["anchors_E"] = dict(collections.Counter(r["anchor"][str(k)] for r, _, _ in U))
    desc[tg] = D
REP_ = units({tg: np.arange(SETS[tg]["nS"]) for tg in TAGS}, cond=lambda r, k, N: x(r, "R3", k) == N + 1 - k)
RD_ = units({tg: np.arange(SETS[tg]["nS"]) for tg in TAGS}, cond=lambda r, k, N: read_ok(r, k, N))
desc["repaired_runs"] = dict(n=len(REP_), W_R3=Wt(REP_, "R3"), W_E=W_E(REP_), late_index=late_index(REP_, "R3"))
desc["read_right_runs"] = dict(n=len(RD_), acc_R3=acc(RD_, "R3"), acc_E=acc(RD_, "E"), W_R3=Wt(RD_, "R3"), late_index=late_index(RD_, "R3"))
desc["pooled"] = {arm: dict(acc=acc(ALL, arm), late_index=late_index(ALL, arm)) for arm in ARMS}
NB_ = SETS[TAGS[0]]["N"]; BND = units({TAGS[0]: np.arange(SETS[TAGS[0]]["nS"])}, ks={TAGS[0]: [1, NB_]}); MID = units({TAGS[0]: np.arange(SETS[TAGS[0]]["nS"])}, ks={TAGS[0]: list(range(2, NB_))})
desc["boundary_k1_kN"] = dict(n=len(BND), acc={arm: acc(BND, arm) for arm in ARMS}, W_E=W_E(BND), W_R3=Wt(BND, "R3"), late_index=late_index(BND, "R3"))
desc["middle_k"] = dict(n=len(MID), acc={arm: acc(MID, arm) for arm in ARMS}, W_E=W_E(MID), W_R3=Wt(MID, "R3"), late_index=late_index(MID, "R3"))
res["descriptive"] = desc
json.dump(res, open(OUT, "w"), indent=1, default=float)
print(json.dumps(dict(I0=res["I0"]["all"], R1=[res["R1"]["verdict"], b1["est"]], R2=[res["R2"]["verdict"], d2["est"], w2["est"]], R3=[res["R3"]["verdict"], b3["est"]],
                      R4=[res["R4"]["verdict"], both["est"]], R5=[res["R5"]["verdict"], b5["est"]], R6=[res["R6"]["verdict"], b6["est"]]), indent=1, default=float))
print("->", OUT)
