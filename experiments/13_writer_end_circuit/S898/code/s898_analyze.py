"""S898 analysis (rules fixed before the data; knowledge/DECISIONS.md S898 entry).  Input results/r3/s898_<tag>_qpathpre.json.
Self-address heads {1, 4, 12, 17, 20}: FA47 attention mass on the anchor block a(X) and on the successor block s(X) (rows B1 / B4 / answer, as the
run's own queries computed it).  Anchor restoration rho_X = mean[a(E_QP_X) - a(E)] / mean[a(R) - a(E)]; FA47's END part F = W_E - W_ALL47R and the
path share f_X = (W_E - W_QP_X) / F (T1 at L51).  Scene bootstrap 4,000 (seed 895).
I0: t1 >= .95; START(p) answers p >= .95; L47 anchor = p for E and R >= .90; QID reproduces E (max late block-score difference <= 1e-3 in >= .99);
  F > 0 with the interval above 0.  Nothing is read if I0 fails.  (Amended before data: the smoke's whole-residual restoration was 1.17 - the
  question tokens' keys and values differ between END and START prompts, so START's queries need not reproduce START's attention; rho_ALL is
  reported descriptively.)
T1 (where within layers 0-34): among PRE20, L20_25, L26, L27_30, L31_34 the group with the largest rho; rho >= .50 with the lower bound > .30 ->
  'the <group> write moves the self-address heads end-ward (through FA47's queries and gates)'; else 'distributed'.
Descriptive: rho and f for every group (PRE35 replicates S895's .704), successor-mass restoration, answers per arm.
Usage: python3 s895_analyze.py TAG"""
from __future__ import annotations

import collections, json, sys
import numpy as np

TAG = sys.argv[1]
J = json.load(open(f"/vlm/results/r3/s898_{TAG}_qpathpre.json"))
OUT = f"/vlm/results/r3/s898_analysis_{TAG}.json"
rng = np.random.default_rng(898); NB = 4000
rows = J["rows"]; S = len(rows); PINS = J["pins"]; lm = J["late_mean"]; LAYS = [str(l) for l in J["lateq"]]
SA = J["selfaddr"]
GROUPS = ("PRE20", "L20_25", "L26", "L27_30", "L31_34", "PRE35")


def zv(r, key, L):
    return np.asarray(r["zl"][key][L], dtype=float)[:, lm[L]].mean(1).mean(0)


def T1(r, key, p, L="51"):
    z = zv(r, key, L); return z[p - 2] - z[p - 1]


def anchor(q):
    q = np.asarray(q); am = [int(np.argmax(q[i])) + 1 for i in (2, 5, 9)]
    c = collections.Counter(am).most_common(); return c[0][0] if c[0][1] >= 2 else am[2]


REC = {}
for si, r in enumerate(rows):
    for p in PINS:
        d = dict(si=si, p=p)
        tb = T1(r, f"Ep{p}_E_BANDR", p); d["W_E"] = T1(r, f"Ep{p}", p) - tb; d["W_47"] = T1(r, f"Ep{p}_ALL47R", p) - tb
        keys = {"E": f"Ep{p}", "R": f"Rp{p}", **{g: f"Ep{p}_QP_{g}" for g in GROUPS}}
        for g, key in keys.items():
            m = np.asarray(r["mass47"][key], float)
            d[f"a_{g}"] = float(m[:, SA, p - 1].mean()); d[f"s_{g}"] = float(m[:, SA, p].mean()) if p < 8 else 0.0
            if g in GROUPS:
                d[f"W_{g}"] = T1(r, key, p) - tb
        d["vid"] = max(float(np.abs(zv(r, f"Ep{p}_QID", L) - zv(r, f"Ep{p}", L)).max()) for L in LAYS)
        d["anch"] = (anchor(r["qk47"][f"Ep{p}"]), anchor(r["qk47"][f"Rp{p}"]))
        d["land"] = {k: v for k, v in r["land"].items() if f"p{p}" in k}
        REC[(si, p)] = d
SC = sorted({k[0] for k in REC})


def bt(fn):
    n = len(SC); est = fn(np.arange(n)); bs = [fn(rng.integers(0, n, n)) for _ in range(NB)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return dict(est=float(est), lo=float(lo), hi=float(hi))


def M(ix, fn):
    return float(np.mean([fn(REC[(SC[i], p)]) for i in ix for p in PINS]))


rho = lambda ix, g: M(ix, lambda d: d[f"a_{g}"] - d["a_E"]) / M(ix, lambda d: d["a_R"] - d["a_E"])
fsh = lambda ix, g: M(ix, lambda d: d["W_E"] - d[f"W_{g}"]) / M(ix, lambda d: d["W_E"] - d["W_47"])
res = dict(tag=TAG, n_scenes=S)
st = float(np.mean([r["land"][f"START{p}"] == p for r in rows for p in PINS]))
reach = float(np.mean([a_ == d["p"] for d in REC.values() for a_ in d["anch"]]))
vid_ok = float(np.mean([d["vid"] <= 1e-3 for d in REC.values()]))
bF = bt(lambda ix: M(ix, lambda d: d["W_E"] - d["W_47"]))
I0 = dict(t1=bool((J["t1_menu"] or 0) >= .95), START=bool(st >= .95), pin_reach=bool(reach >= .90), identity_path=bool(vid_ok >= .99),
          fa47_part=bool(bF["lo"] > 0))
res["I0"] = dict(**I0, all=bool(all(I0.values())), t1_value=J["t1_menu"], START_acc=st, reach=reach, vid_ok=vid_ok, F=bF)
ok = res["I0"]["all"]
rhos = {g: rho(range(len(SC)), g) for g in GROUPS if g != "PRE35"}
top = max(rhos, key=rhos.get); rt = bt(lambda ix: rho(ix, top))
res["T1"] = dict(rho=rhos, top=top, rho_top=rt, verdict=((f"the {top} write moves the self-address heads end-ward (through FA47's queries and gates)" if rt["est"] >= .50 and rt["lo"] > .30
                                                          else "distributed") if ok else "not read"))
allx = range(len(SC))
res["descriptive"] = dict(rho={g: rho(allx, g) for g in GROUPS}, f={g: fsh(allx, g) for g in GROUPS},
                          successor_restoration={g: M(allx, lambda d, g=g: d["s_E"] - d[f"s_{g}"]) / M(allx, lambda d: d["s_E"] - d["s_R"]) for g in GROUPS},
                          attention={k: M(allx, lambda d, k=k: d[k]) for k in ["a_E", "a_R", "s_E", "s_R"] + [f"a_{g}" for g in GROUPS]},
                          answers={g: dict(P_pm1=float(np.mean([d["land"][f"Ep{d['p']}_QP_{g}"] == d["p"] - 1 for d in REC.values()])),
                                           P_p=float(np.mean([d["land"][f"Ep{d['p']}_QP_{g}"] == d["p"] for d in REC.values()]))) for g in GROUPS})
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"]["all"], T1=[res["T1"]["verdict"], top, rt["est"]], rho_PRE35=rho(range(len(SC)), "PRE35")), indent=1))
print("->", OUT)
