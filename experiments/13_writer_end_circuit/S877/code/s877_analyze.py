"""S877 analysis (rules fixed before the data; knowledge/DECISIONS.md S877 entry).  Input results/r3/s877_<tag>_context.json (and S873's
transplant JSON for the reproducibility check).
Address-level writer effect of a run X at late layer L: W_L(X) = T1_L(X) - T1_L(X_BANDR), T1_L = z_(p-1) - z_p of the matcher-head mean block
scores (rows B1 / B4 / answer) - by position, so defined when the pictures are another scene's.  Natural W_c (E_c, E_c_BANDR) and W_d (E_d,
E_d_BANDR) from the same pass; hybrids W(H46) = T1(H46) - T1(H46_BANDR), W(H51) = T1(H51) - T1(H51_BANDR).
follow_d(h) = r(W_h, W_d) - r(W_h, W_c), within-pin centred Pearson over (scene, pin, donor) cells; scene bootstrap 4,000 (seed 877).  Primary
layer L51 (where S873's context dependence already appears, post hoc p19).
I0: t1 >= .95; START(p) answers p >= .95; L47 anchor = p for E_c and R_c >= .90; the refs E_c of the two donors' sub-passes have equal T1 in
  >= .95 of (scene, pin) (no thread race); reproducibility: W_c(L51) here vs S873's W_c(L51) (same scenes and pins), Pearson >= .90.
C1 (is the context effect carried by the prefix?): follow_d(H46) >= .20 with the interval above 0 -> 'the context effect is carried by the
  scene's prefix (its pictures)'; <= -.20 with the interval below 0 -> 'carried by the scene's question / menu rows'; else 'mixed'.
C2 (band or late keys): follow_d(H51) >= .20 with the interval above 0 -> 'the late lookup's key geometry sets the writer magnitude';
  <= -.20 with the interval below 0 -> 'the band's processing of the scene sets it (the late keys do not)'; else 'both'.
Descriptive: the same statistics at L55 / L59 / L63; mean writer effects of every arm; the logit-level T3 (content mismatched in hybrids).
Usage: python3 s877_analyze.py TAG [--s873 s873_27b]"""
from __future__ import annotations

import argparse, collections, json
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("tag"); ap.add_argument("--s873", default="s873_27b"); A = ap.parse_args()
R3 = "/vlm/results/r3"
J = json.load(open(f"{R3}/s877_{A.tag}_context.json"))
OUT = f"{R3}/s877_analysis_{A.tag}.json"
rng = np.random.default_rng(877); NB = 4000
rows = J["rows"]; S = len(rows); PINS = J["pins"]; lm = J["late_mean"]; NSH = len(J["shifts"])
LAYS = ("51", "55", "59", "63")


def T1(r, key, p, L):
    z = np.asarray(r["zl"][key][L], dtype=float)[:, lm[L]].mean(1).mean(0); return z[p - 2] - z[p - 1]


def anchor(q):
    q = np.asarray(q); am = [int(np.argmax(q[i])) + 1 for i in (2, 5, 9)]
    c = collections.Counter(am).most_common(); return c[0][0] if c[0][1] >= 2 else am[2]


CELL = []
for si, r in enumerate(rows):
    for p in PINS:
        for j in range(NSH):
            k = lambda nm: f"Ep{p}_d{j + 1}_{nm}"
            d = dict(si=si, p=p, j=j, dd=r["donors"][f"p{p}_d{j + 1}"])
            for L in LAYS:
                d[f"Wc{L}"] = T1(r, f"Ep{p}@{j}", p, L) - T1(r, k("E_BANDR"), p, L)
                d[f"Wd{L}"] = T1(r, k("Ed"), p, L) - T1(r, k("Ed_BANDR"), p, L)
                d[f"W46_{L}"] = T1(r, k("H46"), p, L) - T1(r, k("H46_BANDR"), p, L)
                d[f"W51_{L}"] = T1(r, k("H51"), p, L) - T1(r, k("H51_BANDR"), p, L)
            d["T1E"] = T1(r, f"Ep{p}@{j}", p, "51")
            d["anch"] = (anchor(r["qk47"][f"Ep{p}@{j}"]), anchor(r["qk47"][f"Rp{p}@{j}"]))
            CELL.append(d)
BY = collections.defaultdict(list)
for i, d in enumerate(CELL):
    BY[d["si"]].append(i)
SC = sorted(BY)


def bt(fn):
    n = len(SC); est = fn(np.arange(n)); bs = [fn(rng.integers(0, n, n)) for _ in range(NB)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return dict(est=float(est), lo=float(lo), hi=float(hi))


def follow(ix, h, L):
    C = [CELL[i] for s in ix for i in BY[SC[s]]]
    P = np.array([c["p"] for c in C]); W = np.array([c[f"{h}_{L}"] for c in C])
    Wd = np.array([c[f"Wd{L}"] for c in C]); Wc = np.array([c[f"Wc{L}"] for c in C])
    for p in PINS:
        m = P == p
        for a_ in (W, Wd, Wc):
            a_[m] -= a_[m].mean()
    return float(np.corrcoef(W, Wd)[0, 1] - np.corrcoef(W, Wc)[0, 1]), float(np.corrcoef(W, Wd)[0, 1]), float(np.corrcoef(W, Wc)[0, 1])


res = dict(tag=A.tag, n_scenes=S, n_cells=len(CELL))
# ---------------------------------------------------------------- I0
st = float(np.mean([rows[s]["land"][f"START{p}"] == p for s in range(S) for p in PINS]))
reach = float(np.mean([a_ == c["p"] for c in CELL for a_ in c["anch"]]))
race = float(np.mean([T1(rows[s], f"Ep{p}@0", p, "51") == T1(rows[s], f"Ep{p}@1", p, "51") for s in range(S) for p in PINS])) if NSH > 1 else 1.0
try:
    J73 = json.load(open(f"{R3}/s873_{A.s873}_transplant.json")); r73 = J73["rows"]; lm73 = J73["late_mean"]
    def T173(r, key, p):
        z = np.asarray(r["zl"][key]["51"], dtype=float)[:, lm73["51"]].mean(1).mean(0); return z[p - 2] - z[p - 1]
    a73 = [T173(r73[c["si"]], f"Ep{c['p']}", c["p"]) - T173(r73[c["si"]], f"Ep{c['p']}_E_BANDR", c["p"]) for c in CELL if c["j"] == 0]
    a77 = [c["Wc51"] for c in CELL if c["j"] == 0]
    repro = float(np.corrcoef(a73, a77)[0, 1]) if len(a73) > 2 else float("nan")
except FileNotFoundError:
    repro = float("nan")
I0 = dict(t1=bool((J["t1_menu"] or 0) >= .95), START=bool(st >= .95), pin_reach=bool(reach >= .90), no_race=bool(race >= .95),
          reproducible=bool(repro >= .90))
res["I0"] = dict(**I0, all=bool(all(I0.values())), t1_value=J["t1_menu"], START_acc=st, reach=reach, ref_equal=race, repro_r=repro)
ok = res["I0"]["all"]
# ---------------------------------------------------------------- C1 / C2
desc = {}
for L in LAYS:
    for h in ("W46", "W51"):
        est = follow(range(len(SC)), h, L)
        desc[f"{h}_L{L}"] = dict(follow_d=est[0], r_d=est[1], r_c=est[2])
c1 = bt(lambda ix: follow(ix, "W46", "51")[0]); c2 = bt(lambda ix: follow(ix, "W51", "51")[0])
res["C1"] = dict(follow_d=c1, r_d=desc["W46_L51"]["r_d"], r_c=desc["W46_L51"]["r_c"],
                 verdict=(("the context effect is carried by the scene's prefix (its pictures)" if c1["est"] >= .20 and c1["lo"] > 0 else
                           "carried by the scene's question / menu rows" if c1["est"] <= -.20 and c1["hi"] < 0 else "mixed") if ok else "not read"))
res["C2"] = dict(follow_d=c2, r_d=desc["W51_L51"]["r_d"], r_c=desc["W51_L51"]["r_c"],
                 verdict=(("the late lookup's key geometry sets the writer magnitude" if c2["est"] >= .20 and c2["lo"] > 0 else
                           "the band's processing of the scene sets it (the late keys do not)" if c2["est"] <= -.20 and c2["hi"] < 0 else "both")
                          if ok else "not read"))
res["descriptive"] = dict(by_layer=desc, mean_W={k: float(np.mean([c[k] for c in CELL])) for k in ("Wc51", "Wd51", "W46_51", "W51_51", "Wc63", "W46_63", "W51_63")})
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"]["all"], C1=[res["C1"]["verdict"], c1["est"]], C2=[res["C2"]["verdict"], c2["est"]]), indent=1))
print("->", OUT)
