"""S908 analysis (rules fixed before the data; knowledge/DECISIONS.md S908 entry).  Input results/r3/s908_<tag>_bind.json.
Natural runs.  Recipient END(k) with a row subset X of its layer-27 (or 31) input <- donor END(k') (k, k' in {2, 5}, both directions).  Transfer
t(X) = P(x(arm) = x(donor)) - P(x(recipient) = x(donor)) pooled over the two directions, scaled by the closure: s(X) = t(X) / t(ALL27).
Scene bootstrap 4,000 (seed 908).
I0: t1 >= .95; END(2) and END(5) answer differently in >= .80 of scenes; closure: ALL27 reproduces the donor's answer in >= .98 of runs.
  Nothing is read if I0 fails.
B1 (where the binding input sits at layer 27): among ORDPRE, FRM, QM, MENU, READ the largest s at layer 27; s >= .60 with the lower bound > .40 ->
  'at layer 27 the k-specific END transform still depends mainly on <subset>'; else 'distributed'.
B2 (timing of the ordinal rows): s(ORDPRE27) - s(ORDPRE31) >= .30 with the interval above 0 -> 'the ordinal rows are read between layers 27
  and 31'; else 'not observed'.
Descriptive: every s per direction, answer distributions, the late block-score profile correlation with the donor (L51, L63).
Usage: python3 s908_analyze.py TAG"""
from __future__ import annotations

import collections, json, sys
import numpy as np

TAG = sys.argv[1]
J = json.load(open(f"/vlm/results/r3/s908_{TAG}_bind.json"))
OUT = f"/vlm/results/r3/s908_analysis_{TAG}.json"
rng = np.random.default_rng(908); NB = 4000
rows = J["rows"]; S = len(rows); lm = J["late_mean"]
SUBS = ("ORDPRE27", "FRM27", "QM27", "MENU27", "READ27", "ALL27", "ORDPRE31")
DIRS = ((2, 5), (5, 2))


def prof(r, k_, L):
    return np.asarray(r["zl"][k_][L], dtype=float)[:, lm[L]].mean(1).mean(0)


REC = []
for si, r in enumerate(rows):
    for rk, dk in DIRS:
        d = dict(si=si, rk=rk, dk=dk, xr=r["land"][f"E{rk}"], xd=r["land"][f"E{dk}"])
        for X in SUBS:
            key = f"E{rk}_{X}_from{dk}"
            d[X] = r["land"][key]
            d[f"r51_{X}"] = float(np.corrcoef(prof(r, key, "51"), prof(r, f"E{dk}", "51"))[0, 1])
        REC.append(d)


def bt(fn):
    est = fn(np.arange(S)); bs = [fn(rng.integers(0, S, S)) for _ in range(NB)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return dict(est=float(est), lo=float(lo), hi=float(hi))


def pick(ix):
    cnt = collections.Counter(int(i) for i in ix)
    return [d for d in REC for _ in range(cnt.get(d["si"], 0))]


def t_(ix, X):
    ds = pick(ix); return float(np.mean([d[X] == d["xd"] for d in ds]) - np.mean([d["xr"] == d["xd"] for d in ds]))


s_ = lambda ix, X: t_(ix, X) / t_(ix, "ALL27")
res = dict(tag=TAG, n_scenes=S, subsets=J.get("subsets"))
diff = float(np.mean([r["land"]["E2"] != r["land"]["E5"] for r in rows]))
clo = float(np.mean([d["ALL27"] == d["xd"] for d in REC]))
I0 = dict(t1=bool((J["t1_menu"] or 0) >= .95), answers_differ=bool(diff >= .80), closure=bool(clo >= .98))
res["I0"] = dict(**I0, all=bool(all(I0.values())), t1_value=J["t1_menu"], differ=diff, closure_value=clo)
ok = res["I0"]["all"]
CAND = ("ORDPRE27", "FRM27", "QM27", "MENU27", "READ27")
allx = np.arange(S)
top = max(CAND, key=lambda X: s_(allx, X))
b1 = bt(lambda ix: s_(ix, top))
res["B1"] = dict(s={X: s_(allx, X) for X in CAND}, top=top, s_top=b1,
                 verdict=((f"at layer 27 the k-specific END transform still depends mainly on {top[:-2]}" if b1["est"] >= .60 and b1["lo"] > .40 else "distributed")
                          if ok else "not read"))
b2 = bt(lambda ix: s_(ix, "ORDPRE27") - s_(ix, "ORDPRE31"))
res["B2"] = dict(diff=b2, verdict=(("the ordinal rows are read between layers 27 and 31" if b2["est"] >= .30 and b2["lo"] > 0 else "not observed") if ok else "not read"))
desc = {}
for rk, dk in DIRS:
    ds = [d for d in REC if d["rk"] == rk]
    desc[f"E{rk}_from{dk}"] = {X: dict(P_donor=float(np.mean([d[X] == d["xd"] for d in ds])), P_recipient=float(np.mean([d[X] == d["xr"] for d in ds])),
                                       dist=dict(collections.Counter(d[X] for d in ds)), r51=float(np.mean([d[f"r51_{X}"] for d in ds]))) for X in SUBS}
    desc[f"E{rk}_from{dk}"]["base"] = dict(P_recipient_equals_donor=float(np.mean([d["xr"] == d["xd"] for d in ds])))
res["descriptive"] = desc
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"]["all"], B1=[res["B1"]["verdict"], {X: round(v, 3) for X, v in res["B1"]["s"].items()}], B2=[res["B2"]["verdict"], b2["est"]]), indent=1))
print("->", OUT)
