"""S791 analysis (frozen with the pre-registration, before any S791 data).  Per model, per arm (START, END, ADDR, FRAME, R3) and k group
(k = 4: target = literal image; near k = 2; far k in {5, 6}; all j != k = {2, 3, 5, 6}): accuracy (landed = j), target / literal share and
query cosine at the matcher heads, and the content test (swapj: the answer names the new object placed at j; swapk: placed at k).
Rules: I0     START acc >= .9 and START swapj >= .9 (the instrument reads content at a stated address).
       C1     R3 (j != k): tgt >= .5, cos >= .95, acc >= .9, swapj >= .9, swapk <= .05.
       C2     ADDR (j != k): 27B tgt <= END tgt + .10 and acc <= .5; smaller models acc >= .5.
       C3     FRAME: k = 4 acc >= .9 and swapj >= .9; j != k acc <= .1 and swapk >= .8.
       C4     27B END: near swapj >= .7, far swapj <= .4.
       C5     (descriptive, amended before data: not evidence of mediation) across the arm x k cells, Spearman(tgt, swapj).
       END_COND  (amended before data) natural END split by its own answer: swapj given a correct answer, swapk given a literal answer.
Usage: python s791_analyze.py TAG [TAG ...] -> results/r3/s791_analysis.json"""
import json, sys
import numpy as np

R = "/vlm/results/r3"
ARMS = ("START", "END", "ADDR", "FRAME", "R3")
GRP = {"k4": (4,), "near": (2,), "far": (5, 6), "jk": (2, 3, 5, 6)}


def rank(x):
    o = np.argsort(np.argsort(x)); return o.astype(float)


def cell(rows, arm, ks):
    rr = [r for r in rows if r["k"] in ks]
    g = lambda f: float(np.mean([f(r) for r in rr])) if rr else None
    out = dict(n=len(rr), acc=g(lambda r: r[f"land_{arm}"] == r["j"]), tgt=g(lambda r: r[f"tgt_{arm}"]), lit=g(lambda r: r[f"lit_{arm}"]),
               cos=g(lambda r: r[f"cos_{arm}"]), swapj=g(lambda r: r[f"swapj_{arm}"]))
    sk = [r[f"swapk_{arm}"] for r in rr if f"swapk_{arm}" in r]
    out["swapk"] = float(np.mean(sk)) if sk else None
    return out


res = {}
for tag in sys.argv[1:]:
    d = json.load(open(f"{R}/s791_{tag}_chain.json")); rows = d["rows"]
    T = {a: {g: cell(rows, a, ks) for g, ks in GRP.items()} for a in ARMS}
    big = "27b" in tag
    ck = {}
    ck["I0"] = bool(T["START"]["jk"]["acc"] >= .9 and T["START"]["jk"]["swapj"] >= .9)
    r3 = T["R3"]["jk"]
    ck["C1"] = bool(r3["tgt"] >= .5 and r3["cos"] >= .95 and r3["acc"] >= .9 and r3["swapj"] >= .9 and r3["swapk"] <= .05)
    ad, en = T["ADDR"]["jk"], T["END"]["jk"]
    ck["C2"] = bool(ad["tgt"] <= en["tgt"] + .10 and ad["acc"] <= .5) if big else bool(ad["acc"] >= .5)
    fr4, frj = T["FRAME"]["k4"], T["FRAME"]["jk"]
    ck["C3"] = bool(fr4["acc"] >= .9 and fr4["swapj"] >= .9 and frj["acc"] <= .1 and frj["swapk"] >= .8)
    if big:
        ck["C4"] = bool(T["END"]["near"]["swapj"] >= .7 and T["END"]["far"]["swapj"] <= .4)
    xs, ys = [], []
    for a in ARMS[1:]:
        for k in (2, 3, 4, 5, 6):
            c = cell(rows, a, (k,)); xs.append(c["tgt"]); ys.append(c["swapj"])
    rho = float(np.corrcoef(rank(np.array(xs)), rank(np.array(ys)))[0, 1])
    ck["C5"] = bool(rho >= .7)
    assert d.get("readout") in ("bare", "max"), f"{tag}: not read from the generated form"
    # amendment (before any S791 data): split the natural END content test by its own answer -- an address failure (the answer reads the
    # picture at the wrong address it chose) vs a content failure (right address, picture not used)
    endc = {}
    for g, ks in GRP.items():
        rr = [r for r in rows if r["k"] in ks]
        cor = [r for r in rr if r["land_END"] == r["j"]]; litr = [r for r in rr if r["land_END"] == r["k"] and r["k"] != r["j"]]
        endc[g] = dict(n_correct=len(cor), swapj_given_correct=(float(np.mean([r["swapj_END"] for r in cor])) if cor else None),
                       n_literal=len(litr), swapk_given_literal=(float(np.mean([r["swapk_END"] for r in litr])) if litr else None))
    res[tag] = dict(table=T, rho_tgt_swapj=rho, t1_menu=d.get("t1_menu"), checks=ck, end_conditional=endc)
    print(f"== {tag}  t1m {d.get('t1_menu')}  rho(tgt, swapj) {rho:.2f} (descriptive)  checks {ck}")
    print("   END by its own answer: " + " · ".join(f"{g}: correct n={c['n_correct']} swapj {c['swapj_given_correct']} / literal n={c['n_literal']} swapk {c['swapk_given_literal']}" for g, c in endc.items()))
    for a in ARMS:
        print("  " + f"{a:6s} " + " · ".join(f"{g}: acc {c['acc']:.2f} tgt {c['tgt']:.2f} lit {c['lit']:.2f} cos {c['cos']:.2f} swapj {c['swapj']:.2f}"
                                           + (f" swapk {c['swapk']:.2f}" if c["swapk"] is not None else "") for g, c in T[a].items() if c["n"]))
json.dump(res, open(f"{R}/s791_analysis.json", "w"), indent=1)
