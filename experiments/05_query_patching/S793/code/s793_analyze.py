"""S793 analysis (frozen with the pre-registration, before any S793 data).  Per model: landing rates of the arms; for the scenes where the
natural END answer is wrong (land_END != j), whether START given END's queries reproduces that wrong answer.
Rules: I0       NOOP reproduces START in >= .95 of rows and t1_menu >= .95.
       SUF[b]   P(ES_b -> j) - P(END -> j) >= .5 (P(START -> j) - P(END -> j))   (the start-relative query repairs the natural error)
       NEC[b]   P(SE_b -> land_END | END wrong) >= .5 and P(SE_b -> j) <= P(START -> j) - .30   (the end-relative query carries the error)
       LOC      SUF and NEC effects of look exceed those of pre by >= .20 each.
       QK       (descriptive) the target block's share at the look-band layers moves with the transplanted query: ES_look vs END, SE_look vs START.
Usage: python s793_analyze.py TAG [TAG ...] -> results/r3/s793_analysis.json"""
import json, sys
import numpy as np

R = "/vlm/results/r3"
res = {}
for tag in sys.argv[1:]:
    d = json.load(open(f"{R}/s793_{tag}_transplant.json")); rows = d["rows"]
    assert d.get("readout") in ("bare", "max")
    P = lambda arm, rr=rows: float(np.mean([r["land"][arm] == r["j"] for r in rr]))
    wrong = [r for r in rows if r["land"]["END"] != r["j"]]
    rep = lambda arm: float(np.mean([r["land"][arm] == r["land"]["END"] for r in wrong])) if wrong else None
    tshare = lambda arm: float(np.mean([np.mean([r["share"][arm][li][r["j"] - 1] for li in r["share"][arm]]) for r in rows if arm in r["share"]]))
    gap = P("START") - P("END")
    eff = {b: dict(suf=P(f"ES_{b}") - P("END"), nec_rep=rep(f"SE_{b}"), nec_drop=P("START") - P(f"SE_{b}")) for b in ("pre", "look")}
    ck = dict(I0=bool(np.mean([r["land"]["NOOP"] == r["land"]["START"] for r in rows]) >= .95 and (d.get("t1_menu") or 0) >= .95))
    for b in ("pre", "look"):
        ck[f"SUF_{b}"] = bool(gap > 0 and eff[b]["suf"] >= .5 * gap)
        ck[f"NEC_{b}"] = bool(eff[b]["nec_rep"] is not None and eff[b]["nec_rep"] >= .5 and eff[b]["nec_drop"] >= .30)
    ck["LOC"] = bool(eff["look"]["suf"] - eff["pre"]["suf"] >= .20 and (eff["look"]["nec_rep"] or 0) - (eff["pre"]["nec_rep"] or 0) >= .20)
    byk = {k: {arm: P(arm, [r for r in rows if r["k"] == k]) for arm in ("START", "END", "ES_pre", "ES_look", "SE_pre", "SE_look")} for k in (2, 3, 5, 6)}
    qk = {arm: tshare(arm) for arm in ("START", "END", "ES_look", "SE_look")}
    res[tag] = dict(n=len(rows), n_wrong=len(wrong), acc={a_: P(a_) for a_ in ("START", "END", "NOOP", "ES_pre", "ES_look", "SE_pre", "SE_look")},
                    effects=eff, by_k=byk, target_share_look=qk, t1_menu=d.get("t1_menu"), checks=ck)
    print(f"== {tag} n {len(rows)} wrong {len(wrong)} t1m {d.get('t1_menu')}  checks {ck}")
    print("   acc " + " · ".join(f"{k_} {v_:.2f}" for k_, v_ in res[tag]["acc"].items()))
    print("   effects " + " · ".join(f"{b}: suf {e['suf']:.2f} rep {e['nec_rep']} drop {e['nec_drop']:.2f}" for b, e in eff.items()))
    print("   target share (look) " + " · ".join(f"{k_} {v_:.2f}" for k_, v_ in qk.items()))
json.dump(res, open(f"{R}/s793_analysis.json", "w"), indent=1)
