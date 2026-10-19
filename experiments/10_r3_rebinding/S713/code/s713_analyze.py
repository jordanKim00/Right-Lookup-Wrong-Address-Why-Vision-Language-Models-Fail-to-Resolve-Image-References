"""S713 analysis, frozen with the pre-registration (knowledge/DECISIONS.md, 2026-10-09) before any S713 data.
Reads the S582-format records /vlm/results/s582_evalrepair_{TAG}.json written by exp/s713_mmiu.py.
Primary (first-use held-out set, --sealed: MMIU 1,082 / MuirBench 371 position items, 298 / 300 position-free controls):
    MMIU_REPAIR:  mmiu_pos|R  rep - base >= .25 and the paired 95% bootstrap interval excludes 0
    MUIR_REPAIR:  muir|R      rep - base >= .15 and the interval excludes 0
    LOCALITY:     answers changed (rep != base) on <= .01 of the forward-form items (mmiu_pos|F, muir|F) and of the position-free items
                  (mmiu_other, muir_all), each set separately
Secondary:
    FRESH_REPLICATES (later MMIU split, --fresh): mmiu_pos|R rep - base >= .25, interval excludes 0, forward changed <= .01
    CLOZE (report): MMIU visual_cloze, natively end-relative options, and its forward rewrite
    PHRASE (report): other end-relative wordings (S628 rev_style) on the development items
Also reported: rescues / harms, the per-task table, and the stage ledger of the end-relative form from the saved plans (ordinals
detected, k_hat = the ordinal word, end anchor found, N_hat = number of images, ordinal write / frame-only write).
Usage: python s713_analyze.py --sealed TAG [--fresh TAG] [--cloze TAG] [--phrase TAG,TAG,...]
"""
import argparse, json
from collections import defaultdict
import numpy as np

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]


def load(tag):
    return json.load(open(f"/vlm/results/s582_evalrepair_{tag}.json"))


def paired(rows, form):
    x0 = np.array([r[f"{form}|base"] == r["gold"] for r in rows], float); x1 = np.array([r[f"{form}|rep"] == r["gold"] for r in rows], float)
    dd = x1 - x0; rng = np.random.default_rng(0)
    bs = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(4000)]
    ch = float(np.mean([r[f"{form}|base"] != r[f"{form}|rep"] for r in rows]))
    return dict(n=len(rows), base=float(x0.mean()), rep=float(x1.mean()), d=float(dd.mean()), lo=float(np.percentile(bs, 2.5)),
                hi=float(np.percentile(bs, 97.5)), rescue=int(((x0 == 0) & (x1 == 1)).sum()), harm=int(((x0 == 1) & (x1 == 0)).sum()), changed=ch)


def ledger(rows, form="R"):
    L = defaultdict(int)
    for r in rows:
        det, codes = r.get(f"{form}|det"), r.get(f"{form}|codes")
        if det is None:
            continue
        L["items"] += 1; L["detected_items"] += bool(det); L["sources"] += len(det)
        N = len(codes) if codes else None
        for tok, kh, t, anch, wmax, act, frame, attn in det:
            w = tok.strip().lower()
            L["kh_word"] += (w in ORDW and ORDW.index(w) + 1 == kh)
            L["anchor"] += int(anch)
            L["Nhat_ok"] += int(anch) and N is not None and max(codes) == N
            L["ord_write"] += int(act)
            L["frame_only"] += (not act) and frame is not None
    return dict(L)


def show(name, rows, forms):
    out = {}
    for f in forms:
        S = [r for r in rows if f"{f}|base" in r]
        if not S:
            continue
        p = paired(S, f); out[f] = p
        print(f"  {name:>10s} / {f}  n {p['n']:4d}  {p['base']:.3f} → {p['rep']:.3f}  Δ {p['d']:+.3f} [{p['lo']:+.3f}, {p['hi']:+.3f}]"
              f"  구출 {p['rescue']} · 손상 {p['harm']} · 바뀜 {p['changed']:.3f}", flush=True)
    return out


def suite(d, s):
    return [r for r in d["rows"] if r["suite"] == s]


def main(a):
    res, checks = {}, {}
    d = load(a.sealed); print(f"[sealed {a.sealed}]")
    R = {s: show(s, suite(d, s), ("F", "R") if s in ("mmiu_pos", "muir") else ("O",)) for s in ("mmiu_pos", "muir", "mmiu_other", "muir_all")}
    for s in ("mmiu_pos", "muir"):
        print(f"  {s} 원장(R): {ledger(suite(d, s))}")
        bt = defaultdict(list)
        for r in suite(d, s):
            bt[r["task"]].append(r)
        print(f"  {s} 과제별 R: " + " · ".join(f"{t[:14]} {np.mean([x['R|base'] == x['gold'] for x in v]):.2f}→{np.mean([x['R|rep'] == x['gold'] for x in v]):.2f}"
                                             for t, v in sorted(bt.items())))
    res["sealed"] = R
    checks["MMIU_REPAIR"] = R["mmiu_pos"]["R"]["d"] >= .25 - 1e-9 and R["mmiu_pos"]["R"]["lo"] > 0
    checks["MUIR_REPAIR"] = R["muir"]["R"]["d"] >= .15 - 1e-9 and R["muir"]["R"]["lo"] > 0
    checks["LOCALITY"] = all(R[s][f]["changed"] <= .01 + 1e-9 for s, f in (("mmiu_pos", "F"), ("muir", "F"), ("mmiu_other", "O"), ("muir_all", "O")) if f in R.get(s, {}))
    if a.fresh:
        d = load(a.fresh); print(f"[fresh {a.fresh}]")
        F = show("mmiu_pos", suite(d, "mmiu_pos"), ("F", "R")); res["fresh"] = F
        print(f"  원장(R): {ledger(suite(d, 'mmiu_pos'))}")
        checks["FRESH_REPLICATES"] = F["R"]["d"] >= .25 - 1e-9 and F["R"]["lo"] > 0 and F["F"]["changed"] <= .01 + 1e-9
    if a.cloze:
        d = load(a.cloze); print(f"[cloze {a.cloze}]")
        res["cloze"] = show("cloze", suite(d, "cloze"), ("F", "R"))
    for t in (a.phrase.split(",") if a.phrase else []):
        d = load(t); print(f"[phrase {t}]")
        res[f"phrase_{t}"] = {s: show(s, suite(d, s), ("F", "R")) for s in ("mmiu_pos", "muir")}
        for s in ("mmiu_pos", "muir"):
            print(f"  {s} 원장(R): {ledger(suite(d, s))}")
    print(f"checks {checks}")
    json.dump(dict(res=res, checks=checks, tags=vars(a)), open("/vlm/results/r3/s713_analysis.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sealed", required=True); ap.add_argument("--fresh"); ap.add_argument("--cloze"); ap.add_argument("--phrase")
    main(ap.parse_args())
