"""S715-diag analysis (READ stage; knowledge/DECISIONS.md 2026-10-09 16:05).  Candidate detector rules, judged by the criteria fixed
before the measurement: recall of option-line / question reference ordinals (detected with k_hat = the word AND an end anchor in the
clause window for end-relative references) >= .98 on the sealed items, false detections on non-ordinal tokens <= .005 on the sealed
items, and the procedural (S712-input) detections unchanged.
Rules on the L_b state of a token with chart distance d1 to the nearest ordinal prototype and d2 to the second nearest:
    (a) d1 <= T (T = th_chart is R³-v2)      (b) d1 / median(d1 over the span) <= r      (c) d1 / d2 <= m      and (a0) OR (c)
Usage: python s715_analyze.py SEALED_TAG PROC_TAG
"""
import json, sys
from collections import defaultdict
import numpy as np

ORD = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]


def is_boundary(s):
    st = s.strip()
    return "\n" in s or (st != "" and st[0] in ".?!")


def window(toks, i, w=8):
    L, q = [], i - 1
    while q >= 0 and len(L) < w and not is_boundary(toks[q]):
        L.append(q); q -= 1
    R, q = [], i + 1
    while q < len(toks) and len(R) < w and not is_boundary(toks[q]):
        R.append(q); q += 1
    return L[::-1] + R


ASCALE = 1.0                                                                       # frame-anchor threshold scale (main sets it)


def tokens(rec, thr):
    """per token: (kind, word index or 0, end-relative?, anchored?) with kind in {ref, ord_other, other}."""
    toks = rec["toks"]; out = []
    for i, t in enumerate(toks):
        w = t.strip().lower()
        nxt = toks[i + 1].strip().lower() if i + 1 < len(toks) else ""
        if w in ORD and nxt in ("image", "images", "picture"):
            win = window(toks, i)
            endrel = any(toks[q].strip().lower() in ("end", "last") for q in win)
            anch = any(rec["d_end"][q] <= ASCALE * thr["end"] or rec["d_last"][q] <= ASCALE * thr["last"] for q in win)
            out.append(("ref", ORD.index(w) + 1, endrel, anch))
        elif w in ORD:
            out.append(("ord_other", ORD.index(w) + 1, False, False))
        else:
            out.append(("other", 0, False, False))
    return out


def rule_mask(rec, rule, p, th):
    d1, d2 = np.array(rec["dc"]), np.array(rec["dc2"])
    if rule == "a":
        return d1 <= p
    if rule == "b":
        return d1 / max(np.median(d1), 1e-6) <= p
    if rule == "c":
        return d1 / np.maximum(d2, 1e-6) <= p
    if rule == "a0c":
        return (d1 <= th) | (d1 / np.maximum(d2, 1e-6) <= p)


def score(recs, thr, rule, p, th, select=None):
    st = defaultdict(int)
    for rec in recs:
        kinds = tokens(rec, thr); m = rule_mask(rec, rule, p, th)
        for (kind, k, endrel, anch), hit, kh in zip(kinds, m, rec["kh"]):
            if kind == "ref":
                st["ref"] += 1; ok = bool(hit) and kh == k
                st["ref_hit"] += ok
                if endrel:
                    st["end_ref"] += 1; st["end_hit"] += ok and anch
            elif kind == "other":
                st["other"] += 1; st["fp"] += bool(hit)
    return dict(recall_end=st["end_hit"] / max(st["end_ref"], 1), recall_ref=st["ref_hit"] / max(st["ref"], 1),
                fp=st["fp"] / max(st["other"], 1), n_end=st["end_ref"], n_ref=st["ref"], n_other=st["other"])


def main(sealed, proc):
    S = json.load(open(f"/vlm/results/r3/s715_{sealed}_detdiag.json")); P = json.load(open(f"/vlm/results/r3/s715_{proc}_detdiag.json"))
    th, thr = S["th_chart"], S["thr"]
    print(f"th_chart {th:.3f} · frame thr {thr}")
    base_s = score(S["rec"], thr, "a", th, th); base_p = score(P["rec"], thr, "a", th, th)
    print(f"R³-v2 (a0): sealed {base_s} | procedural {base_p}")
    # where the misses sit: chart distance of end-relative reference ordinals that a0 misses, by word and image count
    miss = defaultdict(list)
    for rec in S["rec"]:
        for (kind, k, endrel, anch), d1, kh in zip(tokens(rec, thr), rec["dc"], rec["kh"]):
            if kind == "ref" and endrel and not (d1 <= th and kh == k):
                miss[(rec["n_img"], ORD[k - 1])].append((round(d1, 2), kh))
    for key in sorted(miss, key=lambda x: -len(miss[x]))[:10]:
        v_ = miss[key]; print(f"  missed N{key[0]} {key[1]:7s} n {len(v_):4d}  d1 median {np.median([x[0] for x in v_]):.2f}  kh≠k {np.mean([x[1] != ORD.index(key[1]) + 1 for x in v_]):.2f}")
    grids = dict(a=[th * f for f in (1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0)], b=[.2, .3, .4, .5, .6], c=[.3, .4, .5, .6, .7, .8], a0c=[.3, .4, .5, .6, .7])
    res = {}
    for rule, ps in grids.items():
        for p in ps:
            s_ = score(S["rec"], thr, rule, p, th); q_ = score(P["rec"], thr, rule, p, th)
            res[f"{rule}|{p:.3f}"] = dict(sealed=s_, proc=q_)
            ok = s_["recall_end"] >= .98 and s_["fp"] <= .005 and q_["fp"] <= base_p["fp"] + 1e-9 and abs(q_["recall_ref"] - base_p["recall_ref"]) < 1e-9
            print(f"  {rule:3s} {p:6.3f}  sealed recall_end {s_['recall_end']:.3f} · recall_ref {s_['recall_ref']:.3f} · fp {s_['fp']:.4f}"
                  f" | proc recall_ref {q_['recall_ref']:.3f} · fp {q_['fp']:.4f}  {'MEETS' if ok else ''}")
    # frame anchors: end-relative references whose window has no anchor under the R³-v2 thresholds, and anchor false hits elsewhere
    global ASCALE
    for sc in (1.0, 1.1, 1.2, 1.3, 1.5):
        ASCALE = sc; na = nr = fa = fo = 0
        for rec in S["rec"]:
            toks = rec["toks"]
            for (kind, k, endrel, anch) in tokens(rec, thr):
                if kind == "ref" and endrel:
                    nr += 1; na += not anch
            for q, t in enumerate(toks):
                if t.strip().lower() not in ("end", "last"):
                    fo += 1; fa += rec["d_end"][q] <= sc * thr["end"] or rec["d_last"][q] <= sc * thr["last"]
        res[f"anchor_scale|{sc}"] = dict(no_anchor=na / max(nr, 1), anchor_false=fa / max(fo, 1))
        print(f"  frame-anchor scale {sc:.1f}: end-relative refs without anchor {na / max(nr, 1):.3f} · anchor false hits on other tokens {fa / max(fo, 1):.4f}")
    ASCALE = 1.0
    json.dump(dict(base_sealed=base_s, base_proc=base_p, grid=res), open("/vlm/results/r3/s715_analysis.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
