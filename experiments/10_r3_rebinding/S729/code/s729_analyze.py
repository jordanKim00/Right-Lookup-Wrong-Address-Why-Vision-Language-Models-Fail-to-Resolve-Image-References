"""S729 analysis, frozen with the pre-registration (knowledge/DECISIONS.md, 2026-10-10) before any S729 data.
    READER_ADMIT[m]:    held-out calibration scenes (procedural / natural / degraded natural): count accuracy (last-block code = N) of the
                        new reader >= the old one pooled, and >= old - .01 on each kind
    FRESH_COUNT_FIX[m]: fresh draw of the sealed synthetic design (seed 72900): END accuracy (mean over three formats) at the N where the
                        old reader failed on the sealed draw (Qwen3-VL-4B N9, Qwen3.5-9B N10) -- new reader >= old + .20 and >= R³-v1 - .05
    FRESH_NO_HARM[m]:   same draw, every other END set (the other N, training phrasing, sealed anchors) new >= old - .05; forward sets |change| <= .01
    MMIU_NO_HARM[m]:    sealed MMIU and later MMIU end-relative accuracy with the new reader >= the S725 run - .01; forward answers unchanged
Usage: python s729_analyze.py BASE   (4b | 35_9b)
"""
import json, os, sys
import numpy as np

R = "/vlm/results"
FAIL_N = {"4b": "N9", "35_9b": "N10"}


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def end_mean(res, key):
    xs = [v["rep"] for k, v in res.items() if k.endswith(f"|R|{key}") or f"|R|{key}|" in k]
    return float(np.mean(xs)) if xs else None


def main(b):
    out, chk = {}, {}
    rd = load(f"{R}/r3/s729_rd_{b}_reader.json")
    if rd:
        n, o = rd["res"]["new"], rd["res"]["old"]
        chk["READER_ADMIT"] = n["all"]["count"] >= o["all"]["count"] - 1e-12 and all(n[k]["count"] >= o[k]["count"] - .01 - 1e-12 for k in ("proc", "nat", "low"))
        print(f"[{b}] 판독기 개수 정확: 새 " + " · ".join(f"{k} {n[k]['count']:.3f}" for k in ("proc", "nat", "low", "all")) +
              " | 옛 " + " · ".join(f"{k} {o[k]['count']:.3f}" for k in ("proc", "nat", "low", "all")))
    F = {arm: load(f"{R}/s588p_syneval_fs{arm}_{b}.json") for arm in ("new", "old", "v1")}
    if all(F.values()):
        r = {arm: d["res"] for arm, d in F.items()}
        fn = FAIL_N[b]
        m = {arm: end_mean(r[arm], fn) for arm in r}
        chk["FRESH_COUNT_FIX"] = m["new"] >= m["old"] + .20 - 1e-12 and m["new"] >= m["v1"] - .05 - 1e-12
        others = [k for k in r["new"] if "|R|" in k and not k.endswith(f"|R|{fn}")]
        worst = min(r["new"][k]["rep"] - r["old"][k]["rep"] for k in others)
        fwd = max(abs(r["new"][k]["rep"] - r["new"][k]["base"]) for k in r["new"] if "|F|" in k)
        chk["FRESH_NO_HARM"] = worst >= -.05 - 1e-12 and fwd <= .01 + 1e-12
        out["fresh"] = dict(fail_n=fn, end={arm: m[arm] for arm in m},
                            other={arm: {x: end_mean(r[arm], x) for x in ("N9", "N10", "train-phr", "seal-anchor")} for arm in r}, worst_other=worst, fwd=fwd)
        print(f"[{b}] 새 추출 합성 END {fn}: 새 {m['new']:.3f} · 옛 {m['old']:.3f} · v1 {m['v1']:.3f}  | 다른 END 세트 최악 차 {worst:+.3f} · 앞 방향 최대 변화 {fwd:.3f}")
        for arm in ("new", "old", "v1"):
            print(f"[{b}]   {arm}: " + " · ".join(f"{x} {out['fresh']['other'][arm][x]:.3f}" for x in ("N9", "N10", "train-phr", "seal-anchor")
                                               if out['fresh']['other'][arm][x] is not None))
    mm = {}
    for kind, new, old in (("sealed", f"a6sl_{b}", f"a5sl_{b}"), ("later", f"a6fr_{b}", f"a5fr_{b}")):
        a, o = load(f"{R}/s582_evalrepair_{new}.json"), load(f"{R}/s582_evalrepair_{old}.json")
        if not (a and o):
            continue
        A = [x for x in a["rows"] if x["suite"] == "mmiu_pos"]; O = [x for x in o["rows"] if x["suite"] == "mmiu_pos"]
        acc = lambda rs, k: float(np.mean([x[k] == x["gold"] for x in rs]))
        mm[kind] = dict(new=acc(A, "R|rep"), old=acc(O, "R|rep"), fwd_changed=float(np.mean([x["F|rep"] != x["F|base"] for x in A])))
        print(f"[{b}] MMIU {kind}: 새 {mm[kind]['new']:.3f} · S725 {mm[kind]['old']:.3f} · 앞 방향 바뀜 {mm[kind]['fwd_changed']:.3f}")
    if len(mm) == 2:
        chk["MMIU_NO_HARM"] = all(x["new"] >= x["old"] - .01 - 1e-12 and x["fwd_changed"] <= .01 + 1e-12 for x in mm.values())
    out["mmiu"] = mm
    print(f"[{b}] 판정 {chk}")
    P = f"{R}/r3/s729_analysis.json"
    allr = load(P) or {}
    allr[b] = dict(out=out, checks=chk)
    json.dump(allr, open(P, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
