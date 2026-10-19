"""S843 analysis (rules fixed in knowledge/DECISIONS.md before any S843 data).  Word lists, N = 6-9, natural END(k) / START(k); five models.
Per model: I0 = t1 >= .95 and START(k) -> k >= .90 pooled over cells.  On the S801 cell set (N in {6, 7, 8}, k in {1..5, N - 1, N}) and pooled:
non-identity target share (END(k) -> N + 1 - k), copy share (END(k) -> k), identity accuracy (N7 k4; and N9 k5 reported separately).
Verdicts (models passing I0):
  TCOPY   each smaller model (Qwen3.5-4B / 9B, Qwen3-VL-4B / 8B): copy share >= .30 and copy > target -> 'copies in text'.
  T27     Qwen3.5-27B: target share >= .45 with copy share <= .15 -> 'corrects in text'; and identity accuracy (N7 k4 and N9 k5 pooled) <= .70 ->
          'the identity reference collapses in text'.
  SAME    per model, |copy share(text) - copy share(pictures, S801 same cells)| <= .20 and |target share(text) - target share(pictures)| <= .20 ->
          'the same pattern as pictures' (VL-4B's picture values are descriptive only - it missed S801's instrument check).
Descriptive: the full N x k answer grids (for the text counterpart of fig_grid).
Usage: python exp/s843_analyze.py [TAG ...]  ->  results/r3/s843_analysis.json"""
from __future__ import annotations

import json, sys
import numpy as np

V = "/home/storage2/ai26269345/vlm"
TAGS = [("s843_4b", "Qwen3-VL-4B", "s801_4b"), ("s843_8b", "Qwen3-VL-8B", "s801_8b"), ("s843_35_4b", "Qwen3.5-4B", "s801_35_4b"),
        ("s843_35_9b", "Qwen3.5-9B", "s801_35_9b"), ("s843_27b", "Qwen3.5-27B", "s801_27b")]
CELLS = [(N, k) for N in (6, 7, 8) for k in sorted({1, 2, 3, 4, 5, N - 1, N})]


def pic_stats(tag):
    try:
        R = json.load(open(f"{V}/results/r3/s801_{tag}_grounding.json"))["rows"]
    except FileNotFoundError:
        return None
    non = [r for r in R if r["k"] != r["j"]]
    return dict(target=float(np.mean([r["END"]["land"] == r["j"] for r in non])), copy=float(np.mean([r["END"]["land"] == r["k"] for r in non])))


def main(*tags):
    out = dict(models={})
    for tag, name, ptag in TAGS:
        if tags and tag not in tags:
            continue
        try:
            d = json.load(open(f"{V}/results/r3/s843_{tag}_textgrid.json"))
        except FileNotFoundError:
            continue
        R = d["rows"]
        land = lambda r, N, k, fr: r["land"][f"N{N}_k{k}_{fr}"]
        st = float(np.mean([land(r, N, k, "S") == k for r in R for N in (6, 7, 8, 9) for k in range(1, N + 1)]))
        i0 = dict(t1=d["t1_menu"], start_acc=st, ok=bool(d["t1_menu"] >= .95 and st >= .90))
        non = [(r, N, k) for r in R for N, k in CELLS if k != N + 1 - k]
        tgt = float(np.mean([land(r, N, k, "E") == N + 1 - k for r, N, k in non]))
        cpy = float(np.mean([land(r, N, k, "E") == k for r, N, k in non]))
        idn7 = float(np.mean([land(r, 7, 4, "E") == 4 for r in R])); idn9 = float(np.mean([land(r, 9, 5, "E") == 5 for r in R]))
        grid = {f"N{N}": {str(k): np.bincount([land(r, N, k, "E") for r in R], minlength=N + 1).tolist() for k in range(1, N + 1)} for N in (6, 7, 8, 9)}
        m = dict(name=name, I0=i0, target=tgt, copy=cpy, identity_N7=idn7, identity_N9=idn9, grid=grid, pictures=pic_stats(ptag))
        if i0["ok"]:
            if name != "Qwen3.5-27B":
                m["TCOPY"] = "copies in text" if cpy >= .30 and cpy > tgt else "not shown"
            else:
                m["T27"] = dict(corrects=("corrects in text" if tgt >= .45 and cpy <= .15 else "not shown"),
                                identity=("the identity reference collapses in text" if (idn7 + idn9) / 2 <= .70 else "not shown"))
            if m["pictures"]:
                m["SAME"] = ("the same pattern as pictures" if abs(cpy - m["pictures"]["copy"]) <= .20 and abs(tgt - m["pictures"]["target"]) <= .20
                             else "different from pictures")
        out["models"][tag] = m
        print(f"== {name}: I0 {i0} | target {tgt:.3f} copy {cpy:.3f} identity N7 {idn7:.3f} N9 {idn9:.3f} | pictures {m['pictures']} | "
              f"{m.get('TCOPY', '')}{m.get('T27', '')} | {m.get('SAME', '')}")
    json.dump(out, open(f"{V}/results/r3/s843_analysis.json" if not tags else f"{V}/results/r3/s843_analysis_{'_'.join(tags)}.json", "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
