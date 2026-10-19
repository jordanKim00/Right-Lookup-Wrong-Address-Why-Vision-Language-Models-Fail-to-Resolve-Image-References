"""Where does R³ fall short of the start-relative wording?  Saved answers only (no model call; post-hoc descriptive analysis, 2026-10-10).
An item counts as a Read miss when R³ detects fewer ordinal sources in its end-relative form than the same Read detects in the start-relative
form of the same item (every option line carries one ordinal in both forms).  Reports Start / End / R³ accuracy on missed and complete items.
Usage: python3 exp/read_miss_split.py TAG [TAG ...]   (results/s582_evalrepair_TAG.json; suites mmiu_pos, muir)"""
import json, os, sys
import numpy as np

ROOT = "/home/storage2/ai26269345/vlm" if os.path.exists("/home/storage2/ai26269345/vlm") else "/vlm"
out = {}
for tag in sys.argv[1:]:
    rows = json.load(open(f"{ROOT}/results/s582_evalrepair_{tag}.json"))["rows"]
    for suite in ("mmiu_pos", "muir"):
        B = [r for r in rows if r["suite"] == suite]
        if not B:
            continue
        miss = [len(r["R|det"]) < len(r["F|det"]) for r in B]
        res = {}
        for lab, sel in (("missed", [i for i, m in enumerate(miss) if m]), ("complete", [i for i, m in enumerate(miss) if not m])):
            acc = lambda k: float(np.mean([B[i][k] == B[i]["gold"] for i in sel])) if sel else float("nan")
            res[lab] = dict(n=len(sel), start=acc("F|base"), end=acc("R|base"), r3=acc("R|rep"))
        out[f"{tag}|{suite}"] = res
        print(tag, suite, " · ".join(f"{k} n={v['n']} Start {100*v['start']:.1f} End {100*v['end']:.1f} R3 {100*v['r3']:.1f}" for k, v in res.items()))
json.dump(out, open(f"{ROOT}/research/ces/outputs/r3_lure_latest/read_miss_split.json", "w"), indent=1)
