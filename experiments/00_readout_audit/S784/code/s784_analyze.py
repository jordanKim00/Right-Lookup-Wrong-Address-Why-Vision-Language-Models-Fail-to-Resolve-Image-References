"""S784 analysis, frozen with the pre-registration (knowledge/DECISIONS.md, 2026-10-14 18:4x) before any S784 data.
S784 re-runs S643 (all ops, seed 64300) and S660 (seed 66000) with R3_READ=bare: content answers are read from the bare tokens (primary); the old
space-token readout (ans_space / *_space) and free generation's first word (ans_free / *_free) are recorded alongside.
Per model and content op (Start, End, RelO+, RelO-, RelC+, RelC-, EndC, RelOx+, RelOx-):
AGREE     share of questions where the bare answer equals the free answer >= .95 (instrument check).
SPACE_OK  share where the space answer equals the bare answer >= .95 (if it holds, the old S643 numbers for that model and op stand).
Same-scene (S660): comp | both (index and retrieval right) under the bare readout, and SPACE_OK on the composed and retrieval questions.
Descriptive: accuracy and literal share under each readout; free answers outside the menu.
Usage: python s784_analyze.py   (reads every results/r3/s643_s784*_relimg.json and s660_s784ss*_samescene.json)
"""
import glob, json, os
import numpy as np

CONTENT = ("Start", "End", "RelO+", "RelO-", "RelC+", "RelC-", "EndC", "RelOx+", "RelOx-")


def relimg(files):
    out = {}
    for f in files:
        d = json.load(open(f)); m = d["tag"].replace("s784x_", "").replace("s784_", "")
        o = out.setdefault(m, {})
        for op in CONTENT:
            R = [r for r in d["rows"] if r["op"] == op and "ans_space" in r]
            if not R:
                continue
            bare = [r["ans"] for r in R]; sp = [r["ans_space"] for r in R]; fr = [r["ans_free"] for r in R]
            agree = float(np.mean([b.lower() == f_ for b, f_ in zip(bare, fr)])); space_ok = float(np.mean([s_ == b for s_, b in zip(sp, bare)]))
            o[op] = dict(n=len(R), agree=round(agree, 3), space_eq_bare=round(space_ok, 3), AGREE=bool(agree >= .95 - 1e-9), SPACE_OK=bool(space_ok >= .95 - 1e-9),
                         acc_bare=round(float(np.mean([r["kind"] == "correct" for r in R])), 3),
                         copy_bare=round(float(np.mean([r["kind"] == "copy" for r in R])), 3),
                         free_empty=int(sum(1 for r in R if r["ans_free"] == "")))
    return out


def samescene(files):
    out = {}
    for f in files:
        d = json.load(open(f)); m = d["tag"].replace("s784ss_", ""); R = d["rows"]
        both = [r for r in R if r["idx_ok"] and r["ret_ok"]]
        out[m] = dict(n=len(R), both_n=len(both), comp_given_both=(sum(r["comp_ok"] for r in both), len(both)),
                      surface_given_both=sum(r["comp_surface"] for r in both),
                      space_eq_bare_comp=round(float(np.mean([r["comp_space"] == r["comp"] for r in R])), 3),
                      space_eq_bare_ret=round(float(np.mean([r["ret_space"] == r["ret"] for r in R])), 3),
                      bare_eq_free_comp=round(float(np.mean([r["comp"].lower() == r["comp_free"] for r in R])), 3))
        out[m]["SPACE_OK"] = bool(min(out[m]["space_eq_bare_comp"], out[m]["space_eq_bare_ret"]) >= .95 - 1e-9)
    return out


if __name__ == "__main__":
    A = dict(relimg=relimg(sorted(glob.glob("/vlm/results/r3/s643_s784*_relimg.json"))),
             samescene=samescene(sorted(glob.glob("/vlm/results/r3/s660_s784ss*_samescene.json"))))
    for m, o in A["relimg"].items():
        print(f"[{m}] " + " · ".join(f"{op} agree {x['agree']:.2f} sp=bare {x['space_eq_bare']:.2f} acc {x['acc_bare']:.2f}" for op, x in o.items()))
    for m, o in A["samescene"].items():
        print(f"[{m}] same scene {o}")
    json.dump(A, open(os.environ.get("ANALYSIS_OUT", "/vlm/results/r3/s784_analysis.json"), "w"), indent=1)
