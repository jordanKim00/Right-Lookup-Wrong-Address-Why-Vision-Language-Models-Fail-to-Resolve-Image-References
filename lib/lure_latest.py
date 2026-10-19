"""Lure analysis of the latest R³ on MMIU position items (saved answers only, no model call).

Items are rebuilt with the selection rule of exp/s582_evalrepair.run (dev: first 50 per task, sealed: 50-150, fresh: 150-250
without shared images, N <= 9) and aligned with the saved rows by task and gold letter. In the end-relative form, option L that
names forward image k reads "the (N+1-k)-th image from the end". A literal reader maps it to image N+1-k, so the lure is the
option naming image N+1-p, where p is the gold image: its written ordinal equals the target's forward position (same definition
as FINDINGS §659 / §750). Items without a lure option, or whose lure is the gold option, are excluded.

Usage: python3 exp/lure_latest.py TAG:SPLIT [...] --out research/ces/outputs/r3_lure_latest/lure_latest.json
"""
from __future__ import annotations

import argparse, json, os, re

HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))         # archive/legacy_2026-09-08
REPO = os.path.realpath(os.path.join(HERE, "..", ".."))
ROOT = os.path.join(REPO, "assets", "mmiu")
POS_TASKS = ["face_retrieval", "handwritten_retrieval", "image2image_retrieval", "person_reid", "vehicle_retrieval",
             "sketch2image_retrieval", "text2image_retrieval", "forensic_detection_blink", "forensic_detection_forgerynet",
             "visual_coherence", "Icon_Question_Answering_with_Spatial_Context"]
ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
PAT = re.compile(r"^\s*([A-H])\s*:\s*[Tt]he (" + "|".join(ORDW) + r") image\s*$")


def items(split):
    out = []
    for task in POS_TASKS:
        keep = []
        for it in json.load(open(os.path.join(ROOT, f"{task}.json"))):
            ls = [l for l in str(it.get("options", "")).split("\n") if l.strip()]
            if ls and all(PAT.match(l) for l in ls):
                keep.append((it, [(PAT.match(l).group(1), ORDW.index(PAT.match(l).group(2)) + 1) for l in ls]))
        if split == "fresh":
            used = {p for it, _ in keep[:150] for p in it["input"]["input_image_path"]}
            sel = [(it, o) for it, o in keep[150:250] if not used & set(it["input"]["input_image_path"])]
        else:
            sel = keep[50:150] if split == "sealed" else keep[:50]
        for it, opts in sel:
            N = len(it["input"]["input_image_path"])
            if N > 9:
                continue
            out.append(dict(task=task, gold=str(it["output"]["output_text"]).strip()[:1], N=N, opts=opts))
    return out


def analyse(tag, split):
    rows = [r for r in json.load(open(os.path.join(REPO, "results", f"s582_evalrepair_{tag}.json")))["rows"] if r["suite"] == "mmiu_pos"]
    its = items(split)
    assert len(rows) == len(its), (tag, len(rows), len(its))
    n = err = lure_err = 0; chance = []; resc_l = resc_o = 0; after = after_l = 0; new_l = 0
    for r, it in zip(rows, its):
        assert r["task"] == it["task"] and r["gold"] == it["gold"], (tag, r["task"], it["task"])
        img = dict(it["opts"])                                            # letter -> forward image
        p = img.get(it["gold"])
        lure = [L for L, k in it["opts"] if k == it["N"] + 1 - p]
        if p is None or not lure or lure[0] == it["gold"]:
            continue
        lure = lure[0]; n += 1
        b, a = r["R|base"], r["R|rep"]
        if b != it["gold"]:
            err += 1; chance.append(1 / (len(it["opts"]) - 1))
            if b == lure:
                lure_err += 1; resc_l += a == it["gold"]
            else:
                resc_o += a == it["gold"]
        if a != it["gold"]:
            after += 1; after_l += a == lure
            new_l += a == lure and b != lure
    oth = err - lure_err
    return dict(tag=tag, split=split, items=n, base_acc=1 - err / n, errors=err, lure_share=lure_err / err,
                chance=sum(chance) / len(chance), rescued_lure=resc_l / lure_err, rescued_other=resc_o / oth if oth else None,
                errors_after=after, lure_share_after=after_l / after if after else None, new_lure=new_l)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = [analyse(*x.split(":")) for x in a.runs]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    for d in res:
        print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in d.items()))
