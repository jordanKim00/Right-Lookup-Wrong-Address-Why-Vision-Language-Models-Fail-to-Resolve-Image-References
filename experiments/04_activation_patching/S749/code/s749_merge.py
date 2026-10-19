"""Merge S749 parts (exp/s749_swapk.py --start / --n) into the single files that exp/s749_analyze.py reads.
Usage: python s749_merge.py TAG PART_TAG [PART_TAG ...]"""
import json, sys
import numpy as np
tag, parts = sys.argv[1], sys.argv[2:]
meta, arrays, head = [], {}, None
for p in parts:
    D = f"/vlm/results/r3/s749_{p}_swapk"
    j = json.load(open(D + ".json")); z = np.load(D + ".npz")
    head = head or {k: v for k, v in j.items() if k != "meta"}
    meta += j["meta"]; arrays.update({k: z[k] for k in z.files})
meta.sort(key=lambda m: m["si"])
assert len({m["si"] for m in meta}) == len(meta), "duplicate scenes"
head.update(tag=tag, meta=meta, parts=parts)
np.savez_compressed(f"/vlm/results/r3/s749_{tag}_swapk.npz", **arrays)
json.dump(head, open(f"/vlm/results/r3/s749_{tag}_swapk.json", "w"))
print(f"merged {len(meta)} scenes from {parts} -> s749_{tag}_swapk")
