"""Merge S751 parts (exp/s751_ops.py --start / --n) into the single files that exp/s751_analyze.py reads.
Usage: python s751_merge.py TAG PART_TAG [PART_TAG ...]"""
import json, sys
import numpy as np
tag, parts = sys.argv[1], sys.argv[2:]
meta, arrays, head = [], {}, None
for p in parts:
    D = f"/vlm/results/r3/s751_{p}_ops"
    j = json.load(open(D + ".json")); z = np.load(D + ".npz")
    head = head or {k: v for k, v in j.items() if k != "meta"}
    meta += j["meta"]; arrays.update({k: z[k] for k in z.files})
meta.sort(key=lambda m: m["si"])
assert len({m["si"] for m in meta}) == len(meta), "duplicate scenes"
head.update(tag=tag, meta=meta, parts=parts)
np.savez_compressed(f"/vlm/results/r3/s751_{tag}_ops.npz", **arrays)
json.dump(head, open(f"/vlm/results/r3/s751_{tag}_ops.json", "w"))
print(f"merged {len(meta)} scenes from {parts} -> s751_{tag}_ops")
