"""S721 latency / memory: per-input wall time of the unedited forward vs R³ (capture forward + plan + repaired forward when something is
written) on MMIU development items (end-relative and forward forms), and peak GPU memory.  Usage: python s721_latency.py MODEL TAG --base B
[--per_task 10] [R³-v3 flags] [--device_map balanced]"""
import argparse, json, sys
import numpy as np
import torch
sys.path.insert(0, "/vlm/exp")
import s582_evalrepair as E
from uni import Uni
from r3_min_md import R3MinMD

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag"); ap.add_argument("--base", required=True); ap.add_argument("--per_task", type=int, default=10)
    ap.add_argument("--det_rule", default="abs"); ap.add_argument("--det_param", type=float, default=None); ap.add_argument("--anchor_scale", type=float, default=1.0)
    ap.add_argument("--device_map", default=None); ap.add_argument("--assembled", default=None)                     # S725: per-model R³
    a = ap.parse_args()
    v = Uni(a.model, device="cuda:0", attn="sdpa", device_map=a.device_map)
    for p in v.model.parameters():
        p.requires_grad_(False)
    if a.assembled:
        import r3_select
        rep = r3_select.build(v, a.base, a.assembled)
    else:
        rep = R3MinMD(v, f"/vlm/results/r3/s709_{a.base}_cal.pt", v2=True, det_rule=a.det_rule, det_param=a.det_param, anchor_scale=a.anchor_scale)
    rep.timing = []
    for d in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(d)
    E.run(v, rep, a.tag, {"mmiu_pos"}, per_task=a.per_task)
    T = rep.timing[2:]                                                               # drop warm-up
    mem = [torch.cuda.max_memory_allocated(d) / 2**30 for d in range(torch.cuda.device_count())]
    w = [t for t in T if t["wrote"]]; nw = [t for t in T if not t["wrote"]]
    res = dict(n=len(T), base_ms=1e3 * float(np.mean([t["base"] for t in T])), r3_ms=1e3 * float(np.mean([t["r3"] for t in T])),
               r3_ms_written=1e3 * float(np.mean([t["r3"] for t in w])) if w else None, r3_ms_unwritten=1e3 * float(np.mean([t["r3"] for t in nw])) if nw else None,
               tokens=float(np.mean([t["tokens"] for t in T])), peak_gib=mem)
    print(f"[{a.tag}] latency {res}", flush=True)
    json.dump(dict(tag=a.tag, model=a.model, res=res, timing=T), open(f"/vlm/results/r3/s721_{a.tag}_latency.json", "w"), indent=1)
    print(f"[{a.tag}] 끝 -> /vlm/results/r3/s721_{a.tag}_latency.json", flush=True)
