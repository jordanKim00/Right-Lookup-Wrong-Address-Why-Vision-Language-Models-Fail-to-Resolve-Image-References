"""S772 runner: one model load, then exp/s772_faith.py for the cells (N, phrasing) = (7, A), (8, A), (7, B), (8, B); tags TAG7A ... TAG8B.
Usage: python s772_run.py MODEL TAG [--n 80] [--cells 7A,8A,7B,8B]
"""
import argparse, sys
sys.path.insert(0, "/vlm/exp")
from uni import Uni
import s772_faith as S

ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("tag"); ap.add_argument("--n", type=int, default=80)
ap.add_argument("--device_map", default="balanced"); ap.add_argument("--cells", default="7A,8A,7B,8B")
r = ap.parse_args()
v = Uni(r.model, device="cuda:0", attn="sdpa", device_map=r.device_map)
for c in r.cells.split(","):
    N, phr = int(c[:-1]), c[-1]
    S.main(argparse.Namespace(model=r.model, tag=f"{r.tag}{c}", n=r.n, pixels=160, device_map=r.device_map, N=N, phr=phr,
                              seed=77200 + 10 * N + (0 if phr == "A" else 1)), v)
