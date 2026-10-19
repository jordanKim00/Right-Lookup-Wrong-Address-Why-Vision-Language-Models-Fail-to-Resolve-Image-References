"""S780 runner: one model load, then exp/s780_handoff.py at N = 7 (seed 78000, tag TAG7) and N = 8 (seed 78001, tag TAG8), 80 new scenes each.
Usage: python s780_run.py MODEL TAG [--n 80]
"""
import argparse, sys
sys.path.insert(0, "/vlm/exp")
from uni import Uni
import s780_handoff as S

ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("tag"); ap.add_argument("--n", type=int, default=80)
ap.add_argument("--device_map", default="balanced"); ap.add_argument("--cells", default="7,8")
r = ap.parse_args()
v = Uni(r.model, device="cuda:0", attn="sdpa", device_map=r.device_map)
for c in r.cells.split(","):
    N = int(c)
    S.main(argparse.Namespace(model=r.model, tag=f"{r.tag}{N}", n=r.n, pixels=160, device_map=r.device_map, N=N, phr="A", seed=78000 + (N - 7),
                              r3base="r327b", r3rc="rc_27b", r3cal=None), v)
