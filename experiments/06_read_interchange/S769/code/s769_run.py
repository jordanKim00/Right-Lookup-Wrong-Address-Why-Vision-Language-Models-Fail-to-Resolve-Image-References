"""S769 runner: one model load, then exp/s769_end4.py at N = 7 (seed 76900, tag TAG7) and N = 8 (seed 76901, tag TAG8).
Usage: python s769_run.py MODEL TAG [--n 40]
"""
import argparse, sys
sys.path.insert(0, "/vlm/exp")
from uni import Uni
import s769_end4 as S

ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("tag"); ap.add_argument("--n", type=int, default=40)
ap.add_argument("--device_map", default="balanced")
r = ap.parse_args()
v = Uni(r.model, device="cuda:0", attn="sdpa", device_map=r.device_map)
for N in (7, 8):
    S.main(argparse.Namespace(model=r.model, tag=f"{r.tag}{N}", n=r.n, start=0, pixels=160, device_map=r.device_map, N=N,
                              seed={7: 76900, 8: 76901}[N]), v)
