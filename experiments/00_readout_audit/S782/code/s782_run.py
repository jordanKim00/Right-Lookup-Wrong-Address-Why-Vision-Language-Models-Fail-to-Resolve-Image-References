"""S782 runner: one model load, N = 7 then N = 8 (S780's scenes).  Usage: python s782_run.py MODEL TAG [--n 80]"""
import argparse, sys
sys.path.insert(0, "/vlm/exp")
from uni import Uni
import s782_freegen as S

ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("tag"); ap.add_argument("--n", type=int, default=80)
ap.add_argument("--device_map", default="balanced")
r = ap.parse_args()
v = Uni(r.model, device="cuda:0", attn="sdpa", device_map=r.device_map)
for N in (7, 8):
    S.main(argparse.Namespace(model=r.model, tag=f"{r.tag}{N}", n=r.n, pixels=160, device_map=r.device_map, N=N, seed=78000 + (N - 7),
                              r3base="r327b", r3rc="rc_27b"), v)
