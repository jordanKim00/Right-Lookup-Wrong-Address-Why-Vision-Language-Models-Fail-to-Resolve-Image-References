"""S783 runner: re-measure S780, S769, S772, S778, S770 and S758 on Qwen3.5-27B with the generation-consistent answer readout (R3_READ=bare: argmax over the
menu words' bare tokens, which is the first word greedy generation emits whenever that word is a menu word; S782).  Same code, seeds, scenes and
frozen analysis rules as the originals; only the answer readout changes.  One model load.  Pre-registration: knowledge/DECISIONS.md (S783).
Tags: s783_27b{7,8} (S780), s783e_27b{7,8} (S769), s783f_27b{7A,8A,7B,8B} (S772), s783b_27b{7,8} (S778), s783d_27b (S770), s783r_27b (S758).
Every answer also records whether the vocabulary-wide top-1 token is a bare menu token (t1m / t1_menu): when it is, the readout equals greedy generation.
Usage: python s783_run.py MODEL [--parts 780,769,772,758] [--cells 7A,8A,7B,8B]"""
import argparse, os, sys
os.environ["R3_READ"] = "bare"
sys.path.insert(0, "/vlm/exp")
from uni import Uni
import s780_handoff, s769_end4, s772_faith, s758_paths, s778_readrows, s770_dmtg

ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--device_map", default="balanced")
ap.add_argument("--parts", default="780,769,772,778,770,758")
ap.add_argument("--cells", default="7A,8A,7B,8B")   # S772 cells to run (queue v14 re-runs 7B,8B after 7A,8A were saved; each cell sets its own seed)
r = ap.parse_args()
v = Uni(r.model, device="cuda:0", attn="sdpa", device_map=r.device_map)
NS = argparse.Namespace
for part in r.parts.split(","):
    if part == "780":
        for N in (7, 8):
            s780_handoff.main(NS(model=r.model, tag=f"s783_27b{N}", n=80, pixels=160, device_map=r.device_map, N=N, phr="A", seed=78000 + (N - 7),
                                 r3base="r327b", r3rc="rc_27b", r3cal=None), v)
    elif part == "769":
        for N in (7, 8):
            s769_end4.main(NS(model=r.model, tag=f"s783e_27b{N}", n=40, start=0, pixels=160, device_map=r.device_map, N=N,
                              seed={7: 76900, 8: 76901}[N]), v)
    elif part == "772":
        for c in r.cells.split(","):
            N, phr = int(c[:-1]), c[-1]
            s772_faith.main(NS(model=r.model, tag=f"s783f_27b{c}", n=80, pixels=160, device_map=r.device_map, N=N, phr=phr,
                               seed=77200 + 10 * N + (0 if phr == "A" else 1)), v)
    elif part == "778":
        for N in (7, 8):
            s778_readrows.main(NS(model=r.model, tag=f"s783b_27b{N}", n=40, pixels=160, device_map=r.device_map, N=N, phr="A", seed=77800 + (N - 7)), v)
    elif part == "770":
        s770_dmtg.main(NS(model=r.model, tag="s783d_27b", n=40, pixels=160, device_map=r.device_map, seed=77000), v)
    elif part == "758":
        s758_paths.main(NS(model=r.model, tag="s783r_27b", n=40, start=0, pixels=160, device_map=r.device_map, check=False), v)
print("S783 끝", flush=True)
