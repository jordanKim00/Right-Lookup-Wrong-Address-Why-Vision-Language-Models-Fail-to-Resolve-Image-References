"""S787 runner: re-measure, on new scenes and with the generation-consistent answer readout (R3_READ=bare, FINDINGS §846), the experiments behind
the copy/compose figure (S545 code shift +2, S549 last code -1 / +1, S474b forward-ordinal transplant), the frame-before-ordinal test (S658) and
the frame-write test (S736).  Same code and analyses as the originals; only the readout and the seeds change.  One model load.
Pre-registration: knowledge/DECISIONS.md (S787).
Usage: python s787_run.py MODEL TAG --base B --rc RC --heads JSON [--device_map balanced] [--parts 545,549,474,658,736]"""
import argparse, os, sys
os.environ["R3_READ"] = os.environ.get("R3_READ", "max")          # S794: max = the generated form in either tokenization
assert os.environ["R3_READ"] in ("bare", "max")
sys.path.insert(0, "/vlm/exp")
from uni import Uni
import s545_indexshift, s549_endshift, s474b_fwddonor, s658_preposed, s736_frame

ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("tag")
ap.add_argument("--base", required=True); ap.add_argument("--rc", required=True); ap.add_argument("--heads", required=True)
ap.add_argument("--device_map", default=None); ap.add_argument("--parts", default="545,549,474,658,736")
r = ap.parse_args()
v = Uni(r.model, device="cuda:0", attn="sdpa", device_map=r.device_map) if r.device_map else Uni(r.model, device="cuda:0", attn="sdpa")
for p in v.model.parameters():
    p.requires_grad_(False)
T = f"s787_{r.tag}"; NS = argparse.Namespace
for part in r.parts.split(","):
    if part == "545":
        s545_indexshift.run(v, tag=T, ntest=8, seed=78700)
    elif part == "549":
        s549_endshift.run(v, tag=T, ntest=8, seed=78710)
    elif part == "474":
        s474b_fwddonor.run(v, tag=T, npair=96, seed=78720)
    elif part == "658":
        s658_preposed.main(NS(model=r.model, tag=T, n=80, pixels=160, device_map=r.device_map, seed=78730), v)
    elif part == "736":
        s736_frame.main(NS(model=r.model, tag=T, base=r.base, rc=r.rc, heads=r.heads, wl=None, reader_override=None, n=20, pixels=160,
                           device_map=r.device_map, seed=78740), v)
    print(f"[S787 {r.tag}] {part} done", flush=True)
print("S787 끝", flush=True)
