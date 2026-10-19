"""S731: write vectors at the model's rule write layer L_w (S623(b) forward-only consistency), read layer unchanged.
Ordinal prototypes and frame anchors at L_w from the same canonical calibration sentences as the read layer (r3_build.calibrate, seed 60400;
forward questions and frame-marked texts only).  Writes copies the assembled module can load:
    /vlm/results/r3/s709_{base}w{L}_cal.pt         (R3MinMD: the base calibration + Lw + cal_w)
    /vlm/results/r3/r3s731w{L}_{short}_r3.pt       (R3V1C: the earlier module's checkpoint + Lw + cal_w; smaller models only)
Usage: python s731_wcal.py MODEL TAG --base B --Lw L [--device_map balanced]
"""
from __future__ import annotations

import argparse, sys
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
import r3_build as rb

V1 = dict(lm_4b="r3s609c_4b_r3.pt", lm_8b="r3s609c_8b_r3.pt", lm_35_4b="r3s619g_35_4b_r3.pt", lm_35_9b="r3s619g_35_9b_r3.pt",
          lm_ivl3_8b="r3s616b_ivl3_8b_r3.pt", lm_q25_7b="r3s616b_q25_7b_r3.pt", lm_llov_7b="r3s616b_llov_7b_r3.pt")


def ensure(v, base_name, Lw, force=False):
    """compute and save the L_w write vectors for base_name unless they exist; returns the files written / found."""
    import os
    out = [f"/vlm/results/r3/s709_{base_name}w{Lw}_cal.pt"]
    if base_name in V1:
        short = V1[base_name].replace("r3s609c_", "").replace("r3s619g_", "").replace("r3s616b_", "").replace("_r3.pt", "")
        out.append(f"/vlm/results/r3/r3s731w{Lw}_{short}_r3.pt")
    if not force and all(os.path.exists(f) for f in out):
        return out
    base = torch.load(f"/vlm/results/r3/s709_{base_name}_cal.pt", weights_only=False)
    if base_name in V1:
        st = torch.load(f"/vlm/results/r3/{V1[base_name]}", weights_only=False, map_location="cpu")
        M = rb.Model(v, Lw, st["Lr"], st["Lrel"], st.get("reader_tag"))
    else:
        from r3_build_md import ModelMD
        M = ModelMD(v, Lw, Lw, [Lw], base["reader"])
    cal = rb.calibrate(M, nproto=16, seed=60400)
    for h in getattr(M, "_hooks", []):
        h.remove()
    cw = dict(Mb=cal["Mb"], frame=cal["frame"])
    b2 = dict(base); b2["Lw"] = Lw; b2["cal_w"] = cw
    torch.save(b2, out[0])
    if base_name in V1:
        st2 = dict(st); st2["Lw"] = Lw; st2["cal_w"] = cw
        torch.save(st2, out[1])
    return out


def main(a):
    v = Uni(a.model, device="cuda:0", attn="sdpa", device_map=a.device_map) if a.device_map else Uni(a.model, device="cuda:0", attn="sdpa")
    for p in v.model.parameters():
        p.requires_grad_(False)
    out = ensure(v, a.base, a.Lw, force=True)
    print(f"[{a.tag}] 쓰기 L{a.Lw} -- 끝 -> {' · '.join(out)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag"); ap.add_argument("--base", required=True); ap.add_argument("--Lw", type=int, required=True)
    ap.add_argument("--device_map", default=None)
    main(ap.parse_args())
