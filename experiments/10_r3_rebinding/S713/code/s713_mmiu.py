"""S713: R³-v2 with the 27B model-internal readers on natural benchmarks, through the unchanged S582 harness (exp/s582_evalrepair.run:
same items, prompts, splits, restricted option-letter argmax and paired base/repair readout as the saved small-model rows).
The repair is exp/r3_min_md.R3MinMD (R³-min plan + the S711 frame-only rule; readers / calibration exp/s709_cal.py, base r327b) --
nothing is fitted or tuned on benchmark items.  Pre-registration in knowledge/DECISIONS.md before any S713 data.
Usage: python s713_mmiu.py MODEL TAG --base B --suites S [--sealed | --fresh] [--rev_style end] [--cap n] [--v1] [--device_map balanced]
"""
from __future__ import annotations

import argparse, json, sys

sys.path.insert(0, "/vlm/exp")
import s582_evalrepair as E
from uni import Uni
from r3_min_md import R3MinMD
import r3_select

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag"); ap.add_argument("--base", required=True)
    ap.add_argument("--suites", default="mmiu_pos,muir,mmiu_other,muir_all")
    ap.add_argument("--per_task", type=int, default=50); ap.add_argument("--per_other", type=int, default=25)
    ap.add_argument("--cap", type=int, default=None)
    ap.add_argument("--sealed", action="store_true"); ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--rev_style", default="end", choices=["end", "tolast", "reverse", "backwards"])
    ap.add_argument("--v1", action="store_true")                                      # S709 rule (no frame-only write)
    ap.add_argument("--det_rule", default="abs"); ap.add_argument("--det_param", type=float, default=None)   # S716 READ rule
    ap.add_argument("--anchor_scale", type=float, default=1.0)
    ap.add_argument("--ablate", default=None, choices=[None, "no_frame", "no_ord", "rand_target", "oracle_n"])     # S721
    ap.add_argument("--n_mode", default="max", choices=["max", "last"])                                            # S724 variants
    ap.add_argument("--prompt_mode", default="none", choices=["none", "explain", "twopass"])                        # S721 baselines (S640)
    ap.add_argument("--device_map", default=None)
    ap.add_argument("--assembled", default=None); ap.add_argument("--reader_override", default=None); ap.add_argument("--wl", type=int, default=None)
    ap.add_argument("--wait_cal", action="store_true")                                                           # S734: wait for a calibration another job writes                 # S729                                                                   # S725: per-model R³ (s724 selection tag)
    a = ap.parse_args()
    E.REV_STYLE = a.rev_style
    v = Uni(a.model, device="cuda:0", attn="sdpa", device_map=a.device_map)
    for p in v.model.parameters():
        p.requires_grad_(False)
    if a.wait_cal:
        import os, time
        for _ in range(240):
            if os.path.exists(f"/vlm/results/r3/s709_{a.base}_cal.pt"):
                break
            time.sleep(30)
        time.sleep(10)
    rep = r3_select.build(v, a.base, a.assembled, ablate=a.ablate, reader=a.reader_override, wl=a.wl) if a.assembled else R3MinMD(v, f"/vlm/results/r3/s709_{a.base}_cal.pt", v2=not a.v1, det_rule=a.det_rule,
                  det_param=a.det_param, anchor_scale=a.anchor_scale, ablate=a.ablate, n_mode=a.n_mode)
    E.run(v, rep, a.tag, set(a.suites.split(",")), per_task=a.per_task, per_other=a.per_other, cap=a.cap, sealed=a.sealed, fresh=a.fresh,
          prompt_mode=a.prompt_mode)
    json.dump(dict(tag=a.tag, base=a.base, assembled=a.assembled, choice=(r3_select.choice(a.assembled) if a.assembled else None), v2=not a.v1, det_rule=a.det_rule, det_param=a.det_param, anchor_scale=a.anchor_scale, ablate=a.ablate, prompt_mode=a.prompt_mode,
                   rev_style=a.rev_style, sealed=a.sealed, fresh=a.fresh, stats=rep.stats),
              open(f"/vlm/results/r3/s713_{a.tag}_stats.json", "w"), indent=1)
    print(f"[{a.tag}] stats {rep.stats}", flush=True)
