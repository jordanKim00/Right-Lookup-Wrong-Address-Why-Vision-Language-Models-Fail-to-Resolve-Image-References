"""The per-model R³ assembled from the S724 selection (/vlm/results/r3/s724_plan.json; rules in exp/s724_select.py + exp/s724_plan.py).
Resolve (t = N_hat + 1 - k_hat) and Rebind (frame -> START for every end-bound ordinal, address only when t != k_hat) are fixed;
Read -- detector, frame binder, count mode -- is the model's calibrated choice.  A prototype binder gives exp/r3_min_md.R3MinMD
with the chosen detector and anchor radius; the learned frame gate gives exp/r3_v1c.R3V1C (its own detector, gate and windows).
Usage: rep = r3_select.build(v, base, tag)"""
import json

from r3_min_md import R3MinMD
from r3_v1c import R3V1C

V1 = dict(lm_4b=("r3s609c_4b_r3.pt", "s609_4b"), lm_8b=("r3s609c_8b_r3.pt", "s609_8b"),
          lm_35_4b=("r3s619g_35_4b_r3.pt", "s609_35_4b"), lm_35_9b=("r3s619g_35_9b_r3.pt", "s609_35_9b"),
          lm_ivl3_8b=("r3s616b_ivl3_8b_r3.pt", "s609_ivl3_8b"), lm_q25_7b=("r3s616b_q25_7b_r3.pt", "s609_q25_7b"),
          lm_llov_7b=("r3s616b_llov_7b_r3.pt", "s609_llov_7b"))                                  # S733: other families


S724_PLAN = "/vlm/results/r3/s724_plan.json"
DEFAULT_FILE = "/vlm/exp/r3_plan_default.txt"                                         # S738: one line naming the plan file in force


def default_plan():
    """the plan file in force: S738 switches every later --assembled run to its clean-room plan by writing DEFAULT_FILE (time-stamped
    in knowledge/DECISIONS.md); without the file the registered S724 plan holds, as before."""
    try:
        p = open(DEFAULT_FILE).read().strip()
        return p or S724_PLAN
    except FileNotFoundError:
        return S724_PLAN


def choice(tag, path=None):
    """the final Read of exp/s724_plan.py (component constraints + PlanExact on the selection split), or of the plan file in force."""
    d = json.load(open(path or default_plan()))
    if tag not in d:                                                                  # e.g. S733 families before their S738 entry
        d = json.load(open(S724_PLAN))
    f = d[tag]["final"]
    if f[0] == "v1c":
        return dict(detector="v1", binder="v1_gate", count=f[3])
    return dict(detector=f[1], binder=f[2], count=f[3])


def swap_reader(rep, reader):
    """S729: the same READ with a re-calibrated end reader (same layer and site; the hooks stay where they are)."""
    import torch
    R = torch.load(f"/vlm/results/s588c_reader_{reader}.pt", weights_only=False)
    tgt = rep.inner if hasattr(rep, "inner") else rep
    assert list(R["layers"]) == list(tgt.Rlayers) and list(R["sites"]) == list(tgt.Rsites), (R["layers"], tgt.Rlayers, R["sites"], tgt.Rsites)
    tgt.RP = R["P"]
    return rep


def build(v, base, tag, path=None, reader=None, wl=None, **kw):
    """kw (e.g. ablate=...) goes to R3MinMD; the learned-gate assembly has no ablation arms.  reader: S729 end-reader override;
    wl: S731 write layer (write vectors calibrated at that layer by exp/s731_wcal.py; tests stay at the read layer)."""
    ch = choice(tag, path)
    if ch["binder"] == "v1_gate":
        assert not any(kw.values()), kw
        b0 = base[:-1] if (base.endswith("4") and base[:-1] in V1) else base         # S735: lm_x4 = the gated module + calibrated frame words
        ck, rd = V1[b0]
        if wl is not None:
            ck = f"r3s731w{wl}_" + ck.replace("r3s609c_", "").replace("r3s619g_", "").replace("r3s616b_", "")
        rep = R3V1C(v, f"/vlm/results/r3/{ck}", rd, n_mode=ch["count"], ext_cal=(f"/vlm/results/r3/s709_{base}_cal.pt" if b0 != base else None))
        return swap_reader(rep, reader) if reader else rep
    det = ch["detector"]
    rule, prm = ("abs", None) if det == "abs" else ("margin", float(det.split(":")[1]))
    rep = R3MinMD(v, f"/vlm/results/r3/s709_{base}{'' if wl is None else f'w{wl}'}_cal.pt", v2=True, det_rule=rule, det_param=prm,
                  anchor_scale=float(ch["binder"].split(":")[1]), n_mode=ch["count"], **kw)
    return swap_reader(rep, reader) if reader else rep
