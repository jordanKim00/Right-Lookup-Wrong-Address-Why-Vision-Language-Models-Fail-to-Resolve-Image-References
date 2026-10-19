"""S733: other model families -- the earlier modules' calibrations (S616b: InternVL3-8B, Qwen2.5-VL-7B, LLaVA-OneVision-7B) in the R3MinMD
format, exactly as exp/s718_convert.py did for the five paper models (ordinal prototypes, detector, START / end / last anchors at L_b,
end reader).  Usage: python s733_convert.py  (writes /vlm/results/r3/s709_lm_{ivl3_8b,q25_7b,llov_7b}_cal.pt)"""
import torch

SRC = {"ivl3_8b": "r3s616b_ivl3_8b_r3.pt", "q25_7b": "r3s616b_q25_7b_r3.pt", "llov_7b": "r3s616b_llov_7b_r3.pt"}
for base, f in SRC.items():
    st = torch.load(f"/vlm/results/r3/{f}", weights_only=False, map_location="cpu")
    out = dict(cal=st["cal"], Lb=st["Lb"], Lw=st.get("Lw", st["Lb"]), reader=st["reader_tag"], src=f, note="v1 calibration (S616b), S733")
    torch.save(out, f"/vlm/results/r3/s709_lm_{base}_cal.pt")
    print(base, f, "Lb", out["Lb"], "reader", out["reader"], "det_space", st["cal"]["det_space"], "frames", list(st["cal"]["frame"]["mu"]))
