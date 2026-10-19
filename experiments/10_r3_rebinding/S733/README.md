# S733

Category: `10_r3_rebinding`. R3 (Read, Resolve, Rebind): calibration, Read selection, the two writes, their mechanism tests and other families.

From the main script's docstring:

> S733: other model families -- the earlier modules' calibrations (S616b: InternVL3-8B, Qwen2.5-VL-7B, LLaVA-OneVision-7B) in the R3MinMD format, exactly as exp/s718_convert.py did for the five paper models (ordinal prototypes, detector, START / end / last anchors at L_b, end reader). Usage: python s733_convert.py (writes /vlm/results/r3/s709_lm_{ivl3_8b,q25_7b,llov_7b}_cal.pt)

Code: `code/s733_analyze.py`, `code/s733_convert.py`

Shared modules: none

Data: `data/lure_xfam.json`, `data/s733_analysis.json`
