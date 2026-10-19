# S712

Category: `10_r3_rebinding`. R3 (Read, Resolve, Rebind): calibration, Read selection, the two writes, their mechanism tests and other families.

From the main script's docstring:

> S712: independent procedural check of R³-v2 on Qwen3.5-27B before natural benchmarks -- other scene sizes, prompt layouts and end words than S709 / S711, and forward (START) questions as the no-harm control. Pre-registration in knowledge/DECISIONS.md before any S712 data. R³-v2 frozen as registered in S711 (exp/r3_min_md.R3MinMD, v2 = True; readers and calibration exp/s709_cal.py, base r327b). Fresh procedural scenes from the dev pool (seed 71200), N in {3, 4, 6, 8, 9} (S709 / S711 used 5 and 7), 12 scenes per N; each scene gets one prompt layout (exp/s608_layout: images first / newline-separa

Code: `code/s712_analyze.py`, `code/s712_general.py`

Shared modules: `lib/r3_min_md.py`, `lib/r3_proc.py`, `lib/r3ref.py`, `lib/s608_layout.py`, `lib/uni.py`

Data: `data/s712_analysis.json`, `data/s712_gen_27b_general.json`, `data/s712_s712s_general.json`
