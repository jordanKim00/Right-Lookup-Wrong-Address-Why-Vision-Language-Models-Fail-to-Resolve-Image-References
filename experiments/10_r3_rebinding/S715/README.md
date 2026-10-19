# S715

Category: `10_r3_rebinding`. R3 (Read, Resolve, Rebind): calibration, Read selection, the two writes, their mechanism tests and other families.

From the main script's docstring:

> S715-diag: why R³-v2's ordinal detector misses option-line ordinals on MMIU / MuirBench (27B) -- a READ-stage measurement on the first-use (sealed) items, which are development data for the READ fix from now on (knowledge/DECISIONS.md 2026-10-09 16:00). For every input the S582 harness builds (same items, prompts and order; forward and end-relative forms), the capture forward of R³-v2 (exp/r3_min_md.R3MinMD) is run once and, for every token of the text span after the last image, the L_b state's distances are saved: nearest ordinal prototype and its distance in the detector chart and in the ful

Code: `code/s715_analyze.py`, `code/s715_detdiag.py`

Shared modules: `lib/r3_min_md.py`, `lib/r3_proc.py`, `lib/r3ref.py`, `lib/s588c_reader.py`, `lib/s608_layout.py`, `lib/uni.py`

Data: `data/s715_analysis.json`, `data/s715_d715_27b_detdiag.json`, `data/s715_d715c_27b_detdiag.json`, `data/s715_d715p_27b_detdiag.json`, `data/s715_d715s_detdiag.json`
