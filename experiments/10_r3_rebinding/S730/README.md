# S730

Category: `10_r3_rebinding`. R3 (Read, Resolve, Rebind): calibration, Read selection, the two writes, their mechanism tests and other families.

From the main script's docstring:

> S730: why R³ works, stage by stage, inside each model -- the index-matching heads' queries and the answer under five conditions. Pre-registration in knowledge/DECISIONS.md before any S730 data; analysis exp/s730_analyze.py (frozen with it). Scenes: procedural images (160 px), N = 7, seed 73000; the S705a question form "There are 7 images in a row. What is in the k-th image from the {first|last}?" with a one-word menu. For k in {2, ..., 6} (target j = N + 1 - k): START the forward wording of the same target ("the j-th image from the first") END the end-relative wording, unedited R3 END with the

Re-measured with the clean readout in: S786, S794 (files tagged with the re-run name are kept here).

Code: `code/s730_analyze.py`, `code/s730_stagewise.py`, `code/s730d_diag.py`, `code/s730r_recog.py`, `code/s738c_plancheck.py`

Shared modules: `lib/c_tools.py`, `lib/r3_min_md.py`, `lib/r3_proc.py`, `lib/r3_select.py`, `lib/r3ref.py`, `lib/uni.py`

Data: `data/reflect_rate.json`, `data/s730_analysis.json`, `data/s730_s786_27b_stagewise.json`, `data/s730_s786_35_4b_stagewise.json`, `data/s730_s786_35_9b_stagewise.json`, `data/s730_s786_4b_stagewise.json`, `data/s730_s786_8b_stagewise.json`, `data/s730_s794_35_4b_stagewise.json`, `data/s730_stw2_27b_stagewise.json`, `data/s730_stw_27b_stagewise.json`, `data/s730_stw_35_4b_stagewise.json`, `data/s730_stw_35_9b_stagewise.json`, `data/s730_stw_4b_heads.json`, `data/s730_stw_4b_stagewise.json`, `data/s730_stw_8b_heads.json`, `data/s730_stw_8b_stagewise.json`, `data/s730_stw_ivl3_8b_heads.json`, `data/s730_stw_ivl3_8b_stagewise.json`, `data/s730_stw_llov_7b_heads.json`, `data/s730_stw_q25_7b_heads.json`, `data/s730_stw_q25_7b_stagewise.json`, `data/s730_stwn_ivl3_8b_heads.json`, `data/s730_stwn_ivl3_8b_stagewise.json`, `data/s730_stwn_llov_7b_heads.json`, `data/s730d_dg730_llov_7b_diag.json`, `data/s738c_pc738_27b_plancheck.json`
