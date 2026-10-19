# S801

Category: `01_behavior`. Natural answers: scale, answer grids, option-embedded references, text lists, other families and reasoning models.

From the main script's docstring:

> S801 (pre-registered in knowledge/DECISIONS.md before any S801 data): are correct end-relative answers reads of the target picture? Natural runs only (no edit). Per scene, N in {6, 7, 8}, k in {1, 2, 3, 4, 5, N - 1, N} (both ends and the middle; amended before data), j = N + 1 - k: END(k) and START(j), each unedited, with the picture at the target j replaced by a picture of a menu word absent from the scene (target swap), and, as a control, with the picture at another block c (c not in {j, k}, drawn per cell) replaced the same way (control swap). Readout: two-form menu readout (S794), t1. Scen

Code: `code/s801_analyze.py`, `code/s801_grounding.py`, `code/s803_mixture.py`, `code/s803p_predict.py`

Shared modules: `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s801_analysis.json`, `data/s801_analysis_27b_only_20261015.json`, `data/s801_analysis_all5.json`, `data/s801_analysis_small_9b.json`, `data/s801_s801_27b_grounding.json`, `data/s801_s801_35_4b_grounding.json`, `data/s801_s801_35_9b_grounding.json`, `data/s801_s801_4b_grounding.json`, `data/s801_s801_8b_grounding.json`, `data/s803_analysis.json`
