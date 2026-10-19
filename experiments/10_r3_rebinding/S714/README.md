# S714

Category: `10_r3_rebinding`. R3 (Read, Resolve, Rebind): calibration, Read selection, the two writes, their mechanism tests and other families.

From the main script's docstring:

> S714: the k = N boundary route of Qwen3.5-27B -- how "the N-th image from the last" (= the first image) is answered although it does not use the end anchor (S701 y = 5, S703 k = 7 - m). Pre-registration in knowledge/DECISIONS.md before any S714 data. Descriptive lead (S705a saved arrays, re-read 2026-10-09): for END(k = 7) the answer-row image selection of the full-attention heads sits on the last block at L35-43 (block 7 share .30-.35, block 1 .08-.12) and moves to block 1 only at L51-55 (.26-.37; L55 matcher heads up to .58), while END(k = 2) is decided at L35-47 (S705b) and later selection 

Code: `code/s714_analyze.py`, `code/s714_kn.py`

Shared modules: `lib/c_tools.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s714_analysis.json`, `data/s714_kn_27b_kn.json`
