# S791

Category: `10_r3_rebinding`. R3 (Read, Resolve, Rebind): calibration, Read selection, the two writes, their mechanism tests and other families.

From the main script's docstring:

> S791 (pre-registered in knowledge/DECISIONS.md before any S791 data): one causal chain in the same scenes -- address / frame writes -> the lookup query -> the picture read -> the answer. The S730 / S786 design (same code for writes, captures and readout; R3_READ=bare) plus a content test for every arm: the picture at the target j, and separately at the literal image k, is replaced by a new object whose name is already in the menu (one of the two distractors); the plan computed on the original scene is reused unchanged. Scenes: procedural images (160 px), N = 7, seed 79100 (new); k in {2, ..., 

Code: `code/s791_analyze.py`, `code/s791_chain.py`

Shared modules: `lib/c_tools.py`, `lib/r3_proc.py`, `lib/r3_select.py`, `lib/uni.py`

Data: `data/s791_analysis.json`, `data/s791_s791_27b_chain.json`, `data/s791_s791_35_4b_chain.json`, `data/s791_s791_35_9b_chain.json`, `data/s791_s791_4b_chain.json`, `data/s791_s791_8b_chain.json`
