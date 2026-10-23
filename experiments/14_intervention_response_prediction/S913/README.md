# S913

Category: `14_intervention_response_prediction`. Predicting intervention outcomes: fitted profile predictors (S913-S925) and the parameter-free intervention-response prediction from the model's gradients and path responses (S926-S932), with the circuit-restricted test.

From the main script's docstring:

> S913 / S914 data collector (registered in knowledge/DECISIONS.md before the data; the advisor's frozen-prediction design, 10-22) - does the early-address profile carry enough information to predict the later writer computation? Natural runs (no pins) at one N per run: END(k) for every k = 1..N and START(q) for every q; per END run its natural L47 anchor a (matcher-head mean, rows B1 / B4 / answer, majority) and, at a, E (natural END(k)), R (natural START(a)) and E_BANDR (band 46-49 read-row outputs <- R: the writer-free run), so B = z51(E_BANDR) and (with --pin 1, test sets) PIN (the address p

Code: `code/s913_analyze.py`, `code/s913_collect.py`, `code/s913_fit.py`, `code/s913_grid.py`

Shared modules: `lib/c_headpatch.py`, `lib/c_qpath.py`, `lib/c_qpath_k.py`, `lib/c_tools.py`, `lib/c_tritonlock.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s913_frozen_predictors.json`, `data/s913_s913n7_27b_collect.json`, `data/s913_s913n8_27b_collect.json`, `data/s913_s914e8_27b_collect.json`, `data/s913_s914n6_27b_collect.json`, `data/s913_s914n9_27b_collect.json`, `data/s913_s923A_27b_collect.json`
