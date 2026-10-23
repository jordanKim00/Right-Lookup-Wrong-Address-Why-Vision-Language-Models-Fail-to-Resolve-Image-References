# S924

Category: `14_intervention_response_prediction`. Predicting intervention outcomes: fitted profile predictors (S913-S925) and the parameter-free intervention-response prediction from the model's gradients and path responses (S926-S932), with the circuit-restricted test.

From the main script's docstring:

> S924 collector (registered in knowledge/DECISIONS.md before the data; user 10-23: keep going until intervention prediction clearly works) - fresh test of the mechanism-calibrated predictor on NOVEL interventions. S913's collector (natural END(k) and START(q), the natural L47 anchor a, E / R / E_BANDR / PIN / PIN_BANDR with PIN = the address part U of the L35-47 read-row queries <- START(a)'s, S910 / S914) plus four arms never in the predictor's training data, at the same anchor: PINm (U <- START(a - 1)'s), PINp (U <- START(a + 1)'s), PINt (U <- START(t)'s, t = N + 1 - k: the address corrected 

Re-measured with the clean readout in: S925 (files tagged with the re-run name are kept here).

Code: `code/s924_analyze.py`, `code/s924_collect.py`, `code/s924_fit.py`

Shared modules: `lib/c_headpatch.py`, `lib/c_qpath.py`, `lib/c_qpath_k.py`, `lib/c_tools.py`, `lib/c_tritonlock.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s924_analysis_s924_27b.json`, `data/s924_analysis_s925_27b.json`, `data/s924_frozen_MC.json`, `data/s924_s924_27b_collect.json`, `data/s924_s925_27b_collect.json`
