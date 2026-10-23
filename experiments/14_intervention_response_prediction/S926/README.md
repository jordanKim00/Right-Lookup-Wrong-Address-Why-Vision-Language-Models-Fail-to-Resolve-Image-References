# S926

Category: `14_intervention_response_prediction`. Predicting intervention outcomes: fitted profile predictors (S913-S925) and the parameter-free intervention-response prediction from the model's gradients and path responses (S926-S932), with the circuit-restricted test.

From the main script's docstring:

> S926 collector (registered in knowledge/DECISIONS.md before the data; user 10-23: keep going until intervention prediction clearly works) - fresh test of the two-route predictor on intervention types never seen by any predictor. S924's collector unchanged (per k at END(k)'s natural L47 anchor a: E, R, E_BANDR, PIN, PIN_BANDR, PINm, PINp, St, PINt, PINtB - now all seen types) plus a second sub-pass per k with four NEW types: PINmB (U <- START(a - 1)'s and the band 46-49 read-row outputs <- START(a - 1)'s: a one-step-earlier address with the writer removed), PINpB (the same at a + 1), PINm2 (U <

Re-measured with the clean readout in: S930 (files tagged with the re-run name are kept here).

Code: `code/s926_analyze.py`, `code/s926_collect.py`

Shared modules: `lib/c_headpatch.py`, `lib/c_qpath.py`, `lib/c_qpath_k.py`, `lib/c_tools.py`, `lib/c_tritonlock.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s926_analysis_s926n7_27b.json`, `data/s926_analysis_s926n8_27b.json`, `data/s926_s926n7_27b_collect.json`, `data/s926_s926n8_27b_collect.json`, `data/s926_s930n7_27b_collect.json`
