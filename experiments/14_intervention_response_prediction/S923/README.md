# S923

Category: `14_intervention_response_prediction`. Predicting intervention outcomes: fitted profile predictors (S913-S925) and the parameter-free intervention-response prediction from the model's gradients and path responses (S926-S932), with the circuit-restricted test.

From the main script's docstring:

> S923 fit and freeze (registered in knowledge/DECISIONS.md before the S923 test data): the circuit-structured predictor AP - S913's structured model M3 revised by two circuit findings, fitted on the S913 training runs only (N7 / N8, natural END(k) with E_BANDR): (1) no argmax features: the late support and the writer's update depend on the early profile through e(b - 1 .. b + 3), the last-picture and copy indicators only - no anchor indicator, no anchor-margin terms (the early reference is a graded preference; the argmax indicator absorbed the anchor's support in natural data and froze it under

Code: `code/s923_analyze.py`, `code/s923_fit.py`

Shared modules: none

Data: `data/s923_analysis_s923A_27b.json`, `data/s923_frozen_AP.json`
