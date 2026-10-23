# S930

Category: `14_intervention_response_prediction`. Predicting intervention outcomes: fitted profile predictors (S913-S925) and the parameter-free intervention-response prediction from the model's gradients and path responses (S926-S932), with the circuit-restricted test.

From the main script's docstring:

> S930 analysis = S926's (exp/s926_analyze.py) with two execution fixes learned from S926 N7 (I0 failed on bf16 ties, 10-23 08:5x): (1) I0 natural reproduction is judged outside the bf16 tie zone - runs whose natural top-two margin exceeds .25 in both the collector's natural run (B_E menu logits) and the predictor's (z_bf16) - at >= .95 (the share over all runs is reported); (2) if I0 fails only I0 is printed or saved. Everything else (predictor SEP-HERM, arms, P1-P4, thresholds, bootstrap) is S926's. --- S926's doc follows --- S926 analysis (registered in knowledge/DECISIONS.md before the outco

Re-measures: S926 (their files sit in those experiments' folders).

Code: `code/s930_analyze.py`

Shared modules: none

Data: `data/s930_analysis_s930n7_27b.json`
