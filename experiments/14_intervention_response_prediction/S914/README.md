# S914

Category: `14_intervention_response_prediction`. Predicting intervention outcomes: fitted profile predictors (S913-S925) and the parameter-free intervention-response prediction from the model's gradients and path responses (S926-S932), with the circuit-restricted test.

From the main script's docstring:

> S914 analysis (rules fixed before the data; knowledge/DECISIONS.md S914 entry; the advisor's design, 10-22). Frozen predictors main/s913_frozen_predictors.json (fitted by exp/s913_fit.py on the S913 training runs only, N 7 and 8; hash checked). Test runs results/r3/s913_<tag>_collect.json for three sets never used in fitting: new N9 scenes ('from the last'), N6 (an image count the predictors never saw), N8 with the new question form 'from the end'; each with --pin 1 (E, R, E_BANDR, PIN, PIN_BANDR at the natural L47 anchor a). Measurement boundary: the predictors read the L47 (or L43) early blo

Code: `code/s914_analyze.py`

Shared modules: none

Data: `data/s914_analysis_s914n9_27b.json`
