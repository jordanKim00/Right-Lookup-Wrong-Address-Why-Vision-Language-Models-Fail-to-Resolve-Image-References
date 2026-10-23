# S932

Category: `14_intervention_response_prediction`. Predicting intervention outcomes: fitted profile predictors (S913-S925) and the parameter-free intervention-response prediction from the model's gradients and path responses (S926-S932), with the circuit-restricted test.

From the main script's docstring:

> S932dev - every intervention by the quadratic along its OWN path (user 10-23: elegant, model-internal): pins: the real pin path - at every pinned reader the read-row query's U part <- U part of (1 - e) q_nat + e q_donor (q_nat = the natural run's pre-RoPE read-row query at that layer), recomputed from the running input (e = 1 is exactly the real pin; no layer-to-layer double counting); slope by the finite difference at e = .1, value at e = 1/2: HERM = 4 (z(1/2) - z) - (z(.1) - z) / .1 (batched, no gradient); band-replacing arms: the joint path (pin local changes at 35 / 39 / 43 and band set to

Code: `code/s932_analyze.py`, `code/s932_pathq.py`

Shared modules: `lib/c_qpath.py`, `lib/c_tools.py`, `lib/c_tritonlock.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s932_s932d7_27b_pathq.json`, `data/s932_s932d8_27b_pathq.json`
