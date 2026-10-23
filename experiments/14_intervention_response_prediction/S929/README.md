# S929

Category: `14_intervention_response_prediction`. Predicting intervention outcomes: fitted profile predictors (S913-S925) and the parameter-free intervention-response prediction from the model's gradients and path responses (S926-S932), with the circuit-restricted test.

From the main script's docstring:

> S929 - circuit-restricted linear response: does the identified circuit carry the model's first-order response to the interventions? Same computation as S927 (exp/s927_lr.py: z^I = z + sum_s <dz/dh_s, Delta_s^I>, exact local changes, no fitted parameter), with the gradient restricted to paths of the identified circuit (values unchanged; only the backward flow is cut): R0 = none (S927); R1 = read rows only: at every decoder layer >= 35 the hidden state at positions outside the read rows is detached (so the keys and values the late reads see keep their natural values: the menu-name and picture ro

Code: `code/s929_analyze.py`, `code/s929_circ.py`

Shared modules: `lib/c_qpath.py`, `lib/c_tools.py`, `lib/c_tritonlock.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s929_analysis_s929a_27b.json`, `data/s929_analysis_s929b_27b.json`, `data/s929_s929a_27b_circ.json`, `data/s929_s929b_27b_circ.json`
