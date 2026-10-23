# S928

Category: `14_intervention_response_prediction`. Predicting intervention outcomes: fitted profile predictors (S913-S925) and the parameter-free intervention-response prediction from the model's gradients and path responses (S926-S932), with the circuit-restricted test.

From the main script's docstring:

> S928 - path-integral (quadrature) extension of S927's linear response (user 10-23: '모델 내부 그라디언트 ... 수학적으로 우아하게'). For an intervention I with exact local site change Delta (S927), the effect on the menu logits is exactly the integral of the model's own gradient along the intervention path: z^I - z = int_0^1 <grad z(h + e Delta), Delta> de (fundamental theorem of calculus; integrated-gradients completeness). S927's LR is the one-node rule at e = 0. Here one more node at the half-intervened state e = 1/2 gives, from one forward and the backward passes there: phi(1/2) - phi(0) and g(1/2) = <grad z

Code: `code/s928_quad.py`

Shared modules: `lib/c_qpath.py`, `lib/c_tools.py`, `lib/c_tritonlock.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s928_s928a_27b_quad.json`
