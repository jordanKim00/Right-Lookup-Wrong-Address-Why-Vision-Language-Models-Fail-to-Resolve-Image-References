# S927

Category: `14_intervention_response_prediction`. Predicting intervention outcomes: fitted profile predictors (S913-S925) and the parameter-free intervention-response prediction from the model's gradients and path responses (S926-S932), with the circuit-restricted test.

From the main script's docstring:

> S927 - parameter-free linear-response (LR) predictor of intervention outcomes from the model's own gradients (user 10-23: '모델 내부 그라디언트나 뭐 이것저것 잘 조합해서 해봐 수학적으로 우아하게'). Nothing is fitted to any outcome. For a natural END(k) run let z be the menu logits (one per menu word, max over the word's two first tokens; in float32 from the final hidden state, i.e. the bf16 logits without their output rounding; order = labs: positions 1..N are the pictures, the rest distractors). The natural answer x_nat is the collector's (bf16 logits, argmax in menu order). An intervention I changes circuit sites s by Del

Code: `code/s927_analyze.py`, `code/s927_lr.py`

Shared modules: `lib/c_qpath.py`, `lib/c_tools.py`, `lib/c_tritonlock.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s927_analysis_s927a_27b.json`, `data/s927_analysis_s927b_27b.json`, `data/s927_s927a_27b_lr.json`, `data/s927_s927b_27b_lr.json`
