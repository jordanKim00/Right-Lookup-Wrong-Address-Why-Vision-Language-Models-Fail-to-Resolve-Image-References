# S652

Category: `03_index_code_and_copy`. The index code on the image blocks (code shifts, key/value exchange, block size) and copy-versus-compose tests.

From the main script's docstring:

> S652 analysis: intervention x model matrix of predicted-landing rates for A_copy vs A_comp (and A_count where it differs), on rows where the algorithms predict different blocks (knowledge/CAUSAL_MODEL_PREDICTIONS.md; registration exp/QUEUE.md S652). I1 all codes +2 (s545 'R|pre'): A_copy k-2 (k >= 3) · A_comp j · A_count j-2 I2 last code -1 (s549 'R|m1'): A_copy k · A_comp j-1 · A_count j I3 last code +1 (s549 'R|p1'): A_copy k · A_comp j+1 · A_count j I4 forward-ordinal transplant (s474b grid 'ord|L' at the layer nearest relative depth .13, donor-correct rows): A_copy x · A_comp N+1-x baselin

Code: `code/s652_fit.py`

Shared modules: none

Data: `data/s652_fit.json`, `data/s652_fit_with27bI1.json`
