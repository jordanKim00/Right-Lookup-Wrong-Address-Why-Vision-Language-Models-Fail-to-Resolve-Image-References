# S825

Category: `01_behavior`. Natural answers: scale, answer grids, option-embedded references, text lists, other families and reasoning models.

From the main script's docstring:

> S825 fit (pre-data for S825): an END error law fitted on N = 7 and N = 8 natural END landings (27B, clean readouts: the pooled clean runs of research/ces/outputs/agent_math_1015/pool_end_kernel.json and S824's matched conditions a7s7 / a8s8), used to predict N = 6 and N = 9 before any N = 6 / 9 run. For 1 < k < N, j = N + 1 - k: P(p) = w_t [p = j] + w_o [p = j - 1] (one past the target toward the start) + w_s [p = k - sgn(k - j)] (one step from the literal toward the target; for k = j the step toward the start, k - 1) + w_l [p = k] + w_u / N (spread); weights per class c = 'near' (k in {2, N -

Code: `code/s825_analyze.py`, `code/s825_fit.py`, `code/s825_newN.py`

Shared modules: `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s825_analysis.json`, `data/s825_predictions.json`, `data/s825_s825_27b_newN.json`
