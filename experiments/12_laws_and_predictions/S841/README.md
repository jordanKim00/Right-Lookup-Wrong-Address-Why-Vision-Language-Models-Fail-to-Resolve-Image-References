# S841

Category: `12_laws_and_predictions`. Answer laws and predictions fixed in advance, the held-out source-cut predictions and the post hoc analyses.

From the main script's docstring:

> S841 (pre-registered in knowledge/DECISIONS.md before any S841 data): why does the late step move the identity reference? Hypothesis (sign): the late correction is a one-image step whose side comes from comparing the ordinal with the STATED count (a sign, not a distance) and that fires even when the comparison is a tie. S824 (saved): with 7 pictures shown, END(4) is an identity reference, yet saying '8 images' moved its off-target answers to the upper side (5, 6: .22 -> .40; 2, 3: .30 -> .17) - natural answers mix the step with the end-anchored estimate, so here the early lookup is moved (the 

Code: `code/s841_analyze.py`, `code/s841_sign.py`

Shared modules: `lib/c_qpath_b.py`, `lib/pipe_run.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s841_analysis.json`, `data/s841_s841_27b_anchor.json`
