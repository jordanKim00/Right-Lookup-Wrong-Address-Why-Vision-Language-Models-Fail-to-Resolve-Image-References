# S832

Category: `12_laws_and_predictions`. Answer laws and predictions fixed in advance, the held-out source-cut predictions and the post hoc analyses.

From the main script's docstring:

> S832 (pre-registered in knowledge/DECISIONS.md before any S832 data): does the two-component law FROZEN after S831 (research/ces/outputs/ s832_frozen_law.json = the fit of exp/p2_w1b_cross.py on natural runs + S830; FINDINGS §926) predict, before they are run, the landing distributions of END(k) in two NEW cells when the early lookup is moved to p, and their natural distributions? Law: P(x | a, j) = ws Step(x | a, j) + we End(x | j) + wu / N; Step = offsets from the early anchor a toward the end-relative target j (+-1 at random when a = j); End = offsets around j; natural runs use the literal 

Code: `code/s832_analyze.py`, `code/s832_newcells.py`

Shared modules: `lib/c_qpath_b.py`, `lib/pipe_run.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s832_analysis.json`, `data/s832_frozen_law.json`, `data/s832_s832_27b_anchor.json`
