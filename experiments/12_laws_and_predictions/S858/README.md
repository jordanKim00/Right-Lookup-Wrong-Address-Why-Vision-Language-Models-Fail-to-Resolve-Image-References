# S858

Category: `12_laws_and_predictions`. Answer laws and predictions fixed in advance, the held-out source-cut predictions and the post hoc analyses.

From the main script's docstring:

> S858 (P3, user 10-18: scope frozen after S857; pre-registered in knowledge/DECISIONS.md before any S858 data). Prior predictions of what unperformed interventions do in new conditions, at three levels: the late queries (51-63), the late picture scores, the answer logits / answers. New conditions (none used in S852-S857): N_a in {9, 7} pictures with the last two as the source, k {4, 6} at N 9 and {4, 5} at N 7, early lookup pinned at p {2, 5} (address part U over the shared pictures 1..N-2 <- START(p) of the same configuration, layers 35-47), new scenes (seed 85980), all nine single-token label

Code: `code/s858_analyze.py`, `code/s858_pred.py`

Shared modules: `lib/c_cut.py`, `lib/c_qpath_j.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s858_analysis.json`, `data/s858_s858_27b_pred.json`, `data/s858_slopefit.json`
