# S828

Category: `05_query_patching`. Query edits with keys, values and gates fixed: whole-query swaps, address-part pinning, late-query swaps and transfers.

From the main script's docstring:

> S828 predictions, fixed from STAGE 1 (END(k), START(j) natural runs) before STAGE 2; procedure registered in knowledge/DECISIONS.md before any S828 data. The two-route readout law is the one found after S827 and frozen here (research/ces/outputs/s827_post_tworoute.json, entry B51_A51_r2: P(i) proportional to exp(bB zB_i + bA zA_i), bB 0.7, bA 1.4; zB = mean of the boundary newline rows B1, B4, zA = the answer row, index-matching-head block scores at L51). Every arm's internal scores are predicted by replacement: the edited rows take the donor's natural scores (half: the mean of donor and recip

Code: `code/s828_analyze.py`, `code/s828_predict.py`, `code/s828_twoway.py`

Shared modules: `lib/c_qpath_b.py`, `lib/pipe_run.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s828_analysis.json`, `data/s828_predictions_s828_27b.json`, `data/s828_s828_27b_natural.json`, `data/s828_s828_27b_twoway.json`
