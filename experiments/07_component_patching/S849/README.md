# S849

Category: `07_component_patching`. Token-mixer and MLP outputs and residual rows replaced band by band (the writer band 46-49, boundary rows, Shapley shares).

From the main script's docstring:

> S849-P (test B evaluation; registered in knowledge/DECISIONS.md before any S849 data, with the frozen readout and thresholds set on the S848 development data). New scenes and new (p, N_s) combinations: 7 pictures shown, k = 4 (identity at N_s = 7), stated count N_s in {6, 7, 8}, early lookup pinned to p in {2, 3, 5, 6}. Per condition, in one pass: E_EARLY_p (END pinned), S_EARLY_p (START(k) pinned), BANDF_p / BANDE_p (band 46-49 outputs swapped) and the passive weight-only predictions PRED_F_p / PRED_E_p (each late full-attention reader 51 / 55 / 59 / 63 recomputes its query from the unchanged

Code: `code/s849_pred.py`, `code/s849_use.py`

Shared modules: `lib/c_qpath_g.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s849_readout.json`, `data/s849_s849p_27b_pred.json`, `data/s849_s849u2_27b_use.json`, `data/s849_s849u_27b_use.json`
