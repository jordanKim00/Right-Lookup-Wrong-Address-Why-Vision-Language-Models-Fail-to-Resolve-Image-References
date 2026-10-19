# S848

Category: `07_component_patching`. Token-mixer and MLP outputs and residual rows replaced band by band (the writer band 46-49, boundary rows, Shapley shares).

From the main script's docstring:

> S848 (pre-registered in knowledge/DECISIONS.md before any S848 data; user 10-17 15:03, test A + development data for test B). Test A, part 1 (representation): with the early lookup pinned to p (E_EARLY_p, as S846) and the stated target t = N_s - 3 below, at or above p (8 pictures, k = 4, N_s in {6, 7, 8, 9}, p in {3, 4, 5, 6}), is goal attainment (p = t) represented in the read rows' residual stream around the band (inputs of layers 44, 46-50, 52)? The residuals of every E_EARLY_p run are kept (ten read rows). Test B development (not the evaluation): the same pass also holds S_EARLY_p (START(k

Code: `code/s848_analyze.py`, `code/s848_stop.py`

Shared modules: `lib/c_qpath_g.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s848_analysis.json`, `data/s848_s848_27b_stop.json`
