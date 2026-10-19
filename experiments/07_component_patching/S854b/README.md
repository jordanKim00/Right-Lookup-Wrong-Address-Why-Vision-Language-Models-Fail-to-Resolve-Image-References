# S854b

Category: `07_component_patching`. Token-mixer and MLP outputs and residual rows replaced band by band (the writer band 46-49, boundary rows, Shapley shares).

From the main script's docstring:

> S854-2 (pre-registered in knowledge/DECISIONS.md before any S854-2 data; user 10-18 'S854-2 - input-row causal decomposition of the writer'). Which input rows carry the ordinal-related state into the writer (layers 46-49), and do they combine additively? Eight pictures shown and stated, early lookup pinned (address part U of the ten read rows at the early look layers 35-47 <- START(p)); natural states E1 (k 3, p 3), E2 (k 3, p 4), E3 (k 5, p 3), E4 (k 5, p 4). Input groups at the input of layer 46 (positions from the ordinal token on; earlier positions are identical in every run): Q = question

Code: `code/s854b_analyze.py`, `code/s854b_rows.py`

Shared modules: `lib/c_qpath_j.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s854b_analysis.json`, `data/s854b_s854b_27b_rows.json`
