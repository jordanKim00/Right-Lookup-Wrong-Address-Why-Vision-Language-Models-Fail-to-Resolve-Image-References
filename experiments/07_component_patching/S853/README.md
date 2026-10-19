# S853

Category: `07_component_patching`. Token-mixer and MLP outputs and residual rows replaced band by band (the writer band 46-49, boundary rows, Shapley shares).

From the main script's docstring:

> S853 (pre-registered in knowledge/DECISIONS.md before any S853 data; user 10-18 'GO - writer input decomposition'). What does the writer (layers 46-49) take in, what does it compute, and how is the result carried? Eight real pictures, the same menu, the END frame, the early lookup pinned (address part U of the ten read rows at the early look layers 35-47 <- START(p, N_s)). Writer-input states (natural pinned END runs; the stated count N_s is a counterfactual probe when it differs from the eight pictures shown): E1 (k 3, p 3, N_s 8) E2 (k 3, p 4, 8) E3 (k 5, p 3, 8) E4 (k 5, p 4, 8) main 2 x 2 

Code: `code/s853_analyze.py`, `code/s853_writer.py`

Shared modules: `lib/c_qpath_i.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s853_analysis.json`, `data/s853_int.json`, `data/s853_s853_27b_writer.json`
