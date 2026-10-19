# S849p

Category: `07_component_patching`. Token-mixer and MLP outputs and residual rows replaced band by band (the writer band 46-49, boundary rows, Shapley shares).

From the main script's docstring:

> S849-P analysis (rules fixed in knowledge/DECISIONS.md before any S849 data; thresholds and the readout fixed on the S848 development data). Test B: from measured changes of the band 46-49 outputs, a fixed computation predicts the late queries, the image-wise positional scores and the answer distribution in new scenes and new (p, N_s) combinations (7 pictures, k = 4, N_s 6-8, p in {2, 3, 5, 6}). Interventions: BANDF (END pinned, START's band outputs) predicted by PRED_F from E_EARLY; BANDE (START(k) pinned, END's band outputs) predicted by PRED_E from S_EARLY. PRED arms recompute each late ful

Code: `code/s849p_analyze.py`

Shared modules: none

Data: `data/s849p_analysis.json`
