# S850

Category: `07_component_patching`. Token-mixer and MLP outputs and residual rows replaced band by band (the writer band 46-49, boundary rows, Shapley shares).

From the main script's docstring:

> S850 (pre-registered in knowledge/DECISIONS.md before any S850 data; user 10-17 evening): boundary-level stop-signal causal test - can the hypothesis 'goal-attainment information exists but is not used for the address update' be falsified at the writer's input? 8 pictures (the same real images for every condition of a scene), k = 4, stated count N_s in {6, 7, 8} (stated target 3, 4, 5), early lookup pinned to p in {3, 4, 5} (E_EARLY_p, layers 35-47; 3 x 3 grid, diagonal = goal reached; p = 4 gives left / same / right). The boundary rows B1 and B4 are steered at the input of layer 46 (the write

Code: `code/s850_analyze.py`, `code/s850_bstop.py`

Shared modules: `lib/c_qpath_h.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s850_analysis.json`, `data/s850_chain.json`, `data/s850_explore.json`, `data/s850_layers.json`, `data/s850_s850_27b_bstop.json`
