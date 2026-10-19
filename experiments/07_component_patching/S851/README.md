# S851

Category: `07_component_patching`. Token-mixer and MLP outputs and residual rows replaced band by band (the writer band 46-49, boundary rows, Shapley shares).

From the main script's docstring:

> S851 (pre-registered in knowledge/DECISIONS.md before any S851 data; user 10-17 late evening, minimal path test after S850): do the late full-attention queries (layers 51-63) carry the correction that replacing the boundary state (FULLB) produces, and does the steered run (PLUS16) recover when only its late queries are corrected? The S850 grid and arms on fresh scenes (8 pictures, k = 4, N_s 6-8, p 3-5 pinned; 30 scenes, seed 85100). Per p in one pass: E_EARLY_p (natural END pinned), S_EARLY_p, FULLB_p (boundary rows B0-B4 at the input of layer 46 <- S_EARLY_p), PLUS16_p (B1 / B4 steered +16 S

Code: `code/s851_analyze.py`, `code/s851_path.py`

Shared modules: `lib/c_qpath_h.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s851_analysis.json`, `data/s851_desc.json`, `data/s851_s851_27b_path.json`, `data/s851_split.json`
