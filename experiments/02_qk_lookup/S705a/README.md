# S705a

Category: `02_qk_lookup`. Query-key captures of the index-matching heads, the address lens and the nearest start-relative lookup.

From the main script's docstring:

> S705a follow-up (exploratory, registered as such in knowledge/DECISIONS.md): contrastive query-key bases (Lee et al. 2602.04752) for the MATCHER heads, fitted on START runs of the discovery scenes and read on the held-out END runs. Per head (layer, head, row): q = post-RoPE query of the row; positive key = the target block's key, negative = mean of the other blocks' keys, with a block's key = the mean post-RoPE key over its tokens ("blk") or the key of its vision_end token ("ve"). dC = mean_{scene, t} q (k_pos - k_neg)^T -> SVD -> rank (99% squared-singular-value energy), U (query side), V (ke

Code: `code/s705a_analyze.py`, `code/s705a_contrast.py`, `code/s705a_explore_e0.py`, `code/s705a_explore_e2vse5.py`, `code/s705a_explore_inventory.py`, `code/s705a_explore_shift.py`, `code/s705a_qkanatomy.py`, `code/s705a_startonly_v2.py`, `code/s719_analyze.py`, `code/s719_qkanatomy.py`, `code/s805b_window_shape.py`, `code/s805c_query_decomp.py`, `code/s805d_query_decomp_small.py`, `code/s805e_query_resolution.py`, `code/s805f_operative.py`, `code/s805g_decode_split.py`, `code/s805i_transition.py`

Shared modules: `lib/c_tools.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/g_crowding.json`, `data/s705a_analysis.json`, `data/s705a_headsets_27b.json`, `data/s705a_qa_27b_contrast.json`, `data/s705a_qa_27b_matchers.json`, `data/s705a_qa_27b_qkanatomy.json`, `data/s705a_qa_4b_qkanatomy.json`, `data/s705a_qa_9b_qkanatomy.json`, `data/s719_qa5_27b_qkanatomy.json`, `data/s719_qa9_27b_qkanatomy.json`, `data/s805b_window_shape.json`, `data/s805d_query_decomp_small.json`, `data/s805e_query_resolution.json`, `data/s805f_operative.json`, `data/s805g_decode_split.json`, `data/s805i_transition.json`
