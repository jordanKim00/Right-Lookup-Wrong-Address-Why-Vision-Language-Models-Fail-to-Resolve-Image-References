# S891

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S891 (registered in knowledge/DECISIONS.md before its data; predictions frozen and hashed before the run): does the composition map built from the model's own weights and captured states (S885) predict the late-address effect of NEW interventions on FA47's attention? Intervention at FA47's ten read rows (float32 recompute, pins re-applied, keys and values unchanged): for a head set, the attention mass on the anchor block p is rescaled (the removed / added mass is taken from / given to every other position in proportion to its own weight) or moved onto the predecessor block p - 1 (the predecess

Code: `code/s891_analyze.py`, `code/s891_attnpredict.py`

Shared modules: `lib/c_headpatch.py`, `lib/c_qpath.py`, `lib/c_qpath_k.py`, `lib/c_tools.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s891_analysis_s891_27b.json`, `data/s891_s891_27b_attnpredict.json`
