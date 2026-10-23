# S885

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S885 (registered in knowledge/DECISIONS.md before its data; the weights-and-states companion of S884): capture what the composition map needs - which picture each FA47 head's read-row output points the late (L51) lookup at, computed offline from the model's own weights. Scenes = S872's (seed 87200, 48); END(5) (E), START(p) (R) pinned at p in {3, 4, 5} and E_BANDR (band read-row outputs <- R), as S878 / S881 / S883 / S884 (same scenes and pins). Per scene one pass of 9 runs. Stored per run (rows B1 / B4 / answer): FA47 (pins re-applied, float32 recompute of the read rows): per head the attenti

Code: `code/s885_analyze.py`, `code/s885_compcap.py`

Shared modules: `lib/c_headpatch.py`, `lib/c_qpath.py`, `lib/c_qpath_k.py`, `lib/c_tools.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s885_analysis_s885_27b.json`, `data/s885_s885_27b_compcap.json`
