# S878

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S878 (registered in knowledge/DECISIONS.md before its data; advisor reviews 5 and 6, 10-21: the Query-Key crossed counterfactual) - is the scene dependence of the writer's address effect (S873 A1; L51 post hoc p19) made by the writer's query update or by the scene's image keys? Scenes = S872's (seed 87200, 48); END(5) (E), START(p) (R) and E_BANDR (E with the band's read-row outputs <- R) pinned at p in {3, 4, 5}. One forward per scene holds the nine runs (3 pins x E, R, E_BANDR). At the late full-attention layers 51, 55, 59, 63 a forward hook recomputes from the layer's own inputs (input-norm

Code: `code/s878_analyze.py`, `code/s878_qkcross.py`

Shared modules: `lib/c_headpatch.py`, `lib/c_qpath_k.py`, `lib/c_tools.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s878_analysis_s878_27b.json`, `data/s878_s878_27b_qk.json`
