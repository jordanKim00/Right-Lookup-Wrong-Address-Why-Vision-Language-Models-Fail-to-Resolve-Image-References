# S897

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S897 (registered in knowledge/DECISIONS.md before its data; after S890: MLP46's path into FA47's queries and gates runs mainly through the 13 heads outside the anchor heads and h16 (.41 of FA47's END part), and the post hoc END attention changes of h14 (end-ward), h3 / h23 / h10 / h18 / h19 (text -> pictures)) - which single heads carry it? Same path patch (layer-46 MLP output <- START's in FA47's read-row queries and gates) for single heads h3, h14, h23, h10, h18, h19 and the other seven REST heads together, plus the REST group (S890 replication). Scenes = S872's (seed 87200, 48); END(5) pinn

Code: `code/s897_analyze.py`, `code/s897_qpathheads.py`

Shared modules: `lib/c_headpatch.py`, `lib/c_qpath.py`, `lib/c_qpath_k.py`, `lib/c_tools.py`, `lib/c_tritonlock.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s897_analysis_s897_27b.json`, `data/s897_s897_27b_qpathheads.json`
