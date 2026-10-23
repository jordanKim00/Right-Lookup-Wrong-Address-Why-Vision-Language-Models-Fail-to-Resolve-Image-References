# S884

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S884 (registered in knowledge/DECISIONS.md before its data; after S881 / S878 post hoc p21: the writer's late query change has the sign structure of (predecessor content - anchor content) in key space, and the late matcher heads find pictures by content (S880 DUP: START's own late T1 collapses between identical pictures)) - the query side, causally: does the late address follow the CONTENT the read rows read at FA47? Value transplant at the read rows only: FA47's attention of the ten read rows is recomputed (float32, pins re-applied, keys and attention pattern unchanged) with the VALUES of one

Code: `code/s884_analyze.py`, `code/s884_valueswap.py`

Shared modules: `lib/c_headpatch.py`, `lib/c_qpath.py`, `lib/c_qpath_k.py`, `lib/c_tools.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s884_analysis_s884_27b.json`, `data/s884_s884_27b_valueswap.json`
