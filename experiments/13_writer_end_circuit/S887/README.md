# S887

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S887 (registered in knowledge/DECISIONS.md before its data; after S885: FA47's END part is the self-address heads reading less of the anchor, and FA47's natural END effect is about twice its effect without layer 46's END write (K4); S863: the MLPs carry the k switch) - does layer 46's END write move the FA47 self-address heads off the anchor through their queries? Path patch at FA47's ten read rows: the queries and output gates are recomputed (float32, pins re-applied; keys, values and every other position unchanged) from the residual entering layer 47 with the layer-46 MLP output (QM), the la

Code: `code/s887_analyze.py`, `code/s887_qpath46.py`

Shared modules: `lib/c_headpatch.py`, `lib/c_qpath.py`, `lib/c_qpath_k.py`, `lib/c_tools.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s887_analysis_s887_27b.json`, `data/s887_s887_27b_qpath.json`
