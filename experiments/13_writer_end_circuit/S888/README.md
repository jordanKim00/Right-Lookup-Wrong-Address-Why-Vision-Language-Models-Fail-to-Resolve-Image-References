# S888

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S888 (registered in knowledge/DECISIONS.md before its data; after S884 H4: four early FA lookups reading another picture's content at the anchor's position leave START's answer at p (.965) and move its late address toward that picture by only 8-28% of the gap) - through which routes and depths does the picture content that the late lookup uses reach the read rows? Content redirect at the read rows only: at the listed layers the read rows' outputs are recomputed with the anchor block's 64 image-patch VALUES replaced token by token by the far block f's (FA: attention values, exp/s884 FAValueEdit

Code: `code/s888_analyze.py`, `code/s888_dnroute.py`

Shared modules: `lib/c_dnedit.py`, `lib/c_headpatch.py`, `lib/c_qpath.py`, `lib/c_qpath_k.py`, `lib/c_tools.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s888_analysis_s888_27b.json`, `data/s888_s888_27b_dnroute.json`
