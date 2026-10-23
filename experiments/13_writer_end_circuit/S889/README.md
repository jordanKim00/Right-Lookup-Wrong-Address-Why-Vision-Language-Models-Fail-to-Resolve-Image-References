# S889

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S889 (registered in knowledge/DECISIONS.md before its data; after S886 C1 (FA47 writes the predecessor content of the writer's late query change) with S884 H1 / H5 (what FA47 reads at the predecessor's position does not matter) and the post hoc finding that FA47 picture values carry their predecessor's content at about 1/5 of their own) - is the predecessor content that FA47 writes carried as context in the ANCHOR's values? Context transplant at FA47's ten read rows (float32 recompute, pins re-applied, keys and attention pattern unchanged): the anchor block's 64 image-patch values are replaced

Code: `code/s889_analyze.py`, `code/s889_context.py`

Shared modules: `lib/c_headpatch.py`, `lib/c_qpath.py`, `lib/c_qpath_k.py`, `lib/c_tools.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s889_analysis_s889_27b.json`, `data/s889_s889_27b_context.json`
