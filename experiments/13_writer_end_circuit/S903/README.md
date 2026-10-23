# S903

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S903 (registered in knowledge/DECISIONS.md before its data) - which components of the frame-read band L27-34 write the read rows' END state? S896: the read rows' residual at layer 35 carries 89% of the END - START late-address shift; S898: for the FA47 query path the signal is written at L31-34 (.48) and L27-30 (.18), not before 27. Here the donor differs from END(5) only in the frame word: START(5) ('fifth ... from the first'; same ordinal), so the patches carry the frame read only. Component patches (QPathK 'cpatch' at the read rows rowsC: the component's output <- the donor's) and residual 

Code: `code/s903_analyze.py`, `code/s903_framewrite.py`

Shared modules: `lib/c_headpatch.py`, `lib/c_qpath.py`, `lib/c_qpath_k.py`, `lib/c_tools.py`, `lib/c_tritonlock.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s903_analysis_s903_27b.json`, `data/s903_s903_27b_framewrite.json`
