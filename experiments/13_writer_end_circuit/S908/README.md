# S908

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S908 (registered in knowledge/DECISIONS.md before its data) - where are the frame word and the ordinal bound into the k-specific END(k) transform? S906: the read rows' END state at layer 35 is a scene-independent but k-specific vector; S903: it is written at L27-34, mostly by DeltaNet at L31-34; S758: the ordinal's code sits in the question rows (the ordinal and the next three tokens) by L26, and the frame word is used only from L27. Natural runs (no pins), N 8, new scenes (seed 90800, 48). Recipient END(k) with donor END(k') (k, k' in {2, 5}, both directions): the residual of a row subset at 

Code: `code/s908_analyze.py`, `code/s908_bind.py`

Shared modules: `lib/c_headpatch.py`, `lib/c_qpath.py`, `lib/c_qpath_k.py`, `lib/c_tools.py`, `lib/c_tritonlock.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s908_analysis_s908_27b.json`, `data/s908_s908_27b_bind.json`
