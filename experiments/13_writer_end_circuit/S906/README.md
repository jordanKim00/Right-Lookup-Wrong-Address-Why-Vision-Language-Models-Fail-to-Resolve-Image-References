# S906

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S906 (registered in knowledge/DECISIONS.md before its data) - is the END mode vector the same for every ordinal? S904: at the layer-35 input the read rows' END(5) - START(5) difference is one scene-independent direction per row, and adding it to START(5) produces END(5)'s late shift and one-step-back answers. If END(k) = START(k)'s state + a k-independent mode vector, the k = 5 vector added to START(2) / START(3) should produce END(2) / END(3)'s own answers (the end-relative position computed downstream from the ordinal already in the state). Natural runs (no pins). Calibration: 24 scenes (see

Code: `code/s906_analyze.py`, `code/s906_modek.py`

Shared modules: `lib/c_headpatch.py`, `lib/c_qpath.py`, `lib/c_qpath_k.py`, `lib/c_tools.py`, `lib/c_tritonlock.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s906_analysis_s906_27b.json`, `data/s906_s906_27b_modek.json`
