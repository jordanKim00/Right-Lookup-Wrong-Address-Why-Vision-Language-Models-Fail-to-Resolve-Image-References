# S911

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S911 (registered in knowledge/DECISIONS.md before its data; adversarial review 10-22, item D2) - how far does the END(k) transform vector reach, and does END(2) bypass it? S906 / S909: the read rows' END state at layer 35 is a scene-independent, k-specific, end-relative vector d_k; adding it turns START(k) into END(k) for k = 2 but only partly for k = 3, 5; END(2) - d2 still answers near the end. Natural runs at N 8. Calibration: 24 scenes (seed 87330), natural END(k) and START(k) for k = 1..6 -> d_k at the layer-35 input, read rows. Test: 48 new scenes (seed 91100); per scene one pass of 36 r

Code: `code/s911_analyze.py`, `code/s911_reach.py`

Shared modules: `lib/c_headpatch.py`, `lib/c_qpath.py`, `lib/c_qpath_k.py`, `lib/c_tools.py`, `lib/c_tritonlock.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s911_analysis_s911_27b.json`, `data/s911_s911_27b_reach.json`
