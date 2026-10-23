# S909

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S909 (registered in knowledge/DECISIONS.md before its data) - is the END(k) transform vector an end-relative offset? S906: at N8 the read rows' END state at layer 35 is a scene-independent, k-specific vector d_k; the own vector turns START(k) into END(k). Here the N8 vectors (saved by S906: results/r3/s906_s906_27b_dvec.pt, calibration seed 87310) are added to START(k) at another N (7 or 9): if d_k encodes 'k-th from the end' relative to the visible end, START(k) + d_k(N8) should answer like END(k) at that N (target N + 1 - k), not at N8's target position (9 - k). Own-N vectors d_k(N) are cali

Code: `code/s909_analyze.py`, `code/s909_modeN.py`

Shared modules: `lib/c_headpatch.py`, `lib/c_qpath.py`, `lib/c_qpath_k.py`, `lib/c_tools.py`, `lib/c_tritonlock.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s909_analysis_s909n7_27b.json`, `data/s909_analysis_s909n9_27b.json`, `data/s909_s909n7_27b_modeN.json`, `data/s909_s909n9_27b_modeN.json`
