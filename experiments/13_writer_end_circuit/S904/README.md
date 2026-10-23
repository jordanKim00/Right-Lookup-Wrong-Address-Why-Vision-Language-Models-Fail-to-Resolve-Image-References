# S904

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S904 (registered in knowledge/DECISIONS.md before its data) - is the read rows' END state at layer 35 a scene-independent mode vector? S896: the read rows' residual at layer 35 carries 89% of the END - START late-address shift; S903 asks which L27-34 components write it. Here: does one mean direction per read row (END(5) - START(5) at the layer-35 input, averaged over separate calibration scenes) turn START into END when added, and END into START when removed? Calibration: 24 scenes (seed 87300, disjoint from the test scenes), natural END(5) and START(5); the layer-35 input at the read rows ro

Code: `code/s904_analyze.py`, `code/s904_modevec.py`

Shared modules: `lib/c_headpatch.py`, `lib/c_qpath.py`, `lib/c_qpath_k.py`, `lib/c_tools.py`, `lib/c_tritonlock.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s904_analysis_s904_27b.json`, `data/s904_s904_27b_modevec.json`
