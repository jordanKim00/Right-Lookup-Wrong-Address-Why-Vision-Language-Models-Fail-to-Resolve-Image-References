# S892

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S892 (registered in knowledge/DECISIONS.md before its data; natural-run validation of the pinned S883-S887 findings) - do the writer's mechanisms found with pinned anchors operate in natural END runs? New scenes (seed 89200, 48), N 8, nine names; per scene phase A: natural END(k) for k in 2..6 and START(q) for q in 1..8 (13 runs, no pins) - each END run's natural L47 anchor a (head-mean argmax at FA47, rows B1 / B4 / answer, majority). Phase B, per k with 2 <= a <= 8: E (natural END(k)), R = natural START(a), E_BANDR (band 46-49 read-row outputs <- R), E_ALL47R (FA47 read-row output <- R), E_Q

Code: `code/s892_analyze.py`, `code/s892_natural.py`

Shared modules: `lib/c_headpatch.py`, `lib/c_qpath.py`, `lib/c_qpath_k.py`, `lib/c_tools.py`, `lib/c_tritonlock.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s892_analysis_s892_27b.json`, `data/s892_s892_27b_natural.json`
