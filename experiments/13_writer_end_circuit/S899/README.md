# S899

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S899 (registered in knowledge/DECISIONS.md before its data; S892's natural-run design extended to other N, centred on the middle identity cells - N7 END(4) and N9 END(5), where the early anchor is already right and the writer's step breaks it (S879)) - do the same writer measurements hold across N? New scenes per N (seed 89900 + N), nine names; per scene phase A: natural END(k) for k in --ks and START(q) for all q (no pins); each END run's natural L47 anchor a. Phase B, per k with 2 <= a <= N: E, R = natural START(a), E_BANDR (band <- R), E_ALL47R (FA47 <- R), E_QM (MLP46 <- R in FA47's querie

Code: `code/s899_analyze.py`, `code/s899_natural_n.py`

Shared modules: `lib/c_headpatch.py`, `lib/c_qpath.py`, `lib/c_qpath_k.py`, `lib/c_tools.py`, `lib/c_tritonlock.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s899_analysis_s899n7_27b.json`, `data/s899_analysis_s899n9_27b.json`, `data/s899_s899n7_27b_natural.json`, `data/s899_s899n9_27b_natural.json`, `data/s899_s902n8_27b_natural.json`
