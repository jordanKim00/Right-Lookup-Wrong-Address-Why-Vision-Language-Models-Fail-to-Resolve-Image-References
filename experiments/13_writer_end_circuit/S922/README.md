# S922

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S922 analysis = S920's rules on one replication set (registered in knowledge/DECISIONS.md S922 entry): results/r3/s920_<tag>_r3circuit.json at N7, k 1-7 (the end boundaries k = 1, 7 included; k = 4 is the identity cell where R³ writes the frame only). Below, S920's text with 'both sets' read as 'the set', R5 on k = (N + 1) / 2, and the boundary cells also reported apart. Per scene and k (target t = N + 1 - k, a = END(k)'s natural L47 anchor): E, Sa (START(a)), E_B (E + Sa's band outputs), St (START(t)), R3, R3_B (+ St's band), R3A (address write only), R3A_B, R3F (frame write only), Sk (START(

Code: `code/s922_analyze.py`

Shared modules: none

Data: `data/s922_analysis_s922_27b.json`
