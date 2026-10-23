# S874

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S874 (exploratory, descriptive; knowledge/DECISIONS.md 10-21 14:0x plan item 'S874 = the original J-lens, AB vs AQ') - the vocabulary Jacobian lens of Gurnee et al. 2026 (transformer-circuits.pub/2026/workspace; estimator of the public jacobian-lens code 581d398, jlens/fitting.py, local copy research/ces/inputs/jacobian_lens_581d398) restricted to a candidate token set: is the address j* = N + 1 - k, which the model computes without saying it when the answer is a picture's name (AB), readable as a number token, and is it when the question asks for the number (AQ)? In the paper the line-w

Code: `code/s874_analyze.py`, `code/s874_vocablens.py`

Shared modules: `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s874_analysis_s874_27b.json`, `data/s874_s874_27b_rows.json`, `data/s874_s874_27b_vocablens.json`
