# S902

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S902 analysis (rules fixed before the data; knowledge/DECISIONS.md S902 entry). Input results/r3/s899_<tag>_natural.json (exp/s899_natural_n.py run at N 8 with ks 1, 2, 3 on new scenes). Question: S892 found the writer's step on at every k from 2 to 6 (W at k = 2 as large as at k >= 3). Is it on at END(1) - the last picture, which the model answers well - and if it is, why is END(1) right? Per natural END(k) run with its natural L47 anchor a (2 <= a <= N) and the references at a: T1_a = z(a - 1) - z(a) (matcher-head mean LSE block scores at L51, rows B1 / B4 / answer); W = T1_a(E) - T1_a(E_BAN

Code: `code/s902_analyze.py`

Shared modules: none

Data: `data/s902_analysis_s902n8_27b.json`
