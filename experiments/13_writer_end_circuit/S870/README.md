# S870

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S870 (registered in knowledge/DECISIONS.md before its data; advisor review 10-21 (S870: the early route selection law) and the user's question 10-21 'with 5 or 7 images, can we predict every reference from the first to the N-th from the end?') - natural grids at N 5 (an unseen N) and N 7. One scene = nine pictures with all nine names as the menu; layouts show its first N pictures (stated = shown). G: END(k) for every k = 1..N and START(q) for every q = 1..N, N in {5, 7}; head-mean block scores of the ten read rows at every read layer (the early profiles of END and START runs feed the frozen F_

Code: `code/s870_analyze.py`, `code/s870_grid.py`

Shared modules: `lib/c_headpatch.py`, `lib/c_qpath_i.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s870_analysis_s870_27b.json`, `data/s870_s870_27b_grid.json`
