# S862

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S862 (registered in knowledge/DECISIONS.md before its data; user 10-20 'close the last chain and show that the identified algorithm predicts') - the predictive test of the identified circuit on unseen conditions and interventions. One scene = nine pictures with all nine names as the menu; layouts show its first N pictures (N 6-9, stated = shown). G (natural grid): END(k) for every k = 1..N and START(q) for every q = 1..N, N in {6, 7, 8, 9}; the early anchor (L47 head-mean argmax) of every run is recorded - the frozen algorithm predicts each run's answer from it. I (natural interventions in cel

Code: `code/s862_analyze.py`, `code/s862_predict.py`

Shared modules: `lib/c_headpatch.py`, `lib/c_qpath_i.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s862_analysis_s862_27b.json`, `data/s862_s862_27b_predict.json`
