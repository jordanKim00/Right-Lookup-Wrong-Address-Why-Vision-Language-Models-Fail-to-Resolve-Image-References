# S920

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S920 collector (registered in knowledge/DECISIONS.md before the data; advisor reviews 2-3) - R³ against the END circuit on the same scenes: in runs R³ repairs, is the band's END-specific rewriting gone and do the late read outputs approach direct START(t)'s? What do its two writes (27B R³-v2, S713: at layer 3 the ordinal's address Mb[t] - Mb[k_hat] and the frame carrier END -> START; t = k_hat gets the frame write only) do separately? Natural runs at one N (fresh scenes), k in --ks (target t = N + 1 - k): natural END(k), START(q) for every q (batched). Per k: three R³ plans (batch-1 capture fo

Code: `code/s920_analyze.py`, `code/s920_r3circuit.py`

Shared modules: `lib/c_qpath_k.py`, `lib/c_tritonlock.py`, `lib/r3_min_md.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s920_analysis_s920A_27b.json`, `data/s920_s920A_27b_r3circuit.json`
